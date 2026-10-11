#!/usr/bin/env bash
# Usage: bash scripts/references/run_anyloc_crica_small.sh [all|anyloc|crica|evaluate]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
python_bin="${PYTHON_BIN:-python}"
registry=configs/references/anyloc_crica_small.yaml
stage="${1:-all}"
case "$stage" in all|anyloc|crica|evaluate) ;; *) echo "Use all, anyloc, crica or evaluate" >&2; exit 2 ;; esac
mkdir -p runs/reference
# Prevent accidental simultaneous runs against the same checkpoints/caches.
exec 9>runs/reference/anyloc_crica_small.lock
flock -n 9 || { echo "Another AnyLoc/Crica runner is active" >&2; exit 1; }
exec > >(tee -a "runs/reference/anyloc_crica_small_${stage}.log") 2>&1
cli() { "$python_bin" -m a2b_lightvpr.cli "$@"; }

fit_pca() {
    cli fit-reference-pca --config "$registry" --registry "$registry" \
        --model "$1" --output "data/reference_pca/$2.pt" \
        --cache "data/reference_pca/${2%_pca512}_raw_fp16.npy" \
        --components 512 --samples 20000 --seed 0 --batch-size "$3" --workers 4
}

if [[ "$stage" == all || "$stage" == anyloc ]]; then
    cli fit-anyloc-small-vocabulary --registry "$registry" \
        --model anyloc-dinov2-s-gsv-12288 \
        --output data/reference_vocab/anyloc_dinov2_s_plain_gsv224_c32.pt \
        --samples 20000 --patches-per-image 32 --seed 0 --batch-size 16 --workers 4
    fit_pca anyloc-dinov2-s-gsv-12288 anyloc_dinov2_s_plain_gsv224_pca512 16
fi
if [[ "$stage" == all || "$stage" == crica ]]; then
    "$python_bin" scripts/references/train_crica_small_gsv.py
    fit_pca cricavpr-dinov2-s-gsv-5376 cricavpr_dinov2_s_gsv_pca512 1
fi
if [[ "$stage" == all || "$stage" == evaluate ]]; then
    models=(anyloc-dinov2-s-gsv-12288 anyloc-dinov2-s-gsv-pca512
            cricavpr-dinov2-s-gsv-5376 cricavpr-dinov2-s-gsv-pca512)
    for model in "${models[@]}"; do
        cli evaluate-reference-suite --config "$registry" --model "$model" \
            --database-encoder adapted --transform native \
            --datasets robotcar_project svox_night msls_night tokyo_night robotcar_day svox_day svox_sun tokyo_day
    done
    resources=runs/reference/resource_benchmark_anyloc_crica_small.json
    if ! cli validate-reference-resources --registry "$registry" \
        --resources "$resources" --models "${models[@]}"; then
        cli benchmark-reference-models --config "$registry" --models "${models[@]}" \
            --warmup 20 --iterations 100 --output "$resources"
    fi
    cli summarize-small-references --registry "$registry" --reference-root runs/reference \
        --resources "$resources" --models "${models[@]}" \
        --output runs/comparisons/anyloc_crica_small_summary.json
fi
echo "Completed stage: $stage"
