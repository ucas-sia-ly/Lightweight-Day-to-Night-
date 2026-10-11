#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set registry configs/references/small.yaml
set vocabulary data/reference_vocab/anyloc_dinov2_s_l11_token_c32_urban_v2.pt
set anyloc_pca data/reference_pca/anyloc_dinov2_s_block11_train_pca512.pt
set anyloc_cache data/reference_pca/anyloc_dinov2_s_block11_train_raw_fp16.npy
set resources runs/reference/resource_benchmark_small.json
set protocols robotcar_project svox_night msls_night tokyo_night robotcar_day svox_day svox_sun tokyo_day
set models salad-small-544 boq-dinov2-s-12288 boq-dinov2-s-pca512 anyloc-dinov2-s-12288 anyloc-dinov2-s-pca512

function die
    echo "ERROR: $argv" >&2
    exit 1
end

function cli
    command "$python_bin" -m a2b_lightvpr.cli $argv
end

function no_competing_training
    not pgrep -af 'a2b_lightvpr.cli (train-vpr|train-boq-small|fit-anyloc-small-vocabulary)' >/dev/null
end

function gpu_idle
    set -l compute_pids (nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | string trim)
    test (count $compute_pids) -eq 0
end

function resources_complete
    test -f "$resources"; or return 1
    cli validate-reference-resources \
        --registry "$registry" --resources "$resources" --models $models >/dev/null
end

command -s "$python_bin" >/dev/null; or die "missing interpreter: $python_bin"
test -f runs/vpr/salad_small_544/day_fake_seed0/final_suite/summary.json; or die "SALAD-S baseline final suite is incomplete"
test -f runs/vpr/salad_small_544/cdnd_salad_base_seed0/final_suite/summary.json; or die "SALAD-S CDND final suite is incomplete"
test -f runs/reference/boq-dinov2-s-12288-distilled/model_best.pt; or die "BoQ-S distillation training is incomplete"
test -f data/reference_pca/boq_dinov2_s_distilled_train_pca512.pt; or die "BoQ-S distilled PCA is incomplete"
no_competing_training; or die "another VPR/Small-reference job is running; wait before starting completion"
gpu_idle; or die "GPU 0 has active compute processes; wait before starting completion"
"$python_bin" -c 'import shutil, sys, torch; free=shutil.disk_usage(".").free/2**30; print(f"cuda={torch.cuda.is_available()}; free_disk={free:.1f} GiB"); sys.exit(0 if torch.cuda.is_available() and free >= 10 else 1)'; or die "CUDA and at least 10 GiB free disk are required"

cli fit-anyloc-small-vocabulary \
    --registry "$registry" --model anyloc-dinov2-s-12288 \
    --output "$vocabulary" --samples 20000 \
    --patches-per-image 32 --seed 0 --batch-size 16 --workers 4; or exit 1

cli fit-reference-pca \
    --config "$registry" --registry "$registry" \
    --model anyloc-dinov2-s-12288 \
    --output "$anyloc_pca" --cache "$anyloc_cache" \
    --components 512 --samples 20000 --seed 0 \
    --batch-size 16 --workers 4; or exit 1

for model in $models
    cli evaluate-reference-suite \
        --config "$registry" --model "$model" \
        --database-encoder adapted --transform native \
        --datasets $protocols; or exit 1
end

if resources_complete
    echo "[skip] valid Small-reference resource benchmark: $resources"
else
    cli benchmark-reference-models \
        --config "$registry" --models $models \
        --warmup 20 --iterations 100 --output "$resources"; or exit 1
end

cli audit-reference-coverage \
    --registry "$registry" --reference-root runs/reference \
    --resources "$resources" --output runs/reference/coverage_small.json; or exit 1

"$python_bin" -c 'import json, sys; p=json.load(open("runs/reference/coverage_small.json")); sys.exit(0 if p.get("complete") else 1)'; or die "Small-reference coverage remains incomplete"
cli summarize-small-references \
    --registry "$registry" --reference-root runs/reference \
    --resources "$resources" --models $models \
    --output runs/comparisons/dinov2_small_summary.json; or exit 1
echo "Done: runs/reference/coverage_small.json"
echo "Done: runs/comparisons/dinov2_small_summary.json"
