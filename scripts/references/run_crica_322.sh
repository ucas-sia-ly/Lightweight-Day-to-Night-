#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REGISTRY=configs/references/crica_small_322.yaml
"$PYTHON_BIN" scripts/references/train_crica_small_gsv.py --image-size 322 --output runs/reference/cricavpr-dinov2-s-gsv322-5376
"$PYTHON_BIN" -m a2b_lightvpr.cli fit-reference-pca --config "$REGISTRY" --registry "$REGISTRY" \
  --model cricavpr-dinov2-s-gsv322-5376 --output data/reference_pca/cricavpr_dinov2_s_gsv322_pca512.pt \
  --cache data/reference_pca/cricavpr_dinov2_s_gsv322_raw_fp16.npy --components 512 --samples 20000 \
  --seed 0 --batch-size 1 --workers 4
"$PYTHON_BIN" -m a2b_lightvpr.cli evaluate-reference-suite --config "$REGISTRY" \
  --model cricavpr-dinov2-s-gsv322-pca512 --database-encoder adapted --transform native \
  --datasets robotcar_project svox_night msls_night tokyo_night robotcar_day svox_day svox_sun tokyo_day
