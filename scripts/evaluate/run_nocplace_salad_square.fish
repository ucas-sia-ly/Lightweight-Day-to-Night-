#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set registry configs/references/nocplace_salad_square.yaml
set models nocplace-r50-clean5k-resize320 salad-dinov2-s-gsv-eval322
mkdir -p runs/reference/square_retest
for model in $models
    "$python_bin" -m a2b_lightvpr.cli evaluate-reference-suite \
        --config "$registry" --model "$model" --database-encoder adapted --transform native \
        --datasets robotcar_project svox_night tokyo_night msls_night robotcar_day svox_day svox_sun tokyo_day msls_day_val \
        2>&1 | tee "runs/reference/square_retest/$model.log"
    set codes $pipestatus
    if test "$codes[1]" -ne 0
        exit $codes[1]
    end
end
"$python_bin" -m a2b_lightvpr.cli benchmark-reference-models \
    --config "$registry" --models $models --warmup 20 --iterations 100 \
    --output runs/reference/square_retest/resources.json
