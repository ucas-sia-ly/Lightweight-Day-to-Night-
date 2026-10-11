#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set config configs/experiments/data_sources/cdnd_gsv_512.yaml
set output runs/vpr/cdnd_gsv_512/salad_seed0
mkdir -p "$output"
function cli
    "$python_bin" -m a2b_lightvpr.cli $argv
end
"$python_bin" scripts/data/prepare_gsv_cdnd.py; or exit 1
cli translate-train --config "$config"; or exit 1
cli cache-vpr-teacher --config "$config"; or exit 1
if test -f "$output/checkpoints/latest.pt"
    cli train-vpr --config "$config" --set "vpr.resume=$output/checkpoints/latest.pt"; or exit 1
else
    cli train-vpr --config "$config"; or exit 1
end
cli evaluate-trained-suite --config "$config" --checkpoint "$output/model_best.pt" \
    --datasets robotcar_project svox_night tokyo_night msls_night robotcar_day svox_day svox_sun tokyo_day msls_day_val; or exit 1
cli benchmark --config "$config" --checkpoint "$output/model_best.pt" --warmup 20 --iterations 100; or exit 1
echo "Complete: $output/final_suite/summary.json"
