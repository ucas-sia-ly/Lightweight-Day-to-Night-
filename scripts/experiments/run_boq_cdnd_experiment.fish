#!/usr/bin/env fish
set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set config configs/experiments/students/cdnd_boq_small_12288.yaml
set baseline_config configs/experiments/students/day_fake_boq_small_12288.yaml

function cli
    command "$python_bin" -m a2b_lightvpr.cli $argv
end

function complete_run
    set -l output_dir $argv[1]
    test -f "$output_dir/model_best.pt"; and test -f "$output_dir/checkpoints/latest.pt"
end

if complete_run runs/vpr/boq_small_12288/day_fake_seed0
    echo "[skip] completed BoQ-S day+fake baseline"
else
    echo "[train/resume] BoQ-S day+fake baseline"
    cli train-vpr --config "$baseline_config"; or exit 1
end
if test -f data/teacher_cache/boq_dinov2_b_qsattn_official_150k_fp16.npy
    echo "[skip] BoQ-B teacher cache exists"
else
    echo "[cache] building BoQ-B teacher candidate descriptors"
    cli cache-vpr-teacher --config "$config"; or exit 1
end

echo "[train/resume] BoQ-S with CDND neighborhood loss"
if complete_run runs/vpr/boq_small_12288/cdnd_boq_base_seed0
    echo "[skip] completed BoQ-S CDND"
else
    cli train-vpr --config "$config"; or exit 1
end
echo "Done: runs/vpr/boq_small_12288/cdnd_boq_base_seed0"
