#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set expected_samples 1198720
set baseline_config configs/experiments/students/salad_small_day_fake_544.yaml
set cdnd_config configs/experiments/students/salad_small_cdnd_544.yaml

function die
    echo "ERROR: $argv" >&2
    exit 1
end

function cli
    command "$python_bin" -m a2b_lightvpr.cli $argv
end

function run_complete --argument-names directory
    set latest "$directory/checkpoints/latest.pt"
    test -f "$directory/model_best.pt"; and test -f "$latest"; or return 1
    "$python_bin" -c 'import sys, torch; s=torch.load(sys.argv[1], map_location="cpu", weights_only=False); sys.exit(0 if int(s.get("samples_seen", -1)) == int(sys.argv[2]) else 1)' "$latest" "$expected_samples"
end

function train_and_evaluate --argument-names config directory
    if run_complete "$directory"
        echo "[skip] completed training: $directory"
    else
        set train_args
        set latest "$directory/checkpoints/latest.pt"
        if test -f "$latest"
            echo "[resume] $directory from $latest"
            set train_args --set "vpr.resume=$latest"
        else
            echo "[train] $directory"
        end
        cli train-vpr --config "$config" $train_args; or exit 1
        run_complete "$directory"; or die "$directory did not reach samples_seen=$expected_samples"
    end
    cli evaluate-trained-suite \
        --config "$config" \
        --checkpoint "$directory/model_best.pt" \
        --datasets all-conditions; or exit 1
end

command -s "$python_bin" >/dev/null; or die "missing interpreter: $python_bin"
test -f /home/admin123/diffusion_vpr_test/reference_code/weights/dinov2_vits14_reg4_pretrain.pth; or die "missing DINOv2-S-Reg checkpoint"
test -f /home/admin123/diffusion_vpr_test/reference_code/weights/dino_salad_512_32.ckpt; or die "missing SALAD-B checkpoint"
"$python_bin" -c 'import shutil, sys, torch; free=shutil.disk_usage(".").free/2**30; print(f"cuda={torch.cuda.is_available()}; free_disk={free:.1f} GiB"); sys.exit(0 if torch.cuda.is_available() and free >= 8 else 1)'; or die "CUDA and at least 8 GiB free disk are required"

cli audit-data --config "$cdnd_config"; or exit 1
command "$python_bin" -m pytest -q; or exit 1

set baseline_dir runs/vpr/salad_small_544/day_fake_seed0
set cdnd_dir runs/vpr/salad_small_544/cdnd_salad_base_seed0
train_and_evaluate "$baseline_config" "$baseline_dir"
train_and_evaluate "$cdnd_config" "$cdnd_dir"

cli summarize-paired-vpr \
    --baseline "$baseline_dir/final_suite/summary.json" \
    --candidate "$cdnd_dir/final_suite/summary.json" \
    --label salad_base_teacher_to_salad_small_student \
    --output runs/comparisons/salad_family_seed0_summary.json; or exit 1

echo "Done: runs/comparisons/salad_family_seed0_summary.json"
