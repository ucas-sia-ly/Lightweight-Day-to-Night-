#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set train_config configs/experiments/students/boq_dinov2_small_12288.yaml
set registry configs/references/small.yaml
set output runs/reference/boq-dinov2-s-12288-distilled
set pca data/reference_pca/boq_dinov2_s_distilled_train_pca512.pt
set pca_cache data/reference_pca/boq_dinov2_s_distilled_train_raw_fp16.npy
set protocols robotcar_project svox_night msls_night tokyo_night robotcar_day svox_day svox_sun tokyo_day

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

function training_complete
    test -f "$output/model_best.pt"; and test -f "$output/checkpoints/latest.pt"; or return 1
    "$python_bin" -c 'import sys, torch; s=torch.load(sys.argv[1], map_location="cpu", weights_only=False); ok = int(s.get("epoch", -1)) == 40 and int(s.get("descriptor_dim", -1)) == 12288 and s.get("official_checkpoint") is False and s.get("training_status") == "re-trained-and-distilled" and bool(s.get("teacher_checkpoint_sha256")); sys.exit(0 if ok else 1)' "$output/checkpoints/latest.pt"
end

command -s "$python_bin" >/dev/null; or die "missing interpreter: $python_bin"
test -f /home/admin123/diffusion_vpr_test/reference_code/weights/dinov2_vits14_reg4_pretrain.pth; or die "missing DINOv2-S-Reg4 checkpoint"
test -f /home/admin123/diffusion_vpr_test/reference_code/weights/dinov2_12288.pth; or die "missing BoQ-B teacher checkpoint"
test -d /home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/dataset/gsv-cities/Images; or die "missing GSV-Cities images"
no_competing_training; or die "another VPR/Small-reference job is running; wait before starting BoQ-S"
gpu_idle; or die "GPU 0 has active compute processes; wait before starting BoQ-S"
"$python_bin" -c 'import shutil, sys, torch; free=shutil.disk_usage(".").free/2**30; print(f"cuda={torch.cuda.is_available()}; free_disk={free:.1f} GiB"); sys.exit(0 if torch.cuda.is_available() and free >= 10 else 1)'; or die "CUDA and at least 10 GiB free disk are required"

if training_complete
    echo "[skip] completed BoQ-S training: $output"
else
    echo "[train/resume] BoQ-S"
    cli train-boq-small --config "$train_config"; or exit 1
    training_complete; or die "BoQ-S did not complete 40 epochs"
end

cli fit-reference-pca \
    --config "$registry" --registry "$registry" \
    --model boq-dinov2-s-12288 \
    --output "$pca" --cache "$pca_cache" \
    --components 512 --samples 20000 --seed 0 \
    --batch-size 8 --workers 4; or exit 1

for model in boq-dinov2-s-12288 boq-dinov2-s-pca512
    cli evaluate-reference-suite \
        --config "$registry" --model "$model" \
        --database-encoder adapted --transform native \
        --datasets $protocols; or exit 1
end

echo "Done: $output/model_best.pt"
echo "Done: $pca"
