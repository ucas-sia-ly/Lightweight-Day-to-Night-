#!/usr/bin/env fish
set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set -gx CUDA_VISIBLE_DEVICES 0
set -gx PYTHONPATH "$root/src"

command -s "$python_bin" >/dev/null; or begin
    echo "ERROR: missing interpreter: $python_bin" >&2
    exit 1
end
test -f /home/admin123/diffusion_vpr_test/reference_code/salad/dinov2_vits14_pretrain.pth; or begin
    echo "ERROR: missing official DINOv2-S checkpoint" >&2
    exit 1
end
test -d /home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/dataset/gsv-cities/Images; or begin
    echo "ERROR: missing GSV-Cities Images directory" >&2
    exit 1
end

"$python_bin" scripts/references/train_salad_small_gsv.py \
    --output runs/reference/salad-dinov2-s-544-gsv \
    --epochs 4 --batch-size 60 --workers 8
