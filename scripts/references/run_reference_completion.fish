#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set config configs/references/official.yaml
set pca data/reference_pca/boq_train20k_pca512.pt
set pca_cache data/reference_pca/boq_train20k_raw_fp16.npy
set completion_resources runs/reference/resource_benchmark_completion.json

function die
    echo "ERROR: $argv" >&2
    exit 1
end

function cli
    command "$python_bin" -m a2b_lightvpr.cli $argv
end

command -s "$python_bin" >/dev/null; or die "missing interpreter: $python_bin"
test -f /home/admin123/diffusion_vpr_test/reference_code/weights/dinov2_12288.pth; or die "missing BoQ checkpoint"
test -f /home/admin123/diffusion_vpr_test/reference_code/Patch-NetVLAD/pretrained_models/pitts_WPCA4096.pth.tar; or die "missing NetVLAD checkpoint"
"$python_bin" -c 'import shutil, sys, torch; free=shutil.disk_usage(".").free/2**30; print(f"cuda={torch.cuda.is_available()}; free_disk={free:.1f} GiB"); sys.exit(0 if torch.cuda.is_available() and free >= 6 else 1)'; or die "CUDA and at least 6 GiB free disk are required"

cli fit-reference-pca \
    --config "$config" \
    --registry "$config" \
    --model boq-dinov2-12288 \
    --output "$pca" \
    --cache "$pca_cache" \
    --components 512 \
    --samples 20000 \
    --seed 0 \
    --batch-size 16 \
    --workers 4; or exit 1

cli evaluate-reference-suite \
    --config "$config" \
    --model netvlad-vgg16-pitts4096 \
    --database-encoder adapted \
    --transform native \
    --datasets all-conditions; or exit 1

cli evaluate-reference-suite \
    --config "$config" \
    --model boq-dinov2-pca512 \
    --database-encoder adapted \
    --transform native \
    --datasets all-conditions; or exit 1

cli benchmark-reference-models \
    --config "$config" \
    --models mixvpr-r50-128 boq-dinov2-pca512 \
    --warmup 20 \
    --iterations 100 \
    --output "$completion_resources"; or exit 1

cli audit-reference-coverage \
    --registry "$config" \
    --reference-root runs/reference \
    --resources runs/reference/resource_benchmark.json \
    --resources "$completion_resources" \
    --output runs/reference/coverage.json; or exit 1

"$python_bin" -c 'import json, sys; p=json.load(open("runs/reference/coverage.json")); sys.exit(0 if p.get("complete") else 1)'; or die "reference coverage remains incomplete"
echo "Done: runs/reference/coverage.json"
