#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set -gx OMP_NUM_THREADS 4
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set registry configs/references/day_completion.yaml
mkdir -p runs/reference/day_completion
for model in cosplace-r50-512-320 eigenplaces-r50-512-320 anyloc-dinov2-s-gsv-pca512 cricavpr-dinov2-s-gsv322-pca512 salad-dinov2-s-544-gsv boq-dinov2-s-pca512
    "$python_bin" -m a2b_lightvpr.cli evaluate-reference-suite \
        --config "$registry" --model "$model" \
        --database-encoder adapted --transform native --datasets msls_day_val \
        --output-dir "runs/reference/day_completion/$model" \
        2>&1 | tee "runs/reference/day_completion/$model.log"
    set codes $pipestatus
    if test "$codes[1]" -ne 0
        echo "Failed: $model; see runs/reference/day_completion/$model.log" >&2
        exit 1
    end
end
echo "Six missing MSLS day-val evaluations complete. NocPlace still requires a verified official checkpoint."
