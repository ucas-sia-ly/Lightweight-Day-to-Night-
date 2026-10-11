#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"

set config configs/vpr/cdnd_salad_512.yaml
set selection runs/qualitative_top1/selection_all_methods.json
set output runs/qualitative_top1/figures/all_methods
set cache runs/qualitative_top1/all_methods_cache
set log runs/qualitative_top1/export.log

command -s "$python_bin" >/dev/null; or begin
    echo "ERROR: missing interpreter: $python_bin" >&2
    exit 1
end
mkdir -p (dirname "$log")
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set -gx OMP_NUM_THREADS 4

echo "Searching all four night datasets for nine methods; CDND must be correct"
"$python_bin" -m a2b_lightvpr.cli export-qualitative-top1 \
    --config "$config" \
    --search \
    --selection "$selection" \
    --output-dir "$output" \
    --cache-dir "$cache" \
    --batch-size 32 \
    --workers 4 \
    --models cdnd cosplace eigenplaces official_mixvpr512 nocplace anyloc_pca crica_pca salad boq_pca \
    2>&1 | tee "$log"
set status_code $pipestatus[1]
if test $status_code -ne 0
    echo "ERROR: qualitative export failed; inspect $log" >&2
    exit $status_code
end

echo "Done: $output/top1_results.json"
