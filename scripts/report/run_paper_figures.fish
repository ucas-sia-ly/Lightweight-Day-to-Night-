#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1

set -gx PYTHONPATH "$root/src"
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set paper_dir runs/paper
set selection runs/qualitative_top1/selection_all_methods.json
set result_summary runs/comparisons/cdnd_multiseed_ablation_summary.json
if not test -f "$result_summary"
    set result_summary runs/vpr/cdnd_512/salad_no_gate_seed0/final_suite/summary.json
end

command -s "$python_bin" >/dev/null; or begin
    echo "ERROR: missing interpreter: $python_bin" >&2
    exit 1
end

set resource_args --resource-summary runs/reference/resource_benchmark.json
if test -f runs/reference/resource_benchmark_completion.json
    set -a resource_args --resource-summary runs/reference/resource_benchmark_completion.json
end

"$python_bin" -m a2b_lightvpr.cli plot-paper-efficiency \
    --student-summary "$result_summary" \
    --reference-root runs/reference \
    $resource_args \
    --compact-summary runs/comparisons/foundation_small_paper_summary.json \
    --output "$paper_dir/figures/accuracy_efficiency.pdf"; or exit 1

if not test -f "$selection"
    fish scripts/report/run_qualitative_top1.fish; or exit 1
end

"$python_bin" -m a2b_lightvpr.cli export-qualitative-top1 \
    --config configs/vpr/cdnd_salad_512.yaml \
    --selection "$selection" \
    --output-dir "$paper_dir/figures/retrieval" \
    --cache-dir runs/qualitative_top1/cache \
    --models baseline cdnd \
    --batch-size 32 \
    --workers 4; or exit 1

echo "Done: $paper_dir/figures/accuracy_efficiency.pdf"
echo "Done: $paper_dir/figures/retrieval/comparison_all.jpg"
