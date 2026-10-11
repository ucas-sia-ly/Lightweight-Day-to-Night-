#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set run_id (date -u +%Y%m%dT%H%M%SZ)
set output "runs/benchmarks/nine_methods_continuous_$run_id"
mkdir -p runs/benchmarks
"$python_bin" scripts/benchmark/benchmark_nine_methods.py --output "$output" $argv 2>&1 | tee "runs/benchmarks/nine_methods_continuous_$run_id.log"
set codes $pipestatus
exit $codes[1]
