#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
mkdir -p runs/reference/same_weights_224
"$python_bin" scripts/evaluate/evaluate_same_weights_224.py $argv 2>&1 | tee runs/reference/same_weights_224/evaluate.log
set codes $pipestatus
exit $codes[1]
