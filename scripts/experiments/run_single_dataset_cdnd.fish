#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set -g python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set datasets svox msls
if test (count $argv) -gt 0
    contains -- $argv[1] svox msls; or exit 2
    set datasets $argv[1]
end
for ds in $datasets
    mkdir -p "runs/vpr/"$ds"_only_512/logs"
    $python_bin scripts/data/prepare_single_dataset.py $ds; or exit 1
    set cfg "configs/experiments/single_dataset/"$ds"_only_cdnd.yaml"
    $python_bin -m a2b_lightvpr.cli cache-vpr-teacher --config "$cfg"; or exit 1
    set protocols msls_night msls_day_val
    if test $ds = svox
        set protocols svox_night svox_day svox_sun
    end
    for group in day_only day_fake cdnd
        set cfg "configs/experiments/single_dataset/"$ds"_only_"$group".yaml"
        set out "runs/vpr/"$ds"_only_512/"$group"_seed0"
        set resume_args
        if test -f "$out/checkpoints/latest.pt"
            set resume_args --set "vpr.resume=$out/checkpoints/latest.pt"
        end
        $python_bin -m a2b_lightvpr.cli train-vpr --config "$cfg" $resume_args 2>&1 | tee "runs/vpr/"$ds"_only_512/logs/"$group"_train.log"
        set codes $pipestatus
        if test $codes[1] -ne 0
            exit $codes[1]
        end
        $python_bin -m a2b_lightvpr.cli evaluate-trained-suite --config "$cfg" --checkpoint "$out/model_best.pt" --datasets $protocols; or exit 1
    end
    $python_bin scripts/data/prepare_single_dataset.py $ds --summarize; or exit 1
end
