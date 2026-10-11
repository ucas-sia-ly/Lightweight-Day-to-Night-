#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set -g python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set -g cfg configs/experiments/single_dataset/robotcar_only_cdnd.yaml
set stage all
if test (count $argv) -gt 0
    set stage $argv[1]
end
contains -- $stage all prepare generator translate teacher train evaluate summarize; or exit 2
mkdir -p runs/vpr/robotcar_only_512/logs
function run_logged
    set tag $argv[1]
    $python_bin $argv[2..-1] 2>&1 | tee "runs/vpr/robotcar_only_512/logs/$tag.log"
    set codes $pipestatus
    if test $codes[1] -ne 0
        exit $codes[1]
    end
end
if contains -- $stage all prepare
    run_logged prepare scripts/data/prepare_robotcar.py prepare
end
if contains -- $stage all generator
    set resume_args
    set latest runs/generator/robotcar_only_qsattn_seed0/checkpoints/latest.pt
    if test -f $latest
        set resume_args --set "generator.resume=$latest"
    end
    run_logged generator -m a2b_lightvpr.cli train-generator --config $cfg $resume_args
end
if contains -- $stage all translate
    run_logged translate -m a2b_lightvpr.cli translate-train --config $cfg
end
if contains -- $stage all teacher
    run_logged teacher -m a2b_lightvpr.cli cache-vpr-teacher --config $cfg
end
if contains -- $stage all train
    for group in day_only day_fake cdnd
        set resume_args
        set latest "runs/vpr/robotcar_only_512/"$group"_seed0/checkpoints/latest.pt"
        if test -f "$latest"
            set resume_args --set "vpr.resume=$latest"
        end
        run_logged "train_$group" -m a2b_lightvpr.cli train-vpr --config "configs/experiments/single_dataset/robotcar_only_$group.yaml" $resume_args
    end
end
if contains -- $stage all evaluate
    for group in day_only day_fake cdnd
        run_logged "eval_$group" -m a2b_lightvpr.cli evaluate-trained-suite --config "configs/experiments/single_dataset/robotcar_only_$group.yaml" \
            --checkpoint "runs/vpr/robotcar_only_512/"$group"_seed0/model_best.pt" --datasets robotcar_project robotcar_day
    end
    run_logged official -m a2b_lightvpr.cli evaluate-reference-suite --config configs/references/robotcar_only_official.yaml \
        --model mixvpr-robotcar-official --database-encoder adapted --transform unified \
        --output-dir runs/vpr/robotcar_only_512/official --datasets robotcar_project robotcar_day
end
if contains -- $stage all summarize
    run_logged summarize scripts/data/prepare_robotcar.py summarize
end
