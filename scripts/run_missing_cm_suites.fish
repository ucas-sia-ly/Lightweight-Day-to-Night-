#!/usr/bin/env fish

set -l root (path resolve (dirname (status filename))/..)
cd $root

set -l env_bin /home/admin123/miniconda3/envs/img2img-turbo/bin
set -gx PATH $env_bin $PATH
set -gx PYTHONUNBUFFERED 1
mkdir -p logs/paper

set -l names day_only fake_only
set -l configs \
    configs/vpr_loss_official_day_only_compute_matched_seed0.yaml \
    configs/vpr_loss_official_fake_only_compute_matched_seed0.yaml
set -l runs \
    runs/vpr/loss_ablation_official_compute_matched/day_only_seed0 \
    runs/vpr/loss_ablation_official_compute_matched/fake_only_seed0
set -l datasets robotcar_project svox_night msls_night tokyo_night tokyo_sunset

for index in (seq (count $names))
    set -l complete true
    for dataset in $datasets
        if not test -s $runs[$index]/final_suite/$dataset/metrics.json
            set complete false
            break
        end
    end

    if test $complete = true
        echo "SKIP complete formal night/transition suite: $names[$index]"
        continue
    end

    echo "RUN formal night/transition suite: $names[$index]"
    a2b-lightvpr evaluate-trained-suite \
        --config $configs[$index] \
        --checkpoint $runs[$index]/model_best.pt \
        --datasets all \
        2>&1 | tee logs/paper/evaluate_night_$names[$index]_compute_matched.log
    or begin
        echo "FAILED: inspect logs/paper/evaluate_night_$names[$index]_compute_matched.log"
        exit 1
    end
end

python scripts/export_paper_tables.py
or exit 1

lualatex -interaction=nonstopmode -halt-on-error \
    -output-directory=paper paper/a2b_lightvpr_中文论文草稿.tex >/dev/null
or exit 1
lualatex -interaction=nonstopmode -halt-on-error \
    -output-directory=paper paper/a2b_lightvpr_中文论文草稿.tex >/dev/null
or exit 1

echo "DONE: compute-matched suites, generated tables, and paper PDF are current."
