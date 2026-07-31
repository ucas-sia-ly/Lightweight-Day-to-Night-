#!/usr/bin/env fish

set -l root (path resolve (dirname (status filename))/..)
cd $root

set -l check_only 0
if contains -- --check $argv
    set check_only 1
end

function report --argument-names label marker
    if test -s $marker
        echo "[DONE] $label"
    else
        echo "[PENDING] $label -> $marker"
    end
end

echo "A2B-LightVPR seed-0 paper workflow"
report "Official QS-Attn generator" \
    runs/generator/qsattn_official_objective_150k_seed0/generator_final.pt
report "Official QS-Attn translated manifest" \
    data/generated/multidataset_qsattn_official_150k/manifest.csv
report "Day-only loss run" \
    runs/vpr/loss_ablation_official/day_only_seed0/validation/epoch_020.json
report "Fake-only loss run" \
    runs/vpr/loss_ablation_official/fake_only_seed0/validation/epoch_020.json
report "Day+fake loss run" \
    runs/vpr/loss_ablation_official/day_fake_seed0/validation/epoch_020.json
report "Loss selection" \
    runs/comparisons/official_loss_preserve_vs_day_fake_selection.json
report "Official QS-Attn 50k metrics" \
    runs/paper/generation/qsattn_official_tokyo_step_50000/metrics.json
report "Official QS-Attn 100k metrics" \
    runs/paper/generation/qsattn_official_tokyo_step_100000/metrics.json
report "Official QS-Attn 125k metrics" \
    runs/paper/generation/qsattn_official_tokyo_step_125000/metrics.json
report "NetVLAD adapted suite" \
    runs/vpr/netvlad_safe/netvlad_vgg16_4096_seed0/final_suite/summary.json

if test $check_only -eq 1
    echo "Check complete. No long command was started."
    exit 0
end

for prerequisite in \
    runs/generator/qsattn_official_objective_150k_seed0/generator_final.pt \
    data/generated/multidataset_qsattn_official_150k/manifest.csv \
    runs/vpr/qsattn_official/mixvpr_r50_512_seed0/validation/epoch_020.json
    if not test -s $prerequisite
        echo "Missing prerequisite: $prerequisite"
        exit 1
    end
end

fish scripts/run_official_loss_ablation.fish
or exit 1
fish scripts/run_official_loss_selection.fish
or exit 1
fish scripts/evaluate_official_qsattn_checkpoints.fish
or exit 1

set -l netvlad_summary runs/vpr/netvlad_safe/netvlad_vgg16_4096_seed0/final_suite/summary.json
if test -s $netvlad_summary
    echo "[SKIP] existing NetVLAD suite"
else
    mkdir -p logs/paper
    a2b-lightvpr evaluate-trained-suite \
        --config configs/vpr_netvlad_vgg16_4096_safe_60k.yaml \
        --checkpoint runs/vpr/netvlad_safe/netvlad_vgg16_4096_seed0/model_best.pt \
        --datasets all 2>&1 | tee logs/paper/eval_netvlad_adapted.log
    or exit 1
end

python scripts/export_paper_tables.py --output-dir paper/generated_tables
or exit 1

echo "All remaining seed-0 experiments and paper tables completed."
