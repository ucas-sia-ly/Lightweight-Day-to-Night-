#!/usr/bin/env fish

set -l root (path resolve (dirname (status filename))/..)
cd $root
mkdir -p logs/paper runs/comparisons

set -l names day_only fake_only day_fake day_fake_preserve
set -l configs \
    configs/vpr_loss_official_day_only_seed0.yaml \
    configs/vpr_loss_official_fake_only_seed0.yaml \
    configs/vpr_loss_official_day_fake_seed0.yaml \
    configs/vpr_mixvpr_r50_512_qsattn_official_60k.yaml
set -l run_dirs \
    runs/vpr/loss_ablation_official/day_only_seed0 \
    runs/vpr/loss_ablation_official/fake_only_seed0 \
    runs/vpr/loss_ablation_official/day_fake_seed0 \
    runs/vpr/qsattn_official/mixvpr_r50_512_seed0

function completed --argument-names run_dir
    set -l summary_files $run_dir/wandb/run-*/files/wandb-summary.json
    for summary in $summary_files
        if test -f $summary
            if jq -e '.convergence_status == "completed_fixed_budget" and .samples_seen == 1200000' $summary >/dev/null
                return 0
            end
        end
    end
    return 1
end

for index in (seq (count $names))
    set -l name $names[$index]
    set -l config $configs[$index]
    set -l run_dir $run_dirs[$index]
    set -l checkpoint $run_dir/model_best.pt
    set -l summary $run_dir/final_suite/summary.json
    set -l log logs/paper/evaluate_official_loss_$name.log

    if not completed $run_dir
        echo "INCOMPLETE fixed training budget: $run_dir"
        echo "Run fish scripts/run_official_loss_ablation.fish first."
        exit 1
    end
    if test -s $summary
        echo "SKIP existing suite: $summary"
        continue
    end
    if not test -s $checkpoint
        echo "MISSING checkpoint: $checkpoint"
        echo "Run fish scripts/run_official_loss_ablation.fish first."
        exit 1
    end

    echo "EVALUATE $name -> $log"
    a2b-lightvpr evaluate-trained-suite \
        --config $config \
        --checkpoint $checkpoint \
        --datasets msls_night 2>&1 | tee $log
    or begin
        echo "FAILED: $name; inspect $log"
        exit 1
    end
end

set -l preserve_run $run_dirs[4]
set -l baseline_run $run_dirs[3]
set -l preserve_suite $preserve_run/final_suite/summary.json
set -l baseline_suite $baseline_run/final_suite/summary.json

for summary in $preserve_suite $baseline_suite
    if not test -s $summary
        echo "MISSING suite summary: $summary"
        exit 1
    end
end

set -l preserve_epoch (jq -er '.checkpoint_epoch' $preserve_suite)
or exit 1
set -l baseline_epoch (jq -er '.checkpoint_epoch' $baseline_suite)
or exit 1
set -l preserve_validation $preserve_run/validation/epoch_(printf '%03d' $preserve_epoch).json
set -l baseline_validation $baseline_run/validation/epoch_(printf '%03d' $baseline_epoch).json

for validation in $preserve_validation $baseline_validation
    if not test -s $validation
        echo "MISSING matching best-epoch validation: $validation"
        exit 1
    end
end

set -l output runs/comparisons/official_loss_preserve_vs_day_fake_selection.json
a2b-lightvpr compare-loss-selection \
    --candidate-validation $preserve_validation \
    --baseline-validation $baseline_validation \
    --candidate-suite $preserve_suite \
    --baseline-suite $baseline_suite \
    --candidate-name day+fake+preserve \
    --baseline-name day+fake \
    --output $output \
    --samples 10000 \
    --seed 0
or exit 1

echo "Selection result: $output"
jq '{selected, validation, macro_recall_at_1, stratified_paired_bootstrap_r1}' $output
