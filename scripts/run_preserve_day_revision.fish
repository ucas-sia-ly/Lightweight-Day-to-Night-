#!/usr/bin/env fish

set -l root (path resolve (dirname (status filename))/..)
cd $root
mkdir -p logs/paper runs/comparisons

set -l env_bin /home/admin123/miniconda3/envs/img2img-turbo/bin
set -gx PATH $env_bin $PATH
set -gx PYTHONUNBUFFERED 1

for executable in a2b-lightvpr python
    if not test -x $env_bin/$executable
        echo "MISSING executable: $env_bin/$executable"
        exit 1
    end
end
for executable in jq tee
    if not type -q $executable
        echo "MISSING executable on PATH: $executable"
        exit 1
    end
end

function run_logged --argument-names log_file
    echo "RUN -> $log_file"
    command $argv[2..] 2>&1 | tee $log_file
    or begin
        echo "FAILED; inspect $log_file"
        exit 1
    end
end

function training_complete --argument-names run_dir expected_samples
    for summary in $run_dir/wandb/run-*/files/wandb-summary.json
        if test -s $summary; and jq -e \
            ".convergence_status == \"completed_fixed_budget\" and .samples_seen == $expected_samples" \
            $summary >/dev/null
            return 0
        end
    end
    if test -s $run_dir/latest.pt
        python -c \
            'import sys, torch; s=torch.load(sys.argv[1], map_location="cpu", weights_only=False); raise SystemExit(0 if int(s.get("epoch", 0)) >= 20 and int(s.get("samples_seen", 0)) >= int(sys.argv[2]) else 1)' \
            $run_dir/latest.pt $expected_samples
        and return 0
    end
    return 1
end

set -l cm_names day_only fake_only
set -l cm_configs \
    configs/vpr_loss_official_day_only_compute_matched_seed0.yaml \
    configs/vpr_loss_official_fake_only_compute_matched_seed0.yaml
set -l cm_runs \
    runs/vpr/loss_ablation_official_compute_matched/day_only_seed0 \
    runs/vpr/loss_ablation_official_compute_matched/fake_only_seed0

if contains -- --preflight $argv
    echo "Environment: "(command -s a2b-lightvpr)
    for config in $cm_configs \
        configs/vpr_loss_official_day_fake_seed0.yaml \
        configs/vpr_mixvpr_r50_512_qsattn_official_60k.yaml
        python -c \
            'import sys; from a2b_lightvpr.config import load_config; c=load_config(sys.argv[1]); print(f"OK config: {sys.argv[1]} ({c.vpr.samples_per_epoch} samples/epoch, accumulation {c.vpr.gradient_accumulation_steps})")' \
            $config
        or exit 1
    end
    for checkpoint in \
        runs/vpr/loss_ablation_official/day_fake_seed0/model_best.pt \
        runs/vpr/qsattn_official/mixvpr_r50_512_seed0/model_best.pt
        if not test -s $checkpoint
            echo "MISSING checkpoint: $checkpoint"
            exit 1
        end
        echo "OK checkpoint: $checkpoint"
    end
    a2b-lightvpr audit-reference-models \
        --config configs/reference_models_extended.yaml \
        --output /tmp/a2b-reference-audit.json >/dev/null
    or exit 1
    a2b-lightvpr audit-reference-models \
        --config configs/reference_models_three_dataset.yaml \
        --output /tmp/a2b-reference-audit-three.json >/dev/null
    or exit 1
    for key in mixvpr-r50-512 salad-dinov2-544 boq-dinov2-12288
        jq -e ".models[\"$key\"].ready == true" /tmp/a2b-reference-audit.json >/dev/null
        or begin
            echo "Reference model is not ready: $key"
            exit 1
        end
    end
    jq -e '.models["nocplace-r50-ikt-clean5k"].ready == true' \
        /tmp/a2b-reference-audit-three.json >/dev/null
    or begin
        echo "Reference model is not ready: nocplace-r50-ikt-clean5k"
        exit 1
    end
    echo "PREFLIGHT PASSED. No training or descriptor extraction was started."
    exit 0
end

for index in (seq (count $cm_names))
    set -l run_dir $cm_runs[$index]
    if not training_complete $run_dir 2400000
        set -l resume_args
        if test -s $run_dir/latest.pt
            set resume_args --set vpr.resume=$run_dir/latest.pt
            echo "RESUME incomplete training: $run_dir/latest.pt"
        end
        run_logged logs/paper/train_$cm_names[$index]_compute_matched.log \
            a2b-lightvpr train-vpr --config $cm_configs[$index] $resume_args
    else
        echo "SKIP completed compute-matched training: $run_dir"
    end
end

# Evaluate the additional night validation stratum; day/test sets cannot affect selection.
set -l selection_names day_fake day_fake_preserve
set -l selection_configs \
    configs/vpr_loss_official_day_fake_seed0.yaml \
    configs/vpr_mixvpr_r50_512_qsattn_official_60k.yaml
set -l selection_runs \
    runs/vpr/loss_ablation_official/day_fake_seed0 \
    runs/vpr/qsattn_official/mixvpr_r50_512_seed0

for index in (seq (count $selection_names))
    run_logged logs/paper/validate_night_$selection_names[$index].log \
        a2b-lightvpr evaluate-trained-suite \
        --config $selection_configs[$index] \
        --checkpoint $selection_runs[$index]/model_best.pt \
        --datasets msls_night
end

set -l preserve_run $selection_runs[2]
set -l baseline_run $selection_runs[1]
set -l preserve_suite $preserve_run/final_suite/summary.json
set -l baseline_suite $baseline_run/final_suite/summary.json
set -l preserve_epoch (jq -er '.checkpoint_epoch' $preserve_suite); or exit 1
set -l baseline_epoch (jq -er '.checkpoint_epoch' $baseline_suite); or exit 1
set -l selection_output runs/comparisons/official_loss_preserve_vs_day_fake_selection.json

a2b-lightvpr compare-loss-selection \
    --candidate-validation $preserve_run/validation/epoch_(printf '%03d' $preserve_epoch).json \
    --baseline-validation $baseline_run/validation/epoch_(printf '%03d' $baseline_epoch).json \
    --candidate-suite $preserve_suite \
    --baseline-suite $baseline_suite \
    --candidate-name day+fake+preserve \
    --baseline-name day+fake \
    --output $selection_output \
    --samples 10000 \
    --seed 0
or exit 1

# Only after the choice is frozen, evaluate daytime test conditions for all ablations.
set -l day_names day_only_cm fake_only_cm day_fake day_fake_preserve
set -l day_configs $cm_configs[1] $cm_configs[2] $selection_configs[1] $selection_configs[2]
set -l day_runs $cm_runs[1] $cm_runs[2] $selection_runs[1] $selection_runs[2]
for index in (seq (count $day_names))
    run_logged logs/paper/evaluate_day_$day_names[$index].log \
        a2b-lightvpr evaluate-trained-suite \
        --config $day_configs[$index] \
        --checkpoint $day_runs[$index]/model_best.pt \
        --datasets all-day
end

# Compute-matched single-domain models also need the original six-condition suite.
for index in 1 2
    run_logged logs/paper/evaluate_night_$day_names[$index].log \
        a2b-lightvpr evaluate-trained-suite \
        --config $day_configs[$index] \
        --checkpoint $day_runs[$index]/model_best.pt \
        --datasets all
end

run_logged logs/paper/evaluate_day_mixvpr_zero_shot.log \
    a2b-lightvpr evaluate-reference-suite \
    --config configs/reference_models_extended.yaml \
    --model mixvpr-r50-512 \
    --database-encoder adapted \
    --transform native \
    --datasets all-day

python scripts/export_paper_tables.py
echo "DONE. Selection:"
jq '{selected, macro_recall_at_1, stratified_paired_bootstrap_r1}' $selection_output
