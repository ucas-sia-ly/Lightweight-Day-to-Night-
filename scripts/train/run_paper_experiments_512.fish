#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1

set -gx PYTHONPATH "$root/src"
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set baseline_config configs/vpr/day_fake_512.yaml
set cdnd_config configs/vpr/cdnd_salad_512.yaml
set expected_samples 1198720

function die
    echo "ERROR: $argv" >&2
    exit 1
end

function cli
    command "$python_bin" -m a2b_lightvpr.cli $argv
end

function latest_checkpoint --argument-names directory
    if test -d "$directory/checkpoints"
        find "$directory/checkpoints" -maxdepth 1 -type f -name '*.pt' | sort -V | tail -n 1
    end
end

function run_complete --argument-names directory
    set latest "$directory/checkpoints/latest.pt"
    test -f "$directory/model_best.pt"; and test -f "$latest"; or return 1
    "$python_bin" -c 'import sys, torch; state=torch.load(sys.argv[1], map_location="cpu", weights_only=False); sys.exit(0 if int(state.get("samples_seen", -1)) == int(sys.argv[2]) else 1)' "$latest" "$expected_samples"
end

function variant_args --argument-names output seed variant
    set -l args --set "vpr.output_dir=$output" --set "vpr.random_seed=$seed"
    switch "$variant"
        case baseline cdnd
        case day_only
            set -a args --set vpr.train_domains=day
        case fake_to_day_only
            set -a args --set vpr.relation_day_weight=0
        case no_fifo
            set -a args --set vpr.relation_queue_size=0
        case no_self_mask
            set -a args --set vpr.relation_mask_self=false
        case shared_fifo
            set -a args --set vpr.relation_queue_scope=shared
        case '*'
            die "unknown experiment variant: $variant"
    end
    printf '%s\n' $args
end

function train_and_evaluate --argument-names config output seed variant
    set -l args (variant_args "$output" "$seed" "$variant")
    if run_complete "$output"
        echo "[skip] completed training: $output"
    else
        set -l latest (latest_checkpoint "$output")
        set -l train_args $args
        if test -n "$latest"
            echo "[resume] $output from $latest"
            set -a train_args --set "vpr.resume=$latest"
        else
            echo "[train] $output"
        end
        cli train-vpr --config "$config" $train_args; or exit 1
        run_complete "$output"; or die "$output did not reach samples_seen=$expected_samples"
    end

    echo "[evaluate] $output"
    cli evaluate-trained-suite \
        --config "$config" \
        $args \
        --checkpoint "$output/model_best.pt" \
        --datasets all-conditions; or exit 1
end

command -s "$python_bin" >/dev/null; or die "missing interpreter: $python_bin"
"$python_bin" -c 'import shutil, sys, torch; free=shutil.disk_usage(".").free/2**30; print(f"torch={torch.__version__}; cuda={torch.cuda.is_available()}; free_disk={free:.1f} GiB"); sys.exit(0 if torch.cuda.is_available() and free >= 15 else 1)'; or die "CUDA and at least 15 GiB free disk are required"
cli audit-data --config "$cdnd_config"; or exit 1
command "$python_bin" -m pytest -q; or exit 1

train_and_evaluate "$baseline_config" runs/vpr/cdnd_512/day_fake_pk_seed0 0 baseline
train_and_evaluate "$cdnd_config" runs/vpr/cdnd_512/salad_no_gate_seed0 0 cdnd

for seed in 1 2
    train_and_evaluate "$baseline_config" "runs/vpr/cdnd_512/day_fake_pk_seed$seed" "$seed" baseline
    train_and_evaluate "$cdnd_config" "runs/vpr/cdnd_512/salad_no_gate_seed$seed" "$seed" cdnd
end

train_and_evaluate "$baseline_config" runs/vpr/cdnd_512/ablation_day_only_seed0 0 day_only
train_and_evaluate "$cdnd_config" runs/vpr/cdnd_512/ablation_fake_to_day_only_seed0 0 fake_to_day_only
train_and_evaluate "$cdnd_config" runs/vpr/cdnd_512/ablation_no_fifo_seed0 0 no_fifo
train_and_evaluate "$cdnd_config" runs/vpr/cdnd_512/ablation_no_self_mask_seed0 0 no_self_mask
train_and_evaluate "$cdnd_config" runs/vpr/cdnd_512/ablation_shared_fifo_seed0 0 shared_fifo

cli summarize-cdnd-multiseed \
    --baseline 0=runs/vpr/cdnd_512/day_fake_pk_seed0/final_suite/summary.json \
    --baseline 1=runs/vpr/cdnd_512/day_fake_pk_seed1/final_suite/summary.json \
    --baseline 2=runs/vpr/cdnd_512/day_fake_pk_seed2/final_suite/summary.json \
    --candidate 0=runs/vpr/cdnd_512/salad_no_gate_seed0/final_suite/summary.json \
    --candidate 1=runs/vpr/cdnd_512/salad_no_gate_seed1/final_suite/summary.json \
    --candidate 2=runs/vpr/cdnd_512/salad_no_gate_seed2/final_suite/summary.json \
    --ablation day_only=runs/vpr/cdnd_512/ablation_day_only_seed0/final_suite/summary.json \
    --ablation fake_to_day_only=runs/vpr/cdnd_512/ablation_fake_to_day_only_seed0/final_suite/summary.json \
    --ablation no_fifo=runs/vpr/cdnd_512/ablation_no_fifo_seed0/final_suite/summary.json \
    --ablation no_self_mask=runs/vpr/cdnd_512/ablation_no_self_mask_seed0/final_suite/summary.json \
    --ablation shared_fifo=runs/vpr/cdnd_512/ablation_shared_fifo_seed0/final_suite/summary.json \
    --output runs/comparisons/cdnd_multiseed_ablation_summary.json; or exit 1

echo "Done: runs/comparisons/cdnd_multiseed_ablation_summary.json"
