#!/usr/bin/env fish

set -l root (path resolve (dirname (status filename))/..)
cd $root
mkdir -p logs/paper

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

set -l names day_only fake_only day_fake
set -l configs \
    configs/vpr_loss_official_day_only_seed0.yaml \
    configs/vpr_loss_official_fake_only_seed0.yaml \
    configs/vpr_loss_official_day_fake_seed0.yaml

for index in (seq (count $names))
    set -l name $names[$index]
    set -l config $configs[$index]
    set -l run_dir runs/vpr/loss_ablation_official/$name'_seed0'
    set -l log logs/paper/train_official_loss_$name.log
    if completed $run_dir
        echo "SKIP completed: $name"
        continue
    end
    echo "TRAIN $name -> $log"
    a2b-lightvpr train-vpr --config $config 2>&1 | tee $log
    or begin
        echo "FAILED: $name; inspect $log"
        exit 1
    end
end

echo "Training complete. Run each frozen model through evaluate-trained-suite next."
