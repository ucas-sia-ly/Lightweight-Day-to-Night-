#!/usr/bin/env fish

set -l root (path resolve (dirname (status filename))/..)
cd $root
mkdir -p logs/paper

set -l steps 50000 100000 125000
for step in $steps
    set -l padded (printf '%07d' $step)
    set -l checkpoint runs/generator/qsattn_official_objective_150k_seed0/checkpoints/state_$padded.pt
    set -l output runs/paper/generation/qsattn_official_tokyo_step_$step
    set -l metrics $output/metrics.json
    set -l log logs/paper/evaluate_qsattn_official_tokyo_step_$step.log

    if test -s $metrics
        echo "SKIP existing metrics: $metrics"
        continue
    end
    if not test -s $checkpoint
        echo "MISSING checkpoint: $checkpoint"
        exit 1
    end

    echo "EVALUATE official QS-Attn step $step -> $log"
    a2b-lightvpr evaluate-generation \
        --config configs/generator_qsattn_official_objective_150k.yaml \
        --checkpoint $checkpoint \
        --dataset tokyo \
        --output-dir $output \
        --timing-iterations 200 2>&1 | tee $log
    or begin
        echo "FAILED at step $step; inspect $log"
        exit 1
    end
end

echo "Existing 150k result: runs/paper/generation/qsattn_official_tokyo/metrics.json"
echo "Official QS-Attn checkpoint evaluation complete."
