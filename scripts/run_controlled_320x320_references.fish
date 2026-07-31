#!/usr/bin/env fish

set -l root (path resolve (dirname (status filename))/..)
cd $root
mkdir -p logs/paper runs/comparisons

set -l env_bin /home/admin123/miniconda3/envs/img2img-turbo/bin
set -gx PATH $env_bin $PATH
set -gx PYTHONUNBUFFERED 1

function run_logged --argument-names log_file
    echo "RUN -> $log_file"
    command $argv[2..] 2>&1 | tee $log_file
    or begin
        echo "FAILED; inspect $log_file"
        exit 1
    end
end

# Strictly controlled baseline: same 320x320 fit-and-pad preprocessing as A2B.
run_logged logs/paper/eval_mixvpr_official_fitpad_320x320.log \
    a2b-lightvpr evaluate-reference-suite \
    --config configs/reference_models_extended.yaml \
    --set vpr.image_height=320 \
    --set vpr.image_width=320 \
    --model mixvpr-r50-512 \
    --database-encoder adapted \
    --transform unified \
    --datasets all \
    --output-dir runs/reference/mixvpr-r50-512/final_suite/fitpad_320x320/adapted

for encoder in original adapted
    run_logged logs/paper/eval_nocplace_$encoder-fitpad_320x320.log \
        a2b-lightvpr evaluate-reference-suite \
        --config configs/reference_models_three_dataset.yaml \
        --set vpr.image_height=320 \
        --set vpr.image_width=320 \
        --model nocplace-r50-ikt-clean5k \
        --database-encoder $encoder \
        --transform unified \
        --datasets all \
        --output-dir runs/reference/nocplace-r50-ikt-clean5k/final_suite/fitpad_320x320/$encoder
end

set -l candidate runs/vpr/qsattn_official/mixvpr_r50_512_seed0/final_suite/summary.json
a2b-lightvpr compare-suite-bootstrap \
    --candidate $candidate \
    --baseline runs/reference/mixvpr-r50-512/final_suite/fitpad_320x320/adapted/summary.json \
    --output runs/comparisons/official_preserve_vs_mixvpr_fitpad_320x320.json \
    --samples 10000
or exit 1

for encoder in original adapted
    a2b-lightvpr compare-suite-bootstrap \
        --candidate $candidate \
        --baseline runs/reference/nocplace-r50-ikt-clean5k/final_suite/fitpad_320x320/$encoder/summary.json \
        --output runs/comparisons/official_preserve_vs_nocplace_$encoder"_fitpad_320x320.json" \
        --samples 10000
    or exit 1
end

python scripts/export_paper_tables.py
echo "DONE: controlled 320x320 fit-and-pad suites and paired comparisons."
