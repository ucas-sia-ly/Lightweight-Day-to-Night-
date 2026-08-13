#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set python /home/admin123/miniconda3/envs/img2img-turbo/bin/python
set force false
if contains -- --force $argv
    set force true
end

set config configs/vpr_cdnd_salad_512.yaml
set checkpoint runs/vpr/cdnd_512/salad_no_gate_seed0/model_best.pt
set output_dir runs/vpr/cdnd_512/salad_no_gate_seed0/robotcar_temporal
set cache_dir "$output_dir/cache"
set selection "$output_dir/val_selection.json"
set night "$output_dir/night_test.json"
set day "$output_dir/day_test.json"
set ablations "$output_dir/ablations.json"
set comparison "$output_dir/comparison.json"

function die
    echo "ERROR: $argv" >&2
    exit 1
end

function cli
    command "$python" -m a2b_lightvpr.cli $argv
end

function valid_json --argument-names path schema expected_checkpoint expected_manifest expected_selection
    test -f "$path"; or return 1
    command python -c 'import json,sys; x=json.load(open(sys.argv[1])); ok=x.get("schema")==sys.argv[2] and x.get("checkpoint_sha256")==sys.argv[3] and x.get("manifest_sha256")==sys.argv[4]; ok = ok and (not sys.argv[5] or x.get("selection_sha256")==sys.argv[5]); sys.exit(0 if ok else 1)' "$path" "$schema" "$expected_checkpoint" "$expected_manifest" "$expected_selection"
end

test -f "$checkpoint"; or die "missing checkpoint: $checkpoint"
test -x "$python"; or die "missing project interpreter: $python"
set checkpoint_sha (sha256sum "$checkpoint" | string split ' ' --fields 1)
test "$checkpoint_sha" = 36a6c803f5386a86042071469f215765576dfc8bd244941ed814a2901d0a9504; or die "unexpected MixVPR-512 CDND checkpoint"
set manifest_path (command python -c 'from a2b_lightvpr.config import load_config; print(load_config("configs/vpr_cdnd_salad_512.yaml").data.robotcar_manifest)')
test -f "$manifest_path"; or die "missing RobotCar manifest: $manifest_path"
set manifest_sha (sha256sum "$manifest_path" | string split ' ' --fields 1)
mkdir -p "$cache_dir"

echo "[1/5] focused MGTR tests"
command "$python" -m pytest -p no:cacheprovider tests/test_robotcar_temporal.py; or exit 1

echo "[2/5] RobotCar val selection"
if test "$force" = false; and valid_json "$selection" robotcar_temporal_selection_v1 "$checkpoint_sha" "$manifest_sha" ""
    echo "[skip] $selection"
else
    cli select-robotcar-temporal \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --cache "$cache_dir/val_top100.npz" \
        --output "$selection"; or exit 1
end
set selection_sha (sha256sum "$selection" | string split ' ' --fields 1)

echo "[3/5] frozen night evaluation and ablations"
if test "$force" = false; and valid_json "$night" robotcar_temporal_evaluation_v1 "$checkpoint_sha" "$manifest_sha" "$selection_sha"; and valid_json "$ablations" robotcar_temporal_ablations_v1 "$checkpoint_sha" "$manifest_sha" "$selection_sha"
    echo "[skip] night evaluation"
else
    cli evaluate-robotcar-temporal \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --selection "$selection" \
        --condition night \
        --cache "$cache_dir/night_top100.npz" \
        --output "$night" \
        --ablations-output "$ablations"; or exit 1
end

echo "[4/5] frozen RobotCar day control"
if test "$force" = false; and valid_json "$day" robotcar_temporal_evaluation_v1 "$checkpoint_sha" "$manifest_sha" "$selection_sha"
    echo "[skip] $day"
else
    cli evaluate-robotcar-temporal \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --selection "$selection" \
        --condition day \
        --cache "$cache_dir/day_top100.npz" \
        --output "$day"; or exit 1
end

echo "[5/5] system-level comparison"
cli summarize-robotcar-temporal \
    --night "$night" \
    --day "$day" \
    --crica runs/reference/cricavpr-dinov2-b-10752/final_suite/native/adapted/summary.json \
    --salad runs/reference/salad-dinov2-544/final_suite/native/adapted/summary.json \
    --boq runs/reference/boq-dinov2-12288/final_suite/native/adapted/summary.json \
    --reference-resources runs/reference/resource_benchmark.json \
    --output "$comparison"; or exit 1

echo "Done: $comparison"
