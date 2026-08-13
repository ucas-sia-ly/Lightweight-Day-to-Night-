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

set base_config configs/vpr_cdnd_salad_4096.yaml
set route_config configs/vpr_cdnd_route_4096.yaml
set base_checkpoint runs/vpr/cdnd_4096/salad_no_gate_seed0/model_best.pt
set base_root runs/vpr/cdnd_4096/salad_no_gate_seed0/robotcar_ragtr
set route_root runs/vpr/cdnd_4096/route_contrastive_seed0
set final_root runs/vpr/cdnd_4096/ragtr_final
set gardens_root /home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/dataset/extracted_gardens_point
set salad runs/reference/salad-dinov2-544/final_suite/native/adapted/summary.json
set boq runs/reference/boq-dinov2-12288/final_suite/native/adapted/summary.json
set salad_threshold 0.9208472686733556

function die
    echo "ERROR: $argv" >&2
    exit 1
end

function cli
    command "$python" -m a2b_lightvpr.cli $argv
end

function valid_json --argument-names path schema checkpoint_sha manifest_sha selection_sha
    test -f "$path"; or return 1
    command "$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); schema=sys.argv[2]; expected={"schema":schema}; optional=(("checkpoint_sha256",sys.argv[3]),("manifest_sha256",sys.argv[4]),("selection_sha256",sys.argv[5])); required=("query_protocol_fingerprint",) if schema=="robotcar_ragtr_evaluation_v1" else (); ok=all(x.get(k)==v for k,v in expected.items()) and all(not v or x.get(k)==v for k,v in optional) and all(x.get(k) for k in required); sys.exit(0 if ok else 1)' "$path" "$schema" "$checkpoint_sha" "$manifest_sha" "$selection_sha"
end

function route_training_complete --argument-names latest model_best
    test -f "$latest"; and test -f "$model_best"; or return 1
    command "$python" -c 'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); ok=x.get("kind")=="vpr_training_state" and int(x.get("epoch",-1))==5 and int(x.get("samples_seen",-1))==300000; sys.exit(0 if ok else 1)' "$latest"
end

function prepare_val --argument-names config checkpoint output_root
    mkdir -p "$output_root/cache"
    set global "$output_root/cache/val_top100.npz"
    set geometry "$output_root/cache/val_geometry.npz"
    set selection "$output_root/val_selection.json"
    set checkpoint_sha (sha256sum "$checkpoint" | string split ' ' --fields 1)
    set manifest_path (command "$python" -c 'from a2b_lightvpr.config import load_config; import sys; print(load_config(sys.argv[1]).data.robotcar_manifest)' "$config")
    set manifest_sha (sha256sum "$manifest_path" | string split ' ' --fields 1)
    if test "$force" = false; and valid_json "$selection" robotcar_ragtr_selection_v1 "$checkpoint_sha" "$manifest_sha" ""
        echo "[skip] $selection"
    else
        cli select-robotcar-temporal --method ragtr \
            --config "$config" --checkpoint "$checkpoint" \
            --cache "$global" --geometry-cache "$geometry" \
            --output "$selection"; or exit 1
    end
end

test -x "$python"; or die "missing project interpreter: $python"
for path in "$base_checkpoint" "$salad" "$boq" "$gardens_root"
    test -e "$path"; or die "missing required input: $path"
end

echo "[1/8] RA-GTR tests and environment"
command "$python" -c 'import cv2,kornia; from kornia.feature import DISK,LightGlueMatcher; assert hasattr(cv2,"USAC_MAGSAC")'; or die "install with: pip install -e '.[ragtr,test]'"
command "$python" -m pytest -q -p no:cacheprovider; or exit 1

echo "[2/8] Base 4096 val selection"
prepare_val "$base_config" "$base_checkpoint" "$base_root"
set selected_config "$base_config"
set selected_checkpoint "$base_checkpoint"
set selected_root "$base_root"
set base_val (command "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_metrics"]["recalls"]["1"])' "$base_root/val_selection.json")

echo "[3/8] Val-only fallback decision"
if command "$python" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$base_val" "$salad_threshold"
    echo "Base RA-GTR val R@1=$base_val clears the pre-test threshold; skip specialized training."
else
    echo "Base RA-GTR val R@1=$base_val does not clear threshold; run route-only fallback."
    set mining "$route_root/route_mining.npz"
    if test "$force" = true; or not test -f "$mining"
        cli mine-robotcar-route --config "$route_config" \
            --checkpoint "$base_checkpoint" --output "$mining"; or exit 1
    end
    set latest "$route_root/checkpoints/latest.pt"
    set route_checkpoint "$route_root/model_best.pt"
    set resume_args
    if test -f "$latest"
        set resume_args --set "vpr.resume=$latest"
    end
    if test "$force" = true; or not route_training_complete "$latest" "$route_checkpoint"
        cli train-vpr --config "$route_config" $resume_args; or exit 1
    end
    prepare_val "$route_config" "$route_checkpoint" "$route_root/robotcar_ragtr"
    set route_val (command "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_metrics"]["recalls"]["1"])' "$route_root/robotcar_ragtr/val_selection.json")
    if command "$python" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$route_val" "$base_val"
        set selected_config "$route_config"
        set selected_checkpoint "$route_checkpoint"
        set selected_root "$route_root/robotcar_ragtr"
    end
end

mkdir -p "$final_root/cache"
cp "$selected_root/val_selection.json" "$final_root/val_selection.json"
set selection "$final_root/val_selection.json"
set selected_checkpoint_sha (sha256sum "$selected_checkpoint" | string split ' ' --fields 1)
set selected_manifest_path (command "$python" -c 'from a2b_lightvpr.config import load_config; import sys; print(load_config(sys.argv[1]).data.robotcar_manifest)' "$selected_config")
set selected_manifest_sha (sha256sum "$selected_manifest_path" | string split ' ' --fields 1)
set selection_sha (sha256sum "$selection" | string split ' ' --fields 1)
command "$python" -c 'import json,sys; json.dump({"config":sys.argv[1],"checkpoint":sys.argv[2],"checkpoint_sha256":sys.argv[3],"selection":sys.argv[4],"selection_sha256":sys.argv[5]},open(sys.argv[6],"w"),indent=2,sort_keys=True)' "$selected_config" "$selected_checkpoint" "$selected_checkpoint_sha" "$selection" "$selection_sha" "$final_root/frozen_model.json"; or exit 1

echo "[4/8] Frozen night test (exactly once per clean workflow)"
set night "$final_root/night_test.json"
if test "$force" = false; and valid_json "$night" robotcar_ragtr_evaluation_v1 "$selected_checkpoint_sha" "$selected_manifest_sha" "$selection_sha"
    echo "[skip] $night"
else
    cli evaluate-robotcar-temporal --method ragtr \
        --config "$selected_config" --checkpoint "$selected_checkpoint" \
        --selection "$selection" --condition night \
        --cache "$final_root/cache/night_top100.npz" \
        --geometry-cache "$final_root/cache/night_geometry.npz" \
        --output "$night"; or exit 1
end

echo "[5/8] Frozen day control"
set day "$final_root/day_test.json"
if test "$force" = false; and valid_json "$day" robotcar_ragtr_evaluation_v1 "$selected_checkpoint_sha" "$selected_manifest_sha" "$selection_sha"
    echo "[skip] $day"
else
    cli evaluate-robotcar-temporal --method ragtr \
        --config "$selected_config" --checkpoint "$selected_checkpoint" \
        --selection "$selection" --condition day \
        --cache "$final_root/cache/day_top100.npz" \
        --geometry-cache "$final_root/cache/day_geometry.npz" \
        --output "$day"; or exit 1
end

echo "[6/8] RobotCar comparison"
cli summarize-robotcar-temporal --method ragtr \
    --night "$night" --day "$day" --salad "$salad" --boq "$boq" \
    --crica runs/reference/cricavpr-dinov2-b-10752/final_suite/native/adapted/summary.json \
    --reference-resources runs/reference/resource_benchmark.json \
    --output "$final_root/comparison.json"; or exit 1

echo "[7/8] Gardens Point ordinal RA-GTR"
set gardens_out runs/external/gardens_point/models/mixvpr-4096-cdnd-ragtr
if test "$force" = false; and valid_json "$gardens_out/evaluation.json" gardens_ragtr_evaluation_v1 "$selected_checkpoint_sha" "" ""
    echo "[skip] $gardens_out/evaluation.json"
else
    cli evaluate-gardens-temporal --method ragtr \
        --config "$selected_config" --checkpoint "$selected_checkpoint" \
        --root "$gardens_root" --output-dir "$gardens_out"; or exit 1
end

echo "[8/8] Integrity"
command fish -n scripts/run_robotcar_ragtr_4096.fish; or exit 1
command "$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); a=x["acceptance"]; print(json.dumps(a,indent=2)); sys.exit(0 if a["adopt_as_robotcar_method"] else 2)' "$final_root/comparison.json"
set acceptance_status $status
if test $acceptance_status -eq 2
    echo "RA-GTR completed but did not meet the paper adoption threshold."
    exit 2
else if test $acceptance_status -ne 0
    exit $acceptance_status
end
echo "Done: $final_root/comparison.json"
