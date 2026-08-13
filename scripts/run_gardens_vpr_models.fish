#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set python /home/admin123/miniconda3/envs/img2img-turbo/bin/python

set config configs/reference_models_extended.yaml
set mix_config configs/vpr_cdnd_salad_512.yaml
set mix_checkpoint runs/vpr/cdnd_512/salad_no_gate_seed0/model_best.pt
set gardens_root /home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/dataset/extracted_gardens_point
set results_root runs/external/gardens_point/models
set primary runs/external/gardens_point/evaluation.json
set summary runs/external/gardens_point/models_summary.json
set models \
    netvlad-vgg16-pitts4096 \
    cosplace-r50-512 \
    eigenplaces-r50-512 \
    eigenplaces-r50-2048 \
    mixvpr-r50-512 \
    mixvpr-r50-4096 \
    salad-dinov2-544 \
    cricavpr-dinov2-b-10752 \
    boq-dinov2-12288 \
    nocplace-r50-ikt-clean5k

function cli
    command "$python" -m a2b_lightvpr.cli $argv
end

test -x "$python"; or begin
    echo "ERROR: missing project interpreter: $python" >&2
    exit 1
end
test -d "$gardens_root"; or begin
    echo "ERROR: missing Gardens Point: $gardens_root" >&2
    exit 1
end
test -f "$mix_checkpoint"; or begin
    echo "ERROR: missing MixVPR-512 CDND checkpoint: $mix_checkpoint" >&2
    exit 1
end

echo "[1/5] Focused tests and reference-weight audit"
command "$python" -c 'import sys,torch,timm; assert sys.version_info[:2] == (3,10); assert torch.__version__.startswith("2.10."); print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, timm {timm.__version__}")'; or exit 1
command "$python" -m pytest -q -p no:cacheprovider tests/test_external_temporal.py tests/test_robotcar_temporal.py; or exit 1
cli audit-reference-models --config "$config" --output "$results_root/audit.json"; or exit 1

echo "[2/5] MixVPR-512 CDND main result and four-window sweep"
if not command "$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); r=x.get("runtime",{}); ok=x.get("temporal_method")=="mgtr_v1" and x.get("primary_window_size")==4 and set(x.get("window_sweep",{}))=={"3","4","5","6"} and r.get("python","").startswith("3.10.") and r.get("torch","").startswith("2.10."); sys.exit(0 if ok else 1)' "$primary" 2>/dev/null
    cli evaluate-gardens-temporal --config "$mix_config" --checkpoint "$mix_checkpoint" \
        --root "$gardens_root" --output-dir (dirname "$primary"); or exit 1
else
    echo "[skip] mixvpr-512-cdnd"
end

echo "[3/5] Gardens Point reference descriptors and temporal evaluation"
for model in $models
    set output_dir "$results_root/$model"
    set encoder adapted
    if test "$model" = nocplace-r50-ikt-clean5k
        set encoder original
    end
    if command "$python" -c 'import json,sys; p=sys.argv[1]; m=sys.argv[2]; e=sys.argv[3]; x=json.load(open(p)); r=x.get("runtime",{}); ok=x.get("schema")=="gardens_reference_evaluation_v1" and x.get("temporal_method")=="mgtr_v1" and x.get("model")==m and x.get("database_encoder")==e and x.get("primary_window_size")==4 and r.get("python","").startswith("3.10.") and r.get("torch","").startswith("2.10."); sys.exit(0 if ok else 1)' "$output_dir/evaluation.json" "$model" "$encoder" 2>/dev/null
        echo "[skip] $model"
        continue
    end
    cli evaluate-gardens-reference --config "$config" --registry "$config" \
        --model "$model" --root "$gardens_root" --output-dir "$output_dir" \
        --database-encoder "$encoder"; or exit 1
end

echo "[4/5] Single-frame/window=4/window-sweep summary"
cli summarize-gardens-models --results-root "$results_root" \
    --primary-result "$primary" --output "$summary"; or exit 1

echo "[5/5] Integrity"
command fish -n scripts/run_gardens_vpr_models.fish; or exit 1
echo "Done: $summary"
