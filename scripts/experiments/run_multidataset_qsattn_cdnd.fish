#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
mkdir -p runs/vpr/cdnd_multigenerator_512
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set -g python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set config configs/experiments/data_sources/cdnd_multigenerator_512.yaml
set stage all
if test (count $argv) -gt 0
    set stage $argv[1]
end
contains -- $stage all finish; or exit 2
function cli
    $python_bin -m a2b_lightvpr.cli $argv
end
$python_bin scripts/data/prepare_multidataset_qsattn.py filter; or exit 1
set combined_root data/generated/multidataset_qsattn
mkdir -p "$combined_root"
if not test -e "$combined_root/robotcar"
    ln -s (realpath data/generated/robotcar_only_qsattn30k) "$combined_root/robotcar"; or exit 1
end
if test "$stage" = all
    for ds in svox msls
        set cfg "runs/vpr/cdnd_multigenerator_512/qsattn_$ds.yaml"
        # New translation always uses only the corresponding dataset's source rows.
        $python_bin -c 'import sys,yaml; from pathlib import Path; b=yaml.safe_load(Path("configs/experiments/data_sources/cdnd_multigenerator_512.yaml").read_text()); d=sys.argv[1]; b["data"]["generator_manifest"]=f"data/manifests_multidataset_qsattn/{d}_generator_train.csv"; b["data"]["dataset_weights"]={d:1}; b["data"]["vpr_source_manifest"]=f"data/manifests_multidataset_qsattn/{d}_vpr_train.csv"; b["generator"]["output_dir"]=f"runs/generator/qsattn_{d}_30000_seed0"; b["generator"]["resume"]=""; b["data"]["generated_root"]=f"data/generated/multidataset_qsattn/{d}"; Path(sys.argv[2]).write_text(yaml.safe_dump(b,sort_keys=False))' $ds $cfg; or exit 1
        set checkpoint "runs/generator/qsattn_"$ds"_30000_seed0/generator_final.pt"
        if test -f "$checkpoint"
            $python_bin -c 'import sys,torch; s=torch.load(sys.argv[1],map_location="cpu",weights_only=True); assert s["kind"]=="generator_inference" and s["objective"]=="qsattn" and s["step"]==30000' "$checkpoint"; or exit 1
            echo "[reuse] $checkpoint"
        else
            set resume_args
            set latest "runs/generator/qsattn_"$ds"_30000_seed0/checkpoints/latest.pt"
            if test -f "$latest"
                set resume_args --set "generator.resume=$latest"
            end
            cli train-generator --config "$cfg" $resume_args; or exit 1
        end
        if test -f "$combined_root/$ds/manifest.csv"
            echo "[reuse] $ds translated images; validated during merge"
        else
            cli translate-train --config "$cfg"; or exit 1
        end
    end
end
$python_bin scripts/data/prepare_multidataset_qsattn.py merge; or exit 1
cli cache-vpr-teacher --config "$config"; or exit 1
set resume_args
set latest runs/vpr/cdnd_multigenerator_512/seed0/checkpoints/latest.pt
if test -f "$latest"
    set resume_args --set "vpr.resume=$latest"
end
cli train-vpr --config "$config" $resume_args; or exit 1
cli evaluate-trained-suite --config "$config" --checkpoint runs/vpr/cdnd_multigenerator_512/seed0/model_best.pt \
    --datasets robotcar_project robotcar_day svox_night svox_day svox_sun tokyo_night tokyo_day msls_night msls_day_val; or exit 1
echo Complete
