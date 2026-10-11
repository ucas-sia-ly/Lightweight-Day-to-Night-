#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set root (realpath "$script_dir/../..")
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
set expected_samples 1198720
set generator_config configs/generator/qsattn_official_objective_150k.yaml
set generator_dir runs/generator/qsattn_official_objective_150k_seed0
set generator_final "$generator_dir/generator_final.pt"

function die
    echo "ERROR: $argv" >&2
    exit 1
end

function cli
    "$python_bin" -m a2b_lightvpr.cli $argv
end

function verify_sha256 --argument-names path expected
    test -f "$path"; or die "missing required checkpoint: $path"
    set actual (sha256sum "$path" | string split ' ' --fields 1)
    test "$actual" = "$expected"; or die "SHA256 mismatch for $path: $actual"
end

function latest_checkpoint --argument-names directory
    if test -d "$directory/checkpoints"
        find "$directory/checkpoints" -maxdepth 1 -type f -name '*.pt' | sort -V | tail -n 1
    end
end

function vpr_complete --argument-names directory
    set latest "$directory/checkpoints/latest.pt"
    test -f "$directory/model_best.pt"; and test -f "$latest"; or return 1
    "$python_bin" -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=False); sys.exit(0 if int(p.get("samples_seen", -1)) == int(sys.argv[2]) else 1)' "$latest" "$expected_samples"
end

function train_vpr_once --argument-names config_path output_dir
    if vpr_complete "$output_dir"
        echo "[skip] completed VPR run: $output_dir"
        return
    end
    set latest (latest_checkpoint "$output_dir")
    if test -n "$latest"
        echo "[resume] $output_dir from $latest"
        cli train-vpr --config "$config_path" --set "vpr.resume=$latest"; or exit 1
    else
        echo "[run] train VPR: $output_dir"
        cli train-vpr --config "$config_path"; or exit 1
    end
    vpr_complete "$output_dir"; or die "$output_dir did not reach samples_seen=$expected_samples"
end

function evaluate_vpr_once --argument-names config_path output_dir
    set checkpoint "$output_dir/model_best.pt"
    set suite "$output_dir/final_suite/summary.json"
    test -f "$checkpoint"; or die "missing trained checkpoint: $checkpoint"
    if test -f "$suite"
        echo "[skip] development suite: $suite"
    else
        cli evaluate-trained-suite --config "$config_path" --checkpoint "$checkpoint" --datasets all; or exit 1
    end
    if test -f "$output_dir/benchmark.json"
        echo "[skip] resource benchmark: $output_dir/benchmark.json"
    else
        cli benchmark --config "$config_path" --checkpoint "$checkpoint" --warmup 20 --iterations 100; or exit 1
    end
end

echo "[1/10] preflight"
"$python_bin" -c 'import shutil, sys, torch, torchvision, yaml, PIL, numpy; free=shutil.disk_usage(".").free/2**30; print(f"Python OK; torch={torch.__version__}; cuda={torch.cuda.is_available()}; free_disk={free:.1f} GiB"); sys.exit(0 if torch.cuda.is_available() and free >= 20 else 1)'; or die "CUDA and at least 20 GiB free disk are required"
"$python_bin" -c 'from a2b_lightvpr.config import load_config; import sys; [load_config(p) for p in sys.argv[1:]]' \
    "$generator_config" \
    configs/vpr/day_fake_128.yaml configs/vpr/cdnd_salad_128.yaml \
    configs/vpr/day_fake_512.yaml configs/vpr/cdnd_salad_512.yaml \
    configs/vpr/day_fake_4096.yaml configs/vpr/cdnd_salad_4096.yaml; or exit 1
verify_sha256 '/home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/checkpoints/resnet50_MixVPR_128_channels(64)_rows(2).ckpt' d9dc62449ebfbfca0ac138e3062469458f957c5dd8d61157fe07007d1da54b6f
verify_sha256 '/home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/checkpoints/resnet50_MixVPR_512_channels(256)_rows(2).ckpt' 7e1180f77d04ece604f2d12f8b6b9b323a99d6e43863c879028bfa17b34807cf
verify_sha256 '/home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/checkpoints/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt' 97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24
verify_sha256 '/home/admin123/diffusion_vpr_test/reference_code/weights/dino_salad_512_32.ckpt' d25c3434032a3ce5680ffcf51e05de0a7b9962772bd9112d88ca074689f18dcc
cli audit-data --config "$generator_config"; or exit 1

echo "[2/10] tests"
command pytest; or exit 1

echo "[3/10] QS-Attn generator"
if test -f "$generator_final"
    echo "[skip] generator: $generator_final"
else
    set latest (latest_checkpoint "$generator_dir")
    if test -n "$latest"
        cli train-generator --config "$generator_config" --set "generator.resume=$latest"; or exit 1
    else
        cli train-generator --config "$generator_config"; or exit 1
    end
end

echo "[4/10] fake-night training set"
set fake_manifest data/generated/multidataset_qsattn_official_150k/manifest.csv
if test -f "$fake_manifest"
    echo "[skip] translated training set: $fake_manifest"
else
    cli translate-train --config "$generator_config" --checkpoint "$generator_final"; or exit 1
end
"$python_bin" -c 'import csv, sys; n=sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8"))); print(f"fake manifest rows={n}"); sys.exit(0 if n == 118313 else 1)' "$fake_manifest"; or die "fake manifest must contain 118,313 rows"

echo "[5/10] SALAD teacher cache"
set teacher_cache data/teacher_cache/salad_qsattn_official_150k_fp16.npy
set teacher_meta data/teacher_cache/salad_qsattn_official_150k_fp16.meta.json
if test -f "$teacher_cache"; and test -f "$teacher_meta"
    echo "[skip] teacher cache: $teacher_cache"
else
    cli cache-vpr-teacher --config configs/vpr/cdnd_salad_512.yaml; or exit 1
end
"$python_bin" -c 'from pathlib import Path; from a2b_lightvpr.config import load_config; from a2b_lightvpr.vpr.teacher_cache import TeacherDescriptorCache; c=load_config("configs/vpr/cdnd_salad_512.yaml"); x=TeacherDescriptorCache(c, Path(c.data.generated_root)/"manifest.csv"); print(f"teacher cache OK: {x.shape[0]} descriptors")'; or exit 1

echo "[6/10] baseline training"
for dimension in 128 512 4096
    train_vpr_once "configs/vpr/day_fake_$dimension.yaml" "runs/vpr/cdnd_$dimension/day_fake_pk_seed0"
end

echo "[7/10] CDND training"
for dimension in 128 512 4096
    train_vpr_once "configs/vpr/cdnd_salad_$dimension.yaml" "runs/vpr/cdnd_$dimension/salad_no_gate_seed0"
end


echo "[8/10] development evaluation and resources"
for dimension in 128 512 4096
    evaluate_vpr_once "configs/vpr/day_fake_$dimension.yaml" "runs/vpr/cdnd_$dimension/day_fake_pk_seed0"
    evaluate_vpr_once "configs/vpr/cdnd_salad_$dimension.yaml" "runs/vpr/cdnd_$dimension/salad_no_gate_seed0"
end

echo "[9/10] sample-budget verification"
for dimension in 128 512 4096
    for method in day_fake_pk salad_no_gate
        vpr_complete "runs/vpr/cdnd_$dimension/{$method}_seed0"; or die "invalid sample budget for $dimension/$method"
    end
end

echo "[10/10] single-run summary"
mkdir -p runs/comparisons
cli summarize-cdnd-seed0 \
    --baseline 128=runs/vpr/cdnd_128/day_fake_pk_seed0/final_suite/summary.json \
    --candidate 128=runs/vpr/cdnd_128/salad_no_gate_seed0/final_suite/summary.json \
    --baseline 512=runs/vpr/cdnd_512/day_fake_pk_seed0/final_suite/summary.json \
    --candidate 512=runs/vpr/cdnd_512/salad_no_gate_seed0/final_suite/summary.json \
    --baseline 4096=runs/vpr/cdnd_4096/day_fake_pk_seed0/final_suite/summary.json \
    --candidate 4096=runs/vpr/cdnd_4096/salad_no_gate_seed0/final_suite/summary.json \
    --output runs/comparisons/cdnd_seed0_summary.json; or exit 1

echo "Done: runs/comparisons/cdnd_seed0_summary.json"
