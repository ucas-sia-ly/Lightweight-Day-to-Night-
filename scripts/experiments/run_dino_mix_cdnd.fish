#!/usr/bin/env fish
set root (realpath (dirname (status --current-filename))/../..)
cd "$root"; or exit 1
set -gx PYTHONPATH "$root/src"
set -gx OMP_NUM_THREADS 4
set -q CUDA_VISIBLE_DEVICES; or set -gx CUDA_VISIBLE_DEVICES 0
set python_bin python
set -q PYTHON_BIN; and set python_bin "$PYTHON_BIN"
mkdir -p runs/vpr/dino_mix_small_512
$python_bin -m a2b_lightvpr.cli cache-vpr-teacher --config configs/experiments/students/dino_mix_small_cdnd.yaml; or exit 1
for group in day_fake cdnd
    set cfg "configs/experiments/students/dino_mix_small_$group.yaml"
    set out "runs/vpr/dino_mix_small_512/"$group"_seed0"
    set resume_args
    if test -f "$out/checkpoints/latest.pt"
        set resume_args --set "vpr.resume=$out/checkpoints/latest.pt"
    end
    $python_bin -m a2b_lightvpr.cli train-vpr --config "$cfg" $resume_args 2>&1 | tee "runs/vpr/dino_mix_small_512/$group.log"
    set codes $pipestatus
    if test $codes[1] -ne 0
        exit $codes[1]
    end
    $python_bin -m a2b_lightvpr.cli evaluate-trained-suite --config "$cfg" --checkpoint "$out/model_best.pt" \
        --datasets robotcar_seasons_night svox_night tokyo_night msls_night robotcar_seasons_day svox_day svox_sun tokyo_day msls_day_val; or exit 1
    $python_bin -m a2b_lightvpr.cli benchmark --config "$cfg" --checkpoint "$out/model_best.pt" --warmup 20 --iterations 100; or exit 1
end
