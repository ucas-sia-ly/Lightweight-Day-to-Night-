# A2B-LightVPR

Clean research implementation for training-only day-to-night translation and a
single lightweight VPR model at deployment. The old CycleGAN-Turbo project is
used only as a source of datasets and frozen benchmark manifests.

## Design

- One shared ResNet-9 generator and PatchGAN discriminator.
- Exactly one content objective per generator run: `negcut` or `qsattn`.
  Target identity NCE is intentionally absent.
- Datasets are sampled uniformly by default; large datasets cannot dominate by row count.
- Aligned DarkDriving pairs use shared geometry and a low-weight structure loss.
- Generated images are offline VPR training views. The generator and all
  teachers are excluded from deployment.
- Lightweight VPR candidates share a 512-dimensional descriptor contract.
- RobotCar test data cannot be opened by training commands.

## Install

Install a CUDA-compatible PyTorch build first, then:

```bash
pip install -e '.[eval,test]'
```

The provided machine can use its existing environment:

```bash
conda run -n img2img-turbo pip install -e '.[test]'
```

Install the tracking extra and authenticate once before online W&B runs:

```bash
pip install -e '.[eval,test,tracking]'
wandb login
```

## Workflow

```bash
a2b-lightvpr prepare-data \
  --data-root /home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/dataset \
  --output-dir data/manifests
a2b-lightvpr audit-data --config configs/robotcar.yaml
a2b-lightvpr train-generator --config configs/generator_qsattn_v2.yaml \
  --set tracking.enabled=true
a2b-lightvpr translate-train --config configs/generator_qsattn_v2.yaml
a2b-lightvpr train-vpr --config configs/vpr_screen_efficientnet_b0_safe_60k.yaml
a2b-lightvpr evaluate-trained-suite \
  --config configs/vpr_screen_efficientnet_b0_safe_60k.yaml \
  --checkpoint runs/vpr/three_safe/screen_efficientnet_b0_320x480_seed0/model_best.pt \
  --datasets all
```

Training tracking is opt-in. Enable it with `--set tracking.enabled=true`;
generator runs upload losses and generated examples, while VPR runs upload
loss components and RobotCar validation recall after every epoch. See
[`docs/WANDB.md`](docs/WANDB.md) for the complete commands.

当前数据、生成器、VPR、loss、评测公式和下一阶段路线的中文总览见
[`docs/完整技术路线.md`](docs/完整技术路线.md)。当前完成项、checkpoint 和下一轮
单变量实验表见 [`docs/当前实验状态与实验矩阵.md`](docs/当前实验状态与实验矩阵.md)。

Long pilot and final runs are intentionally separate. Checkpoints contain the
resolved config, manifest hashes, optimizer state, and RNG state and can be
resumed without a wrapper script.

For a deliberately small translated smoke subset, lower the place filter only
for that run:

```bash
a2b-lightvpr train-vpr --config configs/vpr_screen_efficientnet_b0_safe_60k.yaml --smoke-epochs 1 \
  --set data.generated_root=data/generated/smoke_qsattn \
  --set vpr.min_class_images=1
```

Formal runs use the dataset-specific minimum view counts in their YAML files.
The generator has no VPR teacher or descriptor-relation loss.

Use `evaluate --baseline-json` with the frozen NocPlace per-query metrics. The
result records a passed acceptance only when R@1 exceeds `81.605%` and the
paired-bootstrap 95% CI lower bound is above zero. A smoke checkpoint or a
single seed is never an acceptance result.

## Reference material

The implementations are independent, with the following repositories used to
verify equations and expected behavior:

- <https://github.com/WeilunWang/NEGCUT>
- <https://github.com/sapphire497/query-selected-attention>
- <https://github.com/gmberton/EigenPlaces>
# A2B-LightVPR
