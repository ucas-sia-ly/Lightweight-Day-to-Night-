# A2B-LightVPR

Research implementation for training-only day-to-night translation and a
single VPR descriptor model at inference. The old CycleGAN-Turbo project is
used only as a source of datasets and frozen benchmark manifests.

## Design

- The formal generator is ResNet-9 + PatchGAN + the official-objective QS-Attn
  structure, including target identity NCE.
- Generated images are offline VPR training views. The generator and all
  training teachers are excluded from query inference.
- Formal VPR training uses MSLS, SVOX and RobotCar with a balanced rolling sampler.
- The paper's primary candidate is MixVPR-R50-512 with a pre-registered seed 0.
- Day/fake loss selection uses RobotCar validation and MSLS night validation;
  frozen test suites do not tune the loss.
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

ONNX export additionally requires the dedicated export dependencies:

```bash
pip install -e '.[export]'
```

## Workflow

```bash
fish scripts/run_remaining_training.fish --check
fish scripts/run_remaining_training.fish
```

Training tracking is opt-in. Enable it with `--set tracking.enabled=true`;
generator runs upload losses and generated examples, while VPR runs upload
loss components and RobotCar validation recall after every epoch. See
[`docs/WANDB.md`](docs/WANDB.md) for the complete commands.

当前数据、生成器、VPR、loss、评测公式和待运行项的中文总览见
[`docs/完整技术路线.md`](docs/完整技术路线.md)。

Long pilot and final runs are intentionally separate. Checkpoints contain the
resolved config, manifest hashes, optimizer state, and RNG state and can be
resumed without a wrapper script.

For a deliberately small translated smoke subset, lower the place filter only
for that run:

```bash
a2b-lightvpr train-vpr --config configs/vpr_mixvpr_r50_512_safe_60k.yaml --smoke-epochs 1 \
  --set data.generated_root=data/generated/smoke_qsattn \
  --set vpr.min_class_images=1
```

Formal runs use the dataset-specific minimum view counts in their YAML files.
The generator has no VPR teacher or descriptor-relation loss.

The paper reports the pre-registered seed 0. Preserve is retained only if its
R@1 does not decrease on either RobotCar validation or MSLS night validation,
and the stratified macro bootstrap 95% CI lower bound is above zero.

Detailed fish commands, skip conditions and result paths are documented in
[`docs/NetVLAD与DINOv2对照实验.md`](docs/NetVLAD与DINOv2对照实验.md).

## Reference material

The implementations are independent, with the following repositories used to
verify equations and expected behavior:

- <https://github.com/WeilunWang/NEGCUT>
- <https://github.com/sapphire497/query-selected-attention>
- <https://github.com/gmberton/EigenPlaces>

NetVLAD、SALAD、CricaVPR、BoQ、资源 benchmark 和官方目标结构 QS-Attn 的
fish 命令见 [`docs/NetVLAD与DINOv2对照实验.md`](docs/NetVLAD与DINOv2对照实验.md)。
