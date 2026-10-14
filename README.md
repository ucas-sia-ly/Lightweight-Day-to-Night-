# 昼夜视觉地点识别

当前主线是 **DINOv2-S/14 + SALAD 544D**：从完整 SALAD-S 权重开始，使用 GSV-Cities 的真实地点标签和 MultiSimilarity 度量训练，比较白天训练与非生成式光照增强。部署时输入一张图像，输出 L2 归一化的 544D 地点描述子。

已有夜间检索比较显示，继续训练在部分测试集改善，但 Tokyo 夜间结果下降，尚未得到全面优于初始模型的方案。两组当前权重已在独立目录重新完成评测，旧 suite 只作为历史证据。实际分数、checkpoint 身份及训练证据统一见 [实验结果](docs/experiment_results.md)。旧生成器、CDND、MixVPR 论文工作流和重复产物已退出推荐入口；历史指标压缩保存在 `reports/retired_experiments_20261014.json.gz`。

## 安装

需要 Python 3.10+。本机训练环境为 Python 3.12、PyTorch 2.5.1、CUDA 12.4；完整训练需要可访问的 NVIDIA GPU。

```bash
conda env create -f environment.yml
conda activate a2b-lightvpr
python -m pip install -e '.[test,eval]'
python -m pip check
python -m a2b_lightvpr.cli --help
```

所有命令从项目根目录执行。数据位置和已有权重见 [资源说明](docs/resource_migration.md)。

## 训练

当前维护两个配对配置，共享初始化、地点标签、物理 batch、训练预算和验证集。

| 配置 | 输入 |
| --- | --- |
| [Day-only](configs/experiments/metric/salad_small_gsv_day_only_544.yaml) | 每张源图一个白天 view |
| [Day-aug](configs/experiments/metric/salad_small_gsv_day_aug_544.yaml) | 每张源图以 0.5 概率替换为光照增强 view |

```bash
a2b-lightvpr audit-resources --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml
a2b-lightvpr train-vpr --config configs/experiments/metric/salad_small_gsv_day_only_544.yaml --set vpr.output_dir=runs/vpr/salad_small_gsv_metric/day_only_new
a2b-lightvpr train-vpr --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml --set vpr.output_dir=runs/vpr/salad_small_gsv_metric/day_aug_new
```

机器内存为 32 GB，训练与评测必须串行。两个配置均使用 `workers: 0`，训练物理 batch 128、评测 batch 64。重新训练必须使用新的 `vpr.output_dir`；已有目录的 `checkpoints/latest.pt` 只用于恢复同一次实验。采样、损失、增强和模型选择见 [训练基线](docs/salad_metric_baseline.md)。

TensorBoard 使用 `tensorboard --logdir runs/vpr/salad_small_gsv_metric --port 6006`，浏览器打开 <http://localhost:6006>。标签与恢复规则见 [训练监控](docs/training_monitoring.md)。

## 评测

```bash
a2b-lightvpr evaluate-trained-suite \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --checkpoint runs/vpr/salad_small_gsv_metric/day_aug_seed0_lowmem/model_recovered_epoch2.pt \
  --datasets msls_night svox_night tokyo247_night tokyo247_sunset robotcar_seasons_night \
  --set vpr.output_dir=runs/vpr/salad_small_gsv_metric/day_aug_reverified_20261014
```

新训练按公开 SVOX Day Validation 选择推理权重，夜间测试不参与选择。本次 Day-aug 从完整 epoch 2 训练 checkpoint 恢复为 `model_recovered_epoch2.pt`，参数身份与恢复证据见实验结果。标准 Tokyo24/7 使用 `tokyo247` 或 `tokyo247_night/day/sunset`，三个条件各 105 Query，排名按 panorama 去重。MSLS Night 是本项目的 55 Query 夜间 val 子集，不能当作标准全部 740 Query 的分数。评测命令应显式列出协议，避免混用历史 Tokyo raw 协议。

公共 [RobotCar Seasons](docs/robotcar_seasons.md)与其他数据集使用相同 suite 入口，固定 v2 test／rear 相机。夜间 429 Query、白天 Database 6,954 张；官方 test 真值隐藏，输出 Top-K，Recall/MRR 为 `null`。

Real Night Query → Fake Night Database 的评测接口可用，但仍需测试 Database 的生成图与 `source_path,generated_path` 映射，详见 [Seasons 文档](docs/robotcar_seasons.md#fake-night-database)。已有训练生成图不属于测试 Database。

## 文档与目录

- [实验结果](docs/experiment_results.md)：统一成绩、证据及当前结论。
- [训练基线](docs/salad_metric_baseline.md)：模型、训练 recipe 与运行命令。
- [资源说明](docs/resource_migration.md)：数据、权重、内存限制与当前缺口。
- [训练监控](docs/training_monitoring.md)：TensorBoard 与断点恢复。
- [RobotCar Seasons](docs/robotcar_seasons.md)：固定公共测试协议和隐藏真值边界。

`src/a2b_lightvpr/` 包含模型、训练、检索、参考模型和资源审计；`configs/experiments/metric/` 保存当前训练配置；`configs/references/salad.yaml` 保存 SALAD-S 参考模型定义；`tests/` 保存回归检查。`data/`、`runs/` 为本地数据与实验产物，`reports/` 保存清理记录和压缩历史指标。
