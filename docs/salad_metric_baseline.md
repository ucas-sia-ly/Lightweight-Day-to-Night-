# SALAD-S 地点度量训练

当前模型使用 **plain DINOv2-S/14 + 标准 SALAD 544D**，从完整 GSV 训练权重继续训练。维护 [Day-only](../configs/experiments/metric/salad_small_gsv_day_only_544.yaml) 和 [Day-aug](../configs/experiments/metric/salad_small_gsv_day_aug_544.yaml) 两份配对配置。

初始模型和继续训练结果按相同协议比较。Day-only 重训后的新权重需要重新评测，旧 suite 只作为历史证据。Day-aug 在 SVOX Night 的改善没有延伸到 Tokyo Night，不能认定增强方案全面胜出。实际结果、训练记录与 checkpoint 身份统一见 [实验结果](experiment_results.md)；本页说明可复现的训练设置。

## 模型与初始化

| 项目 | 设置 |
| --- | --- |
| Backbone | `dinov2_vits14`，384 通道；最后 2 个 block 与最终 norm 可训练 |
| 聚合 | Compact SALAD，16 clusters×32D＋32D CLS，共 544D；标准 OT 和标量 Dustbin |
| 输入 | 224×224 bicubic 直接 resize，ImageNet 归一化，无 padding |
| 输出 | `[B,544]`，L2 归一化 |
| 参数量 | 22,688,977 |
| 聚合器 dropout | 0.3，eval 时关闭 |
| 完整初始化 | `runs/reference/salad-dinov2-s-544-gsv/last.ckpt` |
| 初始化 SHA256 | `40ac2337624ab029026291f6d4d8a89a20001bb45627aa77ea61789f6af36d4c` |

完整初始化加载 backbone 和聚合器，基础 DINOv2 checkpoint 不能代替已训练的 SALAD head。键、通道与形状严格检查；导出初始推理模型不做训练更新。

推理 checkpoint 记录输入尺寸、变换、family 和 backbone 身份，评测时检查与配置一致。改变预处理、模型或输入尺寸应创建独立实验与 checkpoint，不能覆盖原配置后复用旧分数。这里是 SALAD-S 的 544D 紧凑变体，不是官方大维度 DINOv2-B 论文结果的完整复现。

## 地点标签与采样

训练 manifest 为 `data/manifests_gsv_metric/vpr_train.csv`：23 个城市、62,514 个地点、524,701 张不同源图，每地点至少 4 张图。标签来自 GSV-Cities 原始 `city + place_id`，SHA256 为 `32ece2b07ca58e827bddf7460f6ce423dcd145e83edd2d6e3f1515ac205a8ed0`。原元数据、源身份与文件存在性检查见 [audit.json](../data/manifests_gsv_metric/audit.json)。

每个物理 batch 来自同城 32 个地点，每地点抽 4 张不同源图，合计 128 张。每源图只产生一个 Day 或 Aug view，地点正样本由同地点的其他源图提供；不是依靠同源图两份外观完成身份匹配。miner 的候选池为这个物理 batch，梯度累积不会扩大它。

每 epoch 抽样 251,392 张、1,964 batches，覆盖全部合格地点。城市尾部补齐 batch 会少量重复地点；不是每轮读取全部 524,701 张图。两轮预算为 502,784 个源图抽样，实际更新数和是否提前停止以日志为准。训练不使用验证或测试 Query。

## 损失与增强

直接使用 PyTorch Metric Learning：`MultiSimilarityMiner(epsilon=0.1, CosineSimilarity)` 与 `MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, DotProductSimilarity)`，mining 与 loss 使用 FP32。参数沿用 SALAD recipe，避免混用库默认值。

Day-only 使用原始白天图；Day-aug 以 0.5 概率替换为非生成式外观增强：曝光 −1.5～+0.3 stops，亮度 0.8～1.15、对比度 0.8～1.2，红／蓝色温增益 1±w（w∈[−0.12,0.12]），噪声标准差 0～0.02；以 0.25 概率加入轻度 3×3 高斯模糊。增强不做 crop、warp 或强遮挡，按 seed、epoch 与源身份确定。

主线只有地点度量损失，没有可靠性预测、动态 Dustbin、局部自监督或复杂蒸馏。未来增加真实夜间 train 数据或生成图，应作为独立消融，并保持可信的不同源图地点监督。

## 优化与模型选择

训练 2 epochs，backbone LR 1e-5、aggregator LR 3e-5，weight decay 9.5e-9，cosine 最低倍率 0.2，CUDA AMP 开启。两组使用相同初始化、seed 0、物理 batch 与预算。

训练前评估 epoch zero，每轮只按公开 `svox_day_val` 选择 `model_best.pt`，初始模型也进入候选。首轮 Recall@1 比初始下降超过 0.03 时回归保护停止；夜间 test 不参与选择。白天验证的改善不能证明夜间效果。

## 运行

从项目根目录激活环境，先审计资源；只有需要重新生成训练 manifest 时才准备数据：

```bash
a2b-lightvpr audit-resources --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml
a2b-lightvpr prepare-gsv-metric --data-root /home/admin123/github/WORKSPACE/dataset/gsv-cities
```

已有初始推理模型为 `runs/vpr/salad_small_gsv_metric/pretrained_224/model_initial.pt`。重新导出时使用新的文件路径：

```bash
a2b-lightvpr export-vpr-initial \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --output runs/vpr/salad_small_gsv_metric/initial_new.pt
```

训练配置的默认目录保存已有实验。重新训练必须覆盖为新的输出目录，两个任务按顺序运行：

```bash
a2b-lightvpr train-vpr \
  --config configs/experiments/metric/salad_small_gsv_day_only_544.yaml \
  --set vpr.output_dir=runs/vpr/salad_small_gsv_metric/day_only_new

a2b-lightvpr train-vpr \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --set vpr.output_dir=runs/vpr/salad_small_gsv_metric/day_aug_new
```

同一次实验恢复使用 `--set vpr.resume=该目录/checkpoints/latest.pt`。不同 seed 或配置使用不同目录；`--smoke-epochs 1` 只限制 epoch 数，不会自动缩小每轮样本预算。

```bash
a2b-lightvpr evaluate-trained-suite \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --checkpoint runs/vpr/salad_small_gsv_metric/day_aug_seed0_lowmem/model_recovered_epoch2.pt \
  --datasets msls_night svox_night tokyo247_night tokyo247_sunset robotcar_seasons_night \
  --set vpr.output_dir=runs/vpr/salad_small_gsv_metric/day_aug_reverified_20261014

a2b-lightvpr benchmark \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --checkpoint runs/vpr/salad_small_gsv_metric/day_aug_seed0_lowmem/model_recovered_epoch2.pt \
  --warmup 20 --iterations 100 \
  --set vpr.output_dir=runs/benchmarks/salad_metric_recovered
```

本次 Day-aug 的 `model_recovered_epoch2.pt` 从完整 epoch 2 checkpoint 恢复；参数身份核对与原 suite 的对应关系见 [实验结果](experiment_results.md)。序列化文件 hash 的变化不表示参数改变。

TensorBoard 标签与恢复见 [训练监控](training_monitoring.md)。完整训练、suite 和测试必须串行，`workers: 0`，评测 batch 64，32 GB 内存约束见 [资源说明](resource_migration.md#32-gb-内存限制)。

## 协议边界

标准 Tokyo24/7 显式选 `tokyo247`（315 Query）或 `tokyo247_night/day/sunset`（各 105），75,984 个 DB crop 按 6,332 个 panorama 做 NMS。固定名称、坐标、DB 文件清单和正集已核对，来源在 [协议 provenance](../src/a2b_lightvpr/protocols/tokyo247_provenance.json)；这不证明本地图像字节与官方副本完全相同。

MSLS Night 为 55 Query 夜间 val 子集，不能写成标准全部 740 Query 的分数。SVOX Night 为 823 Query／17,166 DB。公共 [Seasons](robotcar_seasons.md)固定 v2 test／rear：429 夜间 Query、6,954 DB，真值隐藏，仅输出排名。`all`／`all-day`／`all-conditions` 已使用标准 Tokyo 条件，旧 raw key 退出活动 suite。

Real Night Query → Fake Night Database 需另外准备完整测试 Database 的生成图和显式映射，格式见 [Seasons 文档](robotcar_seasons.md#fake-night-database)。真实 Query 和源 Database 身份保持原协议；现有训练生成图不能替代测试输入。

## 上游来源

- [SALAD 官方实现](https://github.com/serizba/salad)：OT 聚合和训练 recipe；代码 GPL-3.0。
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)：miner、loss、distance 与 reducer；代码 MIT。
- [DINOv2](https://github.com/facebookresearch/dinov2)：S/14 backbone；代码 Apache-2.0。
- [GSV-Cities](https://github.com/amaralibey/gsv-cities)：原始地点标签与数据；数据使用条件和代码许可分别遵循上游说明。

本项目当前设置、输入维度与继续训练预算需独立报告，不能套用上游论文成绩。
