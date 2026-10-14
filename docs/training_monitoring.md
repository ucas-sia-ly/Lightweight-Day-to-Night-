# 使用 TensorBoard 观察训练

项目的 `train-vpr` 和 `train-generator` 默认记录 TensorBoard 日志。覆盖 MixVPR、SALAD-S、CDND、StableNight-SALAD 的项目训练入口，以及 QS-Attn 生成器训练。checkpoint 与 JSON/JSONL 日志继续保留；当前 VPR 模型选择采用公开白天验证 split，测试评估单独运行。

## 启动与查看

在训练所使用的环境中更新安装，TensorBoard 随项目依赖安装：

```bash
python -m pip install -e '.[test,eval]'
```

在项目根目录打开另一个终端，启动查看服务：

```bash
tensorboard --logdir runs --port 6006
```

浏览器打开 <http://localhost:6006>。`runs` 下不同实验目录会分别显示，可以同时比较 loss、学习率和验证曲线。模型训练与 TensorBoard 服务是两个独立进程，训练不自动启动浏览器或服务。

当前机器可用的 MSLS/SVOX 训练入口为：

```bash
python -m a2b_lightvpr.cli train-vpr \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml
```

默认事件文件写入该实验的 `vpr.output_dir/tensorboard/`。生成器使用 `generator.output_dir/tensorboard/`。已有脚本无需额外传入 TensorBoard 开关；完整训练的资源条件见 [资源迁移说明](resource_migration.md)。

如果训练运行在远程机器，可以在本地建立端口转发后打开同一浏览器地址：

```bash
ssh -L 6006:localhost:6006 用户名@训练机器
```

## 配置

`vpr` 与 `generator` 都支持以下字段：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `tensorboard_enabled` | `true` | 是否写入事件文件 |
| `tensorboard_log_interval` | `10` | 每隔多少个训练 step 记录一次 step 指标 |
| `tensorboard_image_interval` | `100` | 图像与分布诊断的记录间隔；`0` 关闭这些诊断 |
| `tensorboard_flush_secs` | `30` | writer 刷新事件文件的时间间隔，单位秒 |
| `tensorboard_log_dir` | `""` | 空值使用相应 `output_dir/tensorboard`，非空值指定日志目录 |

scalar 在首个有效更新和 epoch 结束时也会记录；生成器则在首个 iteration 和本次训练末尾额外记录。图像诊断启用时也保留首／末记录，方便查看短训练。`run/config` 文本记录本次训练配置。

例如，更密集观察 VPR 的短训练：

```bash
python -m a2b_lightvpr.cli train-vpr \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --set vpr.tensorboard_log_interval=1 \
  --set vpr.output_dir=runs/vpr/stable_night_salad_544/msls_svox_monitor_seed0
```

关闭日志使用 `--set vpr.tensorboard_enabled=false`，生成器对应 `--set generator.tensorboard_enabled=false`。修改这些日志字段不改变实验指纹或 checkpoint 的配置匹配，不会单独阻止恢复训练。

为不同实验、seed 或消融选择不同 `output_dir`；显式自定义 `tensorboard_log_dir` 时，也应分别指定目录。只启动 TensorBoard 不会为历史训练自动补出事件文件。

## 重点观察内容

VPR 的全部曲线使用实际成功的 optimizer 更新次数作为横轴，包括 epoch 平均值与验证指标。梯度累积的多个 batch 共用一次更新，AMP 跳过更新时不把它计作新的 optimizer step。`epoch_progress/epoch` 记录当前完成的 epoch，`epoch_loss/` 记录完整 epoch 的平均值。生成器日志使用绝对训练 iteration 作为横轴。

| 训练路线 | 重点指标 |
| --- | --- |
| 所有项目 VPR 路线 | 总 loss、各已计算损失分量、backbone／aggregator／classifier 学习率、epoch 平均值、训练预算、数据等待时间和吞吐量 |
| 分类与 CDND 基线 | 分类损失；CDND 启用时记录实际计算的蒸馏分量 |
| StableNight-SALAD | 地点、局部稳定性、可靠性损失；有效 anchor／patch 统计；各 view 的 reliability、Dustbin assignment、cluster mass 偏差及 gamma |
| 实际运行验证的 VPR 配置 | 验证集测得的 recall 和查询统计 |
| QS-Attn | Generator／Discriminator loss、GAN 与 content 分量、学习率、AMP 状态，以及 Day／Fake Night／Real Night 图像预览 |

VPR 的 `train/loss` 和 `train/{分量名}` 是当前梯度累积组内各 microbatch 的平均值；完整 epoch 平均值位于 `epoch_loss/`。学习率为 `lr/backbone`、`lr/aggregator` 和 `lr/classifier`。`progress/samples_seen` 是累计源图数量，`amp/scale` 是 AMP scale。在 `epoch_progress/` 下可以观察 `samples`、`student_images`、`reference_images`、`optimizer_steps`、`data_wait_seconds`、`batch_seconds`、`samples_per_second` 和 `amp_skipped_updates`。实际运行验证时记录 `validation/recall_at_K`、`validation/mrr`、有效／总 Query 数及 `validation/best_recall_at_1`。验证配置只允许 `msls_day_val`、`svox_day_val`；配置多个 split 时 Recall/MRR 按协议等权平均，Query 数求和。这些曲线不代表夜间 test 的结果。

StableNight 的 Images 面板提供 `images/day`、`images/aug` 或 `images/fake`，以及 `images/{view}_reliability` 和 `images/{view}_dustbin_assignment` 的原生 patch 网格图。Histograms 面板的 `patches/{view}/reliability` 与 `patches/{view}/dustbin_assignment` 统计最多三张样例中的有效 patch；这是样例诊断，不是整个数据集的分布。局部冻结参考已计算时，另有 `images/reference_reliability_target`，用来比较 predictor 与软监督目标。诊断复用当前梯度累积组最后一个 microbatch 的现有特征，不额外执行 DINOv2 前向计算。

生成器重点查看 `train/loss_g`、`train/loss_d`、`train/loss_gan`、`train/loss_content`、`train/loss_content_source` 和 `lr/generator`／`lr/discriminator`。启用 identity 分支时还记录 `train/loss_content_identity`。`train/amp_scale` 与 `train/amp_step_skipped` 用于观察 AMP 状态。Images 面板包含 `images/day`、`images/fake_night`、`images/real_night`，以及启用 identity 时的 `images/identity_night`，每种最多三张图。这些预览复用已计算的模型输出。

需要减少图像诊断和写盘开销时，可设置 `tensorboard_image_interval: 0`，scalar 记录继续保留。

StableNight 的动态 Dustbin 使用固定 OT 容量。整体 Dustbin assignment 高，不表示模型把所有 patch 任意丢弃；结合 predictor 的分布、gamma、cluster mass 误差和检索指标一起判断。可靠性均值下界只是训练正则，不能把可靠性分数解释成已验证的建筑、道路或车辆语义标签。

当前 MSLS/SVOX 独立 profile 采用 `selection_rule: fixed_budget_last`，不在训练期间跑验证，因此不会产生 recall 曲线，也不会把内部未测量的 `-1` 占位值写成 recall。完整夜间检索仍需要训练后运行 `evaluate-trained-suite --datasets all`；公共 Seasons 也在标准 suite 中，隐藏 test 真值时只输出 Top-K 和 `null` 指标，不产生训练 Recall 曲线。见 [StableNight-SALAD 模型说明](stable_night_salad.md)。

## 恢复训练与日志边界

恢复时继续使用 checkpoint 保存的训练计数，writer 从已保存 step 的下一步设置清理边界。TensorBoard 读取日志时隐藏该边界及之后的旧事件，保留 checkpoint 对应 step 的已完成记录，避免将中断前未保存的日志与重跑结果叠加显示。训练恢复所需的数据、权重和配置约束继续生效；TensorBoard 不替代 checkpoint。旧自建 RobotCar 验证退出后，旧 VPR optimizer checkpoint 因有效配置变化不能直接续训，应使用新输出目录开始新实验。旧完整推理 checkpoint 可用于公共 Seasons 的标准 suite；生成器兼容恢复独立校验。仅改变日志字段仍不阻止同一实验恢复。全新训练不会自动清空已有事件文件，复用旧目录会混入旧日志，因此新实验应使用独立输出目录。

官方参考模型的外部训练脚本有各自的日志机制，本次没有改写第三方源码。`scripts/references/train_salad_small_gsv.py` 使用 Lightning 的默认 logger；具体 logger、目录及记录内容由其安装环境和官方实现决定。参考训练的日志不应与项目 `train-vpr` 的统一监控覆盖范围混淆。

## TensorBoard 接入阶段的历史验证

2026-10-14 已在本机 `a2b-lightvpr` 环境安装 TensorBoard 2.21.0，依赖检查通过。25 项监控专项测试覆盖真实事件文件读取、梯度累积、分类/CDND/StableNight 和生成器入口、日志开关、异常关闭及断点恢复。TensorBoard 接入阶段的完整回归为 253 项通过、6 项因外部资源缺失跳过；其中修复了 NumPy 1.26 兼容包未导出 `multiarray` 时的参考模型 checkpoint 加载问题，并在 NumPy 2 环境复核对应加载测试。

另外使用真实 MSLS/SVOX 图像与预训练 DINOv2-S 完成独立短训练：16 张源图、32 个 Day/Aug view、1 次有效 optimizer 更新。TensorBoard HTTP 服务成功读取 60 个 scalar 标签和 6 个图像标签，事件中另有 4 个 patch 分布标签；checkpoint 重载输出有限、L2 归一化的 544D 描述子。[事件与 checkpoint 检查记录](../runs/smoke/stable_night_tensorboard_20261014_145lhaiz/tensorboard_smoke_report.json)保存在独立 smoke 目录。该检查没有运行检索验证，也没有新增夜间性能结论。
