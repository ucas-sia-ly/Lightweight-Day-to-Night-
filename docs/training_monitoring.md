# TensorBoard 训练监控

当前两个 [SALAD-S／GSV 配置](salad_metric_baseline.md)默认写入 `vpr.output_dir/tensorboard/`，同时保留 `training_metrics.jsonl`、解析后的配置和 checkpoint。已完成实验的结论见 [实验结果](experiment_results.md)。

## 打开监控

```bash
conda activate a2b-lightvpr
tensorboard --logdir runs/vpr/salad_small_gsv_metric --port 6006
```

浏览器打开 <http://localhost:6006>，选择 Day-only 或 Day-aug 的事件目录。同一实验的续训事件在同一目录；新的实验应使用独立输出目录。

## 配置与指标

```yaml
vpr:
  tensorboard_enabled: true
  tensorboard_log_interval: 10
```

`tensorboard_enabled: false` 关闭事件记录。默认还在首个有效更新和 epoch 结束时记录 scalar；`run/config` 保存本次配置。

| 指标 | 观察目的 |
| --- | --- |
| 训练 place loss | 检查数值与趋势；下降本身不证明夜间提升 |
| `active_anchor_fraction`、`active_anchors` | PML reducer 中实际非零 loss 的 anchor 比例与数量 |
| `mined_positive_pairs`、`mined_negative_pairs` | miner 找到的有向训练对数量 |
| `mined_anchor_fraction`、`both_pair_anchor_fraction` | 有 mining 对的 anchor，以及同时有正负对的 anchor 比例 |
| `augmentation_fraction` | 实际增强比例；Day-only 应为 0，Day-aug 期望约 0.5 |
| 学习率、AMP scale／跳步 | 检查优化器和 AMP 是否正常更新 |
| 样本与 optimizer step 计数 | 核对实际完成预算 |
| SVOX Day Validation Recall | 初始和每 epoch 的 checkpoint 选择依据 |

比例分母为实际物理 batch 128。没有 hard pair 的 batch 可以有有限零 loss；长期无有效 mining 时，应检查地点标签、不同源图采样与初始化。当前每源图只产生一个 view，同地点其他源图提供正样本，训练不使用局部可靠性监督。

主线记录 `student_images=源图数量`、`reference_images=0`。兼容日志中可能出现历史 `lr/classifier` tag，但当前模型没有训练分类器；它不属于需要解释的主线参数。

夜间检索在训练后通过 suite 独立运行，不用于 checkpoint 选择。Seasons 的隐藏 test 真值不会产生有效 Recall 曲线。

## 断点恢复

```bash
a2b-lightvpr train-vpr \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --set vpr.output_dir=runs/vpr/salad_small_gsv_metric/day_aug_new \
  --set vpr.resume=runs/vpr/salad_small_gsv_metric/day_aug_new/checkpoints/latest.pt
```

`checkpoints/latest.pt` 恢复同一次训练，`model_best.pt` 用于推理。恢复仍检查数据、配置、训练计数和历史 best 权重身份；不能将新的消融或 seed 当作同一实验恢复。

TensorBoard 按 checkpoint 保存的 step 设置清理边界，隐藏中断后未保存的旧事件，避免重复计数。续训事件文件的时间戳晚于已有文件，打开 writer 时最多额外等待约一秒；已有日志不改写。跨目录恢复会保留经过 hash 校验的 best 推理权重。

全新训练会写入新的训练日志；重复使用已有输出目录可能覆盖 checkpoint 和 JSONL，并混入旧事件。开始新实验必须设置新的 `vpr.output_dir` 和相应日志目录。

## 内存与结果

32 GB 内存机器使用 `workers: 0`，一次只训练或评测一个模型。TensorBoard 可保留打开，但不要同时启动第二个训练任务。评测 batch 64 与训练 batch 128 分别配置，梯度累积不会扩大 miner 的候选集合。

训练记录、validation-best checkpoint、suite 指纹和性能应一起核对，不能用单次 smoke 更新或旧论文数字替代本轮结果。

## CPU 线程与“卡住”排查

`workers: 0` 会让每张图的缩放、归一化和光照增强在主进程执行。CPU 持续满载而 GPU 空闲时，过多 Torch intra-op 线程可能让这些小操作变慢。CLI 默认限制为 2 个 CPU 线程，并尊重显式 `OMP_NUM_THREADS`／`MKL_NUM_THREADS`；新日志打印实际线程数。该修改不会改变正在运行的旧进程。

检查 TensorBoard 的 step 和事件时间是否继续增加，并核对真实 `train-vpr` 进程；终端启动的 `a2b-lightvpr` 与 `python -m a2b_lightvpr.cli` 都要覆盖。检测到其他训练时，评测应等待；不要仅检查自己队列的状态。

若旧进程过慢，可在终端按 Ctrl+C 后从最近完成的 epoch 续训，使用 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 a2b-lightvpr train-vpr --config 配置 --set vpr.resume=对应目录/checkpoints/latest.pt`。当前 checkpoint 按 epoch 保存，中断会重算尚未完成的这一轮。`xFormers is not available` 是可选库缺失提示，不等同于训练退出。
