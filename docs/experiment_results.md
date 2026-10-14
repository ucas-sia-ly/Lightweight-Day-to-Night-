# 当前实验与清理记录

更新日期：2026-10-15。当前比较均为 plain DINOv2-S/14 + SALAD 544D、224 resize。训练只用 GSV-Cities 原始地点标签；两组均为 seed 0、每轮 251,392 次源图采样、两轮、物理 batch 32×4。checkpoint 仅按公开 SVOX Day Validation 选择，不按夜间测试挑选。

## Recall@1

| 模型 | MSLS Night（55 Q） | SVOX Night（823 Q） | Tokyo Night（105 Q） | Tokyo Sunset（105 Q） |
| --- | ---: | ---: | ---: | ---: |
| 完整初始化 | 81.82% | 66.95% | 80.00% | 90.48% |
| Day-only（重训） | 90.91% | 66.95% | 76.19% | 87.62% |
| Day-aug | 87.27% | 68.04% | 74.29% | 88.57% |

这些结果没有证明继续训练或当前光照增强在所有测试集上优于完整初始化。原始比较中，Day-aug 的 SVOX R@1 提升约 1.09 个百分点，Tokyo Night 下降约 5.71 个百分点；不能只报告改善的测试集。单种子、两轮实验不支持统计显著性结论。

MSLS Night 是项目 55 Query 夜间 val 子集，不能写成标准 MSLS 全部 740 Query 的成绩。Tokyo 75,984 DB crop 属于 6,332 个 panorama，按 panorama 做 NMS；标准每条件 105 Query，不与已退役的 raw Tokyo 375 Query 协议混合。公共 Seasons 为 v2 test／rear、6,954 DB、429 夜间 Query，保存完整 Top-10，隐藏真值的 Recall/MRR 为 `null`。不跨协议生成宏平均。

逐数据集 R@1/5/10、MRR、协议指纹与权重身份见 [JSON汇总](../reports/metric_experiments_20261014.json)和[CSV](../reports/metric_experiments_20261014.csv)。

## 当前权重与原始证据

- 完整初始化：[原始 suite](../runs/vpr/salad_small_gsv_metric/pretrained_224_lowmem_eval/final_suite/summary.json)；epoch 0，文件 SHA256 `0bf1b6cf3a8894d088ef26b81f48cd22d1341d080f4e0894f1509c870948e925`。
- Day-only（重训）：[原始 suite](../runs/vpr/salad_small_gsv_metric/day_only_reverified_20261014/final_suite/summary.json)；epoch 2，文件 SHA256 `439a7bd758f5d2835d0051e5a588fb5a42c0307630ccc85c7e1c74989a3f90e1`。
- Day-aug：[原始 suite](../runs/vpr/salad_small_gsv_metric/day_aug_reverified_20261014/final_suite/summary.json)；epoch 2，文件 SHA256 `2a3c212c54b1edcc0621ef2a0c298697b4f34fc5a87444d71281e9abd2e23a2b`。

已完成的实验目录后来再次被用于新训练，导致原 Day-only 模型被替换、Day-aug 的 best/JSONL 被 epoch-zero 覆盖。Day-only 的二次训练权重与原实验不同，已在独立目录完成重测；旧 suite 保留作历史证据。Day-aug 的原始 epoch 2 latest 仍完整，保存的 best hash 与原 suite 一致；独立恢复的 `model_recovered_epoch2.pt` 全部 188 个 tensor 与该 latest 精确相等。恢复文件的序列化 SHA 变化，不能声称字节级恢复。两组新评测用独立目录，不覆盖旧结果。

恢复过程、模型 tensor digest、原文件身份及 TensorBoard 重建的 epoch 记录见 [恢复报告](../runs/resource_audit/metric_checkpoint_recovery.json)。原训练日志和所有 TensorBoard 事件保留；JSONL 重建使用 float32 scalar，损失等数值不具备原始 double 精度。代码现在拒绝对已有训练产物的目录无 resume 开始训练，避免重复覆盖。

## 资源与剩余工作

推理仍为 544D；当前初始模型 batch 1 CUDA AMP 的 p50/p95 为 2.608/2.898 ms，CUDA 分配峰值约 193.49 MiB，机器是本次 RTX 4090，不能当作其他硬件延迟。32 GB 机器训练/评测串行执行，workers=0；训练数据精简存储，最近 gallery 缓存不超过 256 MiB。

Real Night → Day Database 已有当前完整检索证据。Real Night → Fake Night Database 保留公共评测接口，但尚无对应测试生成图和完整映射，不能报告其结果；旧训练生成图不属于测试 Database。后续应先研究真实训练夜间数据与验证方案，再增加新的复杂模块；本次没有重新宣称 StableNight 创新有效。

## 本次删除与保留

已删除项目内约 **169.34 GiB** 旧派生产物。删除清单逐项记录实际路径和原占用：[清单](../reports/cleanup_manifest_20261014.json)。原始公共数据集、GSV 源图和外部预训练权重保留。

- 删除旧 QS-Attn 生成图、teacher/PCA/词典缓存、旧 CDND/MixVPR/BoQ 等实验权重、失败并发产物、定性图片缓存及 smoke 重复权重。
- 删除 98 个旧脚本、配置、论文/生成器/定性流程模块与失效测试文件。活动配置仅两份 metric 训练配置和一份 SALAD reference registry；共用的数据、聚合、checkpoint 和公共评测底层保留。
- 保留完整 SALAD-S 初始化、当前 GSV manifest、初始推理模型、两组 latest/恢复推理权重、TensorBoard、当前完整评测及必要旧 suite 证据。
- 历史指标与身份压缩存于 [旧实验摘要](../reports/retired_experiments_20261014.json.gz)，[旧指标目录](../reports/retired_metrics_20261014.csv)将废弃自建 RobotCar/旧 Tokyo 标为不可比较；[退役源文件文本](../reports/retired_sources_20261014.json.gz)为本机小体积记录，不包含旧图片、缓存或权重。

这些本地产物不入 Git；文档和当前代码保留在工作区中。

进一步评价及后来终端训练版本的身份边界见[2026-10-15 分析](result_analysis_20261015.md)。
