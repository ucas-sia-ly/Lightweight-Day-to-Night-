# 当前资源与运行约束

更新日期：2026-10-14。当前训练主线为 [SALAD-S／GSV 地点度量训练](salad_metric_baseline.md)，结果统一见 [实验结果](experiment_results.md)。本页只记录当前需要的资源；旧生成器、教师缓存和自建 RobotCar 测试不再是待恢复项目。

项目根目录为 `/home/admin123/github/Lightweight-Day-to-Night`。配置中的相对路径从项目根目录解析；数据和权重迁移时应重新审计，不能只修改历史结果中的路径或指纹。

## 已有资源

| 资源 | 本机位置 | 用途 |
| --- | --- | --- |
| GSV-Cities | `/home/admin123/github/WORKSPACE/dataset/gsv-cities` | 当前两份配置唯一训练数据源 |
| GSV 地点训练 manifest | `data/manifests_gsv_metric/vpr_train.csv` | 23 城市、62,514 地点、524,701 张不同源图；每地点至少 4 张 |
| MSLS | `/home/admin123/github/datasets/MSLS` | 公开验证／夜间子集检索 |
| SVOX | `/home/admin123/github/datasets/svox` | 白天验证选择 checkpoint；真实夜间检索 |
| Tokyo24/7 | `/home/admin123/github/datasets/tokyo247` | 标准 315 Query／每条件 105，panorama NMS |
| RobotCar Seasons | `/home/admin123/github/datasets/RobotCar-Seasons` | 官方 v2 test／rear 公共检索 |
| DINOv2 repo | `/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main` | plain DINOv2-S/14 架构 |
| DINOv2-S 权重 | `/home/admin123/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` | 构造预训练 backbone |
| 完整 SALAD-S 初始化 | `runs/reference/salad-dinov2-s-544-gsv/last.ckpt` | 同时初始化 backbone 和聚合器 |

完整初始化 SHA256 为 `40ac2337624ab029026291f6d4d8a89a20001bb45627aa77ea61789f6af36d4c`，训练 manifest SHA256 为 `32ece2b07ca58e827bddf7460f6ce423dcd145e83edd2d6e3f1515ac205a8ed0`。manifest 审计保存在 [audit.json](../data/manifests_gsv_metric/audit.json)，资源审计为文件存在性与元数据检查，不等于逐张解码。

初始推理权重位于 `runs/vpr/salad_small_gsv_metric/pretrained_224/model_initial.pt`，两组训练产物位于 `day_only_seed0_lowmem/` 和 `day_aug_seed0_lowmem/`。当前成绩及对应权重身份以实验结果和原始 suite 为准。

## 32 GB 内存限制

训练和评测一次只运行一个重任务。主配置设置 `workers: 0`，训练为 P32×K4 的物理 batch 128，评测提取 batch 64；减小评测 batch 不改变训练 miner 的候选池。

训练 CSV 按行读取，运行时只保留必要字段并共享重复字符串；完整原始元数据仍在 CSV。suite 只缓存最近一个 Database 的描述子，缓存上限 256 MiB。GPU 不可访问时，`vpr.require_cuda: true` 会停止完整计算。

启动新任务前至少保留 6 GiB MemAvailable；运行中持续低于 4 GiB 时停止当前实验和后续队列。这是监控阈值，不能代替持续观察实际内存。不要并行启动训练、完整 suite 或全量测试，也不要为实验终止其他桌面程序。

## 当前缺口

| 目标 | 仍需要什么 |
| --- | --- |
| Real Night Query → Fake Night Database | 测试 Database 的生成图，以及覆盖完整源 Database 的显式映射 CSV；记录生成器、权重和映射 SHA |
| RobotCar Seasons 本地 Recall | 可追溯的 test 正样本标注；官方 test Query 位姿隐藏，目前只输出排名 |
| 更可靠的夜间提升结论 | 多 seed、相同协议和预算的实验；当前继续训练的改善因数据集而异，Tokyo 有回退 |

无需补回旧 RobotCar 训练日期、旧 MixVPR 初始化或旧教师缓存来运行当前 GSV 主线。未来真实夜间训练必须先核实 train split 和地点正样本定义，不能使用 val/test Query 训练。

## 检查命令

```bash
a2b-lightvpr audit-resources --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml
a2b-lightvpr audit-robotcar-seasons --output runs/resource_audit/robotcar_seasons_v2.json
```

旧实验的大型权重、缓存、重复评测和论文产物已清理；可追溯的历史指标与路径汇总保存于 `reports/retired_experiments_20261014.json.gz`。它们使用原协议身份，不与标准 Tokyo24/7 或新主线成绩合并。
