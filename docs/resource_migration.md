# 当前资源位置与迁移缺口

更新日期：2026-10-14。本文记录路径迁移、资源审计及公共 RobotCar Seasons 的统一 suite 使用方式。旧自建 RobotCar 测试已退出活动代码与配置，不再是待恢复资源；训练数据、模型、维度、采样预算与损失保留，checkpoint 选择改用公开白天验证 split。历史审计和最小训练记录仅证明对应阶段的工程检查，不代表完整夜间性能得到提升。

项目根目录为 `/home/admin123/github/Lightweight-Day-to-Night`。项目内 `data/`、`runs/` 以及配置中的其他相对路径按照原有约定从项目根目录运行。历史 manifest、教师缓存、训练 checkpoint、运行日志和结果文件不批量改写；已确认的旧路径通过读取时解析迁移，避免仅因路径重写改变原 CSV 的 hash、记录顺序或缓存身份。

## 当前可运行范围

| 任务／配置                                    | 当前状态                     | 证据与限制                                                                                   |
| --------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------- |
| StableNight 的 MSLS/SVOX Day/Aug 独立配置     | 资源检查通过，最小训练通过   | 109,366 张源图可访问；真实 DINOv2-S、CUDA AMP 更新及推理重载通过；完整训练和检索结果尚未验证 |
| 原三数据集 StableNight Day/Aug、Day/Fake 配置 | 资源检查未通过               | 118,313 条源图记录中缺 7,190 张原始 RobotCar 训练图；验证改用公开 MSLS/SVOX Day Validation                    |
| 已有 MixVPR 学生完整推理 checkpoint           | GPU 最小推理通过             | 无需重新读取丢失的原初始化文件；新训练仍有初始化资源缺口；旧 VPR optimizer checkpoint 不可按新验证配置直接续训 |
| Real Night Query → Day Database              | MSLS/SVOX/Tokyo 协议解析可用；Seasons 图片与 v2 划分可用 | Seasons 全量 6,954 DB／429 Query Top-10 流程已验证；真值隐藏，不计算 Recall；尚无性能结论 |
| Real Night Query → Fake Night Database       | 评估接口已有，测试输入待准备 | 需要测试 Database 的生成图及显式映射 CSV，不能使用训练伪夜图替代                             |

现有报告：[MSLS/SVOX Day/Aug](../runs/resource_audit/stable_night_salad_msls_svox_day_aug_544.json)、[三数据集 Day/Aug](../runs/resource_audit/stable_night_salad_day_aug_544.json)、[三数据集 Day/Fake](../runs/resource_audit/stable_night_salad_day_fake_544.json)、[公共 Seasons v2](../runs/resource_audit/robotcar_seasons_v2.json)。这些是本机检查快照，资源变化后应重新运行审计。

## 已确认的位置

| 资源                  | 当前位置                                                         | 检查与使用边界                                                                                |
| --------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| MSLS、SVOX、Tokyo247  | `/home/admin123/github/datasets/{MSLS,svox,tokyo247}`          | 根目录存在；使用原来的评测协议解析器                                                          |
| RobotCar Seasons | `/home/admin123/github/datasets/RobotCar-Seasons` | 公共图片与官方 v2 划分可用；与其他测试集共用标准 suite 入口 |
| DarkDriving、BDD100K  | `/home/admin123/github/datasets/{DarkDriving_Base,bdd100k}`    | 保留原生成器数据筛选规则                                                                      |
| GSV-Cities            | `/home/admin123/github/WORKSPACE/dataset/gsv-cities`           | `Images` 与 `Dataframes` 均存在；不替换为 AdaptCities                                     |
| DINOv2 repo           | `/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main` | 本地`hubconf.py` 存在；DINOv2-S register 权重严格加载检查通过                               |
| SALAD 官方 repo       | `/home/admin123/.cache/torch/hub/serizba_salad_main`           | 包含原训练入口和 GSV-Cities dataloader；参考包装脚本使用该缓存，避开另一个已修改的工作区 repo |
| CosPlace repo         | `/home/admin123/.cache/torch/hub/gmberton_cosplace_main`       | 可用于加载原有 R50-GeM-FC 架构；不代表所需 R50 权重也已齐全                                   |
| 预训练与教师权重      | `/home/admin123/.cache/torch/hub/checkpoints`                  | 具体匹配文件见下表                                                                            |
| CLIP 清晰天气筛选权重 | `/home/admin123/.cache/clip/ViT-B-32.pt`                       | 原路径仍然存在，没有改用其他 CLIP 模型                                                        |

| 已有权重文件                        | 对应模型                            | SHA256                                                               |
| ----------------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| `dinov2_vits14_pretrain.pth`      | plain DINOv2-S/14                   | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| `dinov2_vits14_reg4_pretrain.pth` | DINOv2-S/14 + 4 registers           | `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb` |
| `dino_salad_512_32.ckpt`          | DINOv2-B + Compact SALAD，544D 教师 | `d25c3434032a3ce5680ffcf51e05de0a7b9962772bd9112d88ca074689f18dcc` |
| `dinov2_12288.pth`                | 原 BoQ-B，12288D 教师               | `d72ee0ce899e2790be6cf6d1d57f75dfad0c76475bd2198e6dc0c3f0221b3583` |
| `ResNet50_512_eigenplaces.pth`    | EigenPlaces R50，512D               | `886a2fbd910e961d02e5c74ccc5e11cc1fe9a049a4fe290b9514878c691ce491` |

SALAD 544D 教师在 Torch Hub 缓存、`../huggingface/` 和工作区中的副本 hash 相同；配置统一使用缓存副本。没有把其他 `dino_salad.ckpt`、DINOv2-B/G 权重、CosPlace R18 或其他 descriptor 维度的权重当作替代文件。

## 仍需补齐的资源

| 缺口                                         | 配置或代码中的预期位置                                                                | 对运行的影响                                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| RobotCar 原训练白天帧                        | 原清单中的五条训练序列                                                                | VPR 缺 7,190 张；包含生成器训练源图时缺 8,721 张，不能用其他日期代替            |
| Fake Night Database 评估映射与测试生成图     | 配置项`evaluation.fake_database_manifest`                                           | 运行该 Database 域前还需准备；已有训练伪夜图不能当作测试 Database               |
| 公共 Seasons 测试正样本标注 | 需由可追溯的公开 VPR 协议或评测方提供 | 官方测试 Query 位姿隐藏；图片检索可运行，但本地 Recall 仍缺正样本定义 |
| MixVPR 官方初始化权重 128/512/4096D          | `third_party/weights/resnet50_MixVPR_*_channels(*)_rows(*).ckpt`                    | MixVPR 主实验与对应官方参考需要原文件及原 SHA；不能用已有学生训练结果替代初始化 |
| MixVPR 官方代码                              | `third_party/MixVPR/`                                                               | 本地官方代码一致性测试会明确跳过缺失资源                                        |
| BoQ 官方 aggregator repo                     | `third_party/Bag-of-Queries/`                                                       | BoQ-B 教师权重虽在缓存中，BoQ-S 训练及对应重载仍需要其代码                      |
| BoQ CDND 教师缓存                            | `data/teacher_cache/boq_dinov2_b_qsattn_official_150k_fp16.npy` 及配套 metadata/IDs | 该路线还缺对应离线缓存；不能替换为 SALAD 缓存                                   |
| CricaVPR repo                                | `third_party/CricaVPR/`                                                             | CricaVPR 训练及模型重载需要原实现                                               |
| Patch-NetVLAD repo 与原权重                  | `third_party/Patch-NetVLAD/`、`third_party/weights/pitts_WPCA4096.pth.tar`        | 4096D NetVLAD 官方参考暂不可完整构建                                            |
| CosPlace R50/512 官方权重                    | `third_party/weights/resnet50_512.pth`                                              | 原 registry 对应参考暂缺；缓存中的 R18/512 不兼容                               |

这些是明确的预期位置，不是声称文件已经存在。迁移没有自动下载、训练或生成这些缺失资源。旧自建 RobotCar 测试相关入口、配置和解析已移除，后续统一使用下节的公共 RobotCar Seasons。历史结果不批量删除或换名，不能混入新协议与新宏统计。

## 公共 RobotCar Seasons 夜间测试

按用户指定，后续 RobotCar 夜间测试接入 `/home/admin123/github/datasets/RobotCar-Seasons`。本机已有 32,796 张图片，其中白天参考图 20,862 张、night 1,314 张、night-rain 1,320 张，包含 left/rear/right 相机。已从官方发布目录补齐 v2 train/test 清单与 README；统一 suite 严格按 v2 test 选择 rear 的 429 张 night/night-rain Query，另有 1,443 张非夜间条件 Query；两项 Database 均为 6,954 张 rear 白天参考图。公开训练集中的夜间图不被当作测试 Query。

公共 Seasons 与 MSLS/SVOX/Tokyo 共用评测入口；`data.robotcar_seasons_root` 替代旧测试必填字段：

```bash
python -m a2b_lightvpr.cli audit-robotcar-seasons \
  --output runs/resource_audit/robotcar_seasons_v2.json

python -m a2b_lightvpr.cli evaluate-trained-suite \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/msls_svox_day_aug_seed0/model_best.pt \
  --datasets robotcar_seasons_night robotcar_seasons_day
```

官方测试 Query 的位姿隐藏，本地不能凭图片计算可信的 Recall@K；suite 输出 Top-K 排名，`recalls`、`mrr`、`valid_queries` 为 `null`，并记录无公开真值状态。若要报告本地 VPR Recall，仍需可追溯的正样本标注；若进行官方 6DoF 定位评测，还需参考 3D 模型、位姿估计与官方评测服务。标准学生／参考 suite、别名和 Fake Night Database 映射见 [公共 RobotCar Seasons 接入说明](robotcar_seasons.md)。该替换不补齐下文缺失的旧训练白天帧。

前一版独立公共入口的历史回归为 268 项通过、6 项因外部资源缺失跳过，不能当作本次统一 suite 的验证结果。真实 Seasons 图片与 StableNight 544D checkpoint 的 16 张 Database／2 张 Query Top-3 检索检查通过，[输出记录](../runs/smoke/robotcar_seasons_20261014_dhn2gh5a/retrieval.json)明确标记为 subset 和无公开 test 真值，不能当作完整数据性能结果。


本次通过标准 `evaluate-trained-suite` CLI 完成公共 Seasons Night 全量流程：6,954 张 Database、429 张 Query，各输出 Top-10；[统一 suite 记录](../runs/smoke/robotcar_seasons_unified_20261014_4x7pvkuz/final_suite/summary.json)保留 `ground_truth_available: false` 和 `null` 指标。checkpoint 来自仅一次 optimizer 更新的历史 smoke 训练，这不是模型性能评测。Night 监督宏固定为 SVOX Night + Tokyo Night，Day 为 SVOX Day + SVOX Sun + Tokyo Day；Seasons 单列状态，宏内缺测则整体为 `null`，历史宏不能混入新定义。

本次统一后的完整回归为 290 项通过、5 项外部资源缺失跳过；38 份配置加载和相关脚本语法检查通过。上文三份训练资源审计已按当前配置重新生成：MSLS/SVOX profile 的缺失源图为 0、`training_ready: true`；原三数据集 Day/Aug、Day/Fake 的缺失训练源图仍为 7,190，不再报告旧测试协议缺口。

## 旧训练资源审计记录

`data/generated/`、`data/teacher_cache/` 和 `runs/` 中存在历史实验产物，但目录存在、manifest 存在和数组存在分别不能证明所有原始图像与生成图都可访问。尤其是丢失图像时，路径解析只能定位资源，不能补出图像，也不能静默剔除记录来通过计数校验。生成器训练目标图与 VPR 地点源图也应分别审计。

路径迁移阶段对历史 CSV 去重后的路径进行存在性审计：1,690,620 个引用中恢复定位 1,681,899 个，仍缺 8,721 个，剩余缺口为 RobotCar 的原始白天图。全部 888,587 个去重后的生成图路径均可访问。原三数据集 VPR 源 manifest 中，RobotCar 8,947 条源图记录仍有 7,190 条缺失，保留原记录和地点标签；没有删掉缺失行来伪造完整数据集。这些计数是文件存在性检查，不代表逐张完成图像解码。

缺失训练序列和原清单计数如下。生成器源图与 VPR 源图有重叠，两列不能相加作为额外缺失总数。

| RobotCar 训练序列       | 缺失 VPR 源图 | 缺失生成器训练源图 |
| ----------------------- | ------------: | -----------------: |
| `2014-11-28-12-07-13` |         1,644 |              1,725 |
| `2014-12-16-09-14-09` |         1,760 |              1,800 |
| `2015-03-03-11-31-36` |         1,681 |              1,745 |
| `2015-05-19-14-06-38` |         1,752 |              1,807 |
| `2015-08-27-16-07-13` |           353 |              1,644 |
| 合计                    |         7,190 |              8,721 |

当前可恢复的 `2015-02-06-13-57-16` 已按原 timestamp 定位，没有将现存验证／测试序列转为训练数据。公共测试接入不会补出这些训练帧；不能把 Seasons test 或其他日期的图片替代原训练记录。

既有 SALAD 教师缓存的 118,313×544 数组及配套 manifest hash、checkpoint hash 和图像 ID 校验已通过，原 CSV 和缓存不需重写。真实 DINOv2 的 CPU/CUDA 特征与描述子最小推理检查，以及既有 MixVPR 学生 checkpoint 的 GPU 推理检查已通过。完整的已训练 MixVPR checkpoint 重载可直接使用模型状态，不再要求重新读取未参与推理的初始化权重；开始新的 MixVPR 训练仍需要原初始化文件。尚未执行新的完整训练，也没有重新计算全套夜间检索结果。

最初路径迁移阶段的历史回归为 227 项通过、6 项因缺失外部参考资源跳过。另用真实 MSLS/SVOX 图像和预训练 DINOv2-S 完成独立 CUDA AMP 最小训练：16 张源图、32 个 Day/Aug view、1 次有效 optimizer 更新；导出的推理 checkpoint 严格重载后输出有限、L2 归一化的 544D 描述子。[最小训练记录](../runs/smoke/stable_night_migration_20261014_l76t3hle/training_metrics.jsonl)和[当次解析配置](../runs/smoke/stable_night_migration_20261014_l76t3hle/resolved_config.json)位于独立 smoke 目录，仅用于迁移验证，不是检索性能结果；该记录没有验证集 recall。

## 仅使用当前可用源图的独立训练配置

新增 [MSLS/SVOX Day/Aug 配置](../configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml)，仅使用 MSLS/SVOX 的 Day/Aug 源图，不加载生成图、CDND 教师或 RobotCar 验证集。它按最后一个完成的预算 epoch 选择 checkpoint，关闭 epoch-zero 评测；每个 dataset 的 epoch quota 为 20,000，总计 40,000 源图/epoch，20 epochs。它与原三数据集的 60,000 源图/epoch 和验证选择配置不同，因此是独立 profile，不能冒充原三数据集基线。

该 profile 的资源与图像存在性检查通过：109,366 张源图均可访问，MSLS 有 18,960 个合格地点、SVOX 有 9,436 个合格地点；原三数据集 Day/Aug 和 Day/Fake 仍受 RobotCar 训练图缺口影响。上文链接的当前训练资源报告已重新生成，可复查 `runs/resource_audit/*.json`。

原四种训练模式和独立局部监督配置保留训练数据集范围、采样预算及损失；活动验证改为公开白天 val split，旧 checkpoint 选择结果不能当作该配置的新结果。使用当前可用 profile 时，需要另外在现有真实夜间 Query 协议上评估其检索能力，不能把资源审计的 `training_ready` 当作已测性能。

## 验证与运行

最初路径迁移阶段在 37 份既有 YAML 中更新了 317 个路径 scalar，另更新 8 个脚本中的资源路径。三个数据准备脚本的身份与文件检查使用内存中的路径解析，输出仍保留原 CSV 路径字符串；相同输出不重写，缺失源图仍明确失败。当时递归比较确认未改变非路径值；本次另移除旧测试字段、改用公共白天验证集，标准项目配置可通过 `load_config`。安装仍使用 README 的 [environment.yml](../environment.yml)，脚本保持使用当前环境的 `python` 或显式 `PYTHON_BIN`。

SALAD 官方训练包装脚本等外部参考代码还可能需要 `pytorch_lightning` 等对应 repo 依赖，环境文件没有承诺安装每一个官方参考实现的完整环境。应先检查目标脚本的依赖和资源，不通过换架构或换 checkpoint 绕开缺口。

可先执行配置与单元检查：

```bash
python -m a2b_lightvpr.cli --help
python -m pytest
```

使用新增的只读资源审计入口比较当前可用 profile 与原三数据集配置：

```bash
python -m a2b_lightvpr.cli audit-resources \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --output reports/resources_msls_svox.json

python -m a2b_lightvpr.cli audit-resources \
  --config configs/experiments/students/stable_night_salad_day_aug_544.yaml \
  --output reports/resources_three_datasets.json
```

`training_ready` 检查所选训练资源、manifest 标签与 split、地点类别/PK 采样条件及图像文件存在性；报告同时区分评测资源缺口。它不加载权重或解码图片，也不校验教师缓存内部内容、运行依赖和训练显存。使用 `--skip-images` 时不扫描图片，缺失图像计数为未知，readiness 的范围进一步缩小到资源路径和 manifest/sampling 结构。

该配置的 `--smoke-epochs 1` 单独使用时仍会读取完整的 40,000 源图预算。以下命令显式缩小为 16 张源图、两个 PK batch、一次梯度累积更新，并保存到独立输出目录。每次复跑应选择新的 `vpr.output_dir`，避免覆盖已有产物：

```bash
python -m a2b_lightvpr.cli train-vpr \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --smoke-epochs 1 \
  --set vpr.epochs=1 \
  --set vpr.workers=0 \
  --set vpr.samples_per_epoch=16 \
  --set 'vpr.dataset_epoch_quotas={msls: 8, svox: 8}' \
  --set vpr.output_dir=runs/smoke/stable_night_migration_check
```

完整训练使用原配置且不加上述预算覆盖：

```bash
python -m a2b_lightvpr.cli train-vpr \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml
```

VPR 和生成器训练默认记录 TensorBoard 日志。在另一个终端从项目根目录运行：

```bash
tensorboard --logdir runs --port 6006
```

浏览器打开 <http://localhost:6006>。事件文件在各实验的 `output_dir/tensorboard/`；最小训练也会写入对应 smoke 目录。这个 MSLS/SVOX profile 不运行训练验证，因此监控 loss、学习率和 StableNight 聚合诊断，不产生 recall 曲线。TensorBoard 随项目安装；已有环境需重新执行 `python -m pip install -e '.[test,eval]'`。完整配置和日志恢复规则见 [TensorBoard 训练监控](training_monitoring.md)。监控功能不补齐本文列出的图像、协议、权重或参考源码缺口。

本机 `a2b-lightvpr` 环境已安装 TensorBoard 2.21.0，并通过独立 MSLS/SVOX 短训练及 TensorBoard HTTP 读取检查；[监控接入阶段验证记录](../runs/smoke/stable_night_tensorboard_20261014_145lhaiz/tensorboard_smoke_report.json)保留事件标签、更新步数和 544D checkpoint 重载检查。它与上文迁移时的最小训练记录分开保存，均不代表检索性能实验。

完整训练完成后，可用同一配置和推理 checkpoint 在当前具有本地正样本定义的三个真实夜间协议上评估。也可在同一次标准 suite 中包含公共 Seasons；它会导出排名并保留无真值状态：

```bash
python -m a2b_lightvpr.cli evaluate-trained-suite \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/msls_svox_day_aug_seed0/model_best.pt \
  --datasets robotcar_seasons_night msls_night svox_night tokyo_night
```

完整训练和完整四项检索尚未执行。`--datasets all` 运行夜间组（另含 Tokyo Sunset），`all-day` 运行白天与白天验证组，`all-conditions` 运行两组；Seasons Day 可用独立 key。Fake Night Database 的映射格式与评估参数见 [StableNight 模型说明](stable_night_salad.md#real-night-query--两种-database)。

原始 RobotCar 文件补齐后，再运行对应配置的数据审计，例如：

```bash
python -m a2b_lightvpr.cli audit-data \
  --config configs/generator/qsattn_official_objective_150k.yaml \
  --decode
```

`audit-data` 当前覆盖 RobotCar 与 generator manifest，不能替代全部 VPR 源图、生成图和第三方模型文件的检查。每次审计应同时报告记录数、可访问文件数、缺失数和 split；没有对齐的记录不能只报告“根目录存在”。

VPR checkpoint 选择只使用 `msls_day_val` 或 `svox_day_val`；多验证集按协议等权平均，test 不参与训练选择。旧 VPR optimizer checkpoint 因验证／配置变化被严格拒绝，应使用新配置、新输出目录开始新实验。TensorBoard 日志开关不影响配置匹配；生成器旧恢复按训练资源与兼容身份另行校验。旧 snapshot 和结果中的历史路径保留以便追溯；迁移后的活动路径解析不应被理解为已经改写或重新计算了那些历史结果。

现有 MixVPR/NetVLAD 的完整推理 checkpoint 可以直接恢复模型张量，不再需要额外加载丢失的原初始化文件；新训练仍需要对应初始化资源；完整推理不受旧验证配置退出影响。StableNight 的源身份和增强 seed 以规范化源路径为依据，目录迁移前后的增强序列不承诺逐步相同。目前保留了最小训练产物与公共 Seasons 全量流程检查；尚无完整 StableNight 训练或已测真实夜间性能结果。这些限制不影响同一当前位置、相同 seed/epoch 下的 worker 增强可复现性。
