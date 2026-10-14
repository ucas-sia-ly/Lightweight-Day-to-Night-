# RobotCar Seasons 公共测试

RobotCar 测试统一使用 `/home/admin123/github/datasets/RobotCar-Seasons`，与 MSLS、SVOX、Tokyo 一样通过 `evaluate-trained-suite` 或 `evaluate-reference-suite` 运行。旧自建测试解析器、配置字段和协议 key 已移除；不再要求恢复旧测试 manifest/protocol。原始 RobotCar 训练源图与训练 manifest 仍保留，其缺帧问题独立处理。

## 数据与固定协议

本机共有 32,796 张 JPG，包含 left/rear/right 三个相机。统一 suite 固定使用官方 **v2 test、rear 相机**：

| Suite key | Query 条件 | Query 数 | 白天 Database 数 |
| --- | --- | ---: | ---: |
| `robotcar_seasons_night` | night、night-rain | 429 | 6,954 |
| `robotcar_seasons_day` | dawn、dusk、overcast-summer、overcast-winter、rain、snow、sun | 1,443 | 6,954 |

Database 为 `images/overcast-reference/rear/`。官方 [v2 说明](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar_v2.md)区分公开训练位姿与隐藏真值的测试集。本机已有 `robotcar_v2_train.txt`、`robotcar_v2_test.txt` 和官方 README。解析将 test 清单的 `rear/<timestamp>.png` 唯一映射到 `<condition>/rear/<timestamp>.jpg`，检查与 train 清单不重叠；缺失、歧义或重叠时明确失败。

配置只需 `data.robotcar_seasons_root`，默认指向上面的本机目录。其他机器可用 `--set data.robotcar_seasons_root=/path/to/RobotCar-Seasons` 覆盖。统一 suite 不通过相机或条件筛选改动这两项协议；原始三相机图片统计不能作为 v2 测试计数。

## 审计

```bash
python -m a2b_lightvpr.cli audit-robotcar-seasons \
  --data-root /home/admin123/github/datasets/RobotCar-Seasons \
  --output runs/resource_audit/robotcar_seasons_v2.json
```

审计无需模型配置，记录文件数、划分文件 SHA256、图片清单指纹及缺真值状态。`image_retrieval_ready` 只检查文件与划分映射，不代表逐张解码或模型性能验证。审计的 `--protocol original`、`--conditions` 和 `--cameras` 仅用于检查图片集合，不更改统一 suite 的固定协议。

新机器若只有图片，需要从[官方发布目录](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/)补齐 v2 清单。训练与评测不会自动联网下载这些文件，也不会用公开 train 位姿充当 test 真值。

## 学生模型评测

使用与 checkpoint 匹配的模型配置：

```bash
python -m a2b_lightvpr.cli evaluate-trained-suite \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/msls_svox_day_aug_seed0/model_best.pt \
  --datasets robotcar_seasons_night robotcar_seasons_day \
  --set evaluation.database_domain=day
```

输出与其他测试集一致：`vpr.output_dir/final_suite/summary.json` 汇总，`final_suite/robotcar_seasons_night/metrics.json` 和 `final_suite/robotcar_seasons_day/metrics.json` 保存各协议结果。各 Query 的 `retrieval_predictions` 包含 image ID、路径及 Top-K Database ID、路径与相似度；K 取 `evaluation.recalls` 中的最大值，默认 10。结果保留 checkpoint、输入变换、官方划分与协议指纹，不将旧结果改名复用。

两种 suite 入口都支持相同别名：

- `--datasets all`：Seasons Night、SVOX Night、MSLS Night、Tokyo Night、Tokyo Sunset。
- `--datasets all-day`：MSLS/SVOX Day Validation、Seasons Day、SVOX Day/Sun、Tokyo Day。
- `--datasets all-conditions`：上述两组协议。

也可直接指定任意协议 key。MSLS Night 沿用其公开 val 划分；别名只组合评测协议，不改变每个协议的 split。已移除独立公共检索命令，全部模型共用 suite 入口。

## 参考模型评测

例如运行注册表中的 EigenPlaces：

```bash
python -m a2b_lightvpr.cli evaluate-reference-suite \
  --config configs/references/day_completion.yaml \
  --model eigenplaces-r50-512-320 \
  --database-encoder adapted --transform native \
  --datasets robotcar_seasons_night robotcar_seasons_day \
  --output-dir runs/reference/eigenplaces_seasons_v2
```

该入口使用注册表指定的权重、描述子接口和变换，所需参考资源仍需齐全。各协议同样输出 `metrics.json`，根目录输出 `summary.json`；Seasons 与其他测试集可在同一次 suite 中评测。

## Fake Night Database

Real Night Query → Fake Night Database 仍用同一入口，只替换 Database：

```bash
python -m a2b_lightvpr.cli evaluate-trained-suite \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/msls_svox_day_aug_seed0/model_best.pt \
  --datasets robotcar_seasons_night \
  --set evaluation.database_domain=fake_night \
  --set evaluation.fake_database_manifest=data/generated/robotcar_seasons_reference/manifest.csv
```

CSV 字段为 `source_path,generated_path`，可附加 `split=test`，需覆盖所选协议的全部白天 Database。生成图必须存在、映射唯一，并与真实 Database/Query 不重合。Query 仍是真实测试图，Database ID 保留白天源身份；结果位于 `final_suite/fake_night/`。参考 suite 也支持这些配置。示例 CSV 路径不代表该份 Fake Database 已生成；训练伪夜图不能替代测试 Database。

## 无真值结果与比较边界

官方测试 Query 位姿不公开，因此统一返回 `ground_truth_available: false`、`evaluation_status: retrieval_only_no_public_test_gt`，`recalls`、`mrr`、`valid_queries` 为 `null`，同时保留真实 Query 总数和检索排名。未知真值不能作为零 Recall、无正样本或失败 Query，也不能依据检索相似度生成标签。排名不等同于[官方定位评测](https://www.visuallocalization.net/benchmark/)所需的 6DoF 位姿。

报告采用 `fixed_supervised_protocols_v2`：Night 宏固定为 **SVOX Night + Tokyo Night**，Day 宏固定为 **SVOX Day + SVOX Sun + Tokyo Day**。Seasons 两项单列无真值状态；固定宏中任一协议未测，整个宏和配对差值均为 `null`，不改取可用子集。图表排除未测点并记录 `excluded`。旧自建协议结果和旧宏数值不能混入新协议，也不能把历史结果换名当作新测试结果；历史文件仅保留追溯。

定性导出可通过 `export-qualitative-top1 --selection ...` 显式选择公共 Seasons Query，记录 Top-K、展示 Top-1，但不给正确/错误颜色、正样本或几何距离。自动成功/失败案例搜索要求真值，因此只搜索 MSLS/SVOX/Tokyo，并记录 Seasons 的排除理由。

训练 checkpoint 选择只允许 `msls_day_val`、`svox_day_val` 的白天验证 split；同时配置两者时按协议等权平均 Recall。公开 Seasons test 与其他 test 协议不用于 early stopping 或选择最佳 checkpoint。旧完整推理 checkpoint 可继续评测；由于活动验证配置已改变，旧 VPR optimizer checkpoint 不可冒充新实验续训，应使用新输出目录。生成器的兼容恢复独立处理。

## 统一 suite 流程检查

本次完整回归为 **290 项通过、5 项因外部资源缺失跳过**。覆盖统一学生／参考 suite、无真值状态、Fake Database、缓存身份、公开训练验证、旧配置退休、报告宏与定性导出；38 份 YAML 加载及相关脚本语法检查通过。

标准 CLI 已用真实 **6,954 张白天 Database 和 429 张夜间 Query** 完成全量 Top-10 检索，输出[统一 suite 记录](../runs/smoke/robotcar_seasons_unified_20261014_4x7pvkuz/final_suite/summary.json)。429 项 Query 各保留 10 个匹配，隐藏真值对应的 Recall/MRR/有效 Query 数均为 `null`。使用的 StableNight 544D checkpoint 来自仅一次 optimizer 更新的历史 smoke 训练，因此这验证的是完整数据上的运行流程，不能据此报告模型性能或方法收益。

## 历史检查记录

前一版独立接入曾完成 268 项测试、6 项资源缺失跳过，并用真实 Seasons 图片与 StableNight 544D checkpoint 完成 16 张 Database／2 张 Query 的 Top-3 子集检查，保存在[历史 smoke 输出](../runs/smoke/robotcar_seasons_20261014_dhn2gh5a/retrieval.json)。这是旧入口的历史记录，不是本次统一 suite 的回归结果，也不是完整数据性能实验。
