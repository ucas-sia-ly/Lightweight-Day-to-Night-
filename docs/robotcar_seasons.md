# RobotCar Seasons 公共测试

数据位于 `/home/admin123/github/datasets/RobotCar-Seasons`。公共 Seasons 与 MSLS、SVOX、Tokyo 使用相同 `evaluate-trained-suite`／`evaluate-reference-suite` 入口；测试固定为官方 **v2 test／rear 相机**。

| 协议 key | Query | Day Database |
| --- | ---: | ---: |
| `robotcar_seasons_night` | 429 张 night／night-rain | 6,954 张 |
| `robotcar_seasons_day` | 1,443 张其他条件 | 6,954 张 |

公开 train 图像不作为 test Query。固定协议不因本地三相机图片总数或审计筛选而变化。官方 test Query 位姿隐藏，结果保存 Top-K 排名，`ground_truth_available: false`，Recall、MRR 和有效 Query 数为 `null`；未知真值不能记作零 Recall，也不能参与监督指标平均。

## 运行

```bash
a2b-lightvpr audit-robotcar-seasons --output runs/resource_audit/robotcar_seasons_v2.json

a2b-lightvpr evaluate-trained-suite \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --checkpoint runs/vpr/salad_small_gsv_metric/day_aug_seed0_lowmem/model_recovered_epoch2.pt \
  --datasets robotcar_seasons_night robotcar_seasons_day \
  --set vpr.output_dir=runs/evaluation/seasons_day_night
```

迁移到其他机器可覆盖 `data.robotcar_seasons_root`。审计记录划分文件 SHA、图片清单指纹和缺真值状态；文件就绪不等于完成逐张解码或模型性能验证。

suite 与其他数据集一样，在 `vpr.output_dir/final_suite/` 保存 `summary.json` 与各协议的 `metrics.json`。`retrieval_predictions` 记录每张 Query 的 Top-K Database 身份、路径和相似度；K 为 `evaluation.recalls` 的最大值，当前是 10。输出带有 checkpoint、预处理和协议指纹。参考模型使用相同 key，例如：

```bash
a2b-lightvpr evaluate-reference-suite \
  --config configs/references/salad.yaml \
  --model salad-dinov2-s-544-gsv \
  --datasets robotcar_seasons_night
```

参考模型满足 registry 中的权重、仓库和预处理要求后才能运行。

## Fake Night Database

```bash
a2b-lightvpr evaluate-trained-suite \
  --config configs/experiments/metric/salad_small_gsv_day_aug_544.yaml \
  --checkpoint runs/vpr/salad_small_gsv_metric/day_aug_seed0_lowmem/model_recovered_epoch2.pt \
  --datasets robotcar_seasons_night \
  --set evaluation.database_domain=fake_night \
  --set evaluation.fake_database_manifest=data/evaluation/fake_database.csv \
  --set vpr.output_dir=runs/evaluation/seasons_fake_night
```

示例 CSV 需自行准备，字段为 `source_path,generated_path`，可选 `split=test`。相对图像路径从 CSV 所在目录解析；映射必须完整覆盖所选 Day Database，生成图须存在，每个源图与生成图身份唯一。冲突、缺失或与真实 Query／Database 文件重合的生成路径会报错。

Query 保持真实图像，Database 源身份、顺序和地点元数据保持原协议。结果写入 `final_suite/fake_night/`，保存映射 hash；应另记录生成器与权重身份。训练生成图不能替代测试 Database。替换 Database 不会补出 Seasons 的隐藏真值。

## 当前证据

已有评测输出全部 429 张夜间 Query 的 Top-10，权重身份和当前重测状态见 [实验结果](experiment_results.md)。这些排名可用于人工检查，但不能给出本地可信 Recall，也不等同于[官方视觉定位评测](https://www.visuallocalization.net/benchmark/)需要的 6DoF 位姿。
