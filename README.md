# A2B-LightVPR

A2B-LightVPR 面向夜间巡检、配送和自主驾驶机器人的视觉定位：在白天地图和有限夜间标注下，项目用 QS-Attn 离线生成 fake-night 图像，以 SALAD 的白天检索结构监督轻量 MixVPR，并在连续 RobotCar 序列中提供无需再训练的 Temporal Consensus Reranking (TCR)。部署 checkpoint 只保留 MixVPR 学生，代码内部继续保留 `clustered-mgtr` 命名以兼容现有缓存和结果文件。

完整论文草稿见 [PAPER_ZH.md](PAPER_ZH.md)。

## 方法概览

~~~text
day -- QS-Attn --> fake-night
 |                    |
 +-- SALAD cache      +-- MixVPR(fake)
 +-- MixVPR(day)      |
          +---- CDND -+
                 |
        inference-only MixVPR
                 |
        optional Temporal Consensus Reranking (TCR)
~~~

核心训练目标为：

~~~text
L = L_day+fake + 0.5 * (L_fake->day + 0.5 * L_day->day)
~~~

SALAD、teacher cache、地点分类器和 FIFO 队列只存在于训练阶段。推理 checkpoint 只包含 MixVPR 学生。

## 当前结果

以下均为固定随机状态 0 的 development 结果，不表示跨重复训练的均值或统计稳定性。

| 描述子维度 | 方法 | 夜间宏平均 R@1 | 白天宏平均 R@1 | CDND 夜间增益 |
|---:|---|---:|---:|---:|
| 128 | day+fake | 52.40% | 81.20% | - |
| 128 | SALAD-CDND | 55.82% | 82.92% | +3.42pp |
| 512 | day+fake | 73.65% | 88.55% | - |
| 512 | SALAD-CDND | 76.62% | 89.03% | +2.97pp |
| 4096 | day+fake | 81.76% | 91.66% | - |
| 4096 | SALAD-CDND | 82.99% | 91.87% | +1.23pp |

512 维是论文主要轻量配置。128 维用于观察容量瓶颈，4096 维用于观察精度上限和描述子存储代价。

RobotCar 的 10m+30deg R@1 从单帧 88.24% 提升至正式 TCR 的 93.59%。SALAD 的同协议单帧结果为 92.08%。TCR 使用连续顺序和有限未来帧，而 SALAD 没有应用相同的时序重排，因此这是系统级比较，不是同条件算法比较。

Tokyo night 的 512-D CDND 结果为 63.12% R@1，相比配对 baseline 的 60.28% 提升 2.84 个百分点。仓库当前可复现产物不支持将该结果写成 10.6pp；10.91pp 是仅有 55 个有效查询的 MSLS night-val 结果，不能替代 Tokyo night 主结果。

## 环境

建议使用 Python 3.10+、PyTorch 2.1+、CUDA GPU 和 fish shell。

~~~bash
cd /home/admin123/diffusion_vpr_test/A2B-LightVPR
python -m pip install -e '.[test,eval]'
~~~

时序重排不需要额外局部特征模型，只复用 MixVPR top-100 缓存。

## 正式数据

~~~text
data/manifests_clean_night/
├── generator_train.csv
└── vpr_train.csv

data/generated/multidataset_qsattn_official_150k/
├── images/
├── manifest.csv
└── translation_metadata.json

data/teacher_cache/
├── salad_qsattn_official_150k_fp16.npy
├── salad_qsattn_official_150k_fp16.ids.json
└── salad_qsattn_official_150k_fp16.meta.json
~~~

配置中的数据集和第三方权重路径指向当前研究机。迁移环境时修改保留的 YAML 和两个时序评测脚本，不要改变 manifest 中的样本对应关系。

## 初始化权重

| 权重 | SHA256 |
|---|---|
| MixVPR-R50-128，64 x 2 head | d9dc62449ebfbfca0ac138e3062469458f957c5dd8d61157fe07007d1da54b6f |
| MixVPR-R50-512，256 x 2 head | 7e1180f77d04ece604f2d12f8b6b9b323a99d6e43863c879028bfa17b34807cf |
| MixVPR-R50-4096，1024 x 4 head | 97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24 |
| SALAD teacher | d25c3434032a3ce5680ffcf51e05de0a7b9962772bd9112d88ca074689f18dcc |

错误版本必须直接拒绝，不能宽松加载。

## 工作流

完整训练和开发集评测：

~~~bash
fish scripts/run_seed0_all.fish
~~~

RobotCar Temporal Consensus Reranking (TCR)：

~~~bash
fish scripts/run_robotcar_temporal_512.fish
~~~

该流程在同一缓存上评测正式 TCR 的中心窗口 `1/3/5/7/9`。TCR 的主候选使用 top-10，top-100 只用于地点范围补搜；RobotCar TCR 不做 NMS。代码内部参数仍使用 `--method clustered-mgtr`，以保持已有 JSON、缓存和指纹兼容。

TCR 另外报告 map-covered temporal protocol：只有中心前后完整 9 帧都在白天 database 中存在 `10m+30deg` 正样本时，该中心帧才进入指标分母。val 为 `1346/1841` 个中心，night test 为 `1770/1813` 个中心。覆盖标签只用于评测 mask，不传给重排算法；无覆盖路段作为窗口边界，不能把删除后的两段 query 拼接。

脚本会验证已有产物，完成阶段自动跳过；训练中断时从唯一的 checkpoints/latest.pt 恢复。

## 正式配置

~~~text
configs/
├── generator_qsattn_official_objective_150k.yaml
├── vpr_day_fake_pk_128.yaml
├── vpr_cdnd_salad_128.yaml
├── vpr_day_fake_pk_512.yaml
├── vpr_cdnd_salad_512.yaml
├── vpr_day_fake_pk_4096.yaml
├── vpr_cdnd_salad_4096.yaml
└── reference_models_extended.yaml
~~~

排错时可单独执行：

~~~bash
a2b-lightvpr cache-vpr-teacher --config configs/vpr_cdnd_salad_512.yaml

a2b-lightvpr train-vpr --config configs/vpr_cdnd_salad_512.yaml

a2b-lightvpr evaluate-trained-suite \
  --config configs/vpr_cdnd_salad_512.yaml \
  --checkpoint runs/vpr/cdnd_512/salad_no_gate_seed0/model_best.pt \
  --datasets all
~~~

查看全部保留接口：

~~~bash
a2b-lightvpr --help
~~~

## Checkpoint 边界

checkpoints/latest.pt 是训练恢复状态，包含学生、分类器、优化器、调度器、AMP scaler、采样器游标和全部 RNG 状态；CDND 还包含 FIFO 队列与 teacher cache 指纹。

model_best.pt 是推理状态，只包含学生权重、描述子维度、输入信息和必要指纹。显式转换：

~~~bash
a2b-lightvpr export-vpr-inference \
  --config configs/vpr_cdnd_salad_512.yaml \
  --training-checkpoint runs/vpr/cdnd_512/salad_no_gate_seed0/checkpoints/latest.pt \
  --output runs/vpr/cdnd_512/salad_no_gate_seed0/inference.pt
~~~

## 参考模型

configs/reference_models_extended.yaml 注册 NetVLAD、CosPlace、EigenPlaces、官方 MixVPR、SALAD、CricaVPR、BoQ 和已有 NocPlace checkpoint。NocPlace 在本项目中仅作预训练参考模型加载，不提供训练入口。

~~~bash
a2b-lightvpr audit-reference-models --config configs/reference_models_extended.yaml

a2b-lightvpr evaluate-reference-suite \
  --config configs/reference_models_extended.yaml \
  --model salad-dinov2-544 \
  --database-encoder adapted \
  --transform native \
  --datasets all
~~~

## 正式结果目录

~~~text
runs/generator/qsattn_official_objective_150k_seed0/
runs/vpr/cdnd_128/{day_fake_pk,salad_no_gate}_seed0/
runs/vpr/cdnd_512/{day_fake_pk,salad_no_gate}_seed0/
runs/vpr/cdnd_4096/{day_fake_pk,salad_no_gate}_seed0/
runs/reference/
runs/comparisons/cdnd_seed0_summary.json
~~~

## 常见问题

- SHA256 mismatch：权重版本错误。
- manifest fingerprint mismatch：fake-night、源 manifest 与 teacher cache 不属于同一数据版本。
- descriptor dimension mismatch：128、512 和 4096 checkpoint 不可混用。
- relation queue state missing：baseline 训练状态不能恢复 CDND。
- 显存不足：停止其他 GPU 任务；不要改变正式 batch 或样本预算后混入主表。
- 只有 model_best.pt：可以推理，但不能恢复优化器和采样顺序。
- RobotCar temporal cache mismatch：确认 checkpoint、manifest、描述子维度和 val/night/day 条件与缓存元数据一致。

## 结论边界

当前证据支持：CDND 在三种学生容量的一次固定状态实验中均提高夜间宏平均，教师不增加部署模型；TCR 在 RobotCar 开发协议上改善 512 维系统。TCR 的 93.59% 高于 SALAD 单帧的 92.08%，但两者使用的信息不同。

当前证据不支持：跨随机初始化稳定性、统计显著性、所有数据集上的普遍提升、纯视觉时序定位、时序方法跨数据集泛化，或 TCR 与未使用同一重排器的 DINOv2 模型之间的算法公平比较。BoQ 等模型保留为扩展参考；本文不声称达到 SOTA。
