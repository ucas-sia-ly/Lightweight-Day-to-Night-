# A2B-LightVPR

A2B-LightVPR 研究昼间训练数据充足、真实夜间地点标注有限时的视觉地点识别。项目使用 QS-Attn 离线生成 fake-night 图像，以 SALAD 的白天检索结构监督 MixVPR，并在连续序列场景中提供 MGTR v1 与高精度 RA-GTR 时序重排。

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
        optional MGTR v1 / RA-GTR
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

RobotCar 的 10m+30deg R@1 从单帧 88.24% 提升至 MGTR v1 的 91.92%，CPU 重排 P95 约 0.09 ms。Gardens Point 的序列 exact R@1 在固定窗口 4 下从 28.5% 提升至 34.0%。Gardens Point 使用帧序号容差，不是米制定位。

## 环境

建议使用 Python 3.10+、PyTorch 2.1+、CUDA GPU 和 fish shell。

~~~bash
cd /home/admin123/diffusion_vpr_test/A2B-LightVPR
python -m pip install -e '.[test,eval]'
~~~

RA-GTR 还需要冻结的 DISK/LightGlue 局部匹配器和 MAGSAC：

~~~bash
python -m pip install -e '.[test,eval,ragtr]'
~~~

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

RobotCar MGTR v1：

~~~bash
fish scripts/run_robotcar_temporal_512.fish
~~~

Gardens Point 与参考模型：

~~~bash
fish scripts/run_gardens_vpr_models.fish
~~~

4096-CDND 高精度 RA-GTR：

~~~bash
fish scripts/run_robotcar_ragtr_4096.fish
~~~

该脚本先在 RobotCar val 上选择 DISK/LightGlue 几何融合和窗口 3/4/5/6 的有向路线参数。只有 val 未达到预设 SALAD 参考阈值时，才使用 RobotCar train 的 day/fake 做 5 x 60,000 samples 的 route-contrastive 微调。最终 checkpoint 冻结后，night test 只运行一次。RA-GTR 固定延迟两帧，不调用 SALAD 或 BoQ 推理。

RA-GTR 的正式 4096 结果尚未运行，因此当前结果表仍只报告已经完成的 MGTR v1。此前 512 缓存上的路线进度诊断不能替代正式 4096 val 选参与 test 结果。

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
├── vpr_cdnd_route_4096.yaml
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
runs/vpr/cdnd_4096/ragtr_final/
runs/external/gardens_point/
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
- Gardens 指标异常：确认三条序列各 200 张且排序未改变。
- RA-GTR 缺少 Kornia/OpenCV：安装 `.[ragtr]`，并确认 OpenCV 暴露 `USAC_MAGSAC`。
- RA-GTR cache mismatch：不要混用 base、route-finetuned、val、night 或 day 的全局/几何缓存。

## 结论边界

当前证据支持：CDND 在三种学生容量的一次固定状态实验中均提高夜间宏平均，教师不增加部署模型；MGTR v1 在当前 RobotCar 与 Gardens Point 协议上改善 512 维系统。RA-GTR 已实现但尚未形成可写入论文结果表的正式 4096 评测。

当前证据不支持：跨随机初始化稳定性、统计显著性、所有数据集上的普遍提升、纯视觉时序定位，或 MGTR 与未使用同一重排器的 DINOv2 模型之间的算法公平比较。
