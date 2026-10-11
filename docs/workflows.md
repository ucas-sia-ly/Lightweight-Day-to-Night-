# 实验工作流与结果说明

A2B-LightVPR 面向夜间巡检、配送和自主驾驶机器人的视觉定位：在白天地图和有限夜间标注下，项目用 QS-Attn 离线生成 fake-night 图像，以 SALAD 的白天检索结构监督轻量 MixVPR，并在连续 RobotCar 序列中提供无需再训练的 Temporal Consensus Reranking (TCR)。部署 checkpoint 只保留 MixVPR 学生，代码内部继续保留 `clustered-mgtr` 命名以兼容现有缓存和结果文件。

完整论文草稿见 [论文草稿](paper_zh.md)。

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

配置中的数据集和第三方权重路径指向当前研究机。迁移环境时修改对应 YAML 中的数据集和第三方权重路径，不要改变 manifest 中的样本对应关系。

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
fish scripts/train/run_main_seed0.fish
~~~

## 论文补充实验

`vpr.random_seed` 控制 Python、NumPy、PyTorch、CUDA、采样器和 DataLoader 的随机状态。
训练代码同时支持 `train_domains=day`、`relation_day_weight=0`、
`relation_queue_size=0`、`relation_mask_self=false` 和
`relation_queue_scope=shared`，用于论文中的严格配对消融。

三种子 512-D 主实验和 seed-0 组件消融由同一个可恢复脚本执行。该脚本会复用现有
QS-Attn 图像与 SALAD cache，跳过完整运行，并从 `checkpoints/latest.pt` 恢复中断运行：

~~~bash
fish scripts/train/run_paper_experiments_512.fish 2>&1 | tee runs/paper_experiments_512.log
~~~

完成后，多种子均值、样本标准差和配对差值写入
`runs/comparisons/cdnd_multiseed_ablation_summary.json`。汇总器不计算或输出统计显著性。

SVOX 的 yaw 字段可能包含相机方向标记 B/F。该字段不参与 VPR 训练；加载器将
非数值可选位姿解析为 NaN，audit-data 会报告对应数量。

SALAD-B 教师到 SALAD-S 学生的严格配对实验：

~~~bash
fish scripts/experiments/run_salad_family_experiment.fish 2>&1 | tee runs/salad_family_experiment.log
~~~

DINOv2-S 小模型扩展不修改第三方仓库。BoQ-S 使用现有官方 BoQ-DINOv2-B
权重作为冻结教师，同时保留 BoQ 的 Multi-Similarity retrieval loss，并以
scaled descriptor MSE 进行同维度蒸馏。学生使用 DINOv2-S/14-Reg4，不修改
BoQ 第三方聚合器。训练完成后再在 GSV-Cities 上拟合 BoQ-S PCA-512，
并分别评测 PCA 前后的描述子：

~~~bash
fish scripts/references/run_boq_small_experiment.fish 2>&1 | tee runs/boq_small_experiment.log
~~~

BoQ-S 完成后，拟合 AnyLoc-S 的 32-cluster GSV-Cities vocabulary、AnyLoc-S
PCA-512，并补齐 SALAD-S/BoQ-S/AnyLoc-S 的精度、资源和 coverage JSON：

~~~bash
fish scripts/references/run_small_reference_completion.fish 2>&1 | tee runs/small_reference_completion.log
~~~

两个脚本会在检测到其他 VPR 训练或 Small-reference 任务时拒绝启动。
BoQ-S 与 SALAD-S 标记为 `re-trained-and-distilled`，AnyLoc-S 标记为 `re-fitted`；它们都不是
官方发布的 DINOv2-S VPR checkpoint。PCA-512 只减少 descriptor 存储，不减少 encoder 参数。

BoQ-PCA512 拟合、NetVLAD 同协议评测和 reference coverage 审计：

~~~bash
fish scripts/references/run_reference_completion.fish 2>&1 | tee runs/reference_completion.log
~~~

BoQ-PCA512 只压缩描述子和数据库存储，不减少 BoQ 编码器参数或主要前向计算。

论文精度--效率图和 day+fake/CDND 定性检索图使用：

~~~bash
fish scripts/report/run_paper_figures.fish 2>&1 | tee runs/paper_figures.log
~~~

RobotCar Temporal Consensus Reranking (TCR)：

~~~bash
fish scripts/evaluate/run_robotcar_temporal_512.fish
~~~

该流程在同一缓存上评测正式 TCR 的中心窗口 `1/3/5/7/9`。TCR 的主候选使用 top-10，top-100 只用于地点范围补搜；RobotCar TCR 不做 NMS。代码内部参数仍使用 `--method clustered-mgtr`，以保持已有 JSON、缓存和指纹兼容。

TCR 另外报告 map-covered temporal protocol：只有中心前后完整 9 帧都在白天 database 中存在 `10m+30deg` 正样本时，该中心帧才进入指标分母。val 为 `1346/1841` 个中心，night test 为 `1770/1813` 个中心。覆盖标签只用于评测 mask，不传给重排算法；无覆盖路段作为窗口边界，不能把删除后的两段 query 拼接。

脚本会验证已有产物，完成阶段自动跳过；训练中断时从唯一的 checkpoints/latest.pt 恢复。

## 正式配置

主配置位于 `configs/generator/` 和 `configs/vpr/`，参考模型注册表位于
`configs/references/`，扩展实验位于 `configs/experiments/`。
目录与全部入口见 [项目首页](../README.md)。配置里的 `data/`、`runs/` 路径均相对于项目根目录。

排错时可单独执行：

~~~bash
a2b-lightvpr cache-vpr-teacher --config configs/vpr/cdnd_salad_512.yaml

a2b-lightvpr train-vpr --config configs/vpr/cdnd_salad_512.yaml

a2b-lightvpr evaluate-trained-suite \
  --config configs/vpr/cdnd_salad_512.yaml \
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
  --config configs/vpr/cdnd_salad_512.yaml \
  --training-checkpoint runs/vpr/cdnd_512/salad_no_gate_seed0/checkpoints/latest.pt \
  --output runs/vpr/cdnd_512/salad_no_gate_seed0/inference.pt
~~~

## 参考模型

参考模型加载器支持 NetVLAD、CosPlace、EigenPlaces、官方 MixVPR、SALAD、CricaVPR、BoQ 和已有 NocPlace checkpoint；可用注册项以 `configs/references/` 各 YAML 为准。NocPlace 在本项目中仅作预训练参考模型加载，不提供训练入口。`configs/references/small.yaml` 单独注册重新训练/拟合的 DINOv2-S 模型，避免与官方 B/G 权重混淆。

~~~bash
a2b-lightvpr audit-reference-models --config configs/references/official.yaml

a2b-lightvpr evaluate-reference-suite \
  --config configs/references/official.yaml \
  --model boq-dinov2-12288 \
  --database-encoder adapted \
  --transform native \
  --datasets all
~~~

## AnyLoc-S / CricaVPR-S 的 GSV-Cities 对比

独立配置为 `configs/references/anyloc_crica_small.yaml`，不依赖 SALAD/BoQ 产物。
两者使用已有的 `reference_code/salad/dinov2_vits14_pretrain.pth`（无 registers），
训练/拟合和 native 评测分辨率均为 224×224。该配置的通用 `vpr` 分辨率只是项目
配置校验所需的默认值；运行脚本使用 `--transform native`，实际取模型的 `native_size`。
旧 AnyLoc-Reg4 的词典和 PCA 与本次实验不兼容，本次使用独立文件名。

在项目根目录运行（默认 GPU 0，使用当前 Python 环境；可通过 `PYTHON_BIN` 覆盖）：

```bash
bash scripts/references/run_anyloc_crica_small.sh all
```

可将 `all` 换为 `anyloc`（词典和 PCA）、`crica`（训练和 PCA）或 `evaluate`
（两种模型在 PCA 前后的八组评测、资源统计和汇总）。脚本自动将输出追加到
`runs/reference/anyloc_crica_small_<阶段>.log`。可通过 `CUDA_VISIBLE_DEVICES` 和
`PYTHON_BIN` 环境变量选择设备和已有解释器。依赖已包含在 `.[eval,small-reference]` 中。

- AnyLoc：冻结 DINOv2-S，使用 GSV-Cities 的 20,000 张图像、每图 32 个 patch
  拟合 32 个 VLAD 中心；12,288 维描述子再在同一训练采样上拟合 PCA-512。
- CricaVPR：复用官方前向、数据读取、RandAugment 和 MultiSimilarity loss/miner；
  Small 骨干及适配器通道缩半，跨图像模块改为 384 通道，保留 16 heads、2048 维 FFN。
  单 GPU、每批 32 个地点×4 张图、FP32、Adam 1e-4，每 3 轮学习率减半，固定 10 轮。
  只训练适配器及聚合模块，以最后一轮权重作为结果，不使用测试集选择模型。
  这是重新训练的 Small 变体，不是官方 Base checkpoint 或完整官方验证选优流程。
- CricaVPR 原始描述子为 5,376 维；PCA 使用 GSV-Cities 的 20,000 张训练图像。
  该方法存在 batch 依赖，PCA 提取和主表评测均使用 batch=1，保持单图检索协议。
  不应将结果与官方多图 batch 推理直接视为同一协议。

CricaVPR 保存到 `runs/reference/cricavpr-dinov2-s-gsv-5376/model.pt`；
`checkpoints/latest.pt` 保存优化器、调度器及随机状态，重复运行会从完成的 epoch 恢复
（中断所在的未完成 epoch 重跑）。词典/PCA 使用现有元数据校验及缓存恢复机制。
第三方源码文件不作修改。完整过程预计需要数小时，正式运行前为 GPU 留出训练空间。

最终汇总：`runs/comparisons/anyloc_crica_small_summary.json`。
PCA 仅压缩描述子，不减少编码器参数。训练权重、词典和 PCA 拟合完成之前，不能据此报告精度。

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

## CDND on GSV-Cities

`configs/experiments/data_sources/cdnd_gsv_512.yaml` changes the CDND student training source to GSV-Cities.
It retains the existing QS-Attn 150k generator, SALAD teacher, official MixVPR-512
initialization and 20 x 60,000 sampling budget. It is **not** a generator trained only
on GSV-Cities, nor a student initialized from scratch. GSV images with at least four
views per place are used; labels include the city name. Each epoch is a sampling
budget rather than a full pass over the dataset. Validation/checkpoint selection
uses the existing CDND trainer's protocol.

Run from the project root:

```fish
mkdir -p runs/vpr/cdnd_gsv_512
fish --no-config scripts/experiments/run_cdnd_gsv.fish 2>&1 | tee runs/vpr/cdnd_gsv_512/full.log
```

The script prepares the manifest, generates paired fake-night images, caches the
teacher, trains and evaluates four night and five day protocols. GSV manifest,
generated images, teacher cache and student checkpoints use separate paths. An
existing latest training checkpoint is resumed. The large place classifier is
placed on CPU to reduce GPU memory use. Full generation over roughly 530k images
can take substantial time and disk space; no full job is run by configuration setup.

Output: `runs/vpr/cdnd_gsv_512/salad_seed0/final_suite/summary.json`.

## Nine-method single-frame latency

```fish
fish --no-config scripts/benchmark/run_nine_method_latency.fish
```

Measures CosPlace, MixVPR, EigenPlaces, the accepted NocPlace clean5k reproduction,
AnyLoc-PCA, GSV-trained CricaVPR/SALAD/BoQ, and the original CDND seed0 checkpoint.
Uses the verified qualitative model mapping (including NocPlace 320x320 fit-and-pad).
Batch=1, FP16 autocast, 20 warmups, 100 measurements, synchronized GPU timing.
PCA projection is included in descriptor extraction. Input sizes are recorded per
model; this is not a common-resolution comparison.

Three separately measured stages are reported: GPU-resident input to descriptor;
decoded CPU image to descriptor including preprocessing/transfer; FP32 cosine
similarity and top-1 against a normalized synthetic GPU gallery (default 10,000).
Disk I/O, image decoding and gallery feature extraction are excluded. The stacked
plot sums mean encoder and search times; it is not an independently measured
end-to-end latency or a whole-dataset time. A synthetic gallery only benchmarks
computation, not recall. Set `--gallery-size 100000` to change gallery size or
`--image /absolute/path/to/image.jpg` to benchmark a shared real input image.

Output: `runs/benchmarks/nine_methods_single_frame/results.csv`, `results.json`,
`latency.png`, `latency.pdf`. The JSON retains all timing samples and weight hashes.
`runs/benchmarks/nine_methods_smoke/` is a pipeline check only, not reportable timing.

## RobotCar-only adaptation ablation

The `configs/experiments/single_dataset/robotcar_only_{day_only,day_fake,cdnd}.yaml` configurations share official
MixVPR-512 initialization, 320x320 input, seed 0, P=4/K=2, gradient accumulation 2,
one classifier warmup epoch, and 20 epochs of 8,944 source-image samples. This is a
sampling budget, not guaranteed one-pass coverage. Early stopping and first-epoch
regression abort are disabled. Best weights are selected on RobotCar validation
night R@1, including the epoch-zero model if training never improves it.

QS-Attn is newly trained on RobotCar train only (10,543 day, 1,806 night images),
30,000 steps with batch 4 and the existing objective. It generates fake-night images
for 8,947 VPR day images / 1,464 places. Day-only uses classification on day images;
Day+fake adds paired fake images; CDND additionally uses the frozen SALAD teacher.
Official pretrained student and teacher are allowed: this is not RobotCar-only
pretraining. Only RobotCar night and day test protocols are evaluated, with R@1/5/10.
The original official student is also evaluated with the same fit-and-pad transform.

```fish
fish --no-config scripts/experiments/run_robotcar_only.fish all
```

Stages may be run separately with `prepare`, `generator`, `translate`, `teacher`,
`train`, `evaluate`, `summarize`, in that order. Run only one copy at a time. Logs are
written under `runs/vpr/robotcar_only_512/logs/`. Existing generator/student latest
checkpoints are resumed; translations and teacher caches use their existing
fingerprint checks. Changing settings requires a separate output namespace.

Preparation writes `data/manifests_robotcar_only/audit.json`, checks all selected
images exist and belong to the proper train domain, and checks disjoint traverses
and resolved source paths across splits. It preserves VPR place labels. Same roads
may occur across traversals, so the experiment measures cross-traversal adaptation,
not unseen-city generalization. Limited independent night scenes can overfit.

All outputs are separate from previous runs: generator in
`runs/generator/robotcar_only_qsattn_seed0`, generated images in
`data/generated/robotcar_only_qsattn30k`, teacher cache in
`data/teacher_cache/robotcar_only_salad_fp16.npy`, and students in
`runs/vpr/robotcar_only_512/{day_only,day_fake,cdnd}_seed0`.
Final tables: `runs/vpr/robotcar_only_512/ablation_summary.{json,csv}`.
JSON includes valid queries, fingerprints, checkpoints, and percentage-point deltas
Day+fake minus Day-only and CDND minus Day+fake. Do not equate comparisons against
old multidataset runs with a controlled data-source ablation: their budgets differ.

Short validation outputs belong under `runs/smoke/`; regenerate them when needed.
Full training and evaluation are long-running user-launched jobs.

## SVOX-only / MSLS-only adaptation

Run `fish --no-config scripts/experiments/run_single_dataset_cdnd.fish svox` or replace `svox`
with `msls`; omit the argument to run both sequentially. Reuses the completed
30k-step corresponding QS-Attn checkpoint and its own generated images; preparation
validates generator hashes, dataset/source/place identity, uniqueness and files.
It never sends one dataset through the other dataset's generator.

Each experiment runs Day-only, Day+fake, CDND from official MixVPR-512 initialization
at 320x320, seed0, P4/K2, accumulation2, 20 epochs (including classifier warmup).
Source sampling quotas: SVOX 33520/epoch; MSLS 75840/epoch. QS-Attn, teacher cache,
student training data are restricted to the named dataset. SALAD remains an official
pretrained teacher. SVOX has 33526 images / 9436 places; MSLS 75840 / 18960.

Unlike the RobotCar validation-selected experiment, these new experiments use
`selection_rule: fixed_budget_last`: no RobotCar validation or final-evaluation-set
selection. `model_best.pt` is retained as a filename for compatibility but contains
the last completed epoch; `val_recall1=-1` means not measured. If stopped early it is
only a partial-budget model. Evaluate SVOX night/day/sun or MSLS night/day-val only.
The MSLS outputs are benchmark validation-split results, not an independent test set.

Separate outputs: `runs/vpr/{svox,msls}_only_512/`; logs and three-arm JSON/CSV
summaries live there. Repeating the command resumes latest training checkpoints.
Existing RobotCar/multigenerator/GSV experiments and source generated images remain
in place. Full teacher caching, training and evaluation must be user-launched.

## DINOv2-S + MixVPR student

`fish --no-config scripts/experiments/run_dino_mix_cdnd.fish` runs an independent Day+fake
control followed by CDND under `runs/vpr/dino_mix_small_512/`. It reuses the original
three-dataset QS-Attn translations and SALAD teacher cache, not the single-dataset
or multi-generator variants. Both students share the same seed and initialization.

Architecture follows DINO-Mix's patch-token-to-grid approach: plain DINOv2-S/14,
322x322 -> 384x23x23 normalized patch features (CLS excluded), existing four-layer
MixVPR with 256 output channels x 2 rows -> L2-normalized 512D. Total parameters:
24,403,388. Last two DINO blocks and final norm are trainable; the head is randomly
initialized. This is a custom CDND student, not an official DINO-Mix reproduction;
the reference implementation removes final normalization, while this uses the
existing normalized DINO backbone adapter. Third-party source is unchanged.

Official CNN-MixVPR weights are incompatible and are not loaded. New-head learning
rate is 1e-4, backbone LR 1e-5; classifier-only warmup and epoch-zero evaluation are
disabled. Other CDND loss/sampling settings remain those of the original 20-epoch,
60k-samples/epoch experiment. The paired control is necessary because both backbone
and initialization change. Test all four night and five day protocols and benchmark
both final validation-selected checkpoints. Existing outputs remain intact.
