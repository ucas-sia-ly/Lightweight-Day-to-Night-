# StableNight-SALAD：第一阶段研究原型

StableNight-SALAD 使用已有 DINOv2-S/14 和 SALAD-S 的 544D 描述子接口，新增轻量 patch 可靠性预测与动态 Dustbin。当前实现用于开展实验，**没有完成真实夜间数据集上的性能验证，也不代表已经证明优于 SALAD-S、MixVPR 或 CDND**。第一阶段没有引入真实夜间训练数据；真实夜间 Query 的提升仍需独立评测。

推荐先运行 `stable_night_salad_day_aug_544.yaml`，只使用源白天图像和非生成式光照增强，关闭局部教师损失。之后分别比较生成式增强、局部监督和 Dustbin 开关，避免同时加入多个变化后无法判断收益来源。

## 已有实现与兼容边界

| 组件 | 已有行为 | 新原型的使用方式 |
| --- | --- | --- |
| `vpr/salad.py` | `DINOv2SALADBackbone` 调用 `forward_features`，读取归一化的 patch token 与 CLS token；322×322 输入产生 23×23 个 patch | 继续使用该 backbone；新增独立 `ReliableSALADHead` 和模型类 |
| `CompactSALADHead` | CLS 投影为 32D；16 个 cluster 各聚合 32D，最终为 `32 + 16×32 = 544`，L2 归一化 | 原始 head 保留；新 head 沿用投影和 OT 聚合结构 |
| 原始 SALAD OT | log-space Sinkhorn 默认 3 次迭代；全局标量 Dustbin；固定行列边缘分布 | OT 支持标量及逐 patch Dustbin；新 head 默认 20 次、FP32 Sinkhorn |
| `vpr/model.py` | 根据 `model_family` 构造描述子模型 | 增加 `stable_night_salad` 分支，默认推理依旧返回 `[B,544]` Tensor |
| `vpr/losses.py` / `trainer.py` | margin cosine 分类训练；可选 CDND 邻域蒸馏 | 新 family 使用独立的多正样本地点损失，旧分类和 CDND 分支保留 |
| `teachers.py` / `teacher_cache.py` | CDND 使用冻结的 DINOv2-B SALAD 教师和离线全局描述子缓存；缓存校验 manifest、checkpoint、图像 ID、顺序和完成状态 | 原型不依赖该缓存；可选局部参考使用同一预训练 DINOv2-S 的冻结副本，训练时才加载 |
| `data.py` / `transforms.py` | 原始 Day/Fake 数据逐图独立 fit/pad；生成器训练包含裁剪和翻转，生成推理缩放后向下取整到 4 的倍数 | 原始数据和变换路径保留；新 family 使用单独的数据集和保持几何一致的增强路径 |
| 原 SALAD 配置 | `salad` 要求 register 版本 S、544D、1–12 个可训练 block；两个既有实验均为 Day/Fake | 新 family 允许 plain/register DINOv2-S，以及 0–12 个可训练 block；不改变旧约束 |

`backbone_trainable_blocks: 0` 冻结完整 backbone，只训练聚合器，可使用可靠性监督或均值下界正则。由于 `L_stable` 约束原始 backbone 局部特征，启用它时至少解冻一个 block；完全冻结并设置 `stable_stability_weight > 0` 会被配置校验拒绝，避免配置一个无法更新任何参数的损失。设为 `k > 0` 时使用已有策略，仅解冻最后 `k` 个 Transformer block 和最终 norm。`backbone_repo` 与 `backbone_checkpoint` 仍指向本地 DINOv2 repo 和匹配的预训练权重，使用严格加载。第一阶段不使用 DINOv2-B，也不增加大型推理编码器。

可选 `vpr.init_checkpoint` 支持从匹配的 Compact SALAD-S 初始化 backbone、CLS 投影、局部投影、cluster 分配和基础 Dustbin。只允许新增 predictor、gamma 参数/上界及 head 状态缺失；旧组件缺失、额外键或 shape 不匹配会失败，不能把任意教师 checkpoint 当作兼容的 SALAD-S 权重。

新配置要求 `model_family: stable_night_salad`、`aggregator: salad`、`descriptor_dim: 544`、322×322 输入、`cross_domain_aux: none`，并关闭分类器 warmup。原始 MixVPR、SALAD 和 CDND 实验的模型 family、loss 与配置含义不变。

## ReliableSALADHead 与 Dustbin 的实际含义

输入为 patch 特征 `F[B,C,H,W]` 和 CLS token。轻量 predictor 输出 `reliability[B,H×W]`，sigmoid 将其限制在 `[0,1]`。Dustbin 分数为：

```text
gamma = gamma_max * sigmoid(gamma_raw)
dustbin_logit[b,p] = base_dustbin + gamma * (1 - reliability[b,p])
```

默认 `gamma_init=1`、`gamma_max=4`。这种有界参数化保证 gamma 非负，并限制 Dustbin logits 的额外幅度。`stable_dustbin_enabled: false` 使用原有标量形式，供独立消融。OT 接受标量、`[B,P]` 或 `[B,1,P]` Dustbin。

384 通道时，原 head 为 632,401 个参数，新 head 为 645,523 个参数，增加 13,122 个；局部参考只在可选训练损失开启时加载。这个参数增量不等于实际延迟增量，尤其新路径默认增加了 Sinkhorn 迭代数，仍需通过同硬件 benchmark 测量。

Dustbin 开关、gamma 上界和 Sinkhorn 迭代数会保存在 head state 中，训练导出和推理加载保持相同聚合行为。加载时如果配置与 checkpoint 保存的这些设置不一致会报错；不能在评测阶段改开关或迭代数后把结果继续当作原模型。消融需要分别训练并使用对应配置。

SALAD 的边缘分布固定了 Dustbin 总容量。对于 `N` 个 patch、`K` 个 cluster，收敛后的每个 cluster 质量为 1，Dustbin 质量为 `N-K`；322×322 时为 `N=529`、`K=16`。因此动态 Dustbin **改变哪些 patch 进入 cluster，不直接改变丢弃多少 patch**。单纯增大所有 patch 的全局 Dustbin 分数，在完全收敛的固定容量 OT 中会被行归一化抵消。不能把整体 Dustbin assignment 很高当作模型塌缩的证据。

Sinkhorn 使用 logsumexp，并在新 head 中以 FP32 执行，兼容 AMP。有限迭代并不保证精确满足行边缘分布；因此辅助输出包含 `cluster_mass`，用于观察数值收敛。聚合时对每个 cluster 的 log assignment 先归一化，再加权局部特征，避免低质量或极端 logits 使聚合特征仅因数值下溢而消失。固定 cluster 容量、地点判别损失、gamma 上界和可靠性均值下界共同限制退化；它们不能替代检索验证。

正常 `forward` / `encode` 只返回 L2 归一化的 544D 描述子。训练和诊断可调用 `encode_with_aux(images)`，返回 `(descriptor, aux)`：

| 辅助字段 | 形状 | 用途 |
| --- | --- | --- |
| `reliability` | `[B,P]` | 可靠性监督与分布诊断 |
| `dustbin_assignment` | `[B,P]` | patch 的 Dustbin 分配诊断 |
| `cluster_mass` | `[B,16]` | 观察有限迭代 OT 的行质量误差 |
| `local_features` | `[B,C,H,W]` | 可选 Day/Aug 局部一致性训练 |

## 为什么第一阶段不监督 Day/Fake 的同位置 patch

现有生成推理的 `resize_to_fit` 将尺寸向下取整到 4 的倍数，随后 Day 与 Fake 分别经过 VPR fit/pad。即使生成器没有显式改变结构，二者也未必对应同一个 DINO patch：

| 源图尺寸（宽×高） | 生成图尺寸 | Day 的 VPR 内容区 | Fake 的 VPR 内容区 |
| --- | --- | --- | --- |
| MSLS 示例 640×360 | 480×268 | 322×181，top=70 | 322×180，top=71 |
| SVOX 示例 512×384 | 424×320 | 322×242，top=40 | 322×243，top=39 |

这些是依据代码尺寸规则计算的示例，不是对全部图像完成了逐图配准。Generator 训练中另有随机 crop/flip，不能据此推断输出与源图具有严格对应关系。尺寸刚好相同也不能证明生成器保留了全部几何。现有全局教师缓存还使用方形 resize，与 VPR fit/pad 的局部坐标不同。

第一阶段只对**同一白天源图共享几何变换后产生的 Day/Aug**计算逐位置损失，绝不对 Fake 套用同位置局部监督。若后续需要 Day/Fake 局部监督，必须先保存生成过程的缩放/裁剪/padding 元数据并建立可验证的坐标映射，或使用外部配准，并对无效和失败区域显式屏蔽。

非生成式增强只改变曝光、亮度、对比度、色温、轻量噪声和模糊，不做破坏地点结构的 crop、warp 或遮挡。Day 与 Aug 共享 fit/pad；完全落在图像有效内容区内的 14×14 patch 才进入局部监督，padding 和跨越内容边界的 patch 不参与。

当前增强范围为曝光 −0.8～+0.5 stops，亮度倍率 0.8～1.15，对比度倍率 0.8～1.2，红/蓝色温增益 `1±w`（`w∈[-0.12,0.12]`），RGB `[0,1]` 空间中的高斯噪声标准差 0～0.02；以 0.25 概率使用 3×3 高斯模糊，sigma 为 0.1～0.8。增强在拟合后的图像内容区执行，padding 精确恢复。范围偏温和，仍需通过真实夜间验证判断是否覆盖足够的光照变化。

StableNight 数据集以配置 seed、规范化源图身份和 epoch 为每张图建立独立增强 RNG，不消耗模型或 worker 的全局 RNG。共享 epoch 状态会同步到 persistent workers；同一源图在同一 epoch 重复采样时使用相同 Aug，跨 epoch 改变。恢复训练会重建相同源图增强序列；这不代表不同 GPU/PyTorch 版本下的浮点计算完全确定。

## 四种训练模式与地点损失

| `train_domains` | 每个源图使用的 view | 是否需要生成图 |
| --- | --- | --- |
| `day_only` | Day | 否 |
| `day_aug` | Day + 非生成式 Aug | 否 |
| `day_fake` | Day + Fake | 是 |
| `day_aug_fake` | Day + Aug + Fake | 是 |

`day_only` / `day_aug` 从 `data.vpr_source_manifest` 读取源数据，训练不读取生成图、不加载 QS-Attn、不依赖生成 manifest 或 CDND teacher cache。`day_fake` / `day_aug_fake` 则读取 `data.generated_root/manifest.csv` 中的原始 `source_path` 与已有 `generated_path`，不自动运行翻译，也不与 source manifest 隐式合并。四个配置具有相同预算，实际两类 manifest 是否覆盖同一个源图集合需要运行前单独核对。

训练 manifest 必须提供非空 `source_path,place_id,image_id,split`；`split` 为 `train`，`dataset` 缺失时按 `unknown` 处理，通常应明确提供以匹配配置的数据集筛选。Fake 模式还需要 `generated_path`，可选提供 `generator_sha256`。训练路径依照已有数据集约定相对于运行目录解析，建议使用绝对路径。数据集在地点过滤前按 `(dataset,规范化源图路径)` 去重，重复源图只计一次；同源图的不同生成器输出使用第一条兼容记录。冲突的 place/split、同一生成图对应多个源图、同一生成图的矛盾 hash 会被拒绝。

训练只接受 train 数据；place ID 添加 dataset namespace，避免不同数据集的标签冲突。同一地点需要至少两张不同源图，PK batch 默认 `P=4,K=2`，源图 batch size 为 8。地点对比损失把**同地点且不同源图**的全部 view 当作正样本，将同一源图的全部 view 从候选分母中排除，包括 Day/Aug/Fake 和重复记录。这样同源图配对不能单独满足地点学习目标。没有有效正样本或负样本的 anchor 不参与损失，并输出有效 anchor 计数。

这里的源图身份同样带 dataset namespace：同一 dataset 内的路径别名和 symlink 解析为同一个源图，不跨 dataset 合并身份。PK sampler 每个 batch 来自单一 dataset；如果批次完全没有有效地点 anchor，训练会报错，避免只进行辅助正则却误报地点训练正在运行。

```text
L = stable_place_weight * L_place
  + stable_stability_weight * L_stable
  + stable_reliability_weight * L_reliability
```

`L_place` 对每个 anchor 的全部有效正样本取平均 log-probability，温度默认 0.07。各模式在相同源图 batch 下分别产生 8、16、16、24 个 view；梯度累积不会扩大一个 contrastive batch 的候选集合。比较训练开销时必须同时报告源图数量、view 数量和 optimizer step 数。

`stable_reliability_target: none` 时，`L_reliability` 只保留有效 patch 的每图可靠性均值下界 hinge，默认 floor=0.25。它是防止预测全部趋近 0 的正则项，不是静态目标物的语义标签。同时设置 `stable_stability_weight: 0` 时不加载局部参考模型，这是四份主配置的默认设置。设 `stable_reliability_weight: 0` 可完全关闭可靠性损失；`stable_stability_weight: 0` 独立关闭局部一致性。`L_stable` 启用时仍需要冻结参考模型生成 detached 权重，即使可靠性 BCE 关闭。

## 可选的 Day/Aug 局部实验

`stable_night_salad_day_aug_local_544.yaml` 单独启用 `stable_reliability_target: augmentation`，`stable_stability_weight: 0.1` 和 `stable_reliability_weight: 0.1`。冻结的 DINOv2-S 局部参考从同一预训练 checkpoint 初始化，保持 eval、关闭梯度，不随 student 更新，也不进入正常推理接口或导出推理模型。

参考 checkpoint 的 SHA256 会进入训练状态的 reference fingerprint。恢复训练时重新核对，能检测相同文件路径下预训练权重被替换；参考模型本身不会保存进 student、optimizer 或推理 checkpoint。

对几何对应且有效的 patch，软可靠性目标组合以下项，并停止梯度：

```text
appearance = clamp(cos(frozen_day_patch, frozen_aug_patch), 0, 1)
information = clamp(4 * (1 - cos(day_patch, valid_patch_image_mean)), 0, 1)
texture = clamp(within_patch_rgb_spatial_std / 0.05, 0, 1)
target = appearance * information * texture * valid_mask
```

`information` 使用归一化的冻结 patch 特征与有效 patch 图像均值的差异；`texture` 是源图 patch 内的 RGB 空间变化。这些项降低纯色区域仅因外观相似而获得高可靠性的倾向。`L_stable` 对当前 Day/Aug backbone 局部特征的余弦差异按 detached target 加权；`L_reliability` 在有效 patch 上增加 BCE，并保留均值下界。无效区域不监督，完全无有效 patch 时这两个局部损失返回可求导的零值，地点损失仍独立计算。

这是一组可消融的启发式参考，不是静态结构分割器。天空可能有纹理，车辆可能在增强前后保持不变，反光区域可能仍然具有信息量；不能宣称目标必然筛除天空、车辆或伪影。真实昼夜变化的跨外观规律也不能由人工增强完全覆盖。先比较关闭与开启局部监督的效果，再考虑改进参考、加入真正可用的配准或语义信息。

有效区域 mask 目前用于损失和诊断，不作为推理 head 的额外输入；SALAD 聚合仍能看到 fit/pad 后的全部 token。需要单独观察 padding token 的分配，不能把排除了 padding 的监督等同于已经排除了 padding 的描述子贡献。

## 配置与运行

四份主配置的训练预算、seed、输入、backbone、PK sampling 和学习率一致：

- `configs/experiments/students/stable_night_salad_day_only_544.yaml`
- `configs/experiments/students/stable_night_salad_day_aug_544.yaml`
- `configs/experiments/students/stable_night_salad_day_fake_544.yaml`
- `configs/experiments/students/stable_night_salad_day_aug_fake_544.yaml`

局部实验另用 `stable_night_salad_day_aug_local_544.yaml`。这些配置沿用了旧 SALAD 配置中的数据和权重路径；`/home/admin123/diffusion_vpr_test/...` 路径来自迁移前环境。当前工作区检查中，源 manifest 文件存在，但配置指向的 RobotCar manifest/protocol、MSLS/SVOX/Tokyo 根目录和 DINOv2 repo/checkpoint 均不存在。运行前将路径改为真实资源，检查源 manifest 内部的路径也可访问。配置能够加载不等于数据和预训练权重已完整就绪。

安装项目后使用已有 CLI。最小烟雾训练仍需要真实可访问的数据、预训练权重和验证集：

```bash
a2b-lightvpr train-vpr \
  --config configs/experiments/students/stable_night_salad_day_aug_544.yaml \
  --smoke-epochs 1
```

完整训练去掉 `--smoke-epochs 1`。冻结 backbone 可附加 `--set vpr.backbone_trainable_blocks=0`；局部监督关闭或 Dustbin 消融应使用新的 `vpr.output_dir`，避免覆盖已有实验。

`model_best.pt` 已经是推理 checkpoint，可直接用于评测和 benchmark；`checkpoints/latest.pt` 是包含 optimizer 等状态的训练 checkpoint。如需从 latest 导出，使用与训练一致的配置：

```bash
a2b-lightvpr export-vpr-inference \
  --config configs/experiments/students/stable_night_salad_day_aug_544.yaml \
  --training-checkpoint runs/vpr/stable_night_salad_544/day_aug_seed0/checkpoints/latest.pt \
  --output runs/vpr/stable_night_salad_544/day_aug_seed0/model_last_inference.pt

a2b-lightvpr benchmark \
  --config configs/experiments/students/stable_night_salad_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/day_aug_seed0/model_best.pt \
  --warmup 20 --iterations 100
```

已有 `audit-data --config ... --decode` 检查 RobotCar 与 generator manifest；它不是源图训练的必需步骤，也不应被误解为 StableNight 所有输入资源的完整审计。

## Real Night Query → 两种 Database

新 family 支持 `evaluation.database_domain: day` 或 `fake_night`，正常 Query 保持真实协议图像，ground truth 保持源 Database 的地点元数据。Fake Database 必须在评测前由用户准备，不自动调用生成器，不使用训练生成集替代测试 Database。

使用同一个推理 checkpoint，先评估真实夜间 Query → Day Database：

```bash
a2b-lightvpr evaluate-trained-suite \
  --config configs/experiments/students/stable_night_salad_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/day_aug_seed0/model_best.pt \
  --datasets robotcar_project svox_night msls_night tokyo_night \
  --set evaluation.database_domain=day \
  --set vpr.output_dir=runs/vpr/stable_night_salad_544/day_aug_seed0/eval_day
```

`--datasets all` 还包含 Tokyo sunset；上面的四个协议明确选择真实夜间 Query。MSLS 使用既有 val 协议，其余依照已有 suite 协议，不改变 query 筛选或判定半径。

Fake Database 映射通过 `evaluation.fake_database_manifest` 显式指定。CSV 必须包含 `source_path,generated_path`；`split` 可选，缺失或为空时适用于所有 split，提供时只选择与当前评测 split 相同的行。`dataset` 可作为元数据保留，匹配依据是规范化后的真实源路径；CSV 内相对路径相对于映射文件所在目录解析。映射必须完整覆盖所选评测协议的 Database，并提供可访问的生成图。同一源图的相同重复映射可接受，冲突映射、缺失映射、缺失文件或源图和生成图路径相同会报错。不同 Database 源图必须对应不同生成文件，生成文件路径不能是当前评测使用的任何真实 Database 或 Query 文件。替换函数仅改 Database 路径，并保持顺序和位姿元数据；Query 只用于检查映射冲突，保留原始路径。CSV 中未使用行不额外检查其角色。示例：

```csv
source_path,generated_path,split,dataset
/datasets/svox/database/frame001.jpg,/generated_eval/svox/frame001.png,test,svox
/datasets/msls/val/database/frame002.jpg,/generated_eval/msls/frame002.png,val,msls
```

```bash
a2b-lightvpr evaluate-trained-suite \
  --config configs/experiments/students/stable_night_salad_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/day_aug_seed0/model_best.pt \
  --datasets robotcar_project svox_night msls_night tokyo_night \
  --set evaluation.database_domain=fake_night \
  --set evaluation.fake_database_manifest=data/evaluation/fake_database.csv \
  --set vpr.output_dir=runs/vpr/stable_night_salad_544/day_aug_seed0/eval_fake_night
```

`data/evaluation/fake_database.csv` 是示例路径，仓库没有随原型提供测试生成图。测试 Database 的生成图只供评估，不能混入 `vpr_source_manifest` 或训练 generated manifest。报告中需要记录生成器/checkpoint、映射文件 hash、源 Database 数量与实际 Fake 数量，才能解释 Fake Database 的结果和生成器依赖。

Day suite 写到 `vpr.output_dir/final_suite/`，Fake Night suite 写到其 `fake_night/` 子目录。StableNight 缓存校验包含 checkpoint、训练 manifest、Fake 映射 hash、head 设置、recall cutoffs，以及实际 Query/Database/地点判定协议指纹。协议发生变化时重新计算指标，不会只把旧结果的指纹改成新值；旧 family 的缓存行为保持原有含义。当前指纹不逐图计算内容 hash：若在同一路径替换图像字节却保留 CSV 和协议，需要使用新的评测 output directory 或清除相应 suite 缓存后重跑。

## 分阶段验证与后续工作

第一阶段先确认训练、导出与两种 Database 评测路径，然后在相同数据、源图预算、optimizer step、seed、输入大小和 backbone 解冻范围下开展消融：

1. 比较四种 `train_domains`，主配置统一关闭局部教师损失。
2. 在同一训练模式下分别比较 `stable_dustbin_enabled` 开/关，Sinkhorn 迭代数保持一致。
3. 将 `stable_reliability_weight` 设为 0，检查均值下界的作用及可靠性分布。
4. 单独比较 Day/Aug 局部监督关闭与 `day_aug_local` 开启；不同时引入多层融合或 CDND。
5. 与既有 SALAD-S 分类基线、MixVPR、CDND 比较时明确区分 loss 和数据预算；不要将新 family 的 Day/Aug 模式套用到旧实验并继续使用原标签。

每组报告真实夜间 Query 的 R@1/5/10、有效 Query 数、Day/Fake Database 数、源图/view/optimizer step 预算、多个 seed 的方差、参数量、同硬件 AMP 推理 p50/p95、峰值显存，以及 reliability 均值/低分比例、Dustbin patch 分布、cluster mass 误差、有效 anchor 和有效局部 patch 数。静态结构保留、反射/车灯/动态车辆抑制等结论需要真实样例和定量证据，不能由 loss 值下降直接推出。

训练的 `training_metrics.jsonl` 记录源图、view、optimizer step 和参考图像计数；`components_mean` 中记录启用损失的原始值、每个 view 的有效区域 reliability 均值、Dustbin 均值、cluster mass 最小值/最大偏差及 gamma。即使可靠性损失关闭，predictor 均值诊断仍保留。低分比例、空间分配图和语义类别分析仍需使用辅助输出单独统计。

第二阶段才研究第 6 层与第 12 层的轻量融合。第一版保留 `stable_feature_fusion: none` 配置位，启用其他值会被配置校验拒绝；目前没有实现或评估多层融合。待动态 Dustbin 独立消融明确后，再新增轻量投影、冻结/解冻策略和额外推理成本测量。
