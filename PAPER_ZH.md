# 面向夜间机器人定位的训练期跨域邻域蒸馏

**英文题目：Training-Time Cross-Domain Neighborhood Distillation for Lightweight Nighttime Robot Localization**

**作者：** 待填写  
**单位：** 待填写  
**通信作者：** 待填写

## 摘要

夜间巡检、配送和自主驾驶机器人必须在有限算力和显存下，用白天建立的地图定位低照度观测。昼夜外观变化会破坏视觉地点识别（VPR）的相似性结构；直接生成伪夜图像只能扩展外观覆盖，不能保证其检索邻域与真实夜间查询一致。大规模 DINOv2/SALAD 模型虽具备较强鲁棒性，却通常带来机器人难以接受的参数量、延迟和显存开销。本文提出训练期跨域邻域蒸馏（CDND）：QS-Attn 离线生成 fake-night 训练图，冻结 SALAD 教师缓存白天地点邻域，轻量 MixVPR 学生同时匹配 fake-to-day 和 day-to-day 关系分布。数据集隔离 FIFO 扩大困难候选，image-ID mask 消除 day/fake 自匹配；教师、缓存和队列均在训练后移除，部署只保留一个 MixVPR-512。固定随机状态 0 的开发实验中，CDND 的夜间宏平均 R@1 相比 day+fake baseline 提升 2.97 个百分点（73.65%→76.62%）；在 Tokyo night 上为 60.28%→63.12%（+2.84pp）。在具有连续运动观测的 RobotCar 10m+30deg 协议上，轻量学生配合无需再训练的 Temporal Consensus Reranking（TCR）将 R@1 从 88.24% 提升至 93.59%，单帧编码延迟为 3.20 ms。结果表明，CDND 能把强模型的跨域检索结构转移到适合机器人部署的学生，而不增加在线教师前向；TCR 则利用机器人运动连续性进一步抑制单帧误匹配。

**英文摘要（ICRA submission draft）**

Nighttime robots must localize against daytime maps under tight compute and memory budgets. Image translation can broaden appearance coverage, but does not ensure that synthetic-night queries preserve the retrieval neighborhoods required for localization. Large DINOv2/SALAD models are robust, yet their latency and memory footprint are often unsuitable for onboard deployment. We propose Cross-Domain Neighborhood Distillation (CDND), a training-time method that transfers the daytime retrieval structure of a frozen SALAD teacher to a lightweight MixVPR student. QS-Attn is used offline to generate fake-night training images; cached SALAD descriptors define teacher neighborhood distributions, while dataset-isolated FIFO queues enlarge the candidate set and image-ID masking removes trivial self-matches. The teacher and all training-only states are discarded after training, leaving a single MixVPR descriptor at inference. In fixed-seed development experiments, CDND improves the 512-D student's nighttime macro R@1 from 73.65% to 76.62%; on Tokyo night, it improves R@1 from 60.28% to 63.12% (+2.84 percentage points). On the RobotCar 10m+30deg protocol, an optional training-free Temporal Consensus Reranking module raises the student's R@1 from 88.24% to 93.59% by exploiting query-frame continuity, with 3.20 ms single-frame encoding latency. These results support a practical separation of training-time capacity from deployment-time complexity for nighttime robot localization.

**关键词：** 视觉地点识别；昼夜域变化；知识蒸馏；邻域保持；时序重排；轻量模型

## 1 引言

视觉地点识别（Visual Place Recognition，VPR）的目标是根据查询图像，从带有位置标识的数据库中检索同一地点图像。对机器人而言，VPR 不是离线图像检索，而是定位、回环检测和长期自主导航的前端：一次错误的 top-1 可能导致错误重定位，过高的延迟则会使控制回路失去及时观测。因此，机器人 VPR 同时受到跨昼夜鲁棒性、单帧延迟、显存和数据库存储的约束。夜间低照度、局部高光、运动模糊和颜色失真会同时改变局部纹理与全局布局响应，使白天训练的模型明显退化。

当前路线大致分为三类。第一类通过更强骨干网络和大规模预训练提高单帧描述子的泛化能力，但模型、显存和描述子维度往往较大。第二类通过昼夜图像转换构造训练数据，能够直接扩展外观覆盖，但生成图像的像素真实性并不等价于检索结构正确。第三类利用连续序列和运动信息抑制单帧误匹配，适合移动机器人，却不能替代单帧描述子本身的跨域学习。

本文围绕一个机器人部署问题展开：训练阶段可以使用强教师和离线生成图，但车载/机载端只能保留一个紧凑学生；训练数据来自多个地点数据集，不能因为地点编号重名或队列混合破坏监督；在有连续运动观测时，方法还应能利用帧序列降低瞬时误匹配，而不能依赖 GPS 或未来地图真值。

基于这些约束，本文将问题分成两个层次。第一层是模型学习：不要求学生逐元素复制教师描述子，而是让学生学习教师在白天域建立的相对地点邻域。第二层是部署推理：在具有连续顺序的 RobotCar 查询上，以候选序列共识约束当前检索结果。

本文贡献如下：

1. 提出 SALAD 引导的 CDND，将强教师的白天检索邻域蒸馏到轻量学生的跨域检索分布，直接对齐夜间查询到白天地图的检索方向。
2. 设计同数据集地点均衡 batch、数据集隔离 FIFO 和明确梯度边界，使关系学习在多数据集训练中保持可解释且可恢复。
3. 给出面向机器人部署的边界：教师、生成器和队列仅用于训练，在线仅执行一个 512-D MixVPR，编码延迟为 3.20 ms。
4. 提出无需再训练的 TCR 时序扩展，在 RobotCar 连续序列上利用运动连续性抑制单帧误匹配。
5. 报告精度、延迟、显存和描述子存储的联合权衡，并明确固定单次开发实验的结论边界。

## 2 相关工作

### 2.1 全局视觉地点描述子

NetVLAD 将可学习的 VLAD 聚合引入端到端地点识别。CosPlace 将地点识别表述为大规模分类问题，EigenPlaces进一步利用视点组织改善泛化。MixVPR 通过特征混合模块在较小计算量下构造全局描述子，是本文学生模型的基础。SALAD、BoQ 和 CricaVPR 等方法结合 DINOv2 特征与更强聚合机制，表现出较强的零样本地点识别能力，但其骨干、checkpoint 或描述子通常明显大于本文主要学生。

本文不尝试重新设计学生聚合器，而是研究如何在训练期间把强模型的检索结构迁移给现有轻量描述子。

### 2.2 昼夜域转换

非配对图像转换方法通过对抗学习和对比学习在缺少严格成对监督时学习域映射。CUT 使用 patch-wise 对比目标维持内容一致性，QS-Attn 进一步选择与查询相关的特征区域。本文复用正式 QS-Attn 流程生成 fake-night 图像，但将其定位为训练数据构造组件。论文的算法贡献不是生成器结构，而是生成数据进入 VPR 训练后如何保持跨域检索关系。

### 2.3 知识蒸馏与关系蒸馏

经典知识蒸馏通常匹配分类 logits 或中间特征。对于 VPR，描述子坐标本身具有旋转和重参数化自由度，逐元素距离未必能表达真正重要的检索结构。关系蒸馏关注样本间相似度、排序或概率邻域，更符合 VPR 的目标。CDND 使用教师白天邻域作为软目标，使 fake-night 查询与 day 数据库之间的学生相似性分布逼近教师。

### 2.4 时序地点识别

SeqSLAM 等方法表明，连续观测可以通过序列一致性缓解单帧感知混淆。本文的 TCR 不维护完整地图概率模型，也不重复编码历史图像，而是在中心窗口内对连续 top-1 假设进行序号聚类，再限制当前帧候选范围。它属于顺序辅助的部署后处理，而不是纯单帧视觉方法。

## 3 问题定义与系统概览

设训练数据由白天图像 \(x_i^d\)、同一图像经过域转换得到的 \(x_i^f\)、地点标签 \(y_i\)、图像标识 \(q_i\) 和数据集标识 \(s_i\) 构成。学生编码器记为 \(f_\theta\)，冻结教师记为 \(g\)：

$$
z_i^d=f_\theta(x_i^d),\qquad
z_i^f=f_\theta(x_i^f),\qquad
t_i=g(x_i^d).
$$

所有描述子在相似度计算前进行 \(L_2\) 归一化。训练目标是在保持白天能力的同时，使 fake-night 查询能够从 day database 中检索正确地点。推理阶段只计算 \(f_\theta(x)\)。

本文主要符号见下表。

| 符号 | 含义 | 典型形状或取值 |
|---|---|---|
| \(x_i^d,x_i^f\) | 第 \(i\) 个样本的白天图及其 fake-night 图 | \(3\times320\times320\) |
| \(f_\theta,g\) | MixVPR 学生与冻结 SALAD 教师 | 学生可训练，教师冻结 |
| \(z_i^d,z_i^f\) | 学生 day/fake 描述子 | \(D\in\{128,512,4096\}\) |
| \(t_i\) | SALAD 白天描述子 | 544 维 |
| \(s_i,y_i,q_i\) | 数据集、地点类别、图像唯一标识 | 离散标识 |
| \(\mathcal C_s\) | 数据集 \(s\) 的当前 batch 与 FIFO 候选 | 至多 \(B+2048\) 项 |
| \(\mathcal V_i\) | 屏蔽同 image ID 后的有效候选 | \(\mathcal V_i\subseteq\mathcal C_{s_i}\) |
| \(p_i^T,p_i^F,p_i^D\) | 教师、fake-to-day、day-to-day 邻域分布 | 有效候选上的概率向量 |
| \(\tau\) | 关系分布温度 | 0.07 |
| \(P,K\) | 每 batch 的地点数、每地点图像数 | 4、2 |

系统数据流如下：

~~~text
白天训练图 x_day
   ├── QS-Attn 离线转换 ──> x_fake
   ├── 冻结 SALAD ─────────> teacher cache
   └── MixVPR ─────────────> z_day

x_fake ── MixVPR ──────────> z_fake

teacher cache + z_day + z_fake
   └── CDND + 地点分类 ────> 更新 MixVPR

部署：单个 MixVPR
可选部署扩展：
  top-K + 运动增量 ─────────> TCR
  top-K + 连续查询顺序 ─────> TCR
~~~

训练与部署边界可以概括为：生成器只构造数据，SALAD 只构造训练目标，分类器只提供训练期地点监督，最终单帧推理仅保留 MixVPR。时序扩展也不重新训练描述子；它们读取同一 MixVPR 排序缓存，在 CPU 上重排候选。

## 4 方法

### 4.1 QS-Attn 离线伪夜数据

生成器以白天图为源域、真实夜间图为目标域，使用 QS-Attn 正式目标训练 150,000 step。生成器只负责离线转换训练集，VPR 训练不反向传播到生成器。每张 fake-night 图保留源图的 dataset、place ID 和 image ID，从而维持地点监督和 self-mask 所需的一一对应。

正式生成器采用 batch 4、学习率 \(2\times10^{-4}\)、256 个 patch、256 维投影特征、全局 query selection、identity NCE 与 AMP。源图先将短边缩放到 352，再裁剪为 320；离线翻译产物保存为 320×480。正式 checkpoint 位于 step 150,000，其 SHA256 为 `b6474a79a58da02d44cb9b65fbe73d217b4dac27b3750e14db6d94e1ce7c2cb7`；源训练 manifest 指纹为 `1d27372440c76ab1cf335f0a51dde0e521710e9e5aeb5ab4c9802f257df4e299`，生成 manifest 指纹为 `4ef6b81a1fef8bf3949284f7a28f035153dc52284ecb44a060b83053cdf2fbe5`。这些指纹用于确认六个 VPR 运行共享同一生成数据版本。

这种解耦有两个工程优点：第一，六个 VPR 实验共享同一份生成图，避免生成器随机变化进入容量对比；第二，VPR 训练和推理均不承担生成器前向成本。

本文没有保留可用于正式比较的 FID、SSIM 或感知质量评测。因此 QS-Attn 的作用依据是其生成数据进入后续 VPR 实验后的结果，而不是一张未验证的像素质量表；论文不据此声称生成图在视觉真实性上优于其他生成器。

### 4.2 同数据集地点均衡采样

每个 physical batch 固定采用 \(P=4,K=2\)：从一个数据集中选择四个地点，每个地点抽取两张不同图像，共八个 source sample。每个 sample 同时产生 day 与 fake 两个视图，因此学生每个 batch 编码 16 张图。梯度累积步数为 2。

一个 batch 不能混合不同数据集。原因包括：

- 不同数据集的 place ID 命名空间不同；
- 各数据集坐标和地点划分标准不同；
- FIFO 候选必须来自相同数据分布；
- 混合数据集会把无意义的跨数据集负样本写入教师概率分母。

内部地点键使用 dataset 与原 place ID 的组合。image ID 用于排除同一原图的 day/fake 自匹配，分类 label 则是训练期间映射得到的连续整数，三者含义不能互换。

### 4.3 day+fake 分类基线

设分类器为 \(C\)，地点标签为 \(y_i\)。baseline 同等监督两个域：

$$
\mathcal{L}_{day+fake}
=\frac{1}{2}\left[
\mathrm{CE}(C(z_i^d),y_i)
+\mathrm{CE}(C(z_i^f),y_i)
\right].
$$

该目标让伪夜图获得地点判别能力，但它只约束类别决策，不直接规定 fake 查询对 day candidates 的细粒度排序。

### 4.4 教师邻域分布

对 anchor \(i\)，候选集合 \(\mathcal{C}_{s_i}\) 由当前 batch 的 day descriptors 和该数据集 FIFO 中的历史 descriptors 组成。同 image ID 候选被 mask：

$$
\mathcal{V}_i=\{j\in\mathcal{C}_{s_i}\mid q_j\neq q_i\}.
$$

教师在有效集合上定义温度为 \(\tau=0.07\) 的白天邻域分布：

$$
p^T_{ij}
=\frac{\exp(t_i^\top t_j/\tau)}
{\sum_{k\in\mathcal{V}_i}\exp(t_i^\top t_k/\tau)},
\qquad j\in\mathcal{V}_i.
$$

SALAD descriptors 预先以 FP16 缓存，并通过 image ID、维度、dtype 和文件指纹校验。SALAD 不进入 VPR 每步前向，也不会写入最终 checkpoint。

正式 cache 包含 118,313 行 544 维 FP16 描述子，输入尺寸为 322，文件约 123 MiB。教师 checkpoint SHA256 为 `d25c3434032a3ce5680ffcf51e05de0a7b9962772bd9112d88ca074689f18dcc`，cache 数据文件 SHA256 为 `e7cd622455732da5b53e36df9b9a6ebbbc14e81b23902af7f813fa49ca955568`。image-ID 列表和训练 manifest 另有独立 SHA256。训练启动时必须同时验证行数、维度、dtype、完成标志和这些指纹；仅文件路径相同不足以证明 cache 可复用。

### 4.5 学生跨域与日间邻域

学生构造两种分布。fake-to-day 分布直接对应目标部署场景中的跨域检索：

$$
p^F_{ij}
=\frac{\exp((z_i^f)^\top z_j^d/\tau)}
{\sum_{k\in\mathcal{V}_i}\exp((z_i^f)^\top z_k^d/\tau)}.
$$

day-to-day 分布用于稳定学生自身的日间拓扑：

$$
p^D_{ij}
=\frac{\exp((z_i^d)^\top z_j^d/\tau)}
{\sum_{k\in\mathcal{V}_i}\exp((z_i^d)^\top z_k^d/\tau)}.
$$

关系损失使用 teacher-to-student KL：

$$
\mathcal{L}_{f\rightarrow d}
=\frac{1}{|\mathcal{A}|}\sum_{i\in\mathcal{A}}
\mathrm{KL}(p_i^T\Vert p_i^F),
$$

$$
\mathcal{L}_{d\rightarrow d}
=\frac{1}{|\mathcal{A}|}\sum_{i\in\mathcal{A}}
\mathrm{KL}(p_i^T\Vert p_i^D),
$$

其中 \(\mathcal{A}\) 为至少存在一个有效候选的 anchor 集合。若某个 batch 没有有效 anchor，关系项返回与计算图相连的零标量，训练不会产生 NaN，也不会中断反向传播。

最终目标为：

$$
\boxed{
\mathcal{L}
=\mathcal{L}_{day+fake}
+0.5\left(
\mathcal{L}_{f\rightarrow d}
+0.5\mathcal{L}_{d\rightarrow d}
\right)
}
$$

### 4.6 数据集隔离 FIFO 与梯度边界

MSLS、RobotCar 和 SVOX 分别维护容量 2,048 的 FIFO。每个条目保存：

- 冻结 SALAD descriptor；
- detach 后的学生 day descriptor；
- place label；
- image ID。

队列满时按先进先出淘汰。历史学生候选在入队时 detach，因此后续 step 不会跨 batch 反向传播。当前 batch 的 day candidates 保留梯度，这一点使关系损失不仅更新 anchor，也能更新当前候选表示。教师 descriptors 始终冻结。

### 4.7 训练与推理边界

训练状态包含学生、分类器、优化器、调度器、AMP scaler、采样器游标、RNG、队列和 teacher cache 指纹，用于精确恢复。推理 checkpoint 只包含：

- MixVPR 学生权重；
- 描述子维度；
- 输入尺寸与模型类型；
- 数据和配置指纹。

它不包含 SALAD、teacher cache、分类器、FIFO 或优化器。因而 CDND 不增加部署时的模型前向次数。

精确恢复还要求保存 Python、NumPy、PyTorch 与 CUDA RNG 状态。恢复后，采样器从原游标继续，三个数据集队列保持原淘汰顺序，下一 batch 和下一步损失应与未中断运行一致。相比之下，`model_best.pt` 是部署模型，无法恢复优化器或采样顺序；将二者混用会破坏实验配对。

### 4.8 TCR 时序重排

TCR 只处理连续 traverse，不跨序列建立窗口。每个奇数窗口 `1/3/5/7/9` 使用中心帧，窗口内原始 top-1 按数据库帧序号以相邻差不超过 2 做单链聚类。每个 query frame 对同一簇最多贡献一票；最大簇确定地点中心，平局时使用平均外观相似度。当前 top-10 没有中心附近候选时才在缓存的 top-100 中补搜。

TCR 不读取查询 GPS、yaw 或绝对坐标，也不使用空间 NMS 或动态规划。它依赖查询帧的连续顺序以及数据库 traverse 内的 timestamp 排序。窗口 9 对中心帧使用前后各 4 帧，等价于 4 帧输出延迟；序列首尾自动截断，且窗口绝不跨 traverse。若中心附近在 raw top-100 中仍无候选，则保留原始 top-1。该策略只重排已有候选，因此最终 R@K 上限受 raw top-100 覆盖率约束。

RobotCar 同时保留原始协议与 map-covered temporal protocol。后者只对完整 9 帧窗口均存在 `10m+30deg` 白天数据库正样本的中心帧计分，使全部窗口使用相同分母。val 中 1402/1841 张 query 单帧有覆盖，完整窗口中心为 1346 张；night test 对应为 1794/1813 与 1770 张。覆盖真值仅生成评测 mask，不输入重排器，也不把删除无覆盖帧后的不连续片段重新拼接。该诊断用于区分地图无覆盖与候选选择错误；论文主结果仍采用完整 RobotCar night 协议。

## 5 实验设置

### 5.1 数据与协议

训练阶段使用 MSLS、RobotCar 和 SVOX 白天图以及对应 fake-night 图。评测包括：

| 协议 | 数据库数 | 查询数 | 有效查询数 | 正样本定义 | 用途 |
|---|---:|---:|---:|---|---|
| RobotCar project | 1,810 | 1,813 | 1,794 | 10m 且 30deg | 夜间主开发项 |
| SVOX night | 17,166 | 823 | 823 | 25m | 夜间主开发项 |
| Tokyo night | 75,984 | 375 | 141 | 25m | 夜间主开发项 |
| MSLS night val | 18,886 | 55 | 55 | 25m | 小样本验证项 |
| Tokyo sunset | 75,984 | 375 | 141 | 25m | 光照过渡项 |
| RobotCar day | 1,383 | 1,810 | 1,517 | 10m 且 30deg | 白天控制 |
| SVOX day | 17,166 | 14,278 | 14,278 | 25m | 白天控制 |
| SVOX sun | 17,166 | 854 | 854 | 25m | 日照控制 |
| Tokyo day | 75,984 | 375 | 141 | 25m | 白天控制 |
| MSLS day val | 18,886 | 11,065 | 10,989 | 25m | 大规模白天验证 |
| SVOX day val | 17,226 | 14,698 | 14,698 | 25m | 白天验证 |

“有效查询”指在当前 database 与正样本半径/朝向约束下至少存在一个正样本的 query，Recall 的分母采用有效查询数。夜间宏平均只包含 RobotCar project、SVOX night 和 Tokyo night；白天宏平均包含 RobotCar day、SVOX day、SVOX sun 和 Tokyo day。该宏平均对协议等权，而不是按 query 数加权。MSLS night 只有 55 个有效 query，Tokyo sunset 属于过渡光照，因此二者单独报告而不进入主宏平均。

### 5.2 实现细节

学生采用 ResNet50-MixVPR，输入为 320×320。三种 head 分别为 64 channels × 2 rows、256 channels × 2 rows 和 1024 channels × 4 rows，对应 128、512 和 4096 维。完整训练设置如下。

| 类别 | 设置 |
|---|---|
| 训练数据 | MSLS、RobotCar、SVOX 的 day 与同源 fake-night |
| 模型 | ResNet50-MixVPR，描述子 128/512/4096 |
| physical batch | 8 个 source sample，即 16 个 day/fake 视图 |
| PK sampler | \(P=4,K=2\)，每 batch 只来自一个数据集 |
| 梯度累积 | 2；等效 source batch 为 16 |
| epoch 与配额 | 20 epoch；每数据集每 epoch 20,000 source samples |
| 记录的训练预算 | 1,198,720 source samples |
| encoder/aggregator LR | \(10^{-6}\) / \(10^{-6}\) |
| classifier LR | \(10^{-4}\)，第 1 epoch warmup LR \(10^{-3}\) |
| 优化与调度 | AdamW，weight decay \(10^{-4}\)，cosine，最小 LR 比例 0.01 |
| 训练稳定性 | AMP；冻结 BatchNorm；不早停 |
| 关系温度 | 0.07 |
| FIFO | 每个训练数据集 2,048 项 |
| 关系权重 | \(\lambda_{rel}=0.5\) |
| day-to-day 权重 | 0.5（位于关系项内部） |
| teacher | SALAD，322 输入、544 维、FP16 离线 cache |
| 评测检索 | L2 归一化描述子，余弦相似度，R@1/R@5/R@10 |

六个运行除描述子 head、对应官方初始化 checkpoint、是否启用 CDND 和输出目录外保持配对。每个运行都执行固定 20-epoch 预算，但 `model_best.pt` 按开发验证指标保存，因此六个部署 checkpoint 的最佳 epoch 不同。每个运行只使用固定随机状态 0。本文报告的是单次描述性结果，不计算跨训练方差、置信区间或统计显著性，也不把小数点后差异解释为稳定趋势。

### 5.3 对比方法

训练方法比较包括严格配对的 day+fake baseline 和 SALAD-CDND。容量实验保持数据、初始化来源、采样、优化器和预算一致，只改变描述子 head。

系统级参考包括 NetVLAD、CosPlace、EigenPlaces、官方 MixVPR、SALAD、CricaVPR、BoQ 和已有 NocPlace checkpoint。不同方法保持其 native 输入尺寸。DINOv2 参考模型未应用 TCR，因此只能与 MixVPR+TCR 作精度和资源层面的系统比较，不能用于断言同条件时序算法优劣。

参考模型使用 adapted database 结果。除 NocPlace 采用其当前最强 `unified/adapted` 设置外，其余已保留精度结果采用 `native/adapted`。CricaVPR 描述子具有 batch 依赖，正文使用正式 native 结果，batch-16 数字只在附录诊断表展示。NetVLAD 当前保留了资源 benchmark，但没有保留同协议 final-suite 精度 JSON，因此精度表不补写未经当前产物验证的数字。

### 5.4 指标定义

对查询 \(q\)，记协议给出的正样本集合为 \(\mathcal P(q)\)，检索排序前 \(K\) 项为 \(\mathcal R_K(q)\)。有效查询集合 \(\mathcal Q_v\) 只包含 \(|\mathcal P(q)|>0\) 的查询，则：

$$
\operatorname{R@K}=
\frac{1}{|\mathcal Q_v|}
\sum_{q\in\mathcal Q_v}
\mathbf 1\!\left[\mathcal R_K(q)\cap\mathcal P(q)\neq\varnothing\right].
$$

夜间宏平均与白天宏平均分别为：

$$
\overline R_{night}=
\frac{R_{RobotCar}+R_{SVOX\ night}+R_{Tokyo\ night}}{3},
$$

$$
\overline R_{day}=
\frac{R_{RobotCar\ day}+R_{SVOX\ day}+R_{SVOX\ sun}+R_{Tokyo\ day}}{4}.
$$

本文的“提升”均为百分点差：\(\Delta=100(R_{CDND}-R_{baseline})\mathrm{pp}\)，不是相对百分比增长。

## 6 实验结果

### 6.1 CDND 主结果

| 维度 | Baseline 夜间 R@1 | CDND 夜间 R@1 | 夜间差值 | Baseline 白天 R@1 | CDND 白天 R@1 | 白天差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 52.40% | 55.82% | +3.42pp | 81.20% | 82.92% | +1.73pp |
| 512 | 73.65% | 76.62% | +2.97pp | 88.55% | 89.03% | +0.48pp |
| 4096 | 81.76% | 82.99% | +1.23pp | 91.66% | 91.87% | +0.21pp |

三种容量的一次固定状态结果均显示正的夜间宏平均差值，同时白天宏平均没有下降。随着描述子容量增加，绝对精度提高，而 CDND 的相对增益从 3.42pp 降到 1.23pp。这一现象与“容量越受限，结构监督越重要”的解释相符，但六个模型都只训练一次，因此不能把单调趋势上升为统计规律。

128-CDND 相对 512-CDND 的夜间/白天宏平均分别低 20.79/6.11pp；4096-CDND 相对 512-CDND 分别高 6.37/2.83pp。这说明 128 维在当前训练方案下损失过多精度，而 4096 维以显著数据库存储换取更高精度。

### 6.2 512 维分协议结果

表中 R@K 统一写为 R@1/R@5/R@10。

| 协议 | Baseline R@1/5/10 | CDND R@1/5/10 | R@1 差值 |
|---|---:|---:|---:|
| RobotCar project | 86.79/95.32/97.21% | 88.24/96.49/97.49% | +1.45pp |
| SVOX night | 73.88/85.42/88.82% | 78.49/88.34/90.89% | +4.62pp |
| Tokyo night | 60.28/73.76/78.72% | 63.12/78.72/81.56% | +2.84pp |
| MSLS night val | 60.00/81.82/83.64% | 70.91/80.00/85.45% | +10.91pp |
| Tokyo sunset | 74.47/82.27/84.40% | 75.18/82.98/84.40% | +0.71pp |
| RobotCar day | 97.96/99.80/99.87% | 98.02/99.80/99.87% | +0.07pp |
| SVOX day | 95.55/97.66/98.17% | 95.53/97.66/98.19% | -0.01pp |
| SVOX sun | 80.56/89.81/92.39% | 82.44/90.98/92.86% | +1.87pp |
| Tokyo day | 80.14/88.65/88.65% | 80.14/89.36/89.36% | +0.00pp |
| MSLS day val | 74.57/83.81/86.78% | 74.71/83.57/86.80% | +0.14pp |
| SVOX day val | 93.01/96.06/96.82% | 93.02/96.04/96.86% | +0.01pp |

三个主夜间协议的 R@1、R@5 和 R@10 均提高。CDND 在 RobotCar 上主要改善 top-1 选择，而 raw R@10 已达 97.49%，这为后续时序重排留下了候选选择空间。MSLS night 的 R@1 提升 10.91pp，但 R@5 从 81.82% 降为 80.00%；同时该协议只有 55 个有效 query，因此不能只依据 R@1 幅度断言普遍提升。白天控制总体稳定：SVOX day R@1 下降 0.01pp，MSLS day-val R@5 下降 0.24pp，其余变化较小，白天宏平均提高 0.48pp。

### 6.3 容量与资源

| 描述子 | 参数量 | inference checkpoint | P50/P95 编码延迟 | 峰值 CUDA | 每百万图 FP16/FP32 描述子 |
|---:|---:|---:|---:|---:|---:|
| 128 | 9.90M | 37.97 MiB | 1.62/1.68 ms | 112.59 MiB | 0.24/0.48 GiB |
| 512 | 10.09M | 38.72 MiB | 3.20/3.70 ms | 114.09 MiB | 0.95/1.91 GiB |
| 4096 | 10.88M | 41.72 MiB | 1.60/1.63 ms | 119.29 MiB | 7.63/15.26 GiB |

描述子维度主要影响数据库存储和相似度计算，而非 ResNet50 主干。因此 4096 维参数量只比 512 维多约 0.79M，但每百万图的描述子存储增加 8 倍。延迟来自各运行自身 benchmark，并非在完全隔离的统一负载下同时重测；非单调数值不能用于声称 4096 维编码比 512 维更快。

128 维相对 512-CDND 的夜间宏平均低 20.79pp，未形成有效精度效率折中。4096 维比 512-CDND 高 6.37pp，但数据库描述子成本增加 8 倍。由此本文选择 512 维作为主要轻量配置。

### 6.4 RobotCar 时序结果

| 方法 | 附加信息 | R@1 | R@5 | R@10 | 相对学生单帧 R@1 |
|---|---|---:|---:|---:|---:|
| MixVPR-512 CDND 单帧 | 无 | 88.24% | 96.49% | 97.49% | - |
| + TCR | 连续顺序 | 93.59% | 98.49% | 98.89% | +5.35pp |
| SALAD 单帧 | 无 | 92.08% | 99.00% | 99.67% | +3.85pp |

TCR 在 map-covered val 的窗口 1/3/5/7/9 中选择窗口 9，聚类间隔为 2 个数据库帧，中心走廊为正负 5 帧。TCR 使用完全固定的 MixVPR 描述子和 raw top-100 缓存。

| RobotCar day 方法 | R@1 | R@5 | R@10 | 相对单帧 R@1 |
|---|---:|---:|---:|---:|
| MixVPR-512 CDND 单帧 | 98.02% | 99.80% | 99.87% | - |
| + TCR | 98.55% | 99.67% | 99.87% | +0.53pp |

TCR 在 night R@1 提升的同时保持较高的 day R@1。其 night CPU 重排 P50/P95 为 0.057/0.080 ms，窗口状态约 1,112 bytes；稳态每个定位只新增一次图像编码。

同协议下，CricaVPR 和 BoQ 单帧 R@1 分别约为 89.35% 和 96.88%。TCR 的 93.59% 高于 SALAD 单帧的 92.08%，但仍低于 BoQ。参考模型没有应用相同重排器，TCR 又使用连续顺序与有限未来帧，因此该表只说明系统的精度资源位置，不能解释为同条件算法胜负，也不构成 SOTA 声明。

### 6.5 TCR 消融与覆盖诊断

| RobotCar night 10m+30deg | R@1 | R@5 | R@10 | 解释 |
|---|---:|---:|---:|---|
| 单帧 | 88.24% | 96.49% | 97.49% | 不使用时序 |
| TCR, window 3 | 92.57% | 97.18% | 98.51% | 连续 top-1 聚类 |
| TCR, window 5 | 93.24% | 97.62% | 98.66% | 连续 top-1 聚类 |
| TCR, window 7 | 93.31% | 97.85% | 98.59% | 连续 top-1 聚类 |
| TCR, window 9 | 93.46% | 97.99% | 98.59% | 连续 top-1 聚类 |

窗口敏感性表明，TCR 的 R@1 随中心上下文增加而提高，并在窗口 7--9 附近趋于稳定。TCR 只重排已有候选，不会创造新的数据库描述子。

| night 10m+30deg | 评测分母 | 单帧 R@1/5/10 | Clustered R@1/5/10 | R@1 增益 |
|---|---:|---:|---:|---:|
| 原始完整协议 | 1,794 个有效 query | 88.24/96.49/97.49% | 93.59/98.49/98.89% | +5.35pp |
| map-covered 中心诊断 | 1,770 个中心 | 88.70/96.89/97.91% | 93.79/98.70/99.10% | +5.08pp |

map-covered 结果只用于诊断“地图是否覆盖完整时间窗口”。它没有被传入重排器，也不能替代完整协议作为论文主结果。两种分母下的增益接近，说明提升并非完全来自删除无覆盖 query；但该诊断仍由 RobotCar 真值构造，不能用于部署时在线判断。

### 6.6 与 SALAD 的精度和资源位置

| 系统 | 参数量 | 描述子 | batch-1 P50 | 峰值 CUDA | 每百万图 FP16 描述子 | RobotCar R@1 |
|---|---:|---:|---:|---:|---:|---:|
| MixVPR-512 CDND 单帧 | 10.09M | 512 | 3.20 ms | 114.09 MiB | 0.95 GiB | 88.24% |
| MixVPR-512 CDND + TCR | 10.09M | 512 | 3.20 ms + 0.08 ms P95 CPU | 114.09 MiB | 0.95 GiB | 93.59% |
| SALAD-DINOv2 单帧 | 87.80M | 544 | 14.25 ms | 363.11 MiB | 1.01 GiB | 92.08% |

SALAD 的描述子只有 544 维，因此其数据库存储与 512 维学生接近；本文的部署优势主要来自参数量、编码延迟和显存，而不是相对 SALAD 的描述子压缩。TCR 以连续顺序换取超过 SALAD 单帧的 R@1，但 SALAD 的 R@5/R@10 仍更高，且没有对 SALAD 应用同一时序算法。因此合理结论是“轻量时序系统在本开发协议上的 R@1 高于 SALAD 单帧”，而不是“CDND 学生单帧击败教师”或“算法全面优于 SALAD”。

## 7 讨论

### 7.1 为什么邻域蒸馏有效

day+fake 分类只要求正确类别 logit 较大，同一地点内部和不同困难负样本之间的相对结构未被直接约束。SALAD 的白天描述子提供更细粒度的候选概率。CDND 把这种局部排序迁移给 fake-to-day 分布，因此训练目标与实际夜间 query 检索 day database 的方向一致。

### 7.2 为什么教师不应进入部署

强教师若在推理阶段参与融合，系统的参数量和延迟将由教师决定，轻量学生的意义会被削弱。本文将教师前向移到一次性 cache 构建阶段，训练结束后完整剔除。CDND 的主要价值应理解为训练资源换取部署表示，而不是双模型集成。

### 7.3 128、512 与 4096 维的角色

128 维展示了更强相对蒸馏收益，却无法保持足够绝对精度；4096 维展示了容量上限，但数据库存储显著增加。512 维并非所有指标最高，而是当前实验中更合理的精度、存储和时序部署折中。

### 7.4 TCR 的适用边界

TCR 依赖连续顺序和未来帧，窗口 9 引入 4 帧延迟，因此属于顺序辅助系统，不能被描述为纯单帧视觉 VPR。当前时序方法只在 RobotCar 这一个连续米制数据集上验证，尚无证据支持跨数据集泛化。

### 7.5 失败项与不能过度解释的结果

CDND 并非每个单元格都提高：128 维 MSLS night R@1 下降 1.82pp，4096 维 Tokyo sunset R@1 下降 0.71pp，部分白天 R@5/R@10 也有小幅下降。主结论依赖预先定义的夜间宏平均，而不是“所有协议全部获胜”。此外，BoQ 单帧在 RobotCar 达到 96.88%，高于 TCR 的 93.59%；本文保留该结果并明确不作 SOTA 声明。

## 8 局限性

1. 每种方法只训练固定随机状态 0，无法估计跨初始化方差。
2. RobotCar、SVOX、MSLS 和 Tokyo 均参与开发过程，不是完全封存的外部测试。
3. 当前实验未隔离生成器质量、teacher 选择和所有关系损失子项的独立贡献。
4. TCR 的收益依赖连续顺序和有限未来帧。目前只在 RobotCar 上验证，不能推广到无序图像检索或其他连续数据集。
5. 参考模型没有统一应用 TCR，系统级结果不能解释成同条件时序算法排名。
6. 当前速度数据来自单机测量，需要在目标机器人硬件上重新基准测试。

后续工作应增加独立随机重复、封存外部序列、不同教师和队列长度消融，并在真实机器人计算平台上同时测量吞吐、能耗和长期定位稳定性。

## 9 结论

本文提出面向夜间机器人定位的训练期跨域邻域蒸馏 CDND。冻结 SALAD 在白天域提供检索邻域结构，轻量 MixVPR 学习 fake-to-day 与 day-to-day 关系；QS-Attn、teacher cache、分类器和 FIFO 均在训练期结束后移除。固定状态开发结果显示，512-D 学生的夜间宏平均 R@1 从 73.65% 提升到 76.62%，Tokyo night 从 60.28% 提升到 63.12%。对于具有连续运动观测的 RobotCar，训练无关的 TCR 将单帧 R@1 从 88.24% 提升到 93.59%，同时保留 3.20 ms 的单帧编码路径。整体结果支持“训练期强教师、部署期轻学生、连续场景可选时序重排”的机器人系统设计，但仍需要多随机种子、封存测试和目标硬件基准来验证稳定性与实际部署收益。

## 参考文献

[1] Lowry S, Sünderhauf N, Newman P, et al. Visual Place Recognition: A Survey. IEEE Transactions on Robotics, 2016.

[2] Arandjelović R, Gronat P, Torii A, et al. NetVLAD: CNN Architecture for Weakly Supervised Place Recognition. CVPR, 2016.

[3] Berton G, Masone C, Caputo B. Rethinking Visual Geo-localization for Large-Scale Applications. CVPR, 2022.

[4] Berton G, Trivigno G, Caputo B, et al. EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition. ICCV, 2023.

[5] Ali-bey A, Chaib-draa B, Giguère P. MixVPR: Feature Mixing for Visual Place Recognition. WACV, 2023.

[6] Izquierdo S, Civera J. Optimal Transport Aggregation for Visual Place Recognition. CVPR, 2024.

[7] Ali-bey A, Chaib-draa B, Giguère P. BoQ: A Place is Worth a Bag of Learnable Queries. CVPR, 2024.

[8] Lu F, Lan X, Zhang L, et al. CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition. CVPR, 2024.

[9] Oquab M, Darcet T, Moutakanni T, et al. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research, 2024.

[10] Park T, Efros A A, Zhang R, et al. Contrastive Learning for Unpaired Image-to-Image Translation. ECCV, 2020.

[11] Hu X, Zhou X, Huang Q, et al. QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation. CVPR, 2022.

[12] Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network. NIPS Deep Learning Workshop, 2015.

[13] Milford M, Wyeth G. SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights. ICRA, 2012.

## 附录 A 复现路径与结果来源

完整固定状态训练、缓存、评测和汇总：

~~~bash
fish scripts/run_seed0_all.fish
~~~

RobotCar TCR：

~~~bash
fish scripts/run_robotcar_temporal_512.fish
~~~

| 结果类别 | 唯一来源或目录 | 用途 |
|---|---|---|
| 宏平均与容量资源 | `runs/comparisons/cdnd_seed0_summary.json` | 正文表 6.1、6.3 |
| 128 baseline/CDND | `runs/vpr/cdnd_128/*_seed0/final_suite/summary.json` | 附录 B.1 |
| 512 baseline/CDND | `runs/vpr/cdnd_512/*_seed0/final_suite/summary.json` | 正文表 6.2、附录 B.2 |
| 4096 baseline/CDND | `runs/vpr/cdnd_4096/*_seed0/final_suite/summary.json` | 附录 B.3 |
| TCR 选择 | `robotcar_temporal/clustered_val_selection.json` | 附录 C.1 |
| night/day 时序结果 | `robotcar_temporal/clustered_{night,day}_test.json` | 正文 6.4、附录 D |
| 系统比较 | `robotcar_temporal/comparison.json` | 正文 6.6 |
| 参考模型精度 | `runs/reference/*/final_suite/*/*/summary.json` | 附录 E.1 |
| 参考模型资源 | `runs/reference/resource_benchmark.json` | 附录 E.2 |

训练 checkpoint 与 inference checkpoint 不能互换。前者用于中断恢复，包含优化器、调度器、分类器、采样器、RNG 和 CDND 队列；后者用于部署，只保留学生模型和必要元数据。论文发布时应同时提供配置、inference checkpoint SHA256、生成 manifest 指纹与评测协议指纹，但不需要发布含数据路径的完整训练状态。

| 描述子 | 方法 | 最佳 epoch | `model_best.pt` SHA256 |
|---:|---|---:|---|
| 128 | day+fake | 16 | `e41069d9c402b7216d635c86c6267ebf11f075249e9fc6bea06877dae4435184` |
| 128 | CDND | 15 | `c2096dd33b73fe4b18b29491961922e20832359bb92e45e5e590a4653eb4dc63` |
| 512 | day+fake | 18 | `713c3e8c6263c5a6973e045acfdd9aff6a334edbedc8eed23b9ea07fde507ac5` |
| 512 | CDND | 14 | `36a6c803f5386a86042071469f215765576dfc8bd244941ed814a2901d0a9504` |
| 4096 | day+fake | 10 | `69c8eabc9047a59e6f23f5a063c07c1185fd298f078529699bc3043c9365bba6` |
| 4096 | CDND | 8 | `74d5fcc76b89869fa3624f7814325c92981d29854cdf708db2c517a472dea480` |

各协议的 query protocol fingerprint 与 evaluation fingerprint 已逐项保存在相应 `final_suite/summary.json`。复现时应由程序校验，不应从论文排版文本手工恢复。

## 附录 B 三种容量的全部协议结果

表中每个 R@K 均为百分比；“有效/总”是至少存在一个正样本的 query 数与原始 query 数。B 与 C 分别表示 day+fake baseline 和 CDND。

### B.1 MixVPR-128

| 协议 | 有效/总 | B R@1/5/10 | C R@1/5/10 |
|---|---:|---:|---:|
| RobotCar project | 1794/1813 | 75.31/90.64/93.76 | 77.76/91.97/95.32 |
| SVOX night | 823/823 | 47.14/64.76/72.30 | 52.13/69.87/77.28 |
| Tokyo night | 141/375 | 34.75/53.19/63.12 | 37.59/58.87/66.67 |
| MSLS night val | 55/55 | 45.45/61.82/70.91 | 43.64/65.45/78.18 |
| Tokyo sunset | 141/375 | 56.74/71.63/79.43 | 58.16/74.47/78.72 |
| RobotCar day | 1517/1810 | 97.43/99.74/99.80 | 97.63/99.67/99.80 |
| SVOX day | 14278/14278 | 92.88/95.99/96.87 | 93.00/95.99/96.93 |
| SVOX sun | 854/854 | 67.10/80.33/85.71 | 70.14/82.90/87.70 |
| Tokyo day | 141/375 | 67.38/80.14/83.69 | 70.92/79.43/84.40 |
| MSLS day val | 10989/11065 | 66.68/78.61/82.33 | 67.77/79.34/82.81 |
| SVOX day val | 14698/14698 | 89.57/94.06/95.24 | 89.85/94.14/95.33 |

### B.2 MixVPR-512

| 协议 | 有效/总 | B R@1/5/10 | C R@1/5/10 |
|---|---:|---:|---:|
| RobotCar project | 1794/1813 | 86.79/95.32/97.21 | 88.24/96.49/97.49 |
| SVOX night | 823/823 | 73.88/85.42/88.82 | 78.49/88.34/90.89 |
| Tokyo night | 141/375 | 60.28/73.76/78.72 | 63.12/78.72/81.56 |
| MSLS night val | 55/55 | 60.00/81.82/83.64 | 70.91/80.00/85.45 |
| Tokyo sunset | 141/375 | 74.47/82.27/84.40 | 75.18/82.98/84.40 |
| RobotCar day | 1517/1810 | 97.96/99.80/99.87 | 98.02/99.80/99.87 |
| SVOX day | 14278/14278 | 95.55/97.66/98.17 | 95.53/97.66/98.19 |
| SVOX sun | 854/854 | 80.56/89.81/92.39 | 82.44/90.98/92.86 |
| Tokyo day | 141/375 | 80.14/88.65/88.65 | 80.14/89.36/89.36 |
| MSLS day val | 10989/11065 | 74.57/83.81/86.78 | 74.71/83.57/86.80 |
| SVOX day val | 14698/14698 | 93.01/96.06/96.82 | 93.02/96.04/96.86 |

### B.3 MixVPR-4096

| 协议 | 有效/总 | B R@1/5/10 | C R@1/5/10 |
|---|---:|---:|---:|
| RobotCar project | 1794/1813 | 89.69/97.38/98.05 | 90.02/97.21/98.44 |
| SVOX night | 823/823 | 83.96/91.37/93.80 | 85.91/92.71/94.90 |
| Tokyo night | 141/375 | 71.63/78.01/84.40 | 73.05/80.85/85.11 |
| MSLS night val | 55/55 | 69.09/80.00/83.64 | 72.73/83.64/87.27 |
| Tokyo sunset | 141/375 | 82.27/88.65/89.36 | 81.56/88.65/89.36 |
| RobotCar day | 1517/1810 | 98.55/99.74/99.87 | 98.55/99.74/99.87 |
| SVOX day | 14278/14278 | 96.67/98.17/98.66 | 96.57/98.13/98.60 |
| SVOX sun | 854/854 | 87.00/93.21/94.96 | 87.24/93.79/95.20 |
| Tokyo day | 141/375 | 84.40/90.78/91.49 | 85.11/90.78/91.49 |
| MSLS day val | 10989/11065 | 77.96/85.83/88.42 | 77.36/85.32/87.79 |
| SVOX day val | 14698/14698 | 94.35/96.71/97.33 | 94.27/96.58/97.28 |

完整表说明 CDND 不是逐单元格占优。例如 4096 维 RobotCar R@5 下降 0.17pp，MSLS day-val 三项均下降；论文的主要证据是预定义宏平均与跨域主协议，而非选择性挑出所有正差值。

## 附录 C RobotCar val 窗口选择

### C.1 TCR 窗口敏感性

TCR 固定使用 raw top-100、top-10 主候选、聚类间隔 2 帧和中心走廊 5 帧，仅在 val 比较窗口 \(W\in\{1,3,5,7,9\}\)。


### C.1 TCR 窗口敏感性

该表在共同的 map-covered val 中心集合上比较，固定 top-K=10、聚类间隔 2、中心半径 5。

| 窗口 | R@1 | R@5 | R@10 | top-100 回退 | CPU P95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 90.79% | 96.58% | 97.99% | 0 | 0.000 ms |
| 3 | 92.57% | 97.18% | 98.51% | 13 | 0.061 ms |
| 5 | 93.24% | 97.62% | 98.66% | 24 | 0.078 ms |
| 7 | 93.31% | 97.85% | 98.59% | 36 | 0.091 ms |
| **9** | **93.46%** | **97.99%** | **98.59%** | **45** | **0.102 ms** |

窗口 9 按预定 R@1 优先规则入选。其 R@10 并非最高，且需要 4 帧未来信息；因此部署时必须把精度提升与输出延迟一并报告。

## 附录 D RobotCar 完整距离、朝向与 R@K 网格

每个单元格为 R@1/R@5/R@10。Cluster 表示 TCR。不同距离和 yaw 组合的有效 query 数可能不同，不能仅凭跨行数值判断协议难度。

### D.1 Night

| 距离 | 朝向 | 单帧 | TCR | Cluster |
|---:|---:|---:|---:|---:|
| 5m | 10deg | 76.12/94.73/96.85 | 80.24/93.93/95.25 | 79.38/96.62/97.94 |
| 5m | 30deg | 75.70/94.44/96.46 | 80.25/93.88/95.23 | 79.85/96.46/97.70 |
| 5m | 不限 | 74.83/93.35/95.34 | 79.32/92.79/94.12 | 78.99/95.40/96.62 |
| 10m | 10deg | 88.45/96.64/97.72 | 91.64/96.42/97.67 | 92.78/98.52/98.98 |
| 10m | 30deg | 88.24/96.49/97.49 | 93.59/98.49/98.89 |
| 10m | 不限 | 87.47/95.70/96.74 | 91.11/95.75/97.13 | 92.94/97.68/98.12 |
| 25m | 10deg | 91.39/96.90/97.81 | 94.32/96.51/97.81 | 96.00/98.87/99.10 |
| 25m | 30deg | 91.58/97.01/97.73 | 94.96/96.90/98.12 | 97.45/99.00/99.17 |
| 25m | 不限 | 91.40/96.75/97.46 | 94.70/96.58/97.85 | 97.41/98.79/98.95 |
| 50m | 10deg | 92.05/96.64/97.42 | 94.57/96.30/97.37 | 96.47/98.38/98.66 |
| 50m | 30deg | 92.38/96.96/97.57 | 95.36/96.91/97.90 | 98.18/98.73/98.95 |
| 50m | 不限 | 92.83/97.30/98.01 | 95.75/97.30/98.29 | 98.68/99.01/99.34 |

### D.2 Day

| 距离 | 朝向 | 单帧 | TCR | Cluster |
|---:|---:|---:|---:|---:|
| 5m | 10deg | 92.92/99.59/99.66 | 93.46/98.98/99.12 | 93.06/99.46/99.66 |
| 5m | 30deg | 92.56/99.47/99.67 | 93.09/98.80/98.94 | 92.82/99.40/99.67 |
| 5m | 不限 | 91.83/98.75/99.01 | 92.42/98.15/98.29 | 92.09/98.68/99.08 |
| 10m | 10deg | 97.52/99.80/99.87 | 98.39/99.66/99.80 | 97.85/99.66/99.87 |
| 10m | 30deg | 98.02/99.80/99.87 | 98.88/99.67/99.80 | 98.55/99.67/99.87 |
| 10m | 不限 | 96.87/98.83/98.89 | 97.79/98.70/98.83 | 97.52/98.89/99.15 |
| 25m | 10deg | 98.02/99.54/99.74 | 98.42/99.41/99.67 | 98.42/99.67/99.74 |
| 25m | 30deg | 98.90/99.68/99.87 | 99.22/99.61/99.87 | 99.55/99.81/99.87 |
| 25m | 不限 | 97.45/98.28/98.47 | 97.77/98.22/98.54 | 98.35/98.73/98.85 |
| 50m | 10deg | 97.20/98.89/99.48 | 98.24/99.35/99.48 | 97.98/99.41/99.48 |
| 50m | 30deg | 98.02/99.04/99.49 | 99.11/99.43/99.55 | 99.11/99.30/99.49 |
| 50m | 不限 | 95.60/97.09/97.65 | 96.65/97.34/97.83 | 96.90/97.58/97.89 |

完整网格显示，时序方法并非在所有 R@K 上占优。TCR 在严格 5m night 条件提高 R@1，却降低 R@5/R@10；TCR 在多数 night 单元格提高三项指标，但在部分 day 严格单元格只提高 R@1。该现象符合重排器主要改变候选顺序而非生成新候选的性质。

## 附录 E 参考模型扩展比较

### E.1 固定协议 R@1

夜间宏平均只取 RobotCar、SVOX night、Tokyo night。NocPlace 使用 `unified/adapted`，其余为 `native/adapted`。

| 模型 | 设置 | D | RobotCar | SVOX | Tokyo night | MSLS night | Tokyo sunset | 夜间宏平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CosPlace-R50 | native/adapted | 512 | 47.05% | 46.66% | 51.77% | 23.64% | 74.47% | 48.49% |
| EigenPlaces-R50 | native/adapted | 512 | 54.46% | 46.05% | 51.77% | 16.36% | 78.01% | 50.76% |
| EigenPlaces-R50 | native/adapted | 2048 | 56.24% | 50.06% | 53.19% | 21.82% | 80.14% | 53.17% |
| NocPlace-R50 | unified/adapted | 512 | 70.35% | 54.31% | 54.61% | 38.18% | 65.96% | 59.76% |
| MixVPR-R50 official | native/adapted | 4096 | 75.59% | 63.30% | 62.41% | 43.64% | 82.98% | 67.10% |
| MixVPR-512 CDND | trained/fit-pad | 512 | 88.24% | 78.49% | 63.12% | 70.91% | 75.18% | 76.62% |
| CricaVPR-DINOv2 | native/adapted | 10752 | 89.35% | 85.30% | 73.76% | 70.91% | 83.69% | 82.80% |
| SALAD-DINOv2 | native/adapted | 544 | 92.08% | 92.47% | 80.14% | 96.36% | 82.27% | 88.23% |
| BoQ-DINOv2 | native/adapted | 12288 | 96.88% | 97.69% | 87.23% | 100.00% | 88.65% | 93.93% |
| NetVLAD-VGG16 | 同协议精度 JSON 未保留 | 4096 | - | - | - | - | - | - |

BoQ 是该扩展表中精度最高的单帧模型。它没有进入正文主要对手表，是因为论文核心叙事围绕 SALAD 训练期教师与学生部署边界；但在附录完整披露 BoQ 可以避免把系统结果误写成当前最强性能。

### E.2 资源

| 模型 | 参数量 | checkpoint | D | P50/P95 | 峰值 CUDA | 每百万图 FP16 |
|---|---:|---:|---:|---:|---:|---:|
| MixVPR-512 CDND | 10.09M | 38.72 MiB | 512 | 3.20/3.70 ms | 114.09 MiB | 0.95 GiB |
| CosPlace-R50 | 24.56M | 93.98 MiB | 512 | 3.56/3.69 ms | 155.04 MiB | 0.95 GiB |
| EigenPlaces-R50-512 | 24.56M | 93.98 MiB | 512 | 3.54/3.69 ms | 155.60 MiB | 0.95 GiB |
| EigenPlaces-R50-2048 | 27.70M | 105.99 MiB | 2048 | 3.56/3.65 ms | 174.17 MiB | 3.81 GiB |
| MixVPR-R50-4096 official | 10.88M | 41.72 MiB | 4096 | 3.23/3.62 ms | 76.35 MiB | 7.63 GiB |
| NetVLAD-VGG16 | 149.00M | 1080.42 MiB | 4096 | 10.83/11.36 ms | 860.40 MiB | 7.63 GiB |
| NocPlace-R50 | 统一资源记录未保留 | - | 512 | - | - | 0.95 GiB（按维度计算） |
| CricaVPR-DINOv2 | 106.76M | 561.56 MiB | 10752 | 15.89/16.21 ms | 710.02 MiB | 20.03 GiB |
| SALAD-DINOv2 | 87.80M | 335.01 MiB | 544 | 14.25/14.58 ms | 363.11 MiB | 1.01 GiB |
| BoQ-DINOv2 | 95.21M | 363.27 MiB | 12288 | 12.35/12.60 ms | 396.55 MiB | 22.89 GiB |

参考模型资源来自统一 resource benchmark，学生资源来自其正式 checkpoint benchmark。两者均为当前机器上的 batch-1 测量，但运行时间和环境负载未做严格同步控制，因此延迟只适合数量级比较。

### E.3 CricaVPR batch 依赖诊断

| 设置 | RobotCar | SVOX night | Tokyo night | MSLS night | Tokyo sunset |
|---|---:|---:|---:|---:|---:|
| native/adapted 正式结果 | 89.35% | 85.30% | 73.76% | 70.91% | 83.69% |
| native/batch16 appendix | 89.74% | 90.52% | 81.56% | 81.82% | 87.23% |

CricaVPR 的描述子依赖同一 batch 中的其他图像。batch-16 结果明显更高，但这改变了推理条件，不能与逐图编码模型当作完全同条件主表结果；因此正文采用正式 native/adapted 数字，并把 batch-16 作为实现敏感性展示。

## 附录 F 核心张量、梯度与状态

| 名称 | 典型形状 | 是否有梯度 | 生命周期/保存位置 |
|---|---|---|---|
| day image | \(B\times3\times320\times320\) | 否 | 当前 batch |
| fake image | \(B\times3\times320\times320\) | 否 | 当前 batch |
| student day descriptor | \(B\times D\) | 是 | 当前 batch；detach 后入队 |
| student fake descriptor | \(B\times D\) | 是 | 当前 batch |
| SALAD descriptor | \(B\times544\) | 否 | FP16 cache |
| current day candidates | \(B\times D\) | 是 | 当前 batch |
| historical day candidates | 最多 \(2048\times D\)/数据集 | 否 | training checkpoint |
| teacher FIFO candidates | 最多 \(2048\times544\)/数据集 | 否 | training checkpoint |
| classification logits | \(2B\times C\) | 是 | 不进入 inference checkpoint |
| self-mask | \(B\times(B+Q)\) | 否 | 当前关系损失 |
| sampler cursor | 标量及数据集内部状态 | 否 | training checkpoint |
| RNG state | Python/NumPy/Torch/CUDA | 否 | training checkpoint |

其中 \(B=8\)，\(D\in\{128,512,4096\}\)，\(Q\leq2048\)，\(C\) 为训练地点类别数。当前 batch day candidate 保留梯度，而 FIFO 中的历史学生 candidate 已 detach；把二者都 detach 会削弱关系损失，把二者都保留计算图则会造成跨 batch 反向传播和不可控内存增长。

## 附录 G 投稿前检查项

- 核对作者、单位、基金和通信作者信息。
- 对参考文献卷期页码、作者顺序和公开代码链接作最终校验。
- 明确所有数据集、QS-Attn、MixVPR、SALAD 和参考模型许可证。
- 在独立环境执行两个正式工作流的短验证。
- 公布六个 inference checkpoint、配置及 SHA256。
- 公布生成 manifest、SALAD cache 元数据和协议指纹，不公布本机绝对路径。
- 在摘要、正文和结论中持续使用“固定状态开发结果”措辞。
- 不把 TCR 写成纯视觉方法，不把 TCR 写成无延迟单帧方法。
- 不把 TCR 与未应用同一重排器的 SALAD/BoQ 写成算法公平比较。
- 不声称跨随机状态稳定、统计显著、封存测试或 SOTA。
