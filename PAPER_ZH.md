# 面向昼夜视觉地点识别的训练期跨域邻域蒸馏

**英文题目：Training-Time Cross-Domain Neighborhood Distillation for Day-to-Night Visual Place Recognition**

**作者：** 待填写  
**单位：** 待填写  
**通信作者：** 待填写

## 摘要

昼夜外观变化会显著破坏视觉地点识别模型在白天图像上学到的相似性结构。直接使用图像到图像转换扩充夜间外观虽然能够增加训练域覆盖，却不能保证伪夜图与真实夜间查询共享正确的检索邻域；另一方面，采用大规模视觉基础模型可以提高鲁棒性，但会带来较大的参数量、描述子存储和推理开销。本文提出训练期跨域邻域蒸馏方法 CDND。首先使用 QS-Attn 将白天训练图离线转换为 fake-night 图像；随后以冻结的 SALAD 模型为教师，在白天域构造地点邻域概率分布，并约束轻量 MixVPR 学生的 fake-to-day 与 day-to-day 邻域分布。为扩大负候选覆盖，本文为各训练数据集维护相互隔离的 FIFO 队列，同时严格屏蔽同一图像产生的自匹配。SALAD、缓存描述子、分类器和队列均只参与训练，最终推理仍为单个 MixVPR。固定随机状态 0 的开发实验表明，CDND 在 128、512 和 4096 维学生上的夜间宏平均 R@1 分别提高 3.42、2.97 和 1.23 个百分点。作为部署扩展，本文进一步给出运动增量引导的 MGTR v1 时序重排：MixVPR-512 CDND 在 RobotCar 10m+30deg 协议上的 R@1 由 88.24% 提高至 91.92%，CPU 重排 P95 约为 0.09 ms；在 Gardens Point 的 200 帧夜间序列上，固定四帧窗口使 exact R@1 从 28.5% 提高至 34.0%。这些结果说明，训练期强教师可以改善轻量学生的跨域检索结构，而无需增加部署模型复杂度。

**关键词：** 视觉地点识别；昼夜域变化；知识蒸馏；邻域保持；时序重排；轻量模型

## 1 引言

视觉地点识别（Visual Place Recognition，VPR）的目标是根据查询图像，从带有位置标识的数据库中检索同一地点图像。它是机器人长期自主导航、自动驾驶定位、回环检测和图像地理定位的重要组成部分。实际部署中的核心困难不是同域图像匹配，而是天气、季节、视角和光照变化引起的外观偏移，其中昼夜变化尤其明显：夜间低照度、局部高光、运动模糊和颜色失真会同时改变局部纹理与全局布局响应。

当前路线大致分为三类。第一类通过更强骨干网络和大规模预训练提高单帧描述子的泛化能力，但模型、显存和描述子维度往往较大。第二类通过昼夜图像转换构造训练数据，能够直接扩展外观覆盖，但生成图像的像素真实性并不等价于检索结构正确。第三类利用连续序列和运动信息抑制单帧误匹配，适合移动机器人，却不能替代单帧描述子本身的跨域学习。

本文关注如下约束：训练阶段可以使用强教师和离线生成图，部署阶段只能保留一个紧凑学生；训练数据来自多个地点数据集，不能因为地点编号重名或队列混合破坏监督；最终方法还应支持具有连续运动观测的机器人序列。

基于这些约束，本文将问题分成两个层次。第一层是模型学习：不要求学生逐元素复制教师描述子，而是让学生学习教师在白天域建立的相对地点邻域。第二层是部署推理：在数据库具有度量坐标、查询具有相邻运动增量时，以低开销动态规划约束连续候选路径。

本文贡献如下：

1. 提出 SALAD 引导的 CDND，将强教师的白天检索邻域蒸馏到轻量学生的跨域检索分布。
2. 设计同数据集地点均衡 batch、数据集隔离 FIFO 和明确的梯度边界，使跨域关系学习可扩展到多个训练数据集。
3. 保持教师完全退出推理图，系统部署时只需一个 128、512 或 4096 维 MixVPR 学生。
4. 给出 MGTR v1 时序扩展，并在 RobotCar 米制协议与 Gardens Point 序列索引协议上分析连续一致性的作用。
5. 报告固定状态下的容量、精度和描述子存储关系，并明确单次实验与开发协议的结论边界。

## 2 相关工作

### 2.1 全局视觉地点描述子

NetVLAD 将可学习的 VLAD 聚合引入端到端地点识别。CosPlace 将地点识别表述为大规模分类问题，EigenPlaces进一步利用视点组织改善泛化。MixVPR 通过特征混合模块在较小计算量下构造全局描述子，是本文学生模型的基础。SALAD、BoQ 和 CricaVPR 等方法结合 DINOv2 特征与更强聚合机制，表现出较强的零样本地点识别能力，但其骨干、checkpoint 或描述子通常明显大于本文主要学生。

本文不尝试重新设计学生聚合器，而是研究如何在训练期间把强模型的检索结构迁移给现有轻量描述子。

### 2.2 昼夜域转换

非配对图像转换方法通过对抗学习和对比学习在缺少严格成对监督时学习域映射。CUT 使用 patch-wise 对比目标维持内容一致性，QS-Attn 进一步选择与查询相关的特征区域。本文复用正式 QS-Attn 流程生成 fake-night 图像，但将其定位为训练数据构造组件。论文的算法贡献不是生成器结构，而是生成数据进入 VPR 训练后如何保持跨域检索关系。

### 2.3 知识蒸馏与关系蒸馏

经典知识蒸馏通常匹配分类 logits 或中间特征。对于 VPR，描述子坐标本身具有旋转和重参数化自由度，逐元素距离未必能表达真正重要的检索结构。关系蒸馏关注样本间相似度、排序或概率邻域，更符合 VPR 的目标。CDND 使用教师白天邻域作为软目标，使 fake-night 查询与 day 数据库之间的学生相似性分布逼近教师。

### 2.4 时序地点识别

SeqSLAM 等方法表明，连续观测可以通过路径一致性缓解单帧感知混淆。本文的 MGTR v1 不维护完整地图概率模型，也不重复编码历史图像，而是在每帧 top-K 候选上进行短窗口动态规划。它属于运动辅助的部署后处理，而不是纯单帧视觉方法。

## 3 问题定义与系统概览

设训练数据由白天图像 \(x_i^d\)、同一图像经过域转换得到的 \(x_i^f\)、地点标签 \(y_i\)、图像标识 \(q_i\) 和数据集标识 \(s_i\) 构成。学生编码器记为 \(f_\theta\)，冻结教师记为 \(g\)：

$$
z_i^d=f_\theta(x_i^d),\qquad
z_i^f=f_\theta(x_i^f),\qquad
t_i=g(x_i^d).
$$

所有描述子在相似度计算前进行 \(L_2\) 归一化。训练目标是在保持白天能力的同时，使 fake-night 查询能够从 day database 中检索正确地点。推理阶段只计算 \(f_\theta(x)\)。

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
可选部署扩展：top-K + 运动增量 ──> MGTR v1
~~~

## 4 方法

### 4.1 QS-Attn 离线伪夜数据

生成器以白天图为源域、真实夜间图为目标域，使用 QS-Attn 正式目标训练 150,000 step。生成器只负责离线转换训练集，VPR 训练不反向传播到生成器。每张 fake-night 图保留源图的 dataset、place ID 和 image ID，从而维持地点监督和 self-mask 所需的一一对应。

这种解耦有两个工程优点：第一，六个 VPR 实验共享同一份生成图，避免生成器随机变化进入容量对比；第二，VPR 训练和推理均不承担生成器前向成本。

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

### 4.8 MGTR v1 时序重排

MGTR v1 只处理连续 traverse，不跨序列建立窗口。每个新查询帧执行一次 MixVPR 编码并检索 raw top-100。RobotCar 数据库候选先进行空间 NMS：若两候选距离不超过 5m 且朝向差不超过 30deg，则只保留相似度更高者；同位置反方向图像仍保留。NMS 后取 top-K 并作帧内分数归一化：

$$
a_{t,j}=
\frac{s_{t,j}-s_{t,K}}
{s_{t,1}-s_{t,K}+10^{-8}}.
$$

查询 GPS 只转换为相邻位移增量：

$$
\Delta s_t=\lVert p_t-p_{t-1}\rVert_2.
$$

动态规划接口不接收查询绝对位置。候选转移残差为：

$$
r_{t,i,j}=
\left|
\lVert d_{t,j}-d_{t-1,i}\rVert_2-\Delta s_t
\right|.
$$

若数据库候选位移超过查询增量加转移半径，则该转移无效；否则使用 Huber 代价。窗口路径得分为外观分数之和减去运动代价。当前帧输出以获胜路径结尾的数据库图像，不对坐标求平均。

RobotCar 最终参数由 val 冻结为：窗口 4、top-K 10、转移半径 15m、运动权重 0.5。完整窗口无有效路径时逐步缩短窗口，两帧仍失败则回退当前帧 raw top-1。在线稳态每次定位只新增一次图像编码。

Gardens Point 没有米制坐标，因而仅把数据库序列索引当作位置，查询增量固定为一帧，不执行空间 NMS。主窗口预先固定为 4，窗口 3、5、6 只作敏感性分析。

### 4.9 RA-GTR 高精度扩展（待正式评测）

为分析 4096 维学生的候选覆盖上限，代码进一步实现 Route-Aware Geometric Temporal Reranking（RA-GTR）。该扩展当前尚未产生冻结 test 结果，因此不进入本文主结果和结论。

RA-GTR 首先使用 MixVPR-4096 CDND 检索 raw top-100，并在空间 NMS 后保留 top-10 或 top-20。局部候选由冻结的 DISK-depth 与 LightGlue-disk 匹配，并以 USAC_MAGSAC 基础矩阵内点计算几何分数：

$$
G_{t,j}=\log(1+n_{\mathrm{inlier}})
\frac{n_{\mathrm{inlier}}}{\max(n_{\mathrm{match}},1)}.
$$

若一帧所有候选均没有有效几何匹配，则退化为 MixVPR 外观分数。否则外观与几何分数分别在帧内归一化，再以验证集选择的系数融合。

RobotCar 数据库按 traverse 和 timestamp 建立累计路线进度 $s_j$。查询端仍只向动态规划提供相邻 GPS 的标量距离增量，而不提供绝对坐标。每个窗口分别保持固定的正向或反向假设：

$$
r_{t,i,j}=\left|(s_j-s_i)-d\Delta s_t\right|,
\qquad d\in\{-1,+1\}.
$$

一个窗口内不允许改变方向。RA-GTR 固定延迟两帧，以路径相对 raw top-1 的平均优势作为保守覆盖门限；优势不足、raw top-1 不可比较或无有效路径时保留单帧结果。窗口 3、4、5、6 和其他参数只由 RobotCar val 冻结。

如果普通 4096-CDND 的 val 结果未达到预设阈值，代码允许一次 RobotCar 专用 fallback。该阶段只使用 train day 与对应 fake-night，不加载真实夜间图像。它从初始 4096 checkpoint 离线挖掘 day top-100，将 10m 且 30deg 内的不同图像作为正样本、25m 外图像作为负样本、10--25m 候选忽略，并增加权重 0.25 的 fake-to-day supervised contrastive loss。候选 day 描述子冻结，微调预算固定为 5 个 epoch、每 epoch 60,000 samples。

## 5 实验设置

### 5.1 数据与协议

训练阶段使用 MSLS、RobotCar 和 SVOX 白天图以及对应 fake-night 图。评测包括：

| 协议 | 数据库/查询 | 正样本定义 | 用途 |
|---|---|---|---|
| RobotCar project | day / night | 10m 且 30deg | 夜间主开发项 |
| SVOX night | gallery / night | 25m | 夜间主开发项 |
| Tokyo night | day / night | 25m | 夜间主开发项 |
| MSLS night val | day / night | 25m | 小样本补充项 |
| Tokyo sunset | day / sunset | 25m | 光照过渡补充项 |
| RobotCar、SVOX、Tokyo day | day / day-like | 对应数据集协议 | 白天控制 |
| Gardens Point | day-left / night-right | 序列索引容差 | 外部序列分析 |

夜间宏平均只包含 RobotCar project、SVOX night 和 Tokyo night；白天宏平均包含 RobotCar day、SVOX day、SVOX sun 和 Tokyo day。MSLS night 查询数较少，Tokyo sunset 属于过渡光照，因此单独报告而不进入主宏平均。

Gardens Point 包含 day-left、day-right 和 night-right 各 200 张。day-right 用于控制和主窗口协议检查，night-right 用于一次外部序列评测。每个 query 对总体 R@1 的贡献为 0.5 个百分点。

### 5.2 实现细节

学生采用 ResNet50-MixVPR，输入为 320x320。三种 head 分别为 64 channels x 2 rows、256 x 2 rows 和 1024 channels x 4 rows，对应 128、512 和 4096 维。physical batch 为 8，梯度累积为 2。每个 epoch 从三个数据集各采样 20,000 个 source sample，总训练预算记录为 1,198,720 samples。关系温度 0.07，FIFO 容量 2,048，关系权重 0.5，day-to-day 子项权重 0.5。

所有正式模型只运行固定随机状态 0。本文报告的数值是描述性结果，不计算跨训练方差，也不作统计显著性结论。

### 5.3 对比方法

训练方法比较包括严格配对的 day+fake baseline 和 SALAD-CDND。容量实验保持数据、初始化来源、采样、优化器和预算一致，只改变描述子 head。

系统级参考包括 NetVLAD、CosPlace、EigenPlaces、官方 MixVPR、SALAD、CricaVPR、BoQ 和已有 NocPlace checkpoint。不同方法保持其 native 输入尺寸。DINOv2 参考模型未应用 MGTR，因此只能与 MixVPR+MGTR 作精度和资源层面的系统比较，不能用于断言同条件时序算法优劣。

## 6 实验结果

### 6.1 CDND 主结果

| 维度 | day+fake 夜间宏平均 | CDND 夜间宏平均 | 夜间差值 | day+fake 白天宏平均 | CDND 白天宏平均 |
|---:|---:|---:|---:|---:|---:|
| 128 | 52.40% | 55.82% | +3.42pp | 81.20% | 82.92% |
| 512 | 73.65% | 76.62% | +2.97pp | 88.55% | 89.03% |
| 4096 | 81.76% | 82.99% | +1.23pp | 91.66% | 91.87% |

三种容量的一次固定状态结果均显示正的夜间宏平均差值，同时白天宏平均没有下降。随着学生容量增加，绝对精度上升，但 CDND 相对 baseline 的增益减小。这与容量瓶颈假设一致：较小学生更需要教师提供结构约束，但 128 维本身的容量不足仍造成明显绝对精度损失。

### 6.2 512 维分协议结果

| 协议 | day+fake R@1 | CDND R@1 | 差值 |
|---|---:|---:|---:|
| RobotCar project | 86.79% | 88.24% | +1.45pp |
| SVOX night | 73.88% | 78.49% | +4.62pp |
| Tokyo night | 60.28% | 63.12% | +2.84pp |
| MSLS night val | 60.00% | 70.91% | +10.91pp |
| Tokyo sunset | 74.47% | 75.18% | +0.71pp |
| RobotCar day | 97.96% | 98.02% | +0.07pp |
| SVOX day | 95.55% | 95.53% | -0.01pp |
| SVOX sun | 80.56% | 82.44% | +1.87pp |
| Tokyo day | 80.14% | 80.14% | 0.00pp |

主夜间三项均提高。MSLS night 的增益较大，但其有效查询规模较小，不能将该单项幅度直接外推到其他数据集。SVOX day 的变化为 -0.01pp，属于当前固定运行中的轻微波动，白天宏平均仍提高 0.48pp。

### 6.3 容量与资源

| 描述子 | 参数量 | inference checkpoint | P50 编码延迟 | 峰值推理显存 | 每百万图 FP16 描述子 |
|---:|---:|---:|---:|---:|---:|
| 128 | 9.90M | 37.97 MiB | 1.62 ms | 112.59 MiB | 0.24 GiB |
| 512 | 10.09M | 38.72 MiB | 3.20 ms | 114.09 MiB | 0.95 GiB |
| 4096 | 10.88M | 41.72 MiB | 1.60 ms | 119.29 MiB | 7.63 GiB |

描述子维度主要影响数据库存储和相似度计算，而非 ResNet50 主干。因此 4096 维参数量只略高于 512 维，但每百万图 FP16 描述子需要 8 倍存储。已有延迟来自单机测量，受运行负载影响；表中延迟非严格单调，不能据此声称 4096 维编码更快。

128 维相对 512-CDND 的夜间宏平均低 20.79pp，未形成有效精度效率折中。4096 维比 512-CDND 高 6.37pp，但数据库描述子成本增加 8 倍。由此本文选择 512 维作为主要轻量配置。

### 6.4 RobotCar MGTR v1

| 方法 | 10m+30deg R@1 | 变化 |
|---|---:|---:|
| MixVPR-512 CDND 单帧 | 88.24% | - |
| MixVPR-512 CDND + MGTR v1 | 91.92% | +3.68pp |

RobotCar day R@1 从 98.02% 提高到 98.88%。MGTR CPU 重排 P50/P95 为 0.076/0.090 ms，滑窗状态约 492 bytes，每个新定位只新增一次图像编码。结果说明短窗口运动约束可以纠正部分单帧感知混淆，同时没有产生多倍持续编码开销。

同协议下，SALAD、CricaVPR 和 BoQ 单帧 R@1 分别约为 92.08%、89.35% 和 96.88%。MGTR 接近但没有达到 SALAD 单帧结果，也没有超过 BoQ。由于参考模型没有应用相同重排器，这张表只说明轻量系统的精度资源位置。

### 6.5 Gardens Point 序列结果

| 模型 | 描述子维度 | 单帧 exact R@1 | 窗口4 exact R@1 | 差值 |
|---|---:|---:|---:|---:|
| MixVPR-512 CDND | 512 | 28.5% | 34.0% | +5.5pp |
| SALAD | 544 | 36.0% | 38.5% | +2.5pp |
| BoQ | 12288 | 36.5% | 38.0% | +1.5pp |
| EigenPlaces-2048 | 2048 | 25.0% | 33.5% | +8.5pp |
| CosPlace-512 | 512 | 26.5% | 32.5% | +6.0pp |
| CricaVPR | 10752 | 27.5% | 30.0% | +2.5pp |
| NetVLAD | 4096 | 17.5% | 21.5% | +4.0pp |

CDND+窗口4在这组结果中低于 SALAD 和 BoQ，但使用 512 维描述子。更宽松的正负2帧容差下，CDND 窗口4 R@1 为 85.5%，说明不少 exact 错误仍落在相邻位置附近。

CDND 的窗口敏感性为：

| 窗口 | exact R@1 | R@5 | R@10 |
|---:|---:|---:|---:|
| 3 | 32.5% | 81.0% | 88.0% |
| 4 | 34.0% | 81.5% | 88.0% |
| 5 | 37.0% | 82.0% | 88.0% |
| 6 | 36.5% | 83.0% | 88.0% |

窗口 5 的事后 exact 数值更高，但主协议预先固定窗口 4，因此本文不把窗口 5 重新定义为主结果。该现象提示未来应在独立序列上选择窗口，再使用封存序列验证。

### 6.6 RA-GTR 待填结果

RA-GTR 的代码、缓存指纹、val-only 选择和一次性 test 工作流已经实现，但正式长流程尚未运行。提交论文前只有在 RobotCar night `10m+30deg R@1` 严格高于 SALAD 单帧的 92.0847%，且 day 相对 4096 单帧下降不超过 0.5pp 时，才可将 RA-GTR 加入正式结果。不得使用 512 缓存上的可行性诊断值替代最终结果，也不得根据 night test 再次调整参数。

## 7 讨论

### 7.1 为什么邻域蒸馏有效

day+fake 分类只要求正确类别 logit 较大，同一地点内部和不同困难负样本之间的相对结构未被直接约束。SALAD 的白天描述子提供更细粒度的候选概率。CDND 把这种局部排序迁移给 fake-to-day 分布，因此训练目标与实际夜间 query 检索 day database 的方向一致。

### 7.2 为什么教师不应进入部署

强教师若在推理阶段参与融合，系统的参数量和延迟将由教师决定，轻量学生的意义会被削弱。本文将教师前向移到一次性 cache 构建阶段，训练结束后完整剔除。CDND 的主要价值应理解为训练资源换取部署表示，而不是双模型集成。

### 7.3 128、512 与 4096 维的角色

128 维展示了更强相对蒸馏收益，却无法保持足够绝对精度；4096 维展示了容量上限，但数据库存储显著增加。512 维并非所有指标最高，而是当前实验中更合理的精度、存储和时序部署折中。

### 7.4 MGTR 的适用边界

RobotCar MGTR 使用查询相邻位移增量，因此属于视觉与运动辅助系统。它不使用查询绝对坐标，但也不能被描述为纯视觉 VPR。Gardens 版本只使用序列顺序，指标单位是帧索引。两种协议共享短窗口候选路径思想，却不共享相同的物理量。

## 8 局限性

1. 每种方法只训练固定随机状态 0，无法估计跨初始化方差。
2. RobotCar、SVOX、MSLS 和 Tokyo 均参与开发过程，不是完全封存的外部测试。
3. Gardens Point 只有 200 个夜间查询，单个 query 会改变 R@1 约 0.5pp。
4. 当前实验未隔离生成器质量、teacher 选择和所有关系损失子项的独立贡献。
5. MGTR 的收益依赖连续顺序和运动信息，不能推广到无序图像检索。
6. 参考模型没有统一应用 MGTR，系统级结果不能解释成同条件时序算法排名。
7. 当前速度数据来自单机测量，需要在目标机器人硬件上重新基准测试。

后续工作应增加独立随机重复、封存外部序列、不同教师和队列长度消融，并在真实机器人计算平台上同时测量吞吐、能耗和长期定位稳定性。

## 9 结论

本文提出面向昼夜 VPR 的训练期跨域邻域蒸馏 CDND。该方法利用冻结 SALAD 在白天域建立的邻域结构，监督轻量 MixVPR 的 fake-to-day 和 day-to-day 检索分布，并通过同数据集 PK batch、独立 FIFO、self-mask 和明确梯度边界保证训练正确性。固定状态开发结果显示，CDND 在三种描述子容量上均改善夜间宏平均，而推理仍只需单个学生。MGTR v1 进一步说明，在连续机器人序列中，短窗口运动一致性能够以很小 CPU 开销提高轻量系统定位精度。

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

## 附录 A 复现路径

完整固定状态训练：

~~~bash
fish scripts/run_seed0_all.fish
~~~

RobotCar 时序评测：

~~~bash
fish scripts/run_robotcar_temporal_512.fish
~~~

4096-CDND RA-GTR 高精度评测与必要时的 RobotCar-only fallback：

~~~bash
fish scripts/run_robotcar_ragtr_4096.fish
~~~

Gardens Point 与参考模型评测：

~~~bash
fish scripts/run_gardens_vpr_models.fish
~~~

主要结果来源：

~~~text
runs/comparisons/cdnd_seed0_summary.json
runs/vpr/cdnd_512/salad_no_gate_seed0/robotcar_temporal/comparison.json
runs/vpr/cdnd_4096/ragtr_final/comparison.json
runs/external/gardens_point/models_summary.json
~~~

## 附录 B 核心张量与状态

| 名称 | 典型形状 | 是否有梯度 | 保存位置 |
|---|---|---|---|
| day image | B x 3 x 320 x 320 | 否 | batch |
| fake image | B x 3 x 320 x 320 | 否 | batch |
| student day descriptor | B x D | 是 | 当前 batch；detach 后入队 |
| student fake descriptor | B x D | 是 | 当前 batch |
| SALAD descriptor | B x 544 | 否 | FP16 cache |
| current day candidates | B x D | 是 | 当前 batch |
| historical day candidates | 最多 2048 x D/数据集 | 否 | training checkpoint |
| classification logits | 2B x C | 是 | 不进入 inference checkpoint |

其中 \(B=8\)，\(D\in\{128,512,4096\}\)，\(C\) 为训练地点类别数。

## 附录 C 投稿前检查项

- 核对作者、单位、基金和通信作者信息。
- 对参考文献卷期页码和公开代码链接作最终校验。
- 明确所有数据集许可证与下载方式。
- 在独立环境执行三个正式工作流的短验证。
- 公布推理 checkpoint、配置和 SHA256。
- 在正文和摘要中持续使用“固定状态开发结果”措辞。
- 不把 Gardens Point 帧索引写成米制定位。
- 不把 MGTR 的运动辅助结果写成纯视觉单帧结果。
