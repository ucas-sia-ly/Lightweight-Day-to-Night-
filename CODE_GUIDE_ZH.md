# A2B-LightVPR 代码导读

本文档面向第一次阅读本仓库的人。它不重复论文的完整论证，而是回答五个工程问题：数据从哪里来、每个文件负责什么、核心函数如何连接、训练和评测产物如何形成、修改代码时应先看哪里。

> 当前代码固定使用随机状态 `0`。仓库保留 MixVPR 128/512/4096 三种单帧容量实验，但 RobotCar 时序实验只允许 MixVPR-512。Gardens Point、RA-GTR、MGTR v2、Motion-Aligned MGTR 和 4096 维时序均已不在当前实现中。

## 1. 先理解整个项目

项目研究昼夜视觉地点识别（Visual Place Recognition, VPR）。白天数据库与夜间查询存在明显外观域差异，直接使用轻量 MixVPR 往往无法稳定保持白天检索结构。

完整方法分为三个阶段：

```text
原始昼/夜数据
    |
    +-- prepare-data --------------------------> 训练 manifest
    |
    +-- QS-Attn 生成器训练
            |
            +-- translate-train ---------------> fake-night 图像
                                                    |
白天图像 ------------------------------------------+-- day+fake 分类基线
    |
    +-- 冻结 SALAD，仅离线编码白天图 -------------> FP16 teacher cache
                                                    |
                                                    +-- CDND 训练
                                                          |
                                                          +-- 单个 MixVPR 推理
                                                                  |
                                                                  +-- 单帧评测
                                                                  +-- RobotCar TCR
```

核心思想如下：

1. **QS-Attn 只构造训练输入。** 它把白天训练图离线翻译成 fake-night，不参与 VPR 推理。
2. **day+fake 是正式基线。** 同一地点的白天图和 fake-night 图共享分类标签。
3. **SALAD 只提供白天邻域结构。** SALAD 描述子预先缓存，VPR 训练时不再运行教师网络。
4. **CDND 蒸馏邻域分布。** 学生学习 fake-query 到 day-candidates 以及 day-query 到 day-candidates 的关系，而不是强制逐图描述子相等。
5. **部署只保留 MixVPR。** 推理 checkpoint 不含生成器、SALAD、分类器、FIFO、优化器或 teacher cache。
6. **TCR 是 RobotCar 系统扩展。** 它使用连续查询顺序和有限未来帧，因此不是纯单帧视觉 VPR。

训练目标为：

\[
L=L_{day+fake}+0.5\left(L_{fake\rightarrow day}+0.5L_{day\rightarrow day}\right).
\]

## 2. 推荐阅读顺序

如果只想先读懂论文核心，建议按以下顺序：

1. `configs/vpr_cdnd_salad_512.yaml`：先看一遍正式方法的全部超参数。
2. `src/a2b_lightvpr/vpr/losses.py`：理解 baseline 与 CDND 的数学实现。
3. `src/a2b_lightvpr/vpr/neighborhood.py`：理解当前 batch 和历史 FIFO 候选如何拼接。
4. `src/a2b_lightvpr/data.py` 中的 `PairedVPRDataset` 与 `PKBalancedBatchSampler`：理解一个 batch 如何产生。
5. `src/a2b_lightvpr/vpr/trainer.py`：把数据、模型、损失、队列、恢复串起来。
6. `src/a2b_lightvpr/vpr/model.py`：理解 MixVPR 128/512/4096 的模型结构。
7. `src/a2b_lightvpr/vpr/teacher_cache.py` 与 `teachers.py`：理解 SALAD 为什么只在预处理阶段出现。
8. `src/a2b_lightvpr/trained_suite.py`：理解单帧结果如何生成。
9. `src/a2b_lightvpr/robotcar_temporal.py`：最后阅读 TCR。
10. `scripts/run_seed0_all.fish` 与 `scripts/run_robotcar_temporal_512.fish`：查看真实实验是如何编排的。

若要理解 fake-night 的来源，则在第 2 步前阅读 `generator/` 下的四个实现文件。

## 3. 仓库结构

```text
A2B-LightVPR/
├── README.md                 # 使用与复现入口
├── PAPER_ZH.md               # 中文论文草稿
├── CODE_GUIDE_ZH.md          # 本代码导读
├── pyproject.toml            # 安装、依赖和 CLI 声明
├── configs/                  # 七组正式实验/参考模型配置
├── scripts/                  # 两个可执行工作流
├── src/a2b_lightvpr/         # 项目源码
│   ├── generator/            # QS-Attn 图像翻译
│   └── vpr/                  # MixVPR、SALAD cache、CDND 与训练
└── tests/                    # 单元测试和工作流契约测试
```

`data/`、`runs/` 和外部 checkpoint 是运行产物或本机数据，不属于核心 Python 包。它们的路径由配置文件引用。

## 4. 顶层文件

### `pyproject.toml`

定义包名 `a2b-lightvpr`、Python 版本、基础依赖、可选评测依赖和 pytest 设置。安装后命令：

```bash
a2b-lightvpr ...
```

实际映射到 `a2b_lightvpr.cli:main`。`faiss-cpu` 是可选加速；没有 FAISS 时检索代码自动回退到 NumPy 矩阵乘法。

### `README.md`

面向使用者的快速说明，包括环境、数据、正式命令、结果与限制。它回答“怎么运行”。

### `PAPER_ZH.md`

面向论文读者的完整方法和实验草稿。它回答“为什么这样设计、结果意味着什么”。

### `CODE_GUIDE_ZH.md`

即本文档，回答“代码具体在哪里、每个函数做什么”。

## 5. 配置文件

配置均由 `config.py` 解析为不可变 dataclass。命令行可用 `--set section.field=value` 临时覆盖，但公开配置不接受 `seed` 字段。

| 文件 | 用途 | 关键区别 |
|---|---|---|
| `generator_qsattn_official_objective_150k.yaml` | 正式 QS-Attn 生成器 | 训练 150k step，输出正式 fake-night 数据 |
| `vpr_day_fake_pk_128.yaml` | 128 维 baseline | day+fake 分类，无 SALAD/CDND |
| `vpr_cdnd_salad_128.yaml` | 128 维 CDND | 128 维学生，共享 SALAD cache |
| `vpr_day_fake_pk_512.yaml` | 512 维 baseline | 论文主要轻量配置的对照组 |
| `vpr_cdnd_salad_512.yaml` | 512 维 CDND | 论文主要方法，亦为 RobotCar 时序模型 |
| `vpr_day_fake_pk_4096.yaml` | 4096 维 baseline | 单帧容量实验，不用于时序 |
| `vpr_cdnd_salad_4096.yaml` | 4096 维 CDND | 单帧容量实验，不用于时序 |
| `reference_models_extended.yaml` | 参考模型注册表 | checkpoint、仓库、原生尺寸、维度与下载信息 |

六个 VPR 配置的关键配对原则：同一维度内 baseline/CDND 使用相同初始化、训练数据、`P=4,K=2`、epoch、每 epoch 配额、优化器和调度器。CDND 配置只额外启用 SALAD cache、关系队列和关系损失。

注意：当前正式配置包含本机绝对数据路径。这是研究工作区配置，不是完全可移植的发布配置；迁移机器时应统一修改 `data` 路径和初始化 checkpoint 路径。

## 6. 基础模块

### `src/a2b_lightvpr/__init__.py`

包标记文件，不包含运行逻辑。

### `src/a2b_lightvpr/config.py`

集中定义配置结构、读取、覆盖和静态约束。

| 类/函数 | 职责 |
|---|---|
| `DataConfig` | 保存 RobotCar manifest/protocol、生成器/VPR manifest、fake-night 根目录以及 Tokyo/SVOX/MSLS 根目录和图像尺寸。 |
| `GeneratorConfig` | 保存 QS-Attn 目标、训练步数、batch、学习率、AMP、恢复路径和 PatchNCE 参数。 |
| `VPRConfig` | 保存模型族、聚合头、描述子维度、训练预算、采样器、优化器、SALAD/CDND、FIFO 和 PK batch 参数。 |
| `EvalConfig` | 保存正样本距离/朝向阈值、Recall 列表、评测协议和数据库域。 |
| `ExperimentConfig` | 将四个子配置组成一次实验。 |
| `ExperimentConfig.as_dict()` | 将嵌套 dataclass 转为普通字典，供 checkpoint 和 JSON 使用。 |
| `ExperimentConfig.fingerprint()` | 对配置做稳定 SHA256；主动清空 resume 路径，使恢复位置不改变实验身份。 |
| `_construct()` | 将 YAML 字典实例化为指定 dataclass，并把列表字段标准化成 tuple。 |
| `load_config()` | 读取 YAML、构造完整配置并调用严格校验。 |
| `apply_overrides()` | 解析 `section.field=value`，通过 YAML 规则转换标量/列表并返回新的不可变配置。 |
| `validate_config()` | 拒绝已删除目标、非法模型/维度组合、错误 teacher、错误采样预算、非法尺寸和不成对设置。 |

`validate_config()` 是实验防错的第一道门。新增配置字段时，应同时更新 dataclass、覆盖解析和此函数的约束。

### `src/a2b_lightvpr/checkpointing.py`

提供跨生成器与 VPR 共用的状态和原子写入工具。

| 函数 | 职责 |
|---|---|
| `checkpoint_config_matches()` | 只比较指定配置 section，确认恢复 checkpoint 与当前实验相容。 |
| `file_sha256()` | 分块计算文件 SHA256，避免一次读入大权重。 |
| `data_fingerprints()` | 记录 RobotCar、协议、manifest 和生成数据的内容哈希。 |
| `rng_state()` | 保存 Python、NumPy、PyTorch CPU 和 CUDA RNG 状态。 |
| `restore_rng_state()` | 恢复上述 RNG，使断点后的采样和 loss 延续。 |
| `atomic_torch_save()` | 先写临时文件再替换目标，避免中断留下半个 checkpoint。 |
| `write_json()` | 以同样的临时文件替换方式写格式化 JSON。 |

### `src/a2b_lightvpr/transforms.py`

统一生成器和 VPR 的几何及归一化策略。

| 类/函数 | 职责 |
|---|---|
| `_scaled_size()` | 计算保持比例、短边达到目标值后的尺寸。 |
| `resize_short_side()` | 按短边缩放 PIL 图像，不裁剪。 |
| `GeneratorTrainTransform.__call__()` | 对单张生成器训练图执行缩放、随机裁剪、翻转和 `[-1,1]` 归一化。 |
| `GeneratorTrainTransform.paired()` | 对成对图共享同一几何随机参数，保证空间对齐。 |
| `resize_to_fit()` | 保持比例缩放并用 ImageNet 均值填充到固定画布。 |
| `generator_inference_tensor()` | 生成器推理预处理，尺寸调整到网络可处理的倍数。 |
| `VPRTransform.__call__()` | 对 VPR 输入执行 fit-and-pad，再做 ImageNet 归一化。 |
| `denormalize_generator()` | 将生成器 `[-1,1]` 输出恢复到 `[0,1]`。 |
| `normalized_generator_resize()` | 对已归一化 tensor 做不带 padding 的尺寸调整。 |

### `src/a2b_lightvpr/runtime.py`

负责部署态模型加载和资源测试。

| 函数 | 职责 |
|---|---|
| `load_vpr_model()` | 根据配置构造模型，严格检查 inference checkpoint 类型和描述子维度，再加载模型权重。 |
| `benchmark_model()` | 测 batch-1 延迟、参数量、checkpoint 大小和 CUDA 峰值显存，并写 `benchmark.json`。 |

## 7. 数据准备与采样

### `src/a2b_lightvpr/prepare.py`

扫描外部数据集并生成两个正式 manifest：生成器训练 manifest 和 VPR 白天训练 manifest。

| 函数 | 职责 |
|---|---|
| `_write_csv()` | 原子写 CSV，先写 `.tmp` 再替换。 |
| `_image_problem()` | 检查解码、尺寸、近常量、近全黑和近全白图像。 |
| `_paired_rows()` | 将 day/night 池循环配成等长的非配对训练行。 |
| `_even_sample()` | 沿排序后的序列均匀下采样到指定上限。 |
| `_cap_views_per_place()` | 限制每个 dataset/place 的视图数量，避免地点过度主导。 |
| `_parse_svox_train_path()` | 从 SVOX 文件名解析坐标和前/后相机方向。 |
| `build_manifests()` | 汇集 DarkDriving、RobotCar、BDD100K、MSLS、SVOX；做图像/天气筛选、训练划分、地点统计和冻结数量校验，最终写 manifest 与审计报告。 |

数据角色要区分：BDD100K 等可帮助生成器学习昼夜外观，却未必提供 VPR 地点监督；VPR manifest 只收录有可用地点标签的白天图。

### `src/a2b_lightvpr/clean_weather.py`

用于 MSLS 天气筛选，防止雾、雨、雪等因素混入“清晰昼夜”域构造。

| 函数 | 职责 |
|---|---|
| `_candidate_hash()` | 对候选路径和元数据生成缓存身份。 |
| `_image_problem()` | 在加载 CLIP 前先拒绝损坏或明显异常图片。 |
| `filter_clear_weather()` | 使用本地 CLIP 对候选图做天气语义筛选；缓存候选哈希、checkpoint 哈希和接受集合，相同输入可直接复用。 |

### `src/a2b_lightvpr/data.py`

这是数据层核心，既定义 manifest 行，也定义训练 Dataset 和三类平衡采样器。

#### RobotCar 与 manifest

| 类/函数 | 职责 |
|---|---|
| `RobotCarRow` | RobotCar 一行的不可变结构：split、角色、域、traverse、时间戳、坐标、朝向和路径。 |
| `RobotCarRow.image_id` | 由 traverse、timestamp、role 生成稳定图像 ID。 |
| `RobotCarRow.place_id` | 按 10m 网格和 30 度方向桶生成地点分类 ID。 |
| `read_robotcar_manifest()` | 读取 CSV，解析数值并把相对图像路径映射到仓库根。 |
| `read_generator_manifest()` | 读取指定 split，应用 audit invalid IDs 和 `valid` 标记。 |
| `audit_data()` | 检查 train/val/test traverse 泄漏、统计域和角色、可选逐图解码，并核对协议。 |
| `read_generated_manifest()` | 读取 fake-night 输出的 `manifest.csv`。 |

#### Dataset

| 类/方法 | 职责 |
|---|---|
| `BalancedGeneratorDataset.__init__()` | 按 dataset 建 day/night 池并读取数据集权重。 |
| `BalancedGeneratorDataset.__len__()` | 返回按训练步数人为设定的数据长度。 |
| `BalancedGeneratorDataset._open()` | 以 RGB 打开图片。 |
| `BalancedGeneratorDataset.__getitem__()` | 由固定状态与 index 派生独立 RNG，先选数据集，再各选一张 day/night，形成确定性的非配对样本。 |
| `PairedVPRDataset.__init__()` | 读取生成 manifest，仅保留 train；构造 `<dataset>:<place_id>` 防跨数据集冲突；过滤样本不足地点并建立分类标签。 |
| `PairedVPRDataset.__len__()` | 返回过滤后的训练行数。 |
| `PairedVPRDataset.__getitem__()` | 同时读取 day/fake，返回分类 label、稳定 image key、dataset、坐标/朝向；CDND 时附加缓存的 SALAD 描述子。 |
| `PairedVPRDataset.balanced_weights()` | 为通用加权采样器计算“先平衡数据集、再平衡地点、最后平衡地点内图像”的概率。 |
| `ImagePathDataset.__init__()` | 保存待评测行及 transform。 |
| `ImagePathDataset.__len__()` | 返回待编码图像数。 |
| `ImagePathDataset.__getitem__()` | 返回预处理图像和原始索引，保证批量编码后可还原顺序。 |

#### 采样器

| 类/方法 | 职责 |
|---|---|
| `PKBalancedBatchSampler.__init__()` | 按数据集和地点建立索引，校验每个地点至少 K 张、每个数据集至少 P 个地点、quota 可被 `P*K` 整除。 |
| `PKBalancedBatchSampler.__len__()` | 返回所有 dataset quota 对应的 batch 总数。 |
| `PKBalancedBatchSampler.set_epoch()` | 切换 epoch 并重置 cursor/缓存 batch。 |
| `PKBalancedBatchSampler._epoch_batches()` | 用固定状态为每个数据集构造 `P` 地点、每地点 `K` 张不同图的 batch，并打乱数据集 batch 顺序。 |
| `PKBalancedBatchSampler.__iter__()` | 从可恢复 cursor 开始逐 batch 产出索引。 |
| `PKBalancedBatchSampler.state_dict()` | 保存 epoch、cursor、quota 和 PK 结构。 |
| `PKBalancedBatchSampler.load_state_dict()` | 严格核对结构后恢复下一 batch 位置。 |
| `PKBalancedBatchSampler.statistics()` | 输出各数据集 quota、地点数和 batch 统计。 |
| `RollingBalancedSampler.__init__()` | 建立按数据集滚动覆盖的确定性采样状态。 |
| `RollingBalancedSampler.set_epoch()` | 设置 epoch。 |
| `RollingBalancedSampler.__len__()` | 返回 quota 总样本数。 |
| `RollingBalancedSampler._new_cycle()` | 为一个数据集生成新的无放回随机排列周期。 |
| `RollingBalancedSampler._cycle()` | 获取或创建指定周期。 |
| `RollingBalancedSampler._window()` | 从跨周期序列取固定长度窗口，边界处仍尽量保持唯一。 |
| `RollingBalancedSampler.__iter__()` | 交错各数据集 quota，输出覆盖均衡的样本索引。 |
| `RollingBalancedSampler.statistics()` | 报告配额和覆盖统计。 |
| `CoverageBalancedSampler.__init__()` | 建立较早的覆盖均衡采样结构。 |
| `CoverageBalancedSampler.set_epoch()` | 设置 epoch。 |
| `CoverageBalancedSampler.__len__()` | 返回 quota 总量。 |
| `CoverageBalancedSampler._dataset_indices()` | 为单个数据集生成该 epoch 的确定性覆盖索引。 |
| `CoverageBalancedSampler.__iter__()` | 合并并打乱各数据集索引。 |
| `CoverageBalancedSampler.statistics()` | 输出各数据集的样本与 quota 信息。 |
| `benchmark_rows()` | 从 RobotCar manifest 取指定 split 的 database 与 query。 |

正式 CDND 使用 `PKBalancedBatchSampler`。`P=4,K=2` 意味着物理 batch 为 8，且一个 batch 只能来自一个数据集。这样 FIFO 与邻域分布不会把 MSLS、RobotCar、SVOX 的坐标/地点语义混在一起。

## 8. QS-Attn fake-night 生成器

### `src/a2b_lightvpr/generator/__init__.py`

子包标记文件，无逻辑。

### `src/a2b_lightvpr/generator/networks.py`

| 类/函数 | 职责 |
|---|---|
| `_norm()` | 创建不学习仿射参数、不维护 running stats 的 InstanceNorm2d。 |
| `ResnetBlock.__init__()` | 构造反射 padding、卷积、归一化和 ReLU 的残差块。 |
| `ResnetBlock.forward()` | 返回输入与残差分支之和。 |
| `ResnetGenerator.__init__()` | 构造 2 次下采样、9 个残差块和 2 次上采样的全卷积生成器。 |
| `ResnetGenerator.encode_features()` | 运行编码器并抽取 5 层特征供 QS-Attn/ PatchNCE 使用。 |
| `ResnetGenerator.forward()` | 输出翻译图；可选同时返回中间特征。 |
| `PatchDiscriminator.__init__()` | 构造 70x70 PatchGAN 判别器。 |
| `PatchDiscriminator.forward()` | 输出空间真假分数图，而非单个标量。 |
| `init_weights()` | 以均值 0、标准差 0.02 初始化卷积和线性层。 |
| `lsgan_discriminator_loss()` | 计算 LSGAN 判别器的 real/fake 平方损失。 |
| `lsgan_generator_loss()` | 计算生成器欺骗判别器的 LSGAN 损失。 |

### `src/a2b_lightvpr/generator/objectives.py`

| 类/函数 | 职责 |
|---|---|
| `_sample_features()` | 从特征图采样相同空间位置，可复用位置 ID 对齐 source/fake。 |
| `_batch_patch_nce()` | 以同位置为正样本、其他 patch 为负样本计算温度缩放对比损失；key 分支 detach。 |
| `ContentObjective.generator_loss()` | 内容目标抽象接口。 |
| `ContentObjective.identity_loss()` | 默认不提供 identity 目标。 |
| `LazyProjectionHeads.__init__()` | 为 5 个特征层创建延迟确定输入维度的 MLP 投影头。 |
| `QSAttnObjective.__init__()` | 保存 patch 数并创建投影头。 |
| `QSAttnObjective._selected()` | 由 source 自相似注意力熵选择结构性 patch，并将同一注意力路由到 fake。 |
| `QSAttnObjective._feature_loss()` | 浅层随机同位置采样，深层使用 QS 选择，再平均五层 PatchNCE。 |
| `QSAttnObjective.generator_loss()` | 计算 source 到 fake 的内容保持损失。 |
| `QSAttnObjective.identity_loss()` | 可选计算夜图 identity NCE。 |
| `build_content_objective()` | 当前只允许构造 `QSAttnObjective`，拒绝历史目标。 |

### `src/a2b_lightvpr/generator/trainer.py`

| 函数 | 职责 |
|---|---|
| `_device()` | 优先 CUDA，否则 CPU。 |
| `_set_fixed_random_state()` | 将 Python、NumPy、Torch CPU/CUDA 全部固定为 0。 |
| `_objective_state()` | 取内容目标的 state dict。 |
| `_linear_decay()` | 学习率前半程恒定、后半程线性降为 0。 |
| `_checkpoint_payload()` | 组装可精确恢复的生成器训练状态。 |
| `train_generator()` | 创建数据、G/D/QS 目标与优化器；交替更新判别器和生成器；支持 AMP、恢复、`latest.pt` 覆盖写入和 `generator_final.pt` 推理导出。 |

一次生成器 step 的逻辑：

```text
source day -> G -> fake
target night --------> D(real)
fake.detach() -------> D(fake) -> 更新 D
fake ----------------> D(fake) -> GAN loss
source/fake features ----------> QS-Attn PatchNCE
GAN + content -----------------> 更新 G 和投影头
```

### `src/a2b_lightvpr/generator/translate.py`

| 函数 | 职责 |
|---|---|
| `validate_translation_checkpoint()` | 只接受 generator inference checkpoint；完整翻译要求其 step 等于配置总 step，`--limit` 仅用于 smoke test。 |
| `translate_training_set()` | 加载冻结生成器，逐张翻译 VPR 白天训练图，支持已存在图片跳过，写生成图、manifest、checkpoint SHA256 和翻译元数据。 |

## 9. VPR 模型实现

### `src/a2b_lightvpr/vpr/__init__.py`

子包标记文件，无逻辑。

### `src/a2b_lightvpr/vpr/model.py`

此文件同时保留正式 MixVPR 和参考/兼容聚合头，但当前论文训练主线是 `OfficialMixVPRModel`。

| 类/函数 | 职责 |
|---|---|
| `GeM.__init__()` | 创建带可学习幂次 `p` 的广义均值池化。 |
| `GeM.forward()` | 对空间特征做 clamp、幂、平均和开方。 |
| `GeMFC.__init__()` | 将 GeM 与线性投影组合成固定维度描述子。 |
| `GeMFC.forward()` | 池化、投影并 L2 归一化。 |
| `ConvAP.__init__()` | 用 1x1 卷积降维，再自适应池化。 |
| `ConvAP.forward()` | 输出展平且归一化的描述子。 |
| `FeatureMixer.__init__()` | 构造 MixVPR token mixing 的残差 MLP。 |
| `FeatureMixer.forward()` | 在空间 token 维混合特征并残差相加。 |
| `MixVPR.__init__()` | 构造通道投影、多层 FeatureMixer、行投影和最终归一化。128/512/4096 由通道数与 row 数决定。 |
| `MixVPR.forward()` | 将 `[B,C,H,W]` 特征变成 `[B,D]` 地点描述子。 |
| `_feature_grid()` | 用一次 dummy forward 推断 backbone 输出通道和空间网格。 |
| `_build_resnet50()` | 构造去掉分类尾部的 ResNet-50 backbone。 |
| `_build_vgg16()` | 构造去掉末端层的 VGG16 backbone，供 NetVLAD。 |
| `NetVLADPool.__init__()` | 创建可学习聚类中心和 soft assignment 卷积。 |
| `NetVLADPool.forward()` | 聚合每个 cluster 的残差并做 intra/global normalization。 |
| `NetVLADWPCA.__init__()` | 在 NetVLAD 后增加白化 PCA 卷积。 |
| `NetVLADWPCA.forward()` | 输出 4096 维归一化 NetVLAD 描述子。 |
| `DescriptorModel.__init__()` | 组合通用 backbone 与 GeM/ConvAP/MixVPR/NetVLAD 聚合器。 |
| `DescriptorModel.encode()` | backbone -> aggregator。 |
| `DescriptorModel.forward()` | `encode()` 的训练/推理统一入口。 |
| `build_descriptor_model()` | 按 model family、aggregator、尺寸和维度构造模型；官方 MixVPR 使用专门包装并严格加载初始化。 |
| `CroppedResNet50.__init__()` | 将 ResNet-50 截断到 MixVPR 官方使用的卷积层。 |
| `CroppedResNet50.forward()` | 输出截断后的特征图。 |
| `OfficialMixVPRModel.__init__()` | 按官方头映射 128=`64x2`、512=`256x2`、4096=`1024x4`，组合 CroppedResNet50 与 MixVPR。 |
| `OfficialMixVPRModel.encode()` | 提取并 L2 归一化描述子。 |
| `OfficialMixVPRModel.forward()` | 调用 `encode()`。 |
| `_checkpoint_state()` | 从不同 checkpoint 包装格式中提取 state dict 并清理常见前缀。 |
| `_strict_load_state()` | 严格加载官方 MixVPR 权重，任何 missing/unexpected key 都报错。 |
| `_strict_load_netvlad()` | 将官方 NetVLAD key 映射到本地结构并检查 backbone、pool、WPCA 完整。 |
| `eigenplaces_checkpoint()` | 只允许官方 GitHub URL 下载 EigenPlaces 权重，并写大小和 SHA256 元数据。 |
| `resolve_initial_checkpoint()` | 将 `eigenplaces://resnet50` 伪 URI 解析为本地官方 checkpoint。 |
| `load_initial_weights()` | 将 EigenPlaces/CosPlace 风格 key 映射到通用模型，返回 missing/unexpected 列表。 |

## 10. baseline 与 CDND

### `src/a2b_lightvpr/vpr/losses.py`

| 类/函数 | 职责 |
|---|---|
| `MarginCosineClassifier.__init__()` | 创建地点类别权重和缩放/角度 margin。 |
| `MarginCosineClassifier.forward()` | 归一化特征和类别权重，对真类减 margin，再缩放 cosine logits。 |
| `ikt_loss()` | 计算两个描述子相似矩阵的一致性损失，作为仍被测试覆盖的通用辅助函数。 |
| `day_fake_classification_loss()` | day 和 fake 分别过同一 margin classifier，平均两个交叉熵，并返回分项日志。 |
| `cross_domain_neighborhood_loss()` | 用 SALAD teacher logits 构造候选邻域概率，以 KL 蒸馏 fake->day 和 day->day；mask 同 image ID，并在无有效 anchor 时返回可反传的零损失。 |

`cross_domain_neighborhood_loss()` 的张量关系可概括为：

| 名称 | 典型形状 | 梯度 |
|---|---:|---|
| `teacher_anchor` | `[B,544]` | 无，来自 FP16 cache 转 FP32 |
| `teacher_candidates` | `[B+Q,544]` | 无 |
| `fake_anchor` | `[B,D]` | 有 |
| `day_anchor` | `[B,D]` | 有 |
| `student_day_candidates` | `[B+Q,D]` | 当前 B 有，历史 Q 无 |
| 邻域分布 | `[valid_B,B+Q]` | teacher 为目标，student 为预测 |

### `src/a2b_lightvpr/vpr/neighborhood.py`

| 类/方法 | 职责 |
|---|---|
| `NeighborhoodCandidates` | 打包 teacher、student day、label 和 image key 候选。 |
| `DatasetNeighborhoodQueue.__init__()` | 为每个训练数据集建立独立固定容量 FIFO。 |
| `DatasetNeighborhoodQueue.__len__()` | 返回全部数据集历史候选总数。 |
| `DatasetNeighborhoodQueue.size()` | 返回指定数据集队列长度。 |
| `DatasetNeighborhoodQueue._validate_dataset()` | 拒绝未知数据集，保证队列隔离。 |
| `DatasetNeighborhoodQueue.candidates()` | 将当前 batch 候选与指定数据集历史 FIFO 拼接；当前 student day 不 detach，历史项已经 detach。 |
| `DatasetNeighborhoodQueue.enqueue()` | 在 loss 构造后把 teacher/day/label/key detach 入队，并从左侧淘汰超容量历史。 |
| `DatasetNeighborhoodQueue.state_dict()` | 将各数据集 FIFO 搬到 CPU 保存。 |
| `DatasetNeighborhoodQueue.load_state_dict()` | 严格检查数据集、容量和 tensor 形状后恢复队列。 |

为什么 enqueue 在 backward 前发生仍然安全：入队保存的是 `.detach()` 副本；本 step 的计算图已经通过 `candidates()` 中未 detach 的当前 day tensor 建立。

### `src/a2b_lightvpr/vpr/teachers.py`

| 类/函数 | 职责 |
|---|---|
| `_load_dinov2()` | 从本地 DINOv2 仓库构造 ViT-B/14；CPU 时禁用 xFormers CUDA kernel。 |
| `_strict_load()` | 严格加载 SALAD checkpoint。 |
| `_log_sinkhorn()` | 做三次 log-space Sinkhorn 更新。 |
| `_optimal_transport()` | 为 SALAD cluster assignment 加 dustbin 并求最优传输。 |
| `_SaladBackbone.__init__()` | 包装本地 DINOv2 backbone。 |
| `_SaladBackbone.forward()` | 输出 patch feature map 和 CLS token。 |
| `_SaladHead.__init__()` | 创建 token 投影、局部特征、cluster score 和 dustbin。 |
| `_SaladHead.forward()` | 最优传输分配局部特征，拼接 token 与 cluster 描述子，输出 544 维归一化向量。 |
| `SaladTeacher.__init__()` | 组合 backbone/head 并严格加载完整 SALAD 权重。 |
| `SaladTeacher.forward()` | 生成 544 维描述子。 |
| `build_relation_teacher()` | 只接受 `salad`，返回 eval 且全参数冻结的教师。 |

### `src/a2b_lightvpr/vpr/teacher_cache.py`

| 类/函数 | 职责 |
|---|---|
| `_cache_paths()` | 从 `.npy` 主路径派生 `.meta.json` 和 `.ids.json`。 |
| `_ids_sha256()` | 对有序 image ID 列表生成哈希。 |
| `_expected_metadata()` | 生成 teacher、manifest、checkpoint、输入尺寸、维度、行数和 dtype 的期望元数据。 |
| `_TeacherImages.__init__()` | 保存训练行、SALAD 输入尺寸和恢复起点。 |
| `_TeacherImages.__len__()` | 返回尚未缓存的图像数。 |
| `_TeacherImages.__getitem__()` | 读取白天图，直接 resize 为 SALAD 方形输入并 ImageNet 归一化。 |
| `cache_relation_teacher()` | 构建冻结 SALAD，顺序写 FP16 NumPy memmap；每批更新完成行数，支持中断续写和严格指纹检查。 |
| `TeacherDescriptorCache.__init__()` | 训练前检查三件缓存文件、完整标记、manifest/权重/image IDs/shape/dtype，生成缓存 fingerprint。 |
| `TeacherDescriptorCache._array()` | 首次访问时以只读 mmap 延迟加载描述子。 |
| `TeacherDescriptorCache.descriptor()` | 按 manifest index 和 image ID 双重确认后返回 FP32 tensor。 |

### `src/a2b_lightvpr/vpr/trainer.py`

这是核心编排文件。

| 函数 | 职责 |
|---|---|
| `_set_fixed_random_state()` | 固定所有随机源为 0。 |
| `optimizer_base_lrs()` | 返回 backbone、aggregator、classifier 的基础学习率。 |
| `build_vpr_scheduler()` | 构造 cosine 或 ReduceLROnPlateau；constant 返回 `None`。 |
| `learning_rates_at_minimum()` | 判断所有参数组是否到达配置的最小比例。 |
| `accumulation_divisor()` | 正确处理 epoch 尾部不足完整 accumulation group 的 loss 除数。 |
| `freeze_batch_norm_statistics()` | 将所有 BatchNorm 切到 eval，冻结 running statistics。 |
| `configure_training_phase()` | warmup 时冻结官方 backbone/aggregator，只训练分类器；结束时恢复独立学习率。 |
| `build_vpr_optimizer()` | 用三个参数组创建 AdamW。 |
| `_inference_checkpoint()` | 只保存模型及部署必要元数据。 |
| `_vpr_checkpoint()` | 保存模型、分类器、optimizer、scheduler、scaler、RNG、sampler cursor、FIFO 和 samples_seen。 |
| `_evaluate_validation()` | 在 RobotCar val 上调用通用评测，供选 best checkpoint。 |
| `export_vpr_inference_checkpoint()` | 从 training state 精确剥离训练资产，拒绝覆盖已有输出。 |
| `train_vpr()` | 创建数据、采样器、模型、分类器、优化器、scheduler、teacher cache 和 FIFO；恢复状态；执行 baseline/CDND loss、AMP、梯度累积、验证、best/latest 保存与停止规则。 |

一次 CDND step 的实际调用链：

```text
PairedVPRDataset
  -> day, fake, label, image_key, dataset, teacher_descriptor
  -> model.encode(fake), model.encode(day)
  -> day_fake_classification_loss
  -> relation_queue.candidates(dataset, current batch, history)
  -> cross_domain_neighborhood_loss
  -> total loss / accumulation divisor
  -> backward
  -> relation_queue.enqueue(detached history)
  -> 到累积边界后 optimizer.step
```

恢复训练时同时检查 config、teacher cache fingerprint、sampler 状态和 FIFO 状态。只恢复模型权重但不恢复这些状态，不能称为精确续训。

## 11. 单帧评测与汇总

### `src/a2b_lightvpr/evaluation.py`

| 函数 | 职责 |
|---|---|
| `circular_yaw_difference()` | 计算跨 0/360 度边界的最小角差。 |
| `positive_indices()` | 按坐标半径和朝向阈值为每个 query 构造数据库正样本索引。 |
| `retrieval_metrics()` | 内积检索并计算 Recall@K、MRR、有效 query 和逐 query 成功标记；无 FAISS 时回退 NumPy。 |
| `extract_descriptors()` | 批量编码 `ImagePathDataset`，按原索引写入 NumPy 数组。 |
| `evaluate_model()` | 对 RobotCar 指定 split 完成 database/query 编码、正样本构造、指标计算和 JSON 写入。 |

没有任何正样本的 query 被明确标为 invalid，并从 Recall 分母排除，不会被悄悄算成失败。

### `src/a2b_lightvpr/trained_suite.py`

`SUITE_PROTOCOLS` 定义当前夜间开发协议和白天控制协议，包括 RobotCar、SVOX、MSLS、Tokyo。

| 函数 | 职责 |
|---|---|
| `expand_suite_datasets()` | 展开 `all`、`all-day`、`all-conditions` 别名。 |
| `_evaluation_rows()` | 通过参考数据协议层取 database/query 行。 |
| `_suite_summary_values()` | 将嵌套 suite 结果展平成便于记录的键值表。 |
| `evaluate_trained_suite()` | 严格加载训练模型，按协议评测；用 checkpoint/manifest/protocol/evaluation fingerprint 安全复用缓存；附加 benchmark 并写 `final_suite/summary.json`。 |

### `src/a2b_lightvpr/single_run_summary.py`

| 函数 | 职责 |
|---|---|
| `_load_suite()` | 加载一个维度的 suite，检查描述子维度和协议完整性。 |
| `_fingerprint()` | 读取协议/评测 fingerprint。 |
| `_macro()` | 对指定协议的 R@1 做算术平均。 |
| `_resources()` | 整理参数、checkpoint、延迟、显存和每百万图片描述子存储。 |
| `_validate_pair()` | 确保 baseline/CDND 的模型、聚合器、输入、manifest 和初始化完全配对。 |
| `summarize_cdnd_seed0()` | 汇总 128/512/4096 各一次 baseline/CDND，输出逐协议差值、夜/日宏平均和相对 512 的容量资源差；不计算均值、方差或显著性。 |

### `src/a2b_lightvpr/reference_suite.py`

| 函数 | 职责 |
|---|---|
| `evaluate_reference_suite()` | 读取参考模型注册表，将模型名、database encoder、transform 和协议选择转交参考模型评测，并管理输出目录。 |

## 12. 参考模型

### `src/a2b_lightvpr/reference_models.py`

该文件负责 CosPlace、EigenPlaces、NetVLAD、NocPlace、SALAD、BoQ、CricaVPR 等预训练模型的统一加载与协议对齐。它不是 CDND 训练教师实现；CDND 教师由 `vpr/teachers.py` 单独负责。

| 类/函数 | 职责 |
|---|---|
| `SuiteRow` | 用统一字段表示非 RobotCar 数据集的一张评测图及坐标/朝向。 |
| `query_protocol_fingerprint()` | 对有序 database/query 身份、坐标、阈值生成协议哈希。 |
| `evaluation_fingerprint()` | 在协议哈希上加入 transform、尺寸和 padding，区分不同预处理评测。 |
| `FixedResizeTransform.__init__()` | 保存参考模型原生输入尺寸。 |
| `FixedResizeTransform.__call__()` | 直接 resize 并 ImageNet 归一化，不使用 fit-and-pad。 |
| `ReferenceAdapter.__init__()` | 包装不同仓库模型，声明统一 descriptor dimension。 |
| `ReferenceAdapter.encode()` | 兼容 tensor、tuple、dict 等输出并做 L2 归一化。 |
| `ReferenceAdapter.forward()` | 调用 `encode()`。 |
| `_registry()` | 读取参考模型 YAML 的模型注册表。 |
| `audit_reference_models()` | 检查本地 repo/checkpoint 是否存在，记录大小和 SHA256。 |
| `_download_with_resume()` | 通过 HTTP Range 续传到 `.part`，完成后原子改名。 |
| `fetch_reference_model()` | 只按注册表显式下载指定模型并校验 hash/size。 |
| `_strict_state()` | 从参考 checkpoint 提取严格 state dict。 |
| `_isolated_repo_import()` | 临时调整 `sys.path`/模块环境，隔离导入第三方仓库，避免同名模块污染。 |
| `build_reference_model()` | 按注册类型构造各参考模型，加载权重并包装为统一 adapter。 |
| `_reference_checkpoint()` | 解析参考模型 checkpoint 路径。 |
| `benchmark_reference_models()` | 对一个或全部参考模型测参数、权重大小、原生/统一尺寸延迟和显存。 |
| `_svox_rows()` | 将 SVOX 文件名解析成统一 database/query 行。 |
| `_msls_rows()` | 从 MSLS 元数据构造评测行。 |
| `_tokyo_rows()` | 从 Tokyo 目录和坐标信息构造评测行。 |
| `_robotcar_day_rows()` | 构造冻结的 RobotCar 白天控制协议。 |
| `suite_rows()` | 根据数据集别名分派到 RobotCar/SVOX/MSLS/Tokyo 行构造器。 |
| `evaluate_dual_encoder()` | 分别编码 database/query，计算统一指标并附加 protocol/evaluation fingerprint。 |

`database_encoder=original/adapted` 允许评估“原始数据库编码器 + 适配 query 编码器”或统一适配编码器。论文表格比较前必须确认此字段、transform 和 protocol fingerprint 一致。

## 13. RobotCar 时序重排

### `src/a2b_lightvpr/robotcar_temporal.py`

该模块只接受 512 维 checkpoint。它保存 raw top-100 是为了缓存和条件回退；TCR 的地点簇由窗口内 raw top-1 形成。

#### 参数与缓存

| 类/函数 | 职责 |
|---|---|
| `ClusteredTemporalParameters` | TCR 参数：奇数居中窗口、top-10、簇间隔和中心半径。 |
| `_condition_rows()` | 将 `val/night/day` 映射到冻结 RobotCar 协议行。 |
| `ordered_motion_deltas()` | 每个 traverse 内按 timestamp 排序，只输出顺序、序列起点和相邻标量位移；不向重排器传绝对 query 坐标。 |
| `_top_scores()` | 用 FAISS 或 NumPy 生成 raw top-100 索引和相似度。 |
| `_atomic_npz()` | 原子写压缩 NumPy 缓存。 |
| `_cache_metadata_path()` | 从 `.npz` 派生缓存 JSON 路径。 |
| `_expected_cache_metadata()` | 生成 checkpoint、manifest、协议、维度、输入尺寸和行数的期望缓存身份。 |
| `build_robotcar_temporal_cache()` | 严格检查 512 维，提取/复用描述子，生成 top-100、数据库坐标/朝向、query 增量和资源元数据。 |
| `load_robotcar_temporal_cache()` | 校验 NPZ SHA256 后把数组安全载入内存。 |

#### TCR

| 函数 | 职责 |
|---|---|
| `_database_frame_positions()` | 在每个 database traverse 内按 timestamp 编号，返回序号和 route ID。 |
| `_chain_clusters()` | 按 route 隔离，将序号间隔不超过阈值的 top-1 假设连成簇。 |
| `rerank_clustered_temporal()` | 取目标帧两侧的居中奇数窗口，对各帧 raw top-1 聚类；选择支持最大且均分更高的簇；在目标帧 top-10 中寻找簇中心附近候选，必要时才扫描 top-100。 |

TCR 使用未来帧，窗口 `1/3/5/7/9` 对应中心延迟 `0/1/2/3/4` 帧。窗口在 traverse 边界被截断，绝不跨序列。

#### 指标、选参与汇总

| 函数 | 职责 |
|---|---|
| `_positive_indices()` | 为时序完整距离/朝向网格构造正样本。 |
| `ranking_metrics()` | 从已有排名算 R@1/R@5/R@10，并支持额外 query mask。 |
| `map_covered_center_mask()` | 只标记完整 9 帧窗口均有 10m+30度数据库覆盖的中心帧；仅用于诊断，不传入重排器。 |
| `evaluate_grid()` | 评测距离 5/10/25/50m 与朝向 10/30/不限制的完整网格。 |
| `evaluate_grid_masked()` | 在 map-covered mask 上评测同一网格。 |
| `_validate_cache_for_run()` | 核对 condition、维度、checkpoint 和 manifest。 |
| `select_robotcar_temporal()` | 只读 val：TCR 比较窗口 1/3/5/7/9，并冻结选择文件。 |
| `temporal_resource_summary()` | 报告模型、checkpoint、编码、检索、CPU 重排、描述子存储和滑窗状态资源。 |
| `evaluate_robotcar_temporal()` | 用冻结 selection 一次评测 night/day；输出单帧与 TCR 完整网格、诊断和资源。 |
| `_reference_robotcar()` | 从参考模型 suite 中提取 RobotCar 指标和协议身份。 |
| `summarize_robotcar_temporal()` | 验证 night/day/参考模型协议，汇总 MixVPR single/TCR、CricaVPR、SALAD、BoQ 及验收判断。 |

## 14. CLI

### `src/a2b_lightvpr/cli.py`

| 函数 | 职责 |
|---|---|
| `_parser()` | 声明全部子命令、公共 `--config/--set` 和各命令参数；被删除的旧命令不会被解析。 |
| `_dimensions()` | 将重复的 `DIM=JSON` 参数解析为无重复维度字典。 |
| `main()` | 分发命令；无需 ExperimentConfig 的汇总/下载命令直接执行，其余命令先 load+override+validate。 |

当前命令与入口函数：

| CLI | 入口 |
|---|---|
| `prepare-data` | `prepare.build_manifests` |
| `audit-data` | `data.audit_data` |
| `train-generator` | `generator.trainer.train_generator` |
| `translate-train` | `generator.translate.translate_training_set` |
| `cache-vpr-teacher` | `vpr.teacher_cache.cache_relation_teacher` |
| `train-vpr` | `vpr.trainer.train_vpr` |
| `export-vpr-inference` | `vpr.trainer.export_vpr_inference_checkpoint` |
| `evaluate-trained-suite` | `trained_suite.evaluate_trained_suite` |
| `benchmark` | `runtime.benchmark_model` |
| `summarize-cdnd-seed0` | `single_run_summary.summarize_cdnd_seed0` |
| `audit-reference-models` | `reference_models.audit_reference_models` |
| `fetch-reference-model` | `reference_models.fetch_reference_model` |
| `benchmark-reference-models` | `reference_models.benchmark_reference_models` |
| `evaluate-reference-suite` | `reference_suite.evaluate_reference_suite` |
| `select-robotcar-temporal` | `robotcar_temporal.select_robotcar_temporal` |
| `evaluate-robotcar-temporal` | `robotcar_temporal.evaluate_robotcar_temporal` |
| `summarize-robotcar-temporal` | `robotcar_temporal.summarize_robotcar_temporal` |

## 15. 工作流脚本

### `scripts/run_seed0_all.fish`

主训练/评测总入口。它依次：

1. 检查 CUDA、磁盘、配置和四个初始化/教师权重 SHA256。
2. 运行完整 pytest。
3. 检查或恢复 QS-Attn。
4. 检查或生成 118,313 张 fake-night 对应 manifest 行。
5. 检查或生成 SALAD FP16 cache。
6. 训练 128/512/4096 baseline。
7. 训练 128/512/4096 CDND。
8. 评测开发协议并 benchmark。
9. 验证每次训练 `samples_seen=1,198,720`。
10. 输出 `runs/comparisons/cdnd_seed0_summary.json`。

内部函数：

| Fish 函数 | 职责 |
|---|---|
| `die` | 打印错误并退出。 |
| `cli` | 调用 `python -m a2b_lightvpr.cli`。 |
| `verify_sha256` | 检查必需 checkpoint 内容哈希。 |
| `latest_checkpoint` | 找运行目录中最新训练状态。 |
| `vpr_complete` | 同时检查 best/latest，并核对 samples_seen。 |
| `train_vpr_once` | 已完成则跳过，有 latest 则恢复，否则开始训练。 |
| `evaluate_vpr_once` | 复用或生成 final suite 与 benchmark。 |

### `scripts/run_robotcar_temporal_512.fish`

512-CDND RobotCar 时序总入口。它先跑时序专项测试，再完成 TCR val 选窗、night/day 评测和系统比较。

内部函数：

| Fish 函数 | 职责 |
|---|---|
| `die` | 打印错误并退出。 |
| `cli` | 固定使用项目 Python 环境调用 CLI。 |
| `valid_json` | 检查结果 schema、checkpoint/manifest/selection SHA256，决定是否安全跳过。 |

`--force` 会忽略已有合法 JSON 并重跑评测阶段，但不会改变模型维度或 checkpoint 固定 SHA256。

## 16. 测试文件逐项说明

测试是理解隐含契约的最好入口。以下列出每个测试文件和其中每个测试函数。

### `tests/test_cdnd.py`

- `_rows()`：构造多数据集、多地点的合成训练行。
- `test_pk_sampler_builds_single_dataset_four_by_two_batches_and_restores_cursor()`：验证 P=4/K=2、单数据集 batch、不同图像和 cursor 恢复。
- `test_dataset_queue_is_fifo_isolated_and_round_trips()`：验证 FIFO 容量、数据集隔离和序列化恢复。
- `test_cdnd_masks_self_and_backpropagates_only_into_student_descriptors()`：验证 self-mask 与 teacher/student 梯度边界。
- `test_cdnd_returns_differentiable_zero_without_valid_anchors()`：验证空 anchor 仍可 backward。
- `test_sampler_and_queue_restore_produce_identical_next_relation_loss()`：验证恢复后下一批和关系 loss 一致。

### `tests/test_checkpointing.py`

- `test_rng_state_round_trip_restores_torch_sequence()`：验证 RNG 保存恢复后 Torch 随机序列一致。

### `tests/test_clean_weather.py`

- `test_weather_audit_rejects_corrupt_image_before_clip()`：坏图必须在 CLIP 前被拒绝。
- `test_weather_audit_reuses_matching_cache_without_loading_clip()`：相同指纹应直接复用天气缓存。

### `tests/test_config.py`

- `base_config()`：生成最小合法配置夹具。
- `test_only_official_qsattn_objective_is_allowed()`：拒绝历史生成目标。
- `test_public_config_rejects_seed_fields()`：公开配置不接受 seed。
- `test_resume_paths_do_not_change_experiment_fingerprint()`：恢复路径不改变实验身份。
- `test_formal_dimension_configs_are_strictly_paired()`：六个正式 VPR 配置除方法/维度必要字段外严格配对。
- `test_only_salad_can_be_the_training_teacher()`：CDND 训练教师只能是 SALAD。
- `test_reference_config_still_loads()`：参考模型配置保持可读。
- `test_official_generator_config_has_no_public_seed()`：生成器正式配置也无 seed。

### `tests/test_data_audit.py`

- `write_fixture()`：写合成 RobotCar manifest/protocol。
- `test_audit_accepts_disjoint_traversals()`：接受 train/val/test traverse 隔离。
- `test_audit_rejects_traversal_leakage()`：拒绝 traverse 泄漏。
- `test_vpr_training_rejects_non_train_generated_rows()`：VPR manifest 不能含非 train 行。
- `test_vpr_dataset_sampling_weights_match_config()`：平衡权重符合配置。
- `test_frozen_extended_place_statistics()`：冻结正式数据地点/行数统计。
- `test_svox_filename_records_camera_direction()`：SVOX place ID 保留相机方向。
- `test_rolling_sampler_is_balanced_unique_and_resume_deterministic()`：滚动采样平衡、唯一且确定恢复。
- `test_rolling_sampler_repeats_only_after_small_dataset_is_covered()`：小数据集覆盖后才重复。
- `test_rolling_sampler_windows_stay_unique_across_many_cycle_boundaries()`：跨多周期窗口仍保持唯一性。

### `tests/test_evaluation.py`

- `test_retrieval_metrics_known_ranking()`：用手工排名校验 Recall/MRR。
- `test_invalid_queries_are_explicit_and_excluded_from_recall()`：无正样本 query 明确排除。

### `tests/test_generator_objectives.py`

- `test_qsattn_content_objective_produces_scalar_gradient()`：QS-Attn 输出标量且可反传。
- `test_qsattn_identity_nce_is_finite_and_differentiable()`：identity NCE 有限且可反传。
- `test_patch_discriminator_is_spatial()`：判别器输出 Patch 网格。

### `tests/test_reference_models.py`

- `test_reference_registry_reports_ready_and_missing_models()`：审计能区分已准备/缺失模型。
- `test_local_svox_protocol_golden_counts()`：冻结 SVOX 夜间行数。
- `test_local_msls_night_protocol_golden_counts()`：冻结 MSLS 夜间行数。
- `test_local_tokyo_night_and_sunset_are_not_empty()`：Tokyo 夜间/黄昏协议非空。
- `test_local_svox_day_protocol_golden_counts()`：冻结 SVOX day val 行数。
- `test_local_msls_day_protocol_golden_counts()`：冻结 MSLS day val 行数。
- `test_local_tokyo_day_protocol_golden_counts()`：冻结 Tokyo day 行数。
- `test_robotcar_day_uses_fixed_untrained_traversals()`：白天控制集使用未训练 traverse。
- `test_day_validation_queries_do_not_overlap_vpr_training_manifest()`：白天 query 不与训练 manifest 重合。
- `test_native_and_unified_transforms_have_declared_shapes()`：两种预处理尺寸符合声明。
- `test_dual_encoder_uses_distinct_database_and_query_models()`：双编码器模式确实分别调用模型。
- `test_fetch_reference_model_is_explicit_and_records_hash()`：下载必须显式且记录哈希。
- `test_fetch_reference_model_resumes_partial_download()`：支持 `.part` 续传。
- `test_reference_benchmark_reports_storage_and_batch_dependence()`：资源报告包含存储和 batch 依赖。

### `tests/test_reference_suite.py`

- `test_reference_suite_uses_shared_protocol_and_cache()`：参考模型与训练模型共享协议身份并安全复用缓存。

### `tests/test_robotcar_temporal.py`

- `row()`：构造合成时序行。
- `test_motion_deltas_are_translation_invariant_and_traverse_isolated()`：绝对坐标平移不改变增量，traverse 间隔离。
- `test_spatial_nms_suppresses_same_view_but_keeps_opposite_direction()`：同位置同向合并、反向保留。
- `test_topk_path_recovers_consistent_candidate_when_raw_top1_jumps()`：动态规划可从 top-K 修正跳点。
- `test_impossible_transition_falls_back_to_raw_top1()`：无合法转移回退单帧。
- `test_shorter_window_is_used_when_oldest_transition_is_invalid()`：长窗口失败时缩窗。
- `test_clustered_mgtr_uses_center_window_and_top100_fallback()`：居中聚类及受限 top-100 回退。
- `test_clustered_windows_are_odd_and_bounded()`：只接受 1/3/5/7/9。
- `test_map_covered_protocol_uses_common_full_window()`：covered 诊断统一使用完整 9 帧窗口。
- `test_clustered_center_window_never_crosses_sequence()`：窗口不跨序列。
- `test_selection_tie_break_is_fully_deterministic()`：平局规则确定。
- `test_selection_is_hard_limited_to_val()`：选参禁止读取 night/day。
- `test_selection_rejects_checkpoint_or_manifest_mismatch()`：selection/cache 指纹不匹配即拒绝。
- `test_resource_accounting_is_amortized_one_new_encoding()`：滑窗稳态每次只新增一次编码。
- `test_dimension_guard_rejects_unsupported_dimension_before_model_loading()`：128/4096 在加载模型前即被拒绝。
- `test_temporal_resource_summary_uses_actual_descriptor_dimension()`：资源公式读取真实维度。
- `test_temporal_cli_exposes_selection_evaluation_and_summary()`：三个 CLI 存在且参数完整。
- `test_summary_checks_protocol_and_applies_acceptance_rules()`：汇总检查协议并应用验收条件。

### `tests/test_single_run_summary.py`

- `_suite()`：构造合成 suite。
- `test_single_run_summary_reports_deltas_without_statistics()`：只报告固定运行差值，不生成多 seed 统计。
- `test_single_run_summary_rejects_dimension_and_protocol_mismatches()`：拒绝维度/协议错配。
- `test_single_run_summary_rejects_non_paired_inputs()`：拒绝非配对 baseline/CDND。

### `tests/test_single_run_workflow.py`

- `test_clustered_workflow_contract()`：检查时序脚本包含 TCR 固定流程。
- `test_cli_rejects_removed_method()`：已删除时序方法无法解析。
- `test_cli_rejects_removed_gardens_commands()`：已删除 Gardens CLI 无法解析。

### `tests/test_teacher_cache.py`

- `FakeTeacher.forward()`：为测试生成确定性教师描述子。
- `make_config()`：构造临时 cache 配置。
- `test_teacher_cache_builds_and_resumes()`：验证 memmap 分批构建与续写。
- `test_training_cache_checks_ids_and_completion()`：训练加载器检查 ID 和 complete 状态。

### `tests/test_trained_suite.py`

- `test_suite_aliases_cover_active_conditions()`：检查 all/day/all-conditions 展开。
- `_fixture_config()`：构造 suite 配置夹具。
- `test_trained_suite_writes_protocol_bound_metrics()`：结果绑定协议与 evaluation fingerprint。
- `test_trained_suite_reuses_matching_summary()`：身份相同才复用 summary。

### `tests/test_transforms.py`

- `image()`：生成测试图。
- `test_fit_pad_preserves_full_aspect_ratio()`：VPR padding 不裁掉原图。
- `test_generator_resize_is_unpadded_and_multiple_of_four()`：生成器 resize 无 padding 且尺寸适配网络。
- `test_paired_generator_transform_shares_geometry()`：成对增强共享裁剪/翻转参数。

### `tests/test_translate.py`

- `test_full_translation_requires_matching_final_checkpoint()`：完整翻译必须用配置的最终 step。
- `test_smoke_translation_rejects_wrong_objective()`：即使 smoke 也拒绝目标不匹配。

### `tests/test_vpr.py`

- `test_resnet50_aggregators_honor_512_contract()`：通用聚合器输出 512 维。
- `test_mixvpr_matches_official_implementation()`：本地 MixVPR 与官方实现逐元素一致。
- `test_mixvpr_descriptor_dimensions()`：验证 128/512/4096 输出形状。
- `test_mixvpr_initialization_checkpoints_load_strictly()`：三种官方初始化严格加载。
- `test_netvlad_pool_output_contract()`：NetVLAD 输出结构正确。
- `test_eigenplaces_download_records_official_metadata()`：官方下载写来源/hash 元数据。
- `test_classifier_cpu_offload_supports_amp_optimizer_step()`：分类器 CPU offload 与 AMP 更新兼容。
- `test_4096_classifier_amp_accumulation_resume_stays_on_gpu()`：4096 分类器梯度累积恢复设备正确。
- `test_day_fake_classification_loss_is_finite()`：baseline loss 有限。

### `tests/test_vpr_deployment_checkpoint.py`

- `test_training_checkpoint_conversion_is_model_exact_and_strips_training_assets()`：导出模型逐元素一致且剥离训练资产。
- `test_deployment_loader_rejects_checkpoint_dimension_mismatch()`：部署加载拒绝维度错配。

### `tests/test_vpr_scheduler.py`

- `test_plateau_scheduler_reduces_and_respects_minimum()`：plateau 按规则降学习率且不越过下限。
- `test_plateau_scheduler_state_resumes()`：scheduler 状态可恢复。
- `test_convergence_requires_all_learning_rates_at_minimum()`：所有参数组到底才算满足收敛停止条件。
- `test_partial_gradient_accumulation_uses_actual_group_size()`：尾部 accumulation 除数正确。
- `test_optimizer_uses_independent_learning_rates()`：三个参数组学习率独立。
- `test_official_warmup_only_trains_classifier_then_unfreezes()`：官方模型 warmup 冻结/解冻正确。
- `test_frozen_batch_norm_keeps_running_statistics()`：冻结 BN 时统计量不变。
- `test_cosine_scheduler_reaches_configured_ratio_after_main_budget()`：cosine 在主预算末达到最小比例。

## 17. 关键文件之间的调用关系

```text
cli.py
├── prepare.py ── clean_weather.py
├── generator/trainer.py
│   ├── data.BalancedGeneratorDataset
│   ├── generator/networks.py
│   └── generator/objectives.py
├── generator/translate.py
├── vpr/teacher_cache.py ── vpr/teachers.py
├── vpr/trainer.py
│   ├── data.PairedVPRDataset + PKBalancedBatchSampler
│   ├── vpr/model.py
│   ├── vpr/losses.py
│   ├── vpr/neighborhood.py
│   └── evaluation.py
├── trained_suite.py ── reference_models.suite_rows
├── reference_suite.py ── reference_models.py
└── robotcar_temporal.py
    ├── runtime.load_vpr_model
    ├── evaluation.extract_descriptors
    └── reference_models.suite_rows
```

## 18. Checkpoint 与产物

| 产物 | 包含 | 不包含/用途 |
|---|---|---|
| generator `checkpoints/latest.pt` | G、D、QS 投影头、优化器、scheduler、scaler、RNG | 仅恢复训练 |
| `generator_final.pt` | G 权重、step、目标和数据指纹 | 用于离线翻译 |
| VPR `checkpoints/latest.pt` | 模型、分类器、优化器、scheduler、scaler、RNG、sampler、FIFO、samples_seen | 精确恢复训练 |
| VPR `model_best.pt` | 学生模型、维度、验证指标和数据指纹 | 推理/评测 |
| SALAD `.npy` cache | 有序 FP16 544 维白天描述子 | 仅 CDND 训练 |
| `final_suite/summary.json` | 各协议指标、资源和指纹 | 单帧汇总输入 |
| temporal `*_top100.npz` | top-100、DB 坐标/朝向、query 增量、序列起点 | 避免重复图像编码 |
| temporal selection JSON | 只由 val 冻结的参数和身份 | night/day 评测必须匹配 |

## 19. 修改项目时的常见入口

| 想修改的内容 | 首先阅读 | 同时更新 |
|---|---|---|
| 新增 VPR 维度 | `vpr/model.py` | `config.py`、正式配置、模型与部署测试 |
| 修改 CDND 公式 | `vpr/losses.py` | `vpr/trainer.py`、`test_cdnd.py`、论文公式 |
| 修改候选队列 | `vpr/neighborhood.py` | checkpoint 恢复测试和梯度测试 |
| 修改 batch 构造 | `data.py` | quota 配置、恢复测试和样本预算检查 |
| 更换 teacher | `vpr/teachers.py` | cache 元数据、config 校验、许可证与论文描述 |
| 新增评测协议 | `trained_suite.py` 与 `reference_models.py` | protocol fingerprint、summary、测试 |
| 修改 TCR | `robotcar_temporal.py` | val-only 选窗、缓存 schema、时序测试和脚本 |
| 修改命令 | `cli.py` | README、Fish 脚本和 CLI 契约测试 |

## 20. 最容易误解的边界

1. **fake-night 不是测试数据。** 它只用于训练；正式夜间 query 来自真实评测集。
2. **SALAD 不参与最终定位。** 它只离线编码训练白天图，CDND 后部署只有 MixVPR。
3. **CDND 不是逐样本 feature matching。** 它匹配候选集合上的邻域概率分布。
4. **当前 batch day candidate 有梯度。** 只有 FIFO 中的历史 student candidate 在入队时 detach。
5. **同 image ID 必须 mask。** 否则 teacher/student 都能靠自身相似度获得虚假的低 KL。
6. **不同数据集不能共享 FIFO 或 mini-batch。** place ID 虽有 dataset 前缀，邻域统计仍应保持数据集内语义。
7. **固定状态 0 不等于多次稳定性。** 当前结果只能描述为单次 development result，不能声称统计显著。
8. **TCR 使用顺序辅助信息。** 它与 SALAD 单帧结果是系统级比较，不是同条件时序算法比较。
9. **TCR 使用未来帧。** 居中窗口会带来明确的帧延迟；TCR 则只用历史帧。
10. **map-covered 是诊断 mask。** 完整 night 协议仍是主结果，mask 不得进入重排算法。

## 21. 两条正式运行命令

完整单次训练与单帧评测：

```bash
fish scripts/run_seed0_all.fish
```

已有 512-CDND checkpoint 后运行 RobotCar 时序评测：

```bash
fish scripts/run_robotcar_temporal_512.fish
```

这两条命令都包含耗时阶段。阅读或修改代码时应优先运行对应的小范围 pytest；不要为了验证文档或轻微重构直接启动完整训练。
