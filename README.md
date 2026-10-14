# A2B-LightVPR

用于昼夜视觉地点识别的研究项目：QS-Attn 离线生成 fake-night 图像，SALAD 教师通过 CDND 监督轻量 MixVPR 学生。部署 checkpoint 只包含学生模型。

## 目录

```text
configs/
├── generator/                 # QS-Attn 生成器
├── vpr/                       # MixVPR 主实验，128 / 512 / 4096 维
├── references/                # 官方与重新训练的参考模型注册表
└── experiments/
    ├── single_dataset/        # RobotCar、SVOX、MSLS 单数据集消融
    ├── data_sources/          # GSV、多生成器数据源实验
    └── students/              # SALAD-S、BoQ-S、DINO-Mix 学生实验
scripts/
├── data/                      # 数据准备、校验和 manifest 合并
├── train/                     # 主实验、多种子与组件消融
├── evaluate/                  # 单帧评测与同权重评测
├── experiments/               # 扩展实验工作流
├── references/                # 参考模型训练、PCA 和词典拟合
├── benchmark/                 # 延迟与资源测量
└── report/                    # 图表与定性检索导出
src/a2b_lightvpr/
├── generator/                 # 生成器网络、损失、训练与翻译
├── vpr/                       # 学生模型、CDND、教师缓存与训练
├── references/                # 参考模型加载、训练、拟合与评测
├── reporting/                 # 单种子、多种子、配对实验汇总与绘图
├── qualitative/               # 定性样本搜索与图片导出
└── *.py                       # CLI、配置、数据、检索指标、模型运行
tests/                        # 单元测试与可选本地资源检查
docs/                         # 模型设计与资源路径迁移说明
data/                         # 本地数据、生成图像与缓存（不入 Git）
runs/                         # 权重、评测结果、日志与图表（不入 Git）
```

## 安装与检查

需要 Python 3.10+；完整训练需要 CUDA、数据集和第三方权重。所有命令均在项目根目录执行。

推荐在 Linux NVIDIA GPU 机器上使用 [environment.yml](environment.yml)：Python 3.12、PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.4，版本组合与 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v251) 一致。文件包含训练、评估、测试、绘图、小参考模型依赖及 fish 脚本运行工具。

```bash
conda env create -f environment.yml
conda activate a2b-lightvpr
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
a2b-lightvpr --help
```

CUDA wheel 自带运行时，GPU 训练仍需要兼容的 NVIDIA 驱动。环境文件不安装数据集、DINOv2 仓库或模型权重；这些资源仍通过实验 YAML 配置。xFormers 不作为必需依赖。其他平台或已有 Python 环境可直接安装项目：

```bash
python -m pip install -e '.[test,eval]'
python -m a2b_lightvpr.cli --help
python -m pytest
```

绘图使用 `.[paper]`，小参考模型使用 `.[small-reference]`。第三方训练包装脚本还依赖对应官方仓库及其环境。

YAML 与脚本中的资源路径已按当前工作区迁移：评测数据位于 `/home/admin123/github/datasets`，GSV-Cities 位于 `../WORKSPACE/dataset/gsv-cities`，已有 DINOv2 仓库与权重使用本机 Torch Hub 缓存。RobotCar 测试统一使用公共 Seasons；旧自建测试入口、配置字段和协议解析已移除。原 RobotCar 训练帧与部分第三方资源仍有缺口。见 [资源迁移与缺口](docs/resource_migration.md)。项目内部 `data/`、`runs/` 路径相对于项目根目录。脚本默认使用当前环境的 `python`，也可以通过 `PYTHON_BIN=/path/to/python` 指定解释器。

## 常用入口

| 工作 | 命令 |
|---|---|
| 完整主实验 | `fish scripts/train/run_main_seed0.fish` |
| 512 维多种子与消融 | `fish scripts/train/run_paper_experiments_512.fish` |
| RobotCar 单数据集消融 | `fish scripts/experiments/run_robotcar_only.fish all` |
| SVOX / MSLS 单数据集消融 | `fish scripts/experiments/run_single_dataset_cdnd.fish svox`（或 `msls`） |
| 参考模型补充评测 | `fish scripts/references/run_reference_completion.fish` |
| 定性检索图 | `fish scripts/report/run_qualitative_top1.fish` |
| 论文图表 | `fish scripts/report/run_paper_figures.fish` |
| 九模型延迟测量 | `fish scripts/benchmark/run_nine_method_latency.fish` |

单独训练或评测使用统一 CLI，例如：

```bash
a2b-lightvpr cache-vpr-teacher --config configs/vpr/cdnd_salad_512.yaml
a2b-lightvpr train-vpr --config configs/vpr/cdnd_salad_512.yaml
```

`checkpoints/latest.pt` 用于恢复同一实验，`model_best.pt` 用于推理，两者不可混用。VPR checkpoint 选择使用公开白天验证 split（MSLS/SVOX），不使用 test。旧完整推理 checkpoint 可继续评测；旧 VPR optimizer checkpoint 因验证配置变化不能直接续训，应使用新配置与输出目录开始新实验。历史产物中的路径、模型标识和指纹保留以便追溯。

项目的 VPR 和 QS-Attn 训练默认写入 `output_dir/tensorboard/`。在项目根目录另开终端执行 `tensorboard --logdir runs --port 6006`，浏览器打开 <http://localhost:6006>，即可比较训练损失、学习率和已运行的验证指标。配置、StableNight 诊断和恢复训练的日志边界见 [TensorBoard 训练监控](docs/training_monitoring.md)。

StableNight-SALAD 的设计、配置和验证边界见 [模型说明](docs/stable_night_salad.md)，当前数据、权重和第三方仓库位置及剩余缺口见 [资源迁移说明](docs/resource_migration.md)。

RobotCar Seasons 与其他测试集共用 suite 入口：

```bash
a2b-lightvpr evaluate-trained-suite \
  --config configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml \
  --checkpoint runs/vpr/stable_night_salad_544/msls_svox_day_aug_seed0/model_best.pt \
  --datasets robotcar_seasons_night robotcar_seasons_day
```

`evaluate-reference-suite` 支持相同 key。`--datasets all` 运行夜间组，`all-day` 运行白天与白天验证组，`all-conditions` 运行两组；两种 Database 均使用现有 `evaluation.database_domain` 配置。`audit-robotcar-seasons` 检查官方划分与图片。公共 v2 rear 测试包含 429 张夜间 Query、1,443 张其他条件 Query 和 6,954 张白天 Database；测试真值隐藏时记录 Top-K 与状态，Recall/MRR 为 `null`。标准命令、固定监督宏与比较边界见 [公共 RobotCar Seasons 使用说明](docs/robotcar_seasons.md)。
