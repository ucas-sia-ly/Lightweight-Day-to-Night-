# A2B-LightVPR

用于昼夜视觉地点识别的研究项目：QS-Attn 离线生成 fake-night 图像，SALAD 教师通过 CDND 监督轻量 MixVPR 学生；RobotCar 支持推理阶段的时序重排 TCR。部署 checkpoint 只包含学生模型。

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
├── evaluate/                  # 单帧评测、同权重评测和 TCR
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
└── *.py                       # CLI、配置、数据、检索指标、运行与 TCR
tests/                        # 单元测试与可选本地资源检查
docs/                         # 工作流、论文草稿与整理记录
data/                         # 本地数据、生成图像与缓存（不入 Git）
runs/                         # 权重、评测结果、日志与图表（不入 Git）
```

## 安装与检查

需要 Python 3.10+；完整训练需要 CUDA、数据集和第三方权重。所有命令均在项目根目录执行。

```bash
python -m pip install -e '.[test,eval]'
python -m a2b_lightvpr.cli --help
python -m pytest
```

绘图使用 `.[paper]`，小参考模型使用 `.[small-reference]`。第三方训练包装脚本还依赖对应官方仓库及其环境。

YAML 中的外部数据集、第三方仓库和权重路径沿用原研究机路径，迁移后需要修改。项目内部 `data/`、`runs/` 路径相对于项目根目录。脚本默认使用当前环境的 `python`，也可以通过 `PYTHON_BIN=/path/to/python` 指定解释器。

## 常用入口

| 工作 | 命令 |
|---|---|
| 完整主实验 | `fish scripts/train/run_main_seed0.fish` |
| 512 维多种子与消融 | `fish scripts/train/run_paper_experiments_512.fish` |
| RobotCar 时序评测 | `fish scripts/evaluate/run_robotcar_temporal_512.fish` |
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

`checkpoints/latest.pt` 用于恢复训练，`model_best.pt` 用于推理，两者不可混用。旧实验产物中的路径、模型标识和指纹保持原值；目录整理后的命令使用新路径。

详细实验参数、结果、证据边界和排错见 [工作流说明](docs/workflows.md)，论文草稿见 [paper_zh.md](docs/paper_zh.md)，本次删除与迁移说明见 [整理记录](docs/cleanup.md)。
