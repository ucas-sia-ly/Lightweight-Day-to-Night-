# 项目整理记录

本次整理针对源码、配置、脚本和本地冗余产物。现有训练功能、实验超参数、数据集划分、checkpoint 内容格式和正式实验输出目录保持原值。

## 删除

| 内容 | 删除依据 |
|---|---|
| `configs/reference_models_320.yaml` | 无代码或脚本引用；数据、评测参数和两个模型注册项与 `configs/references/day_completion.yaml` 中的内容完全一致 |
| 根目录 `mixvpr_128_eval.log` | 历史评测终端日志；不被训练、评测或结果汇总读取 |
| `__pycache__/`、`.pytest_cache/` | 自动生成的解释器与测试缓存 |
| `runs/smoke/`、`runs/benchmarks/nine_methods_smoke/` | 一步训练及小样本验证产物，不用于正式指标或初始化 |
| 空 manifest 备份 `combined_generator_train.csv.e3b0c44298fc.bak` | 0 字节历史备份，不含样本 |
| 空 `reviews/` | 无文件、无引用 |
| VS Code 的 LaTeX / BibTeX / LTeX 配置 | 对应 LaTeX 项目已不在工作区；改为 Python 设置与扩展推荐 |
| `.gitignore` 中重复的 `tests/` 和失效的论文临时目录规则 | 测试应可进入版本控制；数据、运行产物和缓存使用统一规则 |

已清理的文件数据约 104.5 MiB。正式训练日志保留在 `runs/`，生成图像、teacher cache、PCA、词典、权重与结果 JSON/CSV 均有明确用途，未按目录大小删除。

## 路径迁移

| 原位置 | 新位置 |
|---|---|
| `PAPER_ZH.md` | `docs/paper_zh.md` |
| 原首页的完整实验说明 | `docs/workflows.md`；首页保留结构与常用入口 |
| `configs/generator_*.yaml` | `configs/generator/` |
| 主实验 `configs/vpr_*.yaml` | `configs/vpr/`；去掉冗余 `vpr_` 前缀，baseline 使用 `day_fake_{dim}.yaml` |
| `configs/reference_*.yaml` | `configs/references/`；主注册表为 `official.yaml`，小模型为 `small.yaml` |
| 数据源、单数据集和扩展学生配置 | `configs/experiments/{data_sources,single_dataset,students}/` |
| `scripts/run_seed0_all.fish` | `scripts/train/run_main_seed0.fish` |
| `scripts/robotcar_only.py` | `scripts/data/prepare_robotcar.py` |
| 其他平铺脚本 | `scripts/{data,train,evaluate,experiments,references,benchmark,report}/` |
| `reference_*`、`small_references`、`small_summary` 模块 | `a2b_lightvpr.references.{models,suite,pca,coverage,training,summary}` |
| 实验汇总与论文图模块 | `a2b_lightvpr.reporting.{seed0_summary,paired_summary,multiseed_summary,figures}` |
| `qualitative_retrieval`、`qualitative_search` 模块 | `a2b_lightvpr.qualitative.{export,search}` |

配置、脚本、模块内部引用、测试、论文命令和文档链接已同步更新。CLI 子命令保持原名；直接导入上述旧 Python 模块的外部代码需要改为新路径。

## 运行修正

所有启动脚本以自身位置定位项目根目录，使用当前环境的 `python`，支持 `PYTHON_BIN` 指定解释器。原脚本中不存在的 `img2img-turbo` 环境路径已移除。

论文图写入 `runs/paper/`，九模型定性图写入 `runs/qualitative_top1/figures/all_methods/`；选样文件统一为 `runs/qualitative_top1/selection_all_methods.json`。论文绘图脚本在缺少选样文件时调用定性搜索入口，再导出 baseline/CDND 对比图。

外部数据集和第三方权重的原研究机路径仍在 YAML 与模型包装脚本中，需要按实际安装位置修改。没有运行完整训练、图像生成或 GPU 评测。

## 验证

- 32 份迁移后的 YAML 内容与整理前一致；31 份通用实验配置通过校验，BoQ-S 专用配置保留独立 schema。
- 全部 43 个 Python 模块导入成功，8 个 Python 辅助脚本的 `--help` 入口正常。
- 22 个 fish/bash 脚本通过语法检查；源码、脚本和测试中的具体配置路径与脚本路径存在。
- 测试：135 passed，11 skipped。跳过项需要本机未安装的外部数据集、官方仓库或初始化权重。原先因缺少本地资源而失败的 4 项测试增加了资源缺失检查；资源存在时仍执行原有断言。
- `git diff --check` 通过。

原工作区已有的未提交修改保留。整理前的源码、配置、脚本、测试和文档快照位于 `/tmp/a2b-lightvpr-before-cleanup-20261012.tar.gz`；它不包含大型数据与运行产物，临时目录清理后会消失。
