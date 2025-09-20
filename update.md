# 更新日志

## 2025-09-20 — 专家数据管线与文档梳理（v0.2.1）

- **专家数据集重构**：`pipelines/configs/fuse_fundamentals.yaml` 的 `experts_map` 统一采用 `<Expert>_{base|rich}` 命名，融合输出落入 `data/merged/expert_group/<Expert>_{base|rich}/`，旧的 `baseline` / `expert_*` 等目录已清理。
- **缺失率 / 交集逻辑增强**：`src/fuse_fundamentals.py` 引入 `max_missing_ratio` 行裁剪与全空列过滤；`dataset_group_summary.csv` 记录 `missing_threshold_rows`、`missing_threshold_cols_count`、`intersect_all_null_cols_count` 并生成 `missing_threshold_columns.txt`。
- **配置补全**：所有专家叶子（base & rich）新增 `weights_config.yaml`；每位专家新增 `datasets/base.yaml` 与 `datasets/rich.yaml` 存放融合字段配置，并在 `experts_map` 中引用对应路径；特征筛选配置 `feature_selection(.yaml|_quick.yaml)` 的 `pkl_path` 调整为指向最新专家数据目录。
- **README 更新**：补充“数据融合”与“快速上手”章节，详细说明 `expert_group` 目录结构及运行命令。
- **数据清理**：删除 `data/merged/` 与 `data/merged/expert_group/` 下的遗留主题目录，仅保留按专家划分的数据集。
- **汇总报表更新**：`reports/datasets/experts_group_summary.csv` 汇聚所有 `<Expert>_{base|rich}` 数据，便于巡检缺失率与样本范围。
- **全局配置瘦身**：删除根目录 `configs/targets.yaml`/`configs/weights_config.yaml`；训练与特征筛选脚本改为依赖各专家叶子下的 `targets.yaml`、`weights_config.yaml`，其余工具默认在缺省权重时使用均值策略。
- **文档同步**：`experts.md` 新增“当前实现总览”小节，说明长表数据结构、慢频广播、专家 `datasets/{base,rich}.yaml` 配置、以及训练阶段按 `period` 过滤的方式。

变更文件（关键）：

- `README.md`
- `pipelines/configs/fuse_fundamentals.yaml`
- `pipelines/configs/feature_selection.yaml`
- `pipelines/configs/feature_selection_quick.yaml`
- `src/fuse_fundamentals.py`
- `reports/datasets/experts_group_summary.csv`
- `configs/experts/**/weights_config.yaml`

## 2025-09-19 — Z 层训练与稳定评估（v0.2）

- `MyTFTModule`：`predict_step` 统一输出 `{score, uncertainty, meta}`，预测落盘 parquet 增补 schema/data/expert/train_window 等元信息；验证阶段新增温度缩放、ECE/Brier、P10/P50/P90 覆盖率与 Pinball Loss，并输出 per-symbol × period 评估报表及 TensorBoard 可靠性曲线。
- 单指标监控：`train_multi_tft.py` / `train_resume.py` / `warm_start_train.py` 自动按任务选择 `val_*_ap@period` 或 `val_*_rmse@period`，`utils/stage_summary.py` 去除 composite 字段，仅保留 `best_monitor` 与 `best_val_loss`。
- `pipelines/build_oof_for_z.py`：汇总专家预测、校验版本一致性，结合目标与 Regime 特征生成 `datasets/z_train.parquet`。
- `features/regime_core.py`：提供波动、量能、资金费率/OI 斜率、动量、ATR 斜率、结构 gap 等核心字段。
- `experts/Z-Combiner/train_z.py` + `model_config.yaml`：基于 OOF 数据训练二层 Stacking（Logistic/MLP），输出与“等权”“单最佳专家”基线的 PR-AUC/ECE 或 RMSE 对比。
- `utils/audit_no_leakage.py`：快速检查 `datasets/z_train.parquet` 重复、时序单调与缺失风险。
- 其它辅助：`metrics/calibration.py`、`utils/eval_report.py`、`scripts/dump_batch.py`、`utils/mp_start.py` 完善校准与并发加载工具链。

变更文件（关键）：

- `models/tft_module.py`
- `metrics/calibration.py`
- `utils/eval_report.py`
- `pipelines/build_oof_for_z.py`
- `features/regime_core.py`
- `experts/Z-Combiner/train_z.py`
- `train_multi_tft.py` / `train_resume.py` / `warm_start_train.py`
- `utils/stage_summary.py`
- `utils/audit_no_leakage.py`
- `scripts/dump_batch.py`
- `utils/mp_start.py`

## 2025-09-14 — 基于专家的配置重构（v0.1）

- 新增专家配置树：`configs/experts/<Expert>/<period>/<modality>/model_config.yaml`。
- 训练脚本新增 `--config` 与 `--expert`，可从路径推断 `{expert, period, modality, symbol}`，并将日志/权重按 专家/周期/模态 分桶。
- DataLoader 支持按 `periods`、`symbols` 过滤（来源于配置路径/字段）。
- 扩充 `configs/targets.yaml` 至 9+1 专家条目（暂映射到现有目标列，后续可替换为真实任务）。
- 为多位专家脚手架最小可跑配置（Alpha-Dir/Alpha-Ret/Risk/MicroStruct/OnChain/Regime/KeyLevel/Relative/Factor/Z）。
- 更新 README，加入新的调用方式：`python train_multi_tft.py --config configs/experts/.../model_config.yaml`。

变更文件（关键）：

- `train_multi_tft.py`
- `train_resume.py`
- `warm_start_train.py`
- `data/load_dataset.py`
- `configs/targets.yaml`
- `README.md`
- 新增 `configs/experts/**/model_config.yaml`

备注：

- Z-Combiner 当前为占位；脚本仅支持 `model_type: tft`。
- Base/Rich 门控、Z 层训练、统一输出契约字段（uncertainty/recent_valid_perf/meta）将在下一步实现。

### 2025-09-14 / Patch-1 — 训练入口强化与就近配置

- 修复 `data/load_dataset.get_dataloaders`：优先使用传入 `data_path`，不可用时回退默认 pkl；新增 `selected_features_path` 参数。
- 训练脚本（train_multi_tft/train_resume/warm_start）：
  - 新增“就近查找”配置：优先叶子目录 → 专家根 → 全局 `configs/` 查找 `weights_config.yaml` 与 `selected_features.txt`；
  - 自动按任务选择监控指标（分类→`AP` 最大化；回归→`RMSE` 最小化），可用 `monitor/monitor_mode` 覆盖；
  - 日志与权重路径继续按 专家/周期/模态 分桶；
  - Warm-start 同步上述改进。
- 在各专家根目录新增 `weights_config.yaml`（默认为单目标权重）。

注意：未删除任何现有 root 配置，以便平滑过渡；后续如需清理可再迁移/删除。

### 2025-09-14 / Patch-2 — 彻底专家化与配置瘦身（方案B）

- 删除 `configs/model_config.yaml`，训练脚本 `--config` 改为必填，强制使用专家叶子配置。
- 迁移管线配置至 `pipelines/configs/`：
  - `configs/feature_selection.yaml` → `pipelines/configs/feature_selection.yaml`（并将 `tft_gating.model_config` 指向专家叶子）
  - `configs/feature_selection_quick.yaml` → `pipelines/configs/feature_selection_quick.yaml`
  - `configs/fuse_fundamentals.yaml` → `pipelines/configs/fuse_fundamentals.yaml`
- 同步更新脚本默认路径：
  - `pipelines/run_feature_screening.py` 默认 `--config` 改为 `pipelines/configs/feature_selection.yaml`
  - `src/csv2Pkl.py`、`src/csv2Parquet.py`、`src/fuse_fundamentals.py` 的默认配置路径全部改为 `pipelines/configs/fuse_fundamentals.yaml`
  - `features/selection/tft_gating.py` 默认 `model_cfg_path` 改为专家叶子示例路径
- 训练脚本（train_multi_tft/train_resume/warm_start）：`--config` 必填；其余保持“就近查找”逻辑与任务感知监控指标。

### 2025-09-14 / Patch-3 — 叶子目录自包含要求（强制）

- 训练严格要求每个专家叶子目录必须提供训练所需的 YAML：
  - `model_config.yaml`
  - `targets.yaml`（包含本叶子的 `model_type/targets/weights`）
- 训练脚本在启动时会校验叶子目录，缺失则报错并提示补齐；`weights_config.yaml` 仅作兼容回退，不再强制。

### 2025-09-14 / Patch-4 — 移除复合指标，统一用损失评估

- 删除 `callbacks/custom_checkpoint.py` 与所有 composite 相关逻辑/文档。
- 训练脚本（train_multi_tft/train_resume/warm_start）统一使用 `val_loss_epoch` 作为监控与早停指标。
- README 清理 composite 章节与说明，明确训练仅以损失作为评估依据。

### 2025-09-14 / Patch-5 — 分桶快照与保存 last.ckpt

- `train_resume.py`：配置快照改为写入对应专家分桶 `lightning_logs/experts/<expert>/<period>/<modality>/tft/configs/`，并附 `targets_used.yaml`。
- `train_multi_tft.py` / `train_resume.py` / `warm_start_train.py` 的 `ModelCheckpoint` 增加 `save_last=True`，统一保存 `last.ckpt`。

### 2025-09-14 / Patch-6 — CLI 入口（按专家快速启动）

- 新增 `scripts/experts_cli.py`：
  - `list`：罗列所有叶子 `model_config.yaml`（expert/period/modality → path）。
  - `train|resume|warm --expert --leaf 1h/base`（或 `--period/--modality`）：自动定位叶子并校验必需文件，调用底层脚本。
  - 支持 `--experts-root` 自定义根目录。

### 2025-09-14 / Patch-7 — 叶子 targets.yaml 与 Base/Rich 补全

- 新增 rich 叶子：为现有专家×周期补齐 `<Expert>/<period>/rich/`，并提供最小可跑的 `model_config.yaml`。
- 每个叶子新增 `targets.yaml`：包含 `model_type`、`targets` 与 `weights`（by_target 或 custom 列表）。
- 训练脚本改为强制读取叶子 `targets.yaml`（不再依赖全局 `configs/targets.yaml` 决定目标），权重优先取自该文件，缺失时回退同目录 `weights_config.yaml`，否则默认全 1。

### 2025-09-14 / Patch-8 — 体检脚本并行化修正

- 将 `scripts/inspect_batch.py` 的 `num_workers` 从 `0` 改为 `4`，避免与 DataLoader 的 `prefetch_factor` 冲突导致报错。
初步自检（静态扫描）

- 所有对 `configs/model_config.yaml` 的默认引用已移除/改写；仅保留日志快照写入 `logs/configs/model_config.yaml`（与删除无冲突）。
- 所有对 `configs/feature_selection.yaml`、`configs/fuse_fundamentals.yaml` 的默认引用已改为 `pipelines/configs/...`。

已知影响与后续验证

- 若外部脚本/文档仍假定存在 `configs/model_config.yaml`，需改为显式传入 `--config` 或指向某个专家叶子；README 已更新。
- 运行期验证：建议在本地数据可用时分别跑一次训练与管线，以确认路径迁移无遗漏。

