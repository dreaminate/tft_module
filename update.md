# 更新日志

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
