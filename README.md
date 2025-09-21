# tft_module

多目标 Temporal Fusion Transformer (TFT) 项目，面向加密 / 股票等金融时间序列的多周期、多任务预测。

## 最新亮点（v0.2.2）

- **预测契约统一**：`MyTFTModule.predict_step` 直接输出 `{score, uncertainty, meta}`，并将批次结果写入 `predictions_{expert}_{period}_{timestamp}.parquet`，含 `symbol/period/time_idx/head_scale/head_bias/schema_ver/...` 等元信息。
- **校准与稳定评估**：在验证阶段自动执行温度缩放、ECE、Brier、P10/P50/P90 覆盖率与 Pinball Loss；生成 per-symbol × period 汇总表，并将可靠性曲线、各类指标写入 TensorBoard。
- **单指标监控**：分类任务以 `val_*_ap@period` 监控，回归任务以 `val_*_rmse@period` 监控，`utils/stage_summary.py` 仅保留 `best_monitor` + `best_val_loss`。
- **Regime 核心特征**：`features/regime_core.py` 产出波动、量能、资金费率 / OI 斜率、动量、ATR 斜率、结构 gap 等慢频字段，自动融合进 OOF 数据集。
- **OOF ↦ Z 层训练闭环**：`pipelines/build_oof_for_z.py` 汇总各专家预测、校验版本一致性，生成 `datasets/z_train.parquet`；`experts/Z-Combiner/train_z.py` 基于 OOF 数据训练二层 Stacking（Logistic / MLP），并与“等权”“单最佳专家”基线比较。
- **无泄漏审计**：`utils/audit_no_leakage.py` 快速检测 `z_train.parquet` 是否存在重复、时间倒退、缺失或时间间隔异常。

### 训练与特征筛选体验（v0.2.2 增强）

- XGBoost 2.x 支持：统一使用 `tree_method="hist" + device="cuda"`，旧版自动回退 `gpu_hist/gpu_predictor`。
- Lightning 进度条默认开启，同时日志频率由配置控制：
  - 在专家叶子 `model_config.yaml` 中设置 `log_every_n_steps`（优先）或 `log_interval`，未设置时默认 100。
  - 相关脚本：`train_multi_tft.py`、`train_resume.py`、`warm_start_train.py`。
- 特征筛选线性路径更稳健：
  - `embedded_stage.py` 将 `linear_max_iter` 默认提高到 2000；
  - 在 `LogisticRegressionCV` / `ElasticNetCV` 拟合时静默 `ConvergenceWarning`（不影响结果，仅抑制噪声）。

## 数据融合（Fundamentals + On-chain）

- 主流程：`src/fuse_fundamentals.py` → `fuse()`；推荐通过 `pipelines/configs/fuse_fundamentals.yaml` + `run_with_config()` 驱动。
- 执行命令：

  ```bash
  python -c "from src.fuse_fundamentals import run_with_config; run_with_config('pipelines/configs/fuse_fundamentals.yaml')"
  ```

- 每位专家的具体字段配置存放在 `configs/experts/<Expert>/datasets/base.yaml` 与 `configs/experts/<Expert>/datasets/rich.yaml`；`experts_map` 只引用这些文件，便于按专家自管理。
- 配置约定：
  - `*_base`：`dataset_type=fundamentals_only`，`max_missing_ratio=0.01`（缺失率>1% 的样本行被删除），默认不启用交集裁剪，适合覆盖早期样本。
  - `*_rich`：`dataset_type=combined`，`max_missing_ratio: null`，保留所有链上指标，由 `no_nan_policy(scope:onchain, method:intersect)` 对链上列求交集，适合近段完整数据。
  - `extra_field` 将所有输出命名为 `<Expert>_<base|rich>` 并写入 `data/merged/expert_group/<Expert>_<base|rich>/`。

- 运行结束自动生成：
  - `full_merged_with_fundamentals.{csv,pkl}` 与 `full_merged_slim.csv`；
  - `fundamental_columns.txt`、`dataset_group_summary.csv`（含 `missing_threshold_rows`、`missing_threshold_cols_count`、`intersect_all_null_cols_count` 等统计）；
  - `missing_threshold_columns.txt`、`intersect_columns*.txt`、`fuse_audit/*`、`config_snapshot.yaml`；
  - 汇总报表：`reports/datasets/experts_group_summary.csv`。

- 自定义提示：
  - 关闭自动裁剪 → `max_missing_ratio` 设为 `null` / 删除，或把 `no_nan_policy.enabled` 设为 `false`。
  - 精确控制交集列 → `scope: custom` + `columns: [...]`。
  - 若需直接在代码里控制，可调用：

    ```python
    from src.fuse_fundamentals import fuse

    fuse(
        base_csv='data/merged/full_merged.csv',
        out_csv='data/merged/expert_group/Custom/full_merged_with_fundamentals.csv',
        include_symbol={'funding_oi': True, 'funding_vol': True, ...},
        include_global={'premium': True, 'altcoin': True, ...},
        max_missing_ratio=0.02,
        no_nan_policy={'enabled': True, 'scope': 'custom', 'method': 'intersect', 'columns': ['premium_premium_index']},
        validate=True,
    )
    ```

  - `post_convert` 会沿用全局配置。如需为单个专家单独设置（仅导出 CSV 或关闭 PKL），可在该条目下覆盖 `post_convert` 字段。

## 目录结构（核心）

```
tft_module/
├─ train_multi_tft.py            # 专家训练入口
├─ train_resume.py               # 续训 checkpoint
├─ warm_start_train.py           # Warm-start 微调
├─ experts/
│  └─ Z-Combiner/
│       ├─ model_config.yaml     # Z 层训练配置（示例）
│       └─ train_z.py            # OOF → Stacking 训练脚本
├─ models/
│  └─ tft_module.py              # MyTFTModule + HybridMultiLoss
├─ pipelines/
│  ├─ build_oof_for_z.py         # 汇总专家输出生成 z_train.parquet
│  └─ configs/
│       └─ fuse_fundamentals.yaml  # 数据融合（Fundamentals + On-chain）配置
├─ features/
│  └─ regime_core.py             # Regime 核心特征计算
├─ metrics/
│  └─ calibration.py             # 温度缩放 / ECE / Brier / Reliability
├─ utils/
│  ├─ eval_report.py             # per-symbol × period 指标汇总
│  ├─ audit_no_leakage.py        # OOF 数据检查
│  ├─ mp_start.py                # Windows 多进程启动补丁
│  └─ ...                        # loss_factory / metric_factory 等
├─ scripts/
│  ├─ dump_batch.py              # 调试 DataLoader batch
│  └─ experts_cli.py             # 按专家快速启动（train/resume/warm）
└─ data/
   └─ merged/
        ├─ full_merged.csv        # 基础 K 线 + 技术指标
        └─ expert_group/          # 每个专家（base|rich）融合结果
```

`data/merged/expert_group/` 是融合脚本的主要输出目录，运行后会为每位专家生成 `<Expert>_base/` 与 `<Expert>_rich/` 两套数据集：

- `full_merged_with_fundamentals.{csv,pkl}`：用于训练/特征筛选的全量列。
- `full_merged_slim.csv`：核心价量 + 基本面 + 目标的精简版。
- `fundamental_columns.txt`：新增列清单。
- `dataset_group_summary.csv`：每个 `(symbol, period)` 的样本数、时间范围，以及 `missing_threshold_rows`、`missing_threshold_cols_count`、`intersect_all_null_cols_count` 等裁剪统计。
- `missing_threshold_columns.txt` / `intersect_columns*.txt`：当缺失率阈值或交集策略触发时的列记录。
- `fuse_audit/`：覆盖率报表、时间审计、列统计；`config_snapshot.yaml` 保存本次融合配置。

## 专家训练流程

1. **准备数据**：
   - 按 `src/` 流水线或自有方式生成长表 `data/pkl_merged/full_merged.pkl`（可包含技术、链上、衍生品等特征 + `target_*` 标签）。
   - 确保列包含 `symbol`、`period`、`time_idx` 以及需要的特征 / 目标。

2. **选择叶子配置**：
   - 每个专家 × 周期 × 模态都有独立目录 `configs/experts/<Expert>/<period>/<modality>/`，需要同时提供 `model_config.yaml` 与 `targets.yaml`。
   - 叶子配置可继承 `schema_version/data_version/expert_version/train_window_id` 等元信息（默认可留空）。

3. **启动训练**：
   ```bash
   python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
   # 续训 / 热启动同理：
   python train_resume.py --config configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
   python warm_start_train.py --config configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
   ```

4. **日志与 checkpoint**：
   - 日志：`lightning_logs/experts/<expert>/<period>/<modality>/tft/`
   - 权重：`checkpoints/experts/<expert>/<period>/<modality>/tft/`
   - 默认保存 top-k 与 `last.ckpt`

## 预测契约与评估

- `predict_step` 输出：
  ```python
  {
      'score': tensor[B, T],
      'uncertainty': tensor[B, T],  # 回归: σ，分类: NaN
      'meta': {
          'symbol_idx', 'period_idx', 'time_idx',
          'head_scale', 'head_bias'
      },
      'schema_ver', 'data_ver', 'expert_ver', 'train_window_id',
      'antilog_return', 'future_price'
  }
  ```
- `on_predict_epoch_end` 汇总所有 batch 写入 parquet（默认在当前 lightning log 目录的 `predictions/`）。
- 验证阶段自动：
  - 进行温度缩放，记录 `val_ece@target`、`val_brier@target`；
  - 计算 P10/P50/P90 覆盖率（与 σ 推导的大致区间对齐）；
  - 输出 Pinball Loss（综合、P10、P90）到日志与 CSV；
  - 生成 per-symbol × period 汇总表 `eval_report_*.csv` 并写入 TensorBoard。

## OOF ↦ Z 层训练

1. **收集专家预测**：
   - 确保各专家在预测模式或验证过程中产生 `predictions_*.parquet`（目录结构：`lightning_logs/experts/<expert>/<period>/<modality>/tft/predictions/`）。

2. **构建 OOF 数据集**：
   ```bash
   python pipelines/build_oof_for_z.py        --predictions-root lightning_logs        --data-path data/pkl_merged/full_merged.pkl        --output datasets/z_train.parquet
   ```
   - 脚本会校验所有预测文件的版本字段一致，并自动并入 Regime 特征。

3. **无泄漏审计**：
   ```bash
   python utils/audit_no_leakage.py --path datasets/z_train.parquet
   ```
   - 检查重复、时间倒退、间隔超阈值、缺失等风险；PASS 后再进入下一步。

4. **训练 Z-Combiner**：
   ```bash
   python experts/Z-Combiner/train_z.py --config experts/Z-Combiner/model_config.yaml
   ```
   - 支持分类 / 回归模式：默认使用 LogisticRegression / MLPRegressor；
   - 自动与“等权”“单最佳专家”基线比较 PR-AUC / ECE（分类）或 RMSE（回归），结果落地 `lightning_logs/experts/Z-Combiner/metrics_*.json`。

## 辅助脚本

- `scripts/dump_batch.py`：启动前设置 `PYTHONPATH` 指向项目根，可快速打印一个验证 batch 的张量形状。
- `utils/mp_start.py`：内部在各训练脚本入口调用，确保 Windows 启动多进程 DataLoader 时不报错。
- `scripts/experts_cli.py`：
  ```bash
  python scripts/experts_cli.py list
  python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1h/base
  ```

## 常见问题

- **预测 parquet 不生成**：确认已运行 `Trainer.predict(...)` 或在验证结束后调用了 `trainer.predict(...)`。
- **校准指标全为 NaN**：检查对应目标是否有正样本 / 负样本，或是否误把回归目标当分类使用。
- **OOF 数据列缺失**：确保传入的 `full_merged.pkl` 包含所有 `target_*` 列，并且最新预处理已对慢频特征做 shift/ffill。
- **Z-Combiner 指标不升**：可在配置中增减 `feature_prefixes`、替换模型（例如换成 `Ridge`、`GradientBoosting` 等），或针对分类任务追加更多校准步骤。

## 快速上手（完整流程）

以下步骤将项目从数据准备、专家训练到多专家融合串联起来，帮助新同事快速跑通：

1. **准备基础数据**
   - 确保 `data/pkl_merged/full_merged.pkl`、`data/merged/full_merged.csv` 已包含核心价量、技术指标与目标列。
   - 若需要生成或更新基础数据，可先运行你们的 ETL/特征工程脚本（详见 `src/` 中的数据处理流程）。

2. **融合基本面 / 链上数据**
   - 根据专家需求编辑 `configs/experts/<Expert>/datasets/{base,rich}.yaml`（控制 `include_symbol` / `include_global` / 裁剪策略），`pipelines/configs/fuse_fundamentals.yaml` 负责引用这些文件。
   - 执行：
     ```bash
     python -c "from src.fuse_fundamentals import run_with_config; run_with_config('pipelines/configs/fuse_fundamentals.yaml')"
     ```
   - 输出会落在 `data/merged/expert_group/<Expert>_{base|rich}/`；检查 `dataset_group_summary.csv`、`fuse_audit/` 以确认覆盖率与裁剪情况。

3. **配置专家训练**
   - 为每位专家在 `configs/experts/<Expert>/<period>/<modality>/` 下准备好 `model_config.yaml`、`targets.yaml`、`weights_config.yaml`。
   - 训练脚本直接读取叶子 `targets.yaml` / `weights_config.yaml`，默认不再依赖根目录的旧版配置；需要迁移或复用权重时，修改叶子下的 `weights_config.yaml` 即可。

4. **启动训练 / 续训 / 热启动**
   ```bash
   python train_multi_tft.py --config configs/experts/Risk-TFT/1h/base/model_config.yaml
   python train_resume.py    --config configs/experts/Risk-TFT/1h/base/model_config.yaml
   python warm_start_train.py --config configs/experts/Risk-TFT/1h/rich/model_config.yaml
   ```
   - 常用命令可以通过 `scripts/experts_cli.py` 管理（`list`、`train`、`resume`、`warm`）。

5. **生成预测与 OOF 数据**
   - 训练完成后运行 `Trainer.predict(...)` 或 CLI 中的预测命令，使 `lightning_logs/experts/.../predictions/` 目录生成 `predictions_*.parquet`。
   - 汇总 OOF：
     ```bash
     python pipelines/build_oof_for_z.py --predictions-root lightning_logs \
       --data-path data/pkl_merged/full_merged.pkl --output datasets/z_train.parquet
     python utils/audit_no_leakage.py --path datasets/z_train.parquet
     ```

6. **训练 Z-Combiner / Stack 模型**
   ```bash
   python experts/Z-Combiner/train_z.py --config experts/Z-Combiner/model_config.yaml
   ```
   - 查看 `lightning_logs/experts/Z-Combiner/metrics_*.json`、`predictions/`、`eval_report_*.csv` 评估融合效果。

7. **可选：特征筛选与证据汇总**
   - 对某些专家跑特征筛选，可使用 `pipelines/configs/feature_selection(.yaml|_quick.yaml)`；对应 `pkl_path` 已指向最新 `expert_group` 数据。
   - 运行脚本前确认 `aggregation.weights_yaml`、`wrapper` 参数符合资源预算。

整体流程执行完后，你会得到：
   - 每位专家（base/rich）的最新模型权重与预测；
   - `datasets/z_train.parquet` 和训练好的 Z-Combiner；
   - 报表 / 特征证据输出，便于继续分析和调参。

---

欢迎在此基础上继续扩展：如接入更多专家、引入分位模型或 GNN/LLM 特征等。
