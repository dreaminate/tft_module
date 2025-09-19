# tft_module

多目标 Temporal Fusion Transformer (TFT) 项目，面向加密 / 股票等金融时间序列的多周期、多任务预测。

## 最新亮点（v0.2）

- **预测契约统一**：`MyTFTModule.predict_step` 直接输出 `{score, uncertainty, meta}`，并将批次结果写入 `predictions_{expert}_{period}_{timestamp}.parquet`，含 `symbol/period/time_idx/head_scale/head_bias/schema_ver/...` 等元信息。
- **校准与稳定评估**：在验证阶段自动执行温度缩放、ECE、Brier、P10/P50/P90 覆盖率与 Pinball Loss；生成 per-symbol × period 汇总表，并将可靠性曲线、各类指标写入 TensorBoard。
- **单指标监控**：分类任务以 `val_*_ap@period` 监控，回归任务以 `val_*_rmse@period` 监控，`utils/stage_summary.py` 仅保留 `best_monitor` + `best_val_loss`。
- **Regime 核心特征**：`features/regime_core.py` 产出波动、量能、资金费率 / OI 斜率、动量、ATR 斜率、结构 gap 等慢频字段，自动融合进 OOF 数据集。
- **OOF ↦ Z 层训练闭环**：`pipelines/build_oof_for_z.py` 汇总各专家预测、校验版本一致性，生成 `datasets/z_train.parquet`；`experts/Z-Combiner/train_z.py` 基于 OOF 数据训练二层 Stacking（Logistic / MLP），并与“等权”“单最佳专家”基线比较。
- **无泄漏审计**：`utils/audit_no_leakage.py` 快速检测 `z_train.parquet` 是否存在重复、时间倒退、缺失或时间间隔异常。

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
│  └─ build_oof_for_z.py         # 汇总专家输出生成 z_train.parquet
├─ features/
│  └─ regime_core.py             # Regime 核心特征计算
├─ metrics/
│  └─ calibration.py             # 温度缩放 / ECE / Brier / Reliability
├─ utils/
│  ├─ eval_report.py             # per-symbol × period 指标汇总
│  ├─ audit_no_leakage.py        # OOF 数据检查
│  ├─ mp_start.py                # Windows 多进程启动补丁
│  └─ ...                        # loss_factory / metric_factory 等
└─ scripts/
   ├─ dump_batch.py              # 调试 DataLoader batch
   └─ experts_cli.py             # 按专家快速启动（train/resume/warm）
```

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

---

欢迎在此基础上继续扩展：如接入更多专家、引入分位模型或 GNN/LLM 特征等。
