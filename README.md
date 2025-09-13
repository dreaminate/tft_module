
# tft\_module

多目标 **Temporal Fusion Transformer (TFT)** 项目，用于加密/股票等金融时间序列的多周期、多任务预测。
内置**特征与目标构建流水线**、**自动实验目录**、**复合指标加权保存**、**阶段性损失权重调度**、**Warm-Start 微调**、以及\*\*(symbol, period) 粒度的回归目标标准化\*\*等功能。

> 适配 1h / 4h / 1d 等多周期数据，支持同时预测分类与回归目标（如涨跌、对数收益、回撤概率等）

---

## 目录

* [目录结构](#目录结构)
* [环境准备](#环境准备)
* [数据流程](#数据流程)

  * [技术指标与局部归一化](#技术指标与局部归一化)
  * [目标字段](#目标字段)
  * [多周期对齐/合并](#多周期对齐合并)
  * [数据加载](#数据加载)
* [训练与复现](#训练与复现)

  * [从零开始训练](#从零开始训练)
  * [续训练](#续训练)
  * [Warm-Start 微调](#warmstart-微调)
  * [自动目录](#自动目录)
* [关键组件](#关键组件)
* [配置示例](#配置示例)

  * [模型与训练超参（model\_config.yaml）](#模型与训练超参model_configyaml)
  * [损失权重（weights\_config.yaml）](#损失权重weights_configyaml)
  * [复合指标权重（composite\_scoreyaml）](#复合指标权重composite_scoreyaml)
* [日志与可视化](#日志与可视化)
* [数据规范与要求](#数据规范与要求)
* [技巧与实践建议](#技巧与实践建议)
* [故障排查（FAQ）](#故障排查faq)
* [其他脚本](#其他脚本)
* [清理旧实验](#清理旧实验)
* [致谢](#致谢)
* [许可](#许可)

---

## 目录结构

```
tft_module/
├─ train_multi_tft.py        # 从零训练
├─ train_resume.py           # 按 checkpoint 续训
├─ warm_start_train.py       # 仅载入模型权重做微调
├─ eval_multi_tft.py         #（可选）在验证/测试集做推理评估&导出CSV
├─ configs/                  # 模型/权重/复合指标 YAML 配置
│  ├─ model_config.yaml
│  ├─ weights_config.yaml
│  └─ composite_score.yaml   # 可选：复合指标权重
├─ data/                     # 数据加载脚本与中间文件
│  └─ load_dataset.py
├─ model/                    # LightningModule 封装的 TFT
│  └─ tft_module.py
├─ callbacks/                # 自定义回调（复合指标 checkpoint）
│  └─ custom_checkpoint.py
├─ utils/                    # 训练/日志/损失/指标等工具库
│  ├─ run_helper.py
│  ├─ loss_factory.py
│  ├─ metric_factory.py
│  ├─ loss_scheduler.py
│  ├─ composite.py
│  ├─ weighted_bce.py
│  └─ stage_summary.py
├─ tft/                      # 新的包结构（核心代码归档于此）
│  ├─ models/                # 模型定义
│  │  └─ tft_module.py
│  ├─ data/                  # 数据加载与拆分
│  │  └─ loaders.py
│  ├─ utils/                 # 工具库（loss/metric 等）
│  │  ├─ loss_factory.py
│  │  ├─ metric_factory.py
│  │  ├─ weighted_bce.py
│  │  ├─ loss_scheduler.py
│  │  ├─ composite.py
│  │  ├─ run_helper.py
│  │  ├─ stage_summary.py
│  │  ├─ checkpoint_utils.py
│  │  ├─ compare_logs.py
│  │  └─ plot_loss_weights.py
│  ├─ callbacks/
│  │  └─ custom_checkpoint.py
│  ├─ features/
│  │  └─ selection/          # 特征筛选流水线（树/置换/TFT gating/聚合/优化/滚验）
│  │     ├─ common.py
│  │     ├─ tree_perm.py
│  │     ├─ aggregate_core.py
│  │     ├─ tft_gating.py
│  │     ├─ optimize_subset.py
│  │     └─ rolling_validate.py
│  └─ pipelines/
│     └─ prune_and_time_audit.py
└─ src/                      # 旧数据管线与脚本（保留兼容转发/逐步迁移）
   ├─ indicating.py / indicators.py / ...
   └─ feature_selection/ (已转发至 tft.features.selection.*)
```

---

## 环境准备

1. 安装 Conda 或 Mamba。
2. 创建环境并安装依赖：

```bash
conda env create -f tft.yml      # 或 mamba env create -f tft.yml
conda activate tft
```

> `tft.yml` 内包含 **PyTorch 2.2**、**PyTorch Lightning**、**PyTorch Forecasting**、**pandas**、**scikit-learn** 等依赖。建议使用符合 CUDA 的 PyTorch 版本。

---

## 数据流程

整体流程（从原始 K 线到可训练数据）：

```
原始K线/衍生数据
   │
   ├─▶ src/indicating.py                 # 计算技术指标、可选LOF异常检测、局部滑动归一化
   │
   ├─▶ src/target_config.py              # 构建多任务预测目标（分类/回归，不同周期）
   │
   ├─▶ src/data_fusion.py                # 多周期字段对齐/合并，产出 full_merged.pkl
   │
   └─▶ data/load_dataset.py              # 构建 TimeSeriesDataSet & DataLoader（自动统计标准化）
```

### 技术指标与局部归一化

* `src/indicating.py`：从 `data/crypto/<period>/` 读取原始 K 线，生成多种指标/派生特征（详见 `src/indicators.py`），支持：

  * **分组滑动归一化**（`groupwise_rolling_norm.py`）：支持 z-score & Min-Max；
  * **LOF 异常检测（可选）**：为诸如 log-return、波动率、成交量等字段产生异常提示（只做提示，不删除数据）；
  * 统一**长表结构**（推荐）：各特征列不带后缀，单独用 `symbol` 与 `period` 标识。

### 目标字段

* `src/target_config.py`：按 1h/4h/1d 等周期构建多类预测目标，包括但不限于：

  * **分类**：`target_binarytrend`（涨跌）、`target_pullback_prob`（短期回调）、`target_sideway_detect`（震荡识别）等；
  * **回归**：`target_logreturn`（对数收益率）、`target_logsharpe_ratio`（风险调整收益）、`target_trend_persistence`（趋势持续步数）、`target_breakout_count`（连续突破次数）、`target_drawdown_prob` / `target_max_drawdown` 等。
* **要点**：

  * 未来收益如 `logreturn` 仅作为**目标**，不作为输入，避免信息泄露；
  * 波动率类（rolling/parkinson/ewma/APARCH/EGARCH）常作为**辅助特征**，非 `target_` 字段。
  * 跨周期字段可下放：慢频（1d/4h）可向 1h **repeat/ffill**，保证小周期样本不留空。

### 多周期对齐/合并

* `src/data_fusion.py`：对齐不同周期的 `future_close`、辅助特征、目标字段，输出 **`full_merged.pkl`** 供训练使用。
* 对齐策略：

  * 使用 `timestamp` 对齐；
  * 慢频字段向下广播给快频窗口；
  * 统一列命名与缺失列补齐，确保训练阶段字段一致。

### 数据加载

* `data/load_dataset.py`：读取 `full_merged.pkl`，构建 `pytorch_forecasting.TimeSeriesDataSet` 与 `DataLoader`：

  * 自动统计 (symbol, period, target) 三级粒度的 **mean/std**，供回归目标标准化；
  * 支持 **ConcatDataset** 合并多周期数据；
  * 提供 `max_encoder_length=96`、`max_prediction_length=1` 的默认序列窗口设置；
  * 含 NaN 与组长度校验，防止 `_getitem__=None` 导致的 `default_collate` 报错。

---

## 训练与复现

### 从零开始训练

```bash
python train_multi_tft.py --config configs/model_config.yaml
```

* `configs/model_config.yaml`：模型结构与训练超参
* `configs/weights_config.yaml`：各目标的基础损失权重（与 `loss_scheduler` 配合可阶段切换）

### 续训练

修改 `model_config.yaml` 中的 `resume_ckpt` 指向目标 checkpoint，然后执行：

```bash
python train_resume.py
```

> 将恢复优化器状态、学习率调度器、当前 epoch 等信息，继续同一实验阶段。

### Warm-Start 微调

仅加载**模型权重**，不恢复优化器/epoch；适合新数据/新任务的迁移微调。

```bash
python warm_start_train.py
```

> 可以重置 `head`/`目标列表`/`loss 配置`，以较小学习率对新任务进行适配。

### 自动目录

所有训练脚本通过 `utils/run_helper.prepare_run_dirs()` 生成规范化目录：

```
runs/<script>-YYYYMMDD_HHMM/
├─ checkpoints/        # 模型权重（文件名包含 composite_score 与 val_loss）
├─ lightning_logs/     # TensorBoard 日志
└─ configs/            # 运行时保存的配置快照（便于复现）
```

---

## 关键组件

| 模块                               | 说明                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model/tft_module.py`            | 基于 PyTorch Forecasting 的 **TemporalFusionTransformer** 封装，支持多目标输出、`HybridMultiLoss`（手动权重 × 不确定性自动加权），`OneCycleLR`，分类/回归混合指标，(symbol, period) 粒度回归目标标准化。 |
| `callbacks/custom_checkpoint.py` | 监控自定义复合指标 `val_composite_score`，通过 `_normalize` 将回归/分类指标归一后加权求和，保存 top-k checkpoint。文件名同时包含 `composite_score` 与 `val_loss`。                             |
| `utils/loss_factory.py`          | 目标名到损失的工厂：分类→`WeightedBinaryCrossEntropy`（支持 auto\_pos\_weight），回归目标按分布特性选择 `SmoothL1Loss` 或 `MSELoss`。                                                 |
| `utils/metric_factory.py`        | 指标工厂：分类→F1、Accuracy、Precision、Recall、ROC-AUC、AP；回归→RMSE、MAE。自动按目标与周期展开。                                                                                 |
| `utils/loss_scheduler.py`        | **多阶段损失权重调度器**：到指定 epoch 自动切换 `weights_config`。                                                                                                         |
| `utils/composite.py`             | 过滤/补齐复合指标权重，自动移除数据集中不存在的 `(target@period)` 指标条目。                                                                                                        |
| `utils/weighted_bce.py`          | mini-batch 级别 `auto_pos_weight`，解决类别不平衡的动态加权。                                                                                                           |
| `utils/stage_summary.py`         | 将当前训练阶段的最优 `val_composite_score` 与 `val_loss` 写入 `stage_logs/<log_name>.yaml`，便于多阶段训练对比与回溯。                                                             |

---

## 配置示例

> 下述示例仅为范式，请按你的项目字段与需求调整。建议将**实际运行配置**的快照自动写入 `runs/.../configs/`，以保证可复现。

### 模型与训练超参（`model_config.yaml`）

```yaml
# 任务与数据
data_path: data/merged/full_merged.pkl
targets:                             # 多目标按需要列出
  - target_binarytrend
  - target_logreturn
  - target_logsharpe_ratio
  - target_drawdown_prob
categoricals: [symbol, period]
reals:
  - open; high; low; close; volume   # 示例；实际请替换为你的特征列清单
  - rsi; macd_hist; atr; obv
  - rolling_volatility; parkinson_volatility; ewma_volatility
  - tail_bias; vol_skewness; return_skewness; amplitude_range; atr_slope

# 序列窗口
max_encoder_length: 96
max_prediction_length: 1
time_idx: time_idx                   # 或 timestamp_int / 累计步索引
group_ids: [symbol, period]

# 模型结构（TFT）
hidden_size: 256
lstm_layers: 2
dropout: 0.15
attention_head_size: 4
static_categoricals: [symbol]        # 静态类别特征（如需）
static_reals: []
x_categoricals: [symbol, period]     # 动态类别特征
x_reals: [...]                       # 动态数值特征
output_size:                         # 每个任务的输出维度（多任务=多head），一般为1
  - 1
  - 1
  - 1
  - 1

# 训练超参
seed: 2025
batch_size: 256
max_epochs: 50
optimizer: adam
lr: 1.5e-3
weight_decay: 1e-5
gradient_clip_val: 0.5
precision: "16-mixed"                # AMP
num_workers: 8

# 学习率调度（OneCycleLR）
onecycle:
  pct_start: 0.3
  div_factor: 25
  final_div_factor: 100

# 恢复/微调
resume_ckpt: null                    # train_resume.py 使用
warm_start_ckpt: null                # warm_start_train.py 使用

# 日志/回调
monitor: val_composite_score
monitor_mode: max
top_k: 3
save_last: true
```

### 损失权重（`weights_config.yaml`）

```yaml
# 基础权重（阶段1）
stage_1:
  target_binarytrend: 1.0
  target_logreturn: 1.0
  target_logsharpe_ratio: 1.0
  target_drawdown_prob: 1.0

# 阶段切换（可选阶段2），示例：加强分类目标
stage_2:
  target_binarytrend: 1.5
  target_logreturn: 0.8
  target_logsharpe_ratio: 1.0
  target_drawdown_prob: 1.0

# 在 utils/loss_scheduler.py 中按 epoch 触发：
# e.g. epoch >= 20 -> 切换到 stage_2
```

### 复合指标权重（`composite_score.yaml`）

> Composite Score 会对**回归/分类指标**进行归一化后加权求和，并用于 `CustomCheckpoint` 的监控与保存。
> 推荐显式按 **period** 标注（`@1h/@4h/@1d`），与日志键一致（如：`val_target_binarytrend_f1@1h`）。

```yaml
# 示例：兼顾分类F1、回归RMSE以及整体val_loss
val_target_binarytrend_f1@1h: 0.25
val_target_binarytrend_f1@4h: 0.15
val_target_logreturn_rmse@1h: -0.20      # RMSE 越小越好，可用负权或在normalize中反向
val_target_logsharpe_ratio_mae@1d: -0.10
val_loss_for_ckpt: -0.30
```

> 注：若某些 `(target@period)` 在当前数据集中不存在，`utils/composite.py` 会自动跳过并给出一次性警告。

---

## 日志与可视化

* **TensorBoard**：日志写入 `runs/.../lightning_logs/`
  启动方式：

  ```bash
  tensorboard --logdir runs
  ```
* **图表记录**（在 `model/tft_module.py` 内实现）：

  * 每 5 个 epoch：

    * 各回归目标的 **Pred vs True 散点图**；
    * 各分类目标的 **混淆矩阵**。
* **阶段汇总**：`utils/stage_summary.py` 将每一阶段的最佳 `val_composite_score` 与 `val_loss` 记录到 `stage_logs/*.yaml`。

---

## 数据规范与要求

* **时间列**：`timestamp`（毫秒）与/或 `datetime`（UTC），内部常用整数时间索引 `time_idx`；
* **长表结构（推荐）**：

  * 特征列**不带** `{symbol}_{period}` 后缀；
  * 使用 `symbol` 与 `period` 两列标记；
  * 慢频（1d/4h）字段向快频（4h/1h）**下放**（repeat / ffill），确保小周期样本完整；
  * 宏观/全市场变量（如 M2、ETF 指标）在同一时间戳对**所有 symbol 使用同一数值**，不留空。
* **目标字段**：以 `target_` 前缀命名；未来收益类**仅作为目标**不作为输入；分类目标不参与 LOF；异常检测只作为提示特征。
* **最小序列长度**：`max_encoder_length + max_prediction_length`（默认 97）。任何 `(symbol, period)` 组不足此长度，将无法构造有效样本。
* **类别编码**：对 `symbol/period` 使用 `NaNLabelEncoder`，确保验证集中只出现训练识别过的类别，避免“未知类别→NaN”。

---

## 技巧与实践建议

* **显存与批大小**：

  * 约 **30万行 × 100+特征** 的数据，建议 `batch_size` 从 128\~256 起步；
  * 若 OOM，可优先 **降低 batch**（比盲目减模型宽度更稳），或使用 `precision="16-mixed"`。
* **OneCycleLR**：

  * 前期快速探索学习率空间，后期平滑收敛。`pct_start` 可 0.2\~0.4 调整；
  * **Warm-Start** 不需要重走 OneCycle 的 warmup，可直接以较小 LR 进入细调。
* **损失设计**：

  * 尖峰/重尾回归目标优先 `SmoothL1Loss`；
  * 类别极不平衡时开启 `WeightedBinaryCrossEntropy(auto_pos_weight=True)`；
  * 分阶段提高关键任务权重（如先齐头并进，再强化 `binarytrend`）。
* **Composite Score**：

  * 指标加权要“可对比”：对 RMSE/MAE 这类“越小越好”的指标，要么使用负权，要么在归一化逻辑中做反向处理；
  * 建议将 `val_loss_for_ckpt` 也纳入一部分比例（例如 0.2\~0.4）以稳定保存逻辑。
* **可复现性**：

  * 固定 `seed`，并记录运行时 `configs/` 快照；
  * 尽量固定 **PyTorch Forecasting** 与 **TorchMetrics** 版本，防止指标口径被动变化。

---

## 故障排查（FAQ）

* **`TimeSeriesDataSet` 组长度不足导致 `default_collate` 报错**

  * 说明：某些 `(symbol, period)` 组样本数 < 97（默认）。
  * 处理：提前筛除过短组，或缩短 `max_encoder_length`。

* **类别编码出现未知（验证集中 symbol/period 未出现于训练）**

  * 说明：部分 symbol 仅出现在验证集。
  * 处理：合并划分前先检查分布；或在构建数据集时显式声明所有可见类别集合。

* **`TorchNormalizer` 或版本兼容报错（pytorch-forecasting）**

  * 说明：不同环境版本差异导致导入失败。
  * 处理：统一版本，确保训练与分析工具使用相同的 `pytorch-forecasting` 版本。

* **`val_composite_score` 未生效或指标缺失**

  * 说明：`composite_score.yaml` 中列出的 `(target@period)` 与实际日志键不匹配。
  * 处理：检查日志键名（如 `val_target_logreturn_rmse@1h`），并与 `utils/composite.py` 规则保持一致。

* **`OneCycleLR` 与 Resume/Warm-Start 的衔接**

  * `train_resume.py`：恢复调度器状态，继续曲线；
  * `warm_start_train.py`：仅加载权重，重新创建优化器与调度器（可选较小 LR）。

---

## 其他脚本

* `src/indicators.py`：MA / RSI / MACD / KDJ / Bollinger / ATR / ADX 等常用指标与派生信号；
* `src/groupwise_rolling_norm.py`：分组滑动窗口归一化（z-score、Min-Max）；
* `src/csv2Parquet.py` / `src/csv2Pkl.py`：CSV 转换工具；
* `src/catch_stocks.py` / `src/ccatch.py`：历史行情抓取示例（可作为数据源参考）；
* `eval_multi_tft.py`（若提供）：

  * 自动选择 top-k checkpoint（或 `last.ckpt`）进行验证集推理；
  * 导出预测 vs 真实的 CSV，便于离线分析。

### 特征筛选（分周期&分目标 → 核心集）

已内置基础特征筛选流水线：

1) 树模型重要度 + 置换重要度（按周期、按目标）

```bash
python -m tft.features.selection.tree_perm --val-mode ratio --val-ratio 0.2 --preview 10 \
  --out reports/feature_evidence/tree_perm
```

2) 跨周期聚合 & 统一核心集（可选择融合 TFT gating 打分作为加分项）

```bash
python -m tft.features.selection.aggregate_core --in reports/feature_evidence/tree_perm \
  --weights configs/weights_config.yaml --topk 128 \
  --tft-file reports/feature_evidence/tft_gating.csv --tft-bonus 0.15 \
  --out-summary reports/feature_evidence/aggregated_core.csv \
  --out-allowlist configs/selected_features.txt
```

3) 一键串联（可选）：

```bash
python -m tft.features.selection.run_pipeline --val-mode ratio --val-ratio 0.2 --topk 128
```

4) TFT gating（可选，如有已训练 checkpoint）：

```bash
python -m tft.features.selection.tft_gating --ckpt lightning_logs/<run>/checkpoints/epoch=..-loss=..ckpt \
  --out reports/feature_evidence/tft_gating.csv
```

流水线会导出 `configs/selected_features.txt`，训练数据加载器会自动读取该白名单，仅保留入选特征参与训练。

---

## 清理旧实验

```bash
rm -r runs/train_multi_tft-202507*    # 删除特定时间前的 runs
```

---

## 致谢

本项目基于并致敬以下开源生态：

* **PyTorch**
* **PyTorch Lightning**
* **PyTorch Forecasting**
* **TorchMetrics**
* 以及 **pandas**、**scikit-learn**、**Optuna** 等社区贡献

---

## 许可

根据你的项目需要添加许可证（如 MIT/Apache-2.0 等）。若暂不指定，可先以 “All rights reserved.” 作为占位。




