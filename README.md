# tft_module

多目标 Temporal Fusion Transformer (TFT) 项目，面向加密/股票等金融时间序列的多周期、多任务预测。

当前版本已实现：

- 多任务输出与混合损失（分类 + 回归），支持不确定性自适应加权的 HybridMultiLoss；
- (symbol, period) 粒度的回归目标标准化与验证/推理期反标准化；
- 专家组（experts）目标集管理：通过 `configs/targets.yaml` 切换不同任务组合；
- 阶段性损失权重调度（支持标量/列表/按目标名字典）；
- 指标体系自动展开（按目标 × 周期），记录二分类混淆矩阵与回归散点；
- 数据加载自动统计均值/方差、兼容空值与组长校验、白名单特征过滤；
- GPU + AMP 训练、OneCycleLR 学习率调度、Warm-Start 与 Resume；
- 统一的日志/权重输出目录（按 expert 分桶）。

适配 1h / 4h / 1d 等多周期数据，支持同时预测分类与回归目标（涨跌、对数收益、回撤概率、趋势持续等）。

---

## 目录结构（精简）

```
tft_module/
├─ train_multi_tft.py            # 从零训练（按 expert）
├─ train_resume.py               # 恢复 checkpoint 继续训练
├─ warm_start_train.py           # 仅载入模型权重做微调
├─ scripts/
│  └─ inspect_batch.py           # 调试：打印一个 batch 的 x/y 结构
├─ models/
│  └─ tft_module.py              # MyTFTModule + HybridMultiLoss（核心模型）
├─ data/
│  └─ load_dataset.py            # 读取 full_merged.pkl → TimeSeriesDataSet/DataLoader
├─ callbacks/
│  └─ (空)                       # 如需自定义回调可在此添加
├─ utils/
│  ├─ loss_factory.py            # 损失工厂（分类→加权BCE；回归→SmoothL1/MSE）
│  ├─ metric_factory.py          # 指标工厂（按目标×周期展开）
│  └─ weighted_bce.py            # 加权 BCE（batch 级 auto_pos_weight）
├─ features/
│  └─ selection/                 # 特征筛选流水线（树/置换/TFT gating/聚合/优化/滚验）
└─ configs/
   ├─ experts/                   # 按 专家/周期/模态 的叶子目录（自包含）
   │   └─ <Expert>/<period>/<modality>/
   │        ├─ model_config.yaml   # 训练超参（数据路径/epoch/优化器…）
   │        └─ targets.yaml        # 本叶子训练的目标集合 + 权重（替代单独 weights_config.yaml）
   ├─ weights_config.yaml        # 兼容兜底（不再强制使用）
   ├─ targets.yaml               # 全局专家清单（参考/工具），训练时不再依赖
   └─ selected_features.txt      # 全局白名单（叶子优先）
```

---

## 数据处理流程

整体流程（从原始 K 线到可训练数据）：

```
原始K线/衍生数据
   │
   ├─▶ src/indicating.py                 # 计算技术指标、可选LOF异常检测、分组滑动归一化
   │
   ├─▶ src/target_config.py              # 构建多任务预测目标（分类/回归，不同周期）
   │
   ├─▶ src/data_fusion.py                # 多周期字段对齐/合并，产出 full_merged.pkl
   │
   └─▶ data/load_dataset.py              # 构建 TimeSeriesDataSet & DataLoader（统计标准化元数据）
```

### 数据加载（data/load_dataset.py）

读取 `data/pkl_merged/full_merged.pkl`（或自定义路径），构建 `TimeSeriesDataSet` 与 `DataLoader`：

- 自动统计 (symbol, period, 部分回归 target) 的均值/方差，打包为 `norm_pack`；
- 目标在 TFT 内部使用 `identity` normalizer（不居中不缩放），由模型外部按组做标准化；
- 默认窗口：`max_encoder_length=36`，`max_prediction_length=1`；
- 支持 `val_mode=days|ratio` 两种划分；
- 自动读取 `configs/selected_features.txt` 白名单过滤输入特征；
- 校验 NaN 与组长度，降低 `_getitem__=None` 引发的 `default_collate` 报错。

默认输出：

```
train_loader, val_loader, target_names, train_ds, period_classes, norm_pack
```

---

## 模型与代码逻辑

核心文件：`models/tft_module.py`

- TemporalFusionTransformer：通过 `from_dataset()` 构建，内置一个“安全”MultiLoss 以满足库内部断言，真实训练损失在外部计算；
- HybridMultiLoss：
  - 基础损失由 `utils/loss_factory.py` 决定（分类→WeightedBCE；回归→SmoothL1/MSE）；
  - 乘以“基础权重”后，再叠加可学习的 log_var（不确定性）项，实现自适应加权；
- y 解析与标准化：
  - 兼容 y 为 Tensor / list[Tensor] / (y, weight) 等多形态，自动拆分为 per-target 向量；
  - 对回归目标按 (symbol, period) 使用 `norm_pack` 做标准化/反标准化；
- 指标体系（`utils/metric_factory.py`）：
  - 分类：F1/Accuracy/Precision/Recall/ROC-AUC/AP；回归：RMSE/MAE；
  - 日志键自动按 `target@period` 展开（默认 1h/4h/1d），并带 `val_` 前缀；
  - 二分类任务记录混淆矩阵；回归任务记录 Pred vs True 散点；
- 学习率与训练：
  - OneCycleLR（按 steps_per_epoch 或总步数估算）；
  - AMP、梯度裁剪与梯度累积；
- 阶段性权重调度：
  - `loss_schedule` 支持标量/列表/字典三种形式；
  - 列表长度与目标数不一致时将忽略并打印警告（不再报错）。

推理接口：`predict_step` 返回

```
{
  'preds_std': 标准化空间预测,
  'preds_orig': 原始空间预测,
  'preds_antilog': 针对 logreturn 的 expm1 反变换,
  'future_price': 基于 close 与 logreturn 的未来价格估计
}
```

---

## 专家组（experts）与配置说明

### 专家组逻辑（参考，全局 configs/targets.yaml）

说明：训练流程现已改为“每个叶子自带 targets.yaml 决定目标集合与权重”，全局 `configs/targets.yaml` 仅保留作参考/工具，不再被训练脚本使用。以下示例展示传统写法：

```yaml
default_expert: alpha_reg

experts:
  alpha_reg:        # 单目标回归（对数收益）
    model_type: tft
    targets: [target_logreturn]

  alpha:            # 纯分类多任务
    model_type: tft
    targets:
      - target_binarytrend
      - target_pullback_prob
      - target_sideway_detect
      - target_drawdown_prob

  tech_reg:         # 纯回归多任务
    model_type: tft
    targets:
      - target_logreturn
      - target_logsharpe_ratio
      - target_breakout_count
      - target_max_drawdown
      - target_trend_persistence

  tech_clf:         # 技术类分类任务集
    model_type: tft
    targets:
      - target_binarytrend
      - target_pullback_prob
      - target_sideway_detect
      - target_drawdown_prob
```

训练脚本会按 expert 过滤/重排指标与损失，日志与 checkpoint 也会按 expert 分类保存。

### 叶子配置（model_config.yaml + targets.yaml）

关键字段：

- `expert`: 仅作为标注（建议与叶子路径一致）。
- `data_path`: 训练数据路径（`get_dataloaders` 内部有默认兜底为 `data/pkl_merged/full_merged.pkl`）；
- `batch_size/num_workers/precision/accumulate/max_epochs/grad_clip` 等常规超参；
- `val_mode`: `days|ratio`；对应 `val_days/val_ratio`；
- `loss_schedule`: 按 epoch 配置阶段性损失权重，支持：
  - 标量（或单元素列表）：对所有目标广播同一权重；
  - 与目标数一致的列表：按顺序应用；
  - 字典：以目标名为键，适配不同专家的目标集合；
  长度不匹配时将忽略并保留原权重，同时打印警告。

### 损失权重（configs/weights_config.yaml）

两种写法（推荐“按目标名”以适配专家切换）：

```yaml
# 推荐：按目标名映射（与 expert 无关，更稳健）
custom_weights_by_target:
  target_logreturn: 3.0
  target_logsharpe_ratio: 2.0
  target_breakout_count: 1.5

# 备选：按列表（需与当前目标数一致）
custom_weights: [1.0, 1.0, 1.0]
```


---

## 训练逻辑（脚本工作流）

- `train_multi_tft.py`：
  - 需要 `--config` 指向 `configs/experts/<Expert>/<period>/<modality>/model_config.yaml`，并读取就近 `weights_config.yaml` 与全局 `targets.yaml`；
  - `data/load_dataset.get_dataloaders()` 生成 `train/val` 数据加载器与 `TimeSeriesDataSet`；
  - 计算 `steps_per_epoch` 与有效步数（考虑梯度累积），创建 `MyTFTModule`；
  - 回调：统一使用验证损失 `val_loss_epoch` 作为监控与早停指标；
  - 目录：`lightning_logs/experts/<expert>/<period>/<modality>/tft/` 与 `checkpoints/experts/<expert>/<period>/<modality>/tft/`；
  - AMP 与 cudnn benchmark 默认开启。

- `train_resume.py`：恢复 checkpoint（包含优化器与调度器状态）继续训练。

- `warm_start_train.py`：仅加载模型权重，重新初始化优化器/调度器，适合新数据迁移微调。

（已去除复合指标逻辑，训练仅以损失作为评估与保存依据）

targets.yaml 示例（每个叶子一份）

```yaml
model_type: tft
targets: [target_binarytrend]
weights:
  by_target:
    target_binarytrend: 1.0
# 或
# weights:
#   custom: [1.0, 2.0, ...]  # 与 targets 顺序一致
```

### 便捷 CLI（按专家快速启动）

提供 `scripts/experts_cli.py` 作为简单命令行入口：

- 列出所有叶子：
  ```bash
  python scripts/experts_cli.py list
  ```
- 训练/续训/热启动（自动拼出叶子路径并校验 weights）：
  ```bash
  # 训练 Alpha-Dir-TFT@1h/base（两种写法）
  python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1h/base
  # 或
  python scripts/experts_cli.py train --expert Alpha-Dir-TFT --period 1h --modality base

  # 续训 Alpha-Ret-TFT@4h/base
  python scripts/experts_cli.py resume --expert Alpha-Ret-TFT --period 4h --modality base

  # 热启动（迁移微调）
  python scripts/experts_cli.py warm --expert Risk-TFT --period 1h --modality base
  ```
  可用 `--experts-root` 指定自定义根目录（默认 `configs/experts`）。

---

## 使用方法（一步到位）

1) 准备数据：
   - 按 `src/` 流水线生成 `data/pkl_merged/full_merged.pkl`（或自行提供等价长表）；
   - 确认包含 `symbol, period, datetime/timestamp, time_idx, 特征列, 目标列(target_*)`；
   - 可选：准备 `configs/selected_features.txt` 作为白名单（每行 1 个列名）。

2) 选择专家与超参（按专家配置）：
   - 直接选择对应的配置文件：`configs/experts/<Expert>/<period>/<modality>/model_config.yaml`；
   - 示例：`configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml` / `configs/experts/Alpha-Ret-TFT/4h/base/model_config.yaml`；
   - 如需覆盖专家名称，可加 `--expert` 或设置环境变量 `EXPERT`；
   - 其余超参在该 `model_config.yaml` 中调整。

3) 配置损失：
   - 在 `configs/weights_config.yaml` 使用 `custom_weights_by_target`（建议）或 `custom_weights`；
   - 在 `model_config.yaml` 配置 `loss_schedule`（标量/列表/字典均可）。

4) 开始训练（直接传入专家配置）：
   ```bash
   conda activate tft
   python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
   # 续训/热启动也同理：
   python train_resume.py --config configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
   python warm_start_train.py --config configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
   ```

5) 可视化与检查点：
   - `tensorboard --logdir lightning_logs`；
   - 检查点在 `checkpoints/experts/<expert>/tft/`；
   - 继续训练：`python train_resume.py`；
   - 迁移微调：`python warm_start_train.py`。

---

## 日志与可视化

- TensorBoard：日志写入 `lightning_logs/experts/<expert>/<period>/<modality>/tft/`
- 启动命令：
  ```bash
  tensorboard --logdir lightning_logs
  ```
- 记录内容：
  - 标量：训练/验证损失，分类/回归指标（按目标×周期展开）；
- 图表记录（在 `models/tft_module.py` 内实现）：
  - 回归：Pred vs True 散点；
  - 分类：混淆矩阵；
- 阶段汇总（如需）：可在回调中记录阶段性最优 `val_loss` / 复合指标。

---

## 调试与工具

- `scripts/inspect_batch.py`：打印一个 batch 的 `x/y` 结构（用于排查 y 形态、None、数值异常）；
- 模型内部对 y 解析已做兼容（Tensor / list / (y, weight)），维度不一致将给出明确报错；
- 若切换 expert 导致目标数变化，优先使用基于目标名的权重映射（`custom_weights_by_target`）。

---

## 常见问题（FAQ）

- TimeSeriesDataSet 组长度不足导致 `default_collate` 报错：
  - 某些 `(symbol, period)` 组样本数 < `max_encoder_length + max_prediction_length`；
  - 处理：提前筛除过短组，或缩短窗口（例如 encoder=24/36）。

- 类别编码出现未知（验证集中 symbol/period 未出现在训练）：
  - 处理：合并划分前检查分布；或在构建数据集时显式声明所有可见类别集合。

- Lightning/PyTorch Forecasting 版本兼容：
  - 处理：统一版本；若导入失败，按报错信息固定库的主次版本。

- `loss_schedule` 长度与目标数不匹配：
  - 现已仅给出警告并忽略该 epoch 的调度（不再报错）；
  - 建议改为标量或按目标名字典配置。

---

## 许可

请按项目需要添加许可证（如 MIT/Apache-2.0 等）。
