# tft_module

多目标 Temporal Fusion Transformer (TFT) 项目，面向加密货币 / 股票等金融时间序列的多周期、多任务预测。项目采用9+1专家架构，包含Alpha-Dir、Alpha-Ret、Risk-Prob、Risk-Reg、MicroStruct-Deriv、OnChain-ETF、Regime-Gate、RelativeStrength-Spread、Factor-Bridge等9个专业专家模型，以及一个Z-Combiner融合层。

## 端到端命令速查（从采集 → 融合 → 筛选 → 训练）

1) 采集与准备（API → 原始数据）

- 衍生品/链上指标（时间段、币种、周期可配）

  ```bash
  # 例：按时间范围与币种抓取 1d 指标
  python src/apied.py --start 2020-10-02 --end 2025-12-18 \
  --symbols BTC,ETH,SOL,BNB,ADA --interval 1d
  ```

  参数（来自实际代码 src/apied.py）
  - `--symbols`：逗号分隔，默认 `BTC,ETH,SOL,BNB,ADA`
  - `--interval`：`1h|4h|1d`，默认 `1d`
  - `--limit`：单次 API 上限，默认 `4500`
  - `--pause`：每次调用后的 sleep 秒数，默认 `1.0`
  - `--start`：起始时间；支持 13 位毫秒 / 10 位秒 / `YYYY-MM-DD` / `YYYYMMDD` / ISO8601 / `now`，默认 `2020-10-01`
  - `--end`：终止时间；同上，默认 `now`

- OHLCV/K线（编辑 `src/ccatch.py` 顶部常量后运行）

  ```bash
  python src/ccatch.py
  ```

  关键常量（来自实际代码 src/ccatch.py 顶部）
  - `BASE_SYMBOLS=['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','ADA/USDT']`
  - `TIMEFRAMES=['1h','4h','1d']`
  - `SINCE_STR='2020-10-01T00:00:00Z'`、`LIMIT=1500`、`PRICE_MODE='trade'`、`EXCHANGE_TYPE='usdm'`
  - 输出目录：`data/crypto/<symbol>/<timeframe>/...csv`（按脚本内部组织）

1. 构建技术面与目标（full_merged.csv）

```bash
python src/build_full_merged.py --periods 1h,4h,1d
# 输出：data/merged/full_merged.csv
# 如需生成 PKL：
#   python src/csv2Pkl.py --src data/merged/full_merged.csv --dst data/pkl_merged/full_merged.pkl
```

参数（来自实际代码 src/build_full_merged.py）
- `--periods`：空格或逗号分隔，默认 `1h 4h 1d`（PowerShell 建议加引号，如 `'1h,4h,1d'`）
- `--indicated-root`：技术面输入根目录，默认 `data/crypto_indicated`
- `--targeted-root`：目标输出根目录，默认 `data/crypto_targeted_and_indicated`
- `--merged-out`：最终合并 CSV 路径，默认 `data/merged/full_merged.csv`

1. 融合 Base/Rich（生成 full_merged_with_fundamentals.csv 及专家分组视图）

```bash
# 方式一：直接运行脚本（推荐，更简洁）
python src/fuse_fundamentals.py --config pipelines/configs/fuse_fundamentals.yaml
# 或使用默认配置：
python src/fuse_fundamentals.py

# 方式二：Python -c 调用
python -c "from src.fuse_fundamentals import run_with_config; \
  run_with_config('pipelines/configs/fuse_fundamentals.yaml')"

# 关键配置：pipelines/configs/fuse_fundamentals.yaml
# 输出：
# - data/merged/full_merged_with_fundamentals.{csv,pkl}
# - data/merged/expert_group/<Expert>_{base|rich}/full_merged_with_fundamentals.{csv,pkl}
```

说明（来自实际代码 src/fuse_fundamentals.py 与 datasets 配置）
- `configs/experts/<Expert>/datasets/{base,rich}.yaml`：
  - `pinned_features`：默认保留特征
  - `feature_list_path`：筛选特征清单路径（固定指向 `reports/feature_evidence/.../selected_features.txt`）
  - `include_symbol/include_global`：Rich 模块选择开关

1. 特征筛选（reports/feature_evidence + selected_features.txt）

```bash
# 全量筛选（所有专家，Base+Rich双通道）
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml

# 单个专家筛选
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --experts alpha_dir_tft

# 多个专家筛选（逗号分隔）
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --experts alpha_dir_tft,risk_prob_tft,derivatives_micro

# 指定通道筛选
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --experts alpha_dir_tft --enable-base
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --experts alpha_dir_tft --enable-rich
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --experts alpha_dir_tft --enable-base --enable-rich

# 快速测试版（仅2个专家，1h周期，禁用时间置换）
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection_quick.yaml

# 跳过预聚合步骤（使用已有证据）
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --skip-pre-aggregation
```

**参数说明：**
- `--config`：配置文件路径，默认使用主配置或quick配置
- `--experts`：指定专家键（逗号分隔），可选值：`alpha_dir_tft`, `alpha_ret_tft`, `risk_prob_tft`, `risk_reg_tft`, `derivatives_micro`, `onchain_fundflow`, `regime_gating`, `breakout_levels`, `relative_strength`, `factor_bridge`
- `--enable-base`：仅启用Base通道筛选
- `--enable-rich`：仅启用Rich通道筛选
- `--skip-pre-aggregation`：跳过过滤和嵌入步骤，直接使用现有tree_perm结果进行聚合

**产物：**
- `reports/feature_evidence/<Expert>/<base|rich|comprehensive>/<period>/selected_features.txt`：各专家各通道各周期的筛选结果（例如：`reports/feature_evidence/Alpha-Dir-TFT/base/1h/selected_features.txt`）
- `reports/feature_evidence/<Expert>/<base|rich|comprehensive>/<period>/aggregated_core.csv`：各周期的聚合评估表和证据链
- `reports/feature_evidence/allowlist_core_common.txt`：跨专家跨周期的共同核心特征

**训练集成：**训练时从 `configs/experts/<Expert>/datasets/<base|rich>.yaml` 的 `feature_list_path` 字段读取对应的 selected_features.txt 文件；训练脚本会根据当前训练的周期自动选择对应的特征文件（例如：`reports/feature_evidence/Expert/base/1h/selected_features.txt`）

1. 开始训练（按专家/周期/模态）
```bash
# 列出叶子配置
python scripts/experts_cli.py list

# 训练 / 续训 / 热启动（示例：Alpha-Dir-TFT 的 4h/base）
python scripts/experts_cli.py train  --expert Alpha-Dir-TFT --leaf 4h/base
python scripts/experts_cli.py resume --expert Alpha-Dir-TFT --leaf 4h/base
python scripts/experts_cli.py warm   --expert Alpha-Dir-TFT --leaf 4h/base
```

注意：
- 续训需要在叶子 `model_config.yaml` 设置 `resume_ckpt: <path/to/ckpt>`（脚本读取该字段）。
- 热启动需要设置 `warm_start_ckpt: <path/to/ckpt>`（仅加载模型权重，不恢复优化器/epoch）。
- 训练配置：`configs/experts/<Expert>/<period>/<modality>/model_config.yaml`
  - 关键字段：`batch_size, learning_rate, max_epochs, min_epochs, devices, monitor=val_loss,` 以及
    `max_encoder_length, max_prediction_length, add_relative_time_idx, allow_missing_timesteps` 等。
- 目标与输出头：`configs/experts/<Expert>/<period>/<modality>/targets.yaml`
  - 例如 `targets: [target_binarytrend]`、`output_head: { type: per_symbol_affine, apply_on: logits }`
- 数据集 pinned/selected：`configs/experts/<Expert>/datasets/{base,rich}.yaml`
  - `pinned_features` 默认特征；`feature_list_path` 指向 `reports/feature_evidence/.../selected_features.txt`

1. 训练后（可选）构建 OOF 与 Z 层训练数据
```bash
python pipelines/build_oof_for_z.py \
  --predictions-root lightning_logs \
  --data-path data/pkl_merged/full_merged.pkl \
  --output datasets/z_train.parquet
```

1. 离线评估（预告）
```bash
# 计划中的离线评估脚本（预告，脚本暂未提供，示例仅作参考）
# python scripts/offline_eval.py \
#   --predictions lightning_logs/experts/<expert>/<period>/<modality>/tft/<log_name>/version_*/predictions/*.parquet \
#   --targets-pkl data/pkl_merged/full_merged.pkl \
#   --metrics ap,roc,rmse,mae --calibration temperature
```

## 最新亮点（v0.2.8）

- **专家体系完整实现**：9+1专家架构全面落地，涵盖方向预测、收益回归、风险评估、微观结构、链上数据、市场体制、关键位突破、相对强弱、因子桥接等全领域专业能力
- **Base/Rich并行融合**：长历史基础特征与近年丰富模态并行训练，α门控自适应加权，缺失时自然回退
- **多证据特征选择**：四阶段筛选管线（过滤→内嵌→时间置换→包装搜索），支持TFT-VSN权重、跨时代稳定性评估
- **防泄露数据处理**：慢频数据shift→ffill广播，严格的时序审计，确保训练数据无未来信息泄露
- **嵌套时序CV**：内层特征选择与调参，外层滚动前瞻评估，确保评估无泄露
- **统一预测契约**：标准化输出`{score, uncertainty, meta}`，支持多符号多头架构，自动校准与可靠性评估
- **Z层智能组合**：基于专家预测和市场状态的动态权重分配，支持风格约束和风险制动

### 训练与特征筛选体验（v0.2.8 增强）

- **XGBoost 2.x兼容**：所有特征筛选和树模型统一使用`tree_method="hist" + device="cuda"`，旧版自动回退
- **专家化配置管理**：每个专家×周期×模态自包含完整配置，支持就近查找和自动推断
- **智能监控指标**：分类任务自动选择`val_*_ap@period`，回归任务选择`val_*_rmse@period`作为早停指标
- **特征筛选高级特性**：
  - TFT-VSN重要度纳入综合打分
  - 跨时代稳定性评估与出现率过滤
  - GA多seed支持与RFE近似器优化
  - Base/Rich质量权重自适应缩放
  - 前瞻对照验证确保无泄露
- **训练日志优化**：进度条默认开启，可配置日志频率，自动保存`last.ckpt`

## 专家数据管理（Base/Rich并行）

- **专家数据集配置**：每位专家在`configs/experts/<Expert>/datasets/`下维护base/rich两套配置
  - `base.yaml`：基础技术面数据，`dataset_type=fundamentals_only`，适合长历史覆盖
  - `rich.yaml`：包含链上/ETF/衍生品数据，`dataset_type=combined`，使用交集策略确保数据质量
  - `comprehensive.yaml`：完整特征集合，结合base和rich的优点

- **融合执行**：通过`pipelines/configs/fuse_fundamentals.yaml`统一调度，支持多专家并行融合

  ```bash
  # 方式一：直接运行脚本
  python src/fuse_fundamentals.py --config pipelines/configs/fuse_fundamentals.yaml
  # 或
  python src/fuse_fundamentals.py
  
  # 方式二：Python -c 调用
  python -c "from src.fuse_fundamentals import run_with_config; run_with_config('pipelines/configs/fuse_fundamentals.yaml')"
  ```

- **输出结构**：`data/merged/expert_group/<Expert>_{base|rich|comprehensive}/` 按专家和模态分组存放
- **Pinned特性**：每位专家可定义默认保留字段（pinned_features），在特征筛选时优先保留，确保核心特征不被误删

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

## 专家架构总览（9+1体系）

项目采用完整的9+1专家架构，涵盖金融时间序列预测的全领域能力：

### 9个专业专家

1. **Alpha-Dir-TFT** - 方向预测（二分类：上涨/下跌概率）
2. **Alpha-Ret-TFT** - 收益回归（预期收益幅度预测）
3. **Risk-Prob-TFT** - 风险概率（极端风险事件概率）
4. **Risk-Reg-TFT** - 风险回归（回撤/波动幅度预测）
5. **MicroStruct-Deriv-TFT** - 微观结构（资金费率、持仓变化等）
6. **OnChain-ETF-TFT** - 链上/ETF资金流（链上数据与ETF动向）
7. **Regime-Gate** - 市场体制识别（趋势/震荡/高波动/危机状态）
8. **KeyLevel-Breakout-TFT** - 关键位突破（支撑/阻力突破概率）
9. **RelativeStrength-Spread-TFT** - 相对强弱（跨币种相对表现）

### Z-Combiner融合层
- 基于各专家预测结果的智能加权组合器
- 支持规则融合、Stacking元学习、动态门控等多种策略
- 集成风险控制和风格约束

## 目录结构（核心）

```
tft_module/
├─ train_multi_tft.py            # 专家训练入口
├─ train_resume.py               # 续训 checkpoint
├─ warm_start_train.py           # Warm-start 微调
├─ experts/
│  └─ Z-Combiner/
│       ├─ model_config.yaml     # Z 层训练配置
│       └─ train_z.py            # OOF → Stacking 训练脚本
├─ configs/experts/              # 专家配置体系
│   ├─ Alpha-Dir-TFT/
│   │   ├─ 1h/base/             # 1小时基础模态
│   │   ├─ 1h/rich/             # 1小时丰富模态
│   │   ├─ 1h/comprehensive/    # 1小时完整模态
│   │   ├─ 4h/...               # 4小时各模态
│   │   ├─ 1d/...               # 1天各模态
│   │   └─ datasets/            # 数据集配置
│   └─ [其他8个专家]/
├─ models/
│  └─ tft_module.py              # MyTFTModule + HybridMultiLoss
├─ pipelines/
│  ├─ build_oof_for_z.py         # 汇总专家输出生成 z_train.parquet
│  └─ configs/
│       ├─ fuse_fundamentals.yaml    # 数据融合配置
│       └─ feature_selection.yaml    # 特征筛选配置
├─ features/
│  ├─ regime_core.py             # Regime 核心特征计算
│  └─ selection/                 # 特征筛选管线
├─ metrics/
│  └─ calibration.py             # 温度缩放 / ECE / Brier / Reliability
├─ utils/
│  ├─ eval_report.py             # per-symbol × period 指标汇总
│  ├─ audit_no_leakage.py        # OOF 数据检查
│  ├─ mp_start.py                # Windows 多进程启动补丁
│  └─ ...                        # 其他工具函数
├─ scripts/
│  ├─ dump_batch.py              # 调试 DataLoader batch
│  ├─ experts_cli.py             # 按专家快速启动
│  └─ [其他辅助脚本]
└─ data/
   ├─ merged/
   │   ├─ full_merged.csv           # 基础 K 线 + 技术指标
   │   └─ expert_group/             # 按专家分组的融合数据
   └─ [其他数据目录]
```

`data/merged/expert_group/` 是融合脚本的主要输出目录，运行后会为每位专家生成 `<Expert>_base/` 与 `<Expert>_rich/` 两套数据集：

- `full_merged_with_fundamentals.{csv,pkl}`：用于训练/特征筛选的全量列。
- `full_merged_slim.csv`：核心价量 + 基本面 + 目标的精简版。
- `fundamental_columns.txt`：新增列清单。
- `dataset_group_summary.csv`：每个 `(symbol, period)` 的样本数、时间范围，以及 `missing_threshold_rows`、`missing_threshold_cols_count`、`intersect_all_null_cols_count` 等裁剪统计。
- `missing_threshold_columns.txt` / `intersect_columns*.txt`：当缺失率阈值或交集策略触发时的列记录。
- `fuse_audit/`：覆盖率报表、时间审计、列统计；`config_snapshot.yaml` 保存本次融合配置。

### Alpha-Dir-TFT Pipeline

1. 构建数据（已完成）：
   - `python src/build_full_merged.py`
   - `python src/fuse_fundamentals.py`

2. 运行 Alpha-Dir-TFT 特征筛选：
   ```bash
   python pipelines/run_feature_screening.py --experts alpha_dir_tft --enable-base --enable-rich
   ```

3. 训练 Alpha-Dir-TFT 模型：
   - 直接调用训练入口：
     ```bash
     python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1h/base
     python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 4h/base
     python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1d/base

     python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1h/rich
     python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 4h/rich
     python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1d/rich
     ```
   - 或者手动调用 `train_multi_tft.py`：
     ```bash
     python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
     python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/4h/base/model_config.yaml
     python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/1d/base/model_config.yaml

     python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/1h/rich/model_config.yaml
     python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/4h/rich/model_config.yaml
     python train_multi_tft.py --config configs/experts/Alpha-Dir-TFT/1d/rich/model_config.yaml
     ```

## 预测契约与评估

- **标准化输出格式**：
  ```python
  {
      'score': tensor[B, T],           # 预测分数/概率
      'uncertainty': tensor[B, T],     # 不确定性（回归: σ，分类: NaN）
      'meta': {
          'symbol_idx', 'period_idx', 'time_idx',
          'head_scale', 'head_bias'
      },
      'schema_ver', 'data_ver', 'expert_ver', 'train_window_id',
      'antilog_return', 'future_price'
  }
  ```

- **自动校准与评估**：
  - 温度缩放：优化预测概率分布
  - 可靠性指标：ECE、Brier分数、P10/P50/P90覆盖率
  - 分位数损失：Pinball Loss评估预测区间质量
  - 分币种×周期汇总：`eval_report_*.csv` 详细评估报告

- **预测数据管理**：
  - 自动写入 parquet 格式：`lightning_logs/experts/<expert>/<period>/<modality>/tft/<log_name>/version_*/predictions/*.parquet`
  - 包含完整的元信息：符号、周期、时间索引、模型版本等
  - 支持批量预测和增量更新

## OOF集成与Z层训练

1. **收集专家预测**：
   - 各专家训练完成后自动生成预测文件：`lightning_logs/experts/<expert>/<period>/<modality>/tft/predictions/`
   - 包含标准化的预测分数、不确定性和元信息

2. **构建OOF数据集**：
  ```bash
  python pipelines/build_oof_for_z.py \
      --predictions-root lightning_logs \
      --data-path data/pkl_merged/full_merged.pkl \
      --output datasets/z_train.parquet
  ```
   - 自动校验版本一致性，合并Regime特征和专家预测

3. **数据质量审计**：
   ```bash
   python utils/audit_no_leakage.py --path datasets/z_train.parquet
   ```
   - 检查时序一致性、重复数据、异常间隔等潜在泄露风险

4. **训练Z-Combiner**：
   ```bash
   python experts/Z-Combiner/train_z.py --config experts/Z-Combiner/model_config.yaml
   ```
   - 支持多种融合策略：规则加权、Stacking元学习、动态门控
   - 自动与"等权"和"单最佳专家"基线对比
   - 输出详细评估报告：`lightning_logs/experts/Z-Combiner/metrics_*.json`

## 辅助工具

### 核心辅助脚本
- **`scripts/dump_batch.py`**：调试DataLoader，打印batch张量形状和数据格式
- **`utils/mp_start.py`**：Windows多进程启动补丁，确保DataLoader正常工作
- **`scripts/experts_cli.py`**：专家训练CLI工具
  ```bash
  # 列出所有可用配置
  python scripts/experts_cli.py list

  # 训练指定专家配置
  python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1h/base

  # 续训（需在叶子 model_config.yaml 配置 resume_ckpt）
  python scripts/experts_cli.py resume --expert Alpha-Dir-TFT --leaf 1h/base

  # Warm-start微调（需配置 warm_start_ckpt）
  python scripts/experts_cli.py warm --expert Alpha-Dir-TFT --leaf 1h/base
  ```

### 数据处理工具
- **`src/build_full_merged.py`**：一键构建完整的技术面数据集
  ```bash
  python src/build_full_merged.py --periods 1h,4h,1d
  ```

### 特征筛选工具
- **`pipelines/run_feature_screening.py`**：运行完整的特征筛选管线
- **`features/selection/tft_gating.py`**：导出TFT变量选择网络重要度

### 评估与监控工具
- **`utils/eval_report.py`**：生成分币种×周期的详细评估报告
- **`metrics/calibration.py`**：模型校准和可靠性评估工具
- **`utils/audit_no_leakage.py`**：时序数据泄露审计工具

### 一键构建 full_merged.csv（技术面扩展后）

新增 `src/build_full_merged.py` 串联 指标→目标→合并，生成最新的 `data/merged/full_merged.csv`：

```bash
python src/build_full_merged.py                 # 默认 1h,4h,1d
python src/build_full_merged.py --periods 1h,4h # 仅 1h 与 4h
```

流程说明：

- 指标：`indicating_main(tf)` 计算基础 + 扩展技术面（含 Regime/通道/动量/一目/形态/Tail/MFI/CMF/枢轴/时间特征），并对关键列先 `shift(1)` 再归一化（输出 `_zn96/_mm96`、`_zn56/_mm56`、`_zn30/_mm30` 等，依周期而定）。
- 目标：`generate_targets_auto.convert_selected_periods_to_csv` 写入 `data/crypto_targeted_and_indicated/<tf>`。
- 合并：`merged_main` 自动探测币种并输出 `data/merged/full_merged.csv`。

注意：`beta_btc_60d` 与 `volume_profile_hvn/lvn` 暂留于融合阶段或专用模块实现。

## 常见问题

- 训练指标/校准暂时关闭：为避免在小验证窗或极端不平衡时 AP/ROC-AUC 等指标在某些周期出现“无样本/单类”而导致早停或日志告警，当前默认仅记录并监控 `val_loss`（EarlyStopping/Checkpoint 也基于 `val_loss`）。原有分类/回归指标与概率校准（温度缩放、ECE、Brier、可靠性曲线）已在训练阶段禁用，后续待稳定后再按需恢复；如需查看效果，可在训练完成后离线评估。

- **预测 parquet 不生成**：确认已运行 `Trainer.predict(...)` 或在验证结束后调用了 `trainer.predict(...)`。
- **校准指标全为 NaN**：检查对应目标是否有正样本 / 负样本，或是否误把回归目标当分类使用。
- **OOF 数据列缺失**：确保传入的 `full_merged.pkl` 包含所有 `target_*` 列，并且最新预处理已对慢频特征做 shift/ffill。
- **Z-Combiner 指标不升**：可在配置中增减 `feature_prefixes`、替换模型（例如换成 `Ridge`、`GradientBoosting` 等），或针对分类任务追加更多校准步骤。

## 快速上手指南

### 完整工作流（数据准备 → 专家训练 → 融合）

1. **准备基础数据**
   ```bash
   # 构建完整技术面数据集
   python src/build_full_merged.py --periods 1h,4h,1d
   ```

2. **融合专家数据**
   ```bash
   # 为所有专家生成专用数据集
   python src/fuse_fundamentals.py  # 使用默认配置
   # 或指定配置
   python src/fuse_fundamentals.py --config pipelines/configs/fuse_fundamentals.yaml
   ```

3. **特征筛选（可选）**
   ```bash
   # 运行完整特征筛选管线
   python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml
   ```

4. **训练专家模型**
   ```bash
   # 使用CLI工具快速启动训练
   python scripts/experts_cli.py train --expert Alpha-Dir-TFT --leaf 1h/base
   python scripts/experts_cli.py train --expert Alpha-Ret-TFT --leaf 1h/base
   python scripts/experts_cli.py train --expert Risk-Prob-TFT --leaf 1h/base
   # ... 训练其他专家
   ```

5. **生成预测并构建OOF**
   ```bash
   # 各专家预测（训练时自动生成，或单独调用predict）
   # 构建Z层训练数据
    python pipelines/build_oof_for_z.py \
        --predictions-root lightning_logs \
        --data-path data/pkl_merged/full_merged.pkl \
        --output datasets/z_train.parquet
    ```

6. **训练Z-Combiner**
   ```bash
   python experts/Z-Combiner/train_z.py --config experts/Z-Combiner/model_config.yaml
   ```

### 新增专家接入流程

如需接入新的专家模型：

1. **定义数据集配置**：在 `configs/experts/<NewExpert>/datasets/` 下创建 base/rich/comprehensive 配置
2. **注册到融合管线**：在 `pipelines/configs/fuse_fundamentals.yaml` 的 `experts_map` 中注册
3. **配置训练参数**：创建各周期×模态的配置目录和文件
4. **运行特征筛选**：为新专家运行特征选择管线
5. **训练模型**：使用标准训练流程训练新专家
6. **集成到Z层**：将新专家预测纳入Z-Combiner训练

### 预期产出
- ✅ 9个专业专家模型（各周期×模态完整覆盖）
- ✅ Z-Combiner智能融合层
- ✅ 完整的特征证据链和评估报告
- ✅ 防泄露的数据处理和审计机制

---

欢迎在此基础上继续扩展：如接入更多专家、引入分位模型或 GNN/LLM 特征等。

---

## 端到端工作流详解

本项目的工作流设计精良，可以清晰地划分为四个主要阶段：**数据准备** -> **特征筛选** -> **配置同步** -> **模型训练**。

### 第一阶段：数据准备 (Data Preparation)

**目标**: 将多种来源的数据融合成一个可供后续所有步骤使用的、大而全的“宽表”。

**核心细节**:

1. **输入数据源**:
   - **技术面数据**: 这通常是基础，包含了各个交易对的 OHLCV（开高低收成交量）等价格信息。
   - **链上/基本面数据**: 包括但不限于 ETF 资金流、宏观经济指标、链上活跃地址、持仓量（OI）、资金费率等。这些数据通常是全局性的，不与特定交易对绑定。

2. **核心脚本与产出**:
   - **`src/build_full_merged.py`**:
     - **作用**: 这个脚本负责处理最基础的技术面数据，计算各种技术分析指标（TA Features），如 RSI, MACD, Bollinger Bands 等。
     - **产出**: 生成一个包含所有交易对、所有时间周期的基础技术指标宽表，通常保存为 `data/merged/full_merged.csv` 或 `.pkl` 文件。
   - **`src/fuse_fundamentals.py`**:
     - **作用**: 这是数据准备阶段的**关键一步**。它读取上一步生成的 `full_merged.pkl`，然后将**所有**链上数据、宏观数据等全局特征，通过时间戳对齐的方式，**融合**进去。
     - **输入**: `data/merged/full_merged.pkl` + 多个链上/基本面数据源。
     - **产出**: `data/merged/full_merged_with_fundamentals.pkl`。这个文件是整个项目后续步骤的**唯一数据来源**，它包含了特征筛选和模型训练可能用到的**所有**特征列。

### 第二阶段：特征筛选 (Feature Selection)

**目标**: 针对每一个“专家”模型，从 `full_merged_with_fundamentals.pkl` 这个巨大的特征池中，筛选出最优的特征子集。

**核心细节**:

1. **输入数据源**:
   - **`data/merged/full_merged_with_fundamentals.pkl`**: 特征筛选流程**唯一**的数据输入。所有的筛选操作都是在这个大宽表上进行的。

2. **核心脚本与逻辑**:
   - **入口**: `pipelines/run_feature_screening.py`。
   - **配置文件**: `pipelines/configs/feature_selection.yaml`，在这里定义了要为哪些专家、哪些数据集（`base`, `rich`, `comprehensive`）进行筛选。
   - **核心流水线**: `features/selection/run_pipeline.py`，它会按顺序执行多个筛选步骤（如过滤法、嵌入法等）。

3. **产出**:
   - `reports/feature_evidence/` 目录下会生成详细的中间结果和最终产出。
   - **最关键的产出**: 每个专家、每个数据集类型对应的 `selected_features.txt` 文件（例如：`reports/feature_evidence/Alpha-Dir-TFT/rich/selected_features.txt`）。这个文件**每行包含一个特征名称**，是下一阶段的直接输入。

### 第三阶段：配置同步 (Configuration Synchronization)

**目标**: 将特征筛选的成果（即 `selected_features.txt` 文件）应用到模型训练的配置中，确保数据侧与模型侧完全对齐。

**核心细节**:

1. **数据配置更新**:
   - **需要修改的文件**: `configs/experts/{专家}/datasets/{数据集}.yaml` (例如 `base.yaml`)。
   - **操作**: 在这些文件中，添加或更新 `feature_list_path` 参数，使其**精确地指向**第二阶段生成的对应的 `selected_features.txt` 文件的路径。

2. **训练配置补全**:
   - **需要创建的目录**: 对于每个专家现有的**每个时间周期**（如 `1h`, `4h`），如果缺少 `comprehensive` 训练配置目录，就需要创建它。
   - **操作**: 以同周期下的 `rich` 目录为模板，将内部的 `model_config.yaml`, `targets.yaml` 等文件**完整地复制**到新建的 `comprehensive` 目录中。

### 第四阶段：模型训练 (Model Training)

**目标**: 使用同步好的配置，启动并完成最终的 TFT 模型训练。

**核心细节**:

1.  **入口脚本**: `train_multi_tft.py`。
2.  **工作流程**:
    *   当你运行这个脚本并提供一个配置路径（如 `configs/experts/Alpha-Dir-TFT/1h/comprehensive`）时，它会：
        1.  读取该路径下的 `model_config.yaml` 和 `targets.yaml`，确定模型超参数和预测目标。
        2.  根据 `modality_set` (例如 `comprehensive`)，找到并读取对应的**数据配置文件** `configs/experts/Alpha-Dir-TFT/datasets/comprehensive.yaml`。
        3.  从这个数据配置文件中，读取**`feature_list_path`** 参数，从而获知应该使用哪个特征列表。
        4.  脚本接着读取这个 `selected_features.txt` 文件，得到一个确切的特征名称列表。
        5.  **数据加载器 (`data/load_dataset.py`)** 会加载**全局数据源** `data/merged/full_merged_with_fundamentals.pkl`。
3.  **动态周期筛选**:
    *   数据加载器会读取 `model_config.yaml` 中的 `period: "1h"` 参数。
    *   然后，它会从全局数据中**只筛选出 `period` 列等于 "1h" 的那些行**。
    *   最后，它会从这些筛选过的数据行中，再根据 `selected_features.txt` **只抽取出需要的特征列**。
    *   经过这两层筛选的、纯净的数据才会被送入模型进行训练，确保了模型训练的**周期特异性**和**特征准确性**。

---

## 特征筛选使用教程（Feature Selection）

本项目提供端到端的特征筛选管线，包含：

- Step 1 过滤（coverage/variance/corr/VIF + IC/MI 证据）
- Step 2 内嵌式（树模型重要性 + Boruta-like + 线性路径 + 可选 SHAP）
- Step 3 时间感知置换（cyclic shift / block permutation）
- Step 4 包装式搜索（RFE / GA，多目标加权可选）
- 汇总/清单导出（核心/增强双清单、出现率与排名统计、文档摘要）

### 目标集合（并集）与专家选择

- 目标集合采用“被选择专家的 targets 的并集（去重）”。
- 默认运行配置中所有专家；也可只运行指定专家：

  - CLI：
    ```bash
    python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --experts alpha_dir_tft,derivatives_micro
    ```

  - YAML：
    ```yaml
    # pipelines/configs/feature_selection.yaml
    experts_run: [alpha_dir_tft, derivatives_micro]
    ```

- 若未显式提供 `filter.params.ic_targets/mi_targets`，系统会基于并集自动拆分：回归类→IC，分类类→MI。

### 快速开始

1) 准备数据

- 确认 `experts` 各条目的 `pkl_path` 指向对应融合后的长表（示例见 `pipelines/configs/feature_selection.yaml`）。
- 若需要快速试跑，可用 `pipelines/configs/feature_selection_quick.yaml`（仅 1h，禁用时间置换）。

2) 运行完整筛选

```bash
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml
```

常用可选参数（在 YAML 中配置为主）：

- 外层验证切分：`splits.outer = {mode: days|ratio, days: 60, ratio: 0.2}`
- 时间置换：`permutation.enabled: true|false`、`block_len_by_period`、`group_cols`、`repeats`、`embargo/purge`
- 聚合：`aggregation.topk_core`、`topk_per_pair`、`min_appear_rate`、`weights_yaml`
- 包装式搜索：`wrapper.method: rfe|ga` 及其参数（`sample_cap/pop/gen/cx/mut`）

3) 运行快速筛选

```bash
python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection_quick.yaml
```

4) 仅跑单个专家/通道（带 pinned 注入示例）

```bash
# base 通道（默认字段按 experts.md 映射注入为 pinned，训练清单=核心 ∪ pinned）
python -m features.selection.run_pipeline --with-filter --expert-name "Alpha-Dir-TFT" --channel base --pkl data/pkl_merged/full_merged.pkl

# rich 通道
python -m features.selection.run_pipeline --with-filter --expert-name "Alpha-Dir-TFT" --channel rich --pkl data/pkl_merged/full_merged.pkl
```

5) 导出 TFT 变量选择（VSN）并参与 Embedded 阶段

```bash
python -m features.selection.tft_gating --ckpt <你的ckpt路径> --out reports/feature_evidence/tft_gating.csv
```

### 结果产出

<br>

#### 产出目录结构解读 (`reports/feature_evidence/`)

`reports/feature_evidence` 目录为每位专家保存了从特征输入到最终筛选结果的完整证据链。理解其结构有助于评估特征的有效性和筛选过程的合理性。以下表格对一个典型专家目录（如 `Alpha-Dir-TFT/`）下的文件和目录进行了解析。

| 分类 | 文件 / 目录名 | 意义和作用 |
| :--- | :--- | :--- |
| **输入配置** | `allowlist_*.txt`, `core_allowlist.txt` | **特征白名单**：定义了哪些特征可以进入筛选流程。`core`为通用核心特征，带专家名的为该专家专属的额外特征。 |
| **过程证据** | `<channel>/` | **并行筛选策略**：分别代表从"基础特征集"和"丰富特征集"出发的两套独立的、完整的筛选流程。 |
| | └ `<period>/stage1_filter/` | **阶段一：过滤法**。存放基于统计指标（如IC/MI、缺失率、方差）的快速初筛结果和证据。 |
| | └ `<period>/stage2_embedded/` | **阶段二：嵌入法**。存放基于多种模型（如树模型、线性模型）评估特征重要性的详细证据。 |
| | └ `<period>/tree_perm/` | **专项验证：排列重要性**。通过更耗时但更可靠的排列重要性方法，验证特征在该周期的稳健性。 |
| **决策汇总** | `<period>/aggregated_core.csv` | **聚合评估表**：**最关键的决策依据文件**。它汇总了该周期的过程证据，为每个特征计算最终综合得分、稳健性和排名。 |
| **最终产出** | `<period>/selected_features.txt` | **最终选定特征集**：**最重要的产出**，基于该周期 `aggregated_core.csv` 的评估生成的特征列表，直接用于该周期的模型训练。 |
| | `<period>/optimized_features.txt` | **优化特征集**：该周期的优化特征子集，追求更高的模型效率和鲁棒性。 |
| | `<period>/plus_features.txt` | **增强特征集**：该周期的增强特征集合，用于实验性或更复杂的模型。 |

<br>


- 每位专家/通道/周期的中间产物：
  - `reports/feature_evidence/<Expert>/<base|rich>/<period>/stage1_filter/`：
    - `allowlist.txt`：过滤后保留下来的特征
    - `filter_stats.csv`：coverage / variance 等统计
    - `ic_mi.csv`：IC/MI 证据
  - `reports/feature_evidence/<Expert>/<base|rich>/<period>/stage2_embedded/`：
    - `raw_scores.csv` / `summary.csv`：内嵌式得分与排名
    - `allowlist_embedded.txt`：按综合得分排序的候选清单
  - `reports/feature_evidence/<Expert>/<base|rich>/<period>/tree_perm/`：
    - `<target>_importances.csv`：树重要性 + 置换 Δ 及排名
    - `summary.csv`：该周期各目标聚合后的明细

- 聚合与最终清单：
  - `reports/feature_evidence/<Expert>/<channel>/<period>/aggregated_core.csv`：各周期的核心集合与统计
  - `reports/feature_evidence/<Expert>/<channel>/<period>/selected_features.txt`：各周期的训练用特征清单
  - `reports/feature_evidence/<Expert>/<channel>/<period>/plus_features.txt` / `optimized_features.txt`：各周期的包装式搜索增强或优化集合
  - `<Expert>/summary.json`：本次筛选摘要（通道统计、Wrapper 结果、可选的后验验证）

### 新增专家接入（让特征筛选可跑）

1) 在 `configs/experts/<Expert>/datasets/` 新增数据集定义

- `base.yaml`：从最新 `data/merged/full_merged.csv` 构建（技术/价量等全表），`dataset_type: fundamentals_only`。目标列 `target_*` 会随全表一并保留。
- `rich.yaml`：同样基于最新 `full_merged.csv`，`dataset_type: combined`，并对链上/ETF等 rich 列按 `no_nan_policy(scope: onchain, method: intersect)` 求“最大行交集”。
- OnChain 专家特别规则：无论 base 还是 rich，筛选只使用链上/ETF特征（技术面不作为候选特征）；但该专家的 `target_*` 列始终保留用于监督。

2) 在 `pipelines/configs/fuse_fundamentals.yaml` 的 `experts_map:` 注册

- 增加 `<Expert>_base: path/to/configs/experts/<Expert>/datasets/base.yaml`
- 增加 `<Expert>_rich: path/to/configs/experts/<Expert>/datasets/rich.yaml`

3) 执行融合，生成专家视图

```bash
# 直接运行脚本（推荐）
python src/fuse_fundamentals.py --config pipelines/configs/fuse_fundamentals.yaml

# 或 Python -c 调用
python -c "from src.fuse_fundamentals import run_with_config; run_with_config('pipelines/configs/fuse_fundamentals.yaml')"
```

4) 在 `pipelines/configs/feature_selection.yaml` 的 `experts:` 注册该专家（双通道）

```yaml
custom_expert:
  name:  Custom-Expert
  pkl_base: data/merged/expert_group/Custom-Expert_base/full_merged_with_fundamentals.pkl
  pkl_rich: data/merged/expert_group/Custom-Expert_rich/full_merged_with_fundamentals.pkl
  periods: ["1h", "4h", "1d"]
  targets: ["target_xxx", "target_yyy"]
```

5) 运行筛选（单专家示例，自动注入 pinned，训练清单=核心 ∪ pinned）

```bash
# base
python -m features.selection.run_pipeline --with-filter --expert-name "Custom-Expert" --channel base --pkl data/pkl_merged/full_merged.pkl
# rich
python -m features.selection.run_pipeline --with-filter --expert-name "Custom-Expert" --channel rich --pkl data/pkl_merged/full_merged.pkl
```

6) 训练（优先使用叶子 `selected_features.txt`）

```bash
python train_multi_tft.py --config configs/experts/Custom-Expert/1h/base/model_config.yaml
```

说明：
- 默认保留字段（pinned）来自 `experts.md`：每位专家在 base/rich 小节写到的字段，都会作为该通道 pinned；rich 的 pinned = base 小节 + rich 小节字段的并集；OHLCV 一并视作 pinned。
- 筛选 Step1/Step2 不剔除 pinned；最终训练清单 = 聚合核心集 ∪ pinned；每位专家叶子目录会写入其专属 `selected_features.txt`，训练时优先读取该清单。

### 常见用法

- 仅跑指定专家并筛选：
  ```bash
  python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --experts Alpha-Dir-TFT
  ```

- 指定置换参数（也可在 YAML 内设置）：
  ```bash
  python -m features.selection.tree_perm --periods 1h,4h --time-perm --block-len 48 --group-cols symbol --perm-repeats 5 --targets target_binarytrend,target_logreturn
  ```

### 设计说明与边界

- Regime 与 Z-Combiner 属“整合者”，默认不参与此管线；如需对 Regime 做筛选，请将其作为独立专家提供输入与 `targets`。
- Embedded 阶段已支持 `targets_override`，确保与管线的并集保持一致。
- 过滤阶段若需要“强制并集”，可在 `filter.params.ic_targets/mi_targets` 置空以启用默认并集拆分。

### 新增高级特性（v0.2.3）

- VSN 重要度：若导出 `tft_gating.csv`，Embedded 阶段会自动读取并在 `summary.csv` 新增 `tft_vsn_importance`，纳入综合打分。
- IC/MI 滚动窗：IC/MI 仅在训练窗计算，并在 `ic_mi.csv` 写入 `window_start/window_end`。
- 时间置换不确定度：`tree_perm` 输出 `perm_std`、`perm_ci95_low/high`，并记录 `block_len/embargo/purge`。
- 跨 era 出现率过滤：在聚合中可设置 `aggregation.min_appear_rate_era`，需要 `era` 列（tree_perm 已尝试写入年份）。
- GA 多 seed：在 `wrapper.ga.seeds` 提供多 seed 时，自动多次运行并输出 `ga_gene_frequency.csv`。
 - RFE 近似器：RFE 使用 LightGBM/CB/XGB 作为近似器，降低耗时；仍使用统一评估函数确保一致。
 - Base/Rich 融合质量权重：`aggregation.rich_quality_weights` 可按 period 缩放 Rich 权重，缓解低覆盖“伪优”。
 - 前瞻对照：`finalize.forward_compare.enabled` 开启后，会输出 `forward_eval.csv`（subset vs full）。

## 训练稳定化与配置集中（近期更新）

- 仅 `val_loss` 作为监控键：训练期不再计算 AP/AUC/F1/RMSE 等补充指标，避免因无样本/单类触发早停；验证集 `drop_last=False` 保证小窗也有 batch。
- 按币种仿射头保留：不影响现有建模（`output_head: { type: per_symbol_affine }`）。
- 配置集中化：所有叶子 `model_config.yaml` 新增/统一以下字段（无需改代码即可调整）：
  - 训练器：`monitor: val_loss`、`min_epochs`、`devices`、`log_interval`、`log_val_interval`、`log_every_n_steps`、`precision`、`accumulate` 等。
  - TS 数据集：`max_encoder_length`、`max_prediction_length`、`add_relative_time_idx`、`add_target_scales`、`allow_missing_timesteps`、`static_categoricals`、`static_reals`。
  - 元信息：`schema_version`、`data_version`、`expert_version`、`train_window_id`。
- 特征与数据规模打印 + 落盘：
  - 启动训练时分段打印：
    - `[Features]` 默认（pinned）、筛选（selected，固定从 `datasets/<base|rich>.yaml` 的 `feature_list_path` 读取 `reports/feature_evidence/.../selected_features.txt`）、合并（combined）、最终使用（final_used）清单与数量；
    - `[Data]` 总行数、训练/验证行数，以及按 `symbol/period` 的分布；
    - `[Shape]` 训练/验证二维形状（行×列，列为最终 used 特征数）。
  - 同步写入 `lightning_logs/experts/<expert>/<period>/<modality>/tft/configs/features_used.yaml`，便于复盘。
- 选中特征读取规则：总是从 `configs/experts/<Expert>/datasets/<base|rich>.yaml` 的 `feature_list_path` 指向的 reports 路径读取；若未配置，才回退到叶子目录的 `selected_features.txt`。
- 学习率调度：仍为 `OneCycleLR`（每步，cos 退火），`max_lr=learning_rate`，其形状参数使用默认（`pct_start=0.1`、`div_factor=25`、`final_div_factor=1e4`）。

### 离线评估（预告）
- 训练期间不再记录 AP/AUC 等补充指标与概率校准；建议在训练完成后，使用离线脚本进行评估与温度缩放。
- 预留脚本入口（占位）：
  ```bash
  # 计划中的离线评估脚本（预告，脚本暂未提供，示例仅作参考）
  # python scripts/offline_eval.py \
  #   --predictions lightning_logs/experts/<expert>/<period>/<modality>/tft/<log_name>/version_*/predictions/*.parquet \
  #   --targets-pkl data/pkl_merged/full_merged.pkl \
  #   --metrics ap,roc,rmse,mae --calibration temperature
  ```
  - 功能：读取预测与目标构建对齐集，计算 AP/ROC-AUC/F1/RMSE/MAE；可选做温度缩放并输出 ECE/Brier 与可靠性曲线；生成 `eval_report_*.csv`。

## FAQ / 常见问题

1.  **特征筛选管线太慢怎么办？**
    -  `pipelines/configs/feature_selection_quick.yaml` 提供了一个轻量级版本，减少了树模型数量、GA/RFE 迭代次数。
    -  在 `feature_selection.yaml` 中，可以减少 `tree_perm.params.n_repeats`、`wrapper.ga.n_generations`、`wrapper.rfe.n_features_to_select`。
    -  确保 `device="cuda"` 以使用 GPU 加速。

2.  **如何新增自定义专家？**
    -  请参考 `experts.md` 中的“专家接入指南”与 `README.md` 的“新增专家接入”章节。

3.  **训练时出现 `lr_scheduler.step()` 在 `optimizer.step()` 之前的警告？**
    -  这是一个在 PyTorch Lightning 中使用 `OneCycleLR` 时已知的、无害的“误报”警告。Lightning 框架在后台保证了正确的调用顺序。
    -  为了保持日志整洁，这个警告已被主动屏蔽，不影响实际训练效果。

## 历史版本亮点
### 2025-09-28 — 专家体系全面升级与9+1架构完整实现（v0.2.8）
#### 专家架构全面升级
