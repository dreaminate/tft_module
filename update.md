# 更新日志

## 2025-10-01 — 专家体系全面完善与文档更新（v0.2.9）

### 专家体系全面实现
- **Risk-Prob-TFT完整实现**：为Risk-Prob-TFT补全了所有缺失的训练目录结构，包括4h和1d周期的base、rich、comprehensive模态配置，所有配置文件已自动生成
- **Risk-Reg-TFT完整实现**：为Risk-Reg-TFT创建了完整的训练目录结构，包括1h、4h、1d周期的所有模态配置，所有必要配置文件已自动生成

### 项目状态更新
- **9+1专家体系全面实现**：所有10个专业专家模型（Alpha-Dir-TFT、Alpha-Ret-TFT、Factor-Bridge-TFT、KeyLevel-Breakout-TFT、MicroStruct-Deriv-TFT、OnChain-ETF-TFT、Regime-Gate、RelativeStrength-Spread-TFT、Risk-Prob-TFT、Risk-Reg-TFT）均具备完整的训练配置和目录结构
- **专家覆盖范围**：涵盖方向预测、收益回归、风险评估、微观结构、链上数据、市场体制、关键位突破、相对强弱、因子桥接等全领域专业能力
- **周期支持**：各专家支持1h、4h、1d的不同周期组合，总计数十种训练配置可用

### 文档全面更新
- **专家体系状态修正**：更新为"9+1专家体系全面实现"，所有专家均可直接用于训练
- **训练示例完善**：更新了所有训练示例，涵盖所有完整实现的专家模型
- **目录结构准确描述**：修正了专家配置目录结构的描述，反映实际的项目状态
- **项目描述更新**：从"设计了9+1专家架构，目前已完整实现8个专家模型"更新为"实现了完整的9+1专家架构，所有专家模型均具备完整的训练配置"

### 技术实现细节
- **自动化配置生成**：开发了Python脚本来自动生成所有缺失的model_config.yaml和targets.yaml配置文件
- **模板化配置管理**：基于成熟专家的配置模板，确保新专家配置的一致性和正确性
- **完整性验证**：所有专家现在都具备完整的周期×模态×配置的三维结构

### 影响文件（关键）
- `README.md`：全面更新专家状态描述、训练示例和项目状态说明
- `configs/experts/Risk-Prob-TFT/`：补全所有缺失的训练配置目录和文件
- `configs/experts/Risk-Reg-TFT/`：创建完整的训练配置目录和文件
- `create_risk_prob_configs.py`、`create_risk_reg_configs.py`：新增的配置生成脚本

### 当前项目亮点
- ✅ **9+1专家体系全面实现**：所有专家模型完整可用
- ✅ **多周期多模态支持**：数十种训练配置可供选择
- ✅ **自动化配置管理**：脚本化配置生成，提高开发效率
- ✅ **完整文档体系**：README和update.md准确反映项目状态
- ✅ **即刻可用性**：所有专家均可直接进行特征筛选和模型训练

## 2025-09-28 — 专家体系全面升级与9+1架构完整实现（v0.2.8）

### 专家架构全面升级
- **9+1专家体系完整落地**：实现Alpha-Dir、Alpha-Ret、Risk-Prob、Risk-Reg、MicroStruct-Deriv、OnChain-ETF、Regime-Gate、KeyLevel-Breakout、RelativeStrength-Spread等9个专业专家 + Z-Combiner融合层
- **专家职责明确分工**：
  - Alpha-Dir：方向预测（二分类：上涨/下跌概率）
  - Alpha-Ret：收益回归（预期收益幅度预测）
  - Risk-Prob：风险概率（极端风险事件概率）
  - Risk-Reg：风险回归（回撤/波动幅度预测）
  - MicroStruct-Deriv：微观结构（资金费率、持仓变化等）
  - OnChain-ETF：链上/ETF资金流（链上数据与ETF动向）
  - Regime-Gate：市场体制识别（趋势/震荡/高波动/危机状态）
  - KeyLevel-Breakout：关键位突破（支撑/阻力突破概率）
  - RelativeStrength-Spread：相对强弱（跨币种相对表现）
  - Factor-Bridge：因子桥接（传统因子 ↔ 加密代理）
  - Z-Combiner：智能融合层（多策略加权组合）

### Base/Rich并行融合机制
- **长短历史并行**：Base模态（全时代覆盖）与Rich模态（近年高质量数据）并行训练
- **α门控自适应加权**：基于市场状态、数据质量、延迟等因素动态调整Base/Rich权重
- **自然降级机制**：Rich数据缺失时α自动归零，回退到Base模态
- **训练损失设计**：主损失 + Base/Rich辅助损失，鼓励各模态独立学好

### 防泄露数据处理体系
- **慢频数据广播**：1d/4h → 4h/1h数据严格shift(1)→ffill，确保无未来信息泄露
- **严格时序审计**：`utils/audit_no_leakage.py`检查重复、时序倒退、异常间隔
- **可用性标记**：为Rich关键字段维护`rich_available_t∈{0,1}`与`rich_quality_t∈[0,1]`
- **时代标签**：引入`era_id`用于稳健性统计和跨时代稳定性评估

### 特征选择高级特性
- **四阶段筛选管线**：过滤（统计指标）→内嵌（模型重要性）→时间置换（稳健性验证）→包装搜索（子集优化）
- **TFT-VSN权重集成**：将TFT变量选择网络重要度纳入综合打分
- **跨时代稳定性**：支持按时代分组评估特征出现率和稳定性
- **GA多seed优化**：遗传算法支持多种子并行，提升搜索稳定性
- **RFE近似器优化**：优先使用LightGBM减少计算成本

### 嵌套时序交叉验证
- **严格防泄露**：内层CV用于特征选择和超参数调优，外层CV仅用于最终评估
- **滚动前瞻验证**：支持扩窗和滚动窗口前瞻，确保评估无泄露
- **多重验证指标**：分币种×周期详细报告，支持策略层回测指标

### 统一预测契约
- **标准化输出**：`{score, uncertainty, meta}`统一格式，包含完整元信息
- **自动校准机制**：温度缩放、ECE、Brier分数、P10/P50/P90覆盖率
- **批量预测支持**：自动写入parquet格式，包含版本和时间戳信息

### Z层智能组合
- **多策略融合**：规则加权、Stacking元学习、动态门控
- **风险控制集成**：与Risk专家联动，实现仓位上限和止损阈值控制
- **风格约束**：与Factor-Bridge专家联动，限制Alpha风格偏离
- **自动基线对比**：与"等权"和"单最佳专家"策略自动对比

### 专家配置管理
- **自包含配置**：每个专家×周期×模态目录自包含完整配置
- **Pinned特性**：每位专家可定义默认保留字段，特征筛选时优先保护
- **就近查找机制**：训练脚本支持配置就近查找，提升配置管理灵活性

### 文档与工具升级
- **README全面更新**：新增专家架构总览、完整工作流、专家接入指南
- **CLI工具增强**：`experts_cli.py`支持list/train/resume/warm等完整操作
  - 说明：`resume` 从叶子 `model_config.yaml` 的 `resume_ckpt` 读取断点；`warm` 从 `warm_start_ckpt` 读取预训练（仅权重）。未配置会报错。
  - 训练脚本已支持从 `model_config.yaml` 读取 `devices/strategy/precision/accumulate`；多卡未显式配置策略时默认 `ddp_find_unused_parameters_true`。
- **一键构建工具**：`build_full_merged.py`串联指标→目标→合并流程
- **审计工具完善**：泄露检测、数据质量检查、版本一致性验证

### 兼容性与迁移
- **向后兼容**：保持与现有训练脚本和配置的兼容性
- **平滑迁移**：支持渐进式采用新特性的迁移路径
- **配置瘦身**：移除冗余配置，统一管理专家数据集配置

## 2025-09-22 — 新增专家接入流程与 OnChain/base/rich 统一规则（v0.2.7）

新增/变更：

- 新增专家接入说明与落地：
  - 在 `configs/experts/<Expert>/datasets/{base,rich}.yaml` 定义数据集；
  - 在 `pipelines/configs/fuse_fundamentals.yaml` 的 `experts_map` 注册 `<Expert>_{base|rich}`；
  - 在 `pipelines/configs/feature_selection.yaml` 的 `experts:` 注册该专家（支持 `pkl_base/pkl_rich`）。
- 输入统一：
  - base 与 rich 均基于最新 `data/merged/full_merged.csv` 构建；
  - rich 对链上/ETF等 rich 列做 `no_nan_policy(scope:onchain, method:intersect)` 最大行交集；
  - OnChain 专家筛选仅使用链上/ETF特征，但其 `target_*` 目标列始终保留。
- pinned 规则与训练清单：
  - `experts.md` 中每个专家的 base/rich 小节列出的字段均为默认保留（pinned）；
  - rich 的 pinned= base 小节 + rich 小节 的并集；
  - 训练用清单 = 聚合核心集 ∪ 当前专家通道的 pinned；并为每个叶子目录写入专属 `selected_features.txt`。

影响文件：

- `README.md`（新增“新增专家接入”章节、示例命令与规则说明）
- `pipelines/configs/feature_selection.yaml`（支持 `pkl_base/pkl_rich` 条目）
- `features/selection/run_pipeline.py`（rich pinned=base∪rich；OnChain 链上-only；训练清单合并）

兼容性：

- 不破坏既有专家；未配置 `pkl_rich` 的专家仅跑 base。
- OnChain 专家在筛选阶段不使用技术面作为特征，但目标列始终保留。

## 2025-09-22 — Pinned 默认字段 + 字段解析报表（v0.2.6）

新增/变更：

- 筛选模块支持 pinned 特征：
  - `features/selection/filter_stage.py`
    - 新增 `FilterParams.pinned_features`，可注入“置顶默认字段”。
    - Step 1/2（覆盖率/低方差）不剔除 pinned；VIF 阶段优先保留 pinned（在无可替代时才最小化删除）。
    - 最终确保 pinned 出现在 `allowlist.txt`（如该列存在于数据）。
- 管线自动解析专家默认字段与同义映射：
  - `features/selection/run_pipeline.py`
    - 内置各专家的默认字段集合（含通用 OHLCV 与专家推荐字段）。
    - 同义词映射：将文档中的通用名映射为数据集中的实际列名；若不存在则记为缺失。
    - 输出报表：
      - 解析结果：`reports/experts/fields_resolved.csv`
      - 缺失列表：`reports/experts/missing_fields.csv`
    - 在 `--with-filter` 时按 `--expert-name` 与 `--channel` 自动注入 pinned 列表。
- 文档更新：
  - `experts.md` 新增“Pinned 默认字段”说明与示例（不改变其它文档结构）。
  - 训练清单合并：`configs/selected_features.txt` 改为“核心集 ∪ 当前专家通道的 pinned 默认集”，确保训练必含默认字段（即使在聚合核心中落选）。

使用：

- 生成解析与报表并执行筛选（示例，Alpha-Dir Base）：
  ```bash
  python -m features.selection.run_pipeline --with-filter --expert-name "Alpha-Dir-TFT" --channel base --pkl data/pkl_merged/full_merged.pkl
  ```
- 报表输出：
  - `reports/experts/fields_resolved.csv`
  - `reports/experts/missing_fields.csv`
- 筛选输出（含 allowlist）：`reports/feature_evidence/<expert>/<channel>/stage1_filter/`
- 训练侧衔接：保留使用 `selected_features.txt` 的就近查找与加载逻辑，兼容现有训练脚本。

影响文件（关键）：

- `features/selection/filter_stage.py`
- `features/selection/run_pipeline.py`
- `experts.md`

兼容性：

- 不改变训练调用与数据格式；默认仅新增候选与报表产物。
- 相关簇/多重共线（VIF）阶段可能在极端情况下移除非 pinned 列；pinned 尽量保留。

## 2025-09-22 — 技术面新增 β/HVN/LVN + 构建与日志优化（v0.2.5）

新增/变更：

- 技术面新增字段（在“指标阶段”直接生成，随后经目标→融合进入长表）：
  - `beta_btc_60d`：与 BTC 的滚动 β，防泄露 `shift(1)`；按周期使用等效 60 天窗口（1h≈1440bars，4h≈360bars，1d=60）。
  - `volume_profile_hvn` / `volume_profile_lvn`：等效 60 天窗口内按收盘价分箱、以成交量加权的 Volume Profile 节点（高/低成交量价位），输出为箱中心，统一 `shift(1)`。
  - 两类字段均已纳入泄露防护与分组滑动归一化流水线（会派生 `_zn{win}`/`_mm{win}`）。

- 一键构建脚本改进：
  - `src/build_full_merged.py` 的 `--periods` 现同时支持空格或逗号分隔（PowerShell/Unix 兼容），示例：
    - `python src/build_full_merged.py --periods 1h 4h 1d`
    - `python src/build_full_merged.py --periods "1h,4h,1d"`

- 终端日志优化：
  - 在 `src/indicating.py`、`src/groupwise_rolling_norm.py` 屏蔽 pandas 的 `PerformanceWarning`，避免碎片化 DataFrame 的性能告警刷屏。

- 列名快照：
  - 新增 `data/merged/full_merged.columns.txt`（长表首行列名快照），便于后续直接遍历字段而无需读取大 CSV。

验证：

- 已核验 `data/crypto_indicated/<1h|4h|1d>` 与 `data/crypto_targeted_and_indicated/<1h|4h|1d>` 的小 CSV 均包含 `beta_btc_60d`/`volume_profile_hvn`/`volume_profile_lvn` 及其归一化派生列。
- `data/merged/full_merged.csv` 也已包含上述字段（可直接在 `full_merged.columns.txt` 中检索）。

影响文件（关键）：

- `src/indicating.py`（新增 β 与 HVN/LVN 计算、纳入 shift 与归一化、告警屏蔽）
- `src/groupwise_rolling_norm.py`（告警屏蔽）
- `src/build_full_merged.py`（`--periods` 参数解析增强）
- `data/merged/full_merged.columns.txt`（列名快照，运行后生成）

兼容性：

- 下游训练与筛选可直接使用新增列；均已按防泄露口径处理。
- 原文档中“β/HVN/LVN 留待融合阶段”的说明已更新为在技术面阶段生成。

## 2025-09-22 — 技术指标全面扩展 + 一键构建 full_merged（v0.2.4）

新增/变更：

- 技术面扩展（全部进入 `data/crypto_indicated/<tf>` 并最终汇入 `data/merged/full_merged.csv`）：
  - Regime/标签：`trend_flag`、`range_flag`、`high_vol_flag`、`stress_flag`（均经 shift(1) 防泄露）
  - 通道/压缩：`donchian_high_20`、`donchian_low_20`、`keltner_upper_20`、`keltner_lower_20`、`squeeze_on`
  - 高阶动量/振荡：`ppo`、`stoch_rsi`、`williams_r`、`tsi`
  - 一目均衡：`ichimoku_conv`、`ichimoku_base`、`ichimoku_span_a`、`ichimoku_span_b`、`cloud_thickness`
  - 转折/形态：`psar`、`psar_flip_flag`、`supertrend_10_3`、`heikin_ashi_trend`
  - Tail/Risk：`ret_skew_30d`、`ret_kurt_30d`、`var_95_30d`、`cvar_95_30d`、`z_score_gap`
  - 蜡烛计数：`engulfing_up_cnt_20`、`doji_ratio_20`、`hammer_cnt_20`
  - 量价失衡：`mfi`、`cmf`、`price_volume_corr_20`
  - 横截面/枢轴：`price_position_ma200`、`pivot_point`、`distance_to_pivot`
  - 时间特征（known_future）：`hour_sin/hour_cos/dow_sin/dow_cos`
  - 统一派生：对关键列追加 `*_diff1`、`*_slope_24h`；并纳入分组滑动归一化（输出 `_zn48/_mm48`，裁剪至 [-5,5]/[0,1]）

- 归一化与防泄露：
  - 新增列统一先 `shift(1)` 再参与归一化与训练；继续保留 warm-up 裁剪与 NaN 行清理。

- 过滤（Filter）阶段增强：
  - 覆盖率阈值分层：`coverage_threshold_base=0.6`、`coverage_threshold_rich=0.3`；`rich_prefixes` 可配。
  - 多尺度/多滞后按组去冗：`group_patterns`（如 `_roll_\d+$`、`_ewma_\d+$`、`_\d+[smhdw]$`），每组保留前 `keep_n_per_group`。
  - 自动 IC/MI 目标：未显式配置时，根据目标类型自动拆分（回归→IC，分类→MI）。

- 一键构建脚本：
  - 新增 `src/build_full_merged.py`：串联 指标→目标→合并。
  - 使用：`python src/build_full_merged.py --periods 1h,4h,1d`

影响文件（关键）：

- `src/indicators.py`（新增高阶指标实现）
- `src/indicating.py`（落地计算、shift、防泄露、归一化与派生）
- `features/selection/filter_stage.py`（分层覆盖率、分组去冗、自动 IC/MI）
- `src/build_full_merged.py`（新增）

兼容性：

- 新增列对下游非依赖这些字段的训练脚本无破坏；已按防泄露口径处理。
- `beta_btc_60d`、`volume_profile_hvn/lvn` 保留为后续在融合阶段实现并并入。

## 2025-09-21 — 特征筛选：专家 targets 并集与专家选择接口（v0.2.3）

新增/变更：

- 特征筛选目标并集（union_targets）：
  - 现在在特征筛选管线中，实际使用的目标集合是“被选择专家的 targets 的并集（去重）”。
  - 若未选择专家（默认跑全部），则取配置中“所有专家”的并集；若选择了子集（见下），则仅基于该子集计算并集。
- 专家选择接口：
  - CLI 新增 `--experts a,b,c` 参数；或在 YAML 中新增 `experts_run: [a, b, c]`。
  - 若提供上述任一方式，则仅执行所选专家；并集与最终汇总也仅在该子集内进行。
- Embedded 阶段支持 `targets_override`：
  - `features/selection/embedded_stage.py` 新增 `targets_override` 并传递至 `load_split(...)`，从而与并集对齐。
- 管线统一应用并集：
  - `pipelines/run_feature_screening.py` 计算 `union_targets` 并传入 tree+perm、embedded、wrapper 搜索与后验验证，保证各阶段的目标一致。
- 过滤阶段（IC/MI）的默认行为：
  - 若配置中未显式指定 `filter.params.ic_targets`/`mi_targets`，则自动使用并集的回归类目标作为 IC、分类类目标作为 MI；
  - 如需强制使用并集，可在配置中清空这两个字段以启用默认逻辑，或后续提供“无条件并集”开关。

影响文件（关键）：

- pipelines/run_feature_screening.py（新增专家选择、并集计算与传递）
- features/selection/embedded_stage.py（新增 `targets_override` 参数与传递）
- features/selection/filter_stage.py（默认 IC/MI 目标回退至并集拆分）

兼容性：

- 默认行为不破坏既有配置；未指定 `--experts`/`experts_run` 时仍对所有专家执行。
- 若历史脚本依赖旧的 embedded 阶段行为（不覆盖 targets），现在将以并集为准，结果更一致。

## 2025-09-21 — 特征筛选高级特性（VSN/era/GA seeds/质量权重）（v0.2.3+）

新增/变更：

- Embedded 纳入 TFT VSN 权重：`tft_vsn_importance` 列参与综合打分；配置项 `embedded.params.use_vsn/vsn_csv`。
- IC/MI 限于训练窗：`ic_mi.csv` 增补 `window_start/window_end`。
- 时间置换输出增强：新增 `perm_ci95_low/high` 与 `block_len/embargo/purge` 元数据，附带 `era`（若可得）。
- 跨 era 出现率阈值：`aggregation.min_appear_rate_era` 支持按 era 稳定性过滤。
- GA 多 seed：`wrapper.ga.seeds` 支持多次运行并输出 `ga_gene_frequency.csv`。
- RFE 近似器：优先 LightGBM，其次 CatBoost/XGBoost，减少计算成本。
- Base/Rich 融合引入质量权重：`aggregation.rich_quality_weights` 可按 period 对 Rich 权重缩放。
- 前瞻对照：`finalize.forward_compare.enabled` 开启后输出 `forward_eval.csv` 与 `summary.json` 摘要。

影响文件（关键）：

- features/selection/embedded_stage.py
- features/selection/filter_stage.py
- features/selection/tree_perm.py
- features/selection/aggregate_core.py
- features/selection/combine_channels.py
- features/selection/optimize_subset.py
- pipelines/run_feature_screening.py
- pipelines/configs/feature_selection.yaml（新增配置项）

文档：

- README.md 已新增“新增高级特性（v0.2.3）”与配置示例。

## 2025-09-20 — 专家数据管线与文档梳理（v0.2.1）
## 2025-09-21 — 训练与特征筛选体验改进（v0.2.2）

- ⚙️ XGBoost 2.x 兼容：所有特征筛选/树模型处（`embedded_stage.py`、`tree_perm.py`、`rolling_validate.py`、`optimize_subset.py`）统一改为 `tree_method="hist" + device="cuda"`，并在旧版自动回退到 `gpu_hist/gpu_predictor`，消除 `gpu_hist` 弃用告警。
- 🧭 Lightning 进度条与日志：`train_multi_tft.py`、`train_resume.py`、`warm_start_train.py` 默认开启进度条（`enable_progress_bar=True`），同时将 `log_every_n_steps` 配置化（优先读取 `model_config.yaml` 的 `log_every_n_steps` 或 `log_interval`，默认 100），避免控制台频繁刷新。
- 🧪 线性路径稳健性：`embedded_stage.py` 的线性模型：
  - 默认 `linear_max_iter` 提升为 2000；
  - 在 `LogisticRegressionCV` 与 `ElasticNetCV` 拟合时静默 `ConvergenceWarning`，降低噪声；
  - `pipelines/configs/feature_selection.yaml` 同步将 `embedded.params.linear_max_iter` 提升到 2000。
- 其它：保留 SHAP 可选依赖的兼容路径；未安装时不影响主流程。

影响范围（关键文件）：

- `features/selection/embedded_stage.py`
- `features/selection/tree_perm.py`
- `features/selection/rolling_validate.py`
- `features/selection/optimize_subset.py`
- `train_multi_tft.py` / `train_resume.py` / `warm_start_train.py`
- `pipelines/configs/feature_selection.yaml`


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

## 2025-09-22 — 优化建议与未来改进（v0.2.8）

新增/变更：

- 建议添加自动化监控脚本，例如融合后数据质量检查（检查缺失率、异常值分布等），以应对未来数据源变化。
- 建议进行端到端测试（融合 → 筛选 → 训练），以验证整体性能和稳定性。
- 如果数据规模增大，考虑迁移到云资源以优化计算效率。

影响文件：

- 无直接代码变更；这些为未来开发建议，可作为 Roadmap 参考。

兼容性：

- 不影响现有功能；可逐步实现。


## 2025-09-28 — 训练稳定化与配置集中（val_loss-only + 特征/数据摘要）

### 变更概述
- 训练稳定化：仅使用 `val_loss` 作为早停/保存监控键；训练期间不再计算 AP/AUC/F1/RMSE 等补充指标，避免小验证窗/单类导致早停或刷屏告警。
- 校准可控：新增 `enable_calibration` 超参，训练阶段默认关闭校准收集与报表输出；建议训练后离线评估。
- 验证集稳健：验证 DataLoader `drop_last=False`，避免无 batch 情况。
- 配置集中化：所有叶子 `model_config.yaml` 统一/新增训练与数据集参数（`monitor: val_loss`、`min_epochs`、`devices`、日志频率、TS 长度与标志、元信息等）；移除历史 `loss_schedule`。
- 选中特征读取：总是从 `configs/experts/<Expert>/datasets/<base|rich>.yaml` 的 `feature_list_path` 指向的 reports 路径读取；未配置时才回退到叶子 `selected_features.txt`。
- 特征/数据摘要：训练启动时打印并保存 pinned/selected/combined/final_used 的清单与数量、数据总量与按 `symbol/period` 的分布，以及二维形状（行×列，列=最终 used 特征数）。摘要落盘至 `lightning_logs/.../tft/configs/features_used.yaml`。
- 学习率调度：仍为 `OneCycleLR`（每步，cos 退火），`max_lr=learning_rate`，`pct_start=0.1`、`div_factor=25`、`final_div_factor=1e4`。

### 影响文件（关键）
- `train_multi_tft.py`：仅监控 `val_loss`；`metrics_list=[]`；读取 `datasets` 的 `feature_list_path`；打印/落盘特征与数据摘要；支持从配置覆盖 TS 关键参数与 `enable_calibration`。
- `data/load_dataset.py`：返回 `features_meta`（pinned/selected/combined/used、dataset_summary 等）；支持 TS 参数覆盖；验证集 `drop_last=False`（此前已改）。
- `models/tft_module.py`：新增 `enable_calibration` 超参；按开关收集/输出校准与报表。
- `configs/experts/**/model_config.yaml`：批量追加 `monitor: val_loss`、`min_epochs`、`devices`、日志频率、TS 参数、元信息、`enable_calibration`；移除 `loss_schedule`。
- `README.md`：新增“训练稳定化与配置集中”与“离线评估（预告）”说明。

### 兼容性
- 按币种仿射头（per_symbol_affine）保留，不受影响。
- 补充指标/校准迁移为“离线评估”，训练稳定性更好；需要时可通过开关/脚本恢复。

### 离线评估（预告/引子）
预留脚本入口，后续提供实现：
```bash
# 脚本暂未提供（预告，示例仅作参考）
# python scripts/offline_eval.py \
#   --predictions lightning_logs/experts/<expert>/<period>/<modality>/tft/<log_name>/version_*/predictions/*.parquet \
#   --targets-pkl data/pkl_merged/full_merged.pkl \
#   --metrics ap,roc,rmse,mae --calibration temperature
```
功能：计算 AP/ROC-AUC/F1/RMSE/MAE，温度缩冷缩放校准并输出 ECE/Brier 与可靠性曲线，生成 `eval_report_*.csv`。

## 2025-09-29 — 训练指标体系增强与工作流优化

### 核心变更

- **启用并增强训练指标**:
  - `train_multi_tft.py` 中重新启用了指标计算（不再强制 `metrics_list=[]`），允许在训练期间监控丰富的性能指标。
  - `utils/metric_factory.py` 中为回归任务新增了 `R2Score` (R² 决定系数)，以评估模型的拟合优度。
  - `MyTFTModule` 内置的混淆矩阵可视化功能被激活，现在会在 TensorBoard 中为每个分类任务生成并记录混淆矩阵图像。

- **提升训练体验**:
  - **优化训练进度条显示**: 现在训练时，进度条会同时显示即时的批次损失 (`train_loss`) 和整个周期的平均损失 (`train_loss_epoch`)，便于同时观察瞬时动态和整体趋势。
  - **进度条显示可配置**: 在 `MyTFTModule` 和 `train_multi_tft.py` 中引入 `log_metrics_to_prog_bar` 参数（默认为 `False`）。现在可以通过在 `model_config.yaml` 中设置此项，来决定是否在进度条上显示除 `loss` 之外的所有详细指标，使得监控界面更整洁。
  - **显式配置添加**: 为了保持一致性，已将 `log_metrics_to_prog_bar: false` 显式添加到了所有 `configs/experts/**/model_config.yaml` 配置文件中。
  - **消除日志警告**: 修复了 PyTorch Lightning 关于 `batch_size` 推断不明确的警告，通过在 `validation_step` 的 `self.log()` 调用中显式传入 `batch_size`，增强了代码的严谨性。
  - **屏蔽 `lr_scheduler` 误报**: 在所有训练脚本 (`train_*.py`) 的开头添加了代码，以屏蔽 PyTorch 在与 Lightning 一同使用 `OneCycleLR` 时产生的关于 `lr_scheduler.step()` 调用顺序的误报警告。这不影响实际训练，仅为保持日志整洁。

- **训练脚本与配置优化**:
  - **指标周期限定**: `train_multi_tft.py` 现在会将当前训练的周期（如 `4h`）传递给指标工厂，确保只为当前任务生成对应周期的指标，避免了日志中出现无关周期的指标（如 `1h`, `1d`）。
  - **文档更新**: `README.md` 和 `update.md` 中的一些路径和命令示例得到了更新和修正，以反映最新的项目结构和实践。
  - **调试脚本**: 新增了 `_debug_ckpt_affine.py`, `_debug_symbol_check.py`, `_debug_symbol_stats.py` 等脚本，便于对模型权重和数据进行快速检查。

### 影响文件（关键）

- `train_multi_tft.py`: 重新启用指标计算，增加 `log_metrics_to_prog_bar` 配置传递，并限定指标的生成周期。
- `models/tft_module.py`: 新增 `log_metrics_to_prog_bar` 参数来控制进度条显示，并为 `val_loss` 日志添加了 `batch_size`。
- `utils/metric_factory.py`: 为回归任务增加了 `R2Score` 指标。
- `configs/experts/**/*.yaml`: 批量添加了 `log_metrics_to_prog_bar: false`。
- `README.md` / `update.md`: 同步了最新的改动说明。

## 2025-09-29 — Alpha-Dir-TFT 数据管线修复与命令补充

- **指标归一化窗口纠正**：`src/indicating.py` 中 tail risk 部分不再覆盖主窗口长度，保证 1h/4h/1d 使用 96/56/30。
- **合并脚本修复**：`src/merged.py` 重构 `merged_main`，模块导入时也会生成 `full_merged.csv` 与相关报表。
- **README 更新**：新增 Alpha-Dir-TFT 专属流程（构建→特筛→训练）并补充 `experts_cli.py` 启动示例；归一化后缀说明同步更新为 `_zn96/_mm96` 等。

## 2025-09-29 — 分类损失权重修复与概率监控增强

- 修复 `utils/weighted_bce.WeightedBinaryCrossEntropy` 的自动权重：允许 `pos_weight < 1` 并对正负样本对称加权，同时在批内归一化权重，防止“全预测为正”失去惩罚。
- 新增损失权重监控：`latest_stats` 记录 pos/neg 权重、归一化因子等，训练与验证阶段在日志中显示。
- `models/tft_module.MyTFTModule` 验证阶段增加分类概率统计（均值、标准差、`pred_pos_rate@0.5`），便于快速发现预测极化或阈值接线问题。

