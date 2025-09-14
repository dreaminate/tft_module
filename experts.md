

---

# 专家体系总览（8 + 1 组合器）

> 统一原则：**每位专家 × 每周期（1h/4h/1d）= 1 份共享骨干 Multi-TFT**；下挂**多符号输出头/Adapter**（BTC/ETH/BNB/SOL/ADA…）。再按模态分 **Base（普适） / Rich（富模态）** 两条通道**并行产出**，由门控 $\alpha_t$ 做**软融合**；Rich 不可用时 $\alpha_t=0$ 自然**回退** Base。
> 若出现**长期负迁移/强独占模态**且 A+C 仍不够，再把极少数“问题币”升格为 **B 方案：单币小 TFT 微调**（兜底）。

## 1) Alpha-TFT（收益/趋势主力）

* **任务**

  * 分类：`target_binarytrend`（未来窗口内方向/趋势成立与否）
  * 回归：`target_logreturn`、`target_logsharpe_ratio`
* **Base 输入**（全时代）
  OHLCV、技术面（MA/EMA、RSI、MACD、KDJ、BOLL、OBV、ATR）、结构特征（trend\_persistence、breakout\_count、amplitude\_range、tail\_bias、return\_skewness）、价量背离、滚动/ParK/EWMA 波动（作为特征）。
* **Rich 增强**（仅可用时代）
  衍生品精细子模态（精选版；详见专家 3）、链上摘要（活跃/净流/稳定币供给/拥堵等；详见专家 4）、ETF 溢价/净申赎（BTC/ETH 可用）。
* **输出**
  `proba_up`/`score`、`uncertainty`（温度或近窗误差）、`recent_valid_perf`。
* **角色**：进攻型核心 alpha 提供者。

## 2) Risk-TFT（风险/回撤主力）

* **任务**

  * 分类：`target_drawdown_prob`、`target_sideway_detect`
  * 回归：`target_max_drawdown`、`target_vol_jump_prob`（可选）
* **Base**
  波动族（rolling/parkinson/ewma）、偏度/尾部、极端回撤统计、LOF 异常密度、价量背离。
* **Rich**
  （1d）EGARCH/APARCH 参数摘要、链上拥堵/资金压力、融资利率异常等。
* **输出**
  风险概率/风险预算系数（0–1）、建议杠杆上限/减仓幅度。
* **角色**：防守制动器——Z 层的仓位上限/止损阈值来源。

## 3) 衍生品&微结构专家（Funding/OI/吃单/多空）

* **任务**：短线方向/动量增强（1h/4h），回归对 `logreturn` 做校正项。
* **Base**
  基础衍生品合成：聚合 OI/资金费率、taker 买卖强弱、期现基差（如可得）、多空比、成交集中度（交易所/合约聚合）。
* **Rich**
  交易所分场所结构（重点 venue 单列）、清算簇强度/方向、借币利率极端偏移、盘口/深度（如可得）。
* **输出**
  短线方向概率、`microstructure_heat` 热度分、短期校正项。
* **角色**：短线放大器，给 Alpha-TFT 提供补强。

## 4) 链上资金流专家（On-chain/ETF/溢价）

* **任务**：`fundflow_strength`（回归先验）、提供 risk-on/off 与持续性先验。
* **Base**
  1d 慢频摘要（广播到 1h/4h）：活跃地址、交易所净流入/净流出、稳定币供给/净流、矿工余额/发行、哈希率（BTC）。
* **Rich**
  （BTC/ETH）ETF 净申赎/溢价、Coinbase 溢价、链上拥堵/手续费飙升、鲸鱼地址动向（如可得）。
* **输出**
  资金风向分、先验 on/off、`uncertainty`（数据延迟/缺失驱动）。
* **角色**：中低频风向锚，辅助 Alpha/Risk/Regime。

## 5) 市场状态/体制识别（Regime Gating）

* **任务**：体制分类（trend/range/high-vol/crash-like）或二分类 risk-on/off。
* **Base**
  来自 A/B/C/D 的**摘要**特征（不可泄露未来）：趋势强度、波动水平/变化、资金费率偏离、OI 结构、链上净流、成交密度、价差波动。
* **Rich**
  可用时代下引入 ETF/链上状态的高置信摘要。
* **输出**
  专家权重向量、Base/Rich 融合权重先验、状态标签。
* **角色**：权重路由器，驱动 Z 层加权与门控。

## 6) 结构/突破专家（Key Levels/Breakout/ATR 斜率）

* **任务**：`trend_persistence`、`breakout_count`（回归）；`pullback_prob`、`sideway_detect`（分类）。
* **Base**
  关键水位（成交量耦合的高低密集区）、ATR 斜率、驱动条形（驱动/反转 K 线形态定量化）、amplitude\_range、tail\_bias、return\_skewness。
* **Rich**
  清算簇/异动成交对突破置信度的加权（衍生品侧提供）。
* **输出**
  突破/回落概率、趋势持续步数期望。
* **角色**：入场/加减仓触发与验证器。

## 7) 相对强弱&价差专家（ETH/BTC & 跨币）

* **任务**：相对 `binarytrend` / `logreturn`（对价差/比价，不做绝对价）。
* **Base**
  比价动量/低波、跨币波动差、成交与换手差、相对结构指标。
* **Rich**
  资金费率差、OI 结构差、链上活跃差、交易所溢价差、ETF 申赎差（BTC/ETH）。
* **输出**
  跨币超额强弱分、建议对冲比率。
* **角色**：择币/换手与中性对冲引擎。

## 8) 因子桥接专家（传统因子 ↔ 加密代理）

* **任务**：输出**风格暴露**（Momentum/LowVol/Carry/Liquidity/RelativeStrength/On-chain…）+ 轻量 `style_score` 先验。
* **实现**
  先用**无训练 Score-Card**（标准化线性打分）→ 稳定后可上 **ElasticNet/PLS/LGBM(带单调/复杂度约束)** 小幅监督；对 Alpha-TFT 进行**正交化/残差化**以抑制漂移。
* **输出**
  `factor_exposures{}`、`style_score`、`uncertainty`。
* **角色**：风格锚/约束器（限制 Alpha 的风格漂移，Z 层惩罚偏离）。

## 9) Z. 决策人/组合器（Gating + 加权 + 头寸规模化）

* **输入**
  各专家 `score/proba`、`uncertainty`、`recent_valid_perf`、Regime 输出、因子暴露/风格先验、风险预算、交易成本/滑点估计、保证金/杠杆约束。
* **输出**
  方向（多/空/观望）、头寸大小（杠杆/保证金）、风格（保守/中性/激进）三个档位。
* **机制**
  规则/线性融合（v1）→ stacking/meta-learner（v2）；风险制动来自 Risk-TFT；风格约束来自因子桥接。

---

# Base / Rich 并行融合（而非二选一）

* **最终输出（对每币种/每目标）**

  $$
  \hat{y}_t=(1-\alpha_t)\,\hat{y}^{\text{base}}_t+\alpha_t\,\hat{y}^{\text{rich}}_t,\quad
  \alpha_t=m_t\cdot\sigma\!\left(g(z_t)\right)
  $$

  `m_t∈{0,1}` 是 Rich 可用性；`z_t` 包含 Regime、质量/延迟、known-future 可用性等；$\sigma$ 为 Sigmoid。
  `m_t=0` ⇒ 回退 Base；`m_t=1` ⇒ **同时用** Base+Rich，权重随状态/质量自适应。
* **训练损失**
  主损失对融合输出；辅损失鼓励 Base/Rich 各自学好（Rich 辅损失只在 `m_t=1` 计入）：

  $$
  \mathcal{L}=\ell(y_t,\hat{y}_t)+\lambda_b\ell(y_t,\hat{y}^{\text{base}}_t)+\lambda_r\,m_t\,\ell(y_t,\hat{y}^{\text{rich}}_t)
  $$
* **门控正则**
  可用性硬约束、时间平滑（$\sum|\Delta\alpha_t|$）、状态先验（资金驱动 Regime 允许更大 $\alpha$）。

---

# 数据形态与防泄露（强制规范）

* **长表**：`[ts, symbol, period, <features...>]`，多周期合一。
* **慢频广播**：1d/4h → 1h **先 shift(1) 再 ffill**，并记录 `is_missing_*` 与 `lag_k`。
* **通道归类**：`static / known_future / observed_past`，known-future（如日历/到期/合约属性）独立通道。
* **可用性/质量**：为 Rich 关键字段生成 `rich_available_t`（0/1）与 `rich_quality_t∈[0,1]`（由延迟/异常/缺失率折算）。
* **时代标签**：`era_id`（如 pre-2020、post-2020、post-2024ETF…）用于稳健性统计与门控。

---

# 特征选择（四类证据 + 时间友好 + 双层清单）

1. **候选池（Step 0）**：多模态广覆盖 + 可用时代标注。
2. **快速过滤（Step 1）**：可用性/低方差/强共线/同源冗余（多滞后/多尺度**按组**去冗）。
3. **内嵌式证据（Step 2）**：TFT-VSN（分组权重）、树模重要性+Boruta 找“所有有用”，用 SHAP 复核校准。
4. **时间感知置换（Step 3）**：块置换/循环移位，分 1h/4h/1d 评估 ΔMetric，跨折/跨时代聚合名次与**出现率**。
5. **包装式搜索（Step 4）**：LightGBM+RFE/顺序前进（算力足再上 GA），**仅在内层时序 CV**；把**特征数/推理延迟/训练成本**写为硬约束或惩罚。
6. **定稿双清单（Step 5）**：

   * **通用核心**：跨周期/跨时代稳定靠前 + 多证据一致；
   * **专用增强**：该专家@该周期@该模态/币显著增益的小集合（**限额**）。
7. **稳健性（Step 6）**：滚动/前瞻评估，严格隔离选择窗与评估窗；上线后监控漂移与重要性时变。
8. **证据卡片**（每特征/组）
   VSN 权重统计、Permutation Δ、SHAP、跨折/跨时代出现率、时代覆盖、进入清单的依据与变更历史。

---

# 训练与评估（嵌套时序 CV、单指标早停、分币日志）

* **切分**

  * **内层**：特征选择/门控调参/头部与 Adapter 微调；
  * **外层**：滚动/扩窗前瞻（最终裁决与报告）。
* **指标（按任务、分周期×分币记录）**

  * 分类：**PR-AUC**、**F1（阈值经校准）**；
  * 回归：**RMSE**、**MAE**；
  * 运行期可再计算策略度量（胜率、收益/回撤、卡玛比）供 Z 层调权。
* **早停**：以**单一主指标**早停（分类 PR-AUC、回归 RMSE）。
* **多符号训练**：批内混币，样本 `(x, s)` 只对 `Head[s]` 回传；GroupNormalizer(symbol) 做归一化；必要时为“问题币”加小型 Adapter。
* **两阶段**

  1. **Base 预训练**（全时代×全币）；
  2. **Rich 预热 + 联训**（在 `m_t=1` 子集 warm-start，随后与 Base 共训 + α 门控）。

> 注：**不建议**把多个指标糅成一个“合成分”作为早停/适应度；合成分只用于**报告**或 Z 层策略评估。

---

# 组合与风控（Z 层）

* **Base↔Rich 融合**：由门控 α 完成（并行融合，回退=α 归零）。
* **专家间加权**：Regime 输出 → 不同状态的权重模板；`uncertainty` 与 `recent_valid_perf` 做动态折扣。
* **风格约束**：因子桥接的 `factor_exposures`/`style_score` 作为**偏离惩罚**，限制 Alpha 的风格漂移。
* **风险制动**：Risk-TFT 输出转为**仓位上限/止损阈值**硬约束。
* **择币与对冲**：相对强弱专家驱动跨币权重与对冲比率。
* **成本/延迟**：纳入上线闸门（特征数、推理延迟、内存、训练时长）。

---

# 触发 B 方案（单币小 TFT 微调）的硬标准

满足任一即评估（先灰度）：

1. 某币在 **≥3 个外层前瞻窗**，主指标**持续落后**本周期组内中位数 **≥10%**；
2. 诊断显示**强负迁移**（混币训练显著变差、单币训练显著变好且可复现）；
3. 存在**强独占且高权重模态**（如 BTC-ETF 在 1d 极强），A+C 吃不满；
4. 资源允许且**已扣除成本后**仍显著提升。

做法：从全币预训练权重复制，**冻结 ≥80% 层**，仅末端/Adapter 小步微调；配 **L2-SP/EWC/蒸馏** 防遗忘；与 A 方案同币 Head 做 A/B，显著性达标才投产。

---

# 监控与审计（线上）

* **必记**：分周期×分币的 PR-AUC/F1、RMSE/MAE；`α_t` 使用率与分布；`uncertainty`；`recent_valid_perf`；关键特征分布漂移；门控与权重热力图。
* **告警**：

  * 指标跌破阈值（例如 7/30 天移动均值下穿历史分位）；
  * `α_t` 长期卡死（→ 门控失灵）；
  * 重要性/分布漂移超阈（→ 触发再训练或禁用专用增强）。
* **证据卡片**：每月固化（VSN/Permutation/SHAP/出现率/时代覆盖/α 使用率），为审计与复盘服务。

---

# 版本与命名（不可省）

* **Checkpoint**：`[Expert]_[Period]_[BASE|RICH]_[ALL|SYM]_{metric=..}_{valloss=..}_{ver}.ckpt`
* **数据视图**：`dataset_ver`（含特征字典 hash、时代边界、缺失策略）；
* **特征清单**：`features_core.yaml`、`features_boost_[expert]_[period].yaml`（专用增强限额）；
* **门控/路由**：`gating_ver`（含输入维度、先验/正则参数）；
* **一键回滚**：版本索引表（所有线上组件的 ver → ckpt 路径与依赖）。

---

# 工程目录（参考）

```
/experts/
  alpha/   risk/   derivatives/   onchain/   regime/   structure/   relative/   factor_bridge/
    config/
      base_[1h|4h|1d].yaml
      rich_[1h|4h|1d].yaml
    model/
      backbone.py         # 共享骨干
      heads.py            # 多符号头/Adapter
      gating.py           # α 门控
    train.py              # 嵌套CV与两阶段训练
    infer.py              # 在线推理与融合
/features/
  dictionaries/           # 全量与分组字典
  selection/              # Filter/Embedded/Permutation/Wrapper 管线
/pipelines/
  datamart/               # 广播/滞后/缺失指示/era 标签
  evaluate/               # 分周期×分币指标
  monitor/                # α/漂移/重要性监控
/z_layer/
  router.py               # 专家/通道权重
  allocator.py            # 头寸规模化
  constraints.py          # 风格与风险约束
```

---

# YAML 模板（示例，裁剪即可用）

```yaml
expert: AlphaTFT
period: 1h
symbols: [BTC, ETH, BNB, SOL, ADA]

channels:
  base:
    features_groups:
      price:    [close, high, low, open, vwap, return_1h, return_4h]
      tech:     [rsi_14, macd_12_26, ema_20, atr_14, obv, kdj_k, kdj_d]
      struct:   [trend_persistence, breakout_count, amplitude_range, tail_bias, return_skewness]
      vol:      [vol_roll_24, vol_ewma_24]
    head: multi_symbol
    adapter: {enabled: true, dims: 32}   # 为“问题币”留接口
  rich:
    features_groups:
      derivatives: [funding_rate_z, oi_chg_z, taker_imbalance_z, basis_z]
      onchain_1d:  [active_addr_z, exch_netflow_z, stablesupply_z]   # 已 shift 后广播
      etf_daily:   [etf_flow_z, etf_premium_z]
    available_flag: rich_available
    quality_flag:   rich_quality
    head: multi_symbol
    adapter: {enabled: true, dims: 32}

gating:
  inputs: [rich_available, rich_quality, regime_trend, regime_range, regime_highvol]
  type: mlp
  hidden: 16
  smooth_l1_alpha: 0.01   # 时间平滑正则
  hard_mask_on_unavailable: true

training:
  optimizer: adamw
  lr: 1e-3
  weight_decay: 0.01
  warmup_steps: 1000
  batch_size: 512
  loss:
    main: [bce, mse]      # 视目标而定
    aux_base: 0.2
    aux_rich: 0.2
  early_stopping:
    metric: pr_auc        # 分类；回归时用 rmse
    patience: 10

cv:
  outer: {type: rolling, windows: 6, step: 1}
  inner: {type: rolling, windows: 4}
  seed: 2025

selection:
  filter: {low_variance: true, corr_thresh: 0.9, group_redundancy: true}
  embedded: {vsn: true, boruta: true, shap_check: true}
  permutation: {block_len: 96, method: "circular_shift"}
  wrapper: {method: "rfe", estimator: "lgbm", max_features: 64}
  lists:
    core_thresholds: {appearance_rate: 0.6, min_perm_delta: 0.002}
    boost_limits: {per_expert_period: 12}

logging:
  save_alpha_series: true
  save_importance: true
  save_metrics_by_symbol: true
```

---

# 排期与分工（现实可执行）

* **Week 1**：Alpha / Risk / Regime 的 **Base** 分支跑通（含特征选择、嵌套 CV、单指标早停、分币日志）；Z 层规则版上线。
* **Week 2**：衍生品/结构接入；**Rich** 分支 warm-start + α 联训；并行融合上线；监控 α/漂移。
* **Week 3**：链上与因子桥接补位（风格约束 + 择币/对冲）；完成证据卡片自动化；评审是否需要对极少数币试点 **B 方案**。
* **持续**：每月滚动/前瞻评估、证据卡片固化、阈值回顾与成本/延迟体检。

**分工建议**

* FE（特征&数据）：广播/滞后/缺失指示/era 标签、字典维护与稳定性统计
* ML（建模&训练）：Base/Rich 两塔、门控、嵌套 CV、特征选择与调参
* STRAT（策略\&Z 层）：权重路由、约束、成本/滑点、回测与上线闸门
* OPS（工程&监控）：模型服务、指标与 α/漂移监控、A/B 实验、回滚

---

## 最后三条“红线”确保上线稳健

1. **所有选择/调参**必须在**内层时序 CV**完成，外层仅裁决（防“选特征过程”泄漏）。
2. **Rich 与慢频数据**必须 **shift→broadcast**，并带 `rich_available/quality`；**融合=软加权**，**回退只是 α=0 特例**。
3. **上线闸门**包含**性能（主指标显著优于基线）+ 成本/延迟 + 漂移监控**三件套；不满足任一不得放量。

---

**一句话总括**：

> 专家清单 8+1、职责清晰；**每专家×每周期**一份共享骨干 Multi-TFT + **多符号头/Adapter**；**Base 与 Rich 并行融合**（$\alpha$ 门控），缺失时自然回退；多证据+时间友好选特征，**嵌套时序 CV**裁决；Z 层完成权重路由、风格与风险约束；B 方案仅对极少数币兜底微调。以上规范**完备、可执行、可审计**。

