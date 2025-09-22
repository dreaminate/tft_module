

---

# 专家体系总览（9 + 1 组合器，A+C 并行）

> 当前实现总览（对齐现状）
>
> - 单一长表：所有模态已在统一的 `full_merged_with_fundamentals.{csv,pkl}` 中融合，含 1h/4h/1d 全部周期；慢频链上/宏观指标统一执行 `shift(1) → ffill` 下推，避免信息泄露（在 `src/fuse_fundamentals.py` 的左连接 + 按 symbol 排序后 ffill 管道中实现，线上强制审计 `utils/audit_no_leakage.py`）。
> - 专家配置：每位专家的 Base/Rich 两路数据融合规则在 `configs/experts/<Expert>/datasets/{base,rich}.yaml` 中维护；上层 `pipelines/configs/fuse_fundamentals.yaml` 负责调度与聚合（支持 experts_map/experts 列表与分组落盘、pkl 后处理）。
> - 训练过滤：训练由叶子目录的 `model_config.yaml` 指定 `period`（与可选 `symbols`）裁剪子集，保证“一张长表”即可跑多周期训练（`scripts/experts_cli.py` 支持 list/train/resume/warm）。
> - 叶子完备：每个叶子目录必须自带 `model_config.yaml` / `targets.yaml` / `weights_config.yaml`（Base 与 Rich 各自一套），训练脚本读取叶子文件为准。
> - Base/Rich 并行：并行而非二选一，Rich 不可用时自动回退 Base（权重 α 归零）；“历史长覆盖 × 近年富模态”两全其美。
> - 融合路径：从简单均值 → Stacking 元学习 → Regime 门控，迭代增强。

---

## 统一范式（A+C）

- 每位专家 → 周期分支（1h/4h/1d） → 模态通道（Base=普适骨干；Rich=富模态补强；Rich 失效自动回退） → 多符号 Adapter/头（head 只对所属 symbol 回传梯度）。
- Alpha 系列强制拆分：方向二分类（Alpha-Dir）与收益回归（Alpha-Ret）各自一位专家，避免任务负迁移。

---

## 多专家任务 + 推荐数据字段一览（Markdown 版）

| # | 专家 | 最终任务 | Base 字段（长期可用） | Rich 字段（近年/高置信） |
| - | - | - | - | - |
| 1 | Alpha-Dir（方向二分类） | binarytrend ↑/↓ | 价量技术面：OHLCV、MA/EMA、RSI、MACD、ADX、布林带宽、短-中期动量（5 / 20 / 60 bar return）等 | 衍生品简要摘要（Funding、OI 斜率）、链上活跃度、ETF 溢价（BTC/ETH）等 |
| 2 | Alpha-Ret（收益回归） | logreturn、logsharpe_ratio | 同 Alpha-Dir Base + 滚动/ParK/EWMA 波动、价量背离 | Funding 斜率、OI 结构、清算密度、链上资金流净量、ETF 净申赎 |
| 3a | Risk-Prob（极端风险分类） | drawdown_prob、vol_jump_prob | 历史波动率、ATR、偏度/峰度、价量异常密度 | Funding 异常峰值、OI 激增、隐含波动率曲面突变、深度骤减、爆仓笔数 |
| 3b | Risk-Reg（回撤/波动回归） | max_drawdown、realized_vol | 同 Risk-Prob Base | 同 Risk-Prob Rich |
| 4 | Derivatives & Microstructure | 短线方向 + 校正项 | 聚合 Funding、OI、基差、成交集中度、多空比 | 分 venue OI、期权 IV/BVIV、25Δ RR、吃单量、盘口深度差、ETF 净流入等 |
| 5 | On-chain / ETF 资金流 | fundflow_strength 回归 + risk-on/off 先验 | 日频活跃地址、交易所净流、稳定币供给、矿工余额、哈希率等 | ETF 申赎/溢价、Coinbase 溢价、Gas 费、鲸鱼地址动向 |
| 6 | Regime Gating | 多分类：trend / range / high-vol / stress | 收益分布指标、20 d HistVol、成交活跃度、市场广度、杠杆水平等 | ETF 溢价、链上拥堵、宏观走势 (S&P, VIX) |
| 7 | Structure / Breakout | 多标签：breakout_prob、pullback_prob、sideway_prob | 关键水位密集区、ATR 斜率、成交密度变化、驱动条形、return skew 等 | 清算簇/异常大单强度 |
| 8 | Relative Strength & Spread | 跨币 binarytrend / logreturn | 比价动量、波动差、成交/换手差 | Funding 差、OI 差、链上活跃差、ETF 申赎差 |
| 9 | 因子桥接（Macro ↔ Crypto） | factor_exposures{}、style_score | S&P500、VIX、DXY、10Y Yield、M2、黄金/原油价格等宏观 & 传统因子 | BTC-S&P 滚动相关、全球动量/风格指数、ETF 流向衍生因子 |

### 说明
- Base：历史覆盖 ≥ 2015，价量 & 宏观为主；Rich：2020+ 才完整的链上、衍生品、ETF 等高置信新模态。
- Risk 专家按 分类-回归 分家，避免损失冲突；训练期两路信号可由简易 MLP 融合为仓位上限函数。
- 各字段均来自文档中对专家输入的具体推荐，已按可获取频率与任务适配性归类。

### 训练数据要点（摘录）

| 关键环节 | 落地建议 |
| - | - |
| 标注防泄露 | 所有标签先 shift(1)；慢频因子 shift→ffill 再广播。 |
| 不平衡 | Risk-Prob 采用 事件级过采样 + focal-loss；其他分类用 class_weight。 |
| 特征窗 | 多尺度 (20 / 60 / 240 bar)；Risk 增强杠杆 & 深度信号。 |
| 验证切分 | 嵌套时序 CV；每折保留一次危机（3·12、5·19 等）。 |
| 样本权重 | era-aware；近年或高成交段权重放大。 |

---

## 1) Alpha-Dir-TFT（方向 / 二分类核心）

- 任务：分类 `target_binarytrend`（未来窗口内涨/跌/方向成立与否；阈值经标定）
- Base（全时代）：价量技术面：OHLCV、MA/EMA、RSI、MACD、ADX、布林带宽、短-中期动量（5 / 20 / 60 bar return）等；可辅以结构与波动统计（如 trend_persistence、breakout_count、return_skewness、ParK/EWMA）。
- Rich（可用时代）：衍生品简要摘要（Funding、OI 斜率）、链上活跃度、（BTC/ETH）ETF 溢价/申赎等。
- 输出：`proba_up/score`、`uncertainty`（温度或近窗误差）、`recent_valid_perf`（滚动验证窗表现）。
- 角色：进攻型方向提供者；与 Regime/微结构协同提升短中线胜率。
- A+C 并行：Base 贡献长历史稳健信号，Rich 注入近年“新模态”优势，组合优于单路。

## 2) Alpha-Ret-TFT（收益 / 回归核心）

- 任务：回归 `target_logreturn`、`target_logsharpe_ratio`（风险调整收益）
- Base：同 Alpha-Dir Base + 滚动/ParK/EWMA 波动、价量背离。
- Rich：Funding 斜率、OI 结构、清算密度、链上资金流净量、ETF 净申赎。
- 输出：`return_score` 与区间收益预估、`uncertainty`。
- 角色：收益校正器：与 Alpha-Dir 的方向概率互补，提供“大小与置信”信息，供 Z 层做仓位与杠杆规模化。

## 3a) Risk-Prob（极端风险分类）

- 任务：`drawdown_prob`、`vol_jump_prob`
- Base：历史波动率、ATR、偏度/峰度、价量异常密度。
- Rich：Funding 异常峰值、OI 激增、隐含波动率曲面突变、深度骤减、爆仓笔数。
- 输出/角色：极端风险概率与风险预算系数（0–1），用于 Z 层仓位上限/减仓硬约束。

## 3b) Risk-Reg（回撤/波动回归）

- 任务：`max_drawdown`、`realized_vol`
- Base：同 Risk-Prob Base。
- Rich：同 Risk-Prob Rich。
- 输出/角色：回撤/波动幅度预估，用于尺度化杠杆与止损阈值。

## 4) 衍生品&微结构（Funding/OI/吃单/多空）

- 任务：短线方向 + 校正项（1h/4h）；可对 `logreturn` 做短期校正项。
- Base：聚合 Funding、OI、基差、成交集中度、多空比。
- Rich：分 venue OI、期权 IV/BVIV、25Δ RR、吃单量、盘口深度差、清算簇强度/方向、ETF 净流入等（如可得）。
- 输出/角色：`microstructure_heat`（短线热度分）、短线方向概率/校正项；短线放大器，与 Alpha 系列联动。

## 5) 链上资金流 / ETF / 溢价（On-chain）

- 任务：`fundflow_strength` 回归 + `risk-on/off` 先验。
- Base（1d 慢频 → 广播）：活跃地址、交易所净流入/净流出、稳定币供给/净流、矿工余额/发行、哈希率（BTC）。
- Rich：（BTC/ETH）ETF 申赎/溢价、Coinbase 溢价、链上拥堵/手续费飙升、鲸鱼地址动向。
- 输出/角色：资金风向锚与不确定度（由延迟/缺失率折算），辅助 Alpha/Risk/Regime；与 Base 长历史并行融合。

## 6) 市场状态 / 体制识别（Regime Gating）—统一为分类任务（确定版）

- 最终任务（唯一训练目标）：多分类（primary）`regime_label ∈ {trend, range, high-vol, stress}`（四分类）。
- 派生量（推理期计算，不单独训练）：
  - `risk_on_off`（二分类先验）：由多分类概率映射得到（如 risk_on = P(trend) - P(stress)，经温度标定/阈值表二值化）。
  - `expert_weight_prior{}`：对 9 位专家的先验权重向量（可由 regime_probs 经规则/小 MLP 得到）。
  - `alpha_prior`：Base/Rich 融合的先验权重（0–1，供 α 门控初值参考）。
- 标签构造（无信息泄露）：所有用于打标签的统计量基于过去窗口计算，严格 `shift(1)` 后再决定当前时刻标签；慢频（1d/4h）广播到 1h 时同样先 `shift(1)→ffill`。
- 参考规则（可配置）：
  - trend：ADX、滚动 OLS 的 R²、HH/HL 计数 ≥ 上分位且方向一致；
  - range：振幅/波动处于低分位，方向性指标走平；
  - high-vol：波动分位（ATR/realized vol）≥ 上分位但方向性不显著；
  - stress：收益极端下分位 + 波动与卖压指标急剧放大（含资金费率负偏离/清算密集等）。
- 类别均衡：按 era 统计占比，必要时启用 class weighting / focal-like。
- 输入（摘要特征，严禁窥视未来）
  - Base（observed past）：收益分布指标、20 d HistVol、成交活跃度、市场广度、杠杆水平、趋势强度、波动水平/变化、成交/换手密度、价差波动、资金费率偏离、OI term-structure 摘要、链上净流等；
  - Rich：ETF 溢价/申赎、链上拥堵/手续费异常、分 venue 衍生品结构、宏观走势（S&P, VIX）等高置信慢频摘要；
  - 可用性/质量：为关键 Rich 字段维护 `rich_available_t∈{0,1}` 与 `rich_quality_t∈[0,1]`，训练期亦可作为输入。
- 输出契约（供 Z 层/门控直接消费）
  - `regime_probs{trend,range,high-vol,stress}`、`regime_label`；
  - `expert_weight_prior{Alpha-Dir, Alpha-Ret, Risk, Derivs, Onchain, Regime, Structure, Relative, Factor}`；
  - `alpha_prior`（Base/Rich 初始权重）；
  - `uncertainty`、`recent_valid_perf`（滚动窗 Macro-F1）。
- 训练与评估
  - Loss：多类交叉熵（可含 class weight）。
  - Metric：Macro-F1（主）、Balanced Accuracy；早停：Macro-F1 单指标。
  - 校准：温度标定 / Platt scaling（离线校准，推理期应用）。
  - 日志：分周期×分币统计混淆矩阵、每类 PR 曲线（可选）。

## 7) 结构 / 突破（Key Levels / Breakout / ATR Slope）—统一为分类任务（确定版）

- 任务形态（二选一）
  - A. 多标签二分类（推荐）：`breakout_prob`（关键位突破可信度）、`pullback_prob`（突破后回踩/回落概率）、`sideway_prob`（盘整概率）
  - B. 单头三分类（互斥更强时使用）：`{breakout, pullback, sideway}`
- 不再在本专家内做回归：`trend_persistence`、`breakout_count` 改为输入特征/派生统计或交由 Alpha-Ret/独立 7R 回归头训练（物理上分离，避免多任务互拖）。
- 标签构造（严格过去窗口）：关键位识别（HVN/LVN、分形枢轴、近端极值簇、(Anchored) VWAP 等）→ 突破：收盘穿越关键带 + ATR 扩张/成交放大 +（可选）资金费率/清算共振 → 回踩：突破后的回测触及关键带并持稳 → 盘整：波动率/区间振幅低分位，方向性走平；所有判定均用 `shift(1)` 后统计量，确保无未来泄露。
- 输入：
  - Base：关键位与成交耦合强度、ATR 斜率/扩张、驱动条形定量、`amplitude_range`、`tail_bias`、`return_skewness`、局部成交密度变化；
  - Rich（可选）：衍生品侧清算簇/异常成交强度，用于提升突破置信度。
- 输出契约：`breakout_prob/pullback_prob/sideway_prob`（或三分类 probs{}），附 `uncertainty` 与近窗校准信息；提供给 Z 层作入场/加减仓触发、验证 Alpha-Dir 的方向信号。
- 训练与评估：多标签 BCE/Weighted-BCE；三分类 CE；Metric：PR-AUC（主）+ F1（阈值标定后）；早停：PR-AUC（多标签取 macro/加权平均）；阈值标定：以最近 3–6 个月验证窗做 F1 曲线寻优，按周期分开记录。
- 如确需连续量（如“趋势持续步数期望”），请新建 7R 回归子模型或并入 Alpha-Ret，不与分类混训；并行时使用 多门控 MoE 为各任务独立路由专家。

## 8) 相对强弱 & 价差（ETH/BTC & 跨币）

- 任务：相对 `binarytrend` / `logreturn`（对价差/比价，不做绝对价）。
- Base：比价动量/低波、跨币波动差、成交与换手差、相对结构指标。
- Rich：资金费率差、OI 结构差、链上活跃差、交易所溢价差、ETF 申赎差（BTC/ETH）。
- 输出/角色：跨币超额强弱分、建议对冲比率；择币/换手与中性对冲引擎。

## 9) 因子桥接（传统因子 ↔ 加密代理）

- 任务：输出风格暴露（Momentum/LowVol/Carry/Liquidity/RelativeStrength/On-chain…）与轻量 `style_score` 先验；为 Z 层提供风格约束。
- 实现：先用无训练 Score-Card（标准化线性打分），稳定后引入 ElasticNet/PLS/LGBM（带单调/复杂度约束）；对 Alpha-Ret/Alpha-Dir 进行正交化/残差化抑制漂移与重叠暴露。该“AI × 因子”融合路径以因子框架可解释性为锚、机器学习筛选与加权为增益。
- 输出/角色：`factor_exposures{}`、`style_score`、`uncertainty`；风格锚/约束器（惩罚 Alpha 风格偏离）。

## Z. 决策人 / 组合器（Gating + 加权 + 头寸规模化）

- 输入：各专家 `score/proba`、`uncertainty`、`recent_valid_perf`、Regime 输出、因子暴露/风格先验、风险预算、交易成本/滑点估计、保证金/杠杆约束。
- 输出：方向（多/空/观望）、头寸大小（杠杆/保证金）、风格档位（保守/中性/激进）。
- 机制：v1 规则/线性融合；v2 Stacking 元学习（Logit/小 MLP）；v3 动态门控（Regime/质量/延迟驱动的 α）。风险制动由 Risk-TFT 提供，风格约束由因子桥接提供。

---

## Base / Rich 并行融合（不是二选一）

- 最终输出（对每币/每目标）：

  $$
  \hat{y}_t = (1-\alpha_t)\,\hat{y}^{\text{base}}_t + \alpha_t\,\hat{y}^{\text{rich}}_t,\quad \alpha_t = m_t\cdot\sigma\big(g(z_t)\big)
  $$

  其中 `m_t∈{0,1}`：Rich 可用性标志（缺失/过期=0）；`z_t` 包含 Regime、质量/延迟、known-future 可用性等；`σ` 为 Sigmoid。

  - `m_t=0` ⇒ 自然回退 Base；`m_t=1` ⇒ 并行使用 Base+Rich，权重随状态/质量自适应（历史广度×新模态深度两全）。

- 训练损失：

  $$
  \mathcal{L}=\ell(y_t,\hat{y}_t)+\lambda_b\,\ell(y_t,\hat{y}^{\text{base}}_t)+\lambda_r\,m_t\,\ell(y_t,\hat{y}^{\text{rich}}_t)
  $$

  主损失作用于融合输出；辅损失鼓励 Base/Rich 各自学好（Rich 仅在 `m_t=1` 计入）。

- 门控正则：可用性硬约束、时间平滑（`\sum|\Delta\alpha_t|`）、状态先验（资金驱动 Regime 放宽 α 上限）。

---

## 数据形态与防泄露（强制规范）

- 长表：`[timestamp, symbol, period, <features...>]` 多周期合一。
- 慢频广播：1d/4h → 4h/1h 先 `shift(1)` 再 `ffill`，并记录 `is_missing_*`、`lag_k`。
- 通道归类：`static / known_future / observed_past`（known-future 如日历/合约属性独立通道）。
- 可用性/质量：为 Rich 关键字段生成 `rich_available_t∈{0,1}` 与 `rich_quality_t∈[0,1]`（由延迟/异常/缺失率折算）。
- 时代标签：`era_id`（如 pre-2020、post-2020、post-2024ETF…）用于稳健性统计、门控与报告。

---

## 特征选择（四类证据 + 时间友好 + 双层清单）

1) 候选池（Step 0）：多模态广覆盖，标注可用时代与缺失模式。
2) 快速过滤（Step 1）：覆盖率/低方差/强共线/同源冗余（多滞后/多尺度按组去冗；`_zn/_mm` 只保留一种形态）。
3) 内嵌式证据（Step 2）：树模型重要性 + Boruta 找“所有有用”；TFT-VSN 的全局特征权重作校验；用 SHAP 复核。
4) 时间感知置换（Step 3）：对 1h/4h/1d 分别进行块置换/循环移位，统计 ΔMetric 与出现率，并跨折/跨时代聚合名次。
5) 包装式搜索（Step 4）：LightGBM-RFE 或 GA 进行子集搜索（多目标/多任务可用加权适应度），多次重复取稳定入选特征。
6) 定稿双清单（Step 5）：
   - 通用核心：跨周期/跨时代稳定靠前 + 多证据一致；
   - 专用增强：某专家×某周期×某模态/币显著增益的小集合（限额）。
7) 稳健性（Step 6）：嵌套时序 CV 与前瞻窗复核，确保“选择-评估”严格隔离；必要时保留略大的稳健子集。
8) 证据卡片：VSN 权重统计、Permutation Δ、SHAP、跨折/跨时代出现率、时代覆盖、入选理由与变更历史。

- 多专家场景注意：可先选统一基础集，再为特定专家补充少量“专用增强”，在维护复杂度与任务最优之间折中。

---

## 训练与评估（嵌套时序 CV、单指标早停、分币日志）

- 切分：
  - 内层：特征筛选/门控调参/Adapter 微调；
  - 外层：滚动或扩窗前瞻（裁决与出报告）。
- 指标（按任务、分周期×分币记录）
  - 分类：PR-AUC、F1（阈值经校准）；
  - 回归：RMSE、MAE；
  - 运行期可再计算策略度量（胜率、收益/回撤、卡玛比）供 Z 层调权。
- 早停与保存：采用单一主指标早停（分类用 PR-AUC / Macro-F1，回归用 RMSE），已弃用 composite_score（合成分仅用于分析报告，不用于早停/适应度）。
- 两阶段：支持“Base 预训练（全时代×全币）→ Rich 预热 + 联训（在 `m_t=1` 子集 warm-start，再与 Base 共训 + α 门控）”。
- 混币训练：批内混币，样本 `(x, s)` 只对 `Head[s]` 回传；GroupNormalizer(symbol) 归一化；“问题币”可增设小型 Adapter。
- 多模型融合：先简单加权 baseline，再演进到 Stacking / 门控。

---

## 组合与风控（Z 层）

- Base↔Rich 融合：由 α 门控完成（并行融合，Rich 缺失=α 归零）。
- 专家间加权：Regime 输出驱动状态化权重模板；`uncertainty` 与 `recent_valid_perf` 动态折扣。
- 风格约束：因子桥接的 `factor_exposures / style_score` 作为偏离惩罚，限制 Alpha 风格漂移。
- 风险制动：Risk-TFT 输出转为仓位上限/止损阈值硬约束。
- 择币/对冲：相对强弱专家驱动跨币权重与对冲比率。
- 成本/延迟闸门：将特征数、推理延迟、内存与训练时长作为上线与回滚的硬指标。

---

## 监控与审计（线上）

- 必记：分周期×分币的 PR-AUC/F1、RMSE/MAE；`α_t` 使用率与分布；`uncertainty`；`recent_valid_perf`；关键特征分布漂移；门控与权重热力图。
- 告警：
  - 指标跌破阈值（例：7/30 天移动均值下穿历史分位）；
  - `α_t` 长期卡死（→ 门控失灵）；
  - 重要性/分布漂移超阈（→ 触发再训练或暂时禁用“专用增强”）。
- 证据卡片：按月固化（VSN/Permutation/SHAP/出现率/时代覆盖/α 使用率），服务审计与复盘。

---

## 版本与命名（不可省）

- Checkpoint 命名（去除 composite）：`[Expert]_[Period]_[BASE|RICH]_[ALL|SYM]_{val_metric=...}_{val_loss=...}_{ver}.ckpt`
- 数据视图：`dataset_ver`（含特征字典 hash、时代边界、缺失策略）；
- 特征清单：`features_core.yaml`、`features_boost_[expert]_[period].yaml`（专用增强限额）；
- 门控/路由：`gating_ver`（含输入维度、先验/正则参数）；
- 一键回滚：版本索引（组件 ver → ckpt 路径与依赖）。

---

## 与传统因子量化的融合位点（速览）

- 前置位：因子桥接专家生成 `style_score` 与 `factor_exposures`，用于训练期正交化与线上风格约束；
- 中间位：Regime/Stacking 使用因子摘要作为元特征（可解释、稳健、低噪声）；
- 后置位：Z 层将 AI-Alpha 与因子 Alpha 并行（双模），由元学习/规则加权，形成“人机结合”的稳健组合。

---

## 附：为什么坚持“Base/Rich 并行 + 堆叠/门控”？

历史覆盖与新模态间存在天然张力。Base/Rich 并行能兼顾“长历史稳健”与“新模态增益”，并在 Rich 缺失时自然降级。Stacking/门控能让数据自己学会“何时更信任谁”，比静态设权更鲁棒，是时间序列多专家体系的常用升级路径。

---

## 工程目录（参考）

```
/experts/
  alpha_dir/   alpha_ret/   risk/   derivatives/   onchain/   regime/   structure/   relative/   factor_bridge/
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

## YAML 模板（示例，裁剪即可用）

```yaml
expert: Alpha-Dir-TFT
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
    adapter: {enabled: true, dims: 32}
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
  smooth_l1_alpha: 0.01
  hard_mask_on_unavailable: true

training:
  optimizer: adamw
  lr: 1e-3
  weight_decay: 0.01
  warmup_steps: 1000
  batch_size: 512
  loss:
    main: bce          # Dir 用 bce；Ret 用 mse/rmse
    aux_base: 0.2
    aux_rich: 0.2
  early_stopping:
    metric: pr_auc     # 回归任务改 rmse
    patience: 10

cv:
  outer: {type: rolling, windows: 6, step: 1}
  inner: {type: rolling, windows: 4}
  seed: 2025

selection:
  filter: {low_variance: true, corr_thresh: 0.9, group_redundancy: true}
  embedded: {vsn: true, boruta: true, shap_check: true}
  permutation: {block_len: 96, method: circular_shift}
  wrapper: {method: rfe, estimator: lgbm, max_features: 64}
  lists:
    core_thresholds: {appearance_rate: 0.6, min_perm_delta: 0.002}
    boost_limits: {per_expert_period: 12}

logging:
  save_alpha_series: true
  save_importance: true
  save_metrics_by_symbol: true
```

---

## 排期与分工（现实可执行）

- Week 1：Alpha-Dir / Alpha-Ret / Risk / Regime 的 Base 分支跑通（含特征选择、嵌套 CV、单指标早停、分币日志）；Z 层规则版上线。
- Week 2：衍生品/结构接入；Rich 分支 warm-start + α 联训；并行融合上线；监控 α/漂移。
- Week 3：链上与因子桥接补位（风格约束 + 择币/对冲）；完成证据卡片自动化；评审是否需要对极少数币试点 B 方案。
- 持续：每月滚动/前瞻评估、证据卡片固化、阈值回顾与成本/延迟体检。

分工建议：

- FE（特征&数据）：广播/滞后/缺失指示/era 标签、字典维护与稳定性统计
- ML（建模&训练）：Base/Rich 两塔、门控、嵌套 CV、特征选择与调参
- STRAT（策略&Z 层）：权重路由、约束、成本/滑点、回测与上线闸门
- OPS（工程&监控）：模型服务、指标与 α/漂移监控、A/B 实验、回滚

---

## 触发 B 方案（单币小 TFT 微调）的硬标准

满足任一即评估（先灰度）：

1) 某币在 ≥3 个外层前瞻窗，主指标持续落后本周期组内中位数 ≥10%；
2) 诊断显示强负迁移（混币训练显著变差、单币训练显著变好且可复现）；
3) 存在强独占且高权重模态（如 BTC-ETF 在 1d 极强），A+C 吃不满；
4) 资源允许且已扣除成本后仍显著提升。

做法：从全币预训练权重复制，冻结 ≥80% 层，仅末端/Adapter 小步微调；配 L2-SP/EWC/蒸馏 防遗忘；与 A 方案同币 Head 做 A/B，显著性达标才投产。

---

## 三条“红线”确保上线稳健

1) 所有选择/调参必须在内层时序 CV 完成，外层仅裁决（防“选特征过程”泄漏）。
2) Rich 与慢频数据必须 `shift→broadcast`，并带 `rich_available/quality`；融合=软加权，回退只是 α=0 特例。
3) 上线闸门包含性能（主指标显著优于基线）+ 成本/延迟 + 漂移监控三件套；不满足任一不得放量。

---

一句话总括：

> 专家清单 9+1、职责清晰；每专家×每周期一份共享骨干 Multi-TFT + 多符号头/Adapter；Base 与 Rich 并行融合（α 门控），缺失时自然回退；多证据+时间友好选特征，嵌套时序 CV 裁决；Z 层完成权重路由、风格与风险约束；B 方案仅对极少数币兜底微调。以上规范完备、可执行、可审计。
