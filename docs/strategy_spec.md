# 多重触碰下降阻力 / 上升支撑自动交易规格书 v1

更新时间：2026-04-08

## 1. 文档目标

本规格书定义一套后端驱动型自动交易模块，用于在 `crypto-analysis-` 项目中实现以下能力：

1. 自动识别下降阻力线与上升支撑线
2. 自动识别多次触碰后的结构有效性
3. 自动检测触线拒绝、假突破失败、预挂单窗口
4. 自动生成做空/做多信号
5. 自动通过风控审批后生成订单
6. 自动管理订单、持仓、止损止盈、撤单与失效
7. 将全部状态通过后端提供给前端图表渲染
8. 最终支持 bot 自主下单

本系统不依赖 TradingView。
后端 Python 为唯一真相源。
前端只渲染后端产出的结构、信号、订单与状态，不参与策略计算。

补充约束：

1. 当前实际行情源以 Bitget 为主。
2. 但策略核心必须保持交易所无关，只接收标准化后的 OHLCV、symbol metadata、tick size、precision、order status。
3. Bitget/未来其他交易所只作为市场数据层与执行层适配器存在，不得渗透到核心策略函数签名中。

## 2. 总体架构

系统链路固定为：

```text
交易所数据 / 历史K线 / 实时K线
→ 市场数据层
→ 结构识别层（pivot / 趋势线 / 评分）
→ 信号层（触碰 / 拒绝 / 假突破失败）
→ 风控层
→ 执行层
→ 状态同步层
→ 前端图表渲染
```

禁止的做法：

1. 前端自己计算趋势线或信号
2. 回测逻辑和实时逻辑各写一套
3. 在策略判断函数内直接调用交易所 API
4. 跳过风控直接发真实订单

## 3. 输入数据与基础约束

### 3.1 输入数据结构

最小输入为标准 OHLCV bar 序列：

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

可选字段：

- `symbol`
- `timeframe`
- `quote_volume`
- `trade_count`

### 3.2 时间索引

所有趋势线计算统一使用 bar 序号 `t` 作为横轴，而不是直接用时间戳做斜率计算。

定义：

- 第 `t` 根 bar 的价格为 `P(t)`
- 趋势线在第 `t` 根 bar 的理论价格为 `L(t)`

### 3.3 决策粒度

默认决策以已收盘 bar 为准。
不允许使用未收盘 bar 的最终 `high/low/close` 去确认结构。
实盘中如果需要 intrabar 触发，只允许用于已确认结构上的挂单成交或止损触发，不允许用于提前确认新的结构。

## 4. 结构识别定义

### 4.1 ATR 定义

容差、缓冲、失效判断默认都使用 ATR 归一化。

定义：

```text
ATR(t) = AverageTrueRange(period = atr_period)
默认 atr_period = 14
```

### 4.2 Pivot High / Pivot Low 定义

#### 4.2.1 离线定义

给定左右窗口：

- `pivot_left = nL`
- `pivot_right = nR`

则第 `t` 根 bar 为 pivot high 的条件：

```text
high[t] > max(high[t-nL : t-1])
且
high[t] >= max(high[t+1 : t+nR])
```

第 `t` 根 bar 为 pivot low 的条件：

```text
low[t] < min(low[t-nL : t-1])
且
low[t] <= min(low[t+1 : t+nR])
```

默认参数：

- `pivot_left = 3`
- `pivot_right = 3`

#### 4.2.2 实时定义

实时版本中，第 `t` 根 bar 只有在 `t + pivot_right` 收盘后，才允许确认 `t` 是否为 pivot。
也就是说，pivot 确认存在固定延迟。
禁止偷看未来。

### 4.3 候选趋势线生成

#### 4.3.1 下降阻力线候选

从最近 `lookback_bars` 内的 pivot highs 中，枚举任意两个 pivot high `(t1, p1)`、`(t2, p2)`，要求：

- `t2 > t1`
- `p2 < p1`
- 斜率 `m = (p2 - p1) / (t2 - t1) < 0`

由此生成候选阻力线：

```text
L(t) = m * (t - t1) + p1
```

#### 4.3.2 上升支撑线候选

从最近 `lookback_bars` 内的 pivot lows 中，枚举任意两个 pivot low `(t1, p1)`、`(t2, p2)`，要求：

- `t2 > t1`
- `p2 > p1`
- `m > 0`

生成候选支撑线：

```text
L(t) = m * (t - t1) + p1
```

#### 4.3.3 默认窗口

- `lookback_bars = 300`

#### 4.3.4 候选线剪枝与去重

v1 必须在候选线生成阶段就做剪枝，避免两点枚举导致候选线数量爆炸、相似线堆积和结果不稳定。

默认规则：

1. 同一 side 下，每个最新 pivot 只允许向前回看最近 `max_anchor_combinations_per_pivot` 个可用 anchor。
2. 若两条候选线的 slope 差异不超过 `line_merge_slope_eps`，且当前 bar 的 projected price 差异不超过 `line_merge_price_eps(t)`，则视为相似线，只保留 `line_score` 更高的一条。
3. 同一 side 在任意时刻最多保留 `max_candidate_lines_per_side` 条候选线进入评分阶段。
4. 进入 active pool 的线最多保留 `max_active_lines_per_side` 条。
5. 若某条线在非 touch 条件下被实体穿越超过 `max_non_touch_crosses` 次，则该线不得继续留在候选池。

定义：

```text
line_merge_price_eps(t) = max(
  merge_price_atr_mult * ATR(t),
  merge_price_pct * close[t]
)
```

默认参数：

- `max_anchor_combinations_per_pivot = 8`
- `max_candidate_lines_per_side = 50`
- `max_active_lines_per_side = 5`
- `line_merge_slope_eps = 0.0005`
- `merge_price_atr_mult = 0.10`
- `merge_price_pct = 0.001`
- `max_non_touch_crosses = 2`

### 4.4 Touch 容差定义

趋势线不是一根绝对零厚度的像素线，必须定义容差带。

对任意 bar `t`，定义趋势线容差：

```text
tol(t) = max(
  tol_atr_mult * ATR(t),
  tol_pct * close[t],
  tick_size * tick_mult
)
```

默认参数：

- `tol_atr_mult = 0.15`
- `tol_pct = 0.002`
- `tick_mult = 3`

### 4.5 阻力 Touch 定义

对于下降阻力线 `L(t)`，第 `t` 根 bar 记为一次有效阻力 touch，当且仅当：

```text
abs(high[t] - L(t)) <= tol(t)
```

并且：

```text
close[t] <= L(t) + close_touch_slack
```

其中：

```text
close_touch_slack = max(0.05 * ATR(t), 0.0005 * close[t])
```

目的：避免大幅实质性突破的 bar 被误记为普通 touch。

### 4.6 支撑 Touch 定义

对于上升支撑线 `L(t)`，第 `t` 根 bar 记为一次有效支撑 touch，当且仅当：

```text
abs(low[t] - L(t)) <= tol(t)
```

并且：

```text
close[t] >= L(t) - close_touch_slack
```

### 4.7 Touch 去重与最小间隔

相邻过近的 touch 不能重复计数。

定义：

- `min_touch_spacing_bars = 5`

若两个 touch 的 bar 距离小于该阈值，则仅保留更接近趋势线的一次。

#### 4.7.1 v1 中 touch 的角色划分

为避免 pivot、普通触线 bar、确认 touch 混用，v1 明确冻结如下边界：

1. `anchor pivot` 必须来自已确认 pivot。
2. `confirming touch` 也必须来自已确认 pivot，且仅它能参与结构确认、残差计算、line score。
3. `bar touch` 可以来自任意已收盘 bar 命中容差带，但它只用于 `ARMED`、`REJECTION`、`FAILED_BREAKOUT` 等交易触发，不用于把线从 `CANDIDATE` 升级到 `CONFIRMED`。

因此：

- 结构确认统计使用 `confirming_touch_count`
- 交易触发统计使用 `bar_touch`

### 4.8 三次确认定义

一条候选趋势线从 `CANDIDATE` 升级为 `CONFIRMED`，必须同时满足：

1. `confirming_touch_count >= min_touches`
2. `min_touches = 3`
3. 任意相邻两次 `confirming touch` 间隔 `>= min_touch_spacing_bars`
4. 线的最大残差 `<= max_line_error`
5. 在确认前未发生明确失效

其中残差定义为：

```text
residual_i = abs(confirming_pivot_price_i - L(t_i))
```

最大允许残差：

```text
max_line_error(t) = max(
  line_error_atr_mult * ATR(t),
  line_error_pct * close[t]
)
```

默认参数：

- `line_error_atr_mult = 0.20`
- `line_error_pct = 0.003`

### 4.9 趋势线质量评分

定义线质量分数 `line_score`，范围 `[0, 100]`：

```text
line_score =
  25 * touch_score
+ 20 * fit_score
+ 15 * spacing_score
+ 15 * recency_score
+ 15 * slope_score
+ 10 * cleanliness_score
- 20 * breakout_risk_penalty
```

各项定义如下：

#### touch_score

```text
touch_score = min(confirming_touch_count / 5, 1)
```

#### fit_score

```text
fit_score = 1 - normalized_mean_residual
```

#### spacing_score

```text
spacing_score = clamp(mean_confirming_touch_gap / target_touch_gap, 0, 1)
```

默认：

- `target_touch_gap = 12`

#### recency_score

```text
recency_score = clamp(1 - bars_since_last_confirming_touch / max_fresh_bars, 0, 1)
```

#### slope_score

```text
slope_score =
  1, if min_slope_abs <= abs(m) <= max_slope_abs
  0, otherwise
```

默认：

- `min_slope_abs = 0.0001`
- `max_slope_abs = 0.0200`

#### cleanliness_score

```text
cleanliness_score = clamp(1 - non_touch_cross_count / cleanliness_cross_cap, 0, 1)
```

默认：

- `cleanliness_cross_cap = 3`

#### breakout_risk_penalty

```text
breakout_risk_penalty = clamp(recent_test_count / breakout_risk_test_cap, 0, 1)
```

其中 `recent_test_count` 定义为最近 `recent_test_window_bars` 内对同一条线的 `bar touch` 次数。

默认：

- `recent_test_window_bars = 30`
- `breakout_risk_test_cap = 4`

默认确认阈值：

```text
line_score >= 60
```

## 5. 因子定义

这套模型不是单纯“碰线就做”，而是一个结构事件评分系统。

### 5.0 v1 因子冻结原则

为避免阶段 3 实现时由代码自行发明公式，v1 只保留可直接公式化的因子项，复杂上下文项先冻结为 `0` 或降级为简化版本：

- `VolumeFailure = 0`
- `TrendContext = 0`
- `ConfluenceScore = 0`
- `FreshnessScore = clamp(1 - bars_since_last_confirming_touch / max_fresh_bars, 0, 1)`
- `BreakoutRisk = clamp(recent_test_count / breakout_risk_test_cap, 0, 1)`

后续版本如需启用量能、高周期趋势、VWAP 共振等项，必须先补正式公式和测试，再提高规格版本。

### 5.1 做空因子：ResistanceShortScore

定义：

```text
ResistanceShortScore(t) =
  0.26 * TouchStrength
+ 0.20 * FitTightness
+ 0.26 * RejectionStrength
+ 0.16 * DistanceCompression
+ 0.12 * FreshnessScore
- 0.15 * BreakoutRisk
```

范围建议归一到 `[0, 1]`。

#### 组成项说明

##### TouchStrength

```text
TouchStrength = min(confirming_touch_count / 5, 1)
```

##### FitTightness

```text
FitTightness = clamp(1 - normalized_mean_residual, 0, 1)
```

##### RejectionStrength

当前触线 bar 的拒绝程度：

```text
RejectionStrength = clamp(
  0.6 * wick_score + 0.4 * close_reclaim_score,
  0,
  1
)
```

其中：

```text
wick_score = clamp(upper_wick_ratio / rejection_wick_ratio_cap, 0, 1)
close_reclaim_score = clamp((L(t) - close[t]) / rejection_close_norm, 0, 1)
rejection_close_norm = max(0.12 * ATR(t), 0.0015 * close[t])
```

##### DistanceCompression

```text
DistanceCompression = clamp(
  1 - abs(close[t] - projected_resistance_next) / arm_distance,
  0,
  1
)
```

##### FreshnessScore

```text
FreshnessScore = clamp(1 - bars_since_last_confirming_touch / max_fresh_bars, 0, 1)
```

##### BreakoutRisk

```text
BreakoutRisk = clamp(recent_test_count / breakout_risk_test_cap, 0, 1)
```

### 5.2 做多因子：SupportLongScore

与做空对称定义：

```text
SupportLongScore(t) =
  0.26 * TouchStrength
+ 0.20 * FitTightness
+ 0.26 * RejectionStrength
+ 0.16 * DistanceCompression
+ 0.12 * FreshnessScore
- 0.15 * BreakdownRisk
```

其中 `RejectionStrength`、`DistanceCompression`、`FreshnessScore` 与做空对称定义，`BreakdownRisk` 使用相同的简化公式：

```text
BreakdownRisk = clamp(recent_test_count / breakout_risk_test_cap, 0, 1)
```

### 5.3 因子筛选阈值

默认建议：

- `ResistanceShortScore >= 0.62` 才允许进入做空候选
- `SupportLongScore >= 0.62` 才允许进入做多候选

注意：
这个阈值只是初始值，必须回测后校准。

## 6. 检测逻辑

### 6.1 结构检测顺序

每根 bar 收盘后按以下顺序执行：

1. 更新 OHLCV / ATR / 辅助指标
2. 更新 pivot 确认结果
3. 更新候选趋势线池
4. 对每条候选线重新计算 touch、残差、评分
5. 更新趋势线状态
6. 生成 signal candidate
7. 送入风控审批
8. 审批通过后生成订单意图
9. 交给执行层处理

### 6.2 触发模型 A：预挂单

#### 做空

条件：

1. `line_state == CONFIRMED`
2. `ResistanceShortScore >= score_threshold`
3. 当前价格距下一根 projected resistance 不超过 arm 区域

定义：

```text
projected_resistance_next = L(t+1)
arm_distance = max(
  arm_atr_mult * ATR(t),
  arm_pct * close[t]
)
```

当满足：

```text
abs(close[t] - projected_resistance_next) <= arm_distance
```

则进入 `ARMED` 状态。

挂单价：

```text
entry_price = projected_resistance_next - entry_buffer
```

其中：

```text
entry_buffer = max(
  entry_buffer_atr_mult * ATR(t),
  entry_buffer_pct * close[t],
  tick_size * entry_tick_mult
)
```

默认：

- `arm_atr_mult = 0.20`
- `arm_pct = 0.003`
- `entry_buffer_atr_mult = 0.03`
- `entry_buffer_pct = 0.0005`
- `entry_tick_mult = 1`

#### 做多

对称处理。

### 6.3 触发模型 B：触线拒绝

#### 做空

定义拒绝条件：

```text
high[t] >= L(t) - tol(t)
and close[t] < L(t) - rejection_close_buffer
and upper_wick_ratio(t) >= wick_ratio_threshold
```

其中：

```text
upper_wick_ratio(t) =
  (high[t] - max(open[t], close[t])) / max(abs(close[t] - open[t]), min_body_unit)
```

默认：

- `rejection_close_buffer = max(0.08 * ATR(t), 0.001 * close[t])`
- `wick_ratio_threshold = 1.2`

满足后生成 `REJECTION_SHORT` 信号。

#### 做多

对称定义，用下影线、收在线上方。

### 6.4 触发模型 C：假突破失败

#### 做空

定义：

```text
high[t] > L(t) + break_tol(t)
and close[t] < L(t) - failed_break_close_buffer
and low[t+1] < low[t] - trigger_buffer
```

其中：

```text
break_tol(t) = max(0.10 * ATR(t), 0.001 * close[t])
failed_break_close_buffer = max(0.08 * ATR(t), 0.001 * close[t])
trigger_buffer = max(0.02 * ATR(t), tick_size)
```

说明：
这个定义在实时系统中，最后一条条件必须在下一根 bar 完成后才能确认，因此这是一个延迟确认模型。

#### 做多

对称处理。

## 7. 状态机

必须拆成两套状态机：
一套给趋势线，一套给订单。

### 7.1 趋势线状态机

状态集合：

- `CANDIDATE`
- `CONFIRMED`
- `ARMED`
- `TRIGGERED`
- `INVALIDATED`
- `EXPIRED`
- `CLOSED`

#### 转移规则

##### `CANDIDATE -> CONFIRMED`

满足：

- `confirming_touch_count >= 3`
- `line_score >= confirm_threshold`
- 未失效

##### `CONFIRMED -> ARMED`

满足：

- 当前价格进入 arm 区
- 因子分数超过阈值
- 非冷却期

##### `ARMED -> TRIGGERED`

满足任一：

- 预挂单被允许提交
- 触线拒绝信号出现
- 假突破失败信号出现

##### 任意状态 -> `INVALIDATED`

满足：

- 明确突破 / 跌破
- 连续收盘穿越趋势线
- 突破距离超过阈值
- 超过最大存活期
- 被更高优先级结构替代

##### `CONFIRMED / ARMED -> EXPIRED`

满足：

- `bars_since_last_confirming_touch > max_fresh_bars`
- 结构过旧

##### `TRIGGERED -> CLOSED`

对应订单生命周期结束，且该线本轮交易完成。

### 7.2 订单状态机

状态集合：

- `SIGNAL_DETECTED`
- `RISK_APPROVED`
- `ORDER_PENDING`
- `ORDER_ACKED`
- `PARTIALLY_FILLED`
- `FILLED`
- `STOP_ACTIVE`
- `TP_ACTIVE`
- `CANCELLED`
- `REJECTED`
- `CLOSED`

必须维护：

- `signal_id`
- `line_id`
- `client_order_id`
- `strategy_run_id`

#### signal_id 定义

```text
signal_id = hash(
  symbol,
  timeframe,
  line_id,
  trigger_bar_timestamp,
  signal_type
)
```

相同 `signal_id` 只能被执行一次。
必须有幂等锁。

## 8. 下单规则

### 8.0 同 symbol 信号优先级与冲突处理

当同一 `symbol` / `timeframe` 上同时存在多条候选线或多个待执行信号时，v1 固定使用以下排序：

1. `score` 更高者优先
2. `confirming_touch_count` 更多者优先
3. 当前价格到目标线的距离更近者优先
4. `trigger_mode` 优先级按 `failed_breakout > rejection > pre_limit`
5. 最近一次 confirming touch 更新更近者优先
6. 若仍并列，则按稳定排序键 `line_id` 升序

冲突处理规则：

1. 同一 `symbol` 同时只允许一个活跃方向仓位。
2. 若已有持仓，则反向新信号默认阻塞，不做同 bar 反手。
3. 若已有待成交订单，则低优先级信号不得抢占，除非高优先级信号已触发显式撤单条件。

### 8.1 下单前置条件

任何订单提交前，必须满足：

1. `signal_state == SIGNAL_DETECTED`
2. 风控审批通过
3. 当前无相同 `signal_id` 的活跃订单
4. 当前未超过最大持仓限制
5. 当前未触发 kill switch
6. 当前 `symbol` 未处于冷却期

### 8.2 仓位计算

定义：

```text
risk_amount = account_equity * risk_per_trade
position_size = risk_amount / abs(entry_price - stop_price)
```

还需考虑：

- 合约乘数
- 最小下单单位
- 杠杆限制
- 最小 notional

默认：

- `risk_per_trade = 0.003`（0.3%）
- 初始 live 建议 `0.001`（0.1%）

### 8.3 止损规则

#### 8.3.0 按触发模式拆分止损源

v1 明确不同 trigger mode 的止损参照物，禁止实现时自行推断：

- `pre_limit`
  - 做空：`stop_source = max(projected_line_price, latest_confirming_touch_high)`
  - 做多：`stop_source = min(projected_line_price, latest_confirming_touch_low)`
- `rejection`
  - 做空：`stop_source = rejection_bar_high`
  - 做多：`stop_source = rejection_bar_low`
- `failed_breakout`
  - 做空：`stop_source = breakout_failure_bar_high`
  - 做多：`stop_source = breakout_failure_bar_low`

#### 做空

默认止损：

```text
stop_price = stop_source + stop_buffer
```

其中：

```text
stop_buffer = max(
  stop_atr_mult * ATR(t),
  stop_pct * close[t],
  tick_size * stop_tick_mult
)
```

默认：

- `stop_atr_mult = 0.12`
- `stop_pct = 0.0015`
- `stop_tick_mult = 2`

#### 做多

```text
stop_price = stop_source - stop_buffer
```

### 8.4 止盈规则

支持三种，初始 v1 先实现前两种：

#### 规则 1：固定 RR

```text
tp_price = entry_price ± rr_target * risk_per_unit
```

默认：

- `rr_target = 2.0`

#### 规则 2：结构目标位

- 做空：最近 `swing low` / 支撑区
- 做多：最近 `swing high` / 阻力区

#### 规则 3：分批止盈

- 50% 在 `1R`
- 30% 在 `2R`
- 20% trailing

v1 可以先不实现 trailing。

### 8.5 撤单规则

以下情况必须撤单：

1. 限价单超过 `cancel_after_bars`
2. 趋势线失效
3. 因子分数跌破保持阈值
4. 同 `symbol` 出现更高优先级反向信号
5. 进入风控禁开状态

默认：

- `cancel_after_bars = 3`

### 8.6 同 bar 冲突成交规则

K 线回测与 paper trading 无法知道同一 bar 内部真实成交顺序时，v1 统一采用保守最差成交规则：

1. 若同一 bar 内同时可达 `entry` 与 `stop`，则按“先成交 entry，再触发 stop”处理。
2. 若同一 bar 内同时可达 `entry` 与 `tp`，仍按保守原则，不允许假设先到达 tp。
3. 若同一 bar 内同时可达 `entry`、`stop`、`tp`，则按最不利顺序处理，即视为 `entry -> stop`。
4. 该规则必须同时用于 backtest、replay 导出和 paper trading fill simulation。

## 9. 风控规则

### 9.1 基础风险约束

必须支持：

- `risk_per_trade`
- `max_concurrent_positions`
- `max_positions_per_symbol`
- `max_daily_loss`
- `max_consecutive_losses`
- `max_total_exposure`
- `cooldown_bars_after_loss`

默认：

- `max_concurrent_positions = 3`
- `max_positions_per_symbol = 1`
- `max_daily_loss = 0.02`
- `max_consecutive_losses = 3`
- `cooldown_bars_after_loss = 10`

### 9.1.1 冷却期定义

v1 将冷却期固定为可执行规则，避免执行层和回测层理解不一致：

- `cooldown_scope = symbol + timeframe + direction`
- `cooldown_trigger = stopped_out or invalidated_after_entry`
- `cooldown_bars = cooldown_bars_after_loss`
- `tp_close` 不触发 cooldown
- 未成交订单取消、未入场信号失效，不触发 cooldown

### 9.2 禁开仓条件

以下任一成立时禁止新开仓：

1. 日内亏损达到阈值
2. 连续亏损达到阈值
3. 当前持仓过多
4. 当前 `symbol` 已有活跃仓位
5. 当前信号在冷却期
6. 数据源异常
7. 订单对账异常
8. 实时行情滞后超阈值
9. 时钟不同步
10. 手续费 / 滑点异常放大

### 9.3 Kill Switch

必须实现全局熔断开关。
任一成立即停止新单并视情况平仓：

1. 数据流中断
2. 订单确认异常
3. 持仓与本地状态不一致
4. 异常连续报错超过阈值
5. 超过最大日亏损
6. 人工触发

## 10. 失效规则

### 10.1 阻力线失效

下降阻力线失效，当任一条件成立：

1. 连续 `break_close_count` 根收盘在线上方
2. 单根收盘高于趋势线超过 `break_distance`
3. 趋势线被大阳实体有效穿越
4. 线过旧且长时间未重新测试
5. 出现明显高一级别反向结构

定义：

```text
break_distance = max(
  break_atr_mult * ATR(t),
  break_pct * close[t]
)
```

默认：

- `break_close_count = 2`
- `break_atr_mult = 0.20`
- `break_pct = 0.003`

### 10.2 支撑线失效

与阻力线对称。

### 10.3 线过期

当：

```text
bars_since_last_confirming_touch > max_fresh_bars
```

则线进入 `EXPIRED`。

默认：

- `max_fresh_bars = 80`

## 11. 回测约束

这部分必须硬性执行，否则回测结果没有意义。

### 11.1 禁止未来函数

1. pivot 只能在 `pivot_right` 之后确认
2. 假突破失败只能在下一根触发确认
3. 不允许用未来最高/最低回填前面状态
4. 每根 bar 只能使用该 bar 收盘时可知的信息

### 11.2 逐 bar 回放

回测必须逐 bar 重放。
每根 bar 输出状态快照：

- pivots
- active lines
- line states
- scores
- signals
- orders
- positions

### 11.3 成本模型

回测必须计入：

- 手续费
- 滑点
- 最小跳价
- 成交延迟（若模拟）

### 11.4 订单成交规则

v1 默认：

1. 限价单：若下一 bar 的价格区间包含委托价，则视为成交
2. 市价单：按下一 bar 开盘价加滑点成交
3. 止损：触及则按最差合理价格成交
4. 同 bar 的 `entry / stop / tp` 冲突一律按 8.6 的保守最差成交规则处理
5. 允许未来版本加入部分成交模拟

### 11.5 回测与实时一致性

实时和回测必须共用同一套核心函数：

- `detect_pivots`
- `build_candidate_lines`
- `score_lines`
- `update_line_state`
- `generate_signal`
- `compute_order_plan`

并且必须共享同一套：

- 候选线剪枝 / 去重规则
- 信号优先级规则
- 冷却期规则
- 同 bar 冲突成交规则

禁止：

- 回测写一套逻辑
- 实盘另写一套简化逻辑

## 12. 实盘约束

### 12.1 启动同步

实盘引擎启动时必须先同步：

1. 当前账户持仓
2. 当前挂单
3. 最近订单记录
4. 本地状态缓存

若不一致，必须进入保护模式，不允许直接继续发单。

### 12.2 实盘开关

真实下单必须同时满足：

- `ENABLE_LIVE_TRADING=true`
- `DRY_RUN=false`
- `CONFIRM_LIVE_TRADING=true`

任一不满足，则只能 paper trade。

### 12.3 幂等与对账

每张订单必须带：

- `client_order_id`
- `signal_id`
- `line_id`

执行层必须定时对账：

- 本地订单状态 vs 交易所订单状态
- 本地持仓 vs 实际持仓

若不一致，进入 `reconciliation_required` 状态。

### 12.4 实盘早期限制

刚上线时，强制：

1. 单 `symbol`
2. 单 `timeframe`
3. 小仓位
4. 单方向或低并发
5. 人工监控日志与图表同步

## 13. 前端显示字段

前端只渲染，不计算。

### 13.1 趋势线字段

每条线至少返回：

- `line_id`
- `symbol`
- `timeframe`
- `side` (`resistance` / `support`)
- `state`
- `t_start`
- `t_end`
- `price_start`
- `price_end`
- `slope`
- `intercept`
- `bar_touch_count`
- `confirming_touch_count`
- `recent_bar_touch_count`
- `line_score`
- `projected_price_current`
- `projected_price_next`
- `is_active`
- `is_invalidated`
- `invalidation_reason`

### 13.2 Touch 点字段

- `line_id`
- `timestamp`
- `bar_index`
- `price`
- `touch_type`
- `residual`
- `is_confirming_touch`

### 13.3 信号字段

- `signal_id`
- `line_id`
- `signal_type`
- `direction`
- `timestamp`
- `trigger_bar_index`
- `score`
- `priority_rank`
- `entry_mode`
- `entry_price`
- `stop_price`
- `tp_price`
- `risk_reward`
- `status`

### 13.4 订单字段

- `order_id`
- `client_order_id`
- `signal_id`
- `symbol`
- `side`
- `order_type`
- `price`
- `quantity`
- `status`
- `filled_quantity`
- `avg_fill_price`
- `stop_price`
- `tp_price`
- `cooldown_scope`
- `created_at`
- `updated_at`

### 13.5 持仓字段

- `position_id`
- `symbol`
- `side`
- `quantity`
- `entry_price`
- `mark_price`
- `unrealized_pnl`
- `realized_pnl`
- `stop_price`
- `tp_price`
- `linked_signal_id`
- `opened_at`
- `close_reason`

### 13.6 前端图层要求

图表必须支持开关以下图层：

1. pivot 点
2. active resistance lines
3. active support lines
4. touch markers
5. projected line
6. armed zone
7. rejection marker
8. failed breakout marker
9. order entry marker
10. stop / tp marker
11. invalidation marker
12. position marker

## 14. 建议模块拆分

结合当前仓库真实结构，建议新增或整理为：

```text
server/
  strategy/
    config.py
    types.py
    atr.py
    pivots.py
    trendlines.py
    scoring.py
    factors.py
    signals.py
    state_machine.py
    replay.py
    order_plan.py

  market/
    bitget_client.py
    bitget_adapter.py

  risk/
    risk_rules.py
    kill_switch.py
    sizing.py

  execution/
    exchange_client.py
    order_manager.py
    position_manager.py
    reconciliation.py
    bitget_trader.py

  routers/
    strategy.py
    orders.py
    positions.py

tests/
  strategy/
    test_pivots.py
    test_trendlines.py
    test_scoring.py
    test_signals.py
    test_state_machine.py
    test_order_plan.py
    test_replay.py
```

前端建议：

```text
frontend/
  js/
    workbench/
      overlays/
        trendline_overlay.js
        touch_overlay.js
        signal_overlay.js
        order_overlay.js
        position_overlay.js
```

## 15. 默认参数初始值

这些不是最终最优值，只是 v1 初值：

```text
pivot_left = 3
pivot_right = 3
lookback_bars = 300

tol_atr_mult = 0.15
tol_pct = 0.002
line_error_atr_mult = 0.20
line_error_pct = 0.003
min_touches = 3
min_touch_spacing_bars = 5
confirm_threshold = 60
max_anchor_combinations_per_pivot = 8
max_candidate_lines_per_side = 50
max_active_lines_per_side = 5
line_merge_slope_eps = 0.0005
merge_price_atr_mult = 0.10
merge_price_pct = 0.001
max_non_touch_crosses = 2
target_touch_gap = 12
min_slope_abs = 0.0001
max_slope_abs = 0.0200
cleanliness_cross_cap = 3
recent_test_window_bars = 30
breakout_risk_test_cap = 4

arm_atr_mult = 0.20
arm_pct = 0.003
entry_buffer_atr_mult = 0.03
entry_buffer_pct = 0.0005

stop_atr_mult = 0.12
stop_pct = 0.0015
rr_target = 2.0

break_close_count = 2
break_atr_mult = 0.20
break_pct = 0.003
max_fresh_bars = 80

risk_per_trade = 0.003
max_concurrent_positions = 3
max_positions_per_symbol = 1
max_daily_loss = 0.02
max_consecutive_losses = 3
cancel_after_bars = 3
cooldown_bars_after_loss = 10
cooldown_scope = symbol + timeframe + direction
same_bar_conflict_policy = conservative_worst_case
trigger_mode_priority = failed_breakout > rejection > pre_limit
```

## 16. 给 Codex / Cursor 的直接执行指令

下面这段可以直接贴进 Codex 或 Cursor：

```text
你正在本地仓库 crypto-analysis- 中工作。

目标：
按《多重触碰下降阻力 / 上升支撑自动交易规格书 v1》实现一个后端驱动型自动交易模块。不要接 TradingView，不要把策略写进前端。后端 Python 是唯一真相源，前端只负责渲染后端输出的趋势线、触点、信号、订单和持仓状态。

开发规则：
1. 先读真实仓库结构，不要凭空假设目录。
2. 优先复用现有市场数据、图表、API、执行相关模块。
3. 所有关键阈值必须集中配置。
4. 策略核心必须是纯函数优先。
5. 回测和实时必须共用同一套核心逻辑。
6. 不允许 lookahead bias。
7. 不允许前端自己重算策略。
8. 所有重要状态和事件必须结构化日志。
9. 每一阶段完成后都输出 review 包。

阶段顺序：
阶段1：生成 docs/implementation_plan.md，只做仓库审计和实施计划，不写业务逻辑。
阶段2：生成 docs/strategy_spec.md，把上面的规格书整理落盘，并结合真实仓库结构补充模块映射。
阶段3：实现纯策略核心：
  - pivots
  - trendlines
  - scoring
  - factors
  - signals
  - state_machine
  - replay
并补单元测试。
阶段4：实现后端 API / websocket 输出趋势线、信号、订单、持仓状态。
阶段5：将后端状态接入现有前端图表 overlay，前端只渲染。
阶段6：实现 paper trading 执行层和风控层。
阶段7：在 paper trading 稳定后，再增加 live trading 通道，但默认关闭。

本次只执行阶段1。
请先读取仓库真实结构，输出：
1. 当前结构摘要
2. 可复用模块
3. 需要新增模块
4. 建议目录结构
5. 分阶段实施顺序
6. 每阶段验收标准
7. review 包
```

## 17. 这份 v1 后续必须重点回测的地方

这份规格书已经够 Codex 开工，但还不是最终参数定稿。
后面必须重点验证：

1. 第四次触碰到底比第三次更优还是更危险
2. 预挂单、拒绝确认、假突破失败三种模式谁更稳
3. `tol_atr_mult` 和 `line_error_atr_mult` 对不同币种的敏感度
4. 高波动币与低波动币是否需要分组参数
5. 同一因子在 `5m / 15m / 1h` 的迁移性
6. 做空和做多是否需要非对称参数
