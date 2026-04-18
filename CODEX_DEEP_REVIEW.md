# DEEP CODE REVIEW — 逐行审计指令

> 这份文档指导 Codex/AI 对整个交易系统进行**极其严格、逐行、不留死角**的 code review。
> 不是泛泛而谈"看起来不错"——是**每一个函数、每一个条件判断、每一个API调用**都要验证。

---

## 0. 必读文件（开始之前）

1. `PRINCIPLES.md` — 不可违反的项目原则
2. `TRENDLINE_TRADING_RULES.md` — Axel 的交易逻辑定义（一字不改）
3. `CLAUDE.md` — 开发规则和反模式
4. `CODEX_REVIEW_INSTRUCTIONS.md` — 之前的review checklist

---

## 1. 挂单流程审计（CRITICAL — 每一步都要验证）

### 文件: `server/strategy/trendline_order_manager.py`

**逐行检查：**

- [ ] `update_trendline_orders()` 函数签名：接收 `new_signals`, `current_bar_index`, `cfg`
- [ ] `cfg` 里有哪些字段？列出每一个。确认没有遗漏。
- [ ] `raw_buffer_pct = cfg.get("buffer_pct", 0.10) / 100` — 这个除以100对吗？如果cfg传的是0.10（百分比），除以100变成0.001。验证调用方传的值。
- [ ] `tf_buffer_map = cfg.get("tf_buffer", {})` — 这个字典的format是什么？`{"5m": 0.05}` 还是 `{"5m": 0.0005}`？
- [ ] `buffer_pct = tf_buffer_map.get(tf, raw_buffer_pct * 100) / 100` — 这个逻辑对吗？如果tf_buffer是`{"5m": 0.05}`，那`0.05/100 = 0.0005`。但回测用的buffer=0.0005。验证一致性。
- [ ] support line: `limit_px = proj * (1 + buffer_pct)` — 确认方向正确（long买在线上方）
- [ ] resistance line: `limit_px = proj * (1 - buffer_pct)` — 确认方向正确（short卖在线下方）
- [ ] `stop_px = proj` — 确认SL = 线本身
- [ ] `tp_px = limit_px * (1 + buffer_pct * rr)` for long — 确认TP计算正确
- [ ] `tp_px = limit_px * (1 - buffer_pct * rr)` for short — 确认TP计算正确
- [ ] `risk_usd = equity * per_tf_risk` — per_tf_risk 从哪来？确认是正确的TF对应的risk
- [ ] `qty = risk_usd / stop_distance` — stop_distance = abs(limit_px - stop_px) = buffer距离。确认数值正确。
- [ ] 最小距离filter `dist < 0.003` — 这个0.3%的阈值对吗？和buffer比较？
- [ ] `_is_bar_boundary(tf)` — 在MOVED逻辑里用。验证每个TF的边界条件：
  - 5m: `m % 5 == 0 and s < 90` — 90秒窗口太短？scan可能错过？
  - 但MOVED在order_manager里，不在trailing里。trailing用`last_update_bar`。验证两套逻辑是否一致。

### 文件: `server/execution/live_adapter.py`

**`submit_live_plan_entry()` 逐行：**

- [ ] `orderType: "market"` — 确认是market不是limit
- [ ] `triggerPrice` — 从 `_normalize_price(trigger_price, contract)` 来。确认precision正确。
- [ ] `stopLossTriggerPrice` — preset SL。确认值 = line price（stop_px）
- [ ] `stopSurplusTriggerPrice` — preset TP。确认值 = tp_px
- [ ] `triggerType: "mark_price"` — 用mark price触发。考虑：mark price和last price的偏差是否会导致问题？
- [ ] `size` — normalized_size。确认Bitget最小单位满足。如果qty太小会被reject吗？

**`update_position_sl_tp()` 逐行：**

- [ ] 只cancel SL相关的order（`pos_loss` + `profit_loss`中的SL），不碰TP
- [ ] 如何识别profit_loss中哪个是SL哪个是TP？检查逻辑是否正确：
  - short: SL trigger > entry → `side=="buy" and tradeSide=="close"`
  - long: SL trigger < entry → `side=="sell" and tradeSide=="close"`
- [ ] cancel后place `pos_loss` — triggerPrice = new_sl。确认precision。
- [ ] 如果new_tp不是None才place TP。确认trailing不传TP（`tp = None`）。

---

## 2. Trailing SL审计（CRITICAL — 这是之前一直有bug的地方）

### 文件: `server/strategy/mar_bb_runner.py`

**`_update_trailing_stops()` 逐行：**

- [ ] `current_sl = params.get("last_sl_set", 0.0)` — 从内存读，不从Bitget读。验证这个值和Bitget实际SL一致。
- [ ] `tf = params.get("tf", "1h")` — 确认每个持仓的TF正确
- [ ] `last_bar = params.get("last_update_bar", 0)` — 新bar检测
- [ ] `if current_sl > 0 and bars_since <= last_bar: continue` — 这个逻辑对吗？
  - bars_since=5, last_bar=5 → skip（同一个bar）
  - bars_since=6, last_bar=5 → 更新（新bar）
  - 但如果服务器重启，last_bar=0, current_sl=0 → 第一次设SL
  - 重启后bars_since可能很大（因为opened_ts是旧的）→ 这对吗？
- [ ] `new_sl = _calc_trendline_trailing_sl(symbol, bars_since)` — 验证投影计算
- [ ] never-widen check: short → `new_sl >= current_sl → continue`。确认方向正确。
- [ ] same-value check: `abs(new_sl - current_sl) / current_sl < 0.0001` — 这个阈值对吗？
- [ ] `params["last_sl_set"] = new_sl` — 更新内存中的SL
- [ ] `params["last_update_bar"] = bars_since` — 更新bar编号

**`_calc_trendline_trailing_sl()` 逐行：**

- [ ] `current_bar = params["entry_bar"] + bars_since_entry` — entry_bar是什么？确认是line projection的起点。
- [ ] `projected_line = params["slope"] * current_bar + params["intercept"]` — 确认slope和intercept的定义与trendline_lab一致。
- [ ] `_round_price()` — 验证每个价格区间的精度和Bitget要求匹配。

**`register_trendline_params()` 逐行：**

- [ ] `entry_bar=ao.bar_count - 1` — 为什么是bar_count-1不是bar_count？
- [ ] `created_ts=ao.created_ts` — 这是plan order创建时间，不是fill时间。差异影响bars_since计算。
- [ ] `tp_price=ao.tp_price` — 确认TP price从active order正确传递
- [ ] `if key in _trendline_params: return` — 不重复注册。但重启后全部重新注册，opened_ts变了。

---

## 3. Fill检测审计

**文件: `server/strategy/mar_bb_runner.py` (~line 1163-1200)**

- [ ] `_load_active()` — 从JSON文件加载。确认文件格式和ActiveLineOrder字段匹配。
- [ ] `held_syms_upper` — 从Bitget positions API来。确认`total`字段正确判断持仓存在。
- [ ] 匹配逻辑: `sym_upper in held_syms_upper and sym_upper not in _trendline_params` — 如果同一个币有多个TF的持仓怎么办？
- [ ] PnL追踪: `history-position` API。确认API存在、字段名正确（`netProfit`, `achievedProfits`, `margin`等）。

---

## 4. 数值精度审计（CRITICAL for real money）

**对每一个数值计算，验证：**

- [ ] buffer_pct的单位一致性：是0.001（0.1%）还是0.0001（0.01%）？追踪从config到order的完整路径。
- [ ] 所有price计算的精度：`_round_price()` 对每个币种够吗？Bitget的checkBDScale是合约级别的。
- [ ] qty计算：risk_usd / stop_distance。如果stop_distance很小（buffer很小），qty可能非常大。有没有上限？
- [ ] 手续费是否在计算中？回测扣了0.10% taker，但实盘sizing没有考虑手续费对equity的影响。

---

## 5. 状态一致性审计

**验证内存状态和Bitget状态的一致性：**

- [ ] `_trendline_params` 内存dict vs Bitget实际positions — 可能drift
- [ ] `trendline_active_orders.json` vs Bitget plan orders — 可能drift
- [ ] `last_sl_set` vs Bitget实际SL trigger price — 可能drift
- [ ] 重启后的恢复逻辑：哪些状态丢失？影响什么？
- [ ] 如果Bitget API暂时失败，状态会不会desync？

---

## 6. 边界条件和异常审计

**测试这些edge case：**

- [ ] 同一个币在多个TF都有信号 → 会下多个plan order吗？互相干扰吗？
- [ ] Plan order触发时scan正好在跑 → 会不会miss fill detection？
- [ ] SL和TP在同一个bar都被触发 → 哪个先？Bitget怎么处理？
- [ ] 线的slope为0（水平线）→ SL永不移动 → 这对吗？
- [ ] 线的slope极大（几乎垂直）→ SL每bar移动很多 → 会不会穿过entry变成反向？
- [ ] equity < 最小下单金额 → 会怎样？
- [ ] API rate limit → 100个币×4TF的scan会不会触发？

---

## 7. 回测 vs 实盘差异审计（CRITICAL）

**逐条对比：**

| 项目 | 回测 (passive_retest_v3.py) | 实盘 (trendline_order_manager.py) | 一致？ |
|------|----------------------------|-----------------------------------|--------|
| Entry方式 | bar.low <= entry_price (passive) | plan order market trigger | ？ |
| SL检查 | 每bar内pessimistic ordering | preset SL on Bitget (实时) | ？ |
| SL移动 | 每bar自动 (line_j = slope*j+intercept) | trailing code at bar boundary | ？ |
| TP检查 | bar.high >= target OR BB | preset TP on Bitget | ？ |
| Sweep检查 | entry+stop same bar | 不检查（Bitget处理） | ？ |
| 手续费 | 0.10% taker flat | Bitget actual (varies) | ？ |
| Per-bar dedup | 3-bar spacing | 无dedup（同一条线一个order） | ？ |
| Buffer值 | 固定per-config | per-TF from tf_buffer_map | ？ |

**对每一个"？"，验证具体数值，给出结论。**

---

## 8. Implementation Tasks

完成review后，按优先级修复所有发现的问题：

### P0: 数据流验证
写一个脚本验证：对一个特定币种（如LINKUSDT 1h），追踪从scan detection → signal → plan order → fill → SL set → SL move的完整数据流。打印每一步的实际数值。和回测做对比。

### P1: SL移动验证
写一个测试：模拟3个连续bar boundary，验证SL确实移动了，且cancel和re-place都成功了。打印Bitget上的实际order变化。

### P2: 精度统一
扫描所有buffer/price/quantity计算路径，确保单位一致（percent vs fraction vs absolute）。

### P3: 状态恢复
验证服务器重启后：(a) 现有持仓的SL/TP正确恢复, (b) 现有plan order继续正常移动, (c) 没有重复order产生。

---

## 输出格式

Review结果必须包含：

```
## Finding [编号]
- 严重程度: P0/P1/P2/P3
- 文件: path:line_number
- 问题: 一句话描述
- 影响: 会导致什么后果
- 修复: 具体代码改动
- 验证: 如何证明修好了
```

**不允许说"看起来没问题"。每一个检查项必须有明确的PASS/FAIL和证据。**
