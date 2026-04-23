# Technical Analysis Basics — Required Reading for AI

> **This file is MANDATORY READING before any task that touches:**
> - Order placement logic (entry/SL/TP computation)
> - Line / drawing interpretation (support, resistance, trendline)
> - Trade direction inference (long, short)
> - Chart overlay labels (持仓, 计划, 止损, 止盈)
>
> **Why this exists**: on 2026-04-23 the AI repeatedly got
> support/resistance confused with the DB `side` label and produced
> wrong entry/SL prices on a real-money ZEC order, then described
> the user's trade with the wrong trade-type name ("breakout long"
> for what was obviously a bounce-long at support). The user's fix:
> write down the fundamentals in a checked-in MD so there's a
> source of truth the AI can't drift away from.

---

## 1. Support and Resistance — definitions

These are PRICE LEVELS (or sloping lines) where past price action
shows buyers or sellers historically stepped in.

### Support (支撑)
- A price level **BELOW current price** where buyers have historically
  absorbed sell pressure and caused price to bounce upward.
- Mental model: the floor. Price falls down, hits support, bounces.
- **Price is ABOVE a support**. The support holds price up.

### Resistance (阻力 / 压制)
- A price level **ABOVE current price** where sellers have historically
  absorbed buy pressure and caused price to reverse downward.
- Mental model: the ceiling. Price rises up, hits resistance, rejects.
- **Price is BELOW a resistance**. The resistance caps price from above.

### Role reversal
Once **broken**, roles flip:
- Broken resistance → becomes support on retest
- Broken support → becomes resistance on retest
This is the single most important S/R concept for re-entries.

### Trendlines (斜线)
Same idea, but the level moves with time:
- **Ascending support (上升支撑)**: connects successive higher lows.
  Price is above it; line goes up over time.
- **Descending resistance (下降阻力)**: connects successive lower highs.
  Price is below it; line goes down over time.
- **Ascending resistance** (rare): upper boundary of an ascending channel.
- **Descending support** (rare): lower boundary of a descending channel.

---

## 2. The four base trade types (this project's terminology)

Each combines a LINE's role with a DIRECTION. Four are valid:

| Line role @ now | Direction | Trade type | Thesis |
|---|---|---|---|
| Support | **Long**  | **Bounce long** (反弹多) | Price dips to support, buyers defend, bounce back up |
| Support | **Short** | **Breakdown short** (破位空) | Price falls through support, cascade down |
| Resistance | **Long**  | **Breakout long** (突破多) | Price breaks above resistance, run up |
| Resistance | **Short** | **Bounce short** (反弹空 / 压回空) | Price rallies to resistance, sellers press, reject down |

The trade TYPE is NOT stored in the DB. The DB stores `side` (the
line's original label at draw-time) and `direction` (what the user
clicked). The COMBINATION tells you the trade type.

**IMPORTANT**: the line's DB `side` may be stale because the line was
LABELED when it was drawn, and price has since moved. At any moment,
a line's EFFECTIVE role is determined by `line_at_now` vs `current_price`:

```
if line_at_now < current_price:
    effective_role = "support"   # line is below price → floor
elif line_at_now > current_price:
    effective_role = "resistance"  # line is above price → ceiling
# else: price is exactly on the line (rare)
```

When describing the user's trade, use the **effective role**, not
the DB label.

---

## 3. Entry / Stop / Target geometry

For every trade:
- **Line** is the pivot.
- **Entry** sits a small buffer AWAY from the line on the side the
  trade is "waiting to see" price come from.
- **SL** sits past the line on the loss side — breaking the line
  invalidates the thesis.
- **TP** is far away in the profit direction.

### LONG (direction = long)
- Entry is **ABOVE** the line (price needs to be above entry at fill;
  for a bounce long, we wait for price to dip back down to entry).
- SL is **BELOW** the line (if price breaks below, the support failed).
- TP is **further ABOVE** entry.

```
TP      ─────── far above entry
entry   ─── line × (1 + buffer%) ──
line    ─── the pivot ─────────────
SL      ─── line × (1 - stop%) ──
        ─────── further below
```

SL distance from entry = **buffer + stop** (both percentages of line).

### SHORT (direction = short)
Mirror of LONG. Entry below line, SL above line, TP further below.

```
        ─────── further above
SL      ─── line × (1 + stop%) ──
line    ─── the pivot ─────────────
entry   ─── line × (1 - buffer%) ──
TP      ─────── far below entry
```

SL distance from entry = buffer + stop.

### Code pointers
- Entry formula: `server/routers/conditionals.py` function `api_place_line_order`
- SL formula: same function, includes a distance-cap using `buffer+stop`.
- Stored fields on `OrderConfig` (`server/conditionals/types.py`):
  - `entry_offset_points` = `line_price × tolerance_pct / 100` — **distance line→entry**
  - `stop_points` = `line_price × stop_offset_pct / 100` — **distance line→SL**
  - Total **entry→SL distance** = `entry_offset_points + stop_points`
  - NEVER confuse `stop_points` with "entry→SL distance". It is
    "line→SL" only.

---

## 4. Line projection — critical for matching user's visual

The user reads the line's value where their eye intersects the
current candle. That visual value must equal what the server uses
for the plan trigger. Otherwise the order places at a different
price than the user intended.

### Correct reference time
The live bar is still forming; its rightmost X-coord is ≈ **now**.
So the server must project the line at **`now`**, not at bar-open.

### DO (since 2026-04-23 fix)
```python
line_now = drawing.line_price_at(int(now_ts))
```

### DON'T
```python
# Snap to bar-open — wrong. On 1d this is up to 24h stale.
bar_open_ts = (int(now_ts) // tf_sec) * tf_sec
line_now = drawing.line_price_at(bar_open_ts)
```

### Scale: LOG, not LINEAR
The chart's Y-axis is log by default. A visually-straight line
between two endpoints corresponds to constant **growth rate**
in price (exponential), not constant **price delta** (linear).
Use log interpolation:

```python
return math.exp(
    math.log(p_start) + ratio * (math.log(p_end) - math.log(p_start))
)
```

This was fixed in commit `9cb64ec` and is PERMANENT. Never switch
back to linear, even if it "looks simpler".

---

## 5. Checklist — AI must complete before recommending or writing
   code in the trade-logic area

Before describing a user's trade to them:
- [ ] I fetched the actual line endpoints from the DB.
- [ ] I projected `line_at_now` using LOG + `ts=now`.
- [ ] I compared `line_at_now` to `current_mark_price` to determine
      the line's **effective role** (support vs resistance).
- [ ] I described the trade using the effective role, NOT the DB
      `side` label, when naming the trade type.

Before writing code that computes entry / SL / TP:
- [ ] I re-read Section 3 above to confirm the direction semantics.
- [ ] My SL formula sums `entry_offset + stop` for total distance
      from entry, never just `stop_points` alone.
- [ ] For any time-based projection, I use `now`, not `bar_open`.

Before claiming a fix is done:
- [ ] I placed (or dry-ran) a REAL order on a sample line and
      verified the resulting `fill_price` and `stop_price` match
      the user's visual + intent within a few basis points.
- [ ] I checked the chart overlay labels render at the same prices
      (and that SL label equals Bitget's real SL, not a stale or
      miscomputed value).

---

## 6. Glossary of user's Chinese terms (so AI doesn't mistranslate)

| User's Chinese | Meaning |
|---|---|
| 支撑 / support | support line |
| 阻力 / 压制 / resistance | resistance line |
| 反弹 / bounce | price touches level and reverses |
| 突破 / breakout | price crosses through a level |
| 破位 / breakdown | price drops through support |
| 买上 / 买入 | buy (go long) |
| 做多 | go long |
| 做空 | go short |
| 挂单 / 计划单 | plan/pending order (triggered when price reaches level) |
| 持仓 / 仓位 | open position (already filled, actively held) |
| 入场 / entry | entry price |
| 止损 / SL | stop loss |
| 止盈 / TP / target | take profit |
| intercept (user's word) | where the trendline crosses the current vertical = line_at_now |
| buffer | the % offset from line to entry (this project's `tolerance_pct`) |
| stop (the % amount) | the % offset from line to SL (this project's `stop_offset_pct`) |

The user often says "buffer 0.4" meaning 0.4% buffer. That's 0.4,
not 0.04. DO NOT type a `0.04` default into any modal pre-fill
without checking — this has burned the user.
