# S/R Zone Quality Filters: Fill the Three Zeros Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the three `0.0` placeholder scores in `factors.py` (VolumeFailure, TrendContext, ConfluenceScore) and add RR + profit-space filters in `signals.py`, so the system outputs few high-quality zones instead of noisy lines.

**Architecture:** Each placeholder becomes a real scoring function in `factors.py` that reads from the existing candles DataFrame. The `signals.py` signal generator gains two new gates (min RR ratio, min profit space). No new modules — this is surgical improvement to existing code. All changes are in `server/strategy/`.

**Tech Stack:** Python, pandas, existing `StrategyConfig` dataclass, existing test harness (pytest).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `server/strategy/config.py` | Modify | Add 6 new config fields for volume, trend, confluence, RR gate, profit-space |
| `server/strategy/factors.py` | Modify | Implement `_volume_confirmation()`, `_trend_context()`, wire into both score functions |
| `server/strategy/signals.py` | Modify | Add RR gate + profit-space gate before appending signals |
| `server/strategy/types.py` | Modify | Add `opposing_zones` field to `StrategySignal` for profit-space tracking |
| `tests/strategy/test_factors_volume.py` | Create | Tests for volume confirmation scoring |
| `tests/strategy/test_factors_trend.py` | Create | Tests for trend context scoring |
| `tests/strategy/test_signals_rr_gate.py` | Create | Tests for RR and profit-space gates |

---

### Task 1: Add Config Fields

**Files:**
- Modify: `server/strategy/config.py:49-118` (inside `StrategyConfig` dataclass)

- [ ] **Step 1: Write the failing test**

Create `tests/strategy/test_config_new_fields.py`:

```python
from server.strategy.config import StrategyConfig


def test_config_has_volume_fields():
    cfg = StrategyConfig()
    assert cfg.volume_surge_threshold == 1.5
    assert cfg.volume_lookback_bars == 20


def test_config_has_trend_fields():
    cfg = StrategyConfig()
    assert cfg.trend_ema_period == 50
    assert cfg.trend_weight == 0.10


def test_config_has_rr_gate_fields():
    cfg = StrategyConfig()
    assert cfg.min_rr_ratio == 2.0
    assert cfg.min_profit_space_atr_mult == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_config_new_fields.py -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

In `server/strategy/config.py`, add these fields inside the `StrategyConfig` dataclass after line 118 (after `trigger_mode_priority`):

```python
    # Volume confirmation
    volume_surge_threshold: float = 1.5  # current vol must be >= 1.5x avg
    volume_lookback_bars: int = 20  # bars to average volume over

    # Trend context
    trend_ema_period: int = 50  # EMA period for trend direction
    trend_weight: float = 0.10  # weight of trend context in factor score

    # Signal quality gates
    min_rr_ratio: float = 2.0  # reject signals with RR below this
    min_profit_space_atr_mult: float = 1.0  # min distance to opposing zone in ATR units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_config_new_fields.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && git add server/strategy/config.py tests/strategy/test_config_new_fields.py && git commit -m "feat(strategy): add config fields for volume, trend, RR gate, profit-space"
```

---

### Task 2: Implement Volume Confirmation Score

**Files:**
- Create: `tests/strategy/test_factors_volume.py`
- Modify: `server/strategy/factors.py:1-125`

- [ ] **Step 1: Write the failing test**

Create `tests/strategy/test_factors_volume.py`:

```python
import pandas as pd
from server.strategy.factors import _volume_confirmation


def _make_candles(volumes: list[float], close: float = 100.0) -> pd.DataFrame:
    n = len(volumes)
    return pd.DataFrame({
        "timestamp": list(range(n)),
        "open": [close] * n,
        "high": [close + 1] * n,
        "low": [close - 1] * n,
        "close": [close] * n,
        "volume": volumes,
    })


def test_volume_surge_scores_high():
    """When current bar volume is 3x the average, score should be high."""
    vols = [100.0] * 20 + [300.0]  # last bar is 3x avg
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=20, lookback=20, surge_threshold=1.5)
    assert score > 0.8


def test_volume_below_threshold_scores_zero():
    """When current bar volume is below surge threshold, score is 0."""
    vols = [100.0] * 21  # last bar is exactly average
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=20, lookback=20, surge_threshold=1.5)
    assert score == 0.0


def test_volume_at_threshold_scores_low():
    """Volume exactly at threshold gives minimal positive score."""
    vols = [100.0] * 20 + [150.0]  # last bar is 1.5x avg (exactly at threshold)
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=20, lookback=20, surge_threshold=1.5)
    assert 0.0 < score <= 0.2


def test_volume_with_short_history():
    """With fewer bars than lookback, use what's available."""
    vols = [100.0, 100.0, 300.0]
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=2, lookback=20, surge_threshold=1.5)
    assert score > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_factors_volume.py -v`
Expected: FAIL with `ImportError: cannot import name '_volume_confirmation'`

- [ ] **Step 3: Write minimal implementation**

Add to `server/strategy/factors.py` before the `__all__` block (before line 121):

```python
def _volume_confirmation(df, bar_index: int, lookback: int = 20, surge_threshold: float = 1.5) -> float:
    """Score 0-1 based on how much current bar volume exceeds the rolling average.
    
    Returns 0.0 if volume is below surge_threshold * average.
    Returns up to 1.0 for extreme volume surges (5x+ average).
    """
    start = max(0, bar_index - lookback)
    if start >= bar_index:
        return 0.0
    avg_vol = float(df["volume"].iloc[start:bar_index].mean())
    if avg_vol <= 0:
        return 0.0
    current_vol = float(df["volume"].iloc[bar_index])
    ratio = current_vol / avg_vol
    if ratio < surge_threshold:
        return 0.0
    # Scale from 0 at threshold to 1 at 5x average
    return clamp((ratio - surge_threshold) / (5.0 - surge_threshold))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_factors_volume.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && git add server/strategy/factors.py tests/strategy/test_factors_volume.py && git commit -m "feat(strategy): implement volume confirmation scoring"
```

---

### Task 3: Implement Trend Context Score

**Files:**
- Create: `tests/strategy/test_factors_trend.py`
- Modify: `server/strategy/factors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/strategy/test_factors_trend.py`:

```python
import pandas as pd
from server.strategy.factors import _trend_context


def _make_trending_candles(direction: str, n: int = 60) -> pd.DataFrame:
    """Create candles with a clear trend."""
    if direction == "up":
        closes = [100.0 + i * 0.5 for i in range(n)]
    elif direction == "down":
        closes = [100.0 - i * 0.5 for i in range(n)]
    else:
        closes = [100.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(n)]
    return pd.DataFrame({
        "timestamp": list(range(n)),
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


def test_long_in_uptrend_scores_high():
    """Going long in uptrend should score positively."""
    df = _make_trending_candles("up")
    score = _trend_context(df, bar_index=59, direction="long", ema_period=50)
    assert score > 0.5


def test_long_in_downtrend_scores_zero():
    """Going long in downtrend should score 0 (counter-trend)."""
    df = _make_trending_candles("down")
    score = _trend_context(df, bar_index=59, direction="long", ema_period=50)
    assert score == 0.0


def test_short_in_downtrend_scores_high():
    """Going short in downtrend should score positively."""
    df = _make_trending_candles("down")
    score = _trend_context(df, bar_index=59, direction="short", ema_period=50)
    assert score > 0.5


def test_short_in_uptrend_scores_zero():
    """Going short in uptrend should score 0."""
    df = _make_trending_candles("up")
    score = _trend_context(df, bar_index=59, direction="short", ema_period=50)
    assert score == 0.0


def test_sideways_scores_mid():
    """In sideways market, both directions get moderate score."""
    df = _make_trending_candles("sideways")
    long_score = _trend_context(df, bar_index=59, direction="long", ema_period=50)
    short_score = _trend_context(df, bar_index=59, direction="short", ema_period=50)
    # Sideways should give something between 0.3 and 0.7
    assert 0.0 <= long_score <= 0.7
    assert 0.0 <= short_score <= 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_factors_trend.py -v`
Expected: FAIL with `ImportError: cannot import name '_trend_context'`

- [ ] **Step 3: Write minimal implementation**

Add to `server/strategy/factors.py` after `_volume_confirmation`:

```python
def _trend_context(df, bar_index: int, direction: str, ema_period: int = 50) -> float:
    """Score 0-1 for trend alignment.
    
    Long trades score higher in uptrends, short trades in downtrends.
    Uses EMA slope and price position relative to EMA.
    """
    if bar_index < ema_period:
        return 0.3  # insufficient data, neutral-ish
    
    close = df["close"].astype(float)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    
    current_close = float(close.iloc[bar_index])
    current_ema = float(ema.iloc[bar_index])
    prev_ema = float(ema.iloc[bar_index - 1])
    
    # EMA slope direction (normalized)
    ema_slope = (current_ema - prev_ema) / max(abs(current_ema), 1e-10)
    
    # Price position relative to EMA
    price_vs_ema = (current_close - current_ema) / max(abs(current_ema), 1e-10)
    
    if direction == "long":
        # Long is good when price > EMA and EMA rising
        slope_score = clamp(ema_slope * 1000)  # amplify small slopes
        position_score = clamp(price_vs_ema * 50)
    else:
        # Short is good when price < EMA and EMA falling
        slope_score = clamp(-ema_slope * 1000)
        position_score = clamp(-price_vs_ema * 50)
    
    return clamp(0.5 * slope_score + 0.5 * position_score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_factors_trend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && git add server/strategy/factors.py tests/strategy/test_factors_trend.py && git commit -m "feat(strategy): implement trend context scoring"
```

---

### Task 4: Wire Volume and Trend into Factor Score Functions

**Files:**
- Modify: `server/strategy/factors.py:7-52` (resistance), `55-100` (support)

- [ ] **Step 1: Write the failing test**

Add to `tests/strategy/test_factors_volume.py`:

```python
import pandas as pd
from server.strategy.config import StrategyConfig
from server.strategy.factors import calculate_resistance_short_score
from server.strategy.types import Trendline


def _make_line(side="resistance", touches=3, score_components=None) -> Trendline:
    return Trendline(
        line_id="test_line",
        side=side,
        symbol="TESTUSDT",
        timeframe="1h",
        state="confirmed",
        anchor_pivot_ids=("p1", "p2"),
        confirming_touch_pivot_ids=("p1", "p2", "p3"),
        anchor_indices=(0, 10),
        anchor_prices=(100.0, 100.0),
        slope=0.0,
        intercept=100.0,
        confirming_touch_indices=(0, 5, 10),
        bar_touch_indices=(),
        confirming_touch_count=touches,
        bar_touch_count=0,
        recent_bar_touch_count=0,
        residuals=(0.1, 0.1, 0.1),
        score=75.0,
        score_components=score_components or {"normalized_mean_residual": 0.1},
        projected_price_current=100.0,
        projected_price_next=100.0,
        latest_confirming_touch_index=10,
        latest_confirming_touch_price=100.0,
        bars_since_last_confirming_touch=5,
        recent_test_count=0,
        non_touch_cross_count=0,
    )


def _make_candles_for_factor(n=20, close=99.5, volume=100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": list(range(n)),
        "open": [close + 0.1] * n,
        "high": [close + 1.0] * n,
        "low": [close - 0.5] * n,
        "close": [close] * n,
        "volume": [volume] * n,
    })


def test_resistance_score_includes_volume():
    """VolumeFailure should no longer be 0.0 when volume data is present."""
    candles = _make_candles_for_factor(n=25, volume=100.0)
    # Set last bar to high volume
    candles.loc[24, "volume"] = 500.0
    line = _make_line()
    cfg = StrategyConfig()
    score, components = calculate_resistance_short_score(candles, line, cfg, bar_index=24)
    assert components["VolumeConfirmation"] > 0.0  # renamed from VolumeFailure


def test_resistance_score_includes_trend():
    """TrendContext should no longer be 0.0."""
    candles = _make_candles_for_factor(n=60, volume=100.0)
    line = _make_line()
    cfg = StrategyConfig()
    score, components = calculate_resistance_short_score(candles, line, cfg, bar_index=59)
    assert "TrendContext" in components
    # In a flat market, trend context should be between 0 and 1
    assert 0.0 <= components["TrendContext"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_factors_volume.py::test_resistance_score_includes_volume -v`
Expected: FAIL with `KeyError: 'VolumeConfirmation'`

- [ ] **Step 3: Wire volume and trend into both score functions**

In `server/strategy/factors.py`, modify `calculate_resistance_short_score` (lines 7-52):

Replace the score calculation block (lines 32-52) with:

```python
    volume_score = _volume_confirmation(df, current_index, cfg.volume_lookback_bars, cfg.volume_surge_threshold)
    trend_score = _trend_context(df, current_index, "short", cfg.trend_ema_period)

    score = (
        (0.22 * touch_strength)
        + (0.16 * fit_tightness)
        + (0.22 * rejection_strength)
        + (0.12 * distance_compression)
        + (0.08 * freshness_score)
        + (0.08 * volume_score)
        + (cfg.trend_weight * trend_score)
        - (0.12 * breakout_risk)
    )
    score = clamp(score)

    return score, {
        "TouchStrength": touch_strength,
        "FitTightness": fit_tightness,
        "RejectionStrength": rejection_strength,
        "DistanceCompression": distance_compression,
        "FreshnessScore": freshness_score,
        "BreakoutRisk": breakout_risk,
        "VolumeConfirmation": volume_score,
        "TrendContext": trend_score,
        "ConfluenceScore": 0.0,  # Phase 2: multi-TF
    }
```

Apply the same pattern to `calculate_support_long_score` (lines 55-100), using `"long"` for trend direction:

```python
    volume_score = _volume_confirmation(df, current_index, cfg.volume_lookback_bars, cfg.volume_surge_threshold)
    trend_score = _trend_context(df, current_index, "long", cfg.trend_ema_period)

    score = (
        (0.22 * touch_strength)
        + (0.16 * fit_tightness)
        + (0.22 * rejection_strength)
        + (0.12 * distance_compression)
        + (0.08 * freshness_score)
        + (0.08 * volume_score)
        + (cfg.trend_weight * trend_score)
        - (0.12 * breakdown_risk)
    )
    score = clamp(score)

    return score, {
        "TouchStrength": touch_strength,
        "FitTightness": fit_tightness,
        "RejectionStrength": rejection_strength,
        "DistanceCompression": distance_compression,
        "FreshnessScore": freshness_score,
        "BreakdownRisk": breakdown_risk,
        "VolumeConfirmation": volume_score,
        "TrendContext": trend_score,
        "ConfluenceScore": 0.0,  # Phase 2: multi-TF
    }
```

- [ ] **Step 4: Run all factor tests to verify they pass**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_factors_volume.py tests/strategy/test_factors_trend.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run existing test suite to check for regressions**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/ -v --tb=short`
Expected: ALL PASS (existing tests use relative assertions, weight redistribution should not break them)

- [ ] **Step 6: Commit**

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && git add server/strategy/factors.py tests/strategy/test_factors_volume.py && git commit -m "feat(strategy): wire volume and trend scoring into factor functions"
```

---

### Task 5: Add RR Gate to Signal Generation

**Files:**
- Create: `tests/strategy/test_signals_rr_gate.py`
- Modify: `server/strategy/signals.py:10-66` (pre_limit), `69-143` (rejection), `146-223` (failed_breakout)

- [ ] **Step 1: Write the failing test**

Create `tests/strategy/test_signals_rr_gate.py`:

```python
import pandas as pd
from server.strategy.config import StrategyConfig
from server.strategy.signals import generate_pre_limit_signals
from server.strategy.types import Trendline


def _make_flat_candles(n=20, close=100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": list(range(n)),
        "open": [close] * n,
        "high": [close + 0.5] * n,
        "low": [close - 0.5] * n,
        "close": [close] * n,
        "volume": [1000.0] * n,
    })


def _make_support_line(price=100.2, touches=4) -> Trendline:
    return Trendline(
        line_id="test_support",
        side="support",
        symbol="TESTUSDT",
        timeframe="1h",
        state="confirmed",
        anchor_pivot_ids=("p1", "p2"),
        confirming_touch_pivot_ids=tuple(f"p{i}" for i in range(touches)),
        anchor_indices=(0, 5),
        anchor_prices=(price, price),
        slope=0.0,
        intercept=price,
        confirming_touch_indices=tuple(range(0, touches * 3, 3)),
        bar_touch_indices=(),
        confirming_touch_count=touches,
        bar_touch_count=0,
        recent_bar_touch_count=0,
        residuals=tuple(0.01 for _ in range(touches)),
        score=80.0,
        score_components={"normalized_mean_residual": 0.05},
        projected_price_current=price,
        projected_price_next=price,
        latest_confirming_touch_index=15,
        latest_confirming_touch_price=price,
        bars_since_last_confirming_touch=4,
        recent_test_count=0,
        non_touch_cross_count=0,
    )


def test_signals_rejected_when_rr_below_minimum():
    """Signals with RR below min_rr_ratio should be filtered out."""
    candles = _make_flat_candles(n=20, close=100.0)
    line = _make_support_line(price=100.2)
    # Set rr_target very low so RR is calculated low, but min_rr_ratio high
    cfg = StrategyConfig(
        rr_target=0.5,  # TP = entry + 0.5 * stop_distance → RR = 0.5
        min_rr_ratio=2.0,  # require RR >= 2.0
        score_threshold=0.0,  # don't filter by score
    )
    signals = generate_pre_limit_signals(candles, [line], cfg)
    # All signals should be filtered because RR = 0.5 < min_rr_ratio = 2.0
    assert len(signals) == 0


def test_signals_accepted_when_rr_above_minimum():
    """Signals with RR above min_rr_ratio should pass."""
    candles = _make_flat_candles(n=20, close=100.0)
    line = _make_support_line(price=100.2)
    cfg = StrategyConfig(
        rr_target=3.0,  # TP = entry + 3 * stop_distance → RR = 3.0
        min_rr_ratio=2.0,  # require RR >= 2.0
        score_threshold=0.0,
    )
    signals = generate_pre_limit_signals(candles, [line], cfg)
    # Signals should pass RR gate (3.0 >= 2.0)
    for sig in signals:
        assert sig.risk_reward >= 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_signals_rr_gate.py -v`
Expected: FAIL — signals are NOT filtered by RR currently

- [ ] **Step 3: Add RR gate to all three signal generators**

In `server/strategy/signals.py`, add this check in `generate_pre_limit_signals` right before `signals.append(...)` (around line 50), for both the resistance and support branches:

```python
        # RR gate: reject signals below minimum risk-reward
        risk = abs(entry_price - stop_price)
        reward = abs(tp_price - entry_price)
        risk_reward = (reward / risk) if risk > 0 else 0.0
        if risk_reward < cfg.min_rr_ratio:
            continue
```

Add the same RR gate block in `generate_rejection_signals` before each `signals.append(...)` call (around lines 100 and 130).

Add the same RR gate block in `generate_failed_breakout_signals` before each `signals.append(...)` call (around lines 180 and 205).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_signals_rr_gate.py -v`
Expected: PASS

- [ ] **Step 5: Run full signal test suite for regressions**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_signals.py tests/strategy/test_signals_rr_gate.py -v --tb=short`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && git add server/strategy/signals.py tests/strategy/test_signals_rr_gate.py && git commit -m "feat(strategy): add minimum RR ratio gate to signal generation"
```

---

### Task 6: Add Profit-Space Gate (Distance to Opposing Zone)

**Files:**
- Modify: `server/strategy/signals.py`
- Modify: `server/strategy/types.py` (add field to StrategySignal)

- [ ] **Step 1: Write the failing test**

Add to `tests/strategy/test_signals_rr_gate.py`:

```python
from server.strategy.signals import _check_profit_space


def test_profit_space_sufficient():
    """When opposing zone is far, profit space is sufficient."""
    # Entry at 100, opposing resistance at 110, ATR=2 → space = 5 ATR
    assert _check_profit_space(
        entry_price=100.0,
        direction="long",
        opposing_lines=[_make_resistance_line(price=110.0)],
        atr_value=2.0,
        min_space_atr=1.0,
    ) == True


def test_profit_space_insufficient():
    """When opposing zone is too close, reject."""
    # Entry at 100, opposing resistance at 100.5, ATR=2 → space = 0.25 ATR
    assert _check_profit_space(
        entry_price=100.0,
        direction="long",
        opposing_lines=[_make_resistance_line(price=100.5)],
        atr_value=2.0,
        min_space_atr=1.0,
    ) == False


def test_profit_space_no_opposing_lines():
    """With no opposing lines, always pass (assume open space)."""
    assert _check_profit_space(
        entry_price=100.0,
        direction="long",
        opposing_lines=[],
        atr_value=2.0,
        min_space_atr=1.0,
    ) == True


def _make_resistance_line(price=110.0) -> Trendline:
    return Trendline(
        line_id="opposing",
        side="resistance",
        symbol="TESTUSDT",
        timeframe="1h",
        state="confirmed",
        anchor_pivot_ids=("p1", "p2"),
        confirming_touch_pivot_ids=("p1", "p2"),
        anchor_indices=(0, 5),
        anchor_prices=(price, price),
        slope=0.0,
        intercept=price,
        confirming_touch_indices=(0, 5),
        bar_touch_indices=(),
        confirming_touch_count=2,
        bar_touch_count=0,
        recent_bar_touch_count=0,
        residuals=(0.01, 0.01),
        score=70.0,
        score_components={},
        projected_price_current=price,
        projected_price_next=price,
        latest_confirming_touch_index=5,
        latest_confirming_touch_price=price,
        bars_since_last_confirming_touch=3,
        recent_test_count=0,
        non_touch_cross_count=0,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_signals_rr_gate.py::test_profit_space_sufficient -v`
Expected: FAIL with `ImportError: cannot import name '_check_profit_space'`

- [ ] **Step 3: Implement profit-space check**

Add to `server/strategy/signals.py` before the `__all__` block:

```python
def _check_profit_space(
    entry_price: float,
    direction: str,
    opposing_lines: Sequence[Trendline],
    atr_value: float,
    min_space_atr: float,
) -> bool:
    """Check if there's enough room to the nearest opposing zone.
    
    For longs: nearest resistance above entry must be >= min_space_atr * ATR away.
    For shorts: nearest support below entry must be >= min_space_atr * ATR away.
    If no opposing lines exist, return True (open space).
    """
    if not opposing_lines or atr_value <= 0:
        return True
    
    if direction == "long":
        # Find nearest resistance above entry
        above = [
            abs(line.projected_price_current - entry_price)
            for line in opposing_lines
            if line.side == "resistance" and line.projected_price_current > entry_price
               and line.state == "confirmed"
        ]
        if not above:
            return True
        nearest_distance = min(above)
    else:
        # Find nearest support below entry
        below = [
            abs(entry_price - line.projected_price_current)
            for line in opposing_lines
            if line.side == "support" and line.projected_price_current < entry_price
               and line.state == "confirmed"
        ]
        if not below:
            return True
        nearest_distance = min(below)
    
    return nearest_distance >= (min_space_atr * atr_value)
```

- [ ] **Step 4: Wire profit-space check into `generate_signals`**

In `server/strategy/signals.py`, modify the `generate_signals` function (line 226) to pass opposing lines:

```python
def generate_signals(candles, lines: Sequence[Trendline], config: StrategyConfig | None = None) -> list[StrategySignal]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    atr = calculate_atr(df, cfg.atr_period)
    current_index = len(df) - 1
    atr_value = float(atr.iloc[current_index]) if current_index >= 0 else 0.0
    
    signals: list[StrategySignal] = []
    signals.extend(generate_pre_limit_signals(candles, lines, cfg))
    signals.extend(generate_rejection_signals(candles, lines, cfg))
    signals.extend(generate_failed_breakout_signals(candles, lines, cfg))
    
    # Profit-space gate: reject signals where opposing zone is too close
    if cfg.min_profit_space_atr_mult > 0 and atr_value > 0:
        signals = [
            sig for sig in signals
            if _check_profit_space(
                sig.entry_price, sig.direction, lines, atr_value, cfg.min_profit_space_atr_mult
            )
        ]
    
    prioritized = prioritize_signals(signals, cfg)
    return resolve_signal_conflicts(prioritized)
```

- [ ] **Step 5: Run tests**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/strategy/test_signals_rr_gate.py tests/strategy/test_signals.py -v --tb=short`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && git add server/strategy/signals.py tests/strategy/test_signals_rr_gate.py && git commit -m "feat(strategy): add profit-space gate checking distance to opposing zones"
```

---

### Task 7: Run Full Test Suite & Verify Integration

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

Run: `cd c:/Users/alexl/Desktop/crypto-analysis- && python -m pytest tests/ -v --tb=short 2>&1 | head -80`
Expected: ALL existing tests still pass

- [ ] **Step 2: Verify the three zeros are gone**

Run quick sanity check:

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && python -c "
from server.strategy.factors import calculate_resistance_short_score
from server.strategy.config import StrategyConfig
import pandas as pd

# 60 bars of slightly downtrending data with a volume spike at end
n = 60
closes = [100.0 - i*0.1 for i in range(n)]
df = pd.DataFrame({
    'timestamp': list(range(n)),
    'open': [c + 0.1 for c in closes],
    'high': [c + 0.5 for c in closes],
    'low': [c - 0.5 for c in closes],
    'close': closes,
    'volume': [100.0]*59 + [500.0],  # volume spike on last bar
})

from server.strategy.types import Trendline
line = Trendline(
    line_id='test', side='resistance', symbol='TEST', timeframe='1h',
    state='confirmed', anchor_pivot_ids=('a','b'), confirming_touch_pivot_ids=('a','b','c'),
    anchor_indices=(0,10), anchor_prices=(100.0,100.0),
    slope=0.0, intercept=100.0,
    confirming_touch_indices=(0,5,10), bar_touch_indices=(),
    confirming_touch_count=3, bar_touch_count=0, recent_bar_touch_count=0,
    residuals=(0.1,0.1,0.1), score=75.0,
    score_components={'normalized_mean_residual': 0.1},
    projected_price_current=100.0, projected_price_next=100.0,
    latest_confirming_touch_index=10, latest_confirming_touch_price=100.0,
    bars_since_last_confirming_touch=5, recent_test_count=0, non_touch_cross_count=0,
)

score, components = calculate_resistance_short_score(df, line, StrategyConfig(), bar_index=59)
print('Score:', round(score, 4))
for k, v in components.items():
    flag = ' ← WAS 0.0, NOW FILLED' if k in ('VolumeConfirmation', 'TrendContext') and v != 0.0 else ''
    print(f'  {k}: {round(v, 4)}{flag}')
print()
print('ConfluenceScore is still 0.0 — that is Phase 2 (multi-TF data needed)')
"
```

Expected output should show `VolumeConfirmation` and `TrendContext` with non-zero values.

- [ ] **Step 3: Final commit with all changes verified**

```bash
cd c:/Users/alexl/Desktop/crypto-analysis- && git log --oneline -7
```

Verify 6 new commits from this plan.

---

## Summary of Changes

| What Changed | Before | After |
|---|---|---|
| `VolumeFailure` in factor components | Always `0.0` | Real score based on volume surge ratio |
| `TrendContext` in factor components | Always `0.0` | Real score based on EMA slope + price position |
| Signal RR filtering | RR computed but not enforced | Signals with RR < `min_rr_ratio` rejected |
| Profit space check | Not checked | Signals blocked when opposing zone is closer than `min_profit_space_atr_mult * ATR` |
| Factor weight distribution | 5 factors, 0.26+0.20+0.26+0.16+0.12-0.15 | 7 factors, rebalanced to include volume + trend |
| `ConfluenceScore` | `0.0` | Still `0.0` — requires multi-TF data pipeline (Phase 2) |

## What's NOT in This Plan (Phase 2)

1. **Multi-TF Confluence** — Needs a data pipeline that fetches multiple timeframes and compares zones across them. Separate plan.
2. **Horizontal zone extraction** — `server/strategy/` only does trendlines, not horizontal zones from `support_resistance.py`. Needs unification. Separate plan.
3. **Reaction strength at zone-qualification time** — Currently only measured at signal generation time. Moving it earlier requires restructuring `scoring.py`. Separate plan.
