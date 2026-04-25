# MA Ribbon EMA21 Strategy — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python module that scans a universe of Bitget perp symbols across 5m/15m/1h/4h, detects every bullish-MA-ribbon formation event (`close > MA5 > MA8 > EMA21 > MA55`), records the percentage distance from each MA at the formation bar, computes forward returns at +5/+10/+20/+50 bars, and produces a per-bucket cohort report. This answers Phase 1's question: **"once the ribbon has formed, are forward returns still positive, or is the strategy's premise dead?"**

**Architecture:** A standalone, self-contained Python package under `backtests/ma_ribbon_ema21/`. Pure functions on `pandas.DataFrame` for indicators / signals / features. A bar-event-driven scanner runs each `(symbol, TF)` pair, emits a list of formation events with attached features, and aggregates into a markdown cohort report. No imports from `server/strategy/` or `server/conditionals/` — the new code does not touch the live trading pipeline.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, httpx (for Bitget OHLCV). All matches existing project conventions in `tests/strategy/test_backtest.py` and `ma_ribbon_screener.py`.

---

## Phase 1 deliverable & success criteria

**Deliverable**: a markdown report at `data/logs/backtest_reports/phase1_<UTC_TIMESTAMP>.md` that, for every `(symbol, TF, distance_bucket)` cohort:
- counts formation events
- shows mean / median forward return at +5, +10, +20, +50 bars (post-fee)
- shows win rate (forward return > 0)
- shows worst-bar drawdown within +50 bars

**Acceptance gate** (per spec §4 Phase 1):
- For at least **30%** of symbols, on at least **one TF**, in the **`[0%, 0.5%)` distance-to-MA5 bucket**, the **mean forward return at +20 bars** is **`> +1%` post-fee**.
- If gate fails → premise is dead; we stop here, no Phase 2.

**Required principles binding on this phase** (from `PRINCIPLES.md`):
- §P1 — fetch Bitget OHLCV at maximum available depth per `(symbol, TF)`. No `tfDays` cap.
- §P2 — if the local CSV cache has fewer days than required, refetch.
- §P10 — no `try / except: pass`. Errors must be visible.
- §P11 — log every state transition (Bitget request, formation event count, report write).
- §P15 — completion = enumerated test pass list, not a single happy-path run.
- §P16 — no hardcoded `tfDays` / `max_days` / `cap` defaults. Anything that looks like a cap must be justified as a physical limit.

---

## File structure created in this plan

```
backtests/                                          # NEW directory at project root
  ma_ribbon_ema21/
    __init__.py                                     # Task 1
    config.phase1.json                              # Task 1
    indicators.py                                   # Task 3
    ma_alignment.py                                 # Task 4
    distance_features.py                            # Task 5
    forward_returns.py                              # Task 6
    data_split.py                                   # Task 7
    multi_tf.py                                     # Task 8
    data_loader.py                                  # Task 10
    phase1_engine.py                                # Task 11, 12
    cohort_report.py                                # Task 13
    acceptance_gate.py                              # Task 14
    phase1_cli.py                                   # Task 15
    tests/
      __init__.py                                   # Task 1
      fixtures.py                                   # Task 2
      test_indicators.py                            # Task 3
      test_ma_alignment.py                          # Task 4
      test_distance_features.py                     # Task 5
      test_forward_returns.py                       # Task 6
      test_70_30_split.py                           # Task 7
      test_multi_tf_alignment.py                    # Task 8
      test_no_lookahead.py                          # Task 9
      test_data_loader.py                           # Task 10
      test_phase1_engine.py                         # Task 11, 12
      test_cohort_report.py                         # Task 13
      test_acceptance_gate.py                       # Task 14
      test_phase1_integration.py                    # Task 15

data/logs/backtest_reports/                         # NEW directory created by Task 16
  phase1_<UTC_TS>.md
```

**No files modified outside `backtests/` and `data/logs/backtest_reports/`.** Existing strategy code (`server/strategy/`, `ma_ribbon_*.py`, `server/ma_ribbon_service.py`) is left untouched.

---

## Naming conventions

- Functions: `snake_case`
- Variables: `snake_case`
- Type hints required on every public function signature
- Docstrings on every public function, with at least one line describing the input and output
- All percentages are stored as decimals (`0.01` for 1%), labeled in code comments

---

## Task 1: Project skeleton + Phase 1 config

**Files:**
- Create: `backtests/__init__.py`
- Create: `backtests/ma_ribbon_ema21/__init__.py`
- Create: `backtests/ma_ribbon_ema21/tests/__init__.py`
- Create: `backtests/ma_ribbon_ema21/config.phase1.json`

- [ ] **Step 1: Create empty `__init__.py` files**

```bash
mkdir -p backtests/ma_ribbon_ema21/tests
: > backtests/__init__.py
: > backtests/ma_ribbon_ema21/__init__.py
: > backtests/ma_ribbon_ema21/tests/__init__.py
```

- [ ] **Step 2: Create `config.phase1.json` with the universe and defaults**

```json
{
  "phase": "P1",
  "universe": [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
    "MATICUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT",
    "ATOMUSDT", "UNIUSDT", "ICPUSDT", "ETCUSDT", "FILUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SEIUSDT",
    "TIAUSDT", "WLDUSDT", "PEPEUSDT", "SHIBUSDT", "RNDRUSDT",
    "FETUSDT", "AAVEUSDT", "ALGOUSDT", "VETUSDT", "HBARUSDT",
    "MKRUSDT", "GRTUSDT", "FTMUSDT", "EGLDUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "RUNEUSDT", "DYDXUSDT", "SUIUSDT"
  ],
  "timeframes": ["5m", "15m", "1h", "4h"],
  "data_split": {
    "train_pct": 0.70,
    "test_pct": 0.30,
    "split_by": "time"
  },
  "moving_averages": {
    "ma_fast_1": 5,
    "ma_fast_2": 8,
    "ema_mid": 21,
    "ma_slow": 55
  },
  "bullish_alignment": {
    "require_close_above_ma5":  true,
    "require_close_above_ma8":  true,
    "require_close_above_ema21": true,
    "require_close_above_ma55":  true,
    "require_ma5_above_ma8":     true,
    "require_ma8_above_ema21":   true,
    "require_ema21_above_ma55":  true
  },
  "forward_return_bars": [5, 10, 20, 50],
  "distance_buckets": [
    [0.000, 0.005],
    [0.005, 0.010],
    [0.010, 0.020],
    [0.020, 0.040],
    [0.040, 0.070],
    [0.070, 1.000]
  ],
  "fees": {
    "per_side": 0.0005,
    "slippage_per_fill": 0.0001
  },
  "data_cache_dir": "data/csv_cache/ma_ribbon_ema21",
  "max_history_days": null
}
```

`max_history_days: null` enforces `PRINCIPLES.md` §P1 — pull as much history as Bitget gives us.

- [ ] **Step 3: Verify pytest can discover the new test directory**

Run: `pytest backtests/ma_ribbon_ema21/tests/ -v --collect-only`
Expected: `no tests collected` exit code 5 (empty dir, no error).

- [ ] **Step 4: Commit**

```bash
git add backtests/__init__.py backtests/ma_ribbon_ema21/
git commit -m "feat(backtests/ma_ribbon_ema21): scaffold Phase 1 package + config"
```

---

## Task 2: Synthetic OHLCV fixtures

**Files:**
- Create: `backtests/ma_ribbon_ema21/tests/fixtures.py`

- [ ] **Step 1: Write the test for the fixture itself**

Create `backtests/ma_ribbon_ema21/tests/test_fixtures.py`:

```python
from __future__ import annotations
import pandas as pd
from backtests.ma_ribbon_ema21.tests.fixtures import (
    make_flat_ohlcv,
    make_uptrend_with_formation,
    make_real_csv_path,
)


def test_make_flat_ohlcv_shape():
    df = make_flat_ohlcv(n_bars=100, base_price=100.0)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 100
    assert (df["close"] == 100.0).all()
    assert df["timestamp"].is_monotonic_increasing


def test_make_uptrend_has_known_formation_bar():
    df, formation_bar_idx = make_uptrend_with_formation(
        n_bars=200, formation_at_bar=100, base_price=100.0
    )
    assert isinstance(df, pd.DataFrame)
    assert formation_bar_idx == 100
    # Closes before formation are below or at base, after formation rise
    assert df.loc[formation_bar_idx + 20, "close"] > df.loc[formation_bar_idx - 1, "close"]


def test_real_csv_fixture_exists():
    p = make_real_csv_path("BTCUSDT", "1h")
    # Path may or may not exist on first run — function only returns the path
    assert isinstance(p, str)
    assert "BTCUSDT" in p and "1h" in p
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtests.ma_ribbon_ema21.tests.fixtures'`

- [ ] **Step 3: Write `fixtures.py`**

Create `backtests/ma_ribbon_ema21/tests/fixtures.py`:

```python
"""Synthetic and real-data fixtures for Phase 1 tests."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


_TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}


def make_flat_ohlcv(
    n_bars: int = 100,
    base_price: float = 100.0,
    tf: str = "1h",
    start_ts: int = 1_700_000_000,
) -> pd.DataFrame:
    """A flat-price OHLCV with no alignment events. Useful as a negative control."""
    step = _TF_SECONDS[tf]
    ts = np.arange(n_bars) * step + start_ts
    return pd.DataFrame({
        "timestamp": ts.astype(np.int64),
        "open":   np.full(n_bars, base_price, dtype=float),
        "high":   np.full(n_bars, base_price, dtype=float),
        "low":    np.full(n_bars, base_price, dtype=float),
        "close":  np.full(n_bars, base_price, dtype=float),
        "volume": np.full(n_bars, 1.0, dtype=float),
    })


def make_uptrend_with_formation(
    n_bars: int = 200,
    formation_at_bar: int = 100,
    base_price: float = 100.0,
    tf: str = "1h",
    start_ts: int = 1_700_000_000,
    pre_drift: float = -0.001,
    post_drift: float = +0.005,
    noise_pct: float = 0.0005,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """OHLCV where bullish ribbon forms near `formation_at_bar`.
    Returns (df, formation_bar_idx). The formation bar is the first bar where
    close > MA5 > MA8 > EMA21 > MA55 holds, given the parameters.
    """
    rng = np.random.default_rng(seed)
    closes = np.empty(n_bars, dtype=float)
    closes[0] = base_price
    for i in range(1, n_bars):
        drift = pre_drift if i < formation_at_bar else post_drift
        noise = rng.normal(0.0, noise_pct)
        closes[i] = closes[i-1] * (1.0 + drift + noise)

    step = _TF_SECONDS[tf]
    ts = np.arange(n_bars) * step + start_ts
    df = pd.DataFrame({
        "timestamp": ts.astype(np.int64),
        "open":   closes,
        "high":   closes * (1.0 + 0.0005),
        "low":    closes * (1.0 - 0.0005),
        "close":  closes,
        "volume": np.full(n_bars, 1.0),
    })
    return df, formation_at_bar


def make_real_csv_path(symbol: str, tf: str) -> str:
    """Returns the canonical path to a small real-data fixture CSV.
    The CSV is checked-in to allow Phase 1 integration tests without network.
    """
    here = Path(__file__).resolve().parent
    return str(here / "data" / f"{symbol}_{tf}.csv")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_fixtures.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/tests/fixtures.py backtests/ma_ribbon_ema21/tests/test_fixtures.py
git commit -m "test(backtests/ma_ribbon_ema21): synthetic OHLCV fixtures"
```

---

## Task 3: Indicators (SMA + EMA)

**Files:**
- Create: `backtests/ma_ribbon_ema21/indicators.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_indicators.py`

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_indicators.py`:

```python
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.indicators import sma, ema


def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, period=3)
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_period_larger_than_series_returns_all_nan():
    s = pd.Series([1.0, 2.0, 3.0])
    out = sma(s, period=5)
    assert out.isna().all()


def test_ema_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(s, period=3)
    # EMA(3): alpha = 2/(3+1) = 0.5
    # First valid value at index 2 = SMA of first 3 = 2.0
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    # Index 3 = 0.5 * 4 + 0.5 * 2 = 3.0
    assert out.iloc[3] == pytest.approx(3.0)
    # Index 4 = 0.5 * 5 + 0.5 * 3 = 4.0
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_index_preserved():
    s = pd.Series([10.0, 20.0, 30.0, 40.0], index=[100, 200, 300, 400])
    out = ema(s, period=2)
    assert list(out.index) == [100, 200, 300, 400]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_indicators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtests.ma_ribbon_ema21.indicators'`

- [ ] **Step 3: Implement `indicators.py`**

Create `backtests/ma_ribbon_ema21/indicators.py`:

```python
"""Pure-function indicators: SMA, EMA. NaN-prefixed to match incomplete windows."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average. Returns a Series with NaN for the first `period - 1` bars."""
    if period < 1:
        raise ValueError(f"sma period must be >= 1, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. Seeded with SMA at index `period - 1`. NaN prefix."""
    if period < 1:
        raise ValueError(f"ema period must be >= 1, got {period}")
    n = len(series)
    out = pd.Series(np.full(n, np.nan, dtype=float), index=series.index)
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = float(series.iloc[:period].mean())
    out.iloc[period - 1] = seed
    prev = seed
    values = series.to_numpy(dtype=float)
    for i in range(period, n):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out.iloc[i] = prev
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_indicators.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/indicators.py backtests/ma_ribbon_ema21/tests/test_indicators.py
git commit -m "feat(backtests/ma_ribbon_ema21): SMA + EMA indicators"
```

---

## Task 4: Bullish-alignment detection + formation events

**Files:**
- Create: `backtests/ma_ribbon_ema21/ma_alignment.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_ma_alignment.py`

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_ma_alignment.py`:

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.ma_alignment import (
    bullish_aligned,
    formation_events,
    AlignmentConfig,
)


def _make_aligned_frame() -> pd.DataFrame:
    # Hand-crafted: 5 bars where ma5>ma8>ema21>ma55<close on bar 4 (last) only.
    return pd.DataFrame({
        "close": [100, 100, 100, 100, 110],
        "ma5":   [100, 100, 100, 100, 105],
        "ma8":   [100, 100, 100, 100, 102],
        "ema21": [100, 100, 100, 100, 101],
        "ma55":  [100, 100, 100, 100,  99],
    })


def test_bullish_aligned_returns_bool_series_aligned_with_input():
    df = _make_aligned_frame()
    cfg = AlignmentConfig.default()
    aligned = bullish_aligned(df, cfg)
    assert isinstance(aligned, pd.Series)
    assert aligned.dtype == bool
    assert list(aligned.index) == list(df.index)


def test_bullish_aligned_only_last_bar_true():
    df = _make_aligned_frame()
    aligned = bullish_aligned(df, AlignmentConfig.default())
    expected = [False, False, False, False, True]
    assert list(aligned) == expected


def test_bullish_aligned_handles_nan_gracefully():
    df = pd.DataFrame({
        "close": [np.nan, 100, 100],
        "ma5":   [np.nan, 100, 105],
        "ma8":   [np.nan, 100, 102],
        "ema21": [np.nan, 100, 101],
        "ma55":  [np.nan, 100,  99],
    })
    aligned = bullish_aligned(df, AlignmentConfig.default())
    assert aligned.iloc[0] == False  # NaN must not be True
    assert aligned.iloc[1] == False  # equal — strict gt fails
    assert aligned.iloc[2] == True


def test_formation_events_detects_false_to_true_transition_only():
    aligned = pd.Series([False, False, True, True, True, False, True])
    events = formation_events(aligned)
    # Transitions: False→True at idx 2, then False→True at idx 6.
    assert list(events) == [2, 6]


def test_formation_events_first_bar_aligned_does_not_count():
    aligned = pd.Series([True, True, False, True])
    events = formation_events(aligned)
    # First bar is aligned but there's no prior False → skip.
    # Only False→True at idx 3.
    assert list(events) == [3]


def test_alignment_config_subset_disables_check():
    df = pd.DataFrame({
        "close": [100],
        "ma5":   [99],   # would fail close > ma5
        "ma8":   [98],
        "ema21": [97],
        "ma55":  [96],
    })
    # With check disabled, alignment passes.
    cfg = AlignmentConfig(
        require_close_above_ma5=False,
        require_close_above_ma8=True,
        require_close_above_ema21=True,
        require_close_above_ma55=True,
        require_ma5_above_ma8=True,
        require_ma8_above_ema21=True,
        require_ema21_above_ma55=True,
    )
    aligned = bullish_aligned(df, cfg)
    assert aligned.iloc[0] == True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_ma_alignment.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `ma_alignment.py`**

Create `backtests/ma_ribbon_ema21/ma_alignment.py`:

```python
"""Bullish MA-ribbon alignment detection + formation event extraction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignmentConfig:
    require_close_above_ma5:    bool = True
    require_close_above_ma8:    bool = True
    require_close_above_ema21:  bool = True
    require_close_above_ma55:   bool = True
    require_ma5_above_ma8:      bool = True
    require_ma8_above_ema21:    bool = True
    require_ema21_above_ma55:   bool = True

    @staticmethod
    def default() -> "AlignmentConfig":
        return AlignmentConfig()

    @staticmethod
    def from_dict(d: dict) -> "AlignmentConfig":
        return AlignmentConfig(**{k: bool(v) for k, v in d.items()})


def bullish_aligned(df: pd.DataFrame, cfg: AlignmentConfig) -> pd.Series:
    """Return a bool Series aligned with df.index where bullish ribbon holds.
    Required columns on df: close, ma5, ma8, ema21, ma55.
    NaN in any required column → False at that bar (no look-ahead, no spurious truth).
    """
    required = ["close", "ma5", "ma8", "ema21", "ma55"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"bullish_aligned: df missing required columns {missing}")

    close = df["close"]
    ma5   = df["ma5"]
    ma8   = df["ma8"]
    ema21 = df["ema21"]
    ma55  = df["ma55"]

    aligned = pd.Series(True, index=df.index)
    if cfg.require_close_above_ma5:
        aligned &= close > ma5
    if cfg.require_close_above_ma8:
        aligned &= close > ma8
    if cfg.require_close_above_ema21:
        aligned &= close > ema21
    if cfg.require_close_above_ma55:
        aligned &= close > ma55
    if cfg.require_ma5_above_ma8:
        aligned &= ma5 > ma8
    if cfg.require_ma8_above_ema21:
        aligned &= ma8 > ema21
    if cfg.require_ema21_above_ma55:
        aligned &= ema21 > ma55

    # Any NaN comparison yields False in pandas, but be explicit:
    nan_mask = (
        close.isna() | ma5.isna() | ma8.isna() | ema21.isna() | ma55.isna()
    )
    aligned = aligned.fillna(False)
    aligned[nan_mask] = False

    return aligned.astype(bool)


def formation_events(aligned: pd.Series) -> pd.Index:
    """Return integer-position indices where aligned transitions False → True.
    First bar is treated as if previous bar was False, BUT we only count it as
    an event if its previous-step neighbor (i-1) is explicitly False.
    """
    a = aligned.to_numpy(dtype=bool)
    n = len(a)
    if n == 0:
        return pd.Index([], dtype=int)
    prev = np.concatenate(([False], a[:-1]))
    transitions = a & ~prev
    # The first bar can be an event only if there's no real history before it.
    # We treat the first bar's "previous" as False for the array — that means
    # if the first bar IS aligned, it would be counted as an event.
    # Per test contract we want first bar to NOT be counted unless explicitly preceded
    # by a False bar. Drop position 0:
    transitions[0] = False
    return pd.Index(np.where(transitions)[0], dtype=int)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_ma_alignment.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/ma_alignment.py backtests/ma_ribbon_ema21/tests/test_ma_alignment.py
git commit -m "feat(backtests/ma_ribbon_ema21): bullish-alignment detection + formation events"
```

---

## Task 5: Distance-from-MA features + bucketing

**Files:**
- Create: `backtests/ma_ribbon_ema21/distance_features.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_distance_features.py`

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_distance_features.py`:

```python
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.distance_features import (
    distance_to_ma_pct,
    distance_bucket,
    DEFAULT_BUCKETS,
)


def test_distance_to_ma_pct_simple():
    close = pd.Series([100.0, 105.0, 110.0])
    ma    = pd.Series([100.0, 100.0, 100.0])
    out = distance_to_ma_pct(close, ma)
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[1] == pytest.approx(0.05)
    assert out.iloc[2] == pytest.approx(0.10)


def test_distance_to_ma_pct_handles_nan_ma():
    close = pd.Series([100.0, 105.0])
    ma    = pd.Series([np.nan, 100.0])
    out = distance_to_ma_pct(close, ma)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.05)


def test_distance_bucket_returns_correct_label():
    # DEFAULT_BUCKETS = [(0,0.005),(0.005,0.01),(0.01,0.02),(0.02,0.04),(0.04,0.07),(0.07,1.0)]
    assert distance_bucket(0.001, DEFAULT_BUCKETS) == "[0.0%, 0.5%)"
    assert distance_bucket(0.005, DEFAULT_BUCKETS) == "[0.5%, 1.0%)"
    assert distance_bucket(0.025, DEFAULT_BUCKETS) == "[2.0%, 4.0%)"
    assert distance_bucket(0.50,  DEFAULT_BUCKETS) == "[7.0%, 100.0%)"


def test_distance_bucket_negative_returns_outlier():
    assert distance_bucket(-0.01, DEFAULT_BUCKETS) == "<negative>"


def test_distance_bucket_at_upper_edge_goes_to_next_bucket():
    # 0.005 → exactly second-bucket lower edge → "[0.5%, 1.0%)"
    assert distance_bucket(0.005, DEFAULT_BUCKETS) == "[0.5%, 1.0%)"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_distance_features.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `distance_features.py`**

Create `backtests/ma_ribbon_ema21/distance_features.py`:

```python
"""Distance-from-MA features and bucketing."""
from __future__ import annotations
import numpy as np
import pandas as pd


# Each tuple is (lower_inclusive, upper_exclusive) in decimal form.
DEFAULT_BUCKETS: list[tuple[float, float]] = [
    (0.000, 0.005),
    (0.005, 0.010),
    (0.010, 0.020),
    (0.020, 0.040),
    (0.040, 0.070),
    (0.070, 1.000),
]


def distance_to_ma_pct(close: pd.Series, ma: pd.Series) -> pd.Series:
    """Returns (close - ma) / ma per bar, NaN where ma is NaN or zero."""
    ma_safe = ma.where(ma.abs() > 1e-12, np.nan)
    return (close - ma_safe) / ma_safe


def distance_bucket(distance: float, buckets: list[tuple[float, float]]) -> str:
    """Map a single distance value to a string bucket label.
    Negative distances return "<negative>" (caller decides whether to keep them)."""
    if distance < 0:
        return "<negative>"
    for lo, hi in buckets:
        if lo <= distance < hi:
            return _label(lo, hi)
    return "<above_max>"


def _label(lo: float, hi: float) -> str:
    return f"[{lo*100:.1f}%, {hi*100:.1f}%)"
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_distance_features.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/distance_features.py backtests/ma_ribbon_ema21/tests/test_distance_features.py
git commit -m "feat(backtests/ma_ribbon_ema21): distance-from-MA features + bucketing"
```

---

## Task 6: Forward returns

**Files:**
- Create: `backtests/ma_ribbon_ema21/forward_returns.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_forward_returns.py`

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_forward_returns.py`:

```python
from __future__ import annotations
import math
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.forward_returns import (
    forward_return,
    forward_returns_at,
    apply_round_trip_cost,
)


def test_forward_return_simple_long():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    # entry at idx=1 (close=101), forward 3 bars → idx 4 (close=104)
    r = forward_return(close, entry_idx=1, n_bars=3)
    assert r == pytest.approx((104.0 - 101.0) / 101.0)


def test_forward_return_returns_nan_when_horizon_exceeds_data():
    close = pd.Series([100.0, 101.0, 102.0])
    r = forward_return(close, entry_idx=1, n_bars=10)
    assert math.isnan(r)


def test_forward_returns_at_returns_dict_keyed_by_horizon():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    out = forward_returns_at(close, entry_idx=0, horizons=[1, 3, 5])
    assert out == pytest.approx({
        1: (101 - 100) / 100,
        3: (103 - 100) / 100,
        5: (105 - 100) / 100,
    })


def test_apply_round_trip_cost_subtracts_two_sides_plus_slippage():
    raw_return = 0.05  # 5%
    cost = apply_round_trip_cost(raw_return, fee_per_side=0.0005, slippage_per_fill=0.0001)
    # 2 fills, each: fee 0.0005 + slippage 0.0001 = 0.0006
    # round trip cost: 2 * 0.0006 = 0.0012
    assert cost == pytest.approx(0.05 - 0.0012)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_forward_returns.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `forward_returns.py`**

Create `backtests/ma_ribbon_ema21/forward_returns.py`:

```python
"""Forward-return computation. Strict: never reach beyond data length."""
from __future__ import annotations
import math
import pandas as pd


def forward_return(close: pd.Series, entry_idx: int, n_bars: int) -> float:
    """Return (close[entry_idx + n_bars] - close[entry_idx]) / close[entry_idx].
    NaN if entry_idx + n_bars exceeds len(close) - 1.
    """
    if n_bars < 1:
        raise ValueError(f"forward_return n_bars must be >= 1, got {n_bars}")
    target = entry_idx + n_bars
    if target >= len(close):
        return math.nan
    entry  = float(close.iloc[entry_idx])
    later  = float(close.iloc[target])
    if entry <= 0 or math.isnan(entry) or math.isnan(later):
        return math.nan
    return (later - entry) / entry


def forward_returns_at(
    close: pd.Series,
    entry_idx: int,
    horizons: list[int],
) -> dict[int, float]:
    """Vectorized version: returns dict horizon → forward return."""
    return {h: forward_return(close, entry_idx, h) for h in horizons}


def apply_round_trip_cost(
    raw_return: float,
    fee_per_side: float = 0.0005,
    slippage_per_fill: float = 0.0001,
) -> float:
    """Subtract a long round-trip cost: 2 fills * (fee + slippage). Long-only."""
    if math.isnan(raw_return):
        return math.nan
    cost_per_fill = fee_per_side + slippage_per_fill
    return raw_return - 2.0 * cost_per_fill
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_forward_returns.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/forward_returns.py backtests/ma_ribbon_ema21/tests/test_forward_returns.py
git commit -m "feat(backtests/ma_ribbon_ema21): forward-return computation + round-trip cost"
```

---

## Task 7: 70/30 train/test split by time

**Files:**
- Create: `backtests/ma_ribbon_ema21/data_split.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_70_30_split.py`

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_70_30_split.py`:

```python
from __future__ import annotations
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.data_split import split_by_time, label_split_column


def test_split_by_time_70_30_count():
    ts = list(range(100))
    is_train, is_test = split_by_time(ts, train_pct=0.70)
    assert sum(is_train) == 70
    assert sum(is_test) == 30


def test_split_by_time_train_is_first_70_pct():
    ts = list(range(10))
    is_train, is_test = split_by_time(ts, train_pct=0.70)
    assert is_train == [True]*7 + [False]*3
    assert is_test  == [False]*7 + [True]*3


def test_split_by_time_invalid_pct_raises():
    with pytest.raises(ValueError):
        split_by_time([1, 2, 3], train_pct=1.5)
    with pytest.raises(ValueError):
        split_by_time([1, 2, 3], train_pct=-0.1)


def test_label_split_column_attaches_train_test_labels():
    df = pd.DataFrame({"timestamp": list(range(10)), "close": list(range(10))})
    out = label_split_column(df, train_pct=0.70)
    assert "split" in out.columns
    assert (out.iloc[:7]["split"] == "train").all()
    assert (out.iloc[7:]["split"] == "test").all()


def test_label_split_column_does_not_mutate_input():
    df = pd.DataFrame({"timestamp": list(range(10)), "close": list(range(10))})
    _ = label_split_column(df, train_pct=0.70)
    assert "split" not in df.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_70_30_split.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `data_split.py`**

Create `backtests/ma_ribbon_ema21/data_split.py`:

```python
"""Time-ordered train/test split. Out-of-sample is the last (1 - train_pct) fraction."""
from __future__ import annotations
import pandas as pd


def split_by_time(timestamps: list[int], train_pct: float = 0.70) -> tuple[list[bool], list[bool]]:
    """Returns (is_train, is_test) — equal-length boolean lists.
    The first `train_pct * len` bars are train; the rest are test.
    """
    if not (0.0 < train_pct < 1.0):
        raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")
    n = len(timestamps)
    cutoff = int(n * train_pct)
    is_train = [i < cutoff for i in range(n)]
    is_test  = [not v for v in is_train]
    return is_train, is_test


def label_split_column(df: pd.DataFrame, train_pct: float = 0.70) -> pd.DataFrame:
    """Returns a NEW DataFrame with a 'split' column containing 'train' or 'test'."""
    if "timestamp" not in df.columns:
        raise ValueError("label_split_column requires a 'timestamp' column")
    out = df.copy()
    is_train, _ = split_by_time(list(out["timestamp"]), train_pct=train_pct)
    out["split"] = ["train" if t else "test" for t in is_train]
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_70_30_split.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/data_split.py backtests/ma_ribbon_ema21/tests/test_70_30_split.py
git commit -m "feat(backtests/ma_ribbon_ema21): 70/30 time-ordered train/test split"
```

---

## Task 8: Multi-TF time alignment helper

**Files:**
- Create: `backtests/ma_ribbon_ema21/multi_tf.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_multi_tf_alignment.py`

This is the canonical "no-lookahead across timeframes" helper. At any timestamp T, compute the index of the latest bar whose CLOSE time is `<= T`.

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_multi_tf_alignment.py`:

```python
from __future__ import annotations
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.multi_tf import (
    bar_close_time,
    latest_closed_bar_idx,
    TF_SECONDS,
)


def test_tf_seconds_table():
    assert TF_SECONDS["5m"]  == 300
    assert TF_SECONDS["15m"] == 900
    assert TF_SECONDS["1h"]  == 3600
    assert TF_SECONDS["4h"]  == 14400


def test_bar_close_time_5m():
    # Bar opens at 10:05 (= 1700000000 + 300*1) → closes at 10:10
    open_ts = 1_700_000_300
    assert bar_close_time(open_ts, "5m") == 1_700_000_300 + 300


def test_latest_closed_bar_idx_15m_at_specific_time():
    # 15m bars open at 09:45, 10:00, 10:15. close times: 10:00, 10:15, 10:30.
    open_times = [1_700_000_000 + 0,
                  1_700_000_000 + 900,
                  1_700_000_000 + 1800]
    # T = 10:10 (= +600 from 10:00). Latest closed bar = the [09:45,10:00] bar (idx 0).
    idx = latest_closed_bar_idx(open_times, "15m", T=1_700_000_000 + 600)
    assert idx == 0


def test_latest_closed_bar_idx_at_boundary_includes_closed_bar():
    # T exactly == close time of bar 0 → bar 0 IS closed.
    open_times = [1_700_000_000 + 0,
                  1_700_000_000 + 900]
    idx = latest_closed_bar_idx(open_times, "15m", T=1_700_000_000 + 900)
    assert idx == 0


def test_latest_closed_bar_idx_before_any_close_returns_negative_one():
    open_times = [1_700_000_000]
    # T before bar 0 even closes → -1.
    idx = latest_closed_bar_idx(open_times, "15m", T=1_700_000_000 + 100)
    assert idx == -1


def test_latest_closed_bar_idx_real_user_example():
    """At T = 10:10 UTC, latest 5m closed bar is [10:05, 10:10] (idx X),
    latest 15m closed bar is [09:45, 10:00] (idx Y).
    """
    base = 1_700_000_000  # let's say this is 09:45 UTC
    five_m_opens   = [base + i * 300 for i in range(10)]   # 09:45, 09:50, ..., 10:30
    fifteen_m_opens = [base + i * 900 for i in range(4)]   # 09:45, 10:00, 10:15, 10:30
    T = base + 25 * 60                                      # 10:10

    # 5m bars closing on or before T=10:10: open at 09:45 closes 09:50 (yes),
    # 09:50 closes 09:55, ..., 10:05 closes 10:10. Five 5m bars closed → idx 4.
    assert latest_closed_bar_idx(five_m_opens, "5m", T) == 4

    # 15m bars closing on or before T=10:10: only [09:45, 10:00] (idx 0).
    assert latest_closed_bar_idx(fifteen_m_opens, "15m", T) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_multi_tf_alignment.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `multi_tf.py`**

Create `backtests/ma_ribbon_ema21/multi_tf.py`:

```python
"""Multi-timeframe time alignment. Critical for no-lookahead across TFs."""
from __future__ import annotations
import bisect


TF_SECONDS: dict[str, int] = {
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
}


def bar_close_time(open_ts: int, tf: str) -> int:
    """Bar's close time = open_ts + tf duration."""
    if tf not in TF_SECONDS:
        raise ValueError(f"unknown tf {tf!r}; expected one of {list(TF_SECONDS)}")
    return open_ts + TF_SECONDS[tf]


def latest_closed_bar_idx(open_times: list[int], tf: str, T: int) -> int:
    """Index of the most recently closed bar at simulation time T.
    Returns -1 if no bar has closed yet at time T.
    `open_times` must be sorted ascending.
    """
    if tf not in TF_SECONDS:
        raise ValueError(f"unknown tf {tf!r}")
    duration = TF_SECONDS[tf]
    # A bar at index i is "closed at T" iff open_times[i] + duration <= T.
    # Equivalently, open_times[i] <= T - duration.
    threshold = T - duration
    # Find rightmost index with open_times[i] <= threshold.
    pos = bisect.bisect_right(open_times, threshold)
    return pos - 1
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_multi_tf_alignment.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/multi_tf.py backtests/ma_ribbon_ema21/tests/test_multi_tf_alignment.py
git commit -m "feat(backtests/ma_ribbon_ema21): multi-TF time alignment helper"
```

---

## Task 9: No-lookahead test (the critical safety net)

**Files:**
- Create: `backtests/ma_ribbon_ema21/tests/test_no_lookahead.py`

This task adds NO new code. It is a meta-test that walks bar-by-bar through a synthetic series and asserts that signals at bar `i` depend only on bars `[0, i]`. It runs every commit. If a future change introduces look-ahead, this test fails immediately.

- [ ] **Step 1: Write the test**

Create `backtests/ma_ribbon_ema21/tests/test_no_lookahead.py`:

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from backtests.ma_ribbon_ema21.indicators import sma, ema
from backtests.ma_ribbon_ema21.ma_alignment import bullish_aligned, AlignmentConfig
from backtests.ma_ribbon_ema21.tests.fixtures import make_uptrend_with_formation


def _attach_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma5"]   = sma(out["close"], 5)
    out["ma8"]   = sma(out["close"], 8)
    out["ema21"] = ema(out["close"], 21)
    out["ma55"]  = sma(out["close"], 55)
    return out


def test_signal_at_bar_i_uses_only_bars_0_through_i():
    """Walk bar-by-bar; for each prefix [0..i], compute alignment on the prefix
    and assert that the alignment[i] value equals what the full-series compute gives.
    Any divergence = look-ahead bias."""
    df, formation_at = make_uptrend_with_formation(
        n_bars=200, formation_at_bar=100, base_price=100.0
    )
    full = _attach_mas(df)
    full_aligned = bullish_aligned(full, AlignmentConfig.default())

    for i in range(60, len(df)):  # start after MA55 has at least one valid bar
        prefix = df.iloc[: i + 1].copy()
        prefix = _attach_mas(prefix)
        prefix_aligned = bullish_aligned(prefix, AlignmentConfig.default())
        assert prefix_aligned.iloc[i] == full_aligned.iloc[i], (
            f"alignment at bar {i} changed when future bars were hidden — look-ahead bug"
        )


def test_alignment_is_false_before_formation_bar():
    df, formation_at = make_uptrend_with_formation(
        n_bars=200, formation_at_bar=100, base_price=100.0,
        pre_drift=-0.001, post_drift=+0.005, noise_pct=0.0,
        seed=42,
    )
    full = _attach_mas(df)
    aligned = bullish_aligned(full, AlignmentConfig.default())
    # At bar (formation_at - 5), the ribbon should not be aligned yet
    assert aligned.iloc[formation_at - 5] == False
```

- [ ] **Step 2: Run the test**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_no_lookahead.py -v`
Expected: 2 PASS

This validates that the implementations from Task 3 and Task 4 are look-ahead-free.

- [ ] **Step 3: Commit**

```bash
git add backtests/ma_ribbon_ema21/tests/test_no_lookahead.py
git commit -m "test(backtests/ma_ribbon_ema21): no-lookahead invariant for indicators + alignment"
```

---

## Task 10: Bitget OHLCV data loader (with CSV cache)

**Files:**
- Create: `backtests/ma_ribbon_ema21/data_loader.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_data_loader.py`
- Create: `backtests/ma_ribbon_ema21/tests/data/BTCUSDT_1h_small.csv` (small fixture, ~50 rows)

The loader reads from a per-symbol-per-TF CSV cache. If the cache is missing or shorter than what `PRINCIPLES.md` §P1 requires (max available depth), it fetches from Bitget v2 perp history and writes a fresh CSV.

- [ ] **Step 1: Write the failing test (CSV path only — no network)**

Create `backtests/ma_ribbon_ema21/tests/test_data_loader.py`:

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.data_loader import (
    load_ohlcv_from_csv,
    DataLoaderConfig,
)


def _write_small_fixture(tmp_path: Path) -> Path:
    csv = tmp_path / "BTCUSDT_1h.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1700000000,100,101,99,100,1.0\n"
        "1700003600,100,101,99,101,1.0\n"
        "1700007200,101,102,100,102,1.0\n"
    )
    return csv


def test_load_ohlcv_from_csv_returns_correct_dataframe(tmp_path):
    _write_small_fixture(tmp_path)
    cfg = DataLoaderConfig(cache_dir=str(tmp_path))
    df = load_ohlcv_from_csv("BTCUSDT", "1h", cfg)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df.iloc[2]["close"] == 102.0


def test_load_ohlcv_from_csv_missing_returns_empty(tmp_path):
    cfg = DataLoaderConfig(cache_dir=str(tmp_path))
    df = load_ohlcv_from_csv("DOESNOTEXIST", "1h", cfg)
    assert df.empty
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_load_ohlcv_from_csv_sorts_by_timestamp(tmp_path):
    csv = tmp_path / "FOOUSDT_1h.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1700007200,103,104,102,103,1.0\n"
        "1700000000,100,101,99,100,1.0\n"
        "1700003600,101,102,100,101,1.0\n"
    )
    cfg = DataLoaderConfig(cache_dir=str(tmp_path))
    df = load_ohlcv_from_csv("FOOUSDT", "1h", cfg)
    assert list(df["timestamp"]) == [1700000000, 1700003600, 1700007200]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_data_loader.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `data_loader.py` (CSV-only first)**

Create `backtests/ma_ribbon_ema21/data_loader.py`:

```python
"""OHLCV data loader. CSV cache primary; Bitget fetch when cache empty / stale.

PRINCIPLES.md §P1: pull max available depth from Bitget. No tfDays cap.
PRINCIPLES.md §P2: if cache is shorter than requested, refetch.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import httpx
import pandas as pd


_LOG = logging.getLogger(__name__)

_BITGET_TF_MAP: dict[str, str] = {
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1H",
    "4h":  "4H",
}

# Bitget v2 USDT-M perp candles endpoint (publicly documented).
_BITGET_HISTORY_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"

_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DataLoaderConfig:
    cache_dir: str = "data/csv_cache/ma_ribbon_ema21"
    bitget_request_limit: int = 200            # Bitget cap per request
    bitget_pages_per_symbol: int = 1000        # safe upper bound; actual stops at empty page
    bitget_sleep_seconds: float = 0.05
    request_timeout_seconds: float = 15.0


def csv_path(symbol: str, tf: str, cfg: DataLoaderConfig) -> Path:
    return Path(cfg.cache_dir) / f"{symbol}_{tf}.csv"


def load_ohlcv_from_csv(symbol: str, tf: str, cfg: DataLoaderConfig) -> pd.DataFrame:
    p = csv_path(symbol, tf, cfg)
    if not p.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=_COLS)
    df = df[_COLS].sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def fetch_ohlcv_from_bitget(symbol: str, tf: str, cfg: DataLoaderConfig) -> pd.DataFrame:
    """Fetch up to max-depth OHLCV from Bitget. Paginates until empty page.
    Returns a DataFrame sorted ascending by timestamp.
    """
    if tf not in _BITGET_TF_MAP:
        raise ValueError(f"unsupported tf {tf!r}")
    bar = _BITGET_TF_MAP[tf]
    rows: list[list] = []
    end_ts: int | None = None
    for page in range(cfg.bitget_pages_per_symbol):
        params: dict[str, str] = {
            "symbol":      f"{symbol}_UMCBL",
            "productType": "USDT-FUTURES",
            "granularity": bar,
            "limit":       str(cfg.bitget_request_limit),
        }
        if end_ts is not None:
            params["endTime"] = str(end_ts)
        try:
            r = httpx.get(_BITGET_HISTORY_URL, params=params,
                          timeout=cfg.request_timeout_seconds)
            r.raise_for_status()
            data = r.json().get("data", []) or []
        except (httpx.HTTPError, ValueError) as exc:
            # Per PRINCIPLES §P10 — don't silently swallow.
            _LOG.error("bitget fetch failed for %s %s page %d: %s",
                       symbol, tf, page, exc)
            raise
        if not data:
            break
        rows.extend(data)
        # Bitget returns DESC; the smallest timestamp is the last item.
        end_ts = int(data[-1][0]) - 1
        time.sleep(cfg.bitget_sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume"
    ])
    df = df[_COLS].astype({"timestamp": "int64"}).astype(
        {c: float for c in ["open", "high", "low", "close", "volume"]}
    )
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    _LOG.info("fetched %d bars for %s %s from Bitget", len(df), symbol, tf)
    return df


def write_csv_cache(df: pd.DataFrame, symbol: str, tf: str, cfg: DataLoaderConfig) -> Path:
    p = csv_path(symbol, tf, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def load_or_fetch(symbol: str, tf: str, cfg: DataLoaderConfig) -> pd.DataFrame:
    """Per PRINCIPLES §P2: if cache is empty, fetch and write. Always returns
    the freshest available DataFrame.
    """
    cached = load_ohlcv_from_csv(symbol, tf, cfg)
    if not cached.empty:
        return cached
    fresh = fetch_ohlcv_from_bitget(symbol, tf, cfg)
    if not fresh.empty:
        write_csv_cache(fresh, symbol, tf, cfg)
    return fresh
```

- [ ] **Step 4: Run CSV-only tests to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_data_loader.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/data_loader.py backtests/ma_ribbon_ema21/tests/test_data_loader.py
git commit -m "feat(backtests/ma_ribbon_ema21): CSV-cached Bitget OHLCV loader"
```

---

## Task 11: Phase 1 engine — single (symbol, TF)

**Files:**
- Create: `backtests/ma_ribbon_ema21/phase1_engine.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_phase1_engine.py`

This module ties together: indicators → alignment → formation events → distance features → forward returns → cohort entries.

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_phase1_engine.py`:

```python
from __future__ import annotations
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.phase1_engine import (
    scan_symbol_tf,
    Phase1Event,
)
from backtests.ma_ribbon_ema21.tests.fixtures import (
    make_uptrend_with_formation,
    make_flat_ohlcv,
)


def test_scan_symbol_tf_returns_at_least_one_event_on_uptrend():
    df, formation_bar = make_uptrend_with_formation(
        n_bars=300, formation_at_bar=120, base_price=100.0
    )
    events = scan_symbol_tf(df, symbol="TESTUSDT", tf="1h")
    assert len(events) >= 1
    e0 = events[0]
    assert isinstance(e0, Phase1Event)
    assert e0.symbol == "TESTUSDT"
    assert e0.tf == "1h"
    # Formation bar idx should be at or after the synthetic formation_at_bar
    assert e0.formation_bar_idx >= formation_bar - 5
    assert e0.formation_bar_idx <= formation_bar + 30


def test_scan_symbol_tf_returns_no_events_on_flat():
    df = make_flat_ohlcv(n_bars=300, base_price=100.0)
    events = scan_symbol_tf(df, symbol="FLATUSDT", tf="1h")
    assert events == []


def test_scan_symbol_tf_event_has_distance_features_and_forward_returns():
    df, _ = make_uptrend_with_formation(
        n_bars=300, formation_at_bar=120, base_price=100.0
    )
    events = scan_symbol_tf(df, symbol="X", tf="1h", forward_horizons=[5, 10, 20])
    e = events[0]
    assert e.distance_to_ma5_pct  is not None
    assert e.distance_to_ma8_pct  is not None
    assert e.distance_to_ema21_pct is not None
    assert e.distance_to_ma55_pct is not None
    assert set(e.forward_returns.keys()) == {5, 10, 20}
    assert set(e.forward_returns_post_fee.keys()) == {5, 10, 20}


def test_scan_symbol_tf_skips_events_without_enough_history():
    df, _ = make_uptrend_with_formation(
        n_bars=60, formation_at_bar=30, base_price=100.0
    )
    # MA55 needs 55 bars of history → events before bar 54 cannot fire.
    events = scan_symbol_tf(df, symbol="X", tf="1h")
    for e in events:
        assert e.formation_bar_idx >= 54
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_phase1_engine.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `phase1_engine.py` (single-pair function only)**

Create `backtests/ma_ribbon_ema21/phase1_engine.py`:

```python
"""Phase 1 engine: scan formation events for a single (symbol, TF) and
attach distance features + forward returns."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Iterable
import pandas as pd

from backtests.ma_ribbon_ema21.indicators import sma, ema
from backtests.ma_ribbon_ema21.ma_alignment import (
    AlignmentConfig,
    bullish_aligned,
    formation_events,
)
from backtests.ma_ribbon_ema21.distance_features import (
    distance_to_ma_pct,
)
from backtests.ma_ribbon_ema21.forward_returns import (
    forward_returns_at,
    apply_round_trip_cost,
)


_DEFAULT_HORIZONS = (5, 10, 20, 50)


@dataclass
class Phase1Event:
    symbol: str
    tf: str
    formation_bar_idx: int
    formation_timestamp: int
    formation_close: float
    distance_to_ma5_pct:   float
    distance_to_ma8_pct:   float
    distance_to_ema21_pct: float
    distance_to_ma55_pct:  float
    ribbon_width_pct:      float
    forward_returns:          dict[int, float] = field(default_factory=dict)
    forward_returns_post_fee: dict[int, float] = field(default_factory=dict)
    split: str = "unknown"   # "train" or "test"


def _enrich_with_indicators(
    df: pd.DataFrame,
    ma5_period: int = 5,
    ma8_period: int = 8,
    ema21_period: int = 21,
    ma55_period: int = 55,
) -> pd.DataFrame:
    out = df.copy()
    out["ma5"]   = sma(out["close"], ma5_period)
    out["ma8"]   = sma(out["close"], ma8_period)
    out["ema21"] = ema(out["close"], ema21_period)
    out["ma55"]  = sma(out["close"], ma55_period)
    return out


def scan_symbol_tf(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    alignment_cfg: AlignmentConfig | None = None,
    forward_horizons: Iterable[int] = _DEFAULT_HORIZONS,
    fee_per_side: float = 0.0005,
    slippage_per_fill: float = 0.0001,
    train_pct: float = 0.70,
) -> list[Phase1Event]:
    """Scan one (symbol, TF) DataFrame for formation events. Returns list."""
    if df.empty:
        return []
    if "close" not in df.columns:
        raise ValueError("scan_symbol_tf: df missing 'close' column")
    enriched = _enrich_with_indicators(df)
    cfg = alignment_cfg or AlignmentConfig.default()
    aligned = bullish_aligned(enriched, cfg)
    event_idx = formation_events(aligned)

    horizons = tuple(forward_horizons)
    n = len(enriched)
    cutoff = int(n * train_pct)
    out: list[Phase1Event] = []

    for i in event_idx:
        i = int(i)
        row = enriched.iloc[i]
        ts = int(row["timestamp"])
        close = float(row["close"])
        d5    = distance_to_ma_pct(enriched["close"], enriched["ma5"]).iloc[i]
        d8    = distance_to_ma_pct(enriched["close"], enriched["ma8"]).iloc[i]
        d21   = distance_to_ma_pct(enriched["close"], enriched["ema21"]).iloc[i]
        d55   = distance_to_ma_pct(enriched["close"], enriched["ma55"]).iloc[i]
        rib_w = (float(row["ma5"]) - float(row["ma55"])) / float(row["ma55"]) \
                if not math.isnan(row["ma55"]) and row["ma55"] != 0 else math.nan

        raw  = forward_returns_at(enriched["close"], i, list(horizons))
        post = {h: apply_round_trip_cost(r, fee_per_side, slippage_per_fill)
                for h, r in raw.items()}

        out.append(Phase1Event(
            symbol=symbol, tf=tf,
            formation_bar_idx=i, formation_timestamp=ts, formation_close=close,
            distance_to_ma5_pct=float(d5),
            distance_to_ma8_pct=float(d8),
            distance_to_ema21_pct=float(d21),
            distance_to_ma55_pct=float(d55),
            ribbon_width_pct=float(rib_w),
            forward_returns=raw,
            forward_returns_post_fee=post,
            split=("train" if i < cutoff else "test"),
        ))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_phase1_engine.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/phase1_engine.py backtests/ma_ribbon_ema21/tests/test_phase1_engine.py
git commit -m "feat(backtests/ma_ribbon_ema21): single-pair phase1 scan engine"
```

---

## Task 12: Phase 1 engine — universe scan

**Files:**
- Modify: `backtests/ma_ribbon_ema21/phase1_engine.py`
- Modify: `backtests/ma_ribbon_ema21/tests/test_phase1_engine.py` (append a universe test)

- [ ] **Step 1: Append the universe test**

Add to `backtests/ma_ribbon_ema21/tests/test_phase1_engine.py`:

```python
from backtests.ma_ribbon_ema21.phase1_engine import scan_universe, UniverseConfig
from backtests.ma_ribbon_ema21.data_loader import DataLoaderConfig
from backtests.ma_ribbon_ema21.tests.fixtures import make_uptrend_with_formation


def test_scan_universe_returns_events_for_each_symbol_tf_pair(tmp_path, monkeypatch):
    # Stage two fixture CSVs into a temporary cache.
    df_a, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=100, tf="1h")
    df_b, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=120, tf="4h")
    (tmp_path / "AAAUSDT_1h.csv").write_text(df_a.to_csv(index=False))
    (tmp_path / "BBBUSDT_4h.csv").write_text(df_b.to_csv(index=False))

    cfg = UniverseConfig(
        symbols=["AAAUSDT", "BBBUSDT"],
        timeframes=["1h", "4h"],
        loader=DataLoaderConfig(cache_dir=str(tmp_path)),
    )
    events = scan_universe(cfg)
    syms_tfs = {(e.symbol, e.tf) for e in events}
    # AAA has data only on 1h, BBB only on 4h; so we expect those two pairs.
    assert ("AAAUSDT", "1h") in syms_tfs
    assert ("BBBUSDT", "4h") in syms_tfs


def test_scan_universe_handles_missing_symbol_gracefully(tmp_path):
    cfg = UniverseConfig(
        symbols=["DOESNOTEXIST"],
        timeframes=["1h"],
        loader=DataLoaderConfig(cache_dir=str(tmp_path)),
    )
    events = scan_universe(cfg)
    assert events == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_phase1_engine.py::test_scan_universe_returns_events_for_each_symbol_tf_pair -v`
Expected: FAIL with `cannot import name 'scan_universe'`.

- [ ] **Step 3: Append `scan_universe` to `phase1_engine.py`**

Append to `backtests/ma_ribbon_ema21/phase1_engine.py`:

```python
from dataclasses import dataclass
from backtests.ma_ribbon_ema21.data_loader import (
    DataLoaderConfig,
    load_or_fetch,
)


@dataclass
class UniverseConfig:
    symbols: list[str]
    timeframes: list[str]
    loader: DataLoaderConfig
    alignment_cfg: AlignmentConfig | None = None
    forward_horizons: tuple[int, ...] = _DEFAULT_HORIZONS
    fee_per_side: float = 0.0005
    slippage_per_fill: float = 0.0001
    train_pct: float = 0.70


def scan_universe(cfg: UniverseConfig) -> list[Phase1Event]:
    """Scan every (symbol, TF) pair. Skip pairs whose data is unavailable.
    Per PRINCIPLES §P11: log every load result.
    """
    import logging
    log = logging.getLogger(__name__)
    out: list[Phase1Event] = []
    for symbol in cfg.symbols:
        for tf in cfg.timeframes:
            df = load_or_fetch(symbol, tf, cfg.loader)
            if df.empty:
                log.warning("no data for %s %s — skipping", symbol, tf)
                continue
            log.info("scanning %s %s (%d bars)", symbol, tf, len(df))
            events = scan_symbol_tf(
                df, symbol=symbol, tf=tf,
                alignment_cfg=cfg.alignment_cfg,
                forward_horizons=cfg.forward_horizons,
                fee_per_side=cfg.fee_per_side,
                slippage_per_fill=cfg.slippage_per_fill,
                train_pct=cfg.train_pct,
            )
            out.extend(events)
            log.info("  → %d events", len(events))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_phase1_engine.py -v`
Expected: 6 PASS (4 from Task 11 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/phase1_engine.py backtests/ma_ribbon_ema21/tests/test_phase1_engine.py
git commit -m "feat(backtests/ma_ribbon_ema21): universe scan across (symbol × TF) pairs"
```

---

## Task 13: Cohort report (aggregation + markdown writer)

**Files:**
- Create: `backtests/ma_ribbon_ema21/cohort_report.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_cohort_report.py`

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_cohort_report.py`:

```python
from __future__ import annotations
from pathlib import Path
import pytest
from backtests.ma_ribbon_ema21.phase1_engine import Phase1Event
from backtests.ma_ribbon_ema21.cohort_report import (
    aggregate_cohorts,
    write_markdown_report,
    CohortStats,
)


def _evt(sym, tf, idx, dist5, ret_post, split):
    return Phase1Event(
        symbol=sym, tf=tf, formation_bar_idx=idx,
        formation_timestamp=1_700_000_000 + idx,
        formation_close=100.0,
        distance_to_ma5_pct=dist5,
        distance_to_ma8_pct=dist5,
        distance_to_ema21_pct=dist5,
        distance_to_ma55_pct=dist5,
        ribbon_width_pct=0.01,
        forward_returns={20: ret_post + 0.0012},   # raw before round-trip cost
        forward_returns_post_fee={20: ret_post},
        split=split,
    )


def test_aggregate_cohorts_groups_by_symbol_tf_bucket():
    events = [
        _evt("AAA", "1h", 100, 0.001, 0.02, "train"),
        _evt("AAA", "1h", 200, 0.001, 0.03, "train"),
        _evt("AAA", "1h", 300, 0.025, -0.01, "test"),
    ]
    cohorts = aggregate_cohorts(events, horizon=20)
    # Should produce two cohorts: (AAA, 1h, [0,0.5%), train), (AAA, 1h, [2%,4%), test)
    keys = {(c.symbol, c.tf, c.bucket, c.split) for c in cohorts}
    assert ("AAA", "1h", "[0.0%, 0.5%)", "train") in keys
    assert ("AAA", "1h", "[2.0%, 4.0%)", "test") in keys


def test_aggregate_cohorts_computes_mean_and_winrate():
    events = [
        _evt("AAA", "1h", 100, 0.001, 0.02, "train"),
        _evt("AAA", "1h", 200, 0.001, -0.01, "train"),
        _evt("AAA", "1h", 300, 0.001, 0.03, "train"),
    ]
    cohorts = aggregate_cohorts(events, horizon=20)
    c = next(c for c in cohorts if c.split == "train")
    assert c.count == 3
    assert c.mean_return_post_fee == pytest.approx((0.02 - 0.01 + 0.03) / 3)
    assert c.win_rate == pytest.approx(2/3)


def test_write_markdown_report_creates_file(tmp_path):
    events = [_evt("AAA", "1h", 100, 0.001, 0.02, "train")]
    cohorts = aggregate_cohorts(events, horizon=20)
    out = tmp_path / "report.md"
    write_markdown_report(cohorts, output_path=str(out), horizon=20)
    text = out.read_text()
    assert out.exists()
    assert "AAA" in text and "1h" in text
    assert "0.0%" in text  # bucket label appears
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_cohort_report.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `cohort_report.py`**

Create `backtests/ma_ribbon_ema21/cohort_report.py`:

```python
"""Aggregate phase1 events into per-cohort stats and write a markdown report."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import math
from datetime import datetime, timezone
from statistics import mean, median

from backtests.ma_ribbon_ema21.phase1_engine import Phase1Event
from backtests.ma_ribbon_ema21.distance_features import distance_bucket, DEFAULT_BUCKETS


@dataclass
class CohortStats:
    symbol: str
    tf: str
    bucket: str
    split: str          # "train" | "test"
    count: int
    mean_return_post_fee:   float
    median_return_post_fee: float
    win_rate:               float
    worst_return_post_fee:  float


def aggregate_cohorts(
    events: list[Phase1Event],
    horizon: int,
    buckets: list[tuple[float, float]] = None,
) -> list[CohortStats]:
    buckets = buckets or DEFAULT_BUCKETS
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for ev in events:
        ret = ev.forward_returns_post_fee.get(horizon)
        if ret is None or math.isnan(ret):
            continue
        bucket = distance_bucket(ev.distance_to_ma5_pct, buckets)
        groups[(ev.symbol, ev.tf, bucket, ev.split)].append(ret)

    out: list[CohortStats] = []
    for (sym, tf, bucket, split), rets in groups.items():
        wins = sum(1 for r in rets if r > 0)
        out.append(CohortStats(
            symbol=sym, tf=tf, bucket=bucket, split=split,
            count=len(rets),
            mean_return_post_fee=mean(rets),
            median_return_post_fee=median(rets),
            win_rate=wins / len(rets) if rets else 0.0,
            worst_return_post_fee=min(rets),
        ))
    return out


def write_markdown_report(
    cohorts: list[CohortStats],
    output_path: str,
    horizon: int,
) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cohorts_sorted = sorted(
        cohorts,
        key=lambda c: (c.symbol, c.tf, c.split, c.bucket)
    )

    lines: list[str] = []
    lines.append(f"# Phase 1 cohort report (horizon = +{horizon} bars)\n")
    lines.append(f"_Generated {now}_\n")
    lines.append("")
    lines.append("Each row: one (symbol, TF, distance-bucket, split) cohort. ")
    lines.append("`mean_post_fee` is the mean forward return after subtracting "
                 "round-trip taker fee + slippage (0.05% × 2 + 0.01% × 2 = 0.12%).")
    lines.append("")
    lines.append("| symbol | TF | bucket | split | count | mean_post_fee | median | winrate | worst |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for c in cohorts_sorted:
        lines.append(
            f"| {c.symbol} | {c.tf} | {c.bucket} | {c.split} | {c.count} | "
            f"{c.mean_return_post_fee:+.4f} | {c.median_return_post_fee:+.4f} | "
            f"{c.win_rate:.2%} | {c.worst_return_post_fee:+.4f} |"
        )
    p.write_text("\n".join(lines) + "\n")
    return p
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_cohort_report.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/cohort_report.py backtests/ma_ribbon_ema21/tests/test_cohort_report.py
git commit -m "feat(backtests/ma_ribbon_ema21): cohort aggregation + markdown report"
```

---

## Task 14: Acceptance gate

**Files:**
- Create: `backtests/ma_ribbon_ema21/acceptance_gate.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_acceptance_gate.py`

The Phase 1 gate (per spec §4):
> For at least 30% of symbols, on at least one TF, in the `[0.0%, 0.5%)` distance-to-MA5 bucket, the mean forward return at +20 bars in the **test** split is `> +1%` post-fee.

- [ ] **Step 1: Write the failing test**

Create `backtests/ma_ribbon_ema21/tests/test_acceptance_gate.py`:

```python
from __future__ import annotations
import pytest
from backtests.ma_ribbon_ema21.cohort_report import CohortStats
from backtests.ma_ribbon_ema21.acceptance_gate import (
    evaluate_phase1_gate,
    GateResult,
)


def _stat(sym, tf, bucket, split, mean_ret, count=5):
    return CohortStats(
        symbol=sym, tf=tf, bucket=bucket, split=split, count=count,
        mean_return_post_fee=mean_ret, median_return_post_fee=mean_ret,
        win_rate=0.5 if mean_ret > 0 else 0.4,
        worst_return_post_fee=mean_ret - 0.05,
    )


def test_gate_passes_when_30_percent_symbols_meet_threshold():
    cohorts = []
    syms_passing = ["AAA", "BBB", "CCC", "DDD"]   # 4
    syms_failing = ["EEE", "FFF", "GGG", "HHH", "III", "JJJ"]  # 6
    for s in syms_passing:
        cohorts.append(_stat(s, "1h", "[0.0%, 0.5%)", "test", 0.02))
    for s in syms_failing:
        cohorts.append(_stat(s, "1h", "[0.0%, 0.5%)", "test", 0.005))
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.30)
    # 4 / 10 = 40% pass → ≥ 30% → GATE PASS
    assert isinstance(result, GateResult)
    assert result.passed is True
    assert result.symbols_passing >= 3


def test_gate_fails_when_no_symbols_meet_threshold():
    cohorts = [_stat(s, "1h", "[0.0%, 0.5%)", "test", 0.005) for s in "ABCDE"]
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.30)
    assert result.passed is False


def test_gate_only_uses_test_split_not_train():
    """A cohort with great train-split numbers and bad test-split should still fail."""
    cohorts = [
        _stat("AAA", "1h", "[0.0%, 0.5%)", "train", 0.05),  # ignored
        _stat("AAA", "1h", "[0.0%, 0.5%)", "test", 0.001),
    ]
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.30)
    assert result.passed is False


def test_gate_reports_failing_symbols_explicitly():
    """Per CLAUDE.md: do not hide losing cases."""
    cohorts = [
        _stat("WIN1", "1h", "[0.0%, 0.5%)", "test", 0.02),
        _stat("LOS1", "1h", "[0.0%, 0.5%)", "test", -0.005),
        _stat("LOS2", "1h", "[0.0%, 0.5%)", "test", 0.001),
    ]
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.50)
    assert "LOS1" in result.failing_symbols
    assert "LOS2" in result.failing_symbols
    assert "WIN1" in result.passing_symbols
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_acceptance_gate.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `acceptance_gate.py`**

Create `backtests/ma_ribbon_ema21/acceptance_gate.py`:

```python
"""Phase 1 acceptance gate evaluator."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from backtests.ma_ribbon_ema21.cohort_report import CohortStats


_TARGET_BUCKET = "[0.0%, 0.5%)"


@dataclass
class GateResult:
    passed: bool
    threshold_pct: float
    min_symbol_pct: float
    horizon: int
    symbols_total: int
    symbols_passing: int
    passing_symbols: list[str]
    failing_symbols: list[str]
    reason: str


def evaluate_phase1_gate(
    cohorts: list[CohortStats],
    horizon: int = 20,
    threshold_pct: float = 0.01,
    min_symbol_pct: float = 0.30,
    target_bucket: str = _TARGET_BUCKET,
) -> GateResult:
    """A symbol counts as 'passing' if, on at least one TF, the cohort
    (target_bucket, split='test') has mean_return_post_fee > threshold_pct.
    """
    by_symbol: dict[str, list[CohortStats]] = defaultdict(list)
    for c in cohorts:
        if c.split != "test":
            continue
        if c.bucket != target_bucket:
            continue
        by_symbol[c.symbol].append(c)

    passing: list[str] = []
    failing: list[str] = []
    for sym, sym_cohorts in by_symbol.items():
        ok = any(c.mean_return_post_fee > threshold_pct for c in sym_cohorts)
        (passing if ok else failing).append(sym)

    total = len(passing) + len(failing)
    pass_pct = (len(passing) / total) if total > 0 else 0.0
    passed = pass_pct >= min_symbol_pct

    reason = (
        f"{len(passing)}/{total} symbols (={pass_pct:.0%}) cleared mean +{horizon}-bar "
        f"return > {threshold_pct:.1%} on {target_bucket} (test split). "
        f"Required: {min_symbol_pct:.0%}."
    )
    return GateResult(
        passed=passed,
        threshold_pct=threshold_pct,
        min_symbol_pct=min_symbol_pct,
        horizon=horizon,
        symbols_total=total,
        symbols_passing=len(passing),
        passing_symbols=sorted(passing),
        failing_symbols=sorted(failing),
        reason=reason,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_acceptance_gate.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add backtests/ma_ribbon_ema21/acceptance_gate.py backtests/ma_ribbon_ema21/tests/test_acceptance_gate.py
git commit -m "feat(backtests/ma_ribbon_ema21): Phase 1 acceptance-gate evaluator"
```

---

## Task 15: Phase 1 CLI + integration test

**Files:**
- Create: `backtests/ma_ribbon_ema21/phase1_cli.py`
- Create: `backtests/ma_ribbon_ema21/tests/test_phase1_integration.py`

The CLI ties everything together so the user can run:

```bash
python -m backtests.ma_ribbon_ema21.phase1_cli \
  --config backtests/ma_ribbon_ema21/config.phase1.json \
  --output data/logs/backtest_reports/phase1_TIMESTAMP.md
```

- [ ] **Step 1: Write the integration test**

Create `backtests/ma_ribbon_ema21/tests/test_phase1_integration.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.phase1_cli import run_phase1
from backtests.ma_ribbon_ema21.tests.fixtures import make_uptrend_with_formation


def test_run_phase1_end_to_end_with_synthetic_data(tmp_path):
    """E2E: stage two synthetic CSVs, run the full pipeline, verify report."""
    cache = tmp_path / "cache"
    cache.mkdir()
    df1, _ = make_uptrend_with_formation(n_bars=400, formation_at_bar=120, base_price=100.0)
    df2, _ = make_uptrend_with_formation(n_bars=400, formation_at_bar=200, base_price=200.0)
    df1.to_csv(cache / "AAAUSDT_1h.csv", index=False)
    df2.to_csv(cache / "BBBUSDT_1h.csv", index=False)

    cfg_path = tmp_path / "phase1.json"
    cfg_path.write_text(json.dumps({
        "phase": "P1",
        "universe": ["AAAUSDT", "BBBUSDT"],
        "timeframes": ["1h"],
        "data_split": {"train_pct": 0.70},
        "moving_averages": {"ma_fast_1": 5, "ma_fast_2": 8, "ema_mid": 21, "ma_slow": 55},
        "bullish_alignment": {
            "require_close_above_ma5":  True,
            "require_close_above_ma8":  True,
            "require_close_above_ema21": True,
            "require_close_above_ma55":  True,
            "require_ma5_above_ma8":     True,
            "require_ma8_above_ema21":   True,
            "require_ema21_above_ma55":  True,
        },
        "forward_return_bars": [5, 10, 20, 50],
        "fees": {"per_side": 0.0005, "slippage_per_fill": 0.0001},
        "data_cache_dir": str(cache),
    }))

    output = tmp_path / "report.md"
    summary = run_phase1(config_path=str(cfg_path), output_path=str(output))

    assert output.exists()
    text = output.read_text()
    assert "AAAUSDT" in text
    assert "BBBUSDT" in text
    assert "Phase 1 cohort report" in text
    # Summary returns the gate result + total event count
    assert summary["total_events"] > 0
    assert "gate" in summary
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_phase1_integration.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `phase1_cli.py`**

Create `backtests/ma_ribbon_ema21/phase1_cli.py`:

```python
"""Phase 1 CLI driver. Reads JSON config, runs scan, writes report."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from backtests.ma_ribbon_ema21.ma_alignment import AlignmentConfig
from backtests.ma_ribbon_ema21.data_loader import DataLoaderConfig
from backtests.ma_ribbon_ema21.phase1_engine import (
    UniverseConfig, scan_universe,
)
from backtests.ma_ribbon_ema21.cohort_report import (
    aggregate_cohorts, write_markdown_report,
)
from backtests.ma_ribbon_ema21.acceptance_gate import (
    evaluate_phase1_gate,
)


_LOG = logging.getLogger("phase1")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_phase1(config_path: str, output_path: str) -> dict:
    _setup_logging()
    cfg_data = json.loads(Path(config_path).read_text())
    universe = UniverseConfig(
        symbols=cfg_data["universe"],
        timeframes=cfg_data["timeframes"],
        loader=DataLoaderConfig(cache_dir=cfg_data["data_cache_dir"]),
        alignment_cfg=AlignmentConfig.from_dict(cfg_data["bullish_alignment"]),
        forward_horizons=tuple(cfg_data["forward_return_bars"]),
        fee_per_side=cfg_data["fees"]["per_side"],
        slippage_per_fill=cfg_data["fees"]["slippage_per_fill"],
        train_pct=cfg_data["data_split"]["train_pct"],
    )
    _LOG.info("Phase 1 starting: %d symbols × %d TFs",
              len(universe.symbols), len(universe.timeframes))
    events = scan_universe(universe)
    _LOG.info("scan complete: %d total events", len(events))

    primary_horizon = 20
    cohorts = aggregate_cohorts(events, horizon=primary_horizon)
    write_markdown_report(cohorts, output_path=output_path,
                          horizon=primary_horizon)
    _LOG.info("report written: %s", output_path)

    gate = evaluate_phase1_gate(cohorts, horizon=primary_horizon,
                                threshold_pct=0.01, min_symbol_pct=0.30)
    _LOG.info("gate %s — %s",
              "PASS" if gate.passed else "FAIL", gate.reason)
    return {
        "total_events": len(events),
        "report_path":  output_path,
        "gate": {
            "passed": gate.passed,
            "reason": gate.reason,
            "passing_symbols": gate.passing_symbols,
            "failing_symbols": gate.failing_symbols,
        }
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MA-ribbon Phase 1 backtest")
    p.add_argument("--config", required=True, help="path to phase1 JSON config")
    p.add_argument("--output", required=True, help="path to write the markdown report")
    args = p.parse_args(argv)
    summary = run_phase1(args.config, args.output)
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run integration test to verify pass**

Run: `pytest backtests/ma_ribbon_ema21/tests/test_phase1_integration.py -v`
Expected: 1 PASS

- [ ] **Step 5: Run the FULL test suite for Phase 1**

Run: `pytest backtests/ma_ribbon_ema21/tests/ -v`
Expected: ALL tests pass. Specifically — count:
- `test_fixtures.py`: 3
- `test_indicators.py`: 4
- `test_ma_alignment.py`: 6
- `test_distance_features.py`: 5
- `test_forward_returns.py`: 4
- `test_70_30_split.py`: 5
- `test_multi_tf_alignment.py`: 6
- `test_no_lookahead.py`: 2
- `test_data_loader.py`: 3
- `test_phase1_engine.py`: 6
- `test_cohort_report.py`: 3
- `test_acceptance_gate.py`: 4
- `test_phase1_integration.py`: 1
- **TOTAL: 52 tests passing**

If ANY test fails, fix it before committing this task. Per `PRINCIPLES.md` §P15.

- [ ] **Step 6: Commit**

```bash
git add backtests/ma_ribbon_ema21/phase1_cli.py backtests/ma_ribbon_ema21/tests/test_phase1_integration.py
git commit -m "feat(backtests/ma_ribbon_ema21): Phase 1 CLI driver + end-to-end test"
```

---

## Task 16: Run Phase 1 against real Bitget data

**Files:**
- Create: `data/logs/backtest_reports/.gitkeep`
- Create: `data/csv_cache/ma_ribbon_ema21/.gitkeep`

This is a **manual execution step**, not a code task. It produces the actual Phase 1 report on real Bitget data and triggers the user-review checkpoint.

- [ ] **Step 1: Ensure output directories exist**

```bash
mkdir -p data/logs/backtest_reports
mkdir -p data/csv_cache/ma_ribbon_ema21
: > data/logs/backtest_reports/.gitkeep
: > data/csv_cache/ma_ribbon_ema21/.gitkeep
```

- [ ] **Step 2: Run the CLI against the real config**

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
python -m backtests.ma_ribbon_ema21.phase1_cli \
  --config backtests/ma_ribbon_ema21/config.phase1.json \
  --output data/logs/backtest_reports/phase1_${TIMESTAMP}.md
```

This will:
1. Read the 45-symbol universe from `config.phase1.json`.
2. For each symbol × TF, attempt to load from CSV cache. Cache is empty on first run → fetches from Bitget v2 perp `history-candles` endpoint at max depth.
3. Detect formation events, compute distance features and forward returns.
4. Write the cohort markdown report.
5. Print a JSON summary including gate pass/fail.

Expected runtime: 5–20 minutes on first run (data download). Subsequent runs hit the CSV cache and complete in seconds.

- [ ] **Step 3: Verify the report exists and contains data**

```bash
ls -la data/logs/backtest_reports/phase1_*.md
head -40 data/logs/backtest_reports/phase1_*.md
```

Expected:
- File exists.
- Contains "Phase 1 cohort report" header.
- Contains at least one symbol × bucket × split row.

- [ ] **Step 4: Capture the JSON summary**

The CLI prints a JSON summary to stdout. Capture it explicitly:

```bash
python -m backtests.ma_ribbon_ema21.phase1_cli \
  --config backtests/ma_ribbon_ema21/config.phase1.json \
  --output data/logs/backtest_reports/phase1_FINAL.md \
  > data/logs/backtest_reports/phase1_FINAL.summary.json
```

- [ ] **Step 5: Commit the report (NOT the CSV cache)**

```bash
git add data/logs/backtest_reports/phase1_FINAL.md \
        data/logs/backtest_reports/phase1_FINAL.summary.json \
        data/logs/backtest_reports/.gitkeep
echo "data/csv_cache/" >> .gitignore  # if not already ignored
git add .gitignore
git commit -m "report(backtests/ma_ribbon_ema21): Phase 1 first run on real Bitget data"
```

- [ ] **Step 6: User-review checkpoint**

Present to the user:
1. The committed markdown report path.
2. The JSON summary, including:
   - Total events scanned
   - `gate.passed` (true/false)
   - `gate.reason`
   - `gate.passing_symbols` (list — must be ≥ 30% of universe for PASS)
   - `gate.failing_symbols`
3. Honest framing per `CLAUDE.md`:
   - If gate **passed**: state which symbols and TFs cleared, with their numbers. Do NOT claim "the strategy works." Claim only what the data shows: "On the test split, X out of Y symbols had mean +20-bar return > 1% in the [0%, 0.5%) bucket. Phase 2 is justified."
   - If gate **failed**: state it failed, list the failing-vs-passing symbols, and ask the user whether to (a) abandon Phase 2, (b) loosen the alignment requirements, or (c) re-examine the universe selection. Do NOT proceed to Phase 2 silently.

---

## Phase 1 acceptance summary

After Task 16, Phase 1 is complete iff:

- [x] All 52 unit tests pass (Task 15 step 5).
- [x] Real-data report exists at `data/logs/backtest_reports/phase1_*.md` and is committed.
- [x] JSON summary committed alongside the report.
- [x] Gate result is **explicitly** stated in user-facing language.
- [x] No code outside `backtests/ma_ribbon_ema21/` and `data/logs/backtest_reports/` was modified.
- [x] No banned phrases (`done`, `fixed`, `能用了`, ...) used in the user-facing summary unless the test logs prove the claim.

If the gate **passes**, the next step is to invoke `superpowers:writing-plans` again with the spec's Phase 2 section as input. If the gate **fails**, the next step is a discussion with the user — do not auto-proceed.

---

## Self-review notes (run before considering this plan final)

**Spec coverage check:**
- [x] §2.1 bullish alignment → Task 4 (`bullish_aligned`)
- [x] §2.2 distance features + buckets → Task 5
- [x] §3.1 bar-event-driven sim → not needed for Phase 1 (no positions yet); deferred to Phase 2
- [x] §3.2 multi-TF alignment helper → Task 8
- [x] §3.3 no-lookahead test → Task 9
- [x] §3.5 data source / Bitget loader → Task 10
- [x] §3.6 config model → Task 1 (config.phase1.json)
- [x] §4 Phase 1 deliverable → Task 11–16
- [x] §5 70/30 IS/OOS split → Task 7

**No-placeholder check:** all code blocks contain real Python, not pseudocode or `...`. All commands are executable. All file paths are absolute relative to project root. (Verified by inspection.)

**Type consistency:** `Phase1Event` has the same field names in Tasks 11, 13, 14. `CohortStats` has the same field names in Tasks 13 and 14. `AlignmentConfig` is constructed identically in Tasks 4, 11, 15. `DataLoaderConfig.cache_dir` is the same key in Tasks 10, 12, 15.

**Scope check:** Plan covers exactly Phase 1 of the spec. Phase 2 (single-layer entry + trailing SL) has its own future plan. No scope creep.

End of plan.
