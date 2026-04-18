from __future__ import annotations

import pandas as pd

from server.strategy.drawing_learner import _compute_line_interaction_features


def test_line_interaction_features_count_touches_and_rejections() -> None:
    timestamps = [1_700_000_000 + i * 3600 for i in range(10)]
    rows = []
    for i, ts in enumerate(timestamps):
        line = 100.0 + i
        low = line + 1.0
        high = line + 3.0
        close = line + 2.0
        if i in {2, 5, 8}:
            low = line - 0.02
            high = line + 2.0
            close = line + 1.2
        rows.append({
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        })
    df = pd.DataFrame(rows)

    features = _compute_line_interaction_features(
        df=df,
        timeframe="1h",
        side="support",
        price_start=100.0,
        price_end=109.0,
        t_start=timestamps[0],
        t_end=timestamps[-1],
        slope_per_bar=1.0,
    )

    assert features["touch_count"] == 3
    assert features["wick_rejection_count"] == 3
    assert features["body_violation_count"] == 0
    assert features["last_touch_age_bars"] == 1
    assert features["avg_rejection_atr"] > 0
