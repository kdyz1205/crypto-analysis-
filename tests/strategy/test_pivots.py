import pandas as pd

from server.strategy.config import StrategyConfig
from server.strategy.pivots import detect_pivots


def test_detect_pivots_respects_confirmation_delay() -> None:
    candles = pd.DataFrame(
        {
            "timestamp": list(range(8)),
            "open": [1.0, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08],
            "high": [1.05, 1.08, 1.12, 1.25, 1.13, 1.10, 1.24, 1.30],
            "low": [0.95, 0.96, 0.97, 0.98, 0.97, 0.96, 0.95, 0.94],
            "close": [1.02, 1.04, 1.06, 1.12, 1.08, 1.07, 1.09, 1.10],
            "volume": [100] * 8,
        }
    )

    config = StrategyConfig(pivot_left=2, pivot_right=2)
    pivots = detect_pivots(candles, config)

    high_pivots = [pivot for pivot in pivots if pivot.kind == "high"]

    assert [pivot.index for pivot in high_pivots] == [3]
    assert high_pivots[0].confirmed_at_index == 5
