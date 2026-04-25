"""Make project root importable from any test under backtests/.

Mirrors tests/conftest.py: pytest's rootdir autodetect doesn't add the project
root to sys.path on its own, so imports like
`from backtests.ma_ribbon_ema21.indicators import sma` would fail without this
shim when pytest is invoked with a backtests/* path.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
