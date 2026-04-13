"""Evolved trendline detection variants.

Each variant implements the TrendlineVariant interface and is tested
via the fade+flip backtest harness. The winner after N evolution rounds
replaces nothing — it gets selected via feature flag at query time.

Old trendlines.py is untouched.
"""
