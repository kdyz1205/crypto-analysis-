"""Backtest / research console: replay historical bars through the
inference pipeline, simulate paper PnL, compute metrics, run ablations.

This is the *truth* layer: without backtest results that beat a
sensible baseline, no inference run should be wired to live trading.
"""
