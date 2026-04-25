"""Self-evolving trendline agent.

Continuously:
  1. Watches for new manual lines + outcomes
  2. Retrains when enough new data accumulates
  3. Auto-draws new lines on recent OHLCV in the user's style
  4. Backtests the auto-drawn lines
  5. Reports findings to a JSONL log
  6. Sleeps and repeats

This is the production end-state: the model trains itself on the user's
drawing style, generates more lines like the user would, validates them
against historical data, and reports what works.
"""
