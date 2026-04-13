"""Self-evolving trendline detection loop.

Architecture:
- base.py               — TrendlineVariant interface (imported from server.strategy.evolved)
- harness.py            — fade+flip backtest: lines → trades → metrics
- evaluator.py          — runs variant across symbols × timeframes, returns fitness
- orchestrator.py       — meta-loop: baseline → architect → variants → winner → iterate
- architect_prompt.py   — template used when spawning architect subagents
"""
