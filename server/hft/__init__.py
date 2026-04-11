"""HFT subsystem — separate from bar-based SR strategy pipeline.

Architecture:
  data_feed/   — order book, trade stream, time sync
  strategies/  — imbalance MR, sweep breakout, inventory MM, lead-lag
  execution/   — ms-level order manager, queue tracker
  risk/        — kill switch, inventory limits, strategy caps
  router.py    — regime router (decides which strategy runs)
"""
