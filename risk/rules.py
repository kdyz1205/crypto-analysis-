"""Hard risk rules — NOT modifiable by agent. System enforced."""

# Per-strategy limits
MAX_STRATEGY_LOSS_PCT = 20.0       # Kill strategy if it loses 20% of allocated capital
MAX_STRATEGY_DRAWDOWN_PCT = 15.0   # Kill if drawdown exceeds 15%
MAX_CONSECUTIVE_LOSSES = 8         # Kill after 8 consecutive losses

# Portfolio limits
MAX_AUTO_DEPLOY_CAPITAL = 50.0     # Agent can auto-deploy max $50 total
MAX_SINGLE_STRATEGY_CAPITAL = 20.0 # Single strategy max $20
MAX_CONCURRENT_LIVE = 5            # Max 5 live strategies at once
MAX_TOTAL_RISK_PCT = 10.0          # Total risk exposure max 10% of portfolio

# API safety
MAX_API_LATENCY_MS = 5000          # Kill if API > 5 seconds
MAX_DATA_GAP_SECONDS = 60          # Stop if no data for 60 seconds
MAX_ORDER_RATE_PER_MINUTE = 10     # Max 10 orders per minute

# Factor safety
MIN_BACKTEST_TRADES = 10           # Factor needs 10+ trades to be validated
MIN_BACKTEST_SCORE = 0.3           # Minimum composite score for validation
CANDIDATE_TO_VALIDATED_MIN_TESTS = 3  # Test on 3+ coin/TF combos before promotion
