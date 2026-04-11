"""Agent configuration."""

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]
TIMEFRAMES = ["1h", "4h"]
STRATEGIES_PER_GENERATION = 5
LOOP_INTERVAL_SECONDS = 120  # 2 minutes between generations
MAX_GENERATIONS = 1000
AUTO_DEPLOY_ENABLED = False  # First stage: research only, no auto-deploy
AUTO_DEPLOY_MAX_CAPITAL = 10.0  # Max $10 per auto-deployed strategy
