"""Agent configuration."""

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]
TIMEFRAMES = ["1h", "4h"]
STRATEGIES_PER_GENERATION = 5
LOOP_INTERVAL_SECONDS = 120  # 2 minutes between generations
MAX_GENERATIONS = 1000
AUTO_DEPLOY_ENABLED = False  # First stage: research only, no auto-deploy
AUTO_DEPLOY_MAX_CAPITAL = 10.0  # Max $10 per auto-deployed strategy

# Generation mode:
#   "pattern"  — scan live 2-touch lines, only generate pattern-engine-approved drafts
#   "random"   — original behavior, random factor combinations
#   "mixed"    — try pattern first, fall back to random if 0 drafts produced
GENERATION_MODE = "mixed"
PATTERN_MAX_DRAFTS_PER_SYMBOL = 2
