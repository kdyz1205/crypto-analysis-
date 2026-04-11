"""Run the autonomous research agent.

Usage: python run_agent.py

The agent:
1. Generates strategy drafts (AI + template)
2. Runs backtests via tools/backtest
3. Updates leaderboard via tools/ranking
4. Logs everything to data/logs/agent.log
5. Runs continuously until stopped

It NEVER directly trades. It ONLY researches.
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass

from agent.worker import run

if __name__ == "__main__":
    run()
