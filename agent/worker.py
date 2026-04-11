"""Agent Research Worker — continuous strategy discovery loop.

Uses ONLY tools/ interfaces. Never touches raw data directly.

Loop:
1. Get market data (tools/market)
2. Scan structure (tools/structure)
3. Read factor pool (tools/factors)
4. Generate strategy drafts (tools/strategy)
5. Run simulations (tools/backtest)
6. Update leaderboard (tools/ranking)
7. Log everything
8. Sleep → repeat
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
from pathlib import Path
from dataclasses import asdict

from .config import *

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [agent] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("agent")


async def research_loop():
    """Main research loop. Runs continuously."""
    from tools.strategy import generate_ai_strategy, list_drafts, update_draft_status
    from tools.backtest import run_simulation
    from tools.ranking import update_leaderboard, get_leaderboard
    from tools.factors import list_factors
    from tools.types import SimulationJob

    generation = 0
    log.info(f"Research agent started. symbols={SYMBOLS} tfs={TIMEFRAMES}")
    log.info(f"Config: {STRATEGIES_PER_GENERATION} strategies/gen, {LOOP_INTERVAL_SECONDS}s interval")

    while generation < MAX_GENERATIONS:
        generation += 1
        gen_start = time.time()
        log.info(f"=== Generation {generation} ===")

        try:
            # Step 1: Check factor pool
            core_factors = list_factors("core")
            candidate_factors = list_factors("candidate")
            log.info(f"Factors: {len(core_factors)} core, {len(candidate_factors)} candidate")

            # Step 2: Generate strategy drafts
            drafts = []
            for _ in range(STRATEGIES_PER_GENERATION):
                draft = generate_ai_strategy(symbols=SYMBOLS, timeframes=TIMEFRAMES)
                if "error" not in draft:
                    drafts.append(draft)
            log.info(f"Generated {len(drafts)} strategy drafts")

            # Step 3: Create simulation job
            strategy_ids = [d["id"] for d in drafts]
            job = SimulationJob(
                name=f"Gen-{generation}",
                strategy_ids=strategy_ids,
                capital=10000.0,
            )
            log.info(f"Simulation job: {job.name} with {len(strategy_ids)} strategies")

            # Step 4: Run simulation
            completed_job = await run_simulation(job)
            log.info(f"Simulation complete: {len(completed_job.results)} results")

            # Step 5: Update leaderboard
            from tools.types import SimulationResult
            results_dir = Path(__file__).parent.parent / "data" / "strategies" / "simulated"
            sim_results = []
            for rid in completed_job.results:
                rpath = results_dir / f"{rid}.json"
                if rpath.exists():
                    sim_results.append(SimulationResult(**json.loads(rpath.read_text(encoding="utf-8"))))

            strategy_meta = {d["id"]: d for d in drafts}
            lb = update_leaderboard(sim_results, strategy_meta)
            log.info(f"Leaderboard: {len(lb)} entries")

            # Step 6: Log top performers
            top = lb[:3] if lb else []
            for i, e in enumerate(top):
                log.info(f"  #{i+1} {e.get('symbol','')} {e.get('timeframe','')} | "
                         f"ret={e.get('return_pct',0):+.1f}% WR={e.get('win_rate',0)}% "
                         f"score={e.get('score',0):.3f}")

            # Step 7: Update draft statuses
            for d in drafts:
                update_draft_status(d["id"], "simulated")

            gen_time = time.time() - gen_start
            log.info(f"Generation {generation} done in {gen_time:.1f}s")

        except Exception as e:
            log.error(f"Generation {generation} failed: {e}", exc_info=True)

        # Sleep
        await asyncio.sleep(LOOP_INTERVAL_SECONDS)

    log.info("Research agent finished (max generations reached)")


def run():
    """Entry point."""
    asyncio.run(research_loop())


if __name__ == "__main__":
    run()
