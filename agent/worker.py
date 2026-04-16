"""Agent Research Worker — continuous strategy discovery loop.

Uses ONLY tools/ interfaces. Never touches raw data directly.
State persists to disk (agent/state.py). Every action audited (tools/audit).

Loop:
1. Restore state from disk
2. Get market data (tools/market)
3. Scan structure (tools/structure)
4. Read factor pool (tools/factors)
5. Generate strategy drafts (tools/strategy)
6. Run simulations (tools/backtest)
7. Update leaderboard (tools/ranking)
8. Persist state + audit
9. Sleep → repeat
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
import traceback
from pathlib import Path
from dataclasses import asdict

from .config import *
from .state import AgentState

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
    """Main research loop with state persistence and full audit trail."""
    from tools.strategy import generate_ai_strategy, list_drafts, update_draft_status
    from tools.backtest import run_simulation
    from tools.ranking import update_leaderboard, get_leaderboard
    from tools.factors import list_factors, record_factor_test
    from tools.audit import write_audit
    from tools.types import SimulationJob, SimulationResult

    # Restore state
    state = AgentState.load()
    generation = state.current_generation
    log.info(f"Research agent started. Resuming from generation {generation}.")
    log.info(f"Lifetime stats: {state.total_strategies_generated} strategies, {state.total_results_produced} results, {state.total_profitable} profitable")
    log.info(f"Config: symbols={SYMBOLS} tfs={TIMEFRAMES} {STRATEGIES_PER_GENERATION}/gen {LOOP_INTERVAL_SECONDS}s interval")

    write_audit("agent", "agent_started", "agent", "worker", {
        "resumed_generation": generation,
        "total_strategies": state.total_strategies_generated,
        "total_results": state.total_results_produced,
        "total_profitable": state.total_profitable,
    })

    while generation < MAX_GENERATIONS:
        generation += 1
        gen_start = time.time()
        batch_id = f"gen-{generation}-{int(gen_start)}"
        log.info(f"=== Generation {generation} (batch {batch_id}) ===")

        try:
            # Step 1: Check factor pool
            core_factors = list_factors("core")
            candidate_factors = list_factors("candidate")
            log.info(f"Factors: {len(core_factors)} core, {len(candidate_factors)} candidate")

            # Step 2: Generate strategy drafts
            drafts = []
            gen_errors = []
            pattern_report = None

            # Pattern-driven generation first (if enabled)
            if GENERATION_MODE in ("pattern", "mixed"):
                try:
                    from .pattern_driven import scan_and_generate
                    pattern_report = await scan_and_generate(
                        symbols=SYMBOLS,
                        timeframes=TIMEFRAMES,
                        generation=generation,
                        batch_id=batch_id,
                        max_drafts_per_symbol=PATTERN_MAX_DRAFTS_PER_SYMBOL,
                    )
                    drafts.extend(pattern_report["drafts"])
                    log.info(f"Pattern-driven: evaluated={pattern_report['evaluated']} "
                             f"approved={pattern_report['approved']} "
                             f"decisions={pattern_report['by_decision']}")
                except Exception as e:
                    log.error(f"Pattern-driven generation failed: {e}", exc_info=True)
                    gen_errors.append({"source": "pattern", "error": str(e)[:200]})

            # Random fallback (if pattern produced 0 drafts, or mode=random)
            need_random = (GENERATION_MODE == "random") or (
                GENERATION_MODE == "mixed" and len(drafts) == 0
            )
            if need_random:
                log.info(f"Random fallback: need {STRATEGIES_PER_GENERATION} drafts")
                for i in range(STRATEGIES_PER_GENERATION):
                    try:
                        draft = generate_ai_strategy(symbols=SYMBOLS, timeframes=TIMEFRAMES,
                                                       generation=generation, batch_id=batch_id)
                        if "error" in draft:
                            gen_errors.append({"index": i, "error": draft["error"]})
                        else:
                            drafts.append(draft)
                    except Exception as e:
                        gen_errors.append({"index": i, "error": str(e)[:100]})

            log.info(f"Generated {len(drafts)} drafts total ({len(gen_errors)} failed)")

            if not drafts:
                write_audit("agent", "generation_failed", "generation", batch_id, {
                    "generation": generation, "reason": "no drafts generated",
                    "errors": gen_errors[:5],
                })
                state.record_error(f"Gen {generation}: no drafts generated")
                await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                continue

            # Step 3: Create simulation job
            strategy_ids = [d["id"] for d in drafts]
            job = SimulationJob(
                name=f"Gen-{generation}",
                strategy_ids=strategy_ids,
                capital=10000.0,
                generation=generation,
                batch_id=batch_id,
            )

            # Mark state as running
            state.begin_generation(generation, job.id)

            # Step 4: Run simulation
            log.info(f"Simulation job {job.id}: {len(strategy_ids)} strategies")
            completed_job = await run_simulation(job)

            # Log failures
            if completed_job.failed_items:
                log.warning(f"Simulation failures: {len(completed_job.failed_items)}")
                for fi in completed_job.failed_items[:5]:
                    log.warning(f"  FAIL {fi.get('strategy_id','?')[:8]} {fi.get('symbol','')} "
                                f"{fi.get('timeframe','')} @ {fi.get('stage','')}: {fi.get('error','')}")

            # Step 5: Load simulation results and update leaderboard
            results_dir = Path(__file__).parent.parent / "data" / "strategies" / "simulated"
            sim_results = []
            for rid in completed_job.results:
                rpath = results_dir / f"{rid}.json"
                if rpath.exists():
                    try:
                        sim_results.append(SimulationResult(**json.loads(rpath.read_text(encoding="utf-8"))))
                    except Exception:
                        pass

            strategy_meta = {d["id"]: d for d in drafts}
            lb = update_leaderboard(sim_results, strategy_meta, generation=generation)
            log.info(f"Leaderboard: {len(lb)} entries after update")

            # Step 6: Log top performers
            top = lb[:5] if lb else []
            for i, e in enumerate(top[:3]):
                log.info(f"  #{i+1} {e.get('symbol','')} {e.get('timeframe','')} | "
                         f"ret={e.get('return_pct',0):+.1f}% WR={e.get('win_rate',0)}% "
                         f"score={e.get('score',0):.3f}")

            # Step 7: Feed results back to factor lifecycle
            factor_updates = 0
            for r in sim_results:
                draft = strategy_meta.get(r.strategy_id, {})
                factor_ids = draft.get("factor_ids", [])
                for fid in factor_ids:
                    try:
                        record_factor_test(fid, r.score, r.trade_count,
                                           symbol=r.symbol, timeframe=r.timeframe)
                        factor_updates += 1
                    except Exception:
                        pass
            if factor_updates:
                log.info(f"Factor test records updated: {factor_updates}")

            # Step 8: Update draft statuses
            for d in drafts:
                update_draft_status(d["id"], "simulated")

            # Step 9: Persist state
            profitable = sum(1 for r in sim_results if r.return_pct > 0)
            state.end_generation(
                strategy_ids=strategy_ids,
                result_ids=completed_job.results,
                top=top,
                profitable=profitable,
            )

            gen_time = time.time() - gen_start
            log.info(f"Generation {generation} done in {gen_time:.1f}s | "
                     f"{len(sim_results)} results, {profitable} profitable, "
                     f"{len(completed_job.failed_items)} failed")

            write_audit("agent", "generation_completed", "generation", batch_id, {
                "generation": generation,
                "drafts": len(drafts),
                "pattern_evaluated": pattern_report["evaluated"] if pattern_report else 0,
                "pattern_approved": pattern_report["approved"] if pattern_report else 0,
                "pattern_by_decision": pattern_report["by_decision"] if pattern_report else {},
                "results": len(sim_results),
                "profitable": profitable,
                "failed_items": len(completed_job.failed_items),
                "duration_s": round(gen_time, 1),
                "top_score": top[0].get("score", 0) if top else 0,
            })

        except Exception as e:
            err_msg = f"Generation {generation} failed: {e}"
            log.error(err_msg, exc_info=True)
            state.record_error(err_msg[:200])
            write_audit("agent", "generation_failed", "generation", batch_id, {
                "generation": generation,
                "error": str(e)[:200],
                "traceback": traceback.format_exc()[-500:],
            })

        # Sleep
        await asyncio.sleep(LOOP_INTERVAL_SECONDS)

    log.info("Research agent finished (max generations reached)")
    write_audit("agent", "agent_stopped", "agent", "worker", {
        "final_generation": generation,
        "total_strategies": state.total_strategies_generated,
        "total_results": state.total_results_produced,
        "total_profitable": state.total_profitable,
        "reason": "max_generations",
    })


def run():
    """Entry point."""
    asyncio.run(research_loop())


if __name__ == "__main__":
    run()
