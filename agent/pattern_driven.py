"""Pattern-driven strategy generation for the agent worker.

Instead of randomly combining factors, scan live market data for 2-touch
structures, query the pattern engine for evidence, and only generate
strategies that are approved by the decision layer.

This turns the agent from "random factor combinator" into "evidence-driven
hypothesis generator".
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

log = logging.getLogger("agent.pattern")


async def scan_and_generate(
    symbols: list[str],
    timeframes: list[str],
    generation: int,
    batch_id: str,
    max_drafts_per_symbol: int = 2,
) -> dict:
    """Scan live market data for 2-touch structures and generate approved drafts.

    Returns:
      {
        "drafts": [draft_dict, ...],   # approved + created StrategyDrafts
        "evaluated": int,               # total patterns evaluated
        "approved": int,                # how many got past decision layer
        "rejected": int,                # no_trade + watch_only
        "by_decision": {...},           # count per decision type
        "errors": [...],
      }
    """
    from tools.pattern_engine import (
        extract_features, load_patterns, match_pattern, scan_historical_patterns,
        append_pattern_record,
    )
    from tools.pattern_strategy import (
        decide_strategy_type, generate_strategy_configs, validate_generated_config
    )
    from tools.strategy import create_from_config
    from tools.audit import write_audit
    from server.data_service import get_ohlcv_with_df
    import pandas as pd

    drafts = []
    evaluated = 0
    approved = 0
    by_decision: dict[str, int] = {}
    errors = []

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                # Load live market data
                df_polars, _ = await get_ohlcv_with_df(symbol, timeframe, days=120)
                if df_polars is None or df_polars.is_empty():
                    errors.append({"symbol": symbol, "tf": timeframe, "err": "no data"})
                    continue

                pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
                pdf = pdf.rename(columns={"open_time": "timestamp"})
                for col in ("open", "high", "low", "close", "volume"):
                    pdf[col] = pd.to_numeric(pdf[col], errors="raise")

                # Check if pattern database exists for this pair
                db = load_patterns(symbol, timeframe)
                if not db:
                    log.info(f"[pattern] no database for {symbol}_{timeframe} — skipping")
                    continue

                # Scan the MOST RECENT bars for 2-touch candidates
                # Use a smaller lookback window — we only care about current active structures
                recent_patterns = scan_historical_patterns(
                    pdf.iloc[-250:].reset_index(drop=True),
                    symbol, timeframe,
                    pivot_window=3,
                    max_anchor_distance=80,
                    lookahead_bars=20,
                )

                if not recent_patterns:
                    continue

                # Keep patterns where anchor2 is near the current bar (active structures)
                total_bars = 250
                active_patterns = [p for p in recent_patterns if p.anchor2_idx >= total_bars - 60]
                log.info(f"[pattern] {symbol}_{timeframe}: {len(active_patterns)} active 2-touch structures")

                # Evaluate each active pattern through the pattern engine
                symbol_drafts = 0
                for p in active_patterns:
                    if symbol_drafts >= max_drafts_per_symbol:
                        break
                    evaluated += 1
                    try:
                        # Live scan: current_time_position=1.0 means "now",
                        # so find_similar's leak guard drops anything with
                        # time_position > 1.0 (future writebacks). Without
                        # this we'd match patterns written back AFTER this
                        # call ran — Round 9/10 bug #4.
                        match = match_pattern(
                            pdf.iloc[-250:].reset_index(drop=True),
                            p.anchor1_idx, p.anchor1_price,
                            p.anchor2_idx, p.anchor2_price,
                            p.features.side, symbol, timeframe, k=30,
                            current_time_position=1.0,
                        )
                        decision = decide_strategy_type(match)
                        dec_type = decision["decision"]
                        by_decision[dec_type] = by_decision.get(dec_type, 0) + 1

                        # Persist the live pattern so writeback can find it later
                        p.split_bucket = "live_scan"
                        p.time_position = 1.0
                        append_pattern_record(symbol, timeframe, p)

                        # Only generate for actionable decisions
                        if dec_type in ("reversal", "breakout", "failed_breakout"):
                            configs = generate_strategy_configs(
                                decision, match, symbol, timeframe, both_variants=False
                            )
                            for cfg in configs:
                                validation = validate_generated_config(cfg, match)
                                if not validation["ok"]:
                                    continue
                                cfg_name = f"{symbol}-{dec_type}-G{generation}"
                                # Pass full lineage to create_from_config
                                lineage = {
                                    "source_pattern_id": p.pattern_id,
                                    "decision_rule": decision.get("rule_id", ""),
                                    "pattern_decision": dec_type,
                                    "pattern_ev": match["stats"].get("expected_value", 0.0),
                                    "pattern_confidence": match["stats"].get("confidence", 0.0),
                                    "pattern_reason": decision.get("reason", ""),
                                    "generation": generation,
                                    "batch_id": batch_id,
                                }
                                result = create_from_config(
                                    cfg_name, cfg["config"],
                                    source="pattern_engine",
                                    lineage=lineage,
                                )
                                drafts.append(result)
                                approved += 1
                                symbol_drafts += 1

                                write_audit("agent", "pattern_driven_draft", "strategy", result["id"], {
                                    "symbol": symbol, "timeframe": timeframe,
                                    "decision": dec_type,
                                    "rule_id": decision.get("rule_id", ""),
                                    "source_pattern_id": p.pattern_id,
                                    "ev": match["stats"].get("expected_value"),
                                    "confidence": match["stats"].get("confidence"),
                                    "reason": decision.get("reason"),
                                })
                    except Exception as e:
                        errors.append({"symbol": symbol, "tf": timeframe, "err": str(e)[:100]})
            except Exception as e:
                errors.append({"symbol": symbol, "tf": timeframe, "err": str(e)[:100]})

    log.info(f"[pattern] evaluated={evaluated} approved={approved} by_decision={by_decision}")

    return {
        "drafts": drafts,
        "evaluated": evaluated,
        "approved": approved,
        "rejected": evaluated - approved,
        "by_decision": by_decision,
        "errors": errors,
    }
