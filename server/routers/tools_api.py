"""Tools-layer API — exposes real research data: leaderboard, audit, failures, factors, agent state."""

from __future__ import annotations

from fastapi import APIRouter, Query, Body
from collections import Counter

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("/leaderboard")
async def api_tools_leaderboard(limit: int = Query(20, ge=1, le=50)):
    """Research leaderboard with full traceability."""
    from tools.ranking import get_leaderboard
    from agent.state import AgentState
    state = AgentState.load()
    entries = get_leaderboard(limit)
    return {"ok": True, "error": None, "data": entries, "meta": {
        "generation": state.current_generation,
        "total_strategies": state.total_strategies_generated,
        "total_results": state.total_results_produced,
        "total_profitable": state.total_profitable,
    }}


@router.get("/leaderboard/{entry_id}")
async def api_tools_leaderboard_entry(entry_id: str):
    """Single leaderboard entry with linked draft."""
    import json
    from pathlib import Path
    from tools.ranking import get_entry
    entry = get_entry(entry_id)
    if not entry:
        return {"ok": False, "error": "not found", "data": None}
    draft_path = Path("data/strategies/drafts") / f"{entry.get('strategy_id', '')}.json"
    draft = None
    if draft_path.exists():
        try:
            draft = json.loads(draft_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"ok": True, "error": None, "data": {"entry": entry, "draft": draft}}


@router.get("/audit")
async def api_tools_audit(day: str = None, limit: int = Query(100, ge=1, le=500)):
    """Structured audit log."""
    from tools.audit import read_audit
    entries = read_audit(day=day, limit=limit)
    return {"ok": True, "error": None, "data": entries, "meta": {"day": day or "today", "count": len(entries)}}


@router.get("/failures")
async def api_tools_failures(generation: int = None, limit: int = Query(50, ge=1, le=200)):
    """Backtest failure records."""
    from tools.backtest import list_failures
    failures = list_failures(generation=generation, limit=limit)
    return {"ok": True, "error": None, "data": failures, "meta": {"count": len(failures), "generation_filter": generation}}


@router.get("/factors")
async def api_tools_factors(stage: str = Query("core")):
    """Factors by stage with test history."""
    from tools.factors import list_factors
    factors = list_factors(stage)
    tested = sum(1 for f in factors if f.get("test_count", 0) > 0)
    return {"ok": True, "error": None, "data": factors, "meta": {"stage": stage, "count": len(factors), "tested": tested}}


@router.get("/factors/{factor_id}")
async def api_tools_factor_detail(factor_id: str):
    """Single factor with full test history."""
    from tools.factors import _load_factor
    factor = _load_factor(factor_id)
    if not factor:
        return {"ok": False, "error": "not found", "data": None}
    return {"ok": True, "error": None, "data": factor}


@router.get("/agent-state")
async def api_tools_agent_state():
    """Agent worker persistent state."""
    from agent.state import AgentState
    from dataclasses import asdict
    state = AgentState.load()
    return {"ok": True, "error": None, "data": asdict(state)}


@router.get("/drafts")
async def api_tools_drafts(status: str = None, limit: int = Query(50, ge=1, le=200)):
    """Strategy drafts."""
    from tools.strategy import list_drafts
    drafts = list_drafts(status=status)
    return {"ok": True, "error": None, "data": drafts[:limit], "meta": {"count": len(drafts[:limit]), "status_filter": status}}


@router.get("/patterns/match")
async def api_patterns_match(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    anchor1_idx: int = Query(...),
    anchor1_price: float = Query(...),
    anchor2_idx: int = Query(...),
    anchor2_price: float = Query(...),
    side: str = Query(..., regex="^(support|resistance)$"),
    k: int = Query(30, ge=5, le=200),
):
    """Given a 2-touch line, return historical probability stats.

    This is the Pattern Engine match endpoint. Input a hypothesis line,
    get back evidence from similar historical structures.
    """
    from tools.pattern_engine import match_pattern
    from server.data_service import get_ohlcv_with_df

    df_polars, _ = await get_ohlcv_with_df(symbol, timeframe, days=180)
    if df_polars is None or df_polars.is_empty():
        return {"ok": False, "error": "no market data", "data": None}

    # Convert to pandas for the pattern engine
    import pandas as pd
    pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    for col in ("open", "high", "low", "close", "volume"):
        pdf[col] = pd.to_numeric(pdf[col], errors="raise")

    try:
        result = match_pattern(
            pdf, anchor1_idx, anchor1_price, anchor2_idx, anchor2_price,
            side, symbol, timeframe, k=k,
        )
        return {"ok": True, "error": None, "data": result}
    except Exception as e:
        return {"ok": False, "error": str(e), "data": None}


@router.post("/patterns/batch-build")
async def api_patterns_batch_build(payload: dict = Body(...)):
    """Start a batch pattern database build across many symbols/timeframes.

    Body: {
      "symbols": ["BTCUSDT", ...],
      "timeframes": ["1h", "4h"],
      "days": 730
    }

    Returns immediately — progress tracked in /api/tools/patterns/batch-progress.
    """
    from tools.pattern_batch import start_batch_build
    symbols = payload.get("symbols", [])
    timeframes = payload.get("timeframes", ["4h"])
    days = payload.get("days", 730)
    if not symbols:
        return {"ok": False, "error": "symbols required", "data": None}
    result = await start_batch_build(symbols, timeframes, days=days)
    return {"ok": True, "error": None, "data": result}


@router.get("/patterns/batch-progress")
async def api_patterns_batch_progress():
    """Current progress of the batch pattern build job."""
    from tools.pattern_batch import load_progress, is_running
    progress = load_progress()
    progress["is_running"] = is_running()
    return {"ok": True, "error": None, "data": progress}


@router.post("/patterns/build-max")
async def api_patterns_build_max(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
):
    """Build pattern database using MAX available history for this symbol.

    Uses history_mode='full_history' which pulls up to ~6 years per coin.
    """
    from tools.pattern_engine import scan_historical_patterns, save_patterns
    from server.data_service import get_ohlcv_with_df
    import pandas as pd

    df_polars, _ = await get_ohlcv_with_df(symbol, timeframe, days=365, history_mode="full_history")
    if df_polars is None or df_polars.is_empty():
        return {"ok": False, "error": "no market data", "data": None}

    pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    for col in ("open", "high", "low", "close", "volume"):
        pdf[col] = pd.to_numeric(pdf[col], errors="raise")

    records = scan_historical_patterns(pdf, symbol, timeframe)
    path = save_patterns(records, symbol, timeframe)

    return {
        "ok": True, "error": None, "data": {
            "symbol": symbol, "timeframe": timeframe,
            "patterns_found": len(records),
            "bars_scanned": len(pdf),
            "saved_to": path,
        }
    }


@router.post("/patterns/build")
async def api_patterns_build(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    days: int = Query(365, ge=30, le=730),
):
    """Scan historical data and build the pattern database for a symbol/timeframe.

    Run this ONCE per symbol to populate the pattern database.
    """
    from tools.pattern_engine import scan_historical_patterns, save_patterns
    from server.data_service import get_ohlcv_with_df

    df_polars, _ = await get_ohlcv_with_df(symbol, timeframe, days=days)
    if df_polars is None or df_polars.is_empty():
        return {"ok": False, "error": "no market data", "data": None}

    import pandas as pd
    pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    for col in ("open", "high", "low", "close", "volume"):
        pdf[col] = pd.to_numeric(pdf[col], errors="raise")

    records = scan_historical_patterns(pdf, symbol, timeframe)
    path = save_patterns(records, symbol, timeframe)

    return {
        "ok": True, "error": None, "data": {
            "symbol": symbol, "timeframe": timeframe,
            "patterns_found": len(records),
            "bars_scanned": len(pdf),
            "saved_to": path,
        }
    }


@router.get("/patterns/explore")
async def api_patterns_explore(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
):
    """Find high-EV but under-sampled pattern clusters = exploration priorities.

    Use this to discover promising structures that need more data before exploitation.
    """
    from tools.pattern_engine import load_patterns, discover_exploration_targets
    database = load_patterns(symbol, timeframe)
    if not database:
        return {"ok": False, "error": "no pattern database", "data": None}
    result = discover_exploration_targets(database)
    return {"ok": True, "error": None, "data": result, "meta": {
        "symbol": symbol, "timeframe": timeframe,
        "database_size": len(database),
    }}


@router.get("/patterns/feature-weights")
async def api_patterns_feature_weights(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    target: str = Query("profitable"),
):
    """Get learned feature importance weights from historical outcomes."""
    from tools.pattern_engine import load_patterns, learn_feature_weights
    database = load_patterns(symbol, timeframe)
    if not database:
        return {"ok": False, "error": "no pattern database", "data": None}
    weights = learn_feature_weights(database, target=target)
    names = ["slope_atr", "log_length", "volatility", "trend_up", "trend_down", "trend_range", "support", "rsi", "ma_distance", "touch_quality"]
    return {"ok": True, "error": None, "data": {
        "target": target,
        "weights": [{"feature": n, "weight": round(w, 3)} for n, w in zip(names, weights)],
        "sorted_by_importance": sorted(
            [{"feature": n, "weight": round(w, 3)} for n, w in zip(names, weights)],
            key=lambda x: -x["weight"],
        ),
    }}


@router.get("/patterns/pca")
async def api_patterns_pca(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    n_components: int = Query(5, ge=2, le=10),
):
    """Get PCA decomposition of pattern database — shows principal modes of variation."""
    from tools.pattern_engine import load_patterns, fit_pca
    database = load_patterns(symbol, timeframe)
    if not database:
        return {"ok": False, "error": "no pattern database", "data": None}
    pca = fit_pca(database, n_components=n_components)
    return {"ok": True, "error": None, "data": pca}


@router.get("/patterns/clusters")
async def api_patterns_clusters(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    eps: float = Query(1.5, ge=0.1, le=10.0),
    min_samples: int = Query(10, ge=3, le=100),
):
    """Discover typical pattern clusters in the historical database.
    Each cluster represents a common structural archetype.
    """
    from tools.pattern_engine import load_patterns, find_clusters
    database = load_patterns(symbol, timeframe)
    if not database:
        return {"ok": False, "error": "no pattern database — run /patterns/build first", "data": None}
    clusters = find_clusters(database, eps=eps, min_samples=min_samples)
    return {"ok": True, "error": None, "data": clusters, "meta": {
        "symbol": symbol, "timeframe": timeframe,
        "database_size": len(database), "cluster_count": len(clusters),
    }}


@router.get("/patterns/recommend-strategies")
async def api_patterns_recommend(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    anchor1_idx: int = Query(...),
    anchor1_price: float = Query(...),
    anchor2_idx: int = Query(...),
    anchor2_price: float = Query(...),
    side: str = Query(..., pattern="^(support|resistance)$"),
    k: int = Query(30, ge=5, le=200),
    both_variants: bool = Query(True),
):
    """Pattern → Strategy auto-generator.

    Given a 2-touch line, returns:
    1. Match stats (probabilities, EV, confidence)
    2. Decision (reversal/breakout/failed_breakout/no_trade/watch_only)
    3. Auto-generated StrategyConfig drafts (validated)
    """
    from tools.pattern_engine import match_pattern
    from tools.pattern_strategy import decide_strategy_type, generate_strategy_configs, validate_generated_config
    from server.data_service import get_ohlcv_with_df
    import pandas as pd

    df_polars, _ = await get_ohlcv_with_df(symbol, timeframe, days=180)
    if df_polars is None or df_polars.is_empty():
        return {"ok": False, "error": "no market data", "data": None}

    pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    for col in ("open", "high", "low", "close", "volume"):
        pdf[col] = pd.to_numeric(pdf[col], errors="raise")

    try:
        match_result = match_pattern(
            pdf, anchor1_idx, anchor1_price, anchor2_idx, anchor2_price,
            side, symbol, timeframe, k=k,
        )
        decision = decide_strategy_type(match_result)
        drafts = generate_strategy_configs(decision, match_result, symbol, timeframe, both_variants=both_variants)

        # Validate each draft with the second-layer guard
        validated_drafts = []
        for draft in drafts:
            validation = validate_generated_config(draft, match_result)
            draft["validation"] = validation
            draft["approved"] = validation["ok"]
            validated_drafts.append(draft)

        return {"ok": True, "error": None, "data": {
            "match": match_result,
            "decision": decision,
            "drafts": validated_drafts,
            "approved_drafts": [d for d in validated_drafts if d["approved"]],
        }}
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()[-500:], "data": None}


@router.post("/patterns/create-from-recommendation")
async def api_patterns_create_from_recommendation(payload: dict = Body(...)):
    """Accept a recommendation draft → save as real StrategyDraft.

    Body: { "draft": {draft object from recommend-strategies endpoint} }
    """
    from tools.strategy import create_from_config
    draft = payload.get("draft", {})
    name = draft.get("name", "PatternGen-" + draft.get("decision_type", "unknown"))
    config = draft.get("config", {})
    if not config:
        return {"ok": False, "error": "missing config in draft", "data": None}
    result = create_from_config(name, config, source="pattern_engine")
    return {"ok": True, "error": None, "data": result}


@router.get("/patterns/live-outcomes")
async def api_patterns_live_outcomes(
    limit: int = Query(100, ge=1, le=500),
    symbol: str = Query(""),
    rule_id: str = Query(""),
):
    """List live outcome records from the closed-loop writeback system."""
    from tools.pattern_writeback import list_live_outcomes
    outcomes = list_live_outcomes(limit=limit, symbol=symbol, rule_id=rule_id)
    return {"ok": True, "error": None, "data": outcomes, "meta": {"count": len(outcomes)}}


@router.get("/patterns/live-outcomes/by-pattern/{symbol}/{timeframe}")
async def api_patterns_live_outcomes_by_pattern(symbol: str, timeframe: str):
    """Get sidecar live outcomes for a specific symbol_timeframe."""
    from tools.pattern_writeback import get_pattern_live_outcomes
    outcomes = get_pattern_live_outcomes(symbol, timeframe)
    return {"ok": True, "error": None, "data": outcomes, "meta": {"count": len(outcomes)}}


@router.get("/patterns/recompute-flags")
async def api_patterns_recompute_flags():
    """Show pattern databases that are flagged for stats recomputation."""
    from tools.pattern_writeback import get_recompute_flags
    flags = get_recompute_flags()
    return {"ok": True, "error": None, "data": flags, "meta": {"count": len(flags)}}


@router.post("/patterns/process-recompute")
async def api_patterns_process_recompute():
    """Process all pending recompute flags — batch update pattern stats with live data."""
    from tools.pattern_writeback import process_recompute_flags
    result = process_recompute_flags()
    return {"ok": True, "error": None, "data": result}


@router.get("/health-digest")
async def api_health_digest(hours: int = Query(24, ge=1, le=168)):
    """Closed-loop health digest — the one daily 'am I getting stronger?' check."""
    import asyncio
    from tools.health_digest import build_digest
    digest = await asyncio.to_thread(build_digest, hours=hours)
    return {"ok": True, "error": None, "data": digest}


@router.post("/health-digest/snapshot")
async def api_health_digest_snapshot():
    """Save the current digest to disk for historical comparison."""
    from tools.health_digest import build_digest, save_digest
    digest = build_digest(hours=24)
    path = save_digest(digest)
    return {"ok": True, "error": None, "data": {"path": path, "summary": digest["summary_line"]}}


@router.get("/health-digest/history")
async def api_health_digest_history(limit: int = Query(7, ge=1, le=30)):
    """Recent saved digests for trend comparison."""
    from tools.health_digest import list_recent_digests
    digests = list_recent_digests(limit=limit)
    return {"ok": True, "error": None, "data": digests, "meta": {"count": len(digests)}}


@router.get("/patterns/rule-effectiveness")
async def api_patterns_rule_effectiveness():
    """List all decision rules with live effectiveness stats + lifecycle status.

    Rules in status=suppressed/retired are not used by the decision layer.
    """
    from tools.rule_effectiveness import list_rules
    rules = list_rules()
    return {"ok": True, "error": None, "data": rules, "meta": {"count": len(rules)}}


@router.post("/patterns/rule/{rule_id}/retire")
async def api_patterns_rule_retire(rule_id: str, reason: str = Query("manual")):
    """Manually retire a rule."""
    from tools.rule_effectiveness import manually_retire_rule
    return manually_retire_rule(rule_id, reason)


@router.post("/patterns/rule/{rule_id}/reactivate")
async def api_patterns_rule_reactivate(rule_id: str):
    """Reactivate a retired rule (resets consecutive_losses too)."""
    from tools.rule_effectiveness import manually_reactivate_rule
    return manually_reactivate_rule(rule_id)


@router.post("/patterns/writeback")
async def api_patterns_writeback(payload: dict = Body(...)):
    """Closed-loop: write a real trade outcome back to the pattern database.

    Body: {
      "symbol": "ETHUSDT", "timeframe": "4h",
      "pattern_id": "source pattern", "strategy_id": "strategy that used it",
      "outcome": {
        "profit_atr": 2.3,
        "max_return_atr": 3.1,
        "max_drawdown_atr": 0.8,
        "hit_target": true,
        "hit_stop": false,
        "bars_in_trade": 14
      }
    }
    """
    from tools.pattern_engine import writeback_trade_outcome
    symbol = payload.get("symbol", "")
    timeframe = payload.get("timeframe", "4h")
    pattern_id = payload.get("pattern_id", "")
    strategy_id = payload.get("strategy_id", "")
    outcome = payload.get("outcome", {})
    if not symbol or not pattern_id:
        return {"ok": False, "error": "symbol and pattern_id required", "data": None}
    result = writeback_trade_outcome(symbol, timeframe, pattern_id, strategy_id, outcome)
    return {"ok": True, "error": None, "data": result}


@router.post("/patterns/labels")
async def api_patterns_save_label(payload: dict = Body(...)):
    """Save a human label on a pattern.

    Body: {
      "pattern_id": "...",
      "symbol": "ETHUSDT",
      "timeframe": "4h",
      "quality": "good" | "bad" | "neutral",
      "tags": ["wedge"],
      "similar_to": ["pattern_id_1"],
      "different_from": ["pattern_id_2"],
      "notes": "..."
    }
    """
    from tools.pattern_labels import PatternLabel, save_label
    label = PatternLabel(
        pattern_id=payload.get("pattern_id", ""),
        symbol=payload.get("symbol", ""),
        timeframe=payload.get("timeframe", "4h"),
        quality=payload.get("quality", ""),
        tags=payload.get("tags", []),
        similar_to=payload.get("similar_to", []),
        different_from=payload.get("different_from", []),
        notes=payload.get("notes", ""),
        labeled_by=payload.get("labeled_by", "user"),
    )
    if not label.pattern_id or not label.symbol:
        return {"ok": False, "error": "pattern_id and symbol required", "data": None}
    result = save_label(label)
    return {"ok": True, "error": None, "data": result}


@router.get("/patterns/labels")
async def api_patterns_list_labels(symbol: str = Query(...), timeframe: str = Query("4h")):
    """List all labels for a symbol/timeframe."""
    from tools.pattern_labels import list_labels
    labels = list_labels(symbol, timeframe)
    return {"ok": True, "error": None, "data": labels, "meta": {"count": len(labels)}}


@router.get("/patterns/database")
async def api_patterns_database():
    """List all pattern databases available."""
    from pathlib import Path
    root = Path("data/patterns")
    if not root.exists():
        return {"ok": True, "error": None, "data": [], "meta": {"count": 0}}
    files = []
    for f in root.glob("*.jsonl"):
        name = f.stem
        size = sum(1 for _ in f.open("r", encoding="utf-8")) if f.exists() else 0
        files.append({"symbol_timeframe": name, "pattern_count": size, "file": f.name})
    return {"ok": True, "error": None, "data": files, "meta": {"count": len(files)}}


@router.post("/strategies/create")
async def api_create_strategy(payload: dict = Body(...)):
    """Create a strategy from structured StrategyConfig.

    Body: { "name": "...", "source": "manual"|"ai", "config": { market, logic_tags, conditions, entry, exit, risk } }
    """
    from tools.strategy import create_from_config
    name = payload.get("name", "")
    source = payload.get("source", "manual")
    config = payload.get("config", {})
    if not name:
        return {"ok": False, "error": "name is required", "data": None}
    result = create_from_config(name, config, source)
    return {"ok": True, "error": None, "data": result}


@router.get("/dashboard")
async def api_tools_dashboard():
    """Aggregated system performance — the 'is it getting better?' endpoint."""
    from tools.ranking import get_leaderboard
    from tools.backtest import list_failures
    from tools.factors import list_factors
    from tools.audit import read_audit
    from agent.state import AgentState
    from dataclasses import asdict

    state = AgentState.load()
    entries = get_leaderboard(50)
    failures = list_failures(limit=200)
    audit = read_audit(limit=100)
    core_factors = list_factors("core")
    validated_factors = list_factors("validated")
    candidate_factors = list_factors("candidate")

    # Aggregate leaderboard stats
    if entries:
        returns = [e.get("return_pct", 0) for e in entries]
        win_rates = [e.get("win_rate", 0) for e in entries]
        sharpes = [e.get("sharpe_ratio", 0) for e in entries]
        drawdowns = [e.get("max_drawdown_pct", 0) for e in entries]
        scores = [e.get("score", 0) for e in entries]
        rrs = [e.get("avg_rr", 0) for e in entries if e.get("avg_rr", 0) > 0]
        perf = {
            "avg_return": round(sum(returns) / len(returns), 2),
            "best_return": round(max(returns), 2),
            "avg_win_rate": round(sum(win_rates) / len(win_rates), 1),
            "avg_sharpe": round(sum(sharpes) / len(sharpes), 2),
            "avg_drawdown": round(sum(drawdowns) / len(drawdowns), 2),
            "max_drawdown": round(max(drawdowns), 2),
            "avg_score": round(sum(scores) / len(scores), 4),
            "best_score": round(max(scores), 4),
            "avg_rr": round(sum(rrs) / len(rrs), 2) if rrs else 0,
            "deployable": sum(1 for e in entries if e.get("deployment_eligible")),
        }
    else:
        perf = {"avg_return": 0, "best_return": 0, "avg_win_rate": 0, "avg_sharpe": 0,
                "avg_drawdown": 0, "max_drawdown": 0, "avg_score": 0, "best_score": 0, "avg_rr": 0, "deployable": 0}

    # Generation-over-generation trend (group entries by generation)
    gen_groups = {}
    for e in entries:
        g = e.get("generation", 0)
        if g not in gen_groups:
            gen_groups[g] = []
        gen_groups[g].append(e)
    trend = []
    for g in sorted(gen_groups.keys()):
        items = gen_groups[g]
        trend.append({
            "generation": g,
            "count": len(items),
            "avg_score": round(sum(e.get("score", 0) for e in items) / len(items), 4),
            "avg_return": round(sum(e.get("return_pct", 0) for e in items) / len(items), 2),
            "best_score": round(max(e.get("score", 0) for e in items), 4),
            "profitable": sum(1 for e in items if e.get("return_pct", 0) > 0),
        })

    # Factor usage frequency across all strategies in leaderboard
    factor_usage = Counter()
    for e in entries:
        for fid in e.get("factor_ids", []):
            factor_usage[fid] += 1
    top_factors = [{"factor_id": fid, "usage_count": cnt} for fid, cnt in factor_usage.most_common(15)]

    # Error distribution by stage
    stage_counts = Counter(f.get("stage", "unknown") for f in failures)
    error_dist = [{"stage": s, "count": c} for s, c in stage_counts.most_common()]

    return {"ok": True, "error": None, "data": {
        "state": asdict(state),
        "performance": perf,
        "leaderboard_size": len(entries),
        "trend": trend,
        "top_factors_used": top_factors,
        "error_distribution": error_dist,
        "total_failures": len(failures),
        "factor_counts": {
            "core": len(core_factors),
            "candidate": len(candidate_factors),
            "validated": len(validated_factors),
            "tested": sum(1 for f in core_factors + candidate_factors + validated_factors if f.get("test_count", 0) > 0),
        },
        "recent_audit": audit[-10:],
    }}


@router.get("/failures/summary")
async def api_tools_failures_summary():
    """Aggregated failure stats — by stage, by generation."""
    from tools.backtest import list_failures
    failures = list_failures(limit=500)
    by_stage = {}
    by_gen = {}
    latest_at = 0.0
    for f in failures:
        stage = f.get("stage", "unknown")
        gen = f.get("generation", 0)
        by_stage[stage] = by_stage.get(stage, 0) + 1
        by_gen[gen] = by_gen.get(gen, 0) + 1
        t = f.get("failed_at", 0)
        if t > latest_at:
            latest_at = t
    return {"ok": True, "error": None, "data": {
        "total": len(failures),
        "by_stage": [{"stage": s, "count": c} for s, c in sorted(by_stage.items(), key=lambda x: -x[1])],
        "by_generation": [{"generation": g, "count": c} for g, c in sorted(by_gen.items())],
        "latest_failed_at": latest_at if latest_at > 0 else None,
    }}


@router.get("/factor-rankings")
async def api_tools_factor_rankings():
    """Factors ranked by test performance — the 'which factors actually work?' endpoint."""
    from tools.factors import list_factors

    all_factors = []
    for stage in ("core", "candidate", "validated"):
        for f in list_factors(stage):
            if f.get("test_count", 0) > 0:
                all_factors.append({
                    "id": f.get("id", ""),
                    "name": f.get("name", ""),
                    "category": f.get("category", ""),
                    "stage": f.get("stage", stage),
                    "test_count": f.get("test_count", 0),
                    "avg_score": round(f.get("avg_score", 0), 4),
                    "total_trades": f.get("total_trades", 0),
                    "best_symbol": _best_symbol(f.get("test_history", [])),
                    "score_trend": _score_trend(f.get("test_history", [])),
                })

    # Sort by avg_score descending
    all_factors.sort(key=lambda f: -f["avg_score"])
    return {"ok": True, "error": None, "data": all_factors, "meta": {"count": len(all_factors)}}


def _best_symbol(history: list) -> str:
    """Find which symbol this factor scores highest on."""
    if not history:
        return ""
    by_sym = {}
    for t in history:
        sym = t.get("symbol", "")
        if sym:
            by_sym.setdefault(sym, []).append(t.get("score", 0))
    if not by_sym:
        return ""
    best = max(by_sym.items(), key=lambda x: sum(x[1]) / len(x[1]))
    return best[0]


def _score_trend(history: list) -> str:
    """Simple trend: compare first half avg vs second half avg."""
    if len(history) < 2:
        return "neutral"
    mid = len(history) // 2
    first = sum(t.get("score", 0) for t in history[:mid]) / mid
    second = sum(t.get("score", 0) for t in history[mid:]) / (len(history) - mid)
    if second > first * 1.05:
        return "improving"
    if second < first * 0.95:
        return "declining"
    return "stable"


# ── Live Draft Manager API ──────────────────────────────────────────────

@router.get("/live-drafts")
async def api_live_drafts(status: str = None):
    """List live deployment drafts."""
    from tools.live_manager import list_live_drafts
    return list_live_drafts(status=status)


@router.get("/live-drafts/{draft_id}")
async def api_live_draft_detail(draft_id: str):
    """Get a single live draft."""
    from tools.live_manager import get_live_draft
    return get_live_draft(draft_id)


@router.post("/live-drafts")
async def api_create_live_draft(
    entry_id: str = Query(..., description="Leaderboard entry ID"),
    capital: float = Query(100.0, ge=1),
    risk_per_trade: float = Query(0.01, ge=0.001, le=0.1),
    auto_submit: bool = Query(False),
    notes: str = Query(""),
):
    """Create a live deployment draft from a leaderboard entry."""
    from tools.live_manager import create_draft_from_leaderboard
    return create_draft_from_leaderboard(entry_id, capital, risk_per_trade, auto_submit, notes)


@router.post("/live-drafts/{draft_id}/approve")
async def api_approve_live_draft(draft_id: str):
    """Approve a draft (runs risk guard check)."""
    from tools.live_manager import approve_live_draft
    return approve_live_draft(draft_id)


@router.post("/live-drafts/{draft_id}/deploy")
async def api_deploy_live_draft(draft_id: str):
    """Deploy an approved draft — creates a live instance."""
    from tools.live_manager import create_instance_from_draft
    return create_instance_from_draft(draft_id)


@router.delete("/live-drafts/{draft_id}")
async def api_delete_live_draft(draft_id: str):
    """Delete a draft."""
    from tools.live_manager import delete_live_draft
    return delete_live_draft(draft_id)


@router.get("/live-instances")
async def api_live_instances(status: str = None):
    """List live strategy instances."""
    from tools.live_manager import list_live_instances
    return list_live_instances(status=status)


@router.get("/live-instances/{instance_id}")
async def api_live_instance_detail(instance_id: str):
    """Get a single live instance."""
    from tools.live_manager import get_live_instance
    return get_live_instance(instance_id)


@router.post("/live-instances/{instance_id}/pause")
async def api_pause_instance(instance_id: str):
    """Pause a running instance."""
    from tools.live_manager import pause_live_instance
    return pause_live_instance(instance_id)


@router.post("/live-instances/{instance_id}/resume")
async def api_resume_instance(instance_id: str):
    """Resume a paused instance."""
    from tools.live_manager import resume_live_instance
    return resume_live_instance(instance_id)


@router.post("/live-instances/{instance_id}/stop")
async def api_stop_instance(instance_id: str):
    """Stop a running/paused instance."""
    from tools.live_manager import stop_live_instance
    return stop_live_instance(instance_id)


@router.post("/live-instances/{instance_id}/retire")
async def api_retire_instance(instance_id: str):
    """Retire a stopped instance."""
    from tools.live_manager import retire_live_instance
    return retire_live_instance(instance_id)
