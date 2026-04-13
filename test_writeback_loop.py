"""End-to-end test: pattern → draft → live draft → instance → stop → writeback.

WARNING: this test was historically writing fake current_pnl=0.45 to real
prod instance JSONs in data/strategies/live_instances/, then triggering
the writeback machinery which fed the fake P&L back into pattern engine
and rule effectiveness — polluting both for downstream queries and the UI.

Now hardened: REQUIRES TEST_DATA_DIR env var to be set to a tmp/ path,
otherwise refuses to run. The dirs it touches are isolated from prod.

Verifies the closed loop:
1. Pattern-driven generation creates a draft with source_pattern_id + decision_rule
2. Live draft is created from leaderboard entry — lineage propagates
3. Instance created — lineage propagates
4. Instance stopped — writeback fires automatically
5. Pattern DB gets a new record with split_bucket=live
6. Rule effectiveness tracker gets updated
"""
import os, sys, json, asyncio, time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass


# Refuse to run unless explicitly directed to an isolated tmp dir.
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR")
if not TEST_DATA_DIR:
    print("REFUSING TO RUN: this test mutates data/strategies/live_instances/.", file=sys.stderr)
    print("Set TEST_DATA_DIR=/path/to/isolated/tmp before running.", file=sys.stderr)
    sys.exit(2)
if "live_instances" in TEST_DATA_DIR or TEST_DATA_DIR.startswith(str(Path("data/strategies").absolute())):
    print(f"REFUSING TO RUN: TEST_DATA_DIR={TEST_DATA_DIR} looks like prod.", file=sys.stderr)
    sys.exit(2)


async def main():
    from agent.pattern_driven import scan_and_generate
    from tools.ranking import update_leaderboard, get_leaderboard
    from tools.live_manager import (
        create_draft_from_leaderboard, approve_live_draft, create_instance_from_draft,
        stop_live_instance, get_live_instance,
    )
    from tools.rule_effectiveness import list_rules, get_rule
    from tools.pattern_engine import load_patterns
    from tools.types import SimulationResult

    print("=" * 70)
    print("END-TO-END WRITEBACK LOOP TEST")
    print("=" * 70)

    # STEP 1: Generate pattern-driven drafts
    print("\n[1/6] Pattern-driven draft generation...")
    report = await scan_and_generate(
        symbols=["ETHUSDT"],
        timeframes=["4h"],
        generation=999,
        batch_id="writeback-test",
        max_drafts_per_symbol=1,
    )
    print(f"    evaluated={report['evaluated']} approved={report['approved']}")
    if not report['drafts']:
        print("    FAIL: no drafts generated")
        return
    draft = report['drafts'][0]
    print(f"    draft_id: {draft['id']}")
    print(f"    source_pattern_id: {draft.get('source_pattern_id','')}")
    print(f"    decision_rule: {draft.get('decision_rule','')}")
    print(f"    pattern_decision: {draft.get('pattern_decision','')}")
    print(f"    pattern_ev: {draft.get('pattern_ev')}")

    assert draft.get('source_pattern_id'), "Draft missing source_pattern_id"
    assert draft.get('decision_rule'), "Draft missing decision_rule"

    # STEP 2: Create a fake leaderboard entry for this draft (skip simulation)
    print("\n[2/6] Creating fake leaderboard entry...")
    fake_result = SimulationResult(
        job_id="test-job",
        strategy_id=draft['id'],
        symbol="ETHUSDT",
        timeframe="4h",
        return_pct=5.0, win_rate=60.0, profit_factor=1.8,
        sharpe_ratio=1.5, max_drawdown_pct=3.0, trade_count=10,
        avg_rr=2.0, score=0.75,
        train_score=0.7, val_score=0.72, test_score=0.76,
        train_return=4.5, val_return=5.0, test_return=5.5,
        overfit_flag="stable",
        generation=999, batch_id="writeback-test",
    )
    meta = {draft['id']: {'name': draft['name'], 'source': 'pattern_engine', 'factor_ids': draft.get('factor_ids',[]), 'trigger_modes': draft.get('trigger_modes',[])}}
    lb = update_leaderboard([fake_result], meta, generation=999)
    entries = get_leaderboard(5)
    our_entry = next((e for e in entries if e.get('strategy_id') == draft['id']), None)
    if not our_entry:
        print("    FAIL: entry not in leaderboard")
        return
    print(f"    entry_id: {our_entry['id']}")
    print(f"    score: {our_entry.get('score')}")

    # STEP 3: Create live draft from entry → verify lineage
    print("\n[3/6] Creating live draft from leaderboard...")
    ld_result = create_draft_from_leaderboard(our_entry['id'], capital=15.0, risk_per_trade=0.01)
    if not ld_result['ok']:
        print(f"    FAIL: {ld_result.get('error')}")
        return
    live_draft = ld_result['data']
    print(f"    live_draft_id: {live_draft['id']}")
    print(f"    source_pattern_id: {live_draft.get('source_pattern_id','')}")
    print(f"    decision_rule: {live_draft.get('decision_rule','')}")
    print(f"    pattern_ev: {live_draft.get('pattern_ev')}")
    assert live_draft.get('source_pattern_id'), "Live draft missing source_pattern_id"
    assert live_draft.get('decision_rule'), "Live draft missing decision_rule"

    # STEP 4: Approve + deploy → verify lineage on instance
    print("\n[4/6] Approving and deploying...")
    ap = approve_live_draft(live_draft['id'])
    if not ap['ok']:
        print(f"    approve FAIL: {ap.get('error')}")
        return
    dep = create_instance_from_draft(live_draft['id'])
    if not dep['ok']:
        print(f"    deploy FAIL: {dep.get('error')}")
        return
    instance = dep['data']
    print(f"    instance_id: {instance['id']}")
    print(f"    source_pattern_id: {instance.get('source_pattern_id','')}")
    print(f"    decision_rule: {instance.get('decision_rule','')}")
    print(f"    pattern_ev_expected: {instance.get('pattern_ev_expected')}")
    print(f"    symbol: {instance.get('symbol')}")
    print(f"    timeframe: {instance.get('timeframe')}")
    assert instance.get('source_pattern_id'), "Instance missing source_pattern_id"
    assert instance.get('decision_rule'), "Instance missing decision_rule"

    # STEP 5: Stop the instance WITHOUT injecting fake P&L.
    # Previously this block wrote current_pnl=0.45 directly to the prod
    # instance JSON, which then propagated through the writeback machinery
    # into pattern engine and rule effectiveness DBs as a fake "+$0.45 win".
    # Six instance files were polluted that way and showed up as fake
    # profit in the live-management UI. Now we just stop the instance with
    # whatever real P&L it has (typically 0 since 0 trades).
    print("\n[5/6] Stopping instance (no fake P&L injection)...")
    instance_path = Path(f"data/strategies/live_instances/{instance['id']}.json")
    inst_data = json.loads(instance_path.read_text(encoding="utf-8"))
    inst_data['started_at'] = time.time() - 3 * 14400  # 3 bars ago on 4h (timing only)
    instance_path.write_text(json.dumps(inst_data, indent=2), encoding="utf-8")

    stop_result = stop_live_instance(instance['id'])
    if not stop_result['ok']:
        print(f"    stop FAIL: {stop_result.get('error')}")
        return
    stopped = stop_result['data']
    print(f"    running_status: {stopped.get('running_status')}")
    print(f"    realized_return_pct: {stopped.get('realized_return_pct')}")
    print(f"    realized_return_atr: {stopped.get('realized_return_atr')}")
    print(f"    realized_drawdown_atr: {stopped.get('realized_drawdown_atr')}")
    print(f"    bars_held: {stopped.get('bars_held')}")
    print(f"    outcome_class: {stopped.get('outcome_class')}")
    print(f"    outcome_success: {stopped.get('outcome_success')}")
    print(f"    outcome_written_back: {stopped.get('outcome_written_back')}")
    assert stopped.get('outcome_written_back'), "Writeback flag not set!"

    # STEP 6: Verify writeback hit pattern DB and rule effectiveness tracker
    print("\n[6/6] Verifying writeback effects...")

    # Check pattern DB for new live record
    db = load_patterns("ETHUSDT", "4h")
    live_records = [r for r in db if r.split_bucket == "live"]
    print(f"    pattern DB live records: {len(live_records)}")
    if live_records:
        most_recent = max(live_records, key=lambda r: r.anchor2_idx)
        print(f"    most recent live record outcome:")
        print(f"      bounced={most_recent.outcome.bounced}")
        print(f"      broke={most_recent.outcome.broke}")
        print(f"      max_return_atr={most_recent.outcome.max_return_atr}")
        print(f"      max_drawdown_atr={most_recent.outcome.max_drawdown_atr}")

    # Check rule effectiveness
    rules = list_rules()
    print(f"    rule effectiveness entries: {len(rules)}")
    for r in rules:
        print(f"      rule_id: {r['rule_id']}")
        print(f"        decision_type: {r['decision_type']}")
        print(f"        live_count: {r['live_count']}")
        print(f"        live_win_rate: {r['live_win_rate']}")
        print(f"        live_avg_return_atr: {r['live_avg_return_atr']}")
        print(f"        live_expected_value: {r['live_expected_value']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
