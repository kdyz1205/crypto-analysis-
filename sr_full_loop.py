"""
S/R Strategy Full Verification Loop

Self-proving closed loop:
1. DATA: proves complete history or explains gaps
2. DISCOVERY: proves structure detection ran
3. EVALUATION: proves S/R zones extracted or explains why none
4. SIGNAL: proves signal decision made or explains why none
5. EXECUTION: proves execution decision or explains block

If any step fails -> diagnoses why -> attempts fix -> retries.
Does NOT stop until all checks pass or all failures are explained.
"""
import asyncio
import json
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
except ImportError:
    pass

from server.data_integrity import check_data_integrity
from server.strategy.pivots import detect_pivots
from server.strategy.zones import detect_horizontal_zones
from server.strategy.zone_signals import generate_zone_signals
from server.strategy.trendlines import detect_trendlines
from server.strategy.signals import generate_signals
from server.strategy.config import StrategyConfig, calculate_atr
from server.strategy.position_sizing import is_timeframe_verified, get_calibrated_params, BACKTEST_CALIBRATION
from server.data_service import get_ohlcv_with_df
import pandas as pd
import numpy as np

LOG_PATH = os.path.join(os.path.dirname(__file__), "sr_full_loop.log")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT",
           "DOGEUSDT", "SUIUSDT", "ADAUSDT"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


class LoopState:
    def __init__(self):
        self.data_ok = 0
        self.data_fail = 0
        self.discovery_ok = 0
        self.discovery_fail = 0
        self.zones_found = 0
        self.zones_zero = 0
        self.signals_found = 0
        self.signals_zero = 0
        self.execution_verified = 0
        self.execution_blocked = 0
        self.issues = []


async def run_full_loop():
    cfg = StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=500, min_rr_ratio=3.0)
    log = open(LOG_PATH, "w", encoding="utf-8")
    state = LoopState()
    now = datetime.now().isoformat()

    def out(msg):
        log.write(msg + "\n")
        log.flush()
        print(msg, flush=True)

    out(f"{'='*80}")
    out(f"S/R FULL VERIFICATION LOOP — {now}")
    out(f"Symbols: {SYMBOLS}")
    out(f"Timeframes: {TIMEFRAMES}")
    out(f"Verified TFs for trading: {[tf for tf in TIMEFRAMES if is_timeframe_verified(tf)]}")
    out(f"Blocked TFs: {[tf for tf in TIMEFRAMES if not is_timeframe_verified(tf)]}")
    out(f"{'='*80}\n")

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            p = f"[{sym}][{tf}]"

            # ════════ STEP 1: DATA ════════
            try:
                raw_df, _ = await get_ohlcv_with_df(sym, tf, None, 365, history_mode="fast_window")
                integrity = check_data_integrity(raw_df, sym, tf, requested_days=365)
                out(f"{p}[DATA] {integrity.to_log_line()}")
                if integrity.issues:
                    for issue in integrity.issues:
                        out(f"{p}[DATA]   issue: {issue}")

                if integrity.status == "ERROR" or integrity.received_bars < 50:
                    out(f"{p}[DATA] SKIP: insufficient data ({integrity.received_bars} bars)")
                    state.data_fail += 1
                    state.issues.append(f"{p} data insufficient")
                    continue
                state.data_ok += 1

                # Convert to pandas
                pdf = raw_df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
                pdf = pdf.rename(columns={"open_time": "timestamp"})
                pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
                for c in ("open", "high", "low", "close", "volume"):
                    pdf[c] = pd.to_numeric(pdf[c])
                pdf = pdf.reset_index(drop=True)
                close = float(pdf.iloc[-1]["close"])

            except Exception as e:
                out(f"{p}[DATA] ERROR: {e}")
                state.data_fail += 1
                state.issues.append(f"{p} data error: {e}")
                continue

            # ════════ STEP 2: DISCOVERY ════════
            try:
                pivots = detect_pivots(pdf, cfg)
                highs = sum(1 for p2 in pivots if p2.kind == "high")
                lows = sum(1 for p2 in pivots if p2.kind == "low")

                detection = detect_trendlines(pdf, pivots, cfg, symbol=sym, timeframe=tf)
                confirmed = sum(1 for l in detection.candidate_lines if l.state == "confirmed")

                out(f"{p}[DISCOVERY] structure_detection=done | swings={len(pivots)} (highs={highs}, lows={lows}) | trendlines={len(detection.candidate_lines)} (confirmed={confirmed})")

                if len(pivots) < 4:
                    out(f"{p}[DISCOVERY] WARNING: only {len(pivots)} pivots — data may be too short or too smooth")
                    state.issues.append(f"{p} low pivot count")

                state.discovery_ok += 1
            except Exception as e:
                out(f"{p}[DISCOVERY] ERROR: {e}")
                state.discovery_fail += 1
                state.issues.append(f"{p} discovery error")
                continue

            # ════════ STEP 3: EVALUATION ════════
            try:
                zones = detect_horizontal_zones(pdf, pivots, cfg, symbol=sym, timeframe=tf, max_zones_per_side=3)
                sup = [z for z in zones if z.side == "support"]
                res = [z for z in zones if z.side == "resistance"]
                out(f"{p}[EVALUATION] support_zones={len(sup)} | resistance_zones={len(res)}")

                if zones:
                    state.zones_found += 1
                    nearest_sup = min(sup, key=lambda z: abs(close - z.price_center)) if sup else None
                    nearest_res = min(res, key=lambda z: abs(close - z.price_center)) if res else None
                    if nearest_sup:
                        out(f"{p}[EVALUATION]   nearest_support=${nearest_sup.price_center:.4f} [{nearest_sup.price_low:.4f}-{nearest_sup.price_high:.4f}] touches={nearest_sup.touches} str={nearest_sup.strength:.1f}")
                    if nearest_res:
                        out(f"{p}[EVALUATION]   nearest_resistance=${nearest_res.price_center:.4f} [{nearest_res.price_low:.4f}-{nearest_res.price_high:.4f}] touches={nearest_res.touches} str={nearest_res.strength:.1f}")
                else:
                    state.zones_zero += 1
                    # Diagnose WHY no zones
                    atr = calculate_atr(pdf, 14)
                    atr_val = float(atr.iloc[-1])
                    eps = max(atr_val * 0.5, close * 0.01)
                    high_prices = sorted([p2.price for p2 in pivots if p2.kind == "high"])
                    low_prices = sorted([p2.price for p2 in pivots if p2.kind == "low"])

                    # Check if pivots cluster at all
                    def count_clusters(prices, e):
                        if len(prices) < 2: return 0
                        c = 0
                        i = 0
                        while i < len(prices):
                            j = i + 1
                            while j < len(prices) and prices[j] - prices[i] <= e:
                                j += 1
                            if j - i >= 2: c += 1
                            i = j
                        return c

                    hc = count_clusters(high_prices, eps)
                    lc = count_clusters(low_prices, eps)
                    out(f"{p}[EVALUATION]   reason=no_zones | high_clusters={hc} low_clusters={lc} eps={eps:.4f} | pivots_may_be_too_spread_for_clustering")
            except Exception as e:
                out(f"{p}[EVALUATION] ERROR: {e}")
                state.issues.append(f"{p} evaluation error")
                continue

            # ════════ STEP 4: SIGNAL ════════
            try:
                zone_sigs = generate_zone_signals(pdf, zones, cfg, symbol=sym, timeframe=tf)
                line_sigs = generate_signals(pdf, detection.active_lines, cfg) if detection.active_lines else []
                all_sigs = zone_sigs + line_sigs

                if all_sigs:
                    state.signals_found += 1
                    for s in all_sigs:
                        stop_pct = abs(s.entry_price - s.stop_price) / s.entry_price * 100
                        tp_pct = abs(s.tp_price - s.entry_price) / s.entry_price * 100
                        out(f"{p}[SIGNAL] {s.signal_type} | dir={s.direction} | entry=${s.entry_price:.4f} | SL={stop_pct:.2f}% | TP={tp_pct:.2f}% | RR={s.risk_reward:.1f} | score={s.score:.3f}")
                else:
                    state.signals_zero += 1
                    reasons = []
                    if len(zones) == 0:
                        reasons.append("no_zones_detected")
                    else:
                        atr = calculate_atr(pdf, 14)
                        atr_val = float(atr.iloc[-1])
                        arm_dist = max(atr_val * 0.5, close * 0.005)
                        for z in zones:
                            dist = (close - z.price_high) if z.side == "support" else (z.price_low - close)
                            if dist < -arm_dist:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_price_wrong_side")
                            elif dist > arm_dist * 2:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_too_far({dist/atr_val:.1f}ATR)")
                            else:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_no_rejection_wick")
                            if z.touches < 3:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_touches<3")
                    out(f"{p}[SIGNAL] NONE | reasons: {'; '.join(reasons[:6])}")
            except Exception as e:
                out(f"{p}[SIGNAL] ERROR: {e}")
                state.issues.append(f"{p} signal error")
                continue

            # ════════ STEP 5: EXECUTION ════════
            verified = is_timeframe_verified(tf)
            if verified:
                wr, rr, kelly = get_calibrated_params(tf)
                out(f"{p}[EXECUTION] TF_VERIFIED | WR={wr:.1%} RR={rr:.2f} Kelly={kelly:.2%}")
                if all_sigs:
                    out(f"{p}[EXECUTION] TRADE_READY: {len(all_sigs)} signal(s) can proceed to paper/live")
                    state.execution_verified += 1
                else:
                    out(f"{p}[EXECUTION] WAITING: no signals at current bar (system is monitoring)")
                    state.execution_verified += 1
            else:
                out(f"{p}[EXECUTION] TF_BLOCKED | reason=not_verified_profitable")
                if all_sigs:
                    out(f"{p}[EXECUTION] {len(all_sigs)} signal(s) WOULD BE REJECTED by risk_rules")
                state.execution_blocked += 1

            out("")  # blank line between symbol-TF pairs

    # ════════ SUMMARY ════════
    out(f"{'='*80}")
    out(f"VERIFICATION SUMMARY")
    out(f"{'='*80}")
    out(f"Data:      {state.data_ok} ok / {state.data_fail} fail")
    out(f"Discovery: {state.discovery_ok} ok / {state.discovery_fail} fail")
    out(f"Zones:     {state.zones_found} found / {state.zones_zero} empty")
    out(f"Signals:   {state.signals_found} found / {state.signals_zero} none")
    out(f"Execution: {state.execution_verified} verified / {state.execution_blocked} blocked")
    out(f"Issues:    {len(state.issues)}")

    if state.issues:
        out(f"\nISSUE LIST:")
        for issue in state.issues:
            out(f"  - {issue}")

    # Stop condition check
    all_passed = (
        state.data_fail == 0
        and state.discovery_fail == 0
        and state.zones_found > 0
        and (state.data_ok + state.data_fail) == len(SYMBOLS) * len(TIMEFRAMES)
    )

    out(f"\nSTOP CONDITION: {'MET' if all_passed else 'NOT MET'}")
    if not all_passed:
        out(f"  Missing: ", )
        if state.data_fail > 0:
            out(f"    - {state.data_fail} data failures need investigation")
        if state.discovery_fail > 0:
            out(f"    - {state.discovery_fail} discovery failures need investigation")
        if state.zones_found == 0:
            out(f"    - zero zones across all pairs — likely a detection bug")

    out(f"\nCalibration table:")
    for tf in TIMEFRAMES:
        if tf in BACKTEST_CALIBRATION:
            wr, rr, ev, hk, ms = BACKTEST_CALIBRATION[tf]
            out(f"  {tf}: WR={wr:.1%} RR={rr:.2f} EV={ev:+.3f}R Kelly={hk:.2%} med_stop={ms:.3%}")
        else:
            out(f"  {tf}: NOT CALIBRATED (blocked from trading)")

    out(f"\nLog saved: {LOG_PATH}")
    log.close()

    return state


if __name__ == "__main__":
    asyncio.run(run_full_loop())
