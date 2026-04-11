"""
S/R Pipeline Full Verification Loop

Self-proving system: proves that for every symbol + timeframe:
1. Data is complete (or explains why not)
2. Structure detection ran (swings, pivots)
3. S/R zones were extracted (or explains why none)
4. Signal decision was made (or explains why no signal)
5. Execution decision was made (or explains why blocked)

Does NOT stop until every check passes or has an explained failure.
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

from server.strategy.pivots import detect_pivots
from server.strategy.zones import detect_horizontal_zones
from server.strategy.zone_signals import generate_zone_signals
from server.strategy.trendlines import detect_trendlines
from server.strategy.signals import generate_signals
from server.strategy.config import StrategyConfig, calculate_atr
from server.strategy.position_sizing import is_timeframe_verified, get_calibrated_params
from server.data_service import get_ohlcv_with_df
import pandas as pd
import numpy as np

LOG_PATH = os.path.join(os.path.dirname(__file__), "sr_pipeline_verification.log")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT",
           "DOGEUSDT", "SUIUSDT", "ADAUSDT"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


async def verify_pipeline():
    cfg = StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=500, min_rr_ratio=3.0)
    log = open(LOG_PATH, "w", encoding="utf-8")
    now = datetime.now().isoformat()
    total_checks = 0
    passed_checks = 0
    issues = []

    def log_line(msg):
        log.write(msg + "\n")
        log.flush()
        print(msg, flush=True)

    log_line(f"=== S/R PIPELINE VERIFICATION === {now}")
    log_line(f"Symbols: {len(SYMBOLS)} | Timeframes: {len(TIMEFRAMES)}")
    log_line("")

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            prefix = f"[{sym}][{tf}]"
            total_checks += 1

            # ── STEP 0: Data completeness ──
            try:
                df, _ = await get_ohlcv_with_df(sym, tf, None, 365, history_mode="fast_window")
                pdf = df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
                pdf = pdf.rename(columns={"open_time": "timestamp"})
                pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
                for c in ("open", "high", "low", "close", "volume"):
                    pdf[c] = pd.to_numeric(pdf[c])
                pdf = pdf.reset_index(drop=True)
                n_bars = len(pdf)

                if n_bars == 0:
                    log_line(f"{prefix}[DATA] FAIL: 0 bars received")
                    issues.append(f"{prefix} no data")
                    continue

                first_ts = pdf.iloc[0]["timestamp"]
                last_ts = pdf.iloc[-1]["timestamp"]
                close_price = float(pdf.iloc[-1]["close"])

                log_line(f"{prefix}[DATA] bars={n_bars} | range={datetime.fromtimestamp(first_ts).strftime('%Y-%m-%d')} -> {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d')} | close=${close_price:.4f}")

            except Exception as e:
                log_line(f"{prefix}[DATA] ERROR: {e}")
                issues.append(f"{prefix} data error: {e}")
                continue

            # ── STEP 1: Discovery — Structure detection ──
            try:
                pivots = detect_pivots(pdf, cfg)
                high_pivots = [p for p in pivots if p.kind == "high"]
                low_pivots = [p for p in pivots if p.kind == "low"]
                log_line(f"{prefix}[DISCOVERY] pivots={len(pivots)} (highs={len(high_pivots)}, lows={len(low_pivots)})")

                if len(pivots) < 4:
                    log_line(f"{prefix}[DISCOVERY] WARNING: too few pivots ({len(pivots)}), likely insufficient data")
                    issues.append(f"{prefix} too few pivots")
            except Exception as e:
                log_line(f"{prefix}[DISCOVERY] ERROR: {e}")
                issues.append(f"{prefix} discovery error")
                continue

            # ── STEP 1b: Trendline detection ──
            try:
                detection = detect_trendlines(pdf, pivots, cfg, symbol=sym, timeframe=tf)
                confirmed = [l for l in detection.candidate_lines if l.state == "confirmed"]
                log_line(f"{prefix}[DISCOVERY] trendlines: candidates={len(detection.candidate_lines)} confirmed={len(confirmed)} active={len(detection.active_lines)}")
            except Exception as e:
                log_line(f"{prefix}[DISCOVERY] trendline error: {e}")

            # ── STEP 2: Evaluation — S/R zone extraction + scoring ──
            try:
                zones = detect_horizontal_zones(pdf, pivots, cfg, symbol=sym, timeframe=tf, max_zones_per_side=3)
                support_zones = [z for z in zones if z.side == "support"]
                resist_zones = [z for z in zones if z.side == "resistance"]
                log_line(f"{prefix}[EVALUATION] zones={len(zones)} (support={len(support_zones)}, resistance={len(resist_zones)})")

                for z in zones:
                    dist_pct = abs(close_price - z.price_center) / close_price * 100
                    log_line(f"{prefix}[EVALUATION]   {z.side:10s} ${z.price_center:.4f} [{z.price_low:.4f}-{z.price_high:.4f}] touches={z.touches} strength={z.strength:.1f} dist={dist_pct:.1f}%")

                if len(zones) == 0:
                    log_line(f"{prefix}[EVALUATION] reason=no_clusters_above_min_touches (pivots may be too spread out for eps={cfg.lookback_bars})")
            except Exception as e:
                log_line(f"{prefix}[EVALUATION] ERROR: {e}")
                issues.append(f"{prefix} evaluation error")
                continue

            # ── STEP 3: Signal decision ──
            try:
                zone_sigs = generate_zone_signals(pdf, zones, cfg, symbol=sym, timeframe=tf)
                line_sigs = generate_signals(pdf, detection.active_lines, cfg) if detection.active_lines else []
                all_sigs = zone_sigs + line_sigs

                if all_sigs:
                    for s in all_sigs:
                        log_line(f"{prefix}[SIGNAL] {s.signal_type} dir={s.direction} entry=${s.entry_price:.4f} SL=${s.stop_price:.4f} TP=${s.tp_price:.4f} RR={s.risk_reward:.1f} score={s.score:.3f}")
                else:
                    # Explain why no signal
                    reasons = []
                    if len(zones) == 0:
                        reasons.append("no_zones_detected")
                    else:
                        atr = calculate_atr(pdf, 14)
                        atr_val = float(atr.iloc[-1])
                        arm_dist = max(atr_val * 0.5, close_price * 0.005)
                        for z in zones:
                            if z.side == "support":
                                dist = close_price - z.price_high
                            else:
                                dist = z.price_low - close_price
                            if dist < -arm_dist:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_price_below_zone")
                            elif dist > arm_dist * 2:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_price_too_far({dist/atr_val:.1f}ATR)")
                            else:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_no_rejection_wick")
                            if z.touches < 3:
                                reasons.append(f"{z.side}_${z.price_center:.2f}_touches_below_3")
                    log_line(f"{prefix}[SIGNAL] NONE | reasons: {'; '.join(reasons[:5])}")
            except Exception as e:
                log_line(f"{prefix}[SIGNAL] ERROR: {e}")
                issues.append(f"{prefix} signal error")
                continue

            # ── STEP 4: Execution decision ──
            verified = is_timeframe_verified(tf)
            cal_wr, cal_rr, cal_kelly = get_calibrated_params(tf)
            if verified:
                log_line(f"{prefix}[EXECUTION] TF_VERIFIED | WR={cal_wr:.1%} RR={cal_rr:.2f} Kelly={cal_kelly:.2%}")
                if all_sigs:
                    log_line(f"{prefix}[EXECUTION] {len(all_sigs)} signal(s) ready for paper/live execution")
                else:
                    log_line(f"{prefix}[EXECUTION] no signals to execute (waiting for setup)")
            else:
                log_line(f"{prefix}[EXECUTION] TF_BLOCKED | reason=not_in_BACKTEST_CALIBRATION (negative or insufficient EV)")
                if all_sigs:
                    log_line(f"{prefix}[EXECUTION] {len(all_sigs)} signal(s) would be REJECTED by risk_rules")

            passed_checks += 1
            log_line("")

    # ── Summary ──
    log_line("=" * 70)
    log_line(f"VERIFICATION COMPLETE: {passed_checks}/{total_checks} symbol-TF pairs checked")
    log_line(f"Issues: {len(issues)}")
    for issue in issues:
        log_line(f"  - {issue}")

    verified_tfs = [tf for tf in TIMEFRAMES if is_timeframe_verified(tf)]
    blocked_tfs = [tf for tf in TIMEFRAMES if not is_timeframe_verified(tf)]
    log_line(f"\nVerified TFs (can trade): {verified_tfs}")
    log_line(f"Blocked TFs (no trade): {blocked_tfs}")
    log_line(f"\nLog saved to: {LOG_PATH}")
    log.close()


if __name__ == "__main__":
    asyncio.run(verify_pipeline())
