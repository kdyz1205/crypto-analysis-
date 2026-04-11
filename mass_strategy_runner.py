"""
Mass Strategy Runner v2 — Vectorized.
200 coins x ~10K strategy variants. Each combo gets $100 paper money.
Strategies that blow up get eliminated. Survivors evolve.
All trades logged to mass_trades.log.
Leaderboard every 60s saved to mass_strategy_report.json.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product

import httpx
import numpy as np

try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env):
        load_dotenv(_env, override=True)
except ImportError:
    pass

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BITGET_REST = "https://api.bitget.com"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH = os.path.join(PROJECT_DIR, "mass_trades.log")
REPORT_PATH = os.path.join(PROJECT_DIR, "mass_strategy_report.json")

TOP_N_COINS = 200
CANDLE_LIMIT = 50
TICK_INTERVAL = 5
REPORT_INTERVAL = 60
PAPER_EQUITY = 100.0
ELIMINATION_THRESHOLD = 10.0  # eliminate strategy when equity < $10

# ── Strategy Grid ──────────────────────────────────────────────────────
# Generates ~10K valid combos

GRID = {
    "ema_fast": [2, 3, 5, 8],
    "ema_slow": [8, 13, 21],
    "rsi_period": [4, 6, 9, 14],
    "rsi_buy": [25, 35, 45],
    "rsi_sell": [55, 65, 75],
    "sl_pct": [0.002, 0.005, 0.01],
    "tp_pct": [0.003, 0.008, 0.015],
    "stype": [0, 1, 2],  # 0=momentum, 1=mean_reversion, 2=breakout
}

def build_grid() -> np.ndarray:
    """Build strategy parameter matrix. Shape: (N_strategies, 8)"""
    keys = list(GRID.keys())
    combos = []
    for vals in product(*GRID.values()):
        p = dict(zip(keys, vals))
        if p["ema_fast"] >= p["ema_slow"]:
            continue
        if p["rsi_buy"] >= p["rsi_sell"]:
            continue
        combos.append(list(vals))
    return np.array(combos, dtype=np.float64)

STYPE_NAMES = {0: "momentum", 1: "mean_rev", 2: "breakout"}

def strat_label(params: np.ndarray) -> str:
    st = STYPE_NAMES.get(int(params[7]), "unk")
    return f"{st}_EMA{int(params[0])}/{int(params[1])}_RSI{int(params[2])}_{int(params[3])}/{int(params[4])}_SL{params[5]*100:.1f}_TP{params[6]*100:.1f}"


# ── Indicators (numpy vectorized) ─────────────────────────────────────

def ema_np(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def rsi_np(close: np.ndarray, period: int) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = ema_np(g, period)
    al = ema_np(l, period)
    al_safe = np.where(al == 0, 1e-10, al)
    rs = ag / al_safe
    return 100 - (100 / (1 + rs))


# ── Signal Generation (per coin, all strategies at once) ───────────────

def generate_signals_batch(close: np.ndarray, strategies: np.ndarray) -> np.ndarray:
    """
    For one coin's close data, generate signals for ALL strategies at once.
    Returns: array of shape (N_strategies,) with values: 0=no signal, 1=long, -1=short
    """
    n = len(close)
    if n < 25:
        return np.zeros(len(strategies), dtype=np.int8)

    signals = np.zeros(len(strategies), dtype=np.int8)

    # Pre-compute indicators for all parameter combos we need
    ema_cache = {}
    rsi_cache = {}
    for period in set(strategies[:, 0].astype(int)) | set(strategies[:, 1].astype(int)):
        ema_cache[period] = ema_np(close, period)
    for period in set(strategies[:, 2].astype(int)):
        rsi_cache[period] = rsi_np(close, period)

    # Price momentum
    c0, c1, c2 = close[-1], close[-2], close[-3]
    pct1 = (c0 - c1) / c1 * 100
    pct2 = (c1 - c2) / c2 * 100

    # Recent high/low for breakout
    lookback = min(10, n - 1)
    recent_high = np.max(close[-lookback - 1:-1])
    recent_low = np.min(close[-lookback - 1:-1])

    for i in range(len(strategies)):
        s = strategies[i]
        ef, es, rp = int(s[0]), int(s[1]), int(s[2])
        rb, rs_val, stype = s[3], s[4], int(s[7])

        ema_f = ema_cache[ef]
        ema_s = ema_cache[es]
        rsi = rsi_cache[rp]
        curr_rsi = rsi[-1]

        if stype == 0:  # momentum
            # EMA crossover
            if ema_f[-1] > ema_s[-1] and ema_f[-2] <= ema_s[-2] and curr_rsi < rb:
                signals[i] = 1
                continue
            if ema_f[-1] < ema_s[-1] and ema_f[-2] >= ema_s[-2] and curr_rsi > rs_val:
                signals[i] = -1
                continue
            # Momentum continuation
            if pct1 > 0.03 and pct2 > 0 and curr_rsi < rb:
                signals[i] = 1
                continue
            if pct1 < -0.03 and pct2 < 0 and curr_rsi > rs_val:
                signals[i] = -1
                continue

        elif stype == 1:  # mean reversion
            if curr_rsi < rb - 15:
                signals[i] = 1
                continue
            if curr_rsi > rs_val + 15:
                signals[i] = -1
                continue
            dev = (c0 - ema_s[-1]) / ema_s[-1] * 100
            if dev < -0.15 and curr_rsi < rb:
                signals[i] = 1
                continue
            if dev > 0.15 and curr_rsi > rs_val:
                signals[i] = -1
                continue

        elif stype == 2:  # breakout
            if c0 > recent_high and curr_rsi < rs_val:
                signals[i] = 1
                continue
            if c0 < recent_low and curr_rsi > rb:
                signals[i] = -1
                continue

    return signals


# ── Bitget API ─────────────────────────────────────────────────────────

async def fetch_symbols(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(f"{BITGET_REST}/api/v2/mix/market/tickers",
                            params={"productType": "USDT-FUTURES"}, timeout=10)
    data = resp.json()
    if data.get("code") != "00000":
        return []
    tickers = data.get("data", [])
    tickers.sort(key=lambda t: float(t.get("quoteVolume", 0) or 0), reverse=True)
    return [t["symbol"] for t in tickers[:TOP_N_COINS]]

async def fetch_candles_batch(client: httpx.AsyncClient, symbols: list[str]) -> dict[str, np.ndarray]:
    sem = asyncio.Semaphore(25)
    results = {}
    async def fetch_one(sym):
        async with sem:
            try:
                resp = await client.get(f"{BITGET_REST}/api/v2/mix/market/candles",
                    params={"symbol": sym, "productType": "USDT-FUTURES", "granularity": "1m", "limit": str(CANDLE_LIMIT)},
                    timeout=8)
                data = resp.json()
                if data.get("code") != "00000" or not data.get("data"):
                    return
                rows = data["data"]
                close = np.array([float(r[4]) for r in rows])
                if len(rows) > 1 and float(rows[0][0]) > float(rows[1][0]):
                    close = close[::-1]
                results[sym] = close
            except Exception:
                pass
    await asyncio.gather(*[fetch_one(s) for s in symbols])
    return results


# ── Main Engine ────────────────────────────────────────────────────────

async def main():
    strategies = build_grid()
    n_strats = len(strategies)
    print(f"=" * 80, flush=True)
    print(f"  MASS STRATEGY RUNNER v2", flush=True)
    print(f"  Strategy variants: {n_strats:,}", flush=True)
    print(f"  Target coins: {TOP_N_COINS}", flush=True)
    print(f"  Paper equity per combo: ${PAPER_EQUITY}", flush=True)
    print(f"  Elimination threshold: ${ELIMINATION_THRESHOLD}", flush=True)
    print(f"=" * 80, flush=True)

    trade_log = open(TRADE_LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")

    async with httpx.AsyncClient() as client:
        print(f"[*] Fetching symbols...", flush=True)
        symbols = await fetch_symbols(client)
        n_coins = len(symbols)
        total_combos = n_coins * n_strats
        print(f"[*] {n_coins} coins x {n_strats:,} strategies = {total_combos:,} combos", flush=True)
        print(f"[*] Starting scan loop...\n", flush=True)

        # State arrays — flat: index = coin_idx * n_strats + strat_idx
        equity = np.full(total_combos, PAPER_EQUITY, dtype=np.float64)
        has_pos = np.zeros(total_combos, dtype=np.bool_)
        pos_side = np.zeros(total_combos, dtype=np.int8)  # 1=long, -1=short
        pos_entry = np.zeros(total_combos, dtype=np.float64)
        pos_sl = np.zeros(total_combos, dtype=np.float64)
        pos_tp = np.zeros(total_combos, dtype=np.float64)
        alive = np.ones(total_combos, dtype=np.bool_)  # False = eliminated
        pnl_total = np.zeros(total_combos, dtype=np.float64)
        trade_count = np.zeros(total_combos, dtype=np.int32)
        win_count = np.zeros(total_combos, dtype=np.int32)

        cycle = 0
        total_trades = 0
        start_time = time.time()
        last_report = start_time

        while True:
            cycle += 1
            t0 = time.time()

            candle_data = await fetch_candles_batch(client, symbols)

            cycle_trades = 0
            cycle_signals = 0

            for coin_idx, sym in enumerate(symbols):
                if sym not in candle_data:
                    continue
                close = candle_data[sym]
                if len(close) < 10:
                    continue
                current_price = close[-1]

                base = coin_idx * n_strats

                # ── Check exits for this coin ──
                coin_has_pos = has_pos[base:base + n_strats]
                coin_alive = alive[base:base + n_strats]
                active_mask = coin_has_pos & coin_alive

                if np.any(active_mask):
                    idxs = np.where(active_mask)[0]
                    for local_i in idxs:
                        gi = base + local_i
                        side = pos_side[gi]
                        hit_sl = (side == 1 and current_price <= pos_sl[gi]) or \
                                 (side == -1 and current_price >= pos_sl[gi])
                        hit_tp = (side == 1 and current_price >= pos_tp[gi]) or \
                                 (side == -1 and current_price <= pos_tp[gi])

                        if hit_sl or hit_tp:
                            if side == 1:
                                pnl_pct = (current_price - pos_entry[gi]) / pos_entry[gi]
                            else:
                                pnl_pct = (pos_entry[gi] - current_price) / pos_entry[gi]
                            pnl = pnl_pct * equity[gi]
                            equity[gi] += pnl
                            pnl_total[gi] += pnl
                            trade_count[gi] += 1
                            if pnl > 0:
                                win_count[gi] += 1
                            has_pos[gi] = False
                            cycle_trades += 1

                            reason = "TP" if hit_tp else "SL"
                            side_str = "long" if side == 1 else "short"
                            trade_log.write(
                                f"{datetime.now().isoformat()}|{sym}|S{local_i}|{side_str}|"
                                f"entry={pos_entry[gi]:.6f}|exit={current_price:.6f}|"
                                f"{reason}|PnL={pnl:+.4f}|eq={equity[gi]:.2f}\n"
                            )

                            # Eliminate if busted
                            if equity[gi] < ELIMINATION_THRESHOLD:
                                alive[gi] = False

                # ── Generate new signals ──
                sigs = generate_signals_batch(close, strategies)
                open_mask = (~has_pos[base:base + n_strats]) & alive[base:base + n_strats] & (sigs != 0)
                open_idxs = np.where(open_mask)[0]
                cycle_signals += len(open_idxs)

                for local_i in open_idxs:
                    gi = base + local_i
                    sig = sigs[local_i]
                    sl_pct = strategies[local_i, 5]
                    tp_pct = strategies[local_i, 6]

                    if sig == 1:  # long
                        pos_sl[gi] = current_price * (1 - sl_pct)
                        pos_tp[gi] = current_price * (1 + tp_pct)
                    else:  # short
                        pos_sl[gi] = current_price * (1 + sl_pct)
                        pos_tp[gi] = current_price * (1 - tp_pct)

                    pos_side[gi] = sig
                    pos_entry[gi] = current_price
                    has_pos[gi] = True

            total_trades += cycle_trades
            elapsed = time.time() - t0
            n_alive = int(np.sum(alive))
            n_active = int(np.sum(has_pos & alive))
            runtime_min = (time.time() - start_time) / 60

            print(
                f"\r[Cycle {cycle:4d}] {len(candle_data):3d} coins | "
                f"Sigs {cycle_signals:5d} | "
                f"Exits {cycle_trades:4d} | "
                f"Active {n_active:6d} | "
                f"Alive {n_alive:,}/{total_combos:,} | "
                f"Trades {total_trades:6d} | "
                f"PnL {np.sum(pnl_total):+.2f} | "
                f"{elapsed:.1f}s    ",
                end="", flush=True,
            )

            # Periodic report
            if time.time() - last_report >= REPORT_INTERVAL:
                last_report = time.time()
                _print_leaderboard(
                    strategies, symbols, n_strats,
                    equity, pnl_total, trade_count, win_count, alive, has_pos,
                    runtime_min, total_trades
                )

            # Refresh symbols
            if cycle % 300 == 0:
                new_syms = await fetch_symbols(client)
                if new_syms and len(new_syms) >= 50:
                    # Keep existing state for overlapping symbols
                    symbols = new_syms
                    print(f"\n[*] Refreshed: {len(symbols)} symbols", flush=True)

            await asyncio.sleep(TICK_INTERVAL)


def _print_leaderboard(strategies, symbols, n_strats, equity, pnl_total, trade_count, win_count, alive, has_pos, runtime_min, total_trades):
    total_combos = len(equity)
    n_alive = int(np.sum(alive))
    n_dead = total_combos - n_alive
    total_pnl = float(np.sum(pnl_total))

    # Aggregate PnL by strategy
    strat_pnl = np.zeros(n_strats)
    strat_trades = np.zeros(n_strats, dtype=np.int32)
    strat_wins = np.zeros(n_strats, dtype=np.int32)
    for ci in range(len(symbols)):
        base = ci * n_strats
        strat_pnl += pnl_total[base:base + n_strats]
        strat_trades += trade_count[base:base + n_strats]
        strat_wins += win_count[base:base + n_strats]

    # Top 15 strategies
    top_idxs = np.argsort(strat_pnl)[::-1][:15]
    worst_idxs = np.argsort(strat_pnl)[:5]

    # Aggregate by coin
    coin_pnl = {}
    coin_trades = {}
    for ci, sym in enumerate(symbols):
        base = ci * n_strats
        coin_pnl[sym] = float(np.sum(pnl_total[base:base + n_strats]))
        coin_trades[sym] = int(np.sum(trade_count[base:base + n_strats]))
    sorted_coins = sorted(coin_pnl.items(), key=lambda x: x[1], reverse=True)

    print(f"\n\n{'='*90}", flush=True)
    print(f"  LEADERBOARD | {runtime_min:.1f}min | {total_trades:,} trades | PnL: {total_pnl:+.2f} | Alive: {n_alive:,} | Dead: {n_dead:,}", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"\n  TOP 15 STRATEGIES:", flush=True)
    for idx in top_idxs:
        label = strat_label(strategies[idx])
        tr = int(strat_trades[idx])
        w = int(strat_wins[idx])
        wr = (w / max(tr, 1)) * 100
        print(f"    {label:55s}  PnL: {strat_pnl[idx]:+10.2f}  Trades: {tr:5d}  WR: {wr:.0f}%", flush=True)

    print(f"\n  WORST 5 STRATEGIES:", flush=True)
    for idx in worst_idxs:
        label = strat_label(strategies[idx])
        tr = int(strat_trades[idx])
        print(f"    {label:55s}  PnL: {strat_pnl[idx]:+10.2f}  Trades: {tr:5d}", flush=True)

    print(f"\n  TOP 10 COINS:", flush=True)
    for sym, pnl in sorted_coins[:10]:
        tr = coin_trades.get(sym, 0)
        print(f"    {sym:20s}  PnL: {pnl:+10.2f}  Trades: {tr:5d}", flush=True)

    print(f"\n  WORST 5 COINS:", flush=True)
    for sym, pnl in sorted_coins[-5:]:
        tr = coin_trades.get(sym, 0)
        print(f"    {sym:20s}  PnL: {pnl:+10.2f}  Trades: {tr:5d}", flush=True)

    print(f"{'='*90}\n", flush=True)

    # Save report
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "runtime_min": runtime_min,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "alive": n_alive,
            "dead": n_dead,
            "top_strategies": [
                {"idx": int(idx), "label": strat_label(strategies[idx]), "pnl": float(strat_pnl[idx]),
                 "trades": int(strat_trades[idx]), "wins": int(strat_wins[idx])}
                for idx in top_idxs
            ],
            "top_coins": [{"symbol": sym, "pnl": pnl, "trades": coin_trades.get(sym, 0)} for sym, pnl in sorted_coins[:20]],
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[*] Stopped.", flush=True)
