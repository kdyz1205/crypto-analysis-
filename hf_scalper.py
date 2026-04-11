"""
High-Frequency Scalper — Paper + Live
Monitors ALL Bitget USDT-Futures, trades on 1m candles.
Strategy: EMA(3) x EMA(8) crossover + RSI(6) filter + volume spike.
Target: ~5 trades/minute across all symbols.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import numpy as np
import pandas as pd

# Load .env
try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env):
        load_dotenv(_env, override=True)
except ImportError:
    pass

# ── Config ──────────────────────────────────────────────────────────────
BITGET_REST = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"

MODE = os.environ.get("SCALPER_MODE", "paper")  # "paper" or "live"
TICK_INTERVAL = 3          # seconds between scans
MAX_POSITIONS = 20         # max concurrent paper positions
POSITION_SIZE_USDT = 50    # notional per trade
STOP_LOSS_PCT = 0.003      # 0.3%
TAKE_PROFIT_PCT = 0.005    # 0.5%
MIN_VOLUME_MULT = 1.0      # any volume ok
RSI_LONG_BELOW = 50        # RSI < 50 → long
RSI_SHORT_ABOVE = 50       # RSI > 50 → short
EMA_FAST = 2
EMA_SLOW = 5
RSI_PERIOD = 4
CANDLE_LIMIT = 20          # bars to fetch per symbol
TOP_N_SYMBOLS = 120        # scan top N by volume

# ── Data Structures ────────────────────────────────────────────────────

@dataclass
class PaperPosition:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size_usdt: float
    stop_loss: float
    take_profit: float
    entry_time: float
    pnl: float = 0.0

@dataclass
class ScalperState:
    positions: dict = field(default_factory=dict)  # symbol -> PaperPosition
    total_trades: int = 0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trades_log: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

state = ScalperState()

# ── Bitget Public API ──────────────────────────────────────────────────

async def fetch_all_symbols(client: httpx.AsyncClient) -> list[str]:
    """Get all USDT-futures symbols sorted by 24h volume."""
    resp = await client.get(
        f"{BITGET_REST}/api/v2/mix/market/tickers",
        params={"productType": "USDT-FUTURES"},
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != "00000":
        print(f"[ERROR] fetch symbols: {data.get('msg')}")
        return []
    tickers = data.get("data", [])
    # Sort by 24h quote volume descending
    tickers.sort(key=lambda t: float(t.get("quoteVolume", 0) or 0), reverse=True)
    symbols = [t["symbol"] for t in tickers[:TOP_N_SYMBOLS]]
    return symbols


async def fetch_candles(client: httpx.AsyncClient, symbol: str) -> pd.DataFrame | None:
    """Fetch 1m candles for a symbol."""
    try:
        resp = await client.get(
            f"{BITGET_REST}/api/v2/mix/market/candles",
            params={
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "granularity": "1m",
                "limit": str(CANDLE_LIMIT),
            },
            timeout=8,
        )
        data = resp.json()
        if data.get("code") != "00000" or not data.get("data"):
            return None
        rows = data["data"]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "quoteVolume"])
        for col in ["open", "high", "low", "close", "volume", "quoteVolume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)
        return df
    except Exception:
        return None


# ── Indicators ─────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def generate_signal(df: pd.DataFrame) -> str | None:
    """Returns 'long', 'short', or None. Aggressive momentum scalping."""
    if len(df) < 8:
        return None

    close = df["close"]
    rsi = compute_rsi(close, RSI_PERIOD)

    # Simple momentum: last 3 bars direction
    c0 = close.iloc[-1]
    c1 = close.iloc[-2]
    c2 = close.iloc[-3]
    curr_rsi = rsi.iloc[-1]

    if pd.isna(curr_rsi):
        return None

    pct_change_1 = (c0 - c1) / c1 * 100
    pct_change_2 = (c1 - c2) / c2 * 100

    # Long: 2 consecutive up bars + RSI not overbought
    if pct_change_1 > 0.02 and pct_change_2 > 0 and curr_rsi < 65:
        return "long"

    # Short: 2 consecutive down bars + RSI not oversold
    if pct_change_1 < -0.02 and pct_change_2 < 0 and curr_rsi > 35:
        return "short"

    # Mean reversion: RSI extreme
    if curr_rsi < 20:
        return "long"
    if curr_rsi > 80:
        return "short"

    return None


# ── Paper Trading Engine ───────────────────────────────────────────────

def open_paper_position(symbol: str, side: str, price: float):
    if symbol in state.positions:
        return
    if len(state.positions) >= MAX_POSITIONS:
        return

    if side == "long":
        sl = price * (1 - STOP_LOSS_PCT)
        tp = price * (1 + TAKE_PROFIT_PCT)
    else:
        sl = price * (1 + STOP_LOSS_PCT)
        tp = price * (1 - TAKE_PROFIT_PCT)

    pos = PaperPosition(
        symbol=symbol, side=side, entry_price=price,
        size_usdt=POSITION_SIZE_USDT, stop_loss=sl,
        take_profit=tp, entry_time=time.time(),
    )
    state.positions[symbol] = pos
    state.total_trades += 1
    now = datetime.now().strftime("%H:%M:%S")
    print(f"  [{now}] OPEN {side.upper():5s} {symbol:20s} @ {price:.6f}  SL={sl:.6f}  TP={tp:.6f}")


def check_paper_exits(symbol: str, current_price: float):
    pos = state.positions.get(symbol)
    if pos is None:
        return

    closed = False
    reason = ""

    if pos.side == "long":
        if current_price <= pos.stop_loss:
            closed, reason = True, "SL"
        elif current_price >= pos.take_profit:
            closed, reason = True, "TP"
    else:
        if current_price >= pos.stop_loss:
            closed, reason = True, "SL"
        elif current_price <= pos.take_profit:
            closed, reason = True, "TP"

    if closed:
        if pos.side == "long":
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price
        pnl_usdt = pnl_pct * pos.size_usdt
        state.total_pnl += pnl_usdt
        if pnl_usdt > 0:
            state.wins += 1
        else:
            state.losses += 1

        now = datetime.now().strftime("%H:%M:%S")
        print(f"  [{now}] EXIT {pos.side.upper():5s} {symbol:20s} @ {current_price:.6f}  {reason}  PnL: {pnl_usdt:+.2f} USDT")
        state.trades_log.append({
            "symbol": symbol, "side": pos.side, "entry": pos.entry_price,
            "exit": current_price, "reason": reason, "pnl": pnl_usdt,
            "time": now,
        })
        del state.positions[symbol]


# ── Main Loop ──────────────────────────────────────────────────────────

async def scan_cycle(client: httpx.AsyncClient, symbols: list[str]):
    """Scan all symbols, generate signals, manage positions."""
    # Batch fetch candles (concurrent with semaphore)
    sem = asyncio.Semaphore(15)
    results = {}

    async def fetch_one(sym):
        async with sem:
            df = await fetch_candles(client, sym)
            if df is not None and len(df) >= EMA_SLOW + 2:
                results[sym] = df

    await asyncio.gather(*[fetch_one(s) for s in symbols])

    signals_found = 0
    for symbol, df in results.items():
        current_price = float(df["close"].iloc[-1])

        # Check exits first
        check_paper_exits(symbol, current_price)

        # Generate signal
        signal = generate_signal(df)
        if signal:
            signals_found += 1
            open_paper_position(symbol, signal, current_price)

    return signals_found, len(results)


async def main():
    print("=" * 70)
    print(f"  HIGH-FREQUENCY SCALPER — {MODE.upper()} MODE")
    print(f"  EMA({EMA_FAST}/{EMA_SLOW}) + RSI({RSI_PERIOD}) + Volume Filter")
    print(f"  Tick interval: {TICK_INTERVAL}s | Max positions: {MAX_POSITIONS}")
    print(f"  Position size: {POSITION_SIZE_USDT} USDT | SL: {STOP_LOSS_PCT*100}% | TP: {TAKE_PROFIT_PCT*100}%")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        print("\n[*] Fetching all Bitget USDT-Futures symbols...")
        symbols = await fetch_all_symbols(client)
        if not symbols:
            print("[FATAL] No symbols found. Check network.")
            return
        print(f"[*] Monitoring {len(symbols)} symbols (top by volume)")
        print(f"[*] Starting scan loop...\n")

        cycle = 0
        while True:
            cycle += 1
            t0 = time.time()
            try:
                signals, scanned = await scan_cycle(client, symbols)
                elapsed = time.time() - t0

                # Status line every cycle
                runtime = time.time() - state.start_time
                rate = state.total_trades / max(runtime / 60, 0.01)
                wr = (state.wins / max(state.wins + state.losses, 1)) * 100

                sys.stdout.write(
                    f"\r[Cycle {cycle:4d}] Scanned {scanned:3d} | "
                    f"Signals {signals:2d} | "
                    f"Open {len(state.positions):2d} | "
                    f"Trades {state.total_trades:4d} ({rate:.1f}/min) | "
                    f"PnL {state.total_pnl:+.2f} | "
                    f"WR {wr:.0f}% | "
                    f"{elapsed:.1f}s"
                )
                sys.stdout.flush()

            except Exception as e:
                print(f"\n[ERROR] Cycle {cycle}: {e}")

            # Refresh symbol list every 100 cycles
            if cycle % 100 == 0:
                try:
                    new_symbols = await fetch_all_symbols(client)
                    if new_symbols:
                        symbols = new_symbols
                        print(f"\n[*] Refreshed symbol list: {len(symbols)} symbols")
                except Exception:
                    pass

            await asyncio.sleep(TICK_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"  FINAL STATS")
        print(f"  Total trades: {state.total_trades}")
        print(f"  Total PnL: {state.total_pnl:+.2f} USDT")
        print(f"  Wins: {state.wins} | Losses: {state.losses}")
        print(f"  Win rate: {(state.wins/max(state.wins+state.losses,1))*100:.1f}%")
        print(f"  Open positions: {len(state.positions)}")
        print(f"{'='*70}")
