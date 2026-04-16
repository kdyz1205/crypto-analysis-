"""Batch pattern database builder — scans many symbols/timeframes at once.

Designed to run as a background task. Tracks progress in a JSON file so the
UI can show real-time status.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field

PROGRESS_FILE = Path(__file__).parent.parent / "data" / "patterns" / "_batch_progress.json"
_current_task: asyncio.Task | None = None


@dataclass
class BatchProgress:
    status: str = "idle"           # idle | running | completed | failed
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_patterns: int = 0
    current_job: str = ""
    started_at: float = 0.0
    last_update: float = 0.0
    finished_at: float = 0.0
    results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def save(self):
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.last_update = time.time()
        PROGRESS_FILE.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")


def load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return asdict(BatchProgress())
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return asdict(BatchProgress())


async def build_batch(
    symbols: list[str],
    timeframes: list[str],
    days: int = 730,
    pivot_window: int = 3,
    lookahead_bars: int = 50,
    skip_existing: bool = True,
) -> dict:
    """Sequentially build pattern databases for all (symbol, timeframe) pairs.

    Sequential (not parallel) because we're calling Bitget API and we want to
    respect rate limits. Progress is written to disk after each job.

    skip_existing=True — skip (symbol, timeframe) pairs that already have a
    fresh DB file (within 12 hours). This makes batch rebuild resumable after
    server restarts.
    """
    from .pattern_engine import scan_historical_patterns, save_patterns, DATA_ROOT
    from server.data_service import get_ohlcv_with_df
    import pandas as pd

    progress = BatchProgress(
        status="running",
        total_jobs=len(symbols) * len(timeframes),
        started_at=time.time(),
    )
    progress.save()
    skip_threshold = 12 * 3600  # 12 hours

    for symbol in symbols:
        for tf in timeframes:
            job_name = f"{symbol}_{tf}"
            progress.current_job = job_name
            progress.save()

            # Skip if file already exists and is fresh
            if skip_existing:
                db_path = DATA_ROOT / f"{symbol}_{tf}.jsonl"
                if db_path.exists():
                    age = time.time() - db_path.stat().st_mtime
                    if age < skip_threshold:
                        n = sum(1 for _ in open(db_path, encoding='utf-8'))
                        progress.completed_jobs += 1
                        progress.total_patterns += n
                        progress.results.append({
                            "symbol_timeframe": job_name,
                            "patterns": n,
                            "bars": 0,
                            "skipped": True,
                        })
                        progress.save()
                        continue

            try:
                # Use history_mode="full_history" to get max available per symbol.
                df_polars, _ = await get_ohlcv_with_df(
                    symbol, tf, days=days, history_mode="full_history",
                )
                if df_polars is None or df_polars.is_empty():
                    progress.failed_jobs += 1
                    progress.errors.append({"job": job_name, "error": "no market data"})
                    progress.save()
                    continue

                pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
                pdf = pdf.rename(columns={"open_time": "timestamp"})
                for col in ("open", "high", "low", "close", "volume"):
                    pdf[col] = pd.to_numeric(pdf[col], errors="raise")

                # Cap max bars: 15m × 6 years = 210k bars → too slow for O(N^2) scan.
                # Use last N bars; N tuned per timeframe for reasonable compute time.
                max_bars = {"15m": 20000, "1h": 20000, "4h": 20000, "1d": 20000}.get(tf, 20000)
                if len(pdf) > max_bars:
                    pdf = pdf.iloc[-max_bars:].reset_index(drop=True)

                # CPU-heavy pattern scan runs in thread pool to avoid blocking event loop
                def _scan():
                    return scan_historical_patterns(
                        pdf, symbol, tf,
                        pivot_window=pivot_window,
                        lookahead_bars=lookahead_bars,
                    )

                records = await asyncio.to_thread(_scan)
                await asyncio.to_thread(save_patterns, records, symbol, tf)

                progress.completed_jobs += 1
                progress.total_patterns += len(records)
                progress.results.append({
                    "symbol_timeframe": job_name,
                    "patterns": len(records),
                    "bars": len(pdf),
                })
                progress.save()

                # Small delay to respect rate limits
                await asyncio.sleep(0.5)
            except Exception as e:
                progress.failed_jobs += 1
                progress.errors.append({"job": job_name, "error": str(e)[:200]})
                progress.save()

    progress.status = "completed"
    progress.finished_at = time.time()
    progress.current_job = ""
    progress.save()
    return asdict(progress)


async def start_batch_build(symbols: list[str], timeframes: list[str], days: int = 730) -> dict:
    """Launch batch build as a background task. Returns immediately.

    If a build is already running, returns the current progress instead.
    """
    global _current_task

    # Check if already running
    if _current_task and not _current_task.done():
        return {
            "ok": False,
            "error": "batch build already running",
            "progress": load_progress(),
        }

    _current_task = asyncio.create_task(
        build_batch(symbols, timeframes, days=days)
    )
    # Give it a moment to initialize
    await asyncio.sleep(0.1)
    return {"ok": True, "started": True, "progress": load_progress()}


def is_running() -> bool:
    return _current_task is not None and not _current_task.done()
