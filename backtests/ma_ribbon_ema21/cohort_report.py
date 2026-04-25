"""Aggregate phase1 events into per-cohort stats and write a markdown report."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import math
from datetime import datetime, timezone
from statistics import mean, median

from backtests.ma_ribbon_ema21.phase1_engine import Phase1Event
from backtests.ma_ribbon_ema21.distance_features import distance_bucket, DEFAULT_BUCKETS


@dataclass
class CohortStats:
    symbol: str
    tf: str
    bucket: str
    split: str          # "train" | "test"
    count: int
    mean_return_post_fee:   float
    median_return_post_fee: float
    win_rate:               float
    worst_return_post_fee:  float


def aggregate_cohorts(
    events: list[Phase1Event],
    horizon: int,
    buckets: list[tuple[float, float]] | None = None,
) -> list[CohortStats]:
    buckets = buckets or DEFAULT_BUCKETS
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for ev in events:
        ret = ev.forward_returns_post_fee.get(horizon)
        if ret is None or math.isnan(ret):
            continue
        bucket = distance_bucket(ev.distance_to_ma5_pct, buckets)
        groups[(ev.symbol, ev.tf, bucket, ev.split)].append(ret)

    out: list[CohortStats] = []
    for (sym, tf, bucket, split), rets in groups.items():
        wins = sum(1 for r in rets if r > 0)
        out.append(CohortStats(
            symbol=sym, tf=tf, bucket=bucket, split=split,
            count=len(rets),
            mean_return_post_fee=mean(rets),
            median_return_post_fee=median(rets),
            win_rate=wins / len(rets) if rets else 0.0,
            worst_return_post_fee=min(rets),
        ))
    return out


def write_markdown_report(
    cohorts: list[CohortStats],
    output_path: str,
    horizon: int,
) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cohorts_sorted = sorted(
        cohorts,
        key=lambda c: (c.symbol, c.tf, c.split, c.bucket)
    )

    lines: list[str] = []
    lines.append(f"# Phase 1 cohort report (horizon = +{horizon} bars)")
    lines.append("")
    lines.append(f"_Generated {now}_")
    lines.append("")
    lines.append("Each row: one (symbol, TF, distance-bucket, split) cohort.")
    lines.append("`mean_post_fee` is the mean forward return after subtracting "
                 "round-trip taker fee + slippage (0.05% x 2 + 0.01% x 2 = 0.12%).")
    lines.append("")
    lines.append("| symbol | TF | bucket | split | count | mean_post_fee | median | winrate | worst |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for c in cohorts_sorted:
        lines.append(
            f"| {c.symbol} | {c.tf} | {c.bucket} | {c.split} | {c.count} | "
            f"{c.mean_return_post_fee:+.4f} | {c.median_return_post_fee:+.4f} | "
            f"{c.win_rate:.2%} | {c.worst_return_post_fee:+.4f} |"
        )
    p.write_text("\n".join(lines) + "\n")
    return p
