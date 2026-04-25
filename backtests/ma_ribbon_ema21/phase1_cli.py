"""Phase 1 CLI driver. Reads JSON config, runs scan, writes report."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

from backtests.ma_ribbon_ema21.ma_alignment import AlignmentConfig
from backtests.ma_ribbon_ema21.data_loader import DataLoaderConfig
from backtests.ma_ribbon_ema21.phase1_engine import (
    UniverseConfig, scan_universe,
)
from backtests.ma_ribbon_ema21.cohort_report import (
    aggregate_cohorts, write_markdown_report,
)
from backtests.ma_ribbon_ema21.acceptance_gate import (
    evaluate_phase1_gate,
)


_LOG = logging.getLogger("phase1")


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def run_phase1(config_path: str, output_path: str) -> dict:
    _setup_logging()
    cfg_data = json.loads(Path(config_path).read_text())
    universe = UniverseConfig(
        symbols=cfg_data["universe"],
        timeframes=cfg_data["timeframes"],
        loader=DataLoaderConfig(cache_dir=cfg_data["data_cache_dir"]),
        alignment_cfg=AlignmentConfig.from_dict(cfg_data["bullish_alignment"]),
        forward_horizons=tuple(cfg_data["forward_return_bars"]),
        fee_per_side=cfg_data["fees"]["per_side"],
        slippage_per_fill=cfg_data["fees"]["slippage_per_fill"],
        train_pct=cfg_data["data_split"]["train_pct"],
    )
    _LOG.info("Phase 1 starting: %d symbols x %d TFs",
              len(universe.symbols), len(universe.timeframes))
    events = scan_universe(universe)
    _LOG.info("scan complete: %d total events", len(events))

    primary_horizon = 20
    cohorts = aggregate_cohorts(events, horizon=primary_horizon)
    write_markdown_report(cohorts, output_path=output_path,
                          horizon=primary_horizon)
    _LOG.info("report written: %s", output_path)

    gate = evaluate_phase1_gate(cohorts, horizon=primary_horizon,
                                threshold_pct=0.01, min_symbol_pct=0.30)
    _LOG.info("gate %s -- %s",
              "PASS" if gate.passed else "FAIL", gate.reason)
    return {
        "total_events": len(events),
        "report_path":  output_path,
        "gate": {
            "passed": gate.passed,
            "reason": gate.reason,
            "passing_symbols": gate.passing_symbols,
            "failing_symbols": gate.failing_symbols,
            "symbols_total": gate.symbols_total,
            "symbols_passing": gate.symbols_passing,
        }
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MA-ribbon Phase 1 backtest")
    p.add_argument("--config", required=True, help="path to phase1 JSON config")
    p.add_argument("--output", required=True, help="path to write the markdown report")
    args = p.parse_args(argv)
    summary = run_phase1(args.config, args.output)
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
