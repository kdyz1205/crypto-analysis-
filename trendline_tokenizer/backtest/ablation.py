"""Ablation runner: train/load multiple model variants (rule-only,
learned-only, raw-only, all-three) and compare their backtest metrics
on the same data slice.

Variants share dataset, replay window, and signal-engine config; they
differ only in `use_rule_tokens / use_learned_tokens / use_raw_features`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

from .metrics import TradeMetrics, summarize_metrics


@dataclass
class AblationRow:
    variant_name: str
    artifact_name: str
    metrics: TradeMetrics


@dataclass
class AblationReport:
    rows: list[AblationRow] = field(default_factory=list)

    def add(self, variant_name: str, artifact_name: str, metrics: TradeMetrics):
        self.rows.append(AblationRow(variant_name, artifact_name, metrics))

    def render(self) -> str:
        lines = ["Variant | Artifact | Metrics"]
        for r in self.rows:
            lines.append(f"  {r.variant_name:>12} | {r.artifact_name} | {summarize_metrics(r.metrics)}")
        if len(self.rows) > 1:
            lines.append("")
            best = max(self.rows, key=lambda r: r.metrics.cumulative_return_pct)
            lines.append(f"Best by cumulative return: {best.variant_name} "
                         f"({best.metrics.cumulative_return_pct:+.2%})")
        return "\n".join(lines)


VARIANTS: dict[str, dict] = {
    "rule_only":    {"use_rule_tokens": True,  "use_learned_tokens": False, "use_raw_features": False},
    "learned_only": {"use_rule_tokens": False, "use_learned_tokens": True,  "use_raw_features": False},
    "raw_only":     {"use_rule_tokens": False, "use_learned_tokens": False, "use_raw_features": True},
    "rule_plus_raw": {"use_rule_tokens": True, "use_learned_tokens": False, "use_raw_features": True},
    "all":          {"use_rule_tokens": True,  "use_learned_tokens": True,  "use_raw_features": True},
}
