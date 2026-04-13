"""Reflection agent prompt template.

Produces a failure-pattern report from a variant's traces, used by the
architect to propose targeted mutations. Hermes-GEPA style: mutation must
be trace-informed, not random.

The prompt gives the agent:
  1. The variant's aggregate metrics (overall + per-touch-number + per-vol-regime + per-side)
  2. A sampled subset of raw traces (for variety checking)
  3. A strict output contract (JSON schema) so the orchestrator can parse
     the response deterministically.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from .trace import read_traces_jsonl


REFLECTION_OUTPUT_SCHEMA = {
    "failure_clusters": [
        {
            "cluster_id": "short string id",
            "hypothesis": "natural-language hypothesis about the cause",
            "evidence": {
                "feature": "e.g. span_bars<30, slope_pct_per_bar>0.002, setup_touch_number=2",
                "n_affected": "int",
                "win_rate": "float",
                "avg_R": "float",
            },
            "proposed_fix": "natural-language description of what the variant should change",
            "priority": "high|medium|low",
        }
    ],
    "winning_patterns": [
        {"feature": "string", "win_rate": "float", "avg_R": "float", "n": "int"}
    ],
    "overall_diagnosis": "2-3 sentence summary",
    "recommended_variants": [
        "short description of each targeted mutation, max 6"
    ],
}


REFLECTION_SYSTEM_PROMPT = """\
You are a reflection agent for a trendline detection evolution loop. Your job
is to read execution traces from a backtested trendline variant and identify
WHY it underperforms — not just THAT it underperforms.

You will receive:
1. Aggregate metrics (overall + sliced by touch_number, vol_regime, side)
2. A sampled subset of raw SetupTrace rows
3. The variant's name

Your output MUST be a single JSON object matching the schema. No prose outside
the JSON. No markdown. Just one top-level {…}.

Key analytical priorities:
- Per setup_touch_number analysis: does edge get better or worse as we wait
  for more touches? This is THE central question — the user's thesis is that
  later-touch setups have higher edge. Validate or refute.
- Feature correlations with negative EV: span_bars, slope_pct_per_bar,
  vol_regime, side, symbol, timeframe. Look for clusters of failures.
- Feature correlations with positive EV: what does the variant already get
  right? Don't break those.
- Flip leg performance: is the flip actually capturing breakout edge, or
  just adding noise?

Output rules:
- Cluster hypotheses must be CONCRETE and TESTABLE (a later variant can check it).
- Evidence must be NUMERIC (n, win_rate, avg_R). No vague "some lines".
- Proposed fixes must be IMPLEMENTATION-ACTIONABLE (a coder can do it).
- Priority: high = would change winner picking, medium = measurable but secondary,
  low = edge case.
"""


def build_reflection_input(
    variant_name: str,
    summary: dict,
    traces_path: Path,
    sample_size: int = 150,
) -> str:
    """Build the user-message payload passed to the reflection agent."""
    traces = read_traces_jsonl(traces_path)
    if len(traces) > sample_size:
        rng = random.Random(42)
        sample = rng.sample(traces, sample_size)
    else:
        sample = traces

    payload = {
        "variant": variant_name,
        "aggregate_metrics": summary,
        "sample_traces": sample,
        "output_schema": REFLECTION_OUTPUT_SCHEMA,
    }
    return json.dumps(payload, indent=2, default=str)
