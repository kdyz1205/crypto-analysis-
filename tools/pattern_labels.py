"""Pattern Labels — human-in-the-loop pattern annotation.

Traders can mark patterns with:
- quality: good | bad | neutral
- similar_to: another pattern_id (positive pair for metric learning)
- different_from: another pattern_id (negative pair)
- notes: free text

These labels feed into:
1. Learned distance metric (positive pairs pulled together, negative pairs pushed apart)
2. Outcome weighting (human labels override outcome-based labels when conflict)
3. Auto-discovery training data
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

LABELS_DIR = Path(__file__).parent.parent / "data" / "pattern_labels"


@dataclass
class PatternLabel:
    """A human annotation on a pattern."""
    label_id: str = ""
    pattern_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    quality: str = ""           # good | bad | neutral
    tags: list[str] = field(default_factory=list)  # e.g. ["wedge", "channel"]
    similar_to: list[str] = field(default_factory=list)   # pattern_ids
    different_from: list[str] = field(default_factory=list)
    notes: str = ""
    labeled_by: str = "user"
    labeled_at: float = field(default_factory=time.time)


def _label_file(symbol: str, timeframe: str) -> Path:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    return LABELS_DIR / f"{symbol}_{timeframe}.jsonl"


def save_label(label: PatternLabel) -> dict:
    """Persist a label (appends to jsonl)."""
    from .types import new_id
    if not label.label_id:
        label.label_id = new_id()
    path = _label_file(label.symbol, label.timeframe)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(label), default=str) + "\n")
    return asdict(label)


def list_labels(symbol: str, timeframe: str) -> list[dict]:
    """Load all labels for a symbol/timeframe."""
    path = _label_file(symbol, timeframe)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def get_label_for_pattern(symbol: str, timeframe: str, pattern_id: str) -> dict | None:
    labels = list_labels(symbol, timeframe)
    for l in reversed(labels):  # most recent wins
        if l.get("pattern_id") == pattern_id:
            return l
    return None


def get_positive_pairs(symbol: str, timeframe: str) -> list[tuple[str, str]]:
    """Return pairs of (pattern_a_id, pattern_b_id) that user marked as similar."""
    pairs = []
    for l in list_labels(symbol, timeframe):
        pid = l.get("pattern_id", "")
        for other in l.get("similar_to", []):
            if pid and other:
                pairs.append((pid, other))
    return pairs


def get_negative_pairs(symbol: str, timeframe: str) -> list[tuple[str, str]]:
    """Return pairs of (pattern_a_id, pattern_b_id) that user marked as different."""
    pairs = []
    for l in list_labels(symbol, timeframe):
        pid = l.get("pattern_id", "")
        for other in l.get("different_from", []):
            if pid and other:
                pairs.append((pid, other))
    return pairs


def get_quality_labels(symbol: str, timeframe: str) -> dict[str, str]:
    """Return {pattern_id: quality} — good/bad/neutral."""
    out = {}
    for l in list_labels(symbol, timeframe):
        pid = l.get("pattern_id", "")
        q = l.get("quality", "")
        if pid and q:
            out[pid] = q  # later labels override earlier
    return out
