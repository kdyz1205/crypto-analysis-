"""End-to-end round-trip on the user's real data directories.

Smoke-test only — asserts the pipeline runs and the metrics summary
object has the expected keys. Numeric bounds are checked by the CLI
report, not by this test.
"""
from pathlib import Path

import pytest

from trendline_tokenizer.adapters import load_manual_records, load_legacy_pattern_records
from trendline_tokenizer.tokenizer import encode_rule, decode_rule
from trendline_tokenizer.tokenizer.metrics import round_trip_error, summarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_PATH = PROJECT_ROOT / "data" / "manual_trendlines.json"
PATTERNS_DIR = PROJECT_ROOT / "data" / "patterns"


def test_manual_records_load_when_present():
    if not MANUAL_PATH.exists():
        pytest.skip("no manual trendlines file")
    recs = load_manual_records(MANUAL_PATH)
    assert isinstance(recs, list)
    # not asserting count — user may clear the store


def test_round_trip_on_legacy_sample():
    if not PATTERNS_DIR.exists():
        pytest.skip("no legacy patterns dir")
    recs = load_legacy_pattern_records(PATTERNS_DIR, limit=200)
    if not recs:
        pytest.skip("no legacy patterns rows")
    errs = []
    for r in recs:
        t = encode_rule(r)
        d = decode_rule(t, reference_record=r)
        errs.append(round_trip_error(r, d))
    summary = summarize(errs)
    assert summary["n"] == len(recs)
    assert "aggregate_median" in summary
    assert "role_accuracy" in summary
    # Role must round-trip exactly (coarse token contains it).
    assert summary["role_accuracy"] == 1.0
