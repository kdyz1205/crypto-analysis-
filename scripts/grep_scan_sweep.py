"""Grep-scan sweep for 'modified-but-not-everywhere' bugs.

Prior commits fixed specific call sites but left sibling sites unchanged.
This script runs 6 regex sweeps across the repo to find those orphans.

Usage:
    python scripts/grep_scan_sweep.py               # run all scans
    python scripts/grep_scan_sweep.py --scan current_pnl
    python scripts/grep_scan_sweep.py --json out.json
    python scripts/grep_scan_sweep.py --ci          # exit nonzero on findings

Exit codes:
    0 = no findings
    N = number of findings (useful for CI gating)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────
# Scan rule definitions
# ──────────────────────────────────────────────────────────────
@dataclass
class ScanRule:
    id: str
    title: str
    why: str                      # why we care (shown in report)
    pattern: str                  # python re.compile pattern
    file_globs: list[str]         # globs relative to repo root
    # Hit is INVALID (skipped) if ANY of these regexes match the same line
    line_excludes: list[str] = field(default_factory=list)
    # Hit is INVALID if the line context (±N lines) contains one of these
    context_excludes: list[tuple[int, str]] = field(default_factory=list)
    fix: str = ""                 # one-line guidance the fixer sees per hit
    severity: str = "MED"         # CRIT / HIGH / MED / LOW
    multiline: bool = False       # True = pattern may span lines (re.DOTALL)
    context_before: int = 0       # lines of before-context to capture
    context_after: int = 0        # lines of after-context to capture


RULES: list[ScanRule] = [
    # ──────────────────────────────────────────────────────────────
    # SCAN 1: current_pnl in frontend
    # ──────────────────────────────────────────────────────────────
    ScanRule(
        id="current_pnl",
        title="Frontend still reads current_pnl as real P&L",
        why=(
            "Phase 1 split P&L into pattern_virtual_pnl (simulated) and "
            "realized_pnl_usd (exchange fill). Any frontend that still "
            "reads current_pnl risks showing fake P&L as real."
        ),
        pattern=r"\bcurrent_pnl\b",
        file_globs=["frontend/**/*.js"],
        line_excludes=[
            # Already-acknowledged fallbacks that explicitly chain
            r"pattern_virtual_pnl\s*\?\?\s*[^;]*current_pnl",
        ],
        fix=(
            "Replace `i.current_pnl` with the correct path:\n"
            "  paper/simulated → `i.pattern_virtual_pnl ?? 0` + [虚拟] badge\n"
            "  real            → `i.realized_pnl_usd ?? 0` + exec-pnl-real class\n"
            "Never show current_pnl without distinguishing which one it is."
        ),
        severity="CRIT",
    ),

    # ──────────────────────────────────────────────────────────────
    # SCAN 2: Unescaped ${...error...} in frontend template literals
    # ──────────────────────────────────────────────────────────────
    ScanRule(
        id="xss_error_interp",
        title="Unescaped server error interpolated into HTML",
        why=(
            "Phase 5 removed XSS in decision_rail.js but views.js still "
            "renders ${resp.error}, ${e.message}, ${f.error}, ${wb.*.error} "
            "into innerHTML. Backend error text with HTML chars breaks "
            "Telegram / HTML rendering; attacker-controlled text executes."
        ),
        pattern=(
            # ${...error...} or ${...message...} or ${...reason...} or ${...msg...}
            r"\$\{[^}]*\.(error|message|reason|msg|detail)[^}]*\}"
        ),
        file_globs=["frontend/**/*.js", "frontend/**/*.html"],
        line_excludes=[
            # Allow if explicitly wrapped in esc()/escapeHtml()
            r"esc\([^)]*\.(error|message|reason|msg|detail)",
            r"escapeHtml\([^)]*\.(error|message|reason|msg|detail)",
            # Already using textContent instead of innerHTML on this line
            r"\.textContent\s*=",
        ],
        fix=(
            "Wrap the interpolated value with esc() or escapeHtml(). "
            "If the whole line is innerHTML assignment, consider rebuilding "
            "the DOM via createElement + textContent instead."
        ),
        severity="CRIT",
    ),

    # ──────────────────────────────────────────────────────────────
    # SCAN 3: match_pattern() calls missing current_time_position
    # ──────────────────────────────────────────────────────────────
    ScanRule(
        id="match_pattern_time_leak",
        title="match_pattern() call missing current_time_position",
        why=(
            "Phase 3 added current_time_position param to match_pattern() "
            "to prevent time leaks in live queries. But agent/pattern_driven.py "
            "and other callers don't actually pass it, so max_time_position "
            "defaults to None and ALL historical patterns are matched — "
            "including ones written in the future relative to the query bar."
        ),
        pattern=r"match_pattern\s*\(",
        file_globs=["agent/**/*.py", "server/**/*.py", "tools/**/*.py"],
        # This is a multi-line scan — we capture the full call and check the
        # context for the argument name. Done in a custom post-filter below.
        multiline=True,
        context_after=15,
        line_excludes=[
            # Function definition itself, not a call
            r"def\s+match_pattern",
        ],
        fix=(
            "Add `current_time_position=<value>` to the call:\n"
            "  - in LIVE scan: pass 1.0 (anchors are the most recent pattern)\n"
            "  - in BACKTEST: pass the normalized bar position within the DB\n"
            "If omitted, future patterns leak into the similarity search."
        ),
        severity="CRIT",
    ),

    # ──────────────────────────────────────────────────────────────
    # SCAN 4: find_similar() calls missing exclude_anchors
    # ──────────────────────────────────────────────────────────────
    ScanRule(
        id="find_similar_self_match",
        title="find_similar() call missing exclude_anchors",
        why=(
            "Phase 3 added self-exclusion in find_similar() (pattern_engine.py:874). "
            "But the only caller (match_pattern) doesn't pass exclude_anchors, "
            "so when a pattern has been written back to the DB it can match "
            "itself on the next query, inflating confidence."
        ),
        pattern=r"find_similar\s*\(",
        file_globs=["tools/**/*.py", "server/**/*.py", "agent/**/*.py"],
        multiline=True,
        context_after=15,
        line_excludes=[
            r"def\s+find_similar",
        ],
        fix=(
            "Add `exclude_anchors=(anchor1_idx, anchor2_idx)` to the call. "
            "Without it, any pattern written back via live writeback can "
            "match its own record on the next scan → false confidence."
        ),
        severity="CRIT",
    ),

    # ──────────────────────────────────────────────────────────────
    # SCAN 5: Inline onclick= in frontend (HTML attr or JS assignment)
    # ──────────────────────────────────────────────────────────────
    ScanRule(
        id="inline_onclick",
        title="Inline onclick= still present",
        why=(
            "Phase 5 removed inline onclick in decision_rail.js but views.js, "
            "main.js, and execution/panel.js still have:\n"
            "  (a) HTML-attribute form:  onclick=\"...\" / onclick='...'\n"
            "  (b) JS property form:     b.onclick = (...) => {}\n"
            "Both leak listeners on re-render + XSS-prone if the string "
            "interpolates server data."
        ),
        pattern=(
            r"(?:"
            r"\bonclick\s*=\s*[\"'][^\"'\n]+[\"']"                # HTML form
            r"|"
            r"\.\s*onclick\s*=\s*(?:async\s+)?(?:function|\(|[A-Za-z_]\w*\s*=>)"  # JS form
            r")"
        ),
        file_globs=["frontend/**/*.js", "frontend/**/*.html"],
        line_excludes=[
            # Clearing a handler (which is fine): `.onclick = null`
            r"\.onclick\s*=\s*null",
        ],
        fix=(
            "Use event delegation instead:\n"
            "  1. Add `data-action=\"xxx\"` attribute to the button HTML\n"
            "  2. Wire ONE addEventListener('click', ...) on the container\n"
            "  3. Dispatch on `e.target.closest('[data-action=xxx]')`\n"
            "Single-listener pattern avoids re-binding on every render."
        ),
        severity="HIGH",
    ),

    # ──────────────────────────────────────────────────────────────
    # SCAN 6: realized_pnl reads outside the types.py definition
    # ──────────────────────────────────────────────────────────────
    ScanRule(
        id="realized_pnl_undifferentiated",
        title="realized_pnl accessed without _sim / _usd suffix",
        why=(
            "Phase 1 split realized_pnl into realized_pnl_sim and "
            "realized_pnl_usd. Any code still reading/writing plain "
            "realized_pnl is operating on the legacy alias, which "
            "only updates in __post_init__ (drifts after mutation)."
        ),
        # Match realized_pnl: NOT followed by _ (so realized_pnl_sim/_usd pass),
        # NOT preceded by "un" (skip unrealized_pnl), NOT preceded by "daily_"
        # (skip daily_realized_pnl which is a different concept).
        pattern=r"(?<!un)(?<!daily_)\brealized_pnl(?!_)",
        file_globs=[
            "frontend/**/*.js",
            "server/**/*.py",
            "tools/**/*.py",
            "agent/**/*.py",
        ],
        line_excludes=[
            # The dataclass definition itself (legitimate)
            r"realized_pnl:\s*float",
            # Docstrings / comments explaining the legacy field
            r"^\s*#",
            r"realized_pnl.*legacy|legacy.*realized_pnl|backward.*realized_pnl",
            # __post_init__ sync logic in types.py (intentional)
            r"self\.realized_pnl\s*=\s*self\.realized_pnl_sim",
            r"self\.realized_pnl_sim\s*=\s*self\.realized_pnl",
        ],
        fix=(
            "Rename to realized_pnl_sim (simulated) or realized_pnl_usd "
            "(real exchange fill). If this is a read, pick based on mode:\n"
            "  paper_state → realized_pnl_sim\n"
            "  live_account → realized_pnl_usd\n"
            "Never read the legacy alias in new code."
        ),
        severity="HIGH",
    ),
]


# ──────────────────────────────────────────────────────────────
# File iteration
# ──────────────────────────────────────────────────────────────
DEAD_FILES = {
    # frontend/app.js is the v1 monolith — replaced by v2 (frontend/js/main.js).
    # Still served at /  but only loaded by the legacy /index.html, not by /v2.
    # We don't fix bugs in dead code; we delete it. Until then, exclude.
    "frontend/app.js",
    "frontend/index.html",  # legacy v1 entry, deleted from active route set
}

# Files where `realized_pnl` is unambiguously paper/sim only (PaperPosition,
# PaperFill, etc.). The legacy alias is intentional in these files; the
# split is at the PaperAccountSummary level. Don't flag them.
PAPER_ONLY_FILES = {
    "server/execution/engine.py",
    "server/execution/position_manager.py",
    "server/execution/types.py",
    "server/strategy/backtest.py",
}


def iter_matching_files(globs: list[str], rule_id: str = "") -> Iterator[Path]:
    seen: set[Path] = set()
    for g in globs:
        for p in ROOT.glob(g):
            if p.is_file() and p not in seen:
                # Skip pycache, node_modules, .venv, etc.
                parts = set(p.parts)
                if parts & {"__pycache__", "node_modules", ".venv", "venv", ".git"}:
                    continue
                rel = str(p.relative_to(ROOT)).replace("\\", "/")
                if rel in DEAD_FILES:
                    continue
                if rule_id == "realized_pnl_undifferentiated" and rel in PAPER_ONLY_FILES:
                    continue
                seen.add(p)
                yield p


# ──────────────────────────────────────────────────────────────
# Hit container
# ──────────────────────────────────────────────────────────────
@dataclass
class Hit:
    rule_id: str
    severity: str
    file: str
    line: int
    matched_text: str
    full_line: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Scanners
# ──────────────────────────────────────────────────────────────
_JS_COMMENT_RE = re.compile(r"^\s*(//|\*)")
_PY_COMMENT_RE = re.compile(r"^\s*#")
_SAFE_MARKER_RE = re.compile(r"//\s*SAFE\b|#\s*SAFE\b")


def _is_comment_line(line: str, file: str) -> bool:
    if file.endswith((".js", ".html")):
        return bool(_JS_COMMENT_RE.match(line))
    if file.endswith(".py"):
        return bool(_PY_COMMENT_RE.match(line))
    return False


def _has_safe_marker_nearby(lines: list[str], i: int, window: int = 2) -> bool:
    """Return True if a `// SAFE` or `# SAFE` marker is on this line or
    within `window` lines before it. Lets developers acknowledge a
    sweep finding as intentional without disabling the rule globally.
    """
    start = max(0, i - window)
    for j in range(start, i + 1):
        if _SAFE_MARKER_RE.search(lines[j]):
            return True
    return False


def scan_line_based(rule: ScanRule) -> list[Hit]:
    """Simple per-line regex scan with line-level excludes."""
    hits: list[Hit] = []
    compiled = re.compile(rule.pattern)
    excludes = [re.compile(p) for p in rule.line_excludes]

    for path in iter_matching_files(rule.file_globs, rule.id):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
        for i, line in enumerate(lines):
            m = compiled.search(line)
            if not m:
                continue
            # Skip comment lines (Python # / JS //) — comments aren't bugs
            if _is_comment_line(line, rel_path):
                continue
            # Honor `// SAFE` / `# SAFE` markers within 2 lines before
            if _has_safe_marker_nearby(lines, i, window=2):
                continue
            # Line-level excludes
            if any(ex.search(line) for ex in excludes):
                continue
            hits.append(
                Hit(
                    rule_id=rule.id,
                    severity=rule.severity,
                    file=str(path.relative_to(ROOT)).replace("\\", "/"),
                    line=i + 1,
                    matched_text=m.group(0),
                    full_line=line.rstrip(),
                    context_before=[
                        lines[j].rstrip()
                        for j in range(max(0, i - rule.context_before), i)
                    ],
                    context_after=[
                        lines[j].rstrip()
                        for j in range(i + 1, min(len(lines), i + 1 + rule.context_after))
                    ],
                )
            )
    return hits


def scan_call_with_missing_kwarg(
    rule: ScanRule,
    required_kwarg: str,
) -> list[Hit]:
    """Find function calls that DON'T mention a required kwarg within N
    lines after the open-paren. Used for match_pattern / find_similar.
    """
    hits: list[Hit] = []
    call_open = re.compile(rule.pattern)
    kwarg_check = re.compile(r"\b" + re.escape(required_kwarg) + r"\s*=")
    excludes = [re.compile(p) for p in rule.line_excludes]

    for path in iter_matching_files(rule.file_globs, rule.id):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            m = call_open.search(line)
            if not m:
                continue
            if any(ex.search(line) for ex in excludes):
                continue
            # Collect the call body: from the open-paren line through either
            # matching close or up to context_after lines, whichever first.
            depth = 0
            body_lines: list[str] = []
            start_pos = m.start()
            # For each line from i onwards, track paren depth
            found_close = False
            for j in range(i, min(len(lines), i + rule.context_after + 1)):
                seg = lines[j]
                if j == i:
                    seg = seg[start_pos:]  # start counting from the call site
                depth += seg.count("(") - seg.count(")")
                body_lines.append(lines[j])
                if depth <= 0 and j > i:
                    found_close = True
                    break
                if j == i and depth <= 0:
                    # Whole call on one line — rare but possible
                    found_close = True
                    break
            body = "\n".join(body_lines)
            if kwarg_check.search(body):
                continue  # kwarg present, good
            hits.append(
                Hit(
                    rule_id=rule.id,
                    severity=rule.severity,
                    file=str(path.relative_to(ROOT)).replace("\\", "/"),
                    line=i + 1,
                    matched_text=m.group(0),
                    full_line=line.rstrip(),
                    context_after=body_lines[1:6],  # first few lines of the call body
                )
            )
    return hits


def run_rule(rule: ScanRule) -> list[Hit]:
    if rule.id == "match_pattern_time_leak":
        return scan_call_with_missing_kwarg(rule, "current_time_position")
    if rule.id == "find_similar_self_match":
        return scan_call_with_missing_kwarg(rule, "exclude_anchors")
    return scan_line_based(rule)


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────
def print_report(results: dict[str, list[Hit]]) -> int:
    # Force UTF-8 on Windows consoles so box-drawing chars don't crash
    import sys as _sys
    if _sys.platform == "win32":
        try:
            _sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    total = 0
    print()
    print("=" * 72)
    print("GREP-SCAN SWEEP - orphan fixes from prior 'Phase N done' commits")
    print("=" * 72)

    # Summary first
    print()
    print("SUMMARY")
    print("-" * 72)
    for rule in RULES:
        hits = results.get(rule.id, [])
        total += len(hits)
        mark = "OK " if not hits else rule.severity.ljust(4)
        print(f"  [{mark}] {rule.id:30s} {len(hits):4d} hit(s)")
    print(f"{'':33s} -----")
    print(f"{'':33s} {total:4d} total")
    print()

    # Details
    for rule in RULES:
        hits = results.get(rule.id, [])
        if not hits:
            continue
        print("=" * 72)
        print(f"{rule.severity} SCAN: {rule.id} - {rule.title}")
        print("=" * 72)
        print(f"WHY: {rule.why}")
        print()
        print(f"FIX: {rule.fix}")
        print()
        print(f"HITS ({len(hits)}):")
        print("-" * 72)
        for h in hits:
            print(f"  {h.file}:{h.line}")
            for cb in h.context_before:
                print(f"    | {cb}")
            print(f"    > {h.full_line}")
            for ca in h.context_after[:3]:
                print(f"    | {ca}")
            print()

    print("=" * 72)
    print(f"TOTAL: {total} finding(s) across {len([r for r in RULES if results.get(r.id)])} scan(s)")
    print("=" * 72)
    return total


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--scan", action="append", help="Run only a specific scan id (repeatable)")
    ap.add_argument("--json", help="Write findings to this JSON path")
    ap.add_argument(
        "--ci",
        action="store_true",
        help="Exit with nonzero code equal to finding count (useful in CI)",
    )
    args = ap.parse_args()

    # Filter rules
    rules = RULES
    if args.scan:
        wanted = set(args.scan)
        rules = [r for r in RULES if r.id in wanted]
        if not rules:
            print(f"No rules match {args.scan}. Available: {[r.id for r in RULES]}", file=sys.stderr)
            sys.exit(2)

    # Run
    results: dict[str, list[Hit]] = {}
    for rule in rules:
        results[rule.id] = run_rule(rule)

    # Report
    total = print_report(results)

    # JSON
    if args.json:
        out = {
            "summary": {
                "total_findings": total,
                "per_scan": {rid: len(hs) for rid, hs in results.items()},
            },
            "findings": {
                rid: [
                    {
                        "severity": h.severity,
                        "file": h.file,
                        "line": h.line,
                        "matched_text": h.matched_text,
                        "full_line": h.full_line,
                        "context_after": h.context_after,
                    }
                    for h in hs
                ]
                for rid, hs in results.items()
            },
        }
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {args.json}")

    if args.ci and total > 0:
        sys.exit(min(total, 125))  # clamp to max unix exit code
    sys.exit(0)


if __name__ == "__main__":
    main()
