# Phase 1-3 Release Commit Plan

**Drafted:** 2026-04-19
**Repo:** `C:\Users\alexl\Desktop\crypto-analysis-`
**Remote:** `https://github.com/kdyz1205/crypto-analysis-.git` (public)
**Agent:** release-preparer (team phase3-pack-a)

This is a PLAN only. Nothing has been committed, staged, or pushed. The lead
should review the groups below, then either execute as-is or adjust.

---

## 0. Secret/safety scan — PASSED, with caveats

- `.env` is correctly gitignored (`.gitignore:12`).
- No API keys, bearer tokens, or Bitget secrets found in any modified tracked
  file or untracked candidate file.
- **Caveat (pre-existing, not introduced here):** the repo historically tracks
  `data/mar_bb_daily_risk.json`, `data/mar_bb_state.json`, and
  `data/logs/browser_review/audit.json`, which record live USD equity and
  position snapshots (e.g. `$4.60 · 可用 $4.60`). These are public-by-convention
  in this repo; flagging for the lead in case they want to gitignore going
  forward, but NOT blocking this release.
- `data/mar_bb_state.json` also contains a hard-coded absolute path
  (`C:/Users/alexl/trading-system/...`). Already-present in repo history.

## 1. Line-ending (CRLF) warnings — safe to commit

Every edited `.py` / `.js` file triggers `LF will be replaced by CRLF`. This is
the standard Windows checkout pattern; the blob stored in git keeps LF. No
action needed — just commit through the warnings. The full list matches the
diff; no surprise files.

---

## Proposed commits (7 groups, in recommended order)

### Commit 1 — backend: emergency kill switches (flatten-all + reset-halt)

**Title:** `Add emergency flatten-all + daily-halt reset endpoints`

**Body:** New `/api/live-execution/flatten-all` closes every position and
cancels every plan order for a mode (guarded by `confirm_code=FLATTEN`), and
`/api/mar-bb-runner/reset-halt` manually clears the daily-DD halt
(guarded by `confirm_code=RESET`).

**Files:**
```
git add server/routers/live_execution.py \
        server/schemas/live_execution.py \
        server/routers/mar_bb_runner.py
```

Note: `server/routers/mar_bb_runner.py` also adds `stop_offset_pct` / 
`tf_stop_offset` fields on Start/Update request models — they logically belong
with commit 2 (stop-offset feature) rather than here. If the lead wants strict
isolation, split this file by hunk; I recommend keeping them together since
both depend on changes in `server/strategy/mar_bb_runner.py` landing.

---

### Commit 2 — backend: stop-offset beyond line + cooldown + auto-line persistence

**Title:** `Place stop just beyond line, persist auto-triggered lines`

**Body:** Replace "stop == line" with a small configurable offset
(`stop_offset_pct`, default 0.01 / 0.01%). After a plan fills, the trendline
manager writes the two anchor bars to `manual_trendlines.json` with
`source=auto_triggered` so the user can see what the system drew, and a
per-(symbol, TF, kind) cooldown prevents immediate re-entry after a close.

**Files:**
```
git add server/strategy/mar_bb_runner.py \
        server/strategy/trendline_order_manager.py \
        server/conditionals/watcher.py \
        server/routers/conditionals.py
```

Scope notes:
- `server/routers/conditionals.py` (+329/-? lines): adds mark-price cache, a
  leverage-set cache, restores `stop_offset_pct` as a live parameter on the
  place-line-order path, and tightens timeouts. All of this is coupled to the
  stop-offset change, so grouping them is honest.
- `server/conditionals/watcher.py`: aligns replan cadence to TF bar boundaries
  (prevents intra-bar churn) and uses the new `stop_offset_pct` when
  registering manual trailing.

---

### Commit 3 — backend: drawings schema loosens line_width min to 0.5

**Title:** `Allow line_width down to 0.5px on manual trendlines`

**Body:** Drop the `ge=1.0` floor on manual line width (now `ge=0.5`). Users
can set thinner strokes from the context menu; the server just clamps to
[0.5, 8.0].

**Files:**
```
git add server/routers/drawings.py server/schemas/drawings.py
```

Trivial, small, stands alone.

---

### Commit 4 — frontend: runner halt UX + LIVE confirmation + Pack A polish

**Title:** `Runner UI: halt badge, reset button, LIVE confirm modal`

**Body:** Show `HALTED · -X% today` pill with 1s-refresh countdown, reveal a
reset-halt button while halted (routes to the new `/reset-halt` endpoint),
and require typing `LIVE` before real-money cold-starts. Loosens the live
account polling timeout to 8s with service-side override.

**Files:**
```
git add frontend/js/views/runner_view.js \
        frontend/js/services/live_execution.js \
        frontend/js/services/conditionals.js
```

---

### Commit 5 — frontend: chart + trade-plan modal + auto-line rendering

**Title:** `Render auto-triggered lines and sharpen chart live updates`

**Body:** Auto-triggered lines render dashed with a warm red/amber palette
(both in `chart_drawing.js` canvas path and the `manual_trendline_overlay`
lightweight-charts path) so they stand apart from user-drawn lines. The
chart only reloads full OHLCV on real bar rollover (not every 10s); the
trade-plan modal caches equity for 60s and uses a separate `stop_pct` field
(previously collapsed onto `buffer_pct`). Context-menu adds 0.5px line-width
option.

**Files:**
```
git add frontend/js/workbench/chart.js \
        frontend/js/workbench/drawings/chart_drawing.js \
        frontend/js/workbench/drawings/trade_plan_modal.js \
        frontend/js/workbench/overlays/manual_trendline_overlay.js
```

---

### Commit 6 — tests + training: weight user-drawn lines + test updates

**Title:** `Update tests for stop-offset; weight user data in training`

**Body:** Existing stop-offset tests flip from "stop at line" to "stop beyond
line" numbers; adds new test for `_trade_prices_for_line` + an
"inside-buffer-until-stop-break" acceptance test. Adds a new
`test_watcher_replan.py`. Training script now applies 2× / 4× / 8× / 10×
sample weights for user-drawn lines, outcome-labeled lines, lines with order
intent, and lines with both intent + outcome.

**Files:**
```
git add tests/conditionals/test_line_projection.py \
        tests/conditionals/test_manual_line_order_api.py \
        tests/strategy/test_trendline_order_manager.py \
        tests/conditionals/test_watcher_replan.py \
        scripts/train_trendline_quality.py
```

---

### Commit 7 — scripts + fixtures: codex_tmp fixtures and new audit scripts

**Title:** `Refresh trendline fixtures and add UI audit scripts`

**Body:** `data/codex_tmp/active_*.json` fixtures regenerate with the new
`stop_price` math (line × (1 ± 0.01%)) and fresh timestamps. Two new
Playwright scripts — `scripts/audit_runner_ui.py` dumps the runner/live
panels with screenshots, `scripts/test_auto_line_viz.py` is a smoke test for
the auto-line rendering — land under `scripts/` and their output shard
under `data/logs/ui_tests/`.

**Files:**
```
git add data/codex_tmp/active_broken_before_move.json \
        data/codex_tmp/active_halt_cancel.json \
        data/codex_tmp/active_held.json \
        data/codex_tmp/active_move.json \
        data/codex_tmp/active_new.json \
        data/codex_tmp/active_orphan.json \
        data/codex_tmp/active_stale.json \
        data/codex_tmp/active_stale_fresh.json \
        scripts/audit_runner_ui.py \
        scripts/test_auto_line_viz.py
```

Optional (if the lead wants to preserve the evidence):
```
git add data/logs/ui_tests/auto_line_viz_1776578711.png \
        data/logs/ui_tests/auto_line_viz_1776578796.png \
        data/logs/ui_tests/runner_audit/
```
I lean toward committing these since the repo already tracks
`data/logs/browser_review/*.png`, so the precedent is set.

---

## FILES TO EXPLICITLY EXCLUDE (do NOT commit)

| File | Reason |
|------|--------|
| `server_pid.txt` | Local process-ID scratch, regenerated on every server start. Already not in any commit; should be .gitignored. |

Recommended .gitignore addition (optional, can be a follow-up PR):
```
# local runtime scratch
server_pid.txt
```

---

## FILES I'D SKIP FOR THIS RELEASE (runtime state churn)

These are tracked files that change on every run and add noise; they look
committed only because the working tree has never been reset. Each is
deliberately excluded from any of the 7 commits above:

| File | Why skip |
|------|----------|
| `data/conditional_orders.json` | Live conditional-order state, +467 lines from the past 2 days of trading. Will diverge again the instant the server starts. |
| `data/manual_trendlines.json` | User's drawn lines — bumps on every draw/erase. |
| `data/mar_bb_daily_risk.json` | Today's DD counter (shows an 81% loss from live trading); transient. |
| `data/mar_bb_state.json` | Runner status + last_scan_ts; regenerates per scan. |
| `data/ml_trades.jsonl` | +46 event rows from the runner's actual trades today. |
| `data/trade_log.jsonl` | +46 log rows, same as above. |
| `data/trendline_active_orders.json` | +180 lines from live plan orders. |
| `data/user_drawings_ml.jsonl` | JSONL that reshuffles on every drawing event (the diff includes a BOM strip too). |
| `data/logs/browser_review/*.png` (4 PNGs) | Newer screenshots with live equity visible. The PNGs are tracked; overwriting is fine locally but polluting history with them adds no value. |
| `data/logs/browser_review/audit.json` | Same — live equity + positions snapshot in body_text. |

**Strong recommendation to the lead:**
Add the following to `.gitignore` in a follow-up cleanup PR, then
`git rm --cached` them so they stop showing up every run:

```
data/conditional_orders.json
data/mar_bb_daily_risk.json
data/mar_bb_state.json
data/ml_trades.jsonl
data/trade_log.jsonl
data/trendline_active_orders.json
data/user_drawings_ml.jsonl
data/logs/browser_review/
```

`data/manual_trendlines.json` is more debatable — if the user's drawings are
considered training data, keep it tracked; otherwise ignore.

That cleanup is OUT OF SCOPE for this release. It deserves its own PR so the
Phase 3 diff stays focused on features.

---

## Suggested commit order (why this order)

1. **Kill switches** first — pure additions, no behavioral risk, useful to
   have live before anything else lands.
2. **Stop-offset + auto-line + cooldown** next — the central Phase 2/3 feature.
3. **Drawings width** — trivial, could slot anywhere.
4. **Runner UI** — depends on commit 1 (reset-halt endpoint).
5. **Chart + trade-plan UI** — depends on commit 2 (auto-line source field).
6. **Tests + training** — verifies commits 1/2; training weight change is
   independent but small, fits here.
7. **Fixtures + audit scripts** — evidence that the above is verified. Last
   because the fixtures encode the commit-2 math.

---

## One-shot helper (for the lead to review, NOT for me to run)

```bash
# Dry-run: verify each group first
git status --short
git diff --stat <files-for-commit-1>
# ... then execute in order
```

A full bash script is intentionally NOT provided — the instruction was
"draft a plan, don't commit."

---

**End of plan.**
