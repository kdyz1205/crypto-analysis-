# Session Handoff — 2026-04-23

> Paste this into the new Claude Code session that you open from inside
> `C:\Users\alexl\Desktop\crypto-analysis-\`. The new Claude will read
> `CLAUDE.md` + `PRINCIPLES.md` automatically; this file tells it what
> happened in the previous session, what's still open, and the working
> contract the user expects.

---

## 0. READ FIRST (new Claude)

Before doing ANYTHING:

1. `Read` `C:\Users\alexl\Desktop\crypto-analysis-\CLAUDE.md`
2. `Read` `C:\Users\alexl\Desktop\crypto-analysis-\PRINCIPLES.md`
3. `Read` `C:\Users\alexl\.claude\projects\C--Users-alexl\memory\feedback_exchange_cancel_forensics.md` — sections Q1 through Q5 document the exact bug family that kept recurring.
4. Run `pytest tests/conditionals/test_watcher_replan.py -v` and confirm 17 tests pass.

Only after all 4: read the rest of this file.

---

## 1. The recurring bug the user is furious about

Same class of bug has bitten real-money trades multiple times:

- **2026-04-22 (BAS)**: both Bitget pending endpoints timed out → reconcile marked triggered conds as cancelled → user lost replan coverage.
- **2026-04-23 00:48 LA (ETH + HYPE + ORDI)**: Bitget returned 429 on plan-pending only; reconcile's gate was `if not (ok_regular OR ok_plan): skip`, which let partial-failure fall through; all three plan-type conds got cascade-cancelled in the same tick while Bitget still held them live.

Root cause was the same in both: **using absence-from-a-list as evidence to transition local state**. Every Bitget API glitch looks identical to a real cancel.

### Why it kept recurring

Previous sessions patched ONE failure mode at a time:
- v1: "skip if BOTH pending fetches fail" — missed the 429-on-one-but-not-the-other case.
- v2: "skip if EITHER fails" — catches more but still can't handle silent incompleteness, paginated lists, eventual-consistency blips.

The fundamental pattern wasn't addressed until this session. User is rightly frustrated: "你的做事方法、做事逻辑,感觉全部都已经并没有在真正地解决问题,而是在反复地出现问题."

### The fix applied in this session

Principle, encoded in code and tests:

> **Local state transitions happen ONLY on affirmative Bitget evidence.
> Never on "oid was absent from a list". If we can't tell, stay triggered
> and retry next cycle.**

New module `server/conditionals/oid_state.py` with a 4-state classifier:

| Classifier returns | Evidence | Action |
|---|---|---|
| LIVE | pending list contains oid, OR history row `state=live` | keep triggered |
| FILLED | matching position, OR history row `state=triggered/executed/filled` | transition to filled |
| CANCELLED | history row `state=cancelled` (only way) | transition to cancelled + reverse spawn |
| UNKNOWN | pending absent AND history couldn't confirm | keep triggered, retry |

This is enforced in 3 places that previously all had the same bug:
1. `_reconcile_against_bitget` — main reconcile loop
2. `_cancel_bitget_plan_safely` — the cancel fallback used by replan + line-broken
3. `exchange_cancel.cancel_exchange_order_for_cond` — user-triggered cancel path

All 3 now dispatch through `classify_cond_state`. See `memory/feedback_exchange_cancel_forensics.md` Q4 + Q5.

### The mirror-direction bug was also fixed

Once the classifier made us MORE conservative about cancelling, a new risk appeared: "cancel API broken → we can never transition → stuck forever." So `_cancel_bitget_plan_safely` now returns a 3-way enum `CancelOutcome.{CANCELLED, FILLED, RETRY}` and callers handle FILLED separately (transition to filled, NO reverse — a reverse against a real open position is catastrophic).

### 17 tests locking the invariant

`tests/conditionals/test_watcher_replan.py` — run with `pytest`. Each test encodes ONE row of this truth table:

| Scenario | Correct action |
|---|---|
| pending 429 + history says live | stay triggered |
| pending 429 + no history evidence | stay triggered (UNKNOWN) |
| pending absent + history says cancelled | transition cancelled + reverse |
| pending absent + history says filled | transition filled (no reverse) |
| pending absent + history API error | stay triggered |
| line-broken + cancel API fails + history says cancelled | transition cancelled + reverse |
| line-broken + cancel API fails + history says filled | transition filled (NO reverse — position is real) |
| line-broken + cancel API fails + history UNKNOWN | stay triggered |
| All good (cancel works + classifier agrees) | normal path |

If these tests break, the principle is violated — don't merge the change.

---

## 2. Commits from this session (in order)

```
db50906  fix(popup): fall back to last-good cached equity when fresh fetch aborts
f1a0fc1  feat(indicators): MA Ribbon is now UI-configurable (periods/type/color/width)
e228283  fix(reconcile): never cancel on partial pending-fetch failure (P0, real money)
1acad13  feat(chart): entry/SL/TP price-line markers for every triggered plan + live position
623d5b2  fix(reconcile): structural rewrite — only transition state on affirmative Bitget evidence
40e308b  feat(chart): measure tool + long/short prediction markers
3abfbd8  fix(watcher): mirror-direction guard — cancel-failure paths now self-heal via history
adc75b6  feat(launcher): desktop shortcut to auto-start server + open /v2
```

All committed to `main`, server currently running with all fixes live.

---

## 3. Open issues the user needs help deciding

### 3a. Three orphan conds from the 00:48:12 LA false-cancel tick

These were created before the fix, so local state is stale. Listed below with current Bitget reality (queried 2026-04-23):

| cond_id | symbol | oid | local status | Bitget reality |
|---|---|---|---|---|
| cond_2d9ccc2f7f2c2c | ETHUSDT | 1431058327481081858 | cancelled | **LIVE, 0.57 @ 2383.47 short** |
| cond_7d071cd6e96b02 | HYPEUSDT | 1431109453823860736 | cancelled | cancelled (user manually cancelled ~01:10 LA) |
| cond_26cbfbb4787a58 | ORDIUSDT | 1431058311415324672 | cancelled | executed + position since closed |

**ETH needs a decision from the user**:
- Option A: flip local status back to `triggered` so watcher resumes replanning
- Option B: user cancels manually on Bitget app, both sides will agree
- Do NOT auto-transition without user approval — that would violate the principle we just fixed.

**ORDI**: the position fired AND already closed. We have no fill/exit records in our store. The money-side is fine (no position right now), but the ML training data for that trade is missing. Low priority — worth a note in memory as "lost trade record #1 from 2026-04-23 false-cancel".

**HYPE**: consistent both sides. No action.

### 3b. No monitoring for stuck-in-UNKNOWN

Telegram alert fires after 10 reconcile cycles (~5 min) of UNKNOWN on the same cond. But no UI badge, no auto-resolution, no recurring reminders after the first alert. If the user misses the Telegram, a cond can sit in UNKNOWN indefinitely.

Follow-up candidates (DO NOT implement without user approval):
- UI red badge on sidebar showing "X cond stuck in UNKNOWN for Y min"
- Recurring alert every 30 min until resolved
- Admin endpoint `/api/conditionals/{id}/force-unknown-resolve?as=cancelled|filled|triggered`

---

## 4. Work methodology the user expects (they've said this repeatedly)

Before any code change:

1. **Write down the principle being touched** — one sentence.
2. **Grep the whole repo** for every place that could violate the principle. List them in a table.
3. **Show the user the list. Wait for "OK" before touching code.**
4. **Fix all of them in one commit.**
5. **Write "principle-level" tests, not "this scenario" tests.**
6. **Commit message matches the list — no unrelated changes sneaked in.**

The user explicitly said they worry about random/unrelated changes. Keep commits tight.

When unsure:
- Read PRINCIPLES.md first
- Read the specific feedback_*.md that matches the problem area
- Ask the user rather than guess

---

## 5. UI features shipped this session

- **📏 测量工具** (toolbar + M hotkey): drag a box, see ΔPrice/Δtime/bars
- **📈 多头 / 📉 空头 仓位预测** (toolbar): 2-click to define entry + TP, SL auto at 1% opposite. R:R shown. Persisted to localStorage per symbol+interval.
- **挂单/仓位 坐标线**: every triggered plan + live position auto-gets entry/SL/TP horizontal lines on chart with labels like `计划 空 @ 41.446 (275.22)`. Polls every 10s + reacts to `conditionals.changed` event.
- **MA Ribbon 可配置 UI**: ⚙ gear per indicator opens inline editor with per-line period/SMA-EMA/color/width + preset dropdown.
- **Desktop shortcut**: `Trading OS.lnk` on user's desktop; double-click starts server (if needed) + opens http://127.0.0.1:8000/v2.

---

## 6. First message to paste into the new session

```
读完 SESSION_HANDOFF_2026-04-23.md, 读完 CLAUDE.md 和 PRINCIPLES.md,
然后 read feedback_exchange_cancel_forensics.md Q1-Q5.

跑一下 pytest tests/conditionals/test_watcher_replan.py -v, 确认 17 个测试全绿,
跟我确认这件事。

然后先不要改代码, 告诉我:
1. 你对这次 bug 的根因和现在的修复怎么理解的
2. 你打算怎么工作 — 特别是在改代码前会做哪几步
3. 关于 SESSION_HANDOFF 里那 3 个孤儿 cond (ETH 还 live 在 Bitget,
   HYPE 一致 cancelled, ORDI 已 fill 完也平了), 你建议我怎么处理

回复后等我确认, 再开始做任何事。
```

This forces the new Claude to prove it actually read the docs before touching code.
