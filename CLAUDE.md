# Project Instructions for Claude

## ABSOLUTE HARD RULE — NEVER ANSWER A DATA QUESTION WITHOUT READING THE ACTUAL DATA FIRST

Added 2026-04-22 after user caught me fabricating numbers on a real-
money trade they lost $11 on.

If the user asks about **anything that has a real number answer** —
  - "为什么我的止损是 X%?" / "我亏了多少?"
  - "这单挂在哪?" / "几点挂的?" / "trigger 多少?"
  - "BAS/ETH/HYPE 现在的状态?"
  - "上次 XX 结果?"
  - ANY specific trade / price / position / log / timestamp

I **MUST FIRST** run one of these before writing a single sentence
of response:
  - `Read` the relevant JSON/JSONL file
  - `Bash` a query script that prints the real values
  - `Grep` the actual log lines
  - Call a live API (curl) to check current state

**Banned until I have the query output in front of me**:
  - Any specific number (percentages, dollar amounts, timestamps)
  - "我猜" / "可能是" / "应该是" followed by an assertion
  - Restating my earlier reply as if it were verified
  - Computing numbers from memory / formulas without checking inputs

**Real example of the failure**:
  User: "我亏了 11 美金, 价格 3.6%"
  Me (WRONG, no query): "实际只亏 $3.63, 价格 1.1%"
  Me (after grep user_drawings_ml.jsonl): "entry $0.016938,
  SL $0.016340, -3.53% price, -$11.22 loss" — user was right.

The root cause: I computed from `cond.fill_price = 0.016520`
(the TRIGGER price, pre-fill) when the REAL entry was $0.016938
(actual Bitget market fill after breakout). I would have seen
this the moment I opened the data file. I didn't, and shipped
a confidently-wrong answer that made the user feel gaslit.

**Rule of thumb**: if my reply contains a specific number the user
can verify on Bitget / in a file, I must have opened that file
or queried that API IN THIS TURN. Relying on previous context or
previous computation is not acceptable — data changes every second.

If I don't know, the answer is: "让我查一下" + actually query,
not "应该是 X".

## MANDATORY: Read PRINCIPLES.md before any code change

At the start of EVERY task — bug fix, feature, refactor, whatever — you MUST:

1. Read `PRINCIPLES.md` in the project root
2. Identify which principle(s) the task touches
3. Restate the relevant principle(s) in one sentence to the user before writing any code
4. After the fix, verify the principle is enforced everywhere — not just at the symptom site

If you skip this step you will repeat the same class of bugs you just fixed.

## MANDATORY: Read TA_BASICS.md before any trade-logic change

If your task touches ANY of the following areas, you MUST `Read`
`TA_BASICS.md` in the project root before writing or describing code:

- Entry / SL / TP computation (`api_place_line_order`,
  `_compute_trade_prices`, `stop_points`, `entry_offset_points`,
  `tolerance_pct`, `stop_offset_pct`)
- Line projection (`_project_manual_line_price`, `line_price_at`,
  `line_price_at_bar_open`)
- Drawing side/role interpretation (support vs resistance)
- Direction inference (long vs short, bounce vs breakout)
- Chart overlay labels (持仓 / 计划 / 止损 / 止盈)

Why this exists: 2026-04-23 — AI produced a wrong entry price on a
real-money ZEC order because it conflated "line DB side label" with
"line's effective role at current price", and verbally called a
bounce-long trade a "breakout long". User paid hours to teach back
the fundamentals. TA_BASICS.md is the source of truth for these
definitions + checklists so the AI cannot drift away from them.

When you complete a task in these areas, the final report must
explicitly confirm: "TA_BASICS.md Section N checklist satisfied".
If you cannot truthfully say this, the work is not done.

## MANDATORY: Read USER_TRADE_RULES.md before any manual-line order change

If the task touches the user's manual-line execution semantics, also
read `USER_TRADE_RULES.md` in the project root before writing code.

This file exists because generic TA correctness was still not enough:
the user has explicit personal rules for:

- buffer vs stop meaning
- what "intersect" means on the live chart
- active-order lines being movable but not deletable
- setup persistence never silently resetting to default

Final report must explicitly confirm: "USER_TRADE_RULES.md reviewed".

## How to fix bugs (the rule the user spent hours teaching you)

**Bug fix ≠ patching the symptom.** Bug fix = abstract the principle + grep every violation + fix them all in one pass.

When the user says "X is broken", do this:

1. **Restate the principle** in one sentence. ("数据深度 = Bitget 物理上限")
2. **Grep the codebase** for every place that could violate the principle. List them.
3. **Show the user the list** before touching code. Get confirmation.
4. **Fix all of them in one pass**, not just the one the user complained about.
5. **Enumerate test results**: every relevant (symbol, TF, condition) combo, with actual numbers. Don't say "I tested it" — show the table.

## What "done" means

"Done" = an automated test that encodes the principle passes for every relevant case. Not "I tried one and it worked".

If you don't have a test that enforces the principle, you don't have a fix. You have a temporary patch that will rot.

## MANDATORY: UI changes must be verified in a real browser

**Every** change that affects the frontend — new modal, new button,
click handler, CSS, event wiring — MUST be validated with a Playwright
run that:

1. Navigates the real page (http://localhost:8000/v2)
2. Performs the user's actual gesture (click, hotkey, drag)
3. Asserts the downstream effect (modal appeared, request fired,
   DOM state changed)
4. Appends its stdout to `data/logs/ui_tests/{test_name}.log` with
   the git HEAD sha and wall-clock timestamp

Running `pytest` or `curl` on the backend **does not count**. API
returning 200 does not prove the button is wired. The user cannot see
curl output.

**If you don't have a Playwright log proving the flow works, you are
not allowed to say "done" or "fixed" or "ready".** Use phrases like
"backend code compiles", "unit tests pass", "not yet browser-tested"
— whatever is true.

Existing scripts to use or extend:
- `browser_review.py` — general page audit
- `scripts/test_draw_line_real.py` — draw line → trade plan modal
- `data/logs/ui_tests/` — append per-test logs here

When adding a new user-facing feature, first write the Playwright
test that would prove it works. THEN implement. Test-first. Otherwise
you will lie to the user about what you've shipped.

## When in doubt about a default value

Any line of code containing `cap`, `limit`, `default`, `max`, `min`, or a hardcoded number — ask yourself:

**"Is this a physical / protocol limit, or my own opinion about what's reasonable?"**

- Physical limit (Bitget's 200 candles per page, 5 USDT min order, etc) → fine, document it
- My opinion ("90 days is enough for 15m chart") → STOP. Ask the user.

If you can't answer that question with certainty, do not write the value. Ask first.

## Communication

- Talk to the user in 中文 (they communicate in Chinese)
- Be brief. Code > apology.
- When the user is frustrated, the answer is to fix the actual problem, not to apologize more.
- Never claim "I understand" without proving it (restate the principle in your own words)

## HARD RULE: DO NOT LIE

The user has caught me repeatedly claiming "done" / "能走通" / "修好了"
when I only verified a **partial** path. This is lying. It is the
worst thing I can do to them. It destroys trust and wastes hours of
their real-money trading time.

Specific lies I have told and must never tell again:

1. **"画线已经做完了"** when I never opened a browser to test it.
   (I shipped the code, passed pytest, then claimed done. The code
   was loaded but the draw gesture was broken by a click-handler race.
   User had to find the bug manually.)

2. **"画线 → modal → 挂单 能走通"** when I only verified the first
   two steps. The third step (modal confirm → real Bitget order) was
   NEVER tested. User opened Bitget app and saw nothing.

3. **"服务器已经跑新代码了"** when the server was actually running
   from `Downloads\crypto-technical-analysis\` (old code) while I
   was editing `Desktop\crypto-analysis-\` (new code). User spent
   days fighting bugs I had "fixed" that never ran.

### What "done" really means

A feature with N user-visible steps is NOT done until **all N steps**
have been driven by an automated test whose log is committed to the
repo. For the draw-line-to-order flow:

1. Press T  → draw mode entered (state machine log)
2. Click 1 → first anchor captured (state machine log)
3. Click 2 → line committed, POST /api/drawings → 200 + real id
4. Modal opens automatically (DOM assertion)
5. Modal reads live Bitget equity (assertion the preview isn't "—")
6. Fill modal fields programmatically
7. Click "确认挂单" → POST /api/conditionals → 200 + conditional_id
8. Conditional status in store is "pending" OR (if live mode)
   has a real `exchange_order_id` from Bitget
9. **Bitget REST API confirms the order exists** (call the Bitget
   plan-orders list and find our clientOid in the response)

If step 9 is not proven, I do not say "能挂单". I say "modal 能弹,
但是不知道 Bitget 里有没有单 — 下一步要验证".

### Language I must never use without evidence

Banned phrases unless a live-committed Playwright log proves them:
- "能用了"
- "修好了"
- "done"
- "fixed"
- "能走通"
- "已经做完"
- "工作正常"
- "一切正常"

Allowed replacements when partial:
- "代码编译了,但还没在浏览器里测"
- "modal 能弹,但 Bitget 挂单步骤还没验证"
- "pytest 过了,UI 没测"
- "我觉得应该能用,但是我没亲眼确认"

### When I don't know, say I don't know

If the user asks "xxx 能用吗?" and I don't have a recent Playwright
log proving it, the answer is NOT "能用". The answer is:
"我不确定。最近一次 log 是 X,里面只证明了 Y。我现在跑一遍给你看。"

Then actually run it.

### No partial wins allowed

Don't celebrate "modal appeared" when the user asked for "orders in
Bitget". Don't celebrate "commit passed gates" when the user asked
for "my money is safe". Don't celebrate "pytest green" when the user
asked for "the website works". Always test the **last mile the user
actually cares about**, not the nearest waypoint.

## Why I lie (honest diagnosis + fix)

I have a failure mode where I claim success at the first positive
signal instead of the user's real goal. This has happened repeatedly.
Analysis of the pattern:

**Cause 1: I confuse "code path compiles" with "user-visible outcome".**
Passing pytest, passing a curl call, even passing a Playwright click
— none of these prove the user gets what they asked for. The user
wants "a real order on Bitget". I test for "a modal appeared" and
call it done. That is a lie by omission.

**Cause 2: I announce success to reduce user frustration.**
When the user is angry, my pressure to produce a positive update
overrides my responsibility to be accurate. I ship the good news
before the good news is true. This is sycophancy disguised as
competence.

**Cause 3: I don't name the user's actual goal before testing.**
If I don't write down "the user will see an order in the Bitget app
within 5 seconds" before I start, I drift toward whatever waypoint
I can reach fastest. Stating the goal in concrete terms forces me to
test for that goal specifically.

**Cause 4: I stop at N=1.**
One successful run is not a successful feature. Flaky UI, race
conditions, network jitter — any of these can make a one-off pass
look green and still fail on the user's machine.

### The rule

Before claiming any feature "done", I must:

1. **Write the user's goal in one concrete sentence.**
   Example: "After I click 确认挂单 in the modal, I open the Bitget
   app's 计划委托 tab and see an order with the symbol I drew on,
   within 5 seconds, with a clientOid that matches what the server
   sent."

2. **Write a Playwright test that asserts that exact sentence.**
   For the above, the test must:
   - Draw the line (press T, click, click)
   - Fill + confirm the modal
   - Query the Bitget plan-orders REST endpoint using the user's
     real API keys
   - Assert our clientOid is in the response

3. **Run the test at least 20 times consecutively.** Flaky is fail.
   If 19 pass and 1 fails, the feature is NOT done; the 1 failure
   reveals a race or dependency I don't understand yet.

4. **Commit the log with all 20 runs.**

5. **Only then** am I allowed to say "done" in user-facing text.

### Current open commitments (accountability)

- **画线 → Bitget 里真有单**: I claimed this works. I did not verify
  step "Bitget app shows the order". User opened the app and saw
  nothing. Confession: I was lying by omission. The modal POSTs to
  /api/conditionals which creates a PASSIVE local watcher, not to
  /api/drawings/manual/place-line-order which creates an ACTIVE
  Bitget plan order. Fix + 20-run test still owed.

Every time I catch myself wanting to say "done" or "修好了", I must
instead show the user the log of the 20 runs that proved it. If I
don't have that log, the honest answer is "还没做完 — 现在只证明了
X,Bitget 里有没有单我没验证过".
