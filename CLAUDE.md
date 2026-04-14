# Project Instructions for Claude

## MANDATORY: Read PRINCIPLES.md before any code change

At the start of EVERY task — bug fix, feature, refactor, whatever — you MUST:

1. Read `PRINCIPLES.md` in the project root
2. Identify which principle(s) the task touches
3. Restate the relevant principle(s) in one sentence to the user before writing any code
4. After the fix, verify the principle is enforced everywhere — not just at the symptom site

If you skip this step you will repeat the same class of bugs you just fixed.

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
