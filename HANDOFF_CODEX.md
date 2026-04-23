# Handoff to Codex — 2026-04-22

用户 Axel 在实盘操盘(Bitget U 本位合约),真钱。上一个 agent(Claude)正在修一串跟"手画线挂单"整套流程有关的 bug,每一条都可能让他丢仓或错开方向。现在要你接着干。

**用户偏好**:
- 用中文跟他讲。
- 不要说 "done / 修好了 / 能用了" 除非有 Playwright 或实盘 log 证据。用 "代码改完了、语法过了、还没端到端验证" 之类精确话。
- **严禁凭空瞎编数字**。任何带具体数字的回答都必须先 `Read` / `Grep` / `Bash curl` 真实数据。`CLAUDE.md` 顶部有完整规则,**先读 `CLAUDE.md`**。
- 他的 Bitget 是 live 模式,不是 sandbox。改错就是真钱亏损。
- 不要 commit,不要 push,除非他明确说了要。

---

## 背景:整件事的 shape

用户画趋势线 → 点"快捷挂单" → 选方向(long/short)+ 点 setup → 后端下 Bitget normal_plan trigger-market 单 → watcher 持续 replan(每根新 bar 根据新的线价更新 Bitget 触发价)→ 线破就 cancel。

2026-04-22 凌晨一笔 BASUSDT 1h short(conditional_id=`cond_99f378f01e8bbf`)出事:
- 00:02 LA 以触发价 0.01493 (line=0.01508 × 0.99) 挂下去
- 01:00 LA bar 边界,watcher 的 replan 因为线已经跌到 0.01343(-10.04% drift)把单子 cancel 了、下了新单触发价 0.01343
- Bitget 看"当时标记价 > 新触发价",把方向推断成 ≤(跌到才触发)= 破位短,不是用户本意的反弹短
- 用户手动在 Bitget app 撤了
- 根因:**当时代码里没有"线破就 cancel,不要 replan"这段逻辑**

## 已经做完的事 (Claude session 2026-04-22)

### 1. 线破 cancel 逻辑 — 修了 8 个 bug

全部改在 `server/conditionals/watcher.py` + `server/conditionals/store.py`,尚未 commit。

| Bug # | 修了什么 | 具体位置 |
|-------|---------|----------|
| **1+2** | `_cancel_bitget_plan_safely` 新 helper:只有真的 cancel 成功才返回 True,`(_cr or {}).get("ok")` 防 None | `watcher.py` 大约 line 1460-1500,每条 cancel 路都有独立 try/except |
| **4** | `_fetch_mark_price_strict` 新 helper:markPrice 拿不到就 return None,绝不回退到 lastPr;`_maybe_replan` 在 mark=None 时整个早退,不 replan | `watcher.py` 约 line 1421-1457,调用处 约 1545-1562 |
| **5** | `store.set_status_if(from_status, to_status)` 原子 CAS;线破 short/long 分支 + reconcile 路径都用它;CAS 输 → skip spawn_reverse | `store.py:147-192`,`watcher.py` 3 处调用 |
| **6** | `_spawn_reverse_conditional` 里反手的 `stop_offset_pct_of_line` 之前硬编码 0,现在读 `src.order.reverse_stop_offset_pct`,0/空 fallback 到源单的 stop | `watcher.py` 在 `_spawn_reverse_conditional` 的 OrderConfig 构造 |
| **8** | 线破 check 用 strict mark,不再用 lastPr 兑水,跟 Bitget 的 trigger 语义对齐 | 见 Bug 4 同一处 |
| **12** | 删了 `_reconcile_against_bitget` 里 `return` 之后的一整块死代码 | `watcher.py` reconcile 尾部 |
| **I2** | 三处 early-return(mark 缺失 / cancel 失败 ×2)都会 `cond.last_poll_ts = now; _store.update(cond)` 再 return | 线破 check 的三个早退路径 |

### 2. 测试:5 个新测试全过

文件 `tests/conditionals/test_watcher_replan.py`,加了 5 个线破专项测试:

1. `test_line_broken_short_cancels_when_mark_above_line_plus_stop_pct` — BAS 场景重现
2. `test_line_broken_cancel_FAILS_both_paths_leaves_triggered` — Bug 1+2
3. `test_line_broken_cancel_first_path_raises_then_fallback_succeeds` — Bug 2
4. `test_mark_fetch_returns_none_aborts_maybe_replan` — Bug 4
5. `test_line_broken_cas_lost_skips_reverse_spawn` — Bug 5

跑法:
```bash
cd "C:\Users\alexl\Desktop\crypto-analysis-"
python -m pytest tests/conditionals/test_watcher_replan.py -v
```

应该看到 7 passed(5 新 + 2 原有)。

**同目录下还有 3 个 pre-existing fail**(`test_line_projection.py`, `test_manual_line_order_api.py`)—— 跟本次工作无关,是之前 log 插值和 ref-price 语义变更留下的。**不要当成本次 regression**。

---

## ⚠️ 还没修的 2 个新 bug(用户刚指出来,紧迫)

### BUG A(HTTP 500 真因)— Unicode 字符炸 Windows 控制台

**证据** (`data/uvicorn_codex_err.log`):
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 74: character maps to <undefined>
```

Windows Python 3.12 的 stdout 默认 cp1252 编码,遇到非 ASCII 字符(比如 `→` U+2192、`—` U+2014)会抛 UnicodeEncodeError,被 FastAPI middleware 变成 HTTP 500。

**已扫描到的有问题 print(至少)**:
- `server/routers/conditionals.py:654-656` `→` (SL distance-cap log)
- `server/routers/conditionals.py:662-663` `→`
- `server/routers/conditionals.py:855` `—` (cond create failed rollback)
- `server/routers/conditionals.py:866` `—`
- `server/conditionals/watcher.py:1248` `—` (reconcile SKIP)
- `server/conditionals/watcher.py:1331, 1663, 1710` `—` (CAS lost)
- 其他:跑下面这条扫全 server/:

```bash
cd "C:\Users\alexl\Desktop\crypto-analysis-"
python -c "
import re, os
hits = []
for root, _, files in os.walk('server'):
    for f in files:
        if not f.endswith('.py'): continue
        p = os.path.join(root, f)
        try: src = open(p, encoding='utf-8').read()
        except Exception: continue
        for lineno, line in enumerate(src.split(chr(10)), 1):
            if 'print(' in line:
                for m in re.finditer(r'[^\x00-\x7f]', line):
                    hits.append((p, lineno, ord(m.group()), line.strip()[:100]))
import sys
for h in hits: sys.stdout.buffer.write(f'{h[0]}:{h[1]} U+{h[2]:04X} : {h[3]}\n'.encode('utf-8'))
" 2>&1 | head -50
```

**两种修法任选**:
1. **替换非 ASCII 字符**:`→` → `->`,`—` → `--`,`⏳` → `[wait]`,等等
2. **在 `main.py` 启动时强制 stdout = utf-8**:
   ```python
   import sys
   sys.stdout.reconfigure(encoding='utf-8', errors='replace')
   sys.stderr.reconfigure(encoding='utf-8', errors='replace')
   ```

**推荐 2**(更彻底),但也要把 print 里的中文 / emoji 检查一遍,免得还有其他编码坑。

### BUG B(钱级)— 快捷挂单 popup 默认方向跟用户画的线侧搞反了

**场景**:用户画下降 resistance 线(但在数据里 `side='support'`,这是另一个更深的问题),打开快捷挂单 popup 准备 SHORT,直接点 setup row(**没点 SHORT 切换按钮**),结果 **backend 收到 `dir=long`,Bitget 开了 BUY 单**。

**证据** (`data/uvicorn_codex_out.log`):
```
[place-line-order] line=0.0190->0.0113 ref@0.0134 mark=0.0131 kind=bounce dir=long mark_ms=716
...
[place-line-order] bitget_submit ok=True ... → ended up as buy · 22128 on Bitget
```

用户点的本意是 short,界面以为已经是 short(因为 setup 名字里有"做空"),但 popup 顶部的 long/short 切换按钮默认值还是 `long`(因为 `line.side === 'support'`)。

**代码位置** `frontend/js/workbench/drawings/trade_plan_modal.js:1089`:
```javascript
let direction = line?.side === 'support' ? 'long' : 'short';
```

**两种修法任选**:
1. **保守**:popup 打开时 `direction = null`,必须用户显式点一下 long/short,才能点 setup;否则 setup row 禁用 + 提示"先选方向"
2. **根据 setup 的名字/config 推断**:setup 存的 config 没有 direction 字段(`_stripPerTradeFields` 剥掉了),但 setup 名字里往往有"做多/做空",可以从名字解析 —— 脆弱,不推荐
3. **折中**:保留 line.side 推断的默认值,但在 popup 顶部用大字提示"当前方向 做多 LONG,点这里切换"—— 只改 UI 不改逻辑

推荐 **1**,最安全。

---

## 优先级(按用户钱包影响)

1. **BUG A + BUG B** —— 都是当下真实会亏钱 / 下错方向,最紧急
2. **跑回归**:`pytest tests/conditionals/test_watcher_replan.py -v`,应该 7 passed
3. **Playwright 端到端验证线破 cancel**:用户之前强调必须有 browser log 才能说"修好了"。脚本起点:`scripts/test_draw_line_real.py`,`data/logs/ui_tests/` 下放测试输出
4. **Bug 3 — trailing 不跟 filled**:之前误报说被 reconcile 间接修了,但其实 reconcile 只在 cancel 失败 → 60s 后才补救。真要严密还是得在 `_register_manual_trailing_if_position_open` 那里加上 filled 状态下的线破直接平仓路径
5. **Bug I3 — line 外推到 0**:BAS 当前 cond 是 `extend_right: false` 所以不中,但画线扩展默认是 true,任何挂 10+ 小时没触发的单都会吃到;在 `types.py:line_price_at` 里加 clamp:外推超过原跨度就冻结 `price_end`
6. **Bug I5 — set_status + event 非原子**:nice-to-have,给 store 加个 `set_status_with_event` 合并成一次写,防进程中间崩导致"已撤销但无原因"
7. **新 BUG C:BAS 那根"support"线实际上是下降 resistance**:前端画线时 `side='support'` 但 `price_start > price_end`(降的)—— 要么 UI 根据斜率自动把 side 改 `resistance`,要么强制用户手选 support/resistance 不按首次点击自动分类

---

## 关键文件地图(Codex 接手必读)

| 文件 | 功能 |
|------|------|
| `CLAUDE.md` | **先读**,顶部"绝对硬规则"列了不能瞎编数字、不能谎称 done |
| `server/conditionals/watcher.py` | 有 `_maybe_replan`、`_reconcile_against_bitget`、`_cancel_bitget_plan_safely`、`_fetch_mark_price_strict`、`_spawn_reverse_conditional`;线破所有逻辑都在这 |
| `server/conditionals/store.py` | `set_status`、`set_status_if`(CAS)、`append_event`、`update`;全部 `threading.Lock` 保护 |
| `server/routers/conditionals.py` | place-line-order 入口(544 行起),`PlaceLineOrderReq` schema,SL cap 逻辑(650 行起)—— **这里有 BUG A 的 Unicode 字符** |
| `server/execution/live_adapter.py` | Bitget API 封装;`submit_live_plan_entry`、`cancel_order`、`cancel_plan_order_any_type`;`side = "buy" if intent.side == "long" else "sell"` 在 line 266 |
| `frontend/js/workbench/drawings/trade_plan_modal.js` | 完整挂单 modal 在 `openTradePlanModal`;快捷 popup 在 `openQuickTradePopup`(1079 行)—— **BUG B 在 1089 行** |
| `frontend/js/workbench/drawings/chart_drawing.js` | 图上右键菜单,"⚡ 交易(快捷挂单)"的入口 |
| `tests/conditionals/test_watcher_replan.py` | 我加的 5 个线破专项测试 |
| `data/conditional_orders.json` | 所有 conditional 的真实状态 + event log。BAS 那笔是 `cond_99f378f01e8bbf`(大约 9503 行) |
| `data/user_drawings_ml.jsonl` | 画线 + 挂单 + 平仓 event,**查真实数字的主要来源** |
| `data/uvicorn_codex_out.log` / `_err.log` | 服务 stdout / stderr,`grep "place-line-order"` 看真实挂单流,`grep "Traceback"` 看崩溃 |

## 跑 / 查清单

每次做任何决策前,先跑:

```bash
# 语法检查
cd "C:\Users\alexl\Desktop\crypto-analysis-"
python -c "import ast; [ast.parse(open(p, encoding='utf-8').read()) for p in ['server/conditionals/watcher.py','server/conditionals/store.py','server/routers/conditionals.py']]; print('OK syntax')"

# 跑线破测试(应 7 passed)
python -m pytest tests/conditionals/test_watcher_replan.py -v

# 跑全 conditionals(3 个 pre-existing fail 不是你的锅)
python -m pytest tests/conditionals/ --tb=line

# 查最近服务错误
grep -a "Traceback\|Internal Server Error" data/uvicorn_codex_err.log | tail -20

# 查 BAS 最近的 conditional 状态
grep -A 5 "cond_99f378f01e8bbf" data/conditional_orders.json | head -40
```

## 用户跟你说话的风格

- 中文,直接,往往着急
- 会骂人("我真的要疯了啊") —— 当成信号而不是噪声:他骂人多半是因为你在乱讲 / 走偏 / 没听懂
- 听到"我要自杀"之类的话要严肃对待 —— 短一句关心 + 988 热线,然后回到正事,不要长篇说教
- 他会纠正你的错误,正面接,**承认错、复述对的理解、再继续**,不要绕
- 不确定的事,回答格式是"让我查一下"+ 立刻 Read/Grep/Bash,**绝对不许**"我猜 / 应该是 / 可能"+ 编数字

## 当前 git 状态(别覆盖)

```
M data/user_drawing_labels.jsonl
M data/user_drawing_outcomes.jsonl
M server/conditionals/store.py
M server/conditionals/watcher.py
M tests/conditionals/test_watcher_replan.py
?? scripts/_test_replan_across_bars.py   (一个我没用的探索脚本,可以删)
```

**没 commit,没 push**。用户会在测完 BUG A + B 之后决定一起 commit 还是分开。

---

## TL;DR 你接手后先做什么

1. `Read` `CLAUDE.md` 全文(**必须**)
2. 跑语法 + `pytest tests/conditionals/test_watcher_replan.py -v` 确认上一 agent 的工作是绿的
3. **先修 BUG A**(Unicode → HTTP 500),方法建议用 `sys.stdout.reconfigure(encoding='utf-8')`,改 `server/main.py` 启动时加一行
4. **再修 BUG B**(popup 默认方向可能跟用户实际意图不符),改 `trade_plan_modal.js:1089`,建议用保守修法:没显式点方向就禁掉 setup row
5. 修完 A + B,让用户在浏览器里点一次 short 挂单 + 重现线破,收集 Playwright 日志
6. 所有 A/B/C 都稳了之后,按上面的优先级列表往下推 3、4、5、6

记住:**这是真钱。** 每次改动 commit 前问自己:"这改法万一出现 xxx 情况,会让用户损失多少?" 答不清楚就别改。
