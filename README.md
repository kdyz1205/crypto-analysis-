# Trading OS — Manual Trendline → Bitget

A local TradingView-style chart + Bitget plan-order engine built around
**manually-drawn trendlines**. You draw a line on any candle chart, the
server turns it into a Bitget trigger-market plan order, and as the
line's slope moves across TF bars the watcher cancels + replaces the
plan so the trigger price always tracks "the line at current bar-open".

Not an auto-trading bot. You decide the lines; the system takes care
of the placement + line-tracking mechanics + trailing SL after fill.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add BITGET_API_KEY / SECRET / PASSPHRASE
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/v2
```

## Core Flow (画线 → 挂单 → 跟 TF 移动)

```
1. Press T on chart       →  draw mode
2. Click 2 points         →  line committed (POST /api/drawings)
3. Right-click the line   →  menu: ⚡ 交易 / + 创建交易计划 / 添加警报
4. ⚡ 交易 popup          →  pick 做多/做空 + 反手 + setup → place
                             (3 clicks, line → live Bitget plan order)
5. Watcher each TF bar    →  cancel old plan + place new plan at the
                             line's log-projected value × buffer
6. Order fills            →  trailing SL registers and follows the
                             slope of the same line
7. SL hits                →  optional auto-reverse (flip direction)
```

Everything above works against **real Bitget plan orders** — visible
in the Bitget app under 计划委托. No paper sandbox.

## Key Features

### Chart (`/v2`)
- **TradingView Lightweight Charts** — log/lin, pan/zoom, TF 1m…1w
- **Multi-symbol ticker picker** — sortable by 币种 / 交易量 / 涨跌幅
- **OHLC hover strip** — 开/高/低/收/Δ/成交量 on crosshair
- **Timezone picker** — cycles 洛杉矶 / 国内 UTC+8 / UTC for X-axis
- **% change since UTC 00:00** — matches Bitget basis, not rolling 24h
- **Indicators** — MA5/8/13/21/55, Bollinger, Wyckoff, RSI, MACD, ATR
- **Factor markers** — RSI oversold/overbought, Volume Surge, MACD cross

### Line Drawing
- **Press T → draw** (2 clicks = 1 line)
- **Drag line body or anchor** → slope updates live, server gets PATCH,
  watcher force-replans all attached orders to the new geometry
- **Log-scale projection** — line price at any TS uses log-interpolation
  matching the chart's visual rendering (no 0.5% drift bug)
- **Right-click menu** — ⚡ 交易 / + 创建交易计划 / 🔔 警报 / 线宽 / 删除
- **Sidebar "我的手画线"** — all lines across all symbols, grouped by
  coin, click to jump chart + select line

### Trade Plan Setups (localStorage `v2.tradeplan.setups.v1`)
- Named reusable parameter bundles (buffer / stop / rr / leverage / size)
- Setup names auto-regenerate from values: `0.1%损 1%buffer RR5 $30U 10x`
- Direction + 反手 are per-trade (picked in quick popup), NOT in setup
- Save / rename / delete, 3 default seeded on first run

### Order Engine
- **Bitget normal_plan** trigger-market orders via REST
- **Log-interpolated line projection** at current TF bar_open
- **1% stake × 10x leverage = $10 margin / $100 position** (UI hint matches code)
- **Auto-replan** every TF bar boundary (1m / 5m / 1h / 4h / 1d)
- **Drag-triggered replan** — PATCH line → immediate cancel + replace
- **Plan-existence guard** — only skip replan if THIS plan's orderId
  is no longer pending (user's manual positions don't interfere)
- **Trailing SL** after fill — tracks same slope via `_trendline_params`
- **Auto-reverse** on SL hit — spawn opposite-direction conditional
- **Mark-price guard** — prevents SL placement on wrong side of mark

### ML / Analysis Data
- **`data/user_drawings_ml.jsonl`** — every draw / update / delete /
  trigger / close event with market context (ATR, BB state, slope)
- **`data/line_adjustments.jsonl`** — before/after geometry + affected
  orders + market snapshot on every line drag
- **`data/trade_snapshots/`** — JSONL + PNG per trade for VLA training
- **`scripts/export_manual_lines_excel.py`** — exports all drawings +
  orders + outcomes + daily calendar to multi-sheet Excel

## Project Structure

```
├── frontend/
│   ├── v2.html                 # Main SPA
│   ├── v2.css                  # Dark theme + component styles
│   └── js/
│       ├── main.js             # Boot
│       ├── state/              # Pub/sub state (market, drawings)
│       ├── services/           # REST clients (conditionals, drawings…)
│       ├── workbench/
│       │   ├── chart.js        # Lightweight-charts setup + hover + TZ
│       │   ├── conditional_panel.js  # 我的手画线 + 持仓 + 挂单
│       │   ├── drawings/
│       │   │   ├── chart_drawing.js      # Draw state machine + SVG
│       │   │   ├── manual_trendline_controller.js
│       │   │   ├── trade_plan_modal.js   # Full modal + ⚡ quick popup
│       │   │   ├── draw_toolbar.js       # Compact top-right toolbar
│       │   │   └── manual_trendline_overlay.js
│       │   └── indicators/     # MA/BB/RSI/MACD/ATR + factors
│       └── util/               # fetch, format, dom, events
│
├── server/
│   ├── app.py                  # FastAPI routes
│   ├── data_service.py         # Bitget candle fetch + disk cache
│   ├── market/
│   │   └── bitget_client.py    # Bitget public REST
│   ├── execution/
│   │   ├── live_adapter.py     # Bitget signed REST (orders / positions)
│   │   └── types.py            # OrderIntent etc
│   ├── conditionals/
│   │   ├── types.py            # ConditionalOrder + log-interp projection
│   │   ├── store.py            # JSONL-backed
│   │   └── watcher.py          # Replan loop + reconcile + auto-reverse
│   ├── drawings/
│   │   ├── store.py
│   │   └── types.py
│   ├── routers/
│   │   ├── drawings.py         # POST/PATCH/DELETE + /all endpoint
│   │   ├── conditionals.py     # Place-line-order + log projection
│   │   └── live_execution.py   # Bitget account / pending / cancel
│   ├── strategy/
│   │   ├── mar_bb_runner.py    # Trailing SL via _trendline_params
│   │   └── drawing_learner.py  # ML capture hooks
│   └── pattern_engine.py       # Similar-line matching
│
├── scripts/
│   ├── export_manual_lines_excel.py   # Historical drawings → Excel
│   └── ...
│
├── data/                        # Runtime state (gitignored)
│   ├── *.csv.gz                 # OHLCV cache per symbol × TF
│   ├── manual_trendlines.json   # Drawings state
│   ├── conditional_orders.json  # Placed orders state
│   ├── user_drawings_ml.jsonl   # ML event stream
│   ├── line_adjustments.jsonl   # Drag-event log
│   └── trade_snapshots/         # Per-trade JSONL + PNG
│
├── CLAUDE.md                    # Project guardrails for AI agents
├── PRINCIPLES.md                # Immutable rules (data / drawing / order)
└── requirements.txt
```

## Tech Stack

- **Frontend**: Vanilla JS ES modules, TradingView Lightweight Charts 4.x
- **Backend**: FastAPI + Uvicorn, Python 3.11+
- **Data**: Bitget USDT-M Futures REST v2, Polars for candle processing
- **Trading**: Bitget `/api/v2/mix/order/place-plan-order` (normal_plan)
- **Storage**: JSONL (orders, events), CSV.gz (candles), PNG (snapshots)
- **ML**: Event capture for future trendline outcome prediction

## Environment Variables

```
BITGET_API_KEY=...
BITGET_SECRET=...
BITGET_PASSPHRASE=...
# Optional
TELEGRAM_BOT_TOKEN=...   # trigger alerts
TELEGRAM_CHAT_ID=...
ANTHROPIC_API_KEY=...    # AI chat sidebar
```

## Safety Notes

- Every `submit_to_exchange: true` places a **real Bitget order**. No
  paper fallback unless you set `mode: demo` (which Bitget demo env
  accepts separately).
- Watcher auto-cancels + replaces plan orders every TF bar — this is
  intentional but BURNS API quota. If you disable autoreplan, orders
  stay frozen at their placement time's line projection.
- Auto-reverse fires on SL hit. Disable per-setup by leaving
  `reverse_enabled` off.
- `data/conditional_orders.json` is the source of truth for "what we
  placed"; if it diverges from Bitget, reconcile runs every 30s and
  marks missing oids as cancelled.

## Read Before Committing Code

- **`PRINCIPLES.md`** — immutable project rules (data depth = exchange
  physical limit; line persists until user deletes; etc.)
- **`CLAUDE.md`** — AI-agent workflow including "don't lie" protocol
  (no "修好了" without Playwright log proving the full flow)
