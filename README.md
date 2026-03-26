# Crypto Analysis & Autonomous Trading

TradingView-style crypto technical analysis dashboard with V6 autonomous trading strategy and OKX live/paper trading.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment** (optional, for live trading & AI chat):
   ```bash
   cp .env.example .env
   # Edit .env with your OKX API keys and Anthropic API key
   ```

3. **Start server**:
   ```bash
   python run.py
   ```
   Opens `http://127.0.0.1:8001` automatically.

## Features

### Chart & Analysis
- **Candlestick chart** with volume — multi-symbol, multi-timeframe (5m / 15m / 1h / 4h / 1D)
- **OKX data** — live API, paginated history back to 2021, gzip-compressed disk cache
- **MA overlays** — MA5, MA8, EMA21, MA55, Bollinger Bands (toggle on/off)
- **Support/Resistance** — auto-detected horizontal zones + pattern recognition
- **Hand-drawn tools** — trend lines, horizontal lines
- **Replay mode** — scroll back to any date and replay price action
- **Backtest** — MFI/MA strategy backtesting with parameter optimization

### Autonomous Trading Agent (V6 Strategy)
- **4-layer filtered trend following**:
  1. Trend ordering (P > MA5 > MA8 > EMA21 > MA55)
  2. Fanning distance (ATR-normalized MA gap thresholds)
  3. Slope momentum (all MAs trending in same direction)
  4. Bollinger Band position filter
- **Self-evolving** — mutates strategy parameters based on performance metrics
- **Paper & Live mode** — switch with safety confirmation
- **OKX live trading** — automated order placement via OKX REST API
- **Risk management** — max drawdown, position limits, consecutive loss circuit breaker

### Dashboard Panels
- **Agent Dashboard** — equity, PnL, positions, trades, signals, strategy params
- **Trader Console** — pre-trade checklist, risk calculator, trade journal
- **AI Chat** — built-in Claude-powered market analysis assistant
- **Self-Healer** — automatic error detection and recovery

### Toolbar
- Quick command bar (`btc 5m`, `eth 1h`)
- Layout presets (Scalper / Swing / Position)
- Strategy presets (Default / Momentum / Mean Revert / Defensive)
- Data priority (Fast / Balanced / Deep history)

## Tech Stack

- **Frontend**: HTML / CSS / JavaScript, TradingView Lightweight Charts
- **Backend**: FastAPI, Python (Polars, NumPy)
- **Data**: OKX REST API v5 (candles + history-candles)
- **Trading**: OKX perpetual swaps (HMAC-SHA256 signed requests)
- **AI**: Anthropic Claude API (chat + analysis)

## Project Structure

```
├── frontend/          # Single-page app (HTML/CSS/JS)
│   ├── index.html     # Main page with all panels
│   ├── app.js         # Chart, agent, trading logic
│   └── style.css      # Dark theme styles
├── server/            # FastAPI backend
│   ├── app.py         # API routes
│   ├── data_service.py    # OKX data fetching + caching
│   ├── agent_brain.py     # V6 strategy + autonomous agent
│   ├── okx_trader.py      # OKX order execution
│   ├── backtest_service.py # Backtesting engine
│   └── ...
├── sr_patterns.py     # Support/resistance detection
├── run.py             # Entry point
└── requirements.txt
```
