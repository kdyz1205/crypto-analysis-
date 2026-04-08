# Follow-up Plan: Bitget Adapter + Factor Mining + Multi-Strategy

> Written 2026-04-08 after completing P1 explainable endpoints (commit 9a8bfbb)

**Why these three together:** They share the same architectural pressure —
the current agent_brain.py assumes ONE strategy, ONE trader (OKX), ONE set
of V6 params. To support Bitget, multiple simultaneous strategies, AND a
factor-mining research loop, we need to generalize three abstractions at
once. Doing them separately would require three refactors.

---

## Phase A — Trader Abstraction (Bitget)

**Goal:** Decouple agent_brain.py from okx_trader.py. Support Bitget and
(future) Hyperliquid without rewriting strategy logic.

### Target structure
```
server/
  traders/
    __init__.py
    base.py           # BaseTrader ABC
    okx.py            # Renamed from okx_trader.py
    bitget.py         # NEW — mirrors okx.py with Bitget REST/WS
    paper.py          # NEW — pure simulation (no exchange)
    factory.py        # get_trader(exchange_name) -> BaseTrader
```

### BaseTrader interface
```python
class BaseTrader(ABC):
    state: AgentState
    risk: RiskLimits

    @abstractmethod
    async def get_price(self, symbol: str) -> float: ...

    @abstractmethod
    async def get_account_balance(self) -> dict: ...

    @abstractmethod
    async def open_position(self, symbol: str, side: str, size_usd: float) -> dict: ...

    @abstractmethod
    async def close_position(self, symbol: str, reason: str) -> dict: ...

    @abstractmethod
    async def update_positions(self) -> None: ...

    def has_api_keys(self) -> bool: ...
    def set_api_keys(self, api_key, secret, passphrase) -> None: ...
    def can_trade(self, symbol) -> tuple[bool, str]: ...
    def revive(self) -> None: ...
    def check_daily_reset(self) -> None: ...
```

### Bitget specifics
- Base URL: `https://api.bitget.com`
- Auth: HMAC-SHA256 with API key + secret + passphrase (same pattern as OKX)
- Endpoints:
  - `/api/v2/spot/market/tickers`
  - `/api/v2/mix/account/accounts` (futures balance)
  - `/api/v2/mix/order/place-order`
  - `/api/v2/mix/position/all-position`
- Symbol format: `BTCUSDT` (no dash)
- Product type: `USDT-FUTURES`
- Margin mode: `isolated` or `crossed`

### Tasks
1. Create `server/traders/base.py` with BaseTrader ABC
2. Move `okx_trader.py` → `server/traders/okx.py`, make it inherit BaseTrader
3. Update all imports of `from .okx_trader import ...` to `from .traders.okx import ...`
4. Write `server/traders/bitget.py` mirroring okx.py with Bitget auth/endpoints
5. Write `server/traders/factory.py` with `get_trader(name)` dispatch
6. Update `agent_brain.py` to accept trader from factory instead of direct OKXTrader
7. Add `EXCHANGE` env var + `exchange` field in agent_state.json
8. Add exchange selector dropdown in v2 Ops sub-tab
9. Smoke test both OKX and Bitget paths end-to-end

**Estimated effort:** 1 session (6-8 hours of focused work)

---

## Phase B — Multi-Strategy Engine

**Goal:** Run N concurrent strategies, each with own params, own positions,
own audit log. User can compare results live.

### Current limitation
`agent_brain.py` has ONE `AgentBrain` singleton. ONE strategy_params dict.
ONE position per symbol. Adding a second strategy would require duplicating
the singleton.

### Target architecture
```
server/
  strategies/
    __init__.py
    base.py              # BaseStrategy ABC
    v6_trend.py          # Renamed from current V6 generate_signal logic
    ma_ribbon_mtf.py     # From ma_ribbon_service.py
    mfi_ma.py            # From backtest_service.py
    factor_composite.py  # NEW — dynamic composite (for factor mining)
  strategy_engine.py     # NEW — runs multiple BaseStrategy instances concurrently
```

### BaseStrategy interface
```python
class BaseStrategy(ABC):
    strategy_id: str     # unique per instance
    name: str            # human readable
    params: dict
    trader: BaseTrader   # shared or dedicated

    @abstractmethod
    async def generate_signal(self, symbol: str, df) -> dict | None: ...

    @abstractmethod
    def manage_positions(self, symbol: str, df, position) -> dict | None: ...

    # Optional: strategy can veto another's trade
    def compatible_with(self, other_positions: list) -> bool:
        return True
```

### StrategyEngine
```python
class StrategyEngine:
    strategies: dict[str, BaseStrategy]    # strategy_id -> instance
    position_ownership: dict[str, str]     # symbol -> strategy_id
    shared_trader: BaseTrader              # one trader for all strategies

    async def add_strategy(self, strategy: BaseStrategy): ...
    async def remove_strategy(self, strategy_id: str): ...
    async def tick(self):
        # Each strategy runs independently on its assigned symbols
        # but shares the trader + risk limits
        for strat in self.strategies.values():
            await self._tick_strategy(strat)

    async def _tick_strategy(self, strat):
        for symbol in strat.symbols:
            # Only trade symbols we own, or unassigned symbols
            owner = self.position_ownership.get(symbol)
            if owner and owner != strat.strategy_id:
                continue
            signal = await strat.generate_signal(symbol, df)
            if signal and signal["action"] in ("long", "short"):
                result = await self.shared_trader.open_position(...)
                if result["ok"]:
                    self.position_ownership[symbol] = strat.strategy_id
```

### Routes
- `POST /api/strategies/add` — add strategy instance
- `DELETE /api/strategies/{id}` — remove instance
- `GET /api/strategies/list` — list all running + their PnL
- `GET /api/strategies/{id}/status` — per-strategy status
- `POST /api/strategies/{id}/params` — update params

### Tasks
1. Create `server/strategies/base.py` with BaseStrategy ABC
2. Refactor V6 logic from agent_brain.py into `strategies/v6_trend.py`
3. Write `server/strategy_engine.py`
4. Update `agent_brain.py` to wrap StrategyEngine (backwards compat)
5. Add `/api/strategies/*` endpoints
6. Add multi-strategy panel in v2 (Execution Center new sub-tab "Strategies")
7. Each strategy gets its own glassbox filter

**Estimated effort:** 1-2 sessions

---

## Phase C — Factor Mining Panel (Hermes-style)

**Goal:** Automated research loop that can ingest papers/articles, extract
candidate factors, backtest them, and surface winners to the agent.

### Inspired by Hermes
- Persistent memory already exists (Phase 2 work)
- Skills / factors auto-generated from reading
- Delegated subagent work (research runs while main agent trades)

### Factor mining workflow
```
1. User drops a paper URL or PDF into the mining panel
2. Backend fetches content (WebFetch / PDF parse)
3. LLM (Claude) extracts factor definitions:
   {
     "name": "rsi_divergence",
     "description": "RSI bearish divergence at local high",
     "formula": "rsi(14) decreasing while price making higher highs",
     "required_data": ["close", "rsi_14"],
     "hypothesis": "reversal signal in oversold bounces",
     "paper_reference": "arxiv 2024.12345"
   }
4. Backend converts definition → Python callable (using sandboxed eval
   or a DSL like "rsi(14) - rsi_slope(5)")
5. Automatic backtest across top-50 symbols on 4h/1h/1d
6. Results scored by: sharpe, win_rate, max_dd, trade_count, overfit_penalty
7. Top 10 candidates get a "factor card" in the UI
8. User can promote a factor to a strategy (wraps it in BaseStrategy
   and adds to StrategyEngine)
```

### Target structure
```
server/
  mining/
    __init__.py
    ingest.py         # Fetch + parse papers (PDF, arxiv, web)
    extractor.py      # LLM-driven factor extraction
    dsl.py            # Safe factor DSL: rsi(14), sma(20), crossover(a, b), etc
    evaluator.py      # Runs backtest on extracted factors
    registry.py       # Stores discovered factors + scores
    scheduler.py      # Background mining loop (one per X hours)
  routers/
    mining.py         # /api/mining/* endpoints
```

### Routes
- `POST /api/mining/ingest` — add a paper URL or text
- `GET /api/mining/factors` — list all discovered factors with scores
- `GET /api/mining/factors/{id}` — detailed factor card
- `POST /api/mining/factors/{id}/promote` — turn into strategy
- `POST /api/mining/factors/{id}/backtest` — manual re-run
- `GET /api/mining/queue` — current mining backlog

### Safety: the DSL
Rather than letting the LLM write raw Python (injection risk), define a
small safe DSL:
```python
# Allowed:
"rsi(14) < 30 and close > sma(50)"
"crossover(ema(9), ema(21)) and volume > mean(volume, 20)"
"slope(close, 10) > 0 and atr(14) < atr_median(30)"

# Parser produces an AST that only calls pre-registered safe functions
# (rsi, sma, ema, atr, slope, crossover, volume, close, ...)
```

### Tasks
1. Design + implement the factor DSL parser + evaluator
2. Write ingest.py (supports URL, PDF upload, raw text)
3. Write extractor.py using Claude tool calling to extract factors
4. Write evaluator.py that runs the factor against historical data
5. Write registry.py with SQLite or JSON-backed storage
6. Write background scheduler that processes queue
7. Add /api/mining/* router
8. Build v2 Mining panel (new sub-tab under Research Drawer)
9. Per-factor card: name, description, paper_ref, backtest metrics,
   sample trades, promote button

**Estimated effort:** 2-3 sessions (DSL is the tricky part)

---

## Dependencies between phases

```
Phase A (Bitget / trader abstraction) ──┐
                                         ├──→ Phase B (multi-strategy)
                                         │     depends on BaseTrader
                                         │
Phase C (factor mining)                  │
  depends on Phase B (BaseStrategy)  ────┘
  because factors need to wrap as strategies
```

**Recommended order:** A → B → C

---

## Total estimated scope

| Phase | Name | Sessions | Risk |
|-------|------|----------|------|
| A | Bitget adapter | 1 | Low (pattern match) |
| B | Multi-strategy engine | 1-2 | Medium (refactor agent) |
| C | Factor mining | 2-3 | High (DSL + LLM loop) |
| **Total** | | **4-6 sessions** | |

---

## What we can squeeze into the current session

If time permits:
- [ ] Phase A Task 1-3: BaseTrader ABC + move okx.py + update imports
- [ ] Phase A Task 4: skeleton Bitget class (minimal: get_price, get_balance)

Leave for next session:
- [ ] Full Bitget implementation (orders, positions)
- [ ] Phase B entirely
- [ ] Phase C entirely
