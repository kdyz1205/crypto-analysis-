"""
AI Chat Engine — Multi-model chat with market data tools.

Integrates Claude (and extensible to other models) into the web app.
The AI has access to:
- Real-time price data (OKX)
- Pattern analysis (S/R lines, trendlines)
- Backtest results
- Agent status & trading
- Chart context (current symbol, interval, etc.)

Like OpenClaw: AI reasons about markets, remembers context, and helps trade.
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "autoresearch" / "chat_memory"

# Auto-load .env so API key is available even when run from any directory
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

# ── Model configs ──
MODELS = {
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5",
        "label": "Claude Haiku 4.5 (fast)",
        "max_tokens": 2048,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5",
        "label": "Claude Sonnet 4.5",
        "max_tokens": 4096,
    },
    "claude-opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5",
        "label": "Claude Opus 4.5 (deep thinking)",
        "max_tokens": 4096,
    },
}

DEFAULT_MODEL = "claude-sonnet"

SYSTEM_PROMPT = """你是 Crypto TA 内置的 AI 分析师，直接嵌入在交易分析平台里。

你的能力：
1. **市场分析**：你可以获取实时价格、K线数据、技术指标（MFI、RSI、MA、ATR、布林带）
2. **形态识别**：你可以调用系统的形态识别引擎，分析支撑/阻力线、趋势
3. **回测**：你可以运行策略回测，评估参数表现
4. **交易 Agent**：你可以查看 Agent 状态、启动/停止 Agent

你的风格：
- 直接、简洁、数据驱动
- 中英混用，术语用英文
- 给出具体数字和判断，不要模糊
- 如果不确定，说明原因而不是瞎猜

当用户问到具体币种时，用工具获取最新数据再回答。"""


class ChatSession:
    """One chat conversation with history."""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.messages: list[dict] = []
        self.model: str = DEFAULT_MODEL
        self.created_at: float = time.time()

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "messages": self.messages,
            "created_at": self.created_at,
        }


class AIChatEngine:
    """
    Multi-model chat engine with market data tools.
    """

    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self._anthropic_client = None
        self._init_anthropic()

    def _init_anthropic(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
                print("[AI Chat] Anthropic client initialized")
            except Exception as e:
                print(f"[AI Chat] Anthropic init failed: {e}")
        else:
            print("[AI Chat] No ANTHROPIC_API_KEY set. Chat will use mock mode.")

    def get_session(self, session_id: str = "default") -> ChatSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
        return self.sessions[session_id]

    def list_models(self) -> list[dict]:
        return [
            {"id": k, "label": v["label"], "provider": v["provider"]}
            for k, v in MODELS.items()
        ]

    # ── Tool definitions for Claude ──

    def _get_tools(self) -> list[dict]:
        """Tools that Claude can call to access market data."""
        return [
            {
                "name": "get_price",
                "description": "Get current real-time price for a crypto symbol (e.g. BTCUSDT, ETHUSDT, HYPEUSDT)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol like BTCUSDT"}
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_market_data",
                "description": "Get OHLCV data + technical indicators (MA, EMA, RSI, MFI, ATR, BB) for analysis. Returns last 20 bars with indicators.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol like BTCUSDT"},
                        "interval": {"type": "string", "description": "Timeframe: 5m, 15m, 1h, 4h, 1d", "default": "4h"},
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "run_backtest",
                "description": "Run MFI/MA strategy backtest on a symbol. Returns trade count, win rate, PnL, Sharpe ratio.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "interval": {"type": "string", "default": "4h"},
                        "days": {"type": "integer", "default": 365},
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_patterns",
                "description": "Detect support/resistance lines and trend patterns for a symbol.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "interval": {"type": "string", "default": "4h"},
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_agent_status",
                "description": "Get current trading agent status: equity, positions, PnL, generation, running state.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "agent_action",
                "description": "Control the trading agent: start, stop, or revive it.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["start", "stop", "revive"]},
                    },
                    "required": ["action"],
                },
            },
        ]

    # ── Tool execution ──

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as string."""
        try:
            if tool_name == "get_price":
                return await self._tool_get_price(tool_input["symbol"])
            elif tool_name == "get_market_data":
                return await self._tool_get_market_data(
                    tool_input["symbol"],
                    tool_input.get("interval", "4h"),
                )
            elif tool_name == "run_backtest":
                return await self._tool_run_backtest(
                    tool_input["symbol"],
                    tool_input.get("interval", "4h"),
                    tool_input.get("days", 365),
                )
            elif tool_name == "get_patterns":
                return await self._tool_get_patterns(
                    tool_input["symbol"],
                    tool_input.get("interval", "4h"),
                )
            elif tool_name == "get_agent_status":
                return self._tool_get_agent_status()
            elif tool_name == "agent_action":
                return await self._tool_agent_action(tool_input["action"])
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _tool_get_price(self, symbol: str) -> str:
        import httpx
        symbol = symbol.upper().replace("USDT", "")
        inst_id = f"{symbol}-USDT-SWAP"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://www.okx.com/api/v5/market/ticker",
                    params={"instId": inst_id},
                )
                data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                d = data["data"][0]
                return json.dumps({
                    "symbol": f"{symbol}USDT",
                    "price": float(d["last"]),
                    "24h_high": float(d.get("high24h", 0)),
                    "24h_low": float(d.get("low24h", 0)),
                    "24h_vol": float(d.get("vol24h", 0)),
                    "24h_change_pct": round((float(d["last"]) - float(d.get("open24h", d["last"]))) / float(d.get("open24h", d["last"])) * 100, 2) if float(d.get("open24h", 0)) > 0 else 0,
                })
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"error": "Price not found"})

    async def _tool_get_market_data(self, symbol: str, interval: str) -> str:
        import numpy as np
        from .data_service import get_ohlcv_with_df
        from .backtest_service import _mfi, _sma, _ema, _atr, _rsi

        symbol = symbol.upper().replace("/", "")
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        try:
            df, _ = await get_ohlcv_with_df(symbol, interval, days=90)
            if df is None or df.is_empty():
                return json.dumps({"error": f"No data for {symbol}"})

            close = df["close"].to_numpy().astype(float)
            high = df["high"].to_numpy().astype(float)
            low = df["low"].to_numpy().astype(float)
            volume = df["volume"].to_numpy().astype(float)

            rsi = _rsi(close, 14)
            mfi = _mfi(high, low, close, volume, 14)
            ma8 = _sma(close, 8)
            ema21 = _ema(close, 21)
            ma55 = _sma(close, 55)
            atr = _atr(high, low, close, 14)

            # Return last 20 bars
            n = min(20, len(close))
            bars = []
            for j in range(len(close) - n, len(close)):
                bar = {
                    "close": round(float(close[j]), 4),
                    "high": round(float(high[j]), 4),
                    "low": round(float(low[j]), 4),
                    "volume": round(float(volume[j]), 2),
                }
                if not np.isnan(rsi[j]): bar["rsi"] = round(float(rsi[j]), 1)
                if not np.isnan(mfi[j]): bar["mfi"] = round(float(mfi[j]), 1)
                if not np.isnan(ma8[j]): bar["ma8"] = round(float(ma8[j]), 4)
                if not np.isnan(ema21[j]): bar["ema21"] = round(float(ema21[j]), 4)
                if not np.isnan(ma55[j]): bar["ma55"] = round(float(ma55[j]), 4)
                if not np.isnan(atr[j]): bar["atr"] = round(float(atr[j]), 4)
                bars.append(bar)

            # Summary
            latest = bars[-1]
            ma_stack = "YES" if (latest.get("ma8", 0) > latest.get("ema21", 0) > latest.get("ma55", 0) and latest["close"] > latest.get("ma8", 0)) else "NO"

            return json.dumps({
                "symbol": symbol,
                "interval": interval,
                "total_bars": len(close),
                "latest": latest,
                "ma_stacking": ma_stack,
                "trend": "BULLISH" if ma_stack == "YES" else ("BEARISH" if latest["close"] < latest.get("ma55", latest["close"]) else "NEUTRAL"),
                "last_20_bars": bars,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _tool_run_backtest(self, symbol: str, interval: str, days: int) -> str:
        from .backtest_service import run_backtest
        from .data_service import get_ohlcv_with_df

        symbol = symbol.upper().replace("/", "")
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        try:
            df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
            if df is None or df.is_empty():
                return json.dumps({"error": f"No data for {symbol}"})

            result = run_backtest(df)
            # Return summary only (not all trade details)
            return json.dumps({
                "symbol": symbol,
                "interval": interval,
                "total_trades": result.get("total_trades", 0),
                "wins": result.get("wins", 0),
                "losses": result.get("losses", 0),
                "win_rate": result.get("win_rate", 0),
                "total_pnl_pct": result.get("total_pnl_pct", 0),
                "sharpe_ratio": result.get("sharpe_ratio"),
                "max_drawdown_pct": result.get("max_drawdown_pct"),
                "profit_factor": result.get("profit_factor"),
                "avg_win": result.get("avg_win", 0),
                "avg_loss": result.get("avg_loss", 0),
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _tool_get_patterns(self, symbol: str, interval: str) -> str:
        from .data_service import get_ohlcv_with_df
        from .pattern_service import get_patterns_from_df

        symbol = symbol.upper().replace("/", "")
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        try:
            df, _ = await get_ohlcv_with_df(symbol, interval, days=90)
            if df is None or df.is_empty():
                return json.dumps({"error": f"No data for {symbol}"})

            result = get_patterns_from_df(df, symbol, interval)
            support = result.get("supportLines", [])
            resistance = result.get("resistanceLines", [])
            trend_label = result.get("trendLabel", "UNKNOWN")
            trend_slope = result.get("trendSlope", 0)

            return json.dumps({
                "symbol": symbol,
                "interval": interval,
                "trend": trend_label,
                "trend_slope": round(trend_slope, 4),
                "support_lines": len(support),
                "resistance_lines": len(resistance),
                "support_levels": [round(float(l.get("y2", l.get("y1", 0))), 4) for l in support[:5]],
                "resistance_levels": [round(float(l.get("y2", l.get("y1", 0))), 4) for l in resistance[:5]],
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_get_agent_status(self) -> str:
        try:
            from .app import get_agent
            agent = get_agent()
            return json.dumps(agent.get_status())
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _tool_agent_action(self, action: str) -> str:
        try:
            from .app import get_agent
            agent = get_agent()
            if action == "start":
                agent.start()
                return json.dumps({"ok": True, "message": "Agent started"})
            elif action == "stop":
                agent.stop()
                return json.dumps({"ok": True, "message": "Agent stopped"})
            elif action == "revive":
                agent.trader.revive()
                agent._save_state()
                return json.dumps({"ok": True, "message": "Agent revived"})
            return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Chat with Claude ──

    async def chat(self, message: str, session_id: str = "default", model: str | None = None) -> dict:
        """
        Send a message and get AI response.
        Returns: {"reply": str, "model": str, "tool_calls": list}
        """
        session = self.get_session(session_id)
        if model:
            session.model = model

        session.add_user(message)

        model_key = session.model or DEFAULT_MODEL
        model_cfg = MODELS.get(model_key, MODELS[DEFAULT_MODEL])

        if not self._anthropic_client:
            # Mock mode: return a helpful message about setting up API key
            reply = self._mock_reply(message)
            session.add_assistant(reply)
            return {"reply": reply, "model": model_key, "tool_calls": [], "mock": True}

        # Build messages for Claude
        claude_messages = []
        for m in session.messages:
            claude_messages.append({"role": m["role"], "content": m["content"]})

        try:
            tool_calls_log = []

            # Initial call with tools
            response = self._anthropic_client.messages.create(
                model=model_cfg["model_id"],
                max_tokens=model_cfg["max_tokens"],
                system=SYSTEM_PROMPT,
                messages=claude_messages,
                tools=self._get_tools(),
            )

            # Handle tool use loop (max 5 iterations)
            iterations = 0
            while response.stop_reason == "tool_use" and iterations < 5:
                iterations += 1

                # Collect all tool use blocks
                tool_results = []
                assistant_content = response.content

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_calls_log.append({"tool": tool_name, "input": tool_input})
                        print(f"[AI Chat] Tool call: {tool_name}({json.dumps(tool_input)[:100]})")

                        result = await self._execute_tool(tool_name, tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                # Add assistant response + tool results to messages
                claude_messages.append({"role": "assistant", "content": assistant_content})
                claude_messages.append({"role": "user", "content": tool_results})

                # Continue conversation with tool results
                response = self._anthropic_client.messages.create(
                    model=model_cfg["model_id"],
                    max_tokens=model_cfg["max_tokens"],
                    system=SYSTEM_PROMPT,
                    messages=claude_messages,
                    tools=self._get_tools(),
                )

            # Extract final text response
            reply_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    reply_text += block.text

            session.add_assistant(reply_text)

            # Save memory
            self._save_memory(session)

            return {
                "reply": reply_text,
                "model": model_key,
                "tool_calls": tool_calls_log,
            }

        except Exception as e:
            error_msg = f"AI 调用失败: {str(e)}"
            print(f"[AI Chat] Error: {e}")
            session.add_assistant(error_msg)
            return {"reply": error_msg, "model": model_key, "tool_calls": [], "error": True}

    def _mock_reply(self, message: str) -> str:
        """Mock reply when no API key is set."""
        return (
            "⚠️ ANTHROPIC_API_KEY 未设置。\n\n"
            "要启用 AI 对话功能，请设置环境变量：\n"
            "```\n"
            "set ANTHROPIC_API_KEY=sk-ant-...\n"
            "```\n"
            "然后重启服务器。\n\n"
            "设置后你就可以：\n"
            "- 问我任何币种的分析\n"
            "- 让我跑回测\n"
            "- 让我分析形态\n"
            "- 控制交易 Agent"
        )

    def _save_memory(self, session: ChatSession):
        """Save chat memory to disk."""
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            path = MEMORY_DIR / f"{session.session_id}.json"
            path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2))
        except Exception:
            pass

    def clear_session(self, session_id: str = "default"):
        if session_id in self.sessions:
            del self.sessions[session_id]
