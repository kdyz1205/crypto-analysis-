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
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

# Claude Code CLI mode (compatible with your `claude tg bot` idea)
# When BRIDGE_MODE=true, we call `claude-code` CLI instead of Anthropic SDK tool-calling.
CLAUDE_CODE_CLI_MODE = os.environ.get("BRIDGE_MODE", "false").lower() == "true"
# Debug: confirm env propagation inside the running server process.
print(f"[AI Chat] CLAUDE_CODE_CLI_MODE={CLAUDE_CODE_CLI_MODE} BRIDGE_MODE={os.environ.get('BRIDGE_MODE')!r}")
CLAUDE_CODE_CMD = os.environ.get(
    "CLAUDE_CODE_CMD",
    os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
)
CLAUDE_CODE_TIMEOUT_SECONDS = int(os.environ.get("CLAUDE_CODE_TIMEOUT_SECONDS", "300"))
CLAUDE_CODE_DANGEROUSLY_SKIP_PERMISSIONS = (
    os.environ.get("CLAUDE_CODE_DANGEROUSLY_SKIP_PERMISSIONS", "true").lower() == "true"
)

# Keep system prompt compact (token saver). Claude Code CLI has its own tool abilities.
CLAUDE_CODE_SYSTEM_PROMPT = """You are a coding assistant inside a local project.
User asks you to change or debug the crypto web app.
Rules:
- Only modify files within this project root.
- Prefer small, safe patches; then verify by running the app or checking outputs.
- Allowed areas: server/*.py, frontend/app.js, frontend/index.html, frontend/style.css.
- Do not touch .env or any API keys.
Return: what you changed + what to verify next.
If you are unsure, explain why briefly and propose a minimal next experiment.
"""

# Reuse your `claude tg bot` environment (especially API keys) for better compatibility.
CLAUDE_TG_BOT_ENV = (
    Path("C:/Users/alexl/Desktop/claude tg bot/.env")
    if os.name == "nt"
    else Path(os.path.expanduser("~/Desktop/claude tg bot/.env"))
)

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

SYSTEM_PROMPT = """你是 Crypto TA 内置的 AI 工程师 + 分析师，直接嵌入在交易分析平台里。

你的能力：
## 市场分析
1. **实时价格**：get_price — 获取任何币种实时价格
2. **技术指标**：get_market_data — K线 + MFI/RSI/MA/ATR/BB
3. **形态识别**：get_patterns — 支撑/阻力线、趋势分析
4. **回测**：run_backtest — 策略回测，评估参数表现
5. **交易 Agent**：get_agent_status / agent_action — 查看/控制 Agent

## 代码进化（自我修复/强化）
6. **读代码**：read_file — 读取项目中的任何源文件
7. **写代码**：edit_file — 用 find/replace 修改代码。uvicorn --reload 会自动重启
8. **看结构**：list_files — 查看项目文件列表
9. **Git 快照**：git_snapshot — 修改前自动创建 git commit 保存点

可修改的文件范围（白名单）：
- server/*.py — 后端 Python
- frontend/app.js — 前端 JS
- frontend/index.html — 前端 HTML
- frontend/style.css — 前端 CSS

⚠️ 代码修改安全规则：
- 修改前必须先 git_snapshot 保存
- 用 read_file 确认当前内容再改
- find 必须精确匹配原文
- 每次只改一处，小步迭代
- 不碰 .env、不碰密钥

你的风格：
- 直接、简洁、数据驱动
- 中英混用，术语用英文
- 给出具体数字和判断，不要模糊
- 如果不确定，说明原因而不是瞎猜

当用户问到具体币种时，用工具获取最新数据再回答。
当用户要求改代码/修bug/加功能时，先 read_file 看现状，再 git_snapshot，再 edit_file。"""


class ChatSession:
    """One chat conversation with history."""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.messages: list[dict] = []
        self.model: str = DEFAULT_MODEL
        # Session id used by Claude Code CLI resume (when BRIDGE_MODE=true).
        self.claude_code_session_id: str | None = None
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
            "claude_code_session_id": self.claude_code_session_id,
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

    async def _chat_with_claude_code_cli(
        self, user_message: str, session: ChatSession
    ) -> tuple[str, str | None]:
        """
        Use Claude Code CLI (same idea as your `claude tg bot`):
        - stdin: user message
        - resume: keeps conversation state per chat session
        - stdout: JSON {result, session_id}
        """
        import asyncio
        import json as _json

        session_id = session.claude_code_session_id
        args = [
            CLAUDE_CODE_CMD,
            "-p",
            "--output-format",
            "json",
        ]
        if CLAUDE_CODE_DANGEROUSLY_SKIP_PERMISSIONS:
            args.append("--dangerously-skip-permissions")
        if session_id:
            args.extend(["--resume", session_id])
        else:
            args.extend(["--append-system-prompt", CLAUDE_CODE_SYSTEM_PROMPT])

        # NOTE: On Windows, `asyncio.create_subprocess_exec` may raise NotImplementedError.
        # So we run the Claude Code CLI in a worker thread instead.
        import subprocess as _subprocess

        def _run_cli_sync() -> tuple[bytes, bytes]:
            env = os.environ.copy()
            # Load provider keys for claude-code CLI from your Telegram bot env.
            if CLAUDE_TG_BOT_ENV and CLAUDE_TG_BOT_ENV.exists():
                try:
                    from dotenv import dotenv_values

                    updates = dotenv_values(str(CLAUDE_TG_BOT_ENV))
                    for k, v in (updates or {}).items():
                        if v:
                            env[k] = v
                except Exception:
                    pass

            completed = _subprocess.run(
                args,
                input=user_message.encode("utf-8"),
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                timeout=CLAUDE_CODE_TIMEOUT_SECONDS,
                env=env,
            )
            return completed.stdout or b"", completed.stderr or b""

        try:
            stdout_data, stderr_data = await asyncio.to_thread(_run_cli_sync)
        except _subprocess.TimeoutExpired:
            raise asyncio.TimeoutError()

        raw = stdout_data.decode("utf-8", errors="replace").strip()
        new_session_id = None
        response_text = ""

        if raw:
            try:
                data = _json.loads(raw)
                response_text = (data.get("result") or "").strip()
                new_session_id = data.get("session_id")
                if not response_text and data.get("is_error"):
                    response_text = f"Error: {data.get('error', 'Unknown error')}"
            except Exception:
                # CLI might output non-JSON; fallback to raw stdout
                response_text = raw

        if not response_text:
            err = stderr_data.decode("utf-8", errors="replace").strip()
            if err:
                response_text = f"Error: {err[:1200]}"
            else:
                response_text = "(No response)"

        return response_text, new_session_id

    def _looks_like_billing_or_credit_error(self, text: str) -> bool:
        t = (text or "").lower()
        # Common patterns from Anthropic/Gemini billing messages.
        return (
            ("credit balance is too low" in t)
            or ("credit balance" in t)
            or ("plans & billing" in t)
            or ("your credit balance" in t)
            or ("余额不足" in t)
            or ("billing" in t and "credit" in t)
        )

    def _load_gemini_api_key_from_env(self) -> str:
        # Prefer currently loaded env first
        key = os.environ.get("GEMINI_API_KEY", "") or ""
        if key:
            return key
        # Then try claude tg bot env file
        try:
            if CLAUDE_TG_BOT_ENV and CLAUDE_TG_BOT_ENV.exists():
                from dotenv import dotenv_values

                updates = dotenv_values(str(CLAUDE_TG_BOT_ENV))
                key = (updates or {}).get("GEMINI_API_KEY", "") or ""
                return key
        except Exception:
            pass
        return ""

    def _tools_to_gemini(self, tool_defs: list[dict]) -> list[Any]:
        """Convert OpenAI-style tool defs to google-genai function tools."""
        from google.genai import types as gtypes

        TYPE_MAP = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
        }

        declarations = []
        for t in tool_defs:
            schema = t.get("input_schema") or {}
            props = {}
            for name, prop in (schema.get("properties") or {}).items():
                ptype = TYPE_MAP.get(prop.get("type", "string"), "STRING")
                p = {"type": ptype}
                if "description" in prop:
                    p["description"] = prop["description"]
                if "enum" in prop:
                    p["enum"] = prop["enum"]
                # Arrays
                if ptype == "ARRAY":
                    item_type = TYPE_MAP.get(
                        (prop.get("items") or {}).get("type", "string"),
                        "STRING",
                    )
                    p["items"] = gtypes.Schema(type=item_type)
                props[name] = gtypes.Schema(**p)

            declarations.append(
                gtypes.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=gtypes.Schema(
                        type="OBJECT",
                        properties=props,
                        required=schema.get("required", []),
                    ),
                )
            )

        return [gtypes.Tool(function_declarations=declarations)]

    async def _chat_with_gemini_tool_call(self, session: ChatSession, model_key: str) -> dict:
        """
        Gemini tool calling loop.
        Returns: {"reply": str, "backend": "gemini"}.
        """
        gemini_key = self._load_gemini_api_key_from_env()
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY not set (neither env nor claude tg bot/.env)")

        from google import genai as google_genai
        from google.genai import types as gtypes

        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        client = google_genai.Client(api_key=gemini_key)

        # Short instruction (Gemini + tools)
        system_instruction = (
            "你是 Crypto TA 的工程师。遇到任务请直接使用工具执行并返回结果，"
            "尽量简短，必要时先给诊断再给补丁方案。"
        )

        tool_defs = self._get_tools()
        tools = self._tools_to_gemini(tool_defs)

        # Convert conversation history to Gemini contents
        contents = []
        for m in session.messages:
            content = m.get("content")
            if isinstance(content, str):
                text = content
            else:
                # tool_result lists/dicts: keep as compact JSON text
                try:
                    text = json.dumps(content, ensure_ascii=False)
                except Exception:
                    text = str(content)
            role = "user" if m.get("role") == "user" else "model"
            contents.append(gtypes.Content(role=role, parts=[gtypes.Part(text=text)]))

        cfg = gtypes.GenerateContentConfig(system_instruction=system_instruction, tools=tools)

        loop = asyncio.get_running_loop()
        for _ in range(6):  # tool iterations
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=gemini_model,
                    contents=contents,
                    config=cfg,
                ),
            )

            candidate = (response.candidates or [None])[0]
            parts = []
            if candidate and getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
                parts = candidate.content.parts or []

            fn_parts = [
                p
                for p in parts
                if getattr(p, "function_call", None) and getattr(p.function_call, "name", None)
            ]

            if fn_parts:
                # Append function call(s) from model
                contents.append(gtypes.Content(role="model", parts=fn_parts))

                response_parts = []
                for fn_part in fn_parts:
                    fc = fn_part.function_call
                    tool_name = fc.name
                    tool_input = dict(fc.args) if getattr(fc, "args", None) else {}
                    result = await self._execute_tool(tool_name, tool_input)
                    response_parts.append(
                        gtypes.Part(
                            function_response=gtypes.FunctionResponse(
                                name=tool_name,
                                response={"result": result},
                            )
                        )
                    )

                contents.append(gtypes.Content(role="user", parts=response_parts))
                continue

            # No tool calls: return text
            try:
                text = response.text or ""
            except Exception:
                text = ""
            if not text:
                text = "\n".join(
                    [getattr(p, "text") for p in parts if getattr(p, "text", None)]
                )
            return {"reply": (text or "(Gemini returned empty response)").strip(), "backend": "gemini"}

        # If all iterations used up, return a best-effort
        return {"reply": "(Gemini tool loop hit iteration limit)", "backend": "gemini"}

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
            # ── Code evolution tools ──
            {
                "name": "read_file",
                "description": "Read a project source file. Path relative to project root, e.g. 'server/app.py' or 'frontend/app.js'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to project root"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "edit_file",
                "description": "Edit a project file by exact find/replace. The 'find' string must match exactly. Use read_file first to confirm current content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to project root"},
                        "find": {"type": "string", "description": "Exact string to find in the file"},
                        "replace": {"type": "string", "description": "Replacement string"},
                    },
                    "required": ["path", "find", "replace"],
                },
            },
            {
                "name": "list_files",
                "description": "List project files. Optionally filter by directory (e.g. 'server', 'frontend').",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Subdirectory to list (default: project root)", "default": ""},
                    },
                },
            },
            {
                "name": "git_snapshot",
                "description": "Create a git commit snapshot before making code changes. ALWAYS call this before edit_file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message describing why we're snapshotting", "default": "auto-snapshot before AI edit"},
                    },
                },
            },
            {
                "name": "save_memory",
                "description": "Save a persistent memory that survives across sessions. Use for: user preferences, project facts, learned lessons, trade reasoning, long-term notes. Namespace groups related memories (e.g. 'user_pref', 'project', 'trade_lesson').",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "Namespace like 'user_pref', 'project', 'trade_lesson'"},
                        "key": {"type": "string", "description": "Short unique key within namespace"},
                        "content": {"type": "string", "description": "The memory content as natural language"},
                    },
                    "required": ["namespace", "key", "content"],
                },
            },
            {
                "name": "recall_memory",
                "description": "Retrieve memories from a namespace, or one specific memory by key. Always check memories before answering questions about user preferences or past reasoning.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "Namespace to fetch"},
                        "key": {"type": "string", "description": "Optional: specific memory key"},
                    },
                    "required": ["namespace"],
                },
            },
            {
                "name": "search_memory",
                "description": "Substring search across ALL memory namespaces. Use to find relevant past context.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Substring to search for"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "schedule_task",
                "description": "Schedule a recurring task using natural language. Examples: 'every 15 minutes', 'every hour', 'daily 08:00', 'every 4 hours'. Available actions: 'daily_summary' (send equity/PnL report), 'agent_scan' (scan all symbols for signals), 'memory_note' (record periodic note).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short human-readable task name"},
                        "recurrence": {"type": "string", "description": "Natural language like 'every 15 minutes' or 'daily 08:00'"},
                        "action": {"type": "string", "enum": ["daily_summary", "agent_scan", "memory_note"], "description": "Which handler to run"},
                        "params": {"type": "object", "description": "Optional params for the handler", "default": {}},
                    },
                    "required": ["name", "recurrence", "action"],
                },
            },
            {
                "name": "list_schedules",
                "description": "List all scheduled tasks currently active.",
                "input_schema": {"type": "object", "properties": {}},
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
            elif tool_name == "read_file":
                return self._tool_read_file(tool_input["path"])
            elif tool_name == "edit_file":
                return self._tool_edit_file(tool_input["path"], tool_input["find"], tool_input["replace"])
            elif tool_name == "list_files":
                return self._tool_list_files(tool_input.get("directory", ""))
            elif tool_name == "git_snapshot":
                return self._tool_git_snapshot(tool_input.get("message", "auto-snapshot before AI edit"))
            elif tool_name == "save_memory":
                return self._tool_save_memory(
                    tool_input["namespace"], tool_input["key"], tool_input["content"]
                )
            elif tool_name == "recall_memory":
                return self._tool_recall_memory(
                    tool_input["namespace"], tool_input.get("key")
                )
            elif tool_name == "search_memory":
                return self._tool_search_memory(tool_input["query"])
            elif tool_name == "schedule_task":
                return self._tool_schedule_task(
                    tool_input["name"], tool_input["recurrence"],
                    tool_input["action"], tool_input.get("params", {})
                )
            elif tool_name == "list_schedules":
                return self._tool_list_schedules()
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
                    "24h_change_pct": round((float(d["last"]) - float(d.get("open24h", 0))) / float(d.get("open24h", 1)) * 100, 2) if float(d.get("open24h", 0)) > 0 else 0,
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
                "sharpe_ratio": result.get("sharpe"),
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

    # ── Code evolution tools ──

    # Whitelist of editable file patterns
    EDITABLE_PATTERNS = [
        "server/*.py",
        "frontend/app.js",
        "frontend/index.html",
        "frontend/style.css",
    ]

    def _is_editable(self, rel_path: str) -> bool:
        """Check if a file path is in the editable whitelist."""
        import fnmatch
        for pattern in self.EDITABLE_PATTERNS:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    def _safe_resolve(self, rel_path: str) -> Path | None:
        """Resolve a relative path safely within project root. Returns None if outside.
        Uses Path.relative_to() instead of string prefix to prevent prefix-collision bypass.
        """
        try:
            full = (PROJECT_ROOT / rel_path).resolve()
            full.relative_to(PROJECT_ROOT.resolve())  # raises ValueError if outside
            return full
        except (ValueError, Exception):
            return None

    def _tool_read_file(self, path: str) -> str:
        """Read a project file."""
        full_path = self._safe_resolve(path)
        if full_path is None:
            return json.dumps({"error": f"Path '{path}' is outside project root"})
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {path}"})
        if not full_path.is_file():
            return json.dumps({"error": f"Not a file: {path}"})
        try:
            content = full_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            # Truncate very large files
            if len(lines) > 500:
                content = "\n".join(lines[:500])
                return json.dumps({
                    "path": path,
                    "content": content,
                    "truncated": True,
                    "total_lines": len(lines),
                    "shown_lines": 500,
                })
            return json.dumps({
                "path": path,
                "content": content,
                "total_lines": len(lines),
            })
        except Exception as e:
            return json.dumps({"error": f"Read failed: {e}"})

    def _tool_edit_file(self, path: str, find: str, replace: str) -> str:
        """Edit a project file by find/replace."""
        if not self._is_editable(path):
            return json.dumps({"error": f"File '{path}' is not in the editable whitelist. Allowed: {self.EDITABLE_PATTERNS}"})
        full_path = self._safe_resolve(path)
        if full_path is None:
            return json.dumps({"error": f"Path '{path}' is outside project root"})
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            content = full_path.read_text(encoding="utf-8")
            if find not in content:
                # Show a snippet around where it might be to help AI correct
                return json.dumps({
                    "error": "FIND string not found in file. Use read_file to check exact content.",
                    "file_length": len(content),
                    "hint": "Make sure the find string matches exactly including whitespace and newlines.",
                })
            count = content.count(find)
            if count > 1:
                return json.dumps({
                    "error": f"FIND string appears {count} times. Make it more specific (include more context).",
                })
            new_content = content.replace(find, replace, 1)
            full_path.write_text(new_content, encoding="utf-8")
            print(f"[AI Chat] edit_file: {path} modified ({len(find)} chars -> {len(replace)} chars)")
            return json.dumps({
                "ok": True,
                "path": path,
                "chars_removed": len(find),
                "chars_added": len(replace),
                "message": f"File {path} updated. uvicorn --reload will auto-restart if it's a .py file.",
            })
        except Exception as e:
            return json.dumps({"error": f"Edit failed: {e}"})

    def _tool_list_files(self, directory: str = "") -> str:
        """List project files."""
        target = self._safe_resolve(directory) if directory else PROJECT_ROOT
        if target is None:
            return json.dumps({"error": f"Directory '{directory}' is outside project root"})
        if not target.exists():
            return json.dumps({"error": f"Directory not found: {directory}"})
        try:
            files = []
            for item in sorted(target.iterdir()):
                rel = str(item.relative_to(PROJECT_ROOT)).replace("\\", "/")
                # Skip hidden dirs, __pycache__, node_modules, .git
                if any(part.startswith(".") or part == "__pycache__" or part == "node_modules" for part in rel.split("/")):
                    continue
                if item.is_dir():
                    files.append({"name": rel + "/", "type": "dir"})
                elif item.is_file():
                    size = item.stat().st_size
                    files.append({"name": rel, "type": "file", "size": size})
            return json.dumps({"directory": directory or ".", "files": files})
        except Exception as e:
            return json.dumps({"error": f"List failed: {e}"})

    def _tool_git_snapshot(self, message: str = "auto-snapshot before AI edit") -> str:
        """Create a git commit as safety snapshot."""
        import subprocess
        try:
            # Stage all tracked changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                timeout=10,
            )
            # Check if there's anything to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not status.stdout.strip():
                return json.dumps({"ok": True, "message": "Nothing to commit — working tree clean"})
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", f"[AI snapshot] {message}"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                # Get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                commit_hash = hash_result.stdout.strip()
                print(f"[AI Chat] git_snapshot: {commit_hash} — {message}")
                return json.dumps({"ok": True, "commit": commit_hash, "message": message})
            else:
                return json.dumps({"error": f"Git commit failed: {result.stderr}"})
        except Exception as e:
            return json.dumps({"error": f"Git snapshot failed: {e}"})

    # ── Hermes-inspired memory + scheduler tools ──

    def _tool_save_memory(self, namespace: str, key: str, content: str) -> str:
        try:
            from .core import memory as mem
            entry = mem.save_memory(namespace, key, content)
            return json.dumps({"ok": True, "namespace": namespace, "key": key, "entry": entry}, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_recall_memory(self, namespace: str, key: str | None = None) -> str:
        try:
            from .core import memory as mem
            result = mem.get_memory(namespace, key)
            if result is None:
                return json.dumps({"ok": False, "reason": "not found"})
            return json.dumps({"ok": True, "namespace": namespace, "key": key, "data": result}, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_search_memory(self, query: str) -> str:
        try:
            from .core import memory as mem
            hits = mem.search_memory(query, limit=20)
            return json.dumps({"ok": True, "query": query, "count": len(hits), "results": hits}, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_schedule_task(self, name: str, recurrence: str, action: str, params: dict) -> str:
        try:
            from .core import scheduler as sched
            task = sched.create_task(name, recurrence, action, params or {})
            return json.dumps({"ok": True, "task": task.to_dict()}, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_list_schedules(self) -> str:
        try:
            from .core import scheduler as sched
            tasks = sched.list_tasks()
            return json.dumps({"ok": True, "count": len(tasks), "tasks": tasks}, default=str)
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

        # 1) Optional Claude Code CLI path (BRIDGE_MODE=true)
        if CLAUDE_CODE_CLI_MODE:
            try:
                reply_text, new_session_id = await self._chat_with_claude_code_cli(message, session)
                if new_session_id:
                    session.claude_code_session_id = new_session_id
                cli_billing = self._looks_like_billing_or_credit_error(reply_text)
                backend = "claude_code_cli"

                # If the CLI fails due to billing/credit issues, automatically fall back to Gemini.
                if cli_billing:
                    try:
                        gemini_out = await self._chat_with_gemini_tool_call(session, model_key)
                        reply_text = gemini_out.get("reply", reply_text)
                        backend = gemini_out.get("backend", "gemini")
                    except Exception as ge:
                        # Keep CLI text if Gemini isn't available.
                        backend = "claude_code_cli"
                        print(f"[Gemini fallback failed] {type(ge).__name__}: {repr(ge)}")

                session.add_assistant(reply_text)
                self._save_memory(session)
                return {
                    "reply": reply_text,
                    "model": model_key,
                    "tool_calls": [],
                    "backend": backend,
                }
            except Exception as e:
                # Fallback to Anthropic SDK tool-calling if CLI fails.
                err_msg = (
                    "[Claude Code CLI failed, fallback to Anthropic SDK] "
                    f"{type(e).__name__}: {repr(e)}"
                )
                print(err_msg)
                # Optional: try Gemini before Anthropic (Anthropic may be billing-blocked too).
                try:
                    gemini_out = await self._chat_with_gemini_tool_call(session, model_key)
                    reply_text = gemini_out.get("reply", "Gemini returned empty")
                    session.add_assistant(reply_text)
                    self._save_memory(session)
                    return {
                        "reply": reply_text,
                        "model": model_key,
                        "tool_calls": [],
                        "backend": gemini_out.get("backend", "gemini"),
                    }
                except Exception:
                    pass
                # Do NOT add assistant message here; fallback path will do it.

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
