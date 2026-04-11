"""Standalone Telegram Bot — runs WITHOUT the web server.

Polls for messages, uses Claude CLI (or Codex fallback) to respond.
Also receives evolution engine results and forwards to chat.

Usage: python run_tg_bot.py
Runs as long as computer is on. No web server needed.
"""

import asyncio
import json
import os
import subprocess
import sys
import time

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass

import httpx

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
CLAUDE_CMD = os.environ.get("CLAUDE_CODE_CMD",
    os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"))
CODEX_CMD = os.environ.get("CODEX_CMD",
    os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "codex.cmd"))
TIMEOUT = 300
POLL_INTERVAL = 2


async def tg_send(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[tg] no config, would send: {text[:100]}")
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Split long messages
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks:
                await client.post(
                    f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                    json={"chat_id": TG_CHAT_ID, "text": chunk, "parse_mode": "HTML"},
                )
    except Exception as e:
        print(f"[tg] send failed: {e}")


async def tg_get_updates(offset: int = 0) -> list:
    if not TG_TOKEN:
        return []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 20, "allowed_updates": '["message"]'},
            )
            data = resp.json()
            return data.get("result", [])
    except Exception as e:
        print(f"[tg] getUpdates failed: {e}")
        return []


def run_claude(message: str) -> str:
    """Run Claude CLI with message, return response."""
    args = [CLAUDE_CMD, "-p", "--output-format", "json", "--dangerously-skip-permissions"]
    try:
        result = subprocess.run(
            args, input=message.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=PROJECT_ROOT, timeout=TIMEOUT,
        )
        raw = result.stdout.decode("utf-8", errors="replace").strip()
        if raw:
            try:
                data = json.loads(raw)
                return data.get("result", raw)
            except json.JSONDecodeError:
                return raw
        err = result.stderr.decode("utf-8", errors="replace").strip()
        if "credit" in err.lower() or "billing" in err.lower():
            return "__BILLING__"
        return err or "(no response)"
    except subprocess.TimeoutExpired:
        return "(timeout — Claude took too long)"
    except FileNotFoundError:
        return "__NOT_FOUND__"
    except Exception as e:
        return f"(error: {e})"


def run_codex(message: str) -> str:
    """Fallback: run Codex CLI."""
    try:
        result = subprocess.run(
            [CODEX_CMD, "-q", "--full-auto"],
            input=message.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=PROJECT_ROOT, timeout=TIMEOUT,
        )
        return result.stdout.decode("utf-8", errors="replace").strip() or "(no response from codex)"
    except Exception as e:
        return f"(codex error: {e})"


async def handle_message(text: str) -> str:
    """Process a message: try Claude, fallback to Codex."""
    print(f"[bot] Processing: {text[:80]}...")

    # Run in thread to not block
    response = await asyncio.to_thread(run_claude, text)

    if response == "__BILLING__" or response == "__NOT_FOUND__":
        print("[bot] Claude unavailable, trying Codex...")
        response = await asyncio.to_thread(run_codex, text)

    return response


async def check_leaderboard():
    """Check evolution leaderboard file and return summary if updated."""
    lb_path = os.path.join(PROJECT_ROOT, "data", "strategy_leaderboard.json")
    try:
        if os.path.exists(lb_path):
            data = json.loads(open(lb_path, encoding="utf-8").read())
            if isinstance(data, list) and len(data) > 0:
                top = data[0]
                return f"📊 排行榜 #1: {top.get('symbol','')} {top.get('timeframe','')} | {top.get('name','')} | 回报:{top.get('total_return_pct',0):+.1f}% WR:{top.get('win_rate',0)}%"
    except Exception:
        pass
    return None


async def main():
    if not TG_TOKEN:
        print("[bot] ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        return
    if not TG_CHAT_ID:
        print("[bot] ERROR: TELEGRAM_CHAT_ID not set in .env")
        return

    print(f"[bot] Telegram Bot started (PID {os.getpid()})")
    print(f"[bot] Chat ID: {TG_CHAT_ID}")
    print(f"[bot] Claude: {CLAUDE_CMD}")
    print(f"[bot] Codex: {CODEX_CMD}")
    print(f"[bot] Polling for messages...")
    print()

    await tg_send("🤖 <b>Trading Bot 启动</b>\n发送任何消息，我用 Claude/Codex 回复。\n\n命令：\n/leaderboard — 查看进化排行榜\n/status — 系统状态")

    offset = 0
    last_lb_check = 0

    while True:
        try:
            updates = await tg_get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if chat_id != TG_CHAT_ID:
                    continue
                if not text:
                    continue

                print(f"[bot] Message: {text[:80]}")

                # Built-in commands
                if text.strip() == "/leaderboard":
                    lb = await check_leaderboard()
                    await tg_send(lb or "暂无排行榜数据。运行 python run_evolution.py 启动进化引擎。")
                    continue

                if text.strip() == "/status":
                    await tg_send(
                        f"🤖 <b>系统状态</b>\n"
                        f"Claude: {'✅' if os.path.exists(CLAUDE_CMD) else '❌'}\n"
                        f"Codex: {'✅' if os.path.exists(CODEX_CMD) else '❌'}\n"
                        f"PID: {os.getpid()}\n"
                        f"运行时间: {time.time():.0f}"
                    )
                    continue

                # Process with Claude/Codex
                response = await handle_message(text)
                # Truncate very long responses
                if len(response) > 4000:
                    response = response[:3900] + "\n\n...(truncated)"
                await tg_send(response)

            # Periodic leaderboard check (every 5 min)
            if time.time() - last_lb_check > 300:
                last_lb_check = time.time()
                # Don't send automatically — only on /leaderboard command

        except Exception as e:
            print(f"[bot] Error: {e}")
            await asyncio.sleep(5)

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
