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
    """Check evolution leaderboard file and return top 5."""
    lb_path = os.path.join(PROJECT_ROOT, "data", "strategy_leaderboard.json")
    try:
        if os.path.exists(lb_path):
            data = json.loads(open(lb_path, encoding="utf-8").read())
            if isinstance(data, list) and len(data) > 0:
                lines = ["<b>📊 进化排行榜</b>", ""]
                for i, v in enumerate(data[:5]):
                    emoji = ["🥇","🥈","🥉","4️⃣","5️⃣"][i] if i < 5 else ""
                    ret = v.get('total_return_pct', 0)
                    lines.append(f"{emoji} {v.get('symbol','')} {v.get('timeframe','')} | {v.get('name','')}")
                    lines.append(f"   回报:{ret:+.1f}% WR:{v.get('win_rate',0)}% Sharpe:{v.get('sharpe_ratio',0)}")
                return "\n".join(lines)
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

    await tg_send(
        "🤖 <b>Trading OS Bot 启动</b>\n\n"
        "📝 <b>自然语言</b> — 发任何消息，Claude 回复\n\n"
        "⚡ <b>策略命令</b>\n"
        "/create [策略] [币种] [周期] [资金] — 创建策略\n"
        "/策略库 — 查看所有可用策略\n"
        "/运行中 — 查看运行状态\n"
        "/stop [币种|all] — 停止策略\n"
        "/leaderboard — 进化排行榜\n"
        "/status — 系统状态\n\n"
        "💡 例: /create sr_full BTCUSDT 4h 10000\n"
        "💡 例: /create 做市 HYPEUSDT 1m 100 live"
    )

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

                # Strategy commands (natural language → deploy)
                if text.startswith("/create") or text.startswith("/deploy") or text.startswith("/策略"):
                    await handle_strategy_command(text)
                    continue

                if text.strip() in ("/strategies", "/catalog", "/策略库"):
                    await show_catalog()
                    continue

                if text.strip() in ("/running", "/运行中"):
                    await show_running_strategies()
                    continue

                if text.strip().startswith("/stop"):
                    await handle_stop_command(text)
                    continue

                # ── Pattern Engine Commands ─────────────────────────
                if text.strip() in ("/health", "/digest", "/日报"):
                    await show_health_digest()
                    continue

                if text.strip() in ("/rules", "/规则"):
                    await show_rule_effectiveness()
                    continue

                if text.strip() in ("/live", "/实盘"):
                    await show_live_instances()
                    continue

                if text.strip() in ("/outcomes", "/结果"):
                    await show_live_outcomes()
                    continue

                if text.strip() in ("/dbs", "/db", "/数据库"):
                    await show_pattern_databases()
                    continue

                if text.strip().startswith("/batch"):
                    await trigger_batch_build(text)
                    continue

                if text.strip() in ("/help", "/帮助"):
                    await tg_send(
                        "<b>🎯 Trading OS Commands</b>\n\n"
                        "<b>研究与闭环</b>\n"
                        "/health - 闭环健康日报（24h 摘要）\n"
                        "/rules - 规则实盘表现\n"
                        "/live - 运行中的实盘实例\n"
                        "/outcomes - 最近平仓结果\n"
                        "/dbs - Pattern 数据库列表\n"
                        "/batch - 触发批量建库（全币种最长历史）\n\n"
                        "<b>策略操作</b>\n"
                        "/create [策略] [币] [周期] [资金]\n"
                        "/策略库 - 查看模板\n"
                        "/运行中 - 查看活跃\n"
                        "/stop [币|all] - 停止\n"
                        "/leaderboard - 排行榜\n\n"
                        "<b>系统</b>\n"
                        "/status - 系统状态\n\n"
                        "💬 也可以直接用自然语言问问题 → Claude 会回答"
                    )
                    continue

                # Natural language → Claude processes with full codebase context
                response = await handle_message(text)
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


# ── Strategy Commands ────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000"


async def api_call(method, path, body=None):
    """Call the local web server API."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            if method == "GET":
                r = await c.get(f"{API_BASE}{path}")
            else:
                r = await c.post(f"{API_BASE}{path}", json=body) if body else await c.post(f"{API_BASE}{path}")
            return r.json()
    except Exception as e:
        return {"error": str(e)}


async def handle_strategy_command(text: str):
    """Parse natural language strategy request → create and deploy.

    Examples:
    /create SR反转 BTCUSDT 4h 10000
    /deploy 均线开花 ETHUSDT 1h 5000 live
    /策略 做市 HYPEUSDT 1m 100 实盘
    """
    parts = text.split()
    if len(parts) < 3:
        await tg_send(
            "<b>创建策略</b>\n\n"
            "格式: /create [策略名] [币种] [周期] [资金] [模式]\n\n"
            "策略名:\n"
            "  sr_reversal — S/R 反转\n"
            "  sr_full — S/R 全模式\n"
            "  sr_retest — 突破回测\n"
            "  ma_ribbon — 均线开花\n"
            "  hft_imbalance — 盘口失衡\n"
            "  hft_mm — 做市\n"
            "  hft_sweep — 突破跟随\n\n"
            "例: /create sr_full BTCUSDT 4h 10000\n"
            "例: /create hft_mm HYPEUSDT 1m 100 live"
        )
        return

    # Parse
    template_id = parts[1] if len(parts) > 1 else "sr_full"
    symbol = parts[2].upper() if len(parts) > 2 else "BTCUSDT"
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    timeframe = parts[3] if len(parts) > 3 else "4h"
    equity = float(parts[4]) if len(parts) > 4 else 10000
    mode = "live" if len(parts) > 5 and parts[5] in ("live", "实盘") else "disabled"

    # Map Chinese names
    name_map = {
        "反转": "sr_reversal", "sr反转": "sr_reversal",
        "全模式": "sr_full", "sr全模式": "sr_full",
        "回测": "sr_retest", "突破回测": "sr_retest",
        "均线": "ma_ribbon", "均线开花": "ma_ribbon",
        "做市": "hft_mm", "hft做市": "hft_mm",
        "失衡": "hft_imbalance", "盘口失衡": "hft_imbalance",
        "突破": "hft_sweep", "突破跟随": "hft_sweep",
        "剥头皮": "hf_scalp",
    }
    template_id = name_map.get(template_id.lower(), template_id)

    await tg_send(f"🔧 创建策略...\n{template_id} | {symbol} {timeframe} | ${equity} | {'实盘' if mode == 'live' else '模拟'}")

    # Call API
    url = f"/api/runtime/catalog/{template_id}/launch?symbol={symbol}&timeframe={timeframe}&live_mode={mode}&starting_equity={equity}"
    result = await api_call("POST", url)

    if result.get("error"):
        await tg_send(f"❌ 创建失败: {result['error']}")
    elif result.get("template"):
        await tg_send(
            f"✅ <b>策略已创建并启动</b>\n\n"
            f"策略: {result['template']}\n"
            f"币种: {symbol}\n"
            f"周期: {timeframe}\n"
            f"资金: ${equity}\n"
            f"模式: {'🔴 实盘' if mode == 'live' else '📋 模拟'}\n\n"
            f"在网页面板查看运行状态"
        )
    else:
        await tg_send(f"⚠️ 结果: {json.dumps(result)[:200]}")


async def show_running_strategies():
    """Show currently running strategies."""
    result = await api_call("GET", "/api/runtime/instances")
    if result.get("error"):
        await tg_send(f"❌ 无法连接服务器: {result['error']}")
        return

    instances = result.get("instances", [])
    running = [i for i in instances if i["status"]["runtime_state"] == "running"]
    live = [i for i in instances if i["config"].get("live_mode") == "live"]

    lines = [f"<b>📊 策略状态</b>", f"总数: {len(instances)} | 运行: {len(running)} | 实盘: {len(live)}", ""]

    for i in running[:10]:
        c = i["config"]
        s = i["status"]
        pnl = s.get("paper_state", {}).get("account", {}).get("realized_pnl", 0) or 0
        is_live = "🔴" if c.get("live_mode") == "live" else "📋"
        bar = s.get("last_processed_bar", "-")
        lines.append(f"{is_live} {c['symbol']} {c['timeframe']} | bar={bar} | pnl={pnl:+.1f}")

    if len(running) > 10:
        lines.append(f"... 还有 {len(running) - 10} 个")

    await tg_send("\n".join(lines))


async def show_catalog():
    """Show available strategy templates."""
    result = await api_call("GET", "/api/runtime/catalog")
    if result.get("error"):
        await tg_send(f"❌ {result['error']}")
        return

    templates = result.get("templates", [])
    lines = ["<b>📚 策略库</b>", ""]
    for t in templates:
        risk = {"low": "🟢低", "medium": "🟡中", "high": "🔴高"}.get(t["risk_level"], "")
        tfs = ", ".join(t.get("supported_timeframes", []))
        lines.append(f"<b>{t['name']}</b> {risk}")
        lines.append(f"  ID: {t['template_id']}")
        lines.append(f"  周期: {tfs}")
        lines.append(f"  {t['description'][:60]}...")
        lines.append("")

    lines.append("使用 /create [ID] [币种] [周期] [资金] 创建")
    await tg_send("\n".join(lines))


async def handle_stop_command(text: str):
    """Stop a strategy. /stop BTCUSDT or /stop all"""
    parts = text.split()
    target = parts[1].upper() if len(parts) > 1 else "all"

    result = await api_call("GET", "/api/runtime/instances")
    if result.get("error"):
        await tg_send(f"❌ {result['error']}")
        return

    stopped = 0
    for i in result.get("instances", []):
        c = i["config"]
        s = i["status"]
        if s["runtime_state"] != "running":
            continue
        if target != "ALL" and target not in c["symbol"]:
            continue
        await api_call("POST", f"/api/runtime/instances/{c['instance_id']}/stop")
        stopped += 1

    await tg_send(f"⏹ 已停止 {stopped} 个策略")


# ── Pattern Engine / Closed Loop Commands ────────────────────────────────

async def show_health_digest():
    """Show the closed-loop health digest."""
    result = await api_call("GET", "/api/tools/health-digest?hours=24")
    if result.get("error"):
        await tg_send(f"❌ {result['error']}")
        return

    d = result.get("data", {})
    p = d.get("research_volume", {})
    t = d.get("trade_volume", {})
    rp = d.get("rule_performance", {})
    st = d.get("agent_state", {})

    lines = [
        "<b>🩺 闭环健康日报</b>",
        "",
        f"<i>{d.get('summary_line','-')}</i>",
        "",
        f"<b>📊 研究量</b>",
        f"  总 patterns: {p.get('total_patterns',0):,}",
        f"  数据库: {p.get('pattern_dbs',0)}",
        f"  24h 新增 live: {p.get('new_live_scans_24h',0)}",
        "",
        f"<b>💰 24h 实盘</b>",
        f"  平仓: {t.get('closed_last_24h',0)}",
        f"  胜率: {int(t.get('win_rate',0)*100)}%",
        f"  平均 EV: {t.get('avg_return_atr',0):+.2f} ATR",
        "",
        f"<b>🔧 引擎</b>",
        f"  状态: {st.get('worker_status','?')}",
        f"  Gen: {st.get('current_generation',0)}",
        f"  累计策略: {st.get('total_strategies_generated',0)}",
    ]

    top = rp.get('top_performers', [])
    if top:
        lines += ["", "<b>⭐ 最佳规则</b>"]
        for r in top[:3]:
            lines.append(f"  {r['rule_id']}: {r['lifetime_ev']:+.2f} ATR ({r['lifetime_count']} 次)")

    drift = rp.get('degrading', [])
    if drift:
        lines += ["", "<b>⚠ 漂移规则</b>"]
        for r in drift[:3]:
            lines.append(f"  {r['rule_id']}: 漂移 {r['drift']:+.2f} ATR")

    alerts = d.get('alerts', [])
    if alerts:
        lines += ["", "<b>🚨 需关注</b>"]
        for a in alerts:
            icon = "⚠" if a.get('severity') == 'warning' else "ℹ"
            lines.append(f"  {icon} {a.get('message','')}")

    await tg_send("\n".join(lines))


async def show_rule_effectiveness():
    """Show live effectiveness of all decision rules."""
    result = await api_call("GET", "/api/tools/patterns/rule-effectiveness")
    if result.get("error"):
        await tg_send(f"❌ {result['error']}")
        return

    rules = result.get("data", [])
    if not rules:
        await tg_send("暂无规则统计 — 需要实盘实例停止后才有数据")
        return

    lines = ["<b>📋 规则实盘表现</b>", ""]
    for r in rules:
        status = r.get('status', 'active')
        icon = {'active':'✅','warning':'⚠','suppressed':'🔇','retired':'🔴','needs_recalibration':'🔧'}.get(status, '•')
        ev = r.get('live_expected_value', 0)
        ev_color = '🟢' if ev > 0.5 else '🔴' if ev < 0 else '🟡'
        lines.append(f"{icon} <b>{r['rule_id']}</b>")
        lines.append(f"   {status} | {r['live_count']} 次 | WR {int(r['live_win_rate']*100)}% | {ev_color} EV {ev:+.2f}")
        if r.get('status_reason'):
            lines.append(f"   <i>{r['status_reason']}</i>")

    await tg_send("\n".join(lines))


async def show_live_instances():
    """Show currently running live strategy instances."""
    result = await api_call("GET", "/api/tools/live-instances")
    if result.get("error"):
        await tg_send(f"❌ {result['error']}")
        return

    instances = result.get("data", [])
    if not instances:
        await tg_send("暂无实盘实例")
        return

    lines = [f"<b>🚀 实盘实例 ({len(instances)})</b>", ""]
    for i in instances[:10]:
        status = i.get('running_status', '?')
        icon = {'running':'▶️','paused':'⏸','stopped':'⏹','retired':'📦','error':'❌'}.get(status, '•')
        pnl = i.get('current_pnl', 0)
        pnl_icon = '🟢' if pnl > 0 else '🔴' if pnl < 0 else '⚪'
        lines.append(f"{icon} <b>{i.get('symbol','?')} {i.get('timeframe','')}</b>")
        lines.append(f"   策略: {i.get('pattern_decision','-')} | 规则: {i.get('decision_rule','-')}")
        lines.append(f"   资金: ${i.get('allocated_capital',0):.0f} | {pnl_icon} {pnl:+.2f}")
        if i.get('pattern_ev_expected'):
            lines.append(f"   预期 EV: {i['pattern_ev_expected']:+.2f} ATR")
        if i.get('outcome_written_back'):
            lines.append(f"   ✓ 已回写 ({i.get('outcome_class','?')})")

    await tg_send("\n".join(lines))


async def show_live_outcomes():
    """Show most recent live outcomes (from writeback pipeline)."""
    result = await api_call("GET", "/api/tools/patterns/live-outcomes?limit=10")
    if result.get("error"):
        await tg_send(f"❌ {result['error']}")
        return

    outcomes = result.get("data", [])
    if not outcomes:
        await tg_send("暂无实盘结果记录")
        return

    lines = [f"<b>📈 最近 {len(outcomes)} 笔平仓</b>", ""]
    for o in outcomes:
        ret = o.get('realized_return_atr', 0)
        icon = '🟢' if ret > 0 else '🔴'
        confirmed = '✓' if o.get('pattern_prediction_confirmed') else '✗'
        lines.append(f"{icon} <b>{o.get('symbol','?')} {o.get('timeframe','')}</b> {ret:+.2f} ATR")
        lines.append(f"   {o.get('outcome_class','-')} | 预测: {confirmed} | 规则: {o.get('origin_decision_rule','-')}")
        lines.append(f"   {o.get('bars_held',0)} bars")

    await tg_send("\n".join(lines))


async def show_pattern_databases():
    """List pattern databases and their size."""
    result = await api_call("GET", "/api/tools/patterns/database")
    progress_result = await api_call("GET", "/api/tools/patterns/batch-progress")

    dbs = result.get("data", [])
    prog = progress_result.get("data", {})

    lines = [f"<b>🗄 Pattern 数据库 ({len(dbs)})</b>", ""]

    total = sum(d.get('pattern_count', 0) for d in dbs)
    lines.append(f"总样本: <b>{total:,}</b>")
    lines.append("")

    if prog and prog.get('status') == 'running':
        lines.append(f"🔄 建库中: {prog.get('completed_jobs',0)}/{prog.get('total_jobs',0)}")
        lines.append(f"当前: {prog.get('current_job','?')}")
        lines.append("")

    # Group by symbol
    by_sym = {}
    for d in dbs:
        sym_tf = d.get('symbol_timeframe', '')
        parts = sym_tf.split('_', 1)
        sym = parts[0] if parts else '?'
        tf = parts[1] if len(parts) > 1 else '?'
        by_sym.setdefault(sym, []).append((tf, d.get('pattern_count', 0)))

    for sym in sorted(by_sym.keys())[:15]:
        tfs = sorted(by_sym[sym])
        tf_str = ' · '.join(f"{tf}:{n:,}" for tf, n in tfs)
        lines.append(f"<b>{sym}</b> — {tf_str}")

    if len(by_sym) > 15:
        lines.append(f"...还有 {len(by_sym) - 15} 个币种")

    await tg_send("\n".join(lines))


async def trigger_batch_build(text: str):
    """Trigger a batch build with max history per symbol."""
    await tg_send("🔄 启动批量建库...")
    result = await api_call("POST", "/api/tools/patterns/batch-build", body={
        "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","HYPEUSDT","DOGEUSDT","PEPEUSDT","ADAUSDT","TAOUSDT","SUIUSDT","LINKUSDT","AVAXUSDT","BNBUSDT","DOTUSDT","ATOMUSDT","NEARUSDT","UNIUSDT","AAVEUSDT","OPUSDT","ARBUSDT"],
        "timeframes": ["15m","1h","4h","1d"],
        "days": 2190,
    })
    if result.get("error"):
        await tg_send(f"❌ {result['error']}")
        return
    d = result.get("data", {})
    prog = d.get("progress", {})
    await tg_send(
        f"✅ 批量建库已启动\n"
        f"总 jobs: {prog.get('total_jobs', 0)}\n"
        f"查看进度: /dbs"
    )


if __name__ == "__main__":
    asyncio.run(main())
