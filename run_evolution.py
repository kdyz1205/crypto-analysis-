"""Strategy Evolution Engine — runs as independent process.

- Backtests strategy variants in background
- Sends leaderboard updates to Telegram
- Writes results to data/strategy_leaderboard.json
- Web server reads this file to show leaderboard
- Does NOT interfere with web server API calls

Usage:
    python run_evolution.py
"""

import asyncio
import os
import sys

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
    print("[env] Loaded .env")
except ImportError:
    pass

import httpx
from server.strategy.evolution import EvolutionEngine


# Telegram notification
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()


async def tg_send(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            )
    except Exception as e:
        print(f"[tg] send failed: {e}")


async def main():
    engine = EvolutionEngine()
    print(f"[evolution] PID {os.getpid()}")
    print(f"[evolution] Leaderboard: {len(engine.leaderboard)} entries")
    print(f"[evolution] TG: {'configured' if TG_TOKEN else 'not configured'}")
    print(f"[evolution] Press Ctrl+C to stop")
    print()

    await tg_send("🧬 <b>进化引擎启动</b>\n后台自动回测策略变体中...")

    engine.start()

    last_top_score = 0.0
    last_gen = 0

    try:
        while True:
            await asyncio.sleep(30)

            # Check for new generation results
            if engine.generation > last_gen:
                last_gen = engine.generation
                stats = engine.get_stats()
                lb = engine.get_leaderboard(5)

                # Print to console
                print(f"\n[gen {engine.generation}] tested={stats['total_tested']} profitable={stats['total_profitable']} lb={stats['leaderboard_size']}")
                for i, v in enumerate(lb):
                    print(f"  #{i+1} {v['symbol']} {v['timeframe']} | {v['name']} | ret={v['total_return_pct']}% WR={v['win_rate']}% score={v['score']}")

                # Send to Telegram every 5 generations
                if engine.generation % 5 == 0 and lb:
                    lines = [f"🧬 <b>进化引擎 Gen {engine.generation}</b>",
                             f"已测试: {stats['total_tested']} | 盈利: {stats['total_profitable']}",
                             ""]
                    for i, v in enumerate(lb[:5]):
                        emoji = "🥇🥈🥉" [i] if i < 3 else "  "
                        ret = v['total_return_pct']
                        sign = "+" if ret >= 0 else ""
                        lines.append(f"{emoji} {v['symbol']} {v['timeframe']} | {sign}{ret}% WR={v['win_rate']}%")
                    await tg_send("\n".join(lines))

                # Alert on new #1 with full factor details
                if lb and lb[0]['score'] > last_top_score + 0.01:
                    last_top_score = lb[0]['score']
                    v = lb[0]
                    triggers = ', '.join(v.get('trigger_modes', []))
                    factors = ', '.join(v.get('factors', [])) or '基础S/R'
                    params = v.get('params', {})
                    await tg_send(
                        f"🏆 <b>新冠军策略!</b>\n"
                        f"<b>{v['symbol']} {v['timeframe']}</b>\n"
                        f"触发: {triggers}\n"
                        f"因子: {factors}\n"
                        f"参数: RR={params.get('rr_target','?')} 触碰≥{params.get('min_touches','?')} 回看={params.get('lookback_bars','?')}\n"
                        f"\n📊 回报: {v['total_return_pct']:+.1f}%\n"
                        f"胜率: {v['win_rate']}% | Sharpe: {v['sharpe_ratio']}\n"
                        f"交易: {v['total_trades']}次 | 回撤: {v['max_drawdown_pct']:.1f}%\n"
                        f"\n👉 在网站 /api/runtime/leaderboard 复制"
                    )

    except KeyboardInterrupt:
        engine.stop()
        await tg_send("🛑 <b>进化引擎停止</b>")
        print("\n[evolution] Stopped")


if __name__ == "__main__":
    asyncio.run(main())
