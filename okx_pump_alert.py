"""
OKX 3分钟涨幅监控 — 任意合约涨幅 ≥3% 触发桌面通知
使用 OKX 公开 API（无需 API Key）

用法:
    python okx_pump_alert.py
    python okx_pump_alert.py --threshold 2.5  # 自定义涨幅阈值(%)
    python okx_pump_alert.py --interval 60    # 自定义检测间隔(秒), 默认30s
"""

import asyncio
import argparse
import time
import subprocess
import sys
from collections import deque
from datetime import datetime

import httpx

# ── 配置 ─────────────────────────────────────────────────────────────────────
OKX_TICKERS_URL = "https://www.okx.com/api/v5/market/tickers"
WINDOW_SECONDS = 180          # 3分钟窗口
DEFAULT_THRESHOLD = 3.0       # 涨幅阈值 %
DEFAULT_POLL_INTERVAL = 30    # 每30秒拉一次行情
INST_TYPE = "SWAP"            # SPOT | SWAP | FUTURES — SWAP=永续合约覆盖最多品种
COOLDOWN_SECONDS = 300        # 同一品种5分钟内不重复推送


def notify(symbol: str, pct: float, price: float):
    """Windows 桌面弹窗通知（PowerShell toast）+ 控制台输出。"""
    ts = datetime.now().strftime("%H:%M:%S")
    msg = f"{symbol}  +{pct:.2f}%  现价: {price:.4f}"
    print(f"\n🚀 [{ts}] PUMP ALERT  {msg}\n")

    # PowerShell BurntToast-style toast（无需第三方库）
    ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.BalloonTipTitle = "OKX Pump Alert 🚀"
$n.BalloonTipText = "{msg}"
$n.Visible = $True
$n.ShowBalloonTip(8000)
Start-Sleep -Seconds 2
$n.Dispose()
""".strip()
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        print(f"  [notify] PowerShell 通知失败: {e}")

    # 蜂鸣音提示
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1200, 200)
            time.sleep(0.1)
    except Exception:
        pass


async def fetch_tickers(client: httpx.AsyncClient, inst_type: str) -> dict[str, float]:
    """返回 {symbol: last_price} 字典。"""
    resp = await client.get(OKX_TICKERS_URL, params={"instType": inst_type}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error: {data.get('msg')}")
    return {
        item["instId"]: float(item["last"])
        for item in data["data"]
        if item.get("last") and float(item["last"]) > 0
    }


async def monitor(threshold: float, poll_interval: int, inst_type: str):
    """主监控循环。"""
    # price_history[symbol] = deque of (timestamp, price)
    price_history: dict[str, deque] = {}
    last_alerted: dict[str, float] = {}   # symbol -> unix ts of last alert

    print(f"OKX Pump Monitor 启动")
    print(f"  品种类型  : {inst_type}")
    print(f"  检测窗口  : {WINDOW_SECONDS}s")
    print(f"  涨幅阈值  : +{threshold}%")
    print(f"  轮询间隔  : {poll_interval}s")
    print(f"  重复推送冷却: {COOLDOWN_SECONDS}s")
    print("─" * 50)

    async with httpx.AsyncClient() as client:
        while True:
            loop_start = time.monotonic()
            try:
                now = time.time()
                tickers = await fetch_tickers(client, inst_type)
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] 已获取 {len(tickers)} 个品种行情", end="\r")

                alerts_this_round = []

                for symbol, price in tickers.items():
                    if symbol not in price_history:
                        price_history[symbol] = deque()

                    history = price_history[symbol]

                    # 清除超过窗口期的旧数据
                    while history and (now - history[0][0]) > WINDOW_SECONDS:
                        history.popleft()

                    # 计算与窗口最早价格的涨幅
                    if history:
                        oldest_ts, oldest_price = history[0]
                        age = now - oldest_ts
                        if age >= WINDOW_SECONDS * 0.8 and oldest_price > 0:   # 至少收到80%窗口的数据才算
                            pct = (price - oldest_price) / oldest_price * 100
                            if pct >= threshold:
                                cooldown_ok = (
                                    symbol not in last_alerted
                                    or (now - last_alerted[symbol]) >= COOLDOWN_SECONDS
                                )
                                if cooldown_ok:
                                    alerts_this_round.append((symbol, pct, price))
                                    last_alerted[symbol] = now

                    history.append((now, price))

                for symbol, pct, price in sorted(alerts_this_round, key=lambda x: -x[1]):
                    notify(symbol, pct, price)

            except httpx.RequestError as e:
                print(f"\n[网络错误] {e} — {poll_interval}s 后重试")
            except Exception as e:
                print(f"\n[错误] {e}")

            # 精确等待到下次轮询
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, poll_interval - elapsed)
            await asyncio.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="OKX pump alert - notify when price rises X% in 3 min")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Rise threshold in %% (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help=f"Poll interval in seconds (default {DEFAULT_POLL_INTERVAL})")
    parser.add_argument("--type", default=INST_TYPE, dest="inst_type",
                        choices=["SPOT", "SWAP", "FUTURES"],
                        help="Instrument type (default SWAP)")
    args = parser.parse_args()

    try:
        asyncio.run(monitor(args.threshold, args.interval, args.inst_type))
    except KeyboardInterrupt:
        print("\n已停止监控。")


if __name__ == "__main__":
    main()
