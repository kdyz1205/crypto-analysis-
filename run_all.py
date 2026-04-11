"""Master runner — starts web server + evolution engine + TG bot bridge.

Usage: python run_all.py
Keeps everything running as long as the computer is on.
"""

import asyncio
import os
import subprocess
import sys
import time
import signal

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
    print("[env] Loaded .env")
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def start_web_server():
    """Start the web server as a subprocess."""
    return subprocess.Popen(
        [sys.executable, "run.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_evolution():
    """Start the evolution engine as a subprocess."""
    return subprocess.Popen(
        [sys.executable, "run_evolution.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main():
    print("=" * 60)
    print("  Trading OS — Master Runner")
    print("  Web Server + Evolution Engine + TG Bot")
    print("=" * 60)
    print()

    processes = {}

    # Start web server
    print("[master] Starting web server...")
    processes["web"] = start_web_server()
    time.sleep(3)

    # Start evolution engine
    print("[master] Starting evolution engine...")
    processes["evolution"] = start_evolution()

    print()
    print("[master] All systems running. Press Ctrl+C to stop.")
    print("[master] Web: http://127.0.0.1:8001/v2")
    print("[master] TG Bot: BRIDGE_MODE=true (Claude CLI)")
    print()

    try:
        while True:
            # Monitor and restart crashed processes
            for name, proc in list(processes.items()):
                if proc.poll() is not None:
                    print(f"[master] {name} crashed (exit={proc.returncode}), restarting...")
                    if name == "web":
                        processes["web"] = start_web_server()
                    elif name == "evolution":
                        time.sleep(5)  # wait before restart
                        processes["evolution"] = start_evolution()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[master] Shutting down...")
        for name, proc in processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        print("[master] All stopped.")


if __name__ == "__main__":
    main()
