import os
import platform
import socket
import subprocess
import threading
import time
import webbrowser
import uvicorn

# Default 8001 to avoid conflict with anything left on 8000 (TIME_WAIT or other app)
PORT = int(os.environ.get("PORT", 8001))
URL = f"http://127.0.0.1:{PORT}"


def open_browser(url_holder):
    """Open default browser after server has had time to bind. url_holder = [url] so port fallback works."""
    time.sleep(2.5)  # give uvicorn time to be ready so the first request does not hang (black/loading)
    try:
        webbrowser.open(url_holder[0])
    except Exception:
        pass


def kill_port(port: int) -> None:
    """Kill process on port (Windows: netstat+taskkill; Unix: lsof+SIGTERM)."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            killed_any = False
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5 and parts[-1].isdigit():
                        subprocess.run(
                            ["taskkill", "/F", "/PID", parts[-1]],
                            capture_output=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        killed_any = True
                        time.sleep(0.5)
            if killed_any:
                time.sleep(0.8)
        else:
            import signal
            result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
            pids = result.stdout.strip().split()
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
            if pids:
                time.sleep(0.5)
    except Exception:
        pass


def port_available(host: str, port: int) -> bool:
    """True if we can bind to (host, port). Use this before uvicorn so we never pass a busy port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


if __name__ == "__main__":
    host = "0.0.0.0"
    port = PORT
    for attempt in range(5):
        kill_port(port)
        if attempt > 0:
            time.sleep(1.0)
        if port_available(host, port):
            url_holder = [f"http://127.0.0.1:{port}"]
            threading.Thread(target=open_browser, args=(url_holder,), daemon=True).start()
            print(f"\n  Crypto TA: {url_holder[0]}\n  Open in browser if not auto-opened.\n")
            uvicorn.run("server.app:app", host=host, port=port, reload=False)
            break
        if port < PORT + 4:
            print(f"\n  Port {port} in use, trying {port + 1} ...\n")
            port += 1
        else:
            print(f"\n  Ports {PORT}..{port} in use. Close the other app or run:  set PORT=8010  then  python run.py\n")
            raise SystemExit(1)
