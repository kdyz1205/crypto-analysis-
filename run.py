import os
import platform
import socket
import subprocess
import sys
import threading
import time
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _candidate_python_exes() -> list[str]:
    candidates = [
        os.environ.get("TRADING_OS_PYTHON", ""),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python312", "python.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312\python.exe"),
    ]
    seen = set()
    for path in candidates:
        if not path:
            continue
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen or norm == os.path.normcase(os.path.abspath(sys.executable)):
            continue
        seen.add(norm)
        if os.path.exists(path):
            yield path


try:
    import uvicorn
except ModuleNotFoundError as exc:
    if exc.name != "uvicorn":
        raise
    for python_exe in _candidate_python_exes():
        probe = subprocess.run(
            [python_exe, "-c", "import uvicorn, fastapi"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            print(f"[launcher] current Python lacks uvicorn: {sys.executable}")
            print(f"[launcher] restarting with: {python_exe}")
            os.execv(python_exe, [python_exe, os.path.abspath(__file__), *sys.argv[1:]])
    raise

# Load .env file if it exists (for ANTHROPIC_API_KEY, OKX keys, etc.)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=True)
        print(f"[env] Loaded {_env_path}")
    else:
        load_dotenv()
except ImportError:
    pass

# Default 8001 to avoid conflict with anything left on 8000 (TIME_WAIT or other app)
PORT = int(os.environ.get("PORT", 8001))
# Open v2 (Execution Center) by default; override with LAUNCH_PATH=/ for legacy UI
_LAUNCH_PATH = os.environ.get("LAUNCH_PATH", "/v2").strip() or "/v2"
# Git Bash / MSYS auto-converts POSIX paths like "/v2" into Windows absolute
# paths like "C:/Program Files/Git/v2" before env vars reach Python. Detect
# that + strip. Keeps the browser from landing on the 404 page 2026-04-20.
import re as _re
_msys = _re.match(r"^[A-Za-z]:[/\\].*?[/\\]([^/\\]+)$", _LAUNCH_PATH)
if _msys:
    _LAUNCH_PATH = "/" + _msys.group(1)
    print(f"[env] LAUNCH_PATH was MSYS-mangled ({os.environ.get('LAUNCH_PATH')!r}); recovered to {_LAUNCH_PATH!r}")
if not _LAUNCH_PATH.startswith("/"):
    _LAUNCH_PATH = "/" + _LAUNCH_PATH


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
            launch = _LAUNCH_PATH
            url_holder = [f"http://127.0.0.1:{port}{launch}"]
            threading.Thread(target=open_browser, args=(url_holder,), daemon=True).start()
            print(
                f"\n  Crypto TA (v2): {url_holder[0]}\n"
                f"  Legacy UI: http://127.0.0.1:{port}/\n"
                f"  Open in browser if not auto-opened.\n"
                f"\n  >>> Use HTTP only (not HTTPS). If Chrome says \"invalid response\",\n"
                f"      you opened https:// - copy the line above exactly (starts with http://).\n"
            )
            reload_enabled = os.environ.get("TRADING_OS_RELOAD", "").lower() in {"1", "true", "yes", "on"}
            uvicorn_kwargs = {"host": host, "port": port, "reload": reload_enabled}
            if reload_enabled:
                uvicorn_kwargs["reload_dirs"] = [
                    os.path.join(PROJECT_ROOT, "server"),
                    os.path.join(PROJECT_ROOT, "frontend"),
                ]
            uvicorn.run("server.app:app", **uvicorn_kwargs)
            break
        if port < PORT + 4:
            print(f"\n  Port {port} in use, trying {port + 1} ...\n")
            port += 1
        else:
            print(f"\n  Ports {PORT}..{port} in use. Close the other app or run:  set PORT=8010  then  python run.py\n")
            raise SystemExit(1)
