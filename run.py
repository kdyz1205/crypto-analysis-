import signal
import subprocess
import uvicorn

PORT = 8000

def kill_port(port):
    result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
    pids = result.stdout.strip().split()
    for pid in pids:
        try:
            signal.os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass

if __name__ == "__main__":
    kill_port(PORT)
    uvicorn.run("server.app:app", host="0.0.0.0", port=PORT, reload=True)
