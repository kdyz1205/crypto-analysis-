"""Launcher: set EVOLVED_TRENDLINES=1 + start uvicorn in same process.

Avoids env-propagation headaches on Windows.
"""
import os
os.environ.setdefault("EVOLVED_TRENDLINES", "1")
os.environ.setdefault("EVOLVED_VARIANT", "v2_trader")
print(f"[run_server_evolved] env: "
      f"EVOLVED_TRENDLINES={os.environ['EVOLVED_TRENDLINES']} "
      f"EVOLVED_VARIANT={os.environ['EVOLVED_VARIANT']}")

import uvicorn
uvicorn.run("server.app:app", host="127.0.0.1", port=8000, log_level="info")
