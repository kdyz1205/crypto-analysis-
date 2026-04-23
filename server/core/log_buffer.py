"""
Centralized log capture — ring buffer for all print/logging output.

Installed once at import time. All modules that use print() or logging
will have their output captured here for the frontend /api/agent/logs endpoint.
"""

import time
import logging
import builtins
from collections import deque

_LOG_BUFFER: deque[dict] = deque(maxlen=200)


class _AgentLogHandler(logging.Handler):
    """Captures log records into the shared ring buffer."""
    def emit(self, record):
        try:
            _LOG_BUFFER.append({
                "ts": record.created,
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "msg": self.format(record),
            })
        except Exception:
            pass


# Install handler on root logger
_agent_handler = _AgentLogHandler()
_agent_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_agent_handler)
logging.getLogger().setLevel(logging.INFO)

# Monkey-patch builtins.print to capture print() calls from agent/trader modules
_original_print = builtins.print


def _capturing_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _LOG_BUFFER.append({
        "ts": time.time(),
        "time": time.strftime("%H:%M:%S"),
        "level": "INFO",
        "msg": msg,
    })
    # Bug A hardening 2026-04-22: Windows cp1252 stdout cannot encode
    # non-ASCII chars like `\u2192` (→) or `\u2014` (—), which used to
    # bubble UnicodeEncodeError up through _original_print() → FastAPI
    # → HTTP 500 on the caller. Any print() with a non-cp1252 char on a
    # real-money endpoint would kill the request. Catch here and retry
    # with replaced chars so the caller never sees the encode error.
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = tuple(
            str(a).encode("ascii", errors="replace").decode("ascii")
            for a in args
        )
        try:
            _original_print(*safe_args, **kwargs)
        except Exception:
            pass


builtins.print = _capturing_print
