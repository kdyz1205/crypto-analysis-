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
    _original_print(*args, **kwargs)


builtins.print = _capturing_print
