"""
Self-Healer — AI-powered code self-repair for the running web app.

How it works:
1. Intercepts Python log output to catch Exceptions / HTTP 500s
2. When an error is detected, sends the error + relevant source code to Claude
3. Claude proposes a patch (unified diff or file replacement)
4. The healer applies the patch to the source file
5. uvicorn --reload auto-restarts the server with the fix
6. Health check verifies the fix worked; git revert if not

Safety rails:
- Always git commit before patching (so we can revert)
- Max 3 auto-fix attempts per error signature
- Never touch .env, secrets, or outside project root
- Rate limit: at least 60s between fixes
"""

import asyncio
import logging
import os
import re
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEAL_LOG = PROJECT_ROOT / "autoresearch" / "heal_log.tsv"
MAX_ATTEMPTS_PER_ERROR = 3
MIN_SECONDS_BETWEEN_FIXES = 60
SERVER_BASE = "http://127.0.0.1:8001"

# Files that can be auto-patched (whitelist)
PATCHABLE_FILES = {
    "server/app.py",
    "server/backtest_service.py",
    "server/data_service.py",
    "server/pattern_service.py",
    "server/pattern_features.py",
    "server/agent_brain.py",
    "server/okx_trader.py",
    "server/ai_chat.py",
    "server/ma_ribbon_service.py",
}


class ErrorBuffer:
    """Collects log records that look like errors/tracebacks."""
    def __init__(self, max_size: int = 200):
        self.lines: list[str] = []
        self.max_size = max_size

    def push(self, line: str):
        self.lines.append(line)
        if len(self.lines) > self.max_size:
            self.lines.pop(0)

    def flush_recent(self, n: int = 40) -> str:
        return "\n".join(self.lines[-n:])


class SelfHealerHandler(logging.Handler):
    """Logging handler that feeds into the error buffer."""
    def __init__(self, buffer: ErrorBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record):
        msg = self.format(record)
        self.buffer.push(msg)


class SelfHealer:
    """
    Background service that watches for errors and auto-repairs code.
    """

    def __init__(self):
        self._running = False
        self._task = None
        self._error_buffer = ErrorBuffer()
        self._attempt_counts: dict[str, int] = defaultdict(int)
        self._last_fix_time: float = 0
        self._fix_count: int = 0
        self._client = None
        self._setup_log_interception()
        self._init_anthropic()

    def _setup_log_interception(self):
        """Attach handler to root logger to capture all errors."""
        handler = SelfHealerHandler(self._error_buffer)
        handler.setLevel(logging.ERROR)
        logging.getLogger().addHandler(handler)

    def _init_anthropic(self):
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env", override=True)
        except ImportError:
            pass
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                pass

    # ── Error detection ────────────────────────────────────────────────

    def _extract_error_signature(self, text: str) -> str | None:
        """Extract a short signature from a traceback for deduplication."""
        # Look for: File "...", line N, in <func> / Error: message
        patterns = [
            r'File "([^"]+)", line (\d+)',
            r'([\w]+Error|[\w]+Exception): (.+)',
            r'HTTP 500',
        ]
        parts = []
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                parts.append(m.group(0)[:60])
        if parts:
            return " | ".join(parts)
        return None

    def _has_new_error(self) -> tuple[bool, str]:
        """Check if there's a new error in the buffer worth fixing."""
        recent = self._error_buffer.flush_recent(30)
        error_keywords = ["Traceback", "Error:", "Exception:", "HTTP 500", "500 Internal"]
        has_error = any(kw in recent for kw in error_keywords)
        return has_error, recent

    # ── Source code reading ────────────────────────────────────────────

    def _read_relevant_files(self, error_text: str) -> dict[str, str]:
        """Read source files mentioned in the error traceback."""
        files = {}
        # Extract file paths from traceback
        for m in re.finditer(r'File "([^"]+\.py)"', error_text):
            path = Path(m.group(1))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            # Only read files in our project
            try:
                rel = path.relative_to(PROJECT_ROOT)
                rel_str = str(rel).replace("\\", "/")
                if rel_str in PATCHABLE_FILES or any(rel_str.startswith(d + "/") for d in {"server", "frontend"}):
                    if path.exists() and len(files) < 3:
                        content = path.read_text(encoding="utf-8", errors="replace")
                        files[rel_str] = content[:8000]  # limit size
            except ValueError:
                pass
        return files

    # ── AI fix generation ──────────────────────────────────────────────

    async def _ask_claude_for_fix(self, error_text: str, source_files: dict) -> dict | None:
        """Ask Claude to analyze the error and propose a fix."""
        if not self._client:
            return None

        files_section = ""
        for fname, content in source_files.items():
            files_section += f"\n\n### {fname}\n```python\n{content}\n```"

        prompt = f"""You are an autonomous code repair agent for a Python FastAPI crypto trading web app.

A runtime error occurred. Analyze it and provide a minimal targeted fix.

## Error Log
```
{error_text[-3000:]}
```

## Relevant Source Files{files_section}

## Instructions
1. Identify the root cause (1-2 sentences)
2. Provide the fix as a JSON object with this exact format:
{{
  "root_cause": "brief explanation",
  "file": "server/relative/path.py",
  "find": "exact string to find in the file (unique, 3-10 lines)",
  "replace": "replacement string",
  "confidence": 0.0-1.0
}}

Rules:
- Only fix ONE file per response
- `find` must be an exact match of existing code
- `replace` must be valid Python
- If you cannot safely fix it, return {{"confidence": 0.0, "root_cause": "cannot fix safely"}}
- Respond ONLY with the JSON object, no other text"""

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON: try each '{' position (capped to prevent DoS)
            import json as _json
            attempts = 0
            for m in re.finditer(r'\{', text):
                attempts += 1
                if attempts > 20:
                    break
                try:
                    obj = _json.loads(text[m.start():])
                    return obj
                except _json.JSONDecodeError:
                    pass
                # Try finding the balanced closing brace
                depth, end = 0, -1
                for i, ch in enumerate(text[m.start():]):
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = m.start() + i + 1
                            break
                if end > 0:
                    try:
                        return _json.loads(text[m.start():end])
                    except _json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[Healer] Claude call failed: {e}")
        return None

    # ── Fix application ────────────────────────────────────────────────

    def _apply_fix(self, fix: dict) -> bool:
        """Apply the proposed fix to the source file."""
        file_rel = fix.get("file", "").replace("\\", "/")
        find_text = fix.get("find", "")
        replace_text = fix.get("replace", "")

        if not file_rel or not find_text or replace_text is None or file_rel not in PATCHABLE_FILES:
            print(f"[Healer] Fix rejected: file={file_rel} not in whitelist")
            return False

        target = PROJECT_ROOT / file_rel.replace("/", os.sep)
        if not target.exists():
            print(f"[Healer] File not found: {target}")
            return False

        content = target.read_text(encoding="utf-8")
        if find_text not in content:
            print(f"[Healer] find string not found in {file_rel}")
            return False

        new_content = content.replace(find_text, replace_text, 1)
        if new_content == content:
            print(f"[Healer] No change would be made")
            return False

        # Backup + write
        target.write_text(new_content, encoding="utf-8")
        print(f"[Healer] Applied fix to {file_rel}")
        return True

    def _git_commit(self, message: str) -> bool:
        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=PROJECT_ROOT, capture_output=True, timeout=15, check=True
            )
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=PROJECT_ROOT, capture_output=True, timeout=15
            )
            return result.returncode == 0
        except Exception:
            return False

    def _git_revert_uncommitted(self):
        """Discard uncommitted changes to restore the snapshot state."""
        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=PROJECT_ROOT, capture_output=True, timeout=15
            )
        except Exception:
            pass

    async def _health_check(self, retries: int = 5, delay: float = 3.0) -> bool:
        """Check if the server is healthy after applying a fix."""
        await asyncio.sleep(delay)  # wait for reload
        for _ in range(retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{SERVER_BASE}/api/health")
                    if resp.status_code == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2.0)
        return False

    def _log_heal(self, sig: str, root_cause: str, file: str, success: bool):
        """Log the heal attempt to TSV."""
        try:
            HEAL_LOG.parent.mkdir(parents=True, exist_ok=True)
            if not HEAL_LOG.exists():
                HEAL_LOG.write_text("time\tsig\troot_cause\tfile\tsuccess\n")
            with open(HEAL_LOG, "a", encoding="utf-8") as f:
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                _s = lambda s: s.replace("\t", " ").replace("\n", " ").replace("\r", "")
                f.write(f"{ts}\t{_s(sig[:60])}\t{_s(root_cause[:100])}\t{_s(file)}\t{success}\n")
        except Exception:
            pass

    # ── Main heal loop ─────────────────────────────────────────────────

    async def try_heal(self):
        """Check for errors and attempt to fix them."""
        now = time.time()
        if now - self._last_fix_time < MIN_SECONDS_BETWEEN_FIXES:
            return

        has_error, error_text = self._has_new_error()
        if not has_error:
            return

        sig = self._extract_error_signature(error_text)
        if not sig:
            return

        if self._attempt_counts[sig] >= MAX_ATTEMPTS_PER_ERROR:
            return  # gave up on this error

        self._attempt_counts[sig] += 1
        self._last_fix_time = now
        attempt = self._attempt_counts[sig]

        print(f"[Healer] Error detected (attempt {attempt}/{MAX_ATTEMPTS_PER_ERROR}): {sig[:80]}")

        # Read relevant source files
        source_files = self._read_relevant_files(error_text)

        # Ask Claude for a fix
        fix = await self._ask_claude_for_fix(error_text, source_files)
        if not fix or fix.get("confidence", 0) < 0.5:
            print(f"[Healer] Claude declined to fix (confidence too low or no fix)")
            self._log_heal(sig, fix.get("root_cause", "?") if fix else "no response", "", False)
            return

        root_cause = fix.get("root_cause", "?")
        file_fixed = fix.get("file", "?")
        print(f"[Healer] Claude fix: {root_cause} -> patching {file_fixed}")

        # Commit before patching (safety snapshot)
        snapshot_ok = self._git_commit(f"[auto-snapshot] before self-heal attempt {attempt}: {sig[:50]}")
        if not snapshot_ok:
            print("[Healer] Could not create safety snapshot, aborting")
            self._error_buffer.lines.clear()
            return

        # Apply the fix
        applied = self._apply_fix(fix)
        if not applied:
            self._log_heal(sig, root_cause, file_fixed, False)
            self._error_buffer.lines.clear()
            return

        # Wait for uvicorn reload + health check
        healthy = await self._health_check()
        if healthy:
            self._fix_count += 1
            self._git_commit(f"[self-heal #{self._fix_count}] {root_cause[:80]}")
            print(f"[Healer] Fix verified! Total fixes: {self._fix_count}")
            self._log_heal(sig, root_cause, file_fixed, True)
        else:
            print(f"[Healer] Fix broke things! Reverting uncommitted changes...")
            self._git_revert_uncommitted()
            self._log_heal(sig, root_cause, file_fixed, False)
        # Clear error buffer after any heal attempt to avoid re-processing stale errors
        self._error_buffer.lines.clear()

    async def _loop(self):
        """Background loop: check for errors every 30s."""
        self._running = True
        print("[Healer] Started. Watching for errors...")
        while self._running:
            try:
                await self.try_heal()
            except Exception as e:
                print(f"[Healer] Loop error: {e}")
            await asyncio.sleep(30)

    def start(self):
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "fix_count": self._fix_count,
            "last_fix_time": self._last_fix_time,
            "error_signatures_seen": dict(self._attempt_counts),
            "recent_errors": self._error_buffer.flush_recent(10),
            "has_ai": self._client is not None,
        }
