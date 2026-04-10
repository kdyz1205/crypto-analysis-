from __future__ import annotations

import json
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = (
    "scripts/verify_manual_drawing_phase.py",
    "scripts/verify_history_scale_phase.py",
    "scripts/verify_browser_final_phase.py",
    "scripts/verify_runtime_phase.py",
    "scripts/verify_live_bridge_phase.py",
)


def _run_script(path: str) -> dict:
    completed = subprocess.run(
        [sys.executable, path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parsed = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = {"stdout": stdout}
    return {
        "script": path,
        "returncode": completed.returncode,
        "stdout": parsed,
        "stderr_tail": stderr.splitlines()[-20:] if stderr else [],
    }


def main() -> int:
    results = [_run_script(path) for path in SCRIPTS]
    passed = all(result["returncode"] == 0 for result in results)
    print(json.dumps({"passed": passed, "results": results}, ensure_ascii=True, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
