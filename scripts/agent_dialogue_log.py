#!/usr/bin/env python3
"""Append one turn to the Claude <-> Codex dialogue log and mirror
to frontend/ so the pixel viewer can read it via /static/.

Usage (Claude calls this when invoking codex):
    python scripts/agent_dialogue_log.py claude TASK-1 "title" "<prompt body>"
    python scripts/agent_dialogue_log.py codex  TASK-1 "title" --from-file path/to/codex_output.txt

Or with --content:
    python scripts/agent_dialogue_log.py codex TASK-1 "running" --content "[RUNNING] ..."

Appends to:  data/logs/agent_dialogue.jsonl   (master)
Mirrors to:  frontend/agent_dialogue.jsonl    (served by /static/)
"""
import argparse, json, os, shutil, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "logs" / "agent_dialogue.jsonl"
MIRROR = ROOT / "frontend" / "agent_dialogue.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker", choices=["claude", "codex"])
    ap.add_argument("task_id")
    ap.add_argument("title")
    ap.add_argument("--content", default=None)
    ap.add_argument("--from-file", default=None)
    args = ap.parse_args()

    if args.content is None and args.from_file is None:
        # Read from stdin (reconfigure to utf-8 tolerant of any bytes)
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        content = sys.stdin.read()
    elif args.from_file:
        content = Path(args.from_file).read_text(encoding="utf-8", errors="replace")
    else:
        content = args.content

    # Strip any lone surrogates that would blow up json.dumps below.
    # Codex output sometimes contains orphan \udcXX from mixed encodings.
    content = content.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    MASTER.parent.mkdir(parents=True, exist_ok=True)

    # Count existing turns for turn number
    turn = 1
    if MASTER.exists():
        with open(MASTER, encoding="utf-8") as f:
            turn = sum(1 for _ in f) + 1

    entry = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "turn": turn,
        "speaker": args.speaker,
        "role": "planner" if args.speaker == "claude" else "executor",
        "task_id": args.task_id,
        "title": args.title,
        "content": content.strip(),
    }

    with open(MASTER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Mirror so /static/ serves it
    shutil.copyfile(MASTER, MIRROR)
    print(f"OK turn {turn} appended | speaker={args.speaker} task={args.task_id}")
    print(f"  master : {MASTER}")
    print(f"  mirror : {MIRROR}")


if __name__ == "__main__":
    main()
