#!/usr/bin/env python
"""PostToolUse hook: reject Python edits that do not parse.

Claude Code pipes the tool-call payload in as JSON on stdin. We pull the edited
path out of it, compile the file in-memory (no __pycache__ side effects), and
exit 2 with the error on stderr so Claude sees the failure and fixes it before
moving on. Any other outcome exits 0 and stays silent.
"""
import json
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # Never block on a malformed payload.

    path = (payload.get("tool_input") or {}).get("file_path", "")
    if not path.endswith(".py"):
        return 0

    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return 0  # File moved or deleted — not our problem.

    try:
        compile(source, path, "exec")
    except SyntaxError as exc:
        print(
            f"SyntaxError in {path}:{exc.lineno}: {exc.msg}\n"
            f"Fix the syntax error before continuing.",
            file=sys.stderr,
        )
        return 2  # Exit 2 = blocking error, surfaced to Claude.

    return 0


if __name__ == "__main__":
    sys.exit(main())
