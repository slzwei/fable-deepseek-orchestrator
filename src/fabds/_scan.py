#!/usr/bin/env python3
"""Out-of-process regex scanner.

Worker ``search`` actions carry a model-supplied regex. Python's ``re`` has no
evaluation timeout, so a pattern like ``(a+)+b`` against a long line backtracks
for effectively ever and no in-process time budget can interrupt it: the check
between lines never gets to run.

Running the scan in a child process makes the bound real. The parent kills the
whole process group on timeout, so the worst case for a hostile pattern is one
wasted subprocess, not a wedged orchestration run.

Protocol: a JSON job on stdin, a JSON result on stdout. The parent decides which
files are eligible; this script only matches.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    try:
        job = json.loads(sys.stdin.read())
    except ValueError as exc:
        json.dump({"error": f"bad job: {exc}"}, sys.stdout)
        return 2

    try:
        pattern = re.compile(job["pattern"])
    except re.error as exc:
        json.dump({"error": f"invalid regex: {exc}"}, sys.stdout)
        return 2

    root = Path(job["root"])
    max_results = int(job.get("max_results", 60))
    max_bytes = int(job.get("max_bytes", 4_000_000))
    max_line = int(job.get("max_line", 2000))

    matches: list[dict] = []
    scanned = 0
    files_scanned = 0

    for relative in job.get("files", []):
        if len(matches) >= max_results or scanned >= max_bytes:
            break
        try:
            raw = (root / relative).read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        scanned += len(raw)
        files_scanned += 1
        for lineno, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
            if len(line) > max_line:
                continue
            if pattern.search(line):
                matches.append({"path": relative, "line": lineno, "text": line.strip()[:200]})
                if len(matches) >= max_results:
                    break

    json.dump({"matches": matches, "files_scanned": files_scanned,
               "truncated": len(matches) >= max_results}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
