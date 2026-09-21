"""Structured, redacting logging.

Two sinks:

* a human stream on stderr in the ``[component] message`` shape the spec asks
  for, gated by ``--verbose`` / ``--quiet``;
* a JSONL event stream on disk for the run ledger.

Every record passes through the process redactor on the way out, so a secret
that survived the sanitizer still cannot reach a log file.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, TextIO

from .redaction import REDACTOR

__all__ = ["Level", "RunLogger", "NullLogger"]


class Level(IntEnum):
    QUIET = 0    # errors only
    NORMAL = 1   # milestones
    VERBOSE = 2  # per-step detail
    DEBUG = 3    # payload sizes, cache keys, argv shapes


_COLOURS = {
    "error": "\033[31m",
    "warn": "\033[33m",
    "ok": "\033[32m",
    "dim": "\033[2m",
}
_RESET = "\033[0m"


@dataclass
class RunLogger:
    """Emits to stderr and, when given, to a JSONL file."""

    level: Level = Level.NORMAL
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    jsonl_path: Path | None = None
    colour: bool | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _started: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        if self.colour is None:
            self.colour = bool(getattr(self.stream, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")
        if self.jsonl_path is not None:
            self.jsonl_path = Path(self.jsonl_path)
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    def event(self, component: str, message: str, *, level: Level = Level.NORMAL,
              kind: str = "info", **fields: Any) -> None:
        safe_message = REDACTOR.scrub(str(message))
        safe_fields = _scrub_fields(fields)
        record = {
            "ts": time.time(),
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "component": component,
            "kind": kind,
            "message": safe_message,
            **safe_fields,
        }
        with self._lock:
            if self.jsonl_path is not None:
                try:
                    with self.jsonl_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, default=str) + "\n")
                except OSError:
                    pass
            if level <= self.level or kind == "error":
                self.stream.write(self._format(component, safe_message, kind))
                self.stream.flush()

    def info(self, component: str, message: str, **fields: Any) -> None:
        self.event(component, message, level=Level.NORMAL, **fields)

    def detail(self, component: str, message: str, **fields: Any) -> None:
        self.event(component, message, level=Level.VERBOSE, **fields)

    def debug(self, component: str, message: str, **fields: Any) -> None:
        self.event(component, message, level=Level.DEBUG, kind="debug", **fields)

    def ok(self, component: str, message: str, **fields: Any) -> None:
        self.event(component, message, level=Level.NORMAL, kind="ok", **fields)

    def warn(self, component: str, message: str, **fields: Any) -> None:
        self.event(component, message, level=Level.NORMAL, kind="warn", **fields)

    def error(self, component: str, message: str, **fields: Any) -> None:
        self.event(component, message, level=Level.QUIET, kind="error", **fields)

    # -- formatting ---------------------------------------------------------

    def _format(self, component: str, message: str, kind: str) -> str:
        tag = f"[{component}]"
        if self.colour and kind in _COLOURS:
            tag = f"{_COLOURS[kind]}{tag}{_RESET}"
        return f"{tag} {message}\n"


class NullLogger(RunLogger):
    """Drops everything. Used by tests and by the library API."""

    def __init__(self) -> None:
        super().__init__(level=Level.QUIET, stream=open(os.devnull, "w", encoding="utf-8"))

    def event(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        return


def _scrub_fields(fields: dict) -> dict:
    out = {}
    for key, value in fields.items():
        if isinstance(value, str):
            out[key] = REDACTOR.scrub(value)
        elif isinstance(value, dict):
            out[key] = _scrub_fields(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [REDACTOR.scrub(v) if isinstance(v, str) else v for v in value]
        else:
            out[key] = value
    return out
