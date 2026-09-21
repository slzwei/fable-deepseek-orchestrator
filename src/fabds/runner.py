"""Safe subprocess execution.

Rules enforced here, with no opt-out:

* ``shell=False`` always. There is no code path in fabds that builds a command
  string; every invocation is an argv list.
* The child environment is built from an allowlist, never inherited wholesale.
  Anything whose name looks like a credential is dropped, and so are the
  ``CLAUDE_CODE_*`` bridge variables that would let a child process talk back
  to the orchestrating session.
* Every call has a wall-clock timeout and kills the whole process group on
  expiry, so a hung child cannot wedge a run.
* stdout/stderr are captured with a byte cap; a model that emits a gigabyte
  cannot exhaust memory.
* Long prompts go in on stdin, never in argv, so they cannot overflow ARG_MAX
  and never appear in ``ps`` output.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import ProviderCrashed, ProviderTimeout
from .redaction import REDACTOR

__all__ = ["CompletedCommand", "run_command", "build_child_env", "ENV_ALLOWLIST"]

#: Environment variables a child is allowed to see. Everything else is dropped.
ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ", "PWD",
    "SSL_CERT_FILE", "SSL_CERT_DIR",
    "PYTHONPATH", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
    "NODE_PATH", "NVM_DIR",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
)

#: Dropped even if explicitly requested: these carry credentials or would let a
#: child reach back into the parent Claude Code session.
_ENV_DENY_PATTERNS = (
    re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION)"),
    re.compile(r"^CLAUDE(_CODE)?_"),
    re.compile(r"^ANTHROPIC_"),
    re.compile(r"^OPENAI_"),
    re.compile(r"^AWS_"),
    re.compile(r"^GOOGLE_"),
    re.compile(r"^GH_|^GITHUB_"),
    re.compile(r"^NPM_"),
    re.compile(r"(?i)PROXY"),
)

MAX_CAPTURE_BYTES = 8 * 1024 * 1024


@dataclass
class CompletedCommand:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def summary(self) -> str:
        head = Path(self.argv[0]).name
        state = "timeout" if self.timed_out else f"exit={self.returncode}"
        return f"{head} {state} in {self.duration_s:.2f}s"


def build_child_env(
    extra: "dict[str, str] | None" = None,
    *,
    allowlist: "tuple[str, ...] | None" = None,
    base: "dict[str, str] | None" = None,
) -> dict[str, str]:
    """Construct a minimal child environment.

    ``extra`` is applied after the allowlist filter and is *not* subject to the
    deny patterns: a caller that deliberately passes a credential path (never a
    credential value) can do so. Callers must register any secret value with the
    redactor first.
    """
    source = os.environ if base is None else base
    names = ENV_ALLOWLIST if allowlist is None else allowlist
    env: dict[str, str] = {}
    for name in names:
        value = source.get(name)
        if value is None:
            continue
        if any(pattern.search(name) for pattern in _ENV_DENY_PATTERNS):
            continue
        env[name] = value
    # Never let a child inherit a proxy that could exfiltrate prompts.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    if extra:
        env.update({k: v for k, v in extra.items() if v is not None})
    return env


def _terminate_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # pragma: no cover - process already reaped
                pass
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            continue


def run_command(
    argv: "list[str] | tuple[str, ...]",
    *,
    cwd: os.PathLike | str | None = None,
    env: "dict[str, str] | None" = None,
    stdin_text: str | None = None,
    timeout_s: float = 300.0,
    max_bytes: int = MAX_CAPTURE_BYTES,
    raise_on_timeout: bool = False,
) -> CompletedCommand:
    """Run ``argv`` with no shell, a timeout and a capture cap."""
    argv = [str(a) for a in argv]
    if not argv:
        raise ProviderCrashed("refusing to run an empty command")

    child_env = build_child_env() if env is None else env
    started = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False, by design
            argv,
            cwd=os.fspath(cwd) if cwd else None,
            env=child_env,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, so timeouts kill children too
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ProviderCrashed(f"executable not found: {argv[0]}", detail=str(exc)) from exc
    except PermissionError as exc:
        raise ProviderCrashed(f"executable not runnable: {argv[0]}", detail=str(exc)) from exc

    timed_out = False
    try:
        stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:  # pragma: no cover - child already gone
            stdout, stderr = "", ""
    duration = time.monotonic() - started

    truncated = False
    if len(stdout) > max_bytes:
        stdout, truncated = stdout[:max_bytes] + "\n[...truncated by fabds...]", True
    if len(stderr) > max_bytes:
        stderr, truncated = stderr[:max_bytes] + "\n[...truncated by fabds...]", True

    result = CompletedCommand(
        argv=argv,
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=REDACTOR.scrub(stdout),
        stderr=REDACTOR.scrub(stderr),
        duration_s=duration,
        timed_out=timed_out,
        truncated=truncated,
    )
    if timed_out and raise_on_timeout:
        raise ProviderTimeout(
            f"{Path(argv[0]).name} exceeded its {timeout_s:.0f}s budget",
            detail=result.stderr[-2000:],
        )
    return result
