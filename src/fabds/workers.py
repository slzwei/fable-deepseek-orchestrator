"""DeepSeek workers: the action loop and the bounded pool.

A worker never receives a shell, a filesystem handle or a network socket. It
emits JSON actions; :class:`ActionExecutor` authorises each one against the
packet's permissions and executes only what survives. Anything refused is
recorded in the envelope, so a worker that repeatedly probes its boundary is
visible to the controller rather than silently ignored.

Every loop is bounded on four axes at once - turns, wall clock, retries and
output size - so no single failure mode can produce an unbounded run.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .errors import (
    ContextTooLarge,
    FabdsError,
    MalformedResponse,
    ModelAttestationError,
    PathSafetyError,
    PermissionDeniedError,
    ProviderError,
    ProviderRateLimited,
)
from .logging import RunLogger
from .models import ResolvedModel, attest
from .packets import (
    Action,
    ActionKind,
    ResultEnvelope,
    TaskKind,
    TaskStatus,
    WorkPacket,
    parse_actions,
)
from .permissions import WorkerPermissions
from .prompts import WORKER_SYSTEM, repository_content_block
from .providers.base import CompletionRequest, Provider
from .redaction import REDACTOR
from .runner import build_child_env, run_command
from .sanitizer import ContextSanitizer
from .workspace import Workspace

__all__ = ["ActionExecutor", "ActionResult", "WorkerRunner", "WorkerPool", "PoolLimits"]

MAX_READ_CHARS = 20_000
MAX_WRITE_BYTES = 2_000_000
MAX_SEARCH_RESULTS = 60
MAX_SEARCH_BYTES = 4_000_000
MAX_SEARCH_SECONDS = 10.0
MAX_PATTERN_CHARS = 200
MAX_OBSERVATION_CHARS = 12_000


@dataclass
class ActionResult:
    action: str
    ok: bool
    output: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def render(self) -> str:
        status = "OK" if self.ok else "REFUSED" if self.metadata.get("refused") else "ERROR"
        body = self.output if self.ok else self.error
        if len(body) > MAX_OBSERVATION_CHARS:
            body = body[:MAX_OBSERVATION_CHARS] + "\n... [truncated] ..."
        return f"[{self.action}] {status}\n{body}"


class ActionExecutor:
    """Authorises and performs one worker action at a time."""

    def __init__(self, workspace: Workspace, permissions: WorkerPermissions,
                 sanitizer: ContextSanitizer, *, command_timeout_s: int = 300) -> None:
        self.workspace = workspace
        self.permissions = permissions
        self.sanitizer = sanitizer
        self.command_timeout_s = command_timeout_s
        self.denied: list[dict] = []
        self.commands_run: list[dict] = []
        self.written: list[str] = []

    def execute(self, action: Action) -> ActionResult:
        handler = {
            ActionKind.READ_FILE: self._read_file,
            ActionKind.LIST_DIR: self._list_dir,
            ActionKind.SEARCH: self._search,
            ActionKind.WRITE_FILE: self._write_file,
            ActionKind.DELETE_FILE: self._delete_file,
            ActionKind.RUN_COMMAND: self._run_command,
        }.get(action.kind)
        if handler is None:
            return ActionResult(action.describe(), False, error=f"unsupported action {action.kind}")
        try:
            return handler(action.payload)
        except (PermissionDeniedError, PathSafetyError) as exc:
            self.denied.append({"action": action.describe(), "reason": exc.message, "code": exc.code})
            return ActionResult(
                action.describe(), False, error=f"{exc.message}\n"
                "This boundary is enforced by the controller. Work within your granted "
                "paths, or finish with status \"blocked\" explaining what you need.",
                metadata={"refused": True},
            )
        except FabdsError as exc:
            return ActionResult(action.describe(), False, error=exc.message)
        except OSError as exc:
            return ActionResult(action.describe(), False, error=f"filesystem error: {exc}")

    # -- reads --------------------------------------------------------------

    def _read_file(self, payload: dict) -> ActionResult:
        raw = payload["path"]
        resolved = self.permissions.assert_read(raw)
        if resolved.is_dir():
            return ActionResult(f"read_file({raw!r})", False, error="that path is a directory; use list_dir")
        result = self.sanitizer.read(resolved, max_chars=MAX_READ_CHARS)
        if not result.included:
            return ActionResult(
                f"read_file({raw!r})", False,
                error=f"refused: {result.reason}", metadata={"refused": True},
            )
        return ActionResult(
            f"read_file({raw!r})", True,
            output=repository_content_block(raw, result.text),
            metadata={"redactions": result.redactions.as_dict(), "truncated": result.truncated},
        )

    def _list_dir(self, payload: dict) -> ActionResult:
        raw = payload["path"]
        resolved = self.permissions.assert_read(raw)
        if not resolved.is_dir():
            return ActionResult(f"list_dir({raw!r})", False, error="not a directory")
        entries = []
        for child in sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))[:400]:
            rel = child.name
            if child.is_dir():
                if self.sanitizer.should_skip_dir(rel):
                    continue
                entries.append(f"{rel}/")
            else:
                try:
                    full = child.resolve(strict=False).relative_to(self.workspace.root).as_posix()
                except ValueError:
                    continue
                if self.sanitizer.is_secret_path(full):
                    entries.append(f"{rel}  [excluded: credential path]")
                else:
                    entries.append(f"{rel}  ({child.stat().st_size} bytes)")
        return ActionResult(f"list_dir({raw!r})", True, output="\n".join(entries) or "(empty)")

    def _search(self, payload: dict) -> ActionResult:
        """Regex search, executed out of process so its cost is truly bounded.

        The pattern comes from a model, so it is untrusted in the availability
        sense as well as the security sense: ``re`` cannot be interrupted, and a
        backtracking pattern would otherwise hang the worker forever. The parent
        decides which files are eligible; a child process does the matching and
        is killed if it overruns.
        """
        pattern = payload["pattern"]
        if len(pattern) > MAX_PATTERN_CHARS:
            return ActionResult("search", False,
                                error=f"pattern exceeds {MAX_PATTERN_CHARS} characters")
        try:
            re.compile(pattern)
        except re.error as exc:
            return ActionResult("search", False, error=f"invalid regex: {exc}")

        subtree = payload.get("path") or "."
        base = self.permissions.assert_read(subtree)
        candidates = [base] if base.is_file() else [
            p for p in sorted(base.rglob("*")) if p.is_file()
        ]

        eligible: list[str] = []
        for path in candidates:
            try:
                rel = path.resolve(strict=False).relative_to(self.workspace.root).as_posix()
            except ValueError:
                continue
            if not self.permissions.may_read(path) or self.sanitizer.is_secret_path(rel):
                continue
            if any(self.sanitizer.should_skip_dir(part) for part in Path(rel).parts[:-1]):
                continue
            eligible.append(rel)

        job = json.dumps({
            "root": str(self.workspace.root), "pattern": pattern, "files": eligible,
            "max_results": MAX_SEARCH_RESULTS, "max_bytes": MAX_SEARCH_BYTES,
        })
        scanner = Path(__file__).resolve().parent / "_scan.py"
        result = run_command(
            [sys.executable, str(scanner)],
            cwd=self.workspace.root, env=build_child_env(),
            stdin_text=job, timeout_s=MAX_SEARCH_SECONDS,
        )
        if result.timed_out:
            return ActionResult(
                f"search({pattern!r})", False,
                error=(f"the search was cancelled after {MAX_SEARCH_SECONDS:.0f}s. "
                       "That pattern is too expensive to evaluate - patterns with "
                       "nested quantifiers such as (a+)+ backtrack exponentially. "
                       "Use a simpler pattern or narrow the subtree."),
            )
        try:
            parsed = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return ActionResult(f"search({pattern!r})", False,
                                error=f"the scanner failed: {result.stderr[:300]}")
        if "error" in parsed:
            return ActionResult(f"search({pattern!r})", False, error=parsed["error"])

        matches = parsed.get("matches", [])
        lines = [
            f"{m['path']}:{m['line']}: {REDACTOR.scrub(m['text'])}" for m in matches
        ]
        if parsed.get("truncated"):
            lines.append(f"... [stopped at {MAX_SEARCH_RESULTS} matches] ...")
        return ActionResult(
            f"search({pattern!r})", True,
            output="\n".join(lines) if lines else "(no matches)",
            metadata={"matches": len(matches), "files_scanned": parsed.get("files_scanned", 0)},
        )

    # -- writes -------------------------------------------------------------

    def _write_file(self, payload: dict) -> ActionResult:
        raw = payload["path"]
        content = payload["content"]
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            return ActionResult(
                f"write_file({raw!r})", False,
                error=f"content exceeds {MAX_WRITE_BYTES} bytes",
            )
        resolved = self.permissions.assert_write(raw)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temp = resolved.with_name(resolved.name + ".fabds-tmp")
        temp.write_bytes(encoded)
        temp.replace(resolved)
        rel = self.permissions._relative(resolved)
        if rel not in self.written:
            self.written.append(rel)
        return ActionResult(
            f"write_file({raw!r})", True,
            output=f"wrote {len(encoded)} bytes to {rel}",
            metadata={"bytes": len(encoded)},
        )

    def _delete_file(self, payload: dict) -> ActionResult:
        raw = payload["path"]
        resolved = self.permissions.assert_write(raw)
        if resolved.is_dir():
            return ActionResult(
                f"delete_file({raw!r})", False,
                error="refusing to delete a directory; delete files individually",
                metadata={"refused": True},
            )
        if not resolved.exists():
            return ActionResult(f"delete_file({raw!r})", False, error="file does not exist")
        resolved.unlink()
        rel = self.permissions._relative(resolved)
        if rel not in self.written:
            self.written.append(rel)
        return ActionResult(f"delete_file({raw!r})", True, output=f"deleted {rel}")

    # -- commands -----------------------------------------------------------

    def _run_command(self, payload: dict) -> ActionResult:
        command_id = payload["command_id"]
        spec = self.permissions.assert_command(command_id)
        argv = spec.build_argv(list(payload.get("paths", [])), self.workspace.root)
        result = run_command(
            argv,
            cwd=self.workspace.root,
            env=build_child_env(),
            timeout_s=min(spec.timeout_s, self.command_timeout_s),
        )
        record = {
            "command_id": command_id,
            "argv": argv,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_s": round(result.duration_s, 2),
        }
        self.commands_run.append(record)
        body = f"$ {spec.display()}\nexit={result.returncode}"
        if result.timed_out:
            body += f" (timed out after {spec.timeout_s}s)"
        tail = (result.stdout + ("\n" + result.stderr if result.stderr else ""))[-MAX_OBSERVATION_CHARS:]
        return ActionResult(
            f"run_command({command_id!r})", True,
            output=f"{body}\n{tail}", metadata=record,
        )


class WorkerRunner:
    """Runs one packet to completion inside one workspace."""

    def __init__(self, *, provider: Provider, model: ResolvedModel, packet: WorkPacket,
                 workspace: Workspace, permissions: WorkerPermissions,
                 sanitizer: ContextSanitizer, logger: RunLogger,
                 max_turns: int = 12, timeout_s: int = 600,
                 max_retries: int = 2, command_timeout_s: int = 300) -> None:
        self.provider = provider
        self.model = model
        self.packet = packet
        self.workspace = workspace
        self.permissions = permissions
        self.sanitizer = sanitizer
        self.logger = logger
        self.max_turns = min(max_turns, packet.max_turns)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.command_timeout_s = command_timeout_s

    # -- prompt -------------------------------------------------------------

    def _initial_prompt(self) -> str:
        packet = self.packet
        lines = [
            f"# Work packet {packet.task_id}",
            "",
            f"Kind: {packet.kind.value} ({'read-only' if packet.read_only else 'may write'})",
            "",
            "## Objective",
            packet.objective,
            "",
            "## Paths you own (you may write only to these)",
            "\n".join(f"  - {p}" for p in packet.owned_paths) or "  (none - you are read-only)",
            "",
            "## Paths you may read",
            "\n".join(f"  - {p}" for p in packet.readonly_paths) or "  - (the whole workspace)",
        ]
        if packet.forbidden_paths:
            lines += ["", "## Paths explicitly forbidden",
                      "\n".join(f"  - {p}" for p in packet.forbidden_paths)]
        lines += [
            "",
            "## Commands available to you",
            "\n".join(
                f"  - {c.id}: {c.display()}" + (f"  # {c.description}" if c.description else "")
                for c in packet.commands
            ) or "  (none)",
            "",
            "## Acceptance criteria",
            "\n".join(f"  {i}. {c}" for i, c in enumerate(packet.acceptance_criteria, 1)) or "  (none stated)",
        ]
        if packet.validation_command_ids:
            lines += ["", "## You must run these before finishing",
                      "\n".join(f"  - {cid}" for cid in packet.validation_command_ids)]
        if packet.context:
            lines += ["", "## Context from the controller", packet.context]
        if packet.notes:
            lines += ["", "## Notes", packet.notes]
        lines += [
            "",
            f"You have at most {self.max_turns} turns. Begin by inspecting what you need, "
            "then act. Respond with one JSON object of actions.",
        ]
        return "\n".join(lines)

    # -- loop ---------------------------------------------------------------

    def run(self) -> ResultEnvelope:
        started = time.monotonic()
        envelope = ResultEnvelope(task_id=self.packet.task_id, status=TaskStatus.RUNNING)
        envelope.workspace = str(self.workspace.root)
        executor = ActionExecutor(
            self.workspace, self.permissions, self.sanitizer,
            command_timeout_s=self.command_timeout_s,
        )
        transcript = [self._initial_prompt()]
        attempts = 1
        finish_payload: dict | None = None

        for turn in range(1, self.max_turns + 1):
            if time.monotonic() - started > self.timeout_s:
                envelope.status = TaskStatus.FAILED
                envelope.error = {"code": "worker_timeout",
                                  "message": f"exceeded the {self.timeout_s}s budget"}
                break

            try:
                response, attempts_used = self._complete("\n\n".join(transcript), turn, attempts)
                attempts = attempts_used
            except FabdsError as exc:
                envelope.status = TaskStatus.FAILED
                envelope.error = exc.as_dict()
                self.logger.error(f"worker:{self.packet.task_id}", f"{exc.code}: {exc.message}")
                break

            envelope.attestation = response.attestation()
            envelope.turns_used = turn

            try:
                actions = parse_actions(response.text)
            except MalformedResponse as exc:
                self.logger.detail(
                    f"worker:{self.packet.task_id}",
                    f"turn {turn}: malformed response, re-prompting ({exc.message})",
                )
                transcript.append(
                    f"Your previous reply could not be parsed: {exc.message}\n"
                    "Reply with exactly one JSON object shaped "
                    '{"actions": [...]} and nothing else.'
                )
                if turn >= self.max_turns:
                    envelope.status = TaskStatus.FAILED
                    envelope.error = exc.as_dict()
                continue

            observations: list[str] = []
            finished = False
            for action in actions:
                if action.kind is ActionKind.FINISH:
                    finish_payload = action.payload
                    finished = True
                    break
                result = executor.execute(action)
                observations.append(result.render())
                self.logger.debug(
                    f"worker:{self.packet.task_id}",
                    f"turn {turn}: {result.action} -> {'ok' if result.ok else 'refused/error'}",
                )
            if finished:
                break

            remaining = self.max_turns - turn
            transcript.append(
                "\n\n".join(observations)
                + f"\n\n({remaining} turn(s) remain. Continue, or finish.)"
            )
        else:
            envelope.error = envelope.error or {
                "code": "turn_limit", "message": f"used all {self.max_turns} turns without finishing",
            }

        self._finalise(envelope, executor, finish_payload, started)
        return envelope

    def _complete(self, prompt: str, turn: int, attempts: int):
        """One provider call with bounded retries and attestation."""
        last: Exception | None = None
        for attempt in range(attempts, attempts + self.max_retries + 1):
            request = CompletionRequest(
                system_prompt=WORKER_SYSTEM,
                user_prompt=prompt,
                model_id=self.model.model_id,
                max_output_tokens=self.packet.max_output_tokens,
                reasoning_effort=self.packet.reasoning_effort,
                timeout_s=min(self.timeout_s, 300),
                label=f"{self.packet.task_id}#t{turn}",
            )
            try:
                response = self.provider.complete(request)
                attest(response, self.model)   # fail closed on the wrong model
                return response, attempt
            except ModelAttestationError:
                raise                          # never retry a wrong-model response
            except ContextTooLarge:
                raise
            except ProviderRateLimited as exc:
                last = exc
                delay = min(30.0, 2.0 ** (attempt - attempts + 1))
                self.logger.warn(
                    f"worker:{self.packet.task_id}",
                    f"rate limited, retrying in {delay:.0f}s (attempt {attempt})",
                )
                time.sleep(delay)
            except ProviderError as exc:
                last = exc
                if not exc.retryable:
                    raise
                self.logger.warn(
                    f"worker:{self.packet.task_id}",
                    f"{exc.code}, retrying (attempt {attempt})",
                )
                time.sleep(min(10.0, 1.5 ** (attempt - attempts + 1)))
        raise last or ProviderError("worker call failed with no recorded error")

    def _finalise(self, envelope: ResultEnvelope, executor: ActionExecutor,
                  finish_payload: dict | None, started: float) -> None:
        envelope.duration_s = time.monotonic() - started
        envelope.denied_actions = tuple(executor.denied)
        envelope.command_results = tuple(executor.commands_run)
        envelope.observed_files_changed = tuple(self.workspace.changed_files())

        if finish_payload is not None:
            envelope.summary = str(finish_payload.get("summary", ""))[:4000]
            envelope.files_changed = _string_tuple(finish_payload.get("files_changed"))
            envelope.commands_run = _string_tuple(finish_payload.get("commands_run"))
            envelope.tests_run = _string_tuple(finish_payload.get("tests_run"))
            tests_passed = finish_payload.get("tests_passed")
            envelope.tests_passed = tests_passed if isinstance(tests_passed, bool) else None
            envelope.unresolved = _string_tuple(finish_payload.get("unresolved"))
            envelope.risks = _string_tuple(finish_payload.get("risks"))
            envelope.recommended_next_step = str(finish_payload.get("recommended_next_step", ""))[:1000]
            claimed = str(finish_payload.get("status", "completed")).lower()
            envelope.status = (
                TaskStatus.BLOCKED if claimed == "blocked" else TaskStatus.COMPLETED
            )
        elif envelope.status is TaskStatus.RUNNING:
            envelope.status = TaskStatus.FAILED

        # A writing worker that produced nothing is a failure, whatever it claims.
        if (envelope.status is TaskStatus.COMPLETED
                and not self.packet.read_only
                and not envelope.observed_files_changed):
            envelope.status = TaskStatus.FAILED
            envelope.error = {
                "code": "no_changes",
                "message": "the worker reported success but the controller observed "
                           "no file changes in its workspace",
            }


def _string_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if isinstance(v, (str, int, float)))
    return ()


@dataclass
class PoolLimits:
    """Per-kind concurrency. Implementation is deliberately the narrowest."""

    max_workers: int = 4
    research: int = 3
    implementation: int = 2
    test: int = 2

    def for_kind(self, kind: TaskKind) -> int:
        return {
            TaskKind.RESEARCH: self.research,
            TaskKind.AUDIT: self.research,
            TaskKind.IMPLEMENT: self.implementation,
            TaskKind.TEST: self.test,
            TaskKind.DOCUMENT: self.test,
        }.get(kind, 1)


class WorkerPool:
    """Bounded, dependency-aware, conflict-aware execution.

    Three separate brakes:

    * a global semaphore of ``max_workers``,
    * a per-kind semaphore, so a swarm of research workers cannot crowd out
      the implementation slots,
    * a file-ownership lock, so two packets claiming the same path are
      serialised even when the planner marked them parallel.

    Only this class creates workers. Workers have no mechanism to create more:
    they emit actions, and no action spawns anything.
    """

    def __init__(self, limits: PoolLimits, logger: RunLogger) -> None:
        self.limits = limits
        self.logger = logger
        self._global = threading.Semaphore(limits.max_workers)
        self._per_kind = {kind: threading.Semaphore(limits.for_kind(kind)) for kind in TaskKind}
        self._path_locks: dict[str, threading.Lock] = {}
        self._path_lock_guard = threading.Lock()
        self.peak_concurrency = 0
        self._active = 0
        self._active_guard = threading.Lock()

    def _locks_for(self, packet: WorkPacket) -> list[threading.Lock]:
        locks = []
        with self._path_lock_guard:
            for glob in sorted(packet.owned_paths):
                if glob not in self._path_locks:
                    self._path_locks[glob] = threading.Lock()
                locks.append(self._path_locks[glob])
        return locks

    def run(self, jobs: "list[tuple[WorkPacket, object]]") -> dict[str, ResultEnvelope]:
        """Run ``(packet, runner_factory)`` pairs, honouring ``depends_on``."""
        results: dict[str, ResultEnvelope] = {}
        by_id = {packet.task_id: (packet, factory) for packet, factory in jobs}
        pending = dict(by_id)
        done: set[str] = set()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.limits.max_workers), thread_name_prefix="fabds-worker"
        ) as pool:
            while pending:
                ready = [
                    task_id for task_id, (packet, _) in pending.items()
                    if all(dep in done for dep in packet.depends_on)
                ]
                if not ready:
                    for task_id, (packet, _) in pending.items():
                        blocked = [d for d in packet.depends_on if d not in done]
                        results[task_id] = ResultEnvelope(
                            task_id=task_id, status=TaskStatus.SKIPPED,
                            summary=f"dependencies did not complete: {', '.join(blocked)}",
                        )
                    break

                futures = {}
                for task_id in ready:
                    packet, factory = pending.pop(task_id)
                    futures[pool.submit(self._run_one, packet, factory)] = task_id

                for future in concurrent.futures.as_completed(futures):
                    task_id = futures[future]
                    try:
                        results[task_id] = future.result()
                    except Exception as exc:  # a crash in one worker never kills the run
                        self.logger.error(f"worker:{task_id}", f"crashed: {exc}")
                        results[task_id] = ResultEnvelope(
                            task_id=task_id, status=TaskStatus.FAILED,
                            error={"code": "worker_crashed", "message": str(exc)[:500]},
                        )
                    # Only successful dependencies unblock dependents.
                    if results[task_id].succeeded:
                        done.add(task_id)
        return results

    def _run_one(self, packet: WorkPacket, factory) -> ResultEnvelope:
        path_locks = self._locks_for(packet)
        with self._global, self._per_kind[packet.kind]:
            for lock in path_locks:
                lock.acquire()
            try:
                with self._active_guard:
                    self._active += 1
                    self.peak_concurrency = max(self.peak_concurrency, self._active)
                self.logger.info(
                    f"worker:{packet.task_id}",
                    f"started ({packet.kind.value})",
                )
                try:
                    envelope = factory().run()
                finally:
                    with self._active_guard:
                        self._active -= 1
                self.logger.info(
                    f"worker:{packet.task_id}",
                    f"{envelope.status.value} in {envelope.duration_s:.1f}s "
                    f"({len(envelope.observed_files_changed)} file(s) changed)",
                )
                return envelope
            finally:
                for lock in reversed(path_locks):
                    lock.release()
