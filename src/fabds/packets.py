"""Work packets, the worker action protocol, and result envelopes.

A **work packet** is the complete, self-contained statement of one bounded
job: what to achieve, what may be read, what may be written, which commands
exist, and what "done" means. A worker receives nothing else - no conversation
history, no controller reasoning, no repository dump.

The **action protocol** is how a worker affects anything. A worker emits JSON
actions; the controller executes the safe ones and refuses the rest. Crucially,
``run_command`` carries a command *id*, not a command string: the worker picks
from the packet's allowlist and can never compose an invocation.

A **result envelope** is the structured report the controller reconciles
against evidence it gathers itself. A worker saying ``tests_passed: true`` is a
claim, not a fact, and the orchestrator records it as such.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import MalformedResponse
from .permissions import CommandSpec

__all__ = [
    "TaskKind", "TaskStatus", "WorkPacket", "ResultEnvelope",
    "Action", "ActionKind", "parse_actions", "extract_json_object",
    "RESULT_SCHEMA_DESCRIPTION",
]


class TaskKind(str, Enum):
    RESEARCH = "research"
    IMPLEMENT = "implement"
    TEST = "test"
    AUDIT = "audit"
    DOCUMENT = "document"

    @property
    def read_only(self) -> bool:
        return self in (TaskKind.RESEARCH, TaskKind.AUDIT)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"      # the worker finished and reported
    FAILED = "failed"            # the worker or its provider failed
    BLOCKED = "blocked"          # the worker reported it cannot proceed
    SKIPPED = "skipped"          # a dependency failed, so this never ran
    REJECTED = "rejected"        # the controller refused the result


class ActionKind(str, Enum):
    READ_FILE = "read_file"
    LIST_DIR = "list_dir"
    SEARCH = "search"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    RUN_COMMAND = "run_command"
    FINISH = "finish"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    payload: dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind is ActionKind.RUN_COMMAND:
            return f"run_command({self.payload.get('command_id')!r})"
        target = self.payload.get("path") or self.payload.get("pattern") or ""
        return f"{self.kind.value}({target!r})" if target else self.kind.value


@dataclass
class WorkPacket:
    """One bounded unit of delegated work."""

    task_id: str
    kind: TaskKind
    objective: str
    context: str = ""
    owned_paths: tuple[str, ...] = ()
    readonly_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    commands: tuple[CommandSpec, ...] = ()
    validation_command_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    max_turns: int = 12
    #: Writing a whole file plus reasoning needs headroom; DeepSeek caps at 8192.
    max_output_tokens: int = 8000
    reasoning_effort: str = "high"
    notes: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]{0,63}", self.task_id):
            raise MalformedResponse(
                f"task_id must be lowercase alphanumeric with - or _: {self.task_id!r}"
            )
        if isinstance(self.kind, str):
            self.kind = TaskKind(self.kind)
        if self.kind.read_only and self.owned_paths:
            raise MalformedResponse(
                f"{self.task_id}: a {self.kind.value} packet is read-only and cannot own paths"
            )
        unknown = set(self.validation_command_ids) - {c.id for c in self.commands}
        if unknown:
            raise MalformedResponse(
                f"{self.task_id}: validation command id(s) not in the allowlist: "
                f"{', '.join(sorted(unknown))}"
            )

    @property
    def read_only(self) -> bool:
        return self.kind.read_only

    def command_map(self) -> dict[str, CommandSpec]:
        return {c.id: c for c in self.commands}

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "objective": self.objective,
            "read_only": self.read_only,
            "owned_paths": list(self.owned_paths),
            "readonly_paths": list(self.readonly_paths),
            "forbidden_paths": list(self.forbidden_paths),
            "acceptance_criteria": list(self.acceptance_criteria),
            "commands": [c.as_dict() for c in self.commands],
            "validation_command_ids": list(self.validation_command_ids),
            "depends_on": list(self.depends_on),
            "max_turns": self.max_turns,
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        scope = ", ".join(self.owned_paths) if self.owned_paths else "read-only"
        return f"{self.task_id} [{self.kind.value}] {self.objective[:70]} ({scope})"


@dataclass
class ResultEnvelope:
    """A worker's structured report plus the controller's own observations."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    summary: str = ""
    files_changed: tuple[str, ...] = ()
    commands_run: tuple[str, ...] = ()
    tests_run: tuple[str, ...] = ()
    tests_passed: bool | None = None
    unresolved: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    recommended_next_step: str = ""

    # Controller-observed, never model-reported:
    observed_files_changed: tuple[str, ...] = ()
    command_results: tuple[dict, ...] = ()
    attestation: dict = field(default_factory=dict)
    turns_used: int = 0
    attempts: int = 1
    error: dict | None = None
    denied_actions: tuple[dict, ...] = ()
    workspace: str = ""
    duration_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status is TaskStatus.COMPLETED

    @property
    def claims_contradicted(self) -> bool:
        """True when the worker's report disagrees with what we observed."""
        claimed = set(self.files_changed)
        observed = set(self.observed_files_changed)
        return bool(claimed ^ observed) and bool(claimed or observed)

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "summary": self.summary,
            "files_changed": list(self.files_changed),
            "observed_files_changed": list(self.observed_files_changed),
            "commands_run": list(self.commands_run),
            "tests_run": list(self.tests_run),
            "tests_passed": self.tests_passed,
            "unresolved": list(self.unresolved),
            "risks": list(self.risks),
            "recommended_next_step": self.recommended_next_step,
            "command_results": list(self.command_results),
            "attestation": self.attestation,
            "turns_used": self.turns_used,
            "attempts": self.attempts,
            "error": self.error,
            "denied_actions": list(self.denied_actions),
            "workspace": self.workspace,
            "duration_s": round(self.duration_s, 2),
            "claims_contradicted": self.claims_contradicted,
        }


RESULT_SCHEMA_DESCRIPTION = """\
{
  "summary": "<what you did, 1-3 sentences>",
  "files_changed": ["<workspace-relative path>", ...],
  "commands_run": ["<command_id>", ...],
  "tests_run": ["<command_id>", ...],
  "tests_passed": true | false | null,
  "unresolved": ["<what you could not finish>", ...],
  "risks": ["<what could break>", ...],
  "recommended_next_step": "<one concrete next action for the controller>",
  "status": "completed" | "blocked"
}"""


# -- parsing model output ---------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Pull one JSON object out of model text.

    Models wrap JSON in prose and fences. This tolerates that, then fails
    loudly rather than guessing. It never evaluates the text.
    """
    if not text or not text.strip():
        raise MalformedResponse("model returned an empty response")

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    candidates.extend(match.group(1).strip() for match in _FENCE_RE.finditer(text))

    start = stripped.find("{")
    if start != -1:
        depth, in_string, escape = 0, False, False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start:index + 1])
                    break

    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    salvaged = repair_truncated_json(stripped)
    if salvaged is not None:
        salvaged["_truncated_report"] = True
        return salvaged

    raise MalformedResponse(
        "model response did not contain a JSON object", detail=text[:600]
    )


def repair_truncated_json(text: str) -> dict | None:
    """Close an object that was cut off mid-emission.

    A worker whose *report* is truncated has usually already done the work: the
    files are written and the controller has verified them. Throwing the report
    away and calling the packet failed loses real work over a formatting
    accident. So a response that clearly began a JSON object gets one repair
    attempt - terminate an open string, close open brackets - and the result is
    flagged ``_truncated_report`` so nothing downstream mistakes it for a
    complete envelope.

    Only ever applied as a last resort, and only to text that starts with "{".
    """
    if not text.startswith("{"):
        return None

    in_string = escape = False
    stack: list[str] = []
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()

    repaired = text
    if escape:
        repaired = repaired[:-1]
    if in_string:
        repaired += '"'
    # Drop a dangling "key": with no value, and any trailing comma.
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', "", repaired)
    repaired = re.sub(r',\s*$', "", repaired)
    repaired += "".join(reversed(stack))

    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_actions(text: str, *, max_actions: int = 8) -> list[Action]:
    """Parse a worker turn into validated actions.

    Structure only: this says an action is *well formed*, not that it is
    *allowed*. Authorisation happens in :mod:`fabds.permissions` when the
    action is executed.
    """
    payload = extract_json_object(text)
    # A repaired object flags itself on the outside; carry that onto the action
    # it produced, or the warning is lost when the envelope is assembled.
    truncated = bool(payload.pop("_truncated_report", False))
    raw_actions = payload.get("actions")
    if raw_actions is None and "action" in payload:
        raw_actions = [payload["action"]]
    if raw_actions is None:
        # A bare result envelope is treated as an implicit finish.
        if "summary" in payload or "status" in payload:
            if truncated:
                payload["_truncated_report"] = True
            return [Action(ActionKind.FINISH, payload)]
        raise MalformedResponse(
            "worker response has no 'actions' list and is not a result envelope",
            detail=json.dumps(payload)[:400],
        )
    if not isinstance(raw_actions, list):
        raise MalformedResponse("'actions' must be a list")
    if not raw_actions:
        raise MalformedResponse("'actions' was empty; emit at least one action or finish")

    actions: list[Action] = []
    for index, entry in enumerate(raw_actions[:max_actions]):
        if not isinstance(entry, dict):
            raise MalformedResponse(f"action {index} is not an object")
        raw_kind = entry.get("op") or entry.get("action") or entry.get("kind")
        if not isinstance(raw_kind, str):
            raise MalformedResponse(f"action {index} has no 'op'")
        try:
            kind = ActionKind(raw_kind.strip().lower())
        except ValueError as exc:
            raise MalformedResponse(
                f"unknown action {raw_kind!r}; valid actions are "
                f"{', '.join(a.value for a in ActionKind)}"
            ) from exc
        payload_fields = {k: v for k, v in entry.items() if k not in ("op", "action", "kind")}
        _validate_action_shape(kind, payload_fields, index)
        if truncated:
            payload_fields["_truncated_report"] = True
        actions.append(Action(kind, payload_fields))
    return actions


def _validate_action_shape(kind: ActionKind, payload: dict[str, Any], index: int) -> None:
    def require(field_name: str, types: tuple) -> None:
        if field_name not in payload or not isinstance(payload[field_name], types):
            raise MalformedResponse(
                f"action {index} ({kind.value}) requires a {field_name!r} of type "
                f"{'/'.join(t.__name__ for t in types)}"
            )

    if kind in (ActionKind.READ_FILE, ActionKind.LIST_DIR, ActionKind.DELETE_FILE):
        require("path", (str,))
    elif kind is ActionKind.WRITE_FILE:
        require("path", (str,))
        require("content", (str,))
    elif kind is ActionKind.SEARCH:
        require("pattern", (str,))
    elif kind is ActionKind.RUN_COMMAND:
        require("command_id", (str,))
        extra = payload.get("paths", [])
        if not isinstance(extra, list) or any(not isinstance(p, str) for p in extra):
            raise MalformedResponse(f"action {index} (run_command) 'paths' must be a list of strings")
    elif kind is ActionKind.FINISH:
        if "summary" not in payload:
            raise MalformedResponse(f"action {index} (finish) requires a 'summary'")
