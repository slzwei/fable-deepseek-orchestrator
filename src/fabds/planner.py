"""Fable as planner and critic.

Fable is expensive, so it is consulted deliberately rather than reflexively.
Two entry points, both optional, both capped:

``plan``      once per run, at the start, when the task is big enough to be
              worth an architecture pass.
``critique``  only when :func:`should_escalate` finds a concrete reason.

Everything Fable returns is a *proposal*. The planner cannot assign a command,
widen a path grant or cause anything to execute: the orchestrator reads the
proposal, applies its own permission policy, and builds the real packets. A
planner that suggests ``rm -rf`` produces, at most, a suggestion the controller
declines.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from .cache import FileCache, plan_cache_key
from .context import ContextPacket
from .errors import MalformedResponse
from .logging import RunLogger
from .models import ResolvedModel, attest
from .packets import TaskKind, extract_json_object
from .prompts import (
    CRITIC_OUTPUT_CONTRACT,
    CRITIC_SYSTEM,
    PLANNER_OUTPUT_CONTRACT,
    PLANNER_SYSTEM,
)
from .providers.base import CompletionRequest, Provider

__all__ = [
    "PlannedPacket", "Plan", "Critique", "Planner",
    "EscalationDecision", "should_escalate", "task_fingerprint",
]


def task_fingerprint(task: str, constraints: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(task.strip().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(constraints.strip().encode("utf-8"))
    return digest.hexdigest()[:24]


@dataclass
class PlannedPacket:
    """A packet the planner *proposed*. Not yet authorised."""

    task_id: str
    kind: TaskKind
    objective: str
    owned_paths: tuple[str, ...] = ()
    readonly_paths: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    parallel_safe: bool = True
    rationale: str = ""

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id, "kind": self.kind.value, "objective": self.objective,
            "owned_paths": list(self.owned_paths), "readonly_paths": list(self.readonly_paths),
            "acceptance_criteria": list(self.acceptance_criteria),
            "depends_on": list(self.depends_on), "parallel_safe": self.parallel_safe,
            "rationale": self.rationale,
        }


@dataclass
class Plan:
    approach: str = ""
    key_decisions: tuple[dict, ...] = ()
    packets: tuple[PlannedPacket, ...] = ()
    risks: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    verification_strategy: str = ""
    attestation: dict = field(default_factory=dict)
    from_cache: bool = False
    raw: str = ""

    def as_dict(self) -> dict:
        return {
            "approach": self.approach,
            "key_decisions": list(self.key_decisions),
            "packets": [p.as_dict() for p in self.packets],
            "risks": list(self.risks),
            "open_questions": list(self.open_questions),
            "verification_strategy": self.verification_strategy,
            "attestation": self.attestation,
            "from_cache": self.from_cache,
        }


@dataclass
class Critique:
    verdict: str = "unknown"
    findings: tuple[dict, ...] = ()
    unverified_areas: tuple[str, ...] = ()
    summary: str = ""
    attestation: dict = field(default_factory=dict)

    @property
    def high_severity(self) -> list[dict]:
        return [f for f in self.findings if f.get("severity") == "high"]

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict, "findings": list(self.findings),
            "unverified_areas": list(self.unverified_areas),
            "summary": self.summary, "attestation": self.attestation,
        }


class Planner:
    """Wraps the planner provider with caching, attestation and round caps."""

    def __init__(self, *, provider: Provider, model: ResolvedModel, logger: RunLogger,
                 cache: FileCache | None = None, version: str = "0",
                 timeout_s: int = 900, max_rounds: int = 2,
                 plan_cache_ttl_s: int = 6 * 3600) -> None:
        self.provider = provider
        self.model = model
        self.logger = logger
        self.cache = cache
        self.version = version
        self.timeout_s = timeout_s
        self.max_rounds = max_rounds
        self.plan_cache_ttl_s = plan_cache_ttl_s
        self.rounds_used = 0
        self.calls = 0
        self.cache_hits = 0
        self.total_cost_usd = 0.0

    # -- planning -----------------------------------------------------------

    def plan(self, *, task: str, constraints: str, context: ContextPacket,
             repo_identity: str, git_state: dict, max_packets: int) -> Plan:
        if self.rounds_used >= self.max_rounds:
            raise MalformedResponse(
                f"planner round cap reached ({self.max_rounds}); the controller must "
                "continue without further planning"
            )

        key = plan_cache_key(
            repo_identity=repo_identity, git_state=git_state,
            task_fingerprint=task_fingerprint(task, constraints),
            planner_model=self.model.model_id, context_digest=context.digest(),
            version=self.version, round_index=self.rounds_used,
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                self.logger.info("planner", f"reusing cached plan ({cached.age_s:.0f}s old)")
                plan = _plan_from_dict(cached.payload)
                plan.from_cache = True
                return plan

        prompt = self._plan_prompt(task, constraints, context, max_packets)
        self.logger.info("planner", f"{self.model.display_name} started ({len(prompt)} chars of context)")
        started = time.monotonic()

        response = self.provider.complete(CompletionRequest(
            system_prompt=PLANNER_SYSTEM,
            user_prompt=prompt,
            model_id=self.model.model_id,
            max_output_tokens=8192,
            timeout_s=self.timeout_s,
            label="plan",
        ))
        attest(response, self.model)   # fail closed if another model answered
        self.calls += 1
        self.rounds_used += 1
        self.total_cost_usd += response.cost_usd or 0.0
        self.logger.ok(
            "planner",
            f"completed in {time.monotonic() - started:.1f}s "
            f"[{response.canonical_model} via {response.provider}]",
        )

        plan = _parse_plan(response.text, max_packets)
        plan.attestation = response.attestation()
        plan.raw = response.text
        if self.cache is not None:
            self.cache.put(key, plan.as_dict(), ttl_s=self.plan_cache_ttl_s,
                           metadata={"model": self.model.model_id})
        return plan

    def _plan_prompt(self, task: str, constraints: str, context: ContextPacket,
                     max_packets: int) -> str:
        return "\n".join([
            "# Planning request",
            "",
            f"Decompose the task below into at most {max_packets} work packets.",
            "Workers are inexpensive and fast but literal: give each one a narrow,",
            "unambiguous objective and objectively checkable acceptance criteria.",
            "",
            context.render(),
            "",
            PLANNER_OUTPUT_CONTRACT,
        ])

    # -- critique -----------------------------------------------------------

    def critique(self, *, material: str, question: str) -> Critique:
        if self.rounds_used >= self.max_rounds:
            raise MalformedResponse(f"planner round cap reached ({self.max_rounds})")
        self.logger.info("critic", f"{self.model.display_name} started: {question[:80]}")
        started = time.monotonic()

        response = self.provider.complete(CompletionRequest(
            system_prompt=CRITIC_SYSTEM,
            user_prompt="\n".join([
                "# Adversarial review request", "", f"Question: {question}", "",
                material, "", CRITIC_OUTPUT_CONTRACT,
            ]),
            model_id=self.model.model_id,
            max_output_tokens=6144,
            timeout_s=self.timeout_s,
            label="critique",
        ))
        attest(response, self.model)
        self.calls += 1
        self.rounds_used += 1
        self.total_cost_usd += response.cost_usd or 0.0
        self.logger.ok(
            "critic",
            f"completed in {time.monotonic() - started:.1f}s [{response.canonical_model}]",
        )

        critique = _parse_critique(response.text)
        critique.attestation = response.attestation()
        return critique

    def stats(self) -> dict:
        return {
            "calls": self.calls, "rounds_used": self.rounds_used,
            "max_rounds": self.max_rounds, "cache_hits": self.cache_hits,
            "cost_usd": round(self.total_cost_usd, 4),
        }


# -- escalation policy ------------------------------------------------------

@dataclass
class EscalationDecision:
    escalate: bool
    reason: str

    def __bool__(self) -> bool:
        return self.escalate


def should_escalate(*, results, plan: Plan | None, rounds_used: int, max_rounds: int,
                    repeated_failures: int, final_review_requested: bool = False,
                    architecture_changed: bool = False) -> EscalationDecision:
    """Decide whether a second Fable call is worth its cost.

    The default answer is no. Escalation happens only on a concrete, named
    trigger, so a run cannot drift into a planner/worker ping-pong loop.
    """
    if rounds_used >= max_rounds:
        return EscalationDecision(False, f"planner round cap reached ({max_rounds})")

    envelopes = list(results.values()) if hasattr(results, "values") else list(results)
    failed = [e for e in envelopes if e.status.value in ("failed", "blocked")]
    contradicted = [e for e in envelopes if e.claims_contradicted]
    denied = [e for e in envelopes if e.denied_actions]

    if repeated_failures >= 2:
        return EscalationDecision(
            True, f"{repeated_failures} attempted fixes failed; the approach may be wrong"
        )
    if architecture_changed:
        return EscalationDecision(True, "the architecture changed materially during the run")
    if envelopes and len(failed) >= max(1, len(envelopes) // 2):
        return EscalationDecision(
            True, f"{len(failed)} of {len(envelopes)} packets failed or were blocked"
        )
    if contradicted:
        ids = ", ".join(e.task_id for e in contradicted)
        return EscalationDecision(
            True, f"worker reports contradict observed changes ({ids})"
        )
    if any(e.tests_passed is False for e in envelopes):
        return EscalationDecision(True, "a validation command reported failing tests")
    if len(denied) >= 2:
        return EscalationDecision(
            True, f"{len(denied)} workers hit permission boundaries; the decomposition may be wrong"
        )
    if final_review_requested:
        return EscalationDecision(True, "a final adversarial review was explicitly requested")
    return EscalationDecision(False, "worker results are consistent and within scope")


# -- parsing ----------------------------------------------------------------

def _parse_plan(text: str, max_packets: int) -> Plan:
    payload = extract_json_object(text)
    raw_packets = payload.get("packets")
    if not isinstance(raw_packets, list) or not raw_packets:
        raise MalformedResponse(
            "the planner returned no packets", detail=json.dumps(payload)[:500]
        )

    packets: list[PlannedPacket] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_packets[:max_packets]):
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("task_id") or f"task_{index + 1:02d}").strip().lower()
        task_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:64] or f"task_{index + 1:02d}"
        while task_id in seen:
            task_id = f"{task_id}_{index + 1}"
        seen.add(task_id)

        try:
            kind = TaskKind(str(entry.get("kind", "implement")).strip().lower())
        except ValueError:
            kind = TaskKind.IMPLEMENT

        owned = _string_tuple(entry.get("owned_paths"))
        if kind.read_only:
            owned = ()   # the schema says read-only; enforce it rather than trust it
        packets.append(PlannedPacket(
            task_id=task_id,
            kind=kind,
            objective=str(entry.get("objective", "")).strip()[:2000],
            owned_paths=owned,
            readonly_paths=_string_tuple(entry.get("readonly_paths")),
            acceptance_criteria=_string_tuple(entry.get("acceptance_criteria")),
            depends_on=_string_tuple(entry.get("depends_on")),
            parallel_safe=bool(entry.get("parallel_safe", True)),
            rationale=str(entry.get("rationale", ""))[:1000],
        ))

    packets = _serialise_conflicts(packets)
    return Plan(
        approach=str(payload.get("approach", ""))[:4000],
        key_decisions=tuple(d for d in payload.get("key_decisions", []) if isinstance(d, dict)),
        packets=tuple(packets),
        risks=_string_tuple(payload.get("risks")),
        open_questions=_string_tuple(payload.get("open_questions")),
        verification_strategy=str(payload.get("verification_strategy", ""))[:2000],
    )


def _serialise_conflicts(packets: list[PlannedPacket]) -> list[PlannedPacket]:
    """Force packets that claim the same path into a dependency chain.

    The planner is asked not to overlap ownership, but the controller does not
    rely on the planner getting it right. Overlaps become dependencies, so the
    conflicting work runs one after another instead of racing.
    """
    claimed: dict[str, str] = {}
    adjusted: list[PlannedPacket] = []
    for packet in packets:
        extra_deps: list[str] = []
        for glob in packet.owned_paths:
            owner = claimed.get(glob)
            if owner and owner != packet.task_id:
                extra_deps.append(owner)
            else:
                claimed[glob] = packet.task_id
        if extra_deps:
            merged = tuple(dict.fromkeys(packet.depends_on + tuple(extra_deps)))
            packet = PlannedPacket(
                **{**packet.as_dict(), "kind": packet.kind,
                   "depends_on": merged, "parallel_safe": False,
                   "owned_paths": packet.owned_paths,
                   "readonly_paths": packet.readonly_paths,
                   "acceptance_criteria": packet.acceptance_criteria}
            )
        adjusted.append(packet)
    return adjusted


def _parse_critique(text: str) -> Critique:
    payload = extract_json_object(text)
    findings = [f for f in payload.get("findings", []) if isinstance(f, dict)]
    return Critique(
        verdict=str(payload.get("verdict", "unknown")).lower(),
        findings=tuple(findings),
        unverified_areas=_string_tuple(payload.get("unverified_areas")),
        summary=str(payload.get("summary", ""))[:4000],
    )


def _plan_from_dict(data: dict) -> Plan:
    packets = []
    for entry in data.get("packets", []):
        packets.append(PlannedPacket(
            task_id=entry["task_id"],
            kind=TaskKind(entry["kind"]),
            objective=entry.get("objective", ""),
            owned_paths=tuple(entry.get("owned_paths", ())),
            readonly_paths=tuple(entry.get("readonly_paths", ())),
            acceptance_criteria=tuple(entry.get("acceptance_criteria", ())),
            depends_on=tuple(entry.get("depends_on", ())),
            parallel_safe=entry.get("parallel_safe", True),
            rationale=entry.get("rationale", ""),
        ))
    return Plan(
        approach=data.get("approach", ""),
        key_decisions=tuple(data.get("key_decisions", ())),
        packets=tuple(packets),
        risks=tuple(data.get("risks", ())),
        open_questions=tuple(data.get("open_questions", ())),
        verification_strategy=data.get("verification_strategy", ""),
        attestation=data.get("attestation", {}),
    )


def _string_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip())
    return ()
