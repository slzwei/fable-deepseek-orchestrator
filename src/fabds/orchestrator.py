"""The controller.

This module is the part that keeps authority where it belongs. It:

* resolves both models before anything else and aborts if either is missing,
* builds a sanitised, budgeted context packet,
* asks the planner for a decomposition and treats the answer as a *proposal*,
* authorises each proposed packet against controller policy - the command
  allowlist, the ownership map, the read-only rule - producing the real packets,
* runs them under bounded concurrency in isolated workspaces,
* verifies the results itself rather than believing the workers,
* escalates to the critic only on a named trigger,
* and stops, leaving patches staged for the controller to integrate.

It never merges into the user's working tree. ``integrate`` exists, is separate,
and applies only the packets it is explicitly told to apply.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import __version__
from .cache import FileCache
from .config import Config
from .context import ContextBuilder, ContextPacket, git_state, repo_identity
from .errors import (
    ConfigError,
    FabdsError,
    LimitExceeded,
    ModelResolutionError,
    PermissionDeniedError,
    WorkspaceError,
)
from .ledger import RunLedger, new_run_id
from .logging import Level, RunLogger
from .models import ModelResolver, ResolvedModel, Role
from .packets import ResultEnvelope, TaskKind, TaskStatus, WorkPacket
from .permissions import CommandSpec, WorkerPermissions
from .planner import Plan, PlannedPacket, Planner, should_escalate
from .runner import build_child_env, run_command
from .sanitizer import ContextSanitizer
from .workers import PoolLimits, WorkerPool, WorkerRunner
from .workspace import create_workspace, supports_worktrees

__all__ = ["Orchestrator", "RunOutcome", "detect_validation_commands"]


@lru_cache(maxsize=8)
def _module_available(module: str) -> bool:
    """Is ``module`` importable by the interpreter a worker command would use?

    Offering a command that cannot run wastes a worker turn and produces a
    confusing failure that looks like the worker's fault. Checked once per
    process.
    """
    return run_command(["python3", "-c", f"import {module}"], timeout_s=60).ok


def detect_validation_commands(repo_root: Path) -> list[CommandSpec]:
    """Infer a small, safe validation allowlist from repository conventions.

    Controller-authored by construction: the *controller* inspects the repo and
    decides which commands exist. A worker can only pick from the result.
    """
    root = Path(repo_root)
    specs: list[CommandSpec] = []

    looks_python = any((root / name).exists() for name in
                       ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "tests", "test"))
    if looks_python:
        if _module_available("pytest"):
            specs.append(CommandSpec(
                "pytest", ("python3", "-m", "pytest", "-q"),
                "run the Python test suite", timeout_s=900, max_extra_paths=1,
            ))
        else:
            # pytest is not importable here, so offering it would guarantee a
            # confusing failure. unittest is always present - but plain
            # `discover` silently finds nothing when tests/ is not an importable
            # package, which is the common layout. Point it at the directory and
            # make that directory the import root.
            tests_dir = next((d for d in ("tests", "test") if (root / d).is_dir()), None)
            if tests_dir and not (root / tests_dir / "__init__.py").exists():
                argv = ("python3", "-m", "unittest", "discover",
                        "-s", tests_dir, "-t", tests_dir)
            else:
                argv = ("python3", "-m", "unittest", "discover")
            specs.append(CommandSpec(
                "unittest", argv, "run the standard-library test suite", timeout_s=900,
            ))
        specs.append(CommandSpec(
            "py_compile", ("python3", "-m", "compileall", "-q", "."),
            "byte-compile the tree to catch syntax errors", timeout_s=300,
        ))

    package_json = root / "package.json"
    if package_json.is_file():
        try:
            import json

            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if "test" in scripts:
            specs.append(CommandSpec(
                "npm_test", ("npm", "test", "--silent"),
                "run the package's own test script", timeout_s=900,
            ))
        if "lint" in scripts:
            specs.append(CommandSpec(
                "npm_lint", ("npm", "run", "lint", "--silent"),
                "run the package's own lint script", timeout_s=600,
            ))

    if (root / "Makefile").is_file():
        try:
            targets = (root / "Makefile").read_text(encoding="utf-8", errors="replace")
        except OSError:
            targets = ""
        if "\ntest:" in targets or targets.startswith("test:"):
            specs.append(CommandSpec("make_test", ("make", "test"), "run the Makefile test target",
                                     timeout_s=900))

    specs.append(CommandSpec(
        "git_status", ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        "show what has changed in this workspace", timeout_s=60,
    ))
    return specs


@dataclass
class RunOutcome:
    run_id: str
    ok: bool
    task: str
    models: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    packets: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    critique: dict | None = None
    verification: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    error: dict | None = None
    ledger_root: str = ""
    dry_run: bool = False

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id, "ok": self.ok, "task": self.task,
            "models": self.models, "plan": self.plan, "packets": self.packets,
            "results": {k: v.as_dict() if hasattr(v, "as_dict") else v
                        for k, v in self.results.items()},
            "critique": self.critique, "verification": self.verification,
            "stats": self.stats, "error": self.error,
            "ledger_root": self.ledger_root, "dry_run": self.dry_run,
            "fabds_version": __version__,
        }

    def summary_lines(self) -> list[str]:
        lines = [f"run {self.run_id}: {'ok' if self.ok else 'FAILED'}"]
        for role, info in self.models.items():
            lines.append(f"  {role:8} {info.get('display_name')} = {info.get('model_id')} "
                         f"via {info.get('provider')}")
        for task_id, result in self.results.items():
            status = result.status.value if hasattr(result, "status") else result.get("status")
            lines.append(f"  {task_id:16} {status}")
        if self.error:
            lines.append(f"  error: {self.error.get('message')}")
        return lines


class Orchestrator:
    """Owns one orchestration run."""

    def __init__(self, config: Config, repo_root: Path, *, logger: RunLogger | None = None,
                 resolver: ModelResolver | None = None, run_id: str | None = None) -> None:
        self.config = config
        self.repo_root = Path(repo_root).resolve(strict=False)
        self.run_id = run_id or new_run_id()
        self.ledger = RunLedger(self.repo_root, self.run_id)
        self.logger = logger or RunLogger(level=Level.NORMAL, jsonl_path=self.ledger.events_path)
        self.resolver = resolver or ModelResolver(config)
        self.cache = FileCache(Path(config.cache_dir), enabled=config.cache_enabled)
        self.sanitizer = ContextSanitizer(
            self.repo_root,
            extra_secret_patterns=config.extra_secret_paths,
            allowlist=config.context_allowlist,
            max_file_chars=config.limits.max_file_excerpt_chars,
        )
        self._started = time.monotonic()

    # -- model resolution ---------------------------------------------------

    def resolve_models(self) -> dict[str, ResolvedModel]:
        """Resolve every required model up front. Fail closed on any miss."""
        resolved: dict[str, ResolvedModel] = {}
        for role in (Role.PLANNER, Role.WORKER):
            model = self.resolver.resolve(role)
            resolved[role] = model
            marker = "  [EXPLICIT FALLBACK]" if model.via_fallback else ""
            self.logger.info(
                "controller",
                f"{role} resolved: {model.display_name} = {model.model_id} "
                f"via {model.provider} ({model.source}){marker}",
            )
        return resolved

    # -- context ------------------------------------------------------------

    def build_context(self, task: str, constraints: str, *, focus_paths=(),
                      previous_attempts=()) -> ContextPacket:
        builder = ContextBuilder(
            self.repo_root, self.sanitizer,
            budget_chars=self.config.limits.max_planner_context_chars,
        )
        builder.add_task(task, constraints)
        builder.add_git_state()
        builder.add_tree(max_entries=250)
        builder.add_previous_attempts(previous_attempts)
        if focus_paths:
            builder.add_files(focus_paths)
        commands = detect_validation_commands(self.repo_root)
        builder.add(
            "Validation commands available to workers",
            "\n".join(f"- {c.id}: {c.display()}  # {c.description}" for c in commands),
            priority=28,
        )
        return builder.build()

    # -- packet authorisation ----------------------------------------------

    def authorise(self, proposals: "list[PlannedPacket]", *,
                  commands: "list[CommandSpec]") -> list[WorkPacket]:
        """Turn planner proposals into packets the controller is willing to run.

        This is where the planner's advice stops being advice. The controller
        decides the commands, clamps the counts, enforces the read-only rule and
        refuses anything whose ownership it cannot express safely.
        """
        limits = self.config.limits
        if len(proposals) > limits.max_total_tasks:
            self.logger.warn(
                "controller",
                f"planner proposed {len(proposals)} packets; clamping to "
                f"max_total_tasks={limits.max_total_tasks}",
            )
            proposals = proposals[: limits.max_total_tasks]

        command_map = {c.id: c for c in commands}
        validation_ids = tuple(
            cid for cid in ("pytest", "unittest", "npm_test", "make_test")
            if cid in command_map
        )
        packets: list[WorkPacket] = []
        known_ids = {p.task_id for p in proposals}

        for proposal in proposals:
            granted = tuple(command_map.values()) if not proposal.kind.read_only else (
                tuple(c for c in commands if c.id in ("git_status",))
            )
            deps = tuple(d for d in proposal.depends_on if d in known_ids and d != proposal.task_id)
            try:
                packet = WorkPacket(
                    task_id=proposal.task_id,
                    kind=proposal.kind,
                    objective=proposal.objective,
                    context=proposal.rationale,
                    owned_paths=proposal.owned_paths,
                    readonly_paths=proposal.readonly_paths,
                    acceptance_criteria=proposal.acceptance_criteria,
                    commands=granted,
                    validation_command_ids=(
                        tuple(v for v in validation_ids if v in {g.id for g in granted})
                        if not proposal.kind.read_only else ()
                    ),
                    depends_on=deps,
                    max_turns=limits.max_worker_turns,
                )
            except FabdsError as exc:
                self.logger.warn("controller", f"rejected packet {proposal.task_id}: {exc.message}")
                continue
            packets.append(packet)

        if not packets:
            raise LimitExceeded("no proposed packet survived controller authorisation")
        return packets

    # -- execution ----------------------------------------------------------

    def execute(self, packets: "list[WorkPacket]", worker_model: ResolvedModel,
                *, base_rev: str = "HEAD") -> dict[str, ResultEnvelope]:
        from .providers import get_provider

        limits = self.config.limits
        pool = WorkerPool(
            PoolLimits(
                max_workers=limits.max_workers,
                research=limits.max_research_workers,
                implementation=limits.max_implementation_workers,
                test=limits.max_test_workers,
            ),
            self.logger,
        )
        workspaces = {}
        jobs = []

        for packet in packets:
            workspace = create_workspace(
                packet.task_id, self.repo_root,
                read_only=packet.read_only, base_rev=base_rev,
                include_globs=tuple(packet.owned_paths) + tuple(packet.readonly_paths) or ("**/*",),
                sanitizer=self.sanitizer,
            )
            workspace.setup()
            workspaces[packet.task_id] = workspace
            permissions = WorkerPermissions(
                workspace_root=workspace.root,
                owned=packet.owned_paths,
                readonly=packet.readonly_paths,
                forbidden=packet.forbidden_paths,
                read_only=packet.read_only,
                commands=packet.command_map(),
            )
            sanitizer = ContextSanitizer(
                workspace.root,
                extra_secret_patterns=self.config.extra_secret_paths,
                allowlist=self.config.context_allowlist,
                max_file_chars=self.config.limits.max_file_excerpt_chars,
            )

            def factory(packet=packet, workspace=workspace, permissions=permissions,
                        sanitizer=sanitizer):
                return WorkerRunner(
                    provider=get_provider(worker_model.provider, self.config),
                    model=worker_model,
                    packet=packet,
                    workspace=workspace,
                    permissions=permissions,
                    sanitizer=sanitizer,
                    logger=self.logger,
                    max_turns=limits.max_worker_turns,
                    timeout_s=limits.worker_timeout_s,
                    max_retries=limits.max_retries_per_task,
                    command_timeout_s=limits.command_timeout_s,
                )

            jobs.append((packet, factory))

        try:
            results = pool.run(jobs)
            self.logger.info(
                "controller",
                f"peak concurrency {pool.peak_concurrency} (limit {limits.max_workers})",
            )
            for task_id, workspace in workspaces.items():
                envelope = results.get(task_id)
                if envelope is None:
                    continue
                patch = workspace.export_patch()
                saved = self.ledger.save_patch(task_id, patch)
                if saved is not None:
                    self.logger.detail("controller", f"staged patch for {task_id} at {saved}")
            return results
        finally:
            self._verify_and_cleanup(workspaces, packets)

    def _verify_and_cleanup(self, workspaces: dict, packets: "list[WorkPacket]") -> None:
        self._verification: dict = getattr(self, "_verification", {})
        by_id = {p.task_id: p for p in packets}
        for task_id, workspace in workspaces.items():
            packet = by_id.get(task_id)
            if packet is not None and not packet.read_only:
                self._verification[task_id] = self._independent_check(packet, workspace)
            workspace.cleanup()

    def _independent_check(self, packet: WorkPacket, workspace) -> dict:
        """Run the packet's validation commands ourselves.

        A worker's ``tests_passed: true`` is a claim. This is the evidence.
        """
        checks = []
        for command_id in packet.validation_command_ids:
            spec = packet.command_map().get(command_id)
            if spec is None:
                continue
            result = run_command(
                list(spec.argv), cwd=workspace.root, env=build_child_env(),
                timeout_s=min(spec.timeout_s, self.config.limits.command_timeout_s),
            )
            checks.append({
                "command_id": command_id,
                "argv": list(spec.argv),
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "passed": result.ok,
                "tail": result.stdout[-3000:] + ("\n" + result.stderr[-1500:] if result.stderr else ""),
            })
            self.logger.info(
                "controller",
                f"independent check {command_id} for {packet.task_id}: "
                f"{'passed' if result.ok else 'FAILED'}",
            )
        return {"checks": checks, "all_passed": all(c["passed"] for c in checks) if checks else None}

    # -- the run ------------------------------------------------------------

    def load_packets(self, path: Path) -> "list[PlannedPacket]":
        """Read a controller-authored decomposition.

        The same schema the planner emits, so a controller can hand-write a
        decomposition, or edit one Fable produced, without involving the planner
        at all. These are still only *proposals*: they go through authorise()
        exactly like a planner's, so commands and read-only rules are applied by
        the controller either way.
        """
        import json as _json

        try:
            data = _json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read packets file {path}: {exc}") from exc
        entries = data.get("packets") if isinstance(data, dict) else data
        if not isinstance(entries, list) or not entries:
            raise ConfigError(f"{path} contains no packets")

        proposals: list[PlannedPacket] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ConfigError(f"packet {index} in {path} is not an object")
            try:
                kind = TaskKind(str(entry.get("kind", "implement")).lower())
            except ValueError as exc:
                raise ConfigError(
                    f"packet {index} has unknown kind {entry.get('kind')!r}") from exc
            proposals.append(PlannedPacket(
                task_id=str(entry.get("task_id") or f"task_{index + 1:02d}"),
                kind=kind,
                objective=str(entry.get("objective", "")),
                owned_paths=() if kind.read_only else tuple(entry.get("owned_paths", ())),
                readonly_paths=tuple(entry.get("readonly_paths", ())),
                acceptance_criteria=tuple(entry.get("acceptance_criteria", ())),
                depends_on=tuple(entry.get("depends_on", ())),
                parallel_safe=bool(entry.get("parallel_safe", True)),
                rationale=str(entry.get("context") or entry.get("rationale", "")),
            ))
        return proposals

    def run(self, *, task: str, constraints: str = "", focus_paths=(),
            use_planner: bool = True, dry_run: bool = False,
            final_review: bool = False, base_rev: str = "HEAD",
            packets_file: "Path | None" = None) -> RunOutcome:
        outcome = RunOutcome(run_id=self.run_id, ok=False, task=task,
                             ledger_root=str(self.ledger.root), dry_run=dry_run)
        try:
            models = self.resolve_models()
        except ModelResolutionError as exc:
            self.logger.error("controller", exc.message)
            outcome.error = exc.as_dict()
            self.ledger.save_manifest(outcome.as_dict())
            return outcome

        outcome.models = {role: model.as_dict() for role, model in models.items()}
        context = self.build_context(task, constraints, focus_paths=focus_paths)
        commands = detect_validation_commands(self.repo_root) + list(self.config.command_specs())
        state = git_state(self.repo_root)

        if dry_run:
            return self._dry_run(outcome, models, context, commands, task, use_planner,
                                 packets_file=packets_file)

        from .providers import get_provider

        planner = Planner(
            provider=get_provider(models[Role.PLANNER].provider, self.config),
            model=models[Role.PLANNER],
            logger=self.logger,
            cache=self.cache,
            version=__version__,
            timeout_s=self.config.limits.planner_timeout_s,
            max_rounds=self.config.limits.max_planner_rounds,
            plan_cache_ttl_s=self.config.plan_cache_ttl_s,
        )

        try:
            if packets_file is not None:
                proposals = self.load_packets(packets_file)
                self.logger.info(
                    "planner",
                    f"skipped: using {len(proposals)} controller-authored packet(s) "
                    f"from {packets_file}",
                )
                plan = Plan(
                    approach=f"Controller-authored decomposition from {packets_file}.",
                    packets=tuple(proposals),
                    verification_strategy="the controller's own validation commands",
                )
            elif use_planner:
                plan = planner.plan(
                    task=task, constraints=constraints, context=context,
                    repo_identity=repo_identity(self.repo_root), git_state=state,
                    max_packets=self.config.limits.max_total_tasks,
                )
            else:
                self.logger.info("planner", "skipped: the controller is planning directly")
                plan = self._fallback_plan(task)
            outcome.plan = plan.as_dict()
            self.ledger.write_json("plan.json", plan.as_dict())

            packets = self.authorise(list(plan.packets), commands=commands)
            outcome.packets = [p.as_dict() for p in packets]
            for packet in packets:
                self.ledger.save_packet(packet)
            self.logger.info("controller", f"authorised {len(packets)} worker packet(s)")

            results = self.execute(packets, models[Role.WORKER], base_rev=base_rev)
            outcome.results = results
            for envelope in results.values():
                self.ledger.save_result(envelope)
            outcome.verification = getattr(self, "_verification", {})

            decision = should_escalate(
                results=results, plan=plan,
                rounds_used=planner.rounds_used,
                max_rounds=self.config.limits.max_planner_rounds,
                repeated_failures=sum(1 for e in results.values()
                                      if e.status is TaskStatus.FAILED),
                final_review_requested=final_review,
            )
            if decision.escalate and use_planner:
                self.logger.info("critic", f"escalating: {decision.reason}")
                critique = planner.critique(
                    material=self._critique_material(plan, results, outcome.verification),
                    question=decision.reason,
                )
                outcome.critique = critique.as_dict()
                self.ledger.write_json("critique.json", critique.as_dict())
            else:
                self.logger.info("critic", f"review skipped: {decision.reason}")
                outcome.critique = {"skipped": True, "reason": decision.reason}

            succeeded = [e for e in results.values() if e.succeeded]
            outcome.ok = bool(succeeded) and len(succeeded) == len(results)
            outcome.stats = {
                "planner": planner.stats(),
                "workers": {
                    "total": len(results),
                    "completed": len(succeeded),
                    "failed": sum(1 for e in results.values() if e.status is TaskStatus.FAILED),
                    "blocked": sum(1 for e in results.values() if e.status is TaskStatus.BLOCKED),
                    "skipped": sum(1 for e in results.values() if e.status is TaskStatus.SKIPPED),
                    "denied_actions": sum(len(e.denied_actions) for e in results.values()),
                },
                "cache": self.cache.stats(),
                "context": context.summary(),
                "wall_clock_s": round(time.monotonic() - self._started, 2),
            }
        except FabdsError as exc:
            self.logger.error("controller", f"{exc.code}: {exc.message}")
            outcome.error = exc.as_dict()
        finally:
            self.ledger.save_manifest(outcome.as_dict())
        return outcome

    def _fallback_plan(self, task: str) -> Plan:
        """The controller's own single-packet decomposition.

        Used with ``--no-planner`` and in dry runs. Deliberately minimal: when
        the planner is not consulted, the controller does not pretend to have a
        decomposition it did not derive.
        """
        return Plan(
            approach="Controller-directed: a single bounded packet, no planner call.",
            packets=(PlannedPacket(
                task_id="task_01",
                kind=TaskKind.IMPLEMENT,
                objective=task,
                owned_paths=("**/*",),
                acceptance_criteria=("the stated objective is met and validation commands pass",),
                rationale="no planner round was used",
            ),),
            verification_strategy="run the repository's own validation commands",
        )

    def _critique_material(self, plan: Plan, results: dict, verification: dict) -> str:
        parts = ["## Plan", plan.approach, "", "## Packets and outcomes"]
        for task_id, envelope in results.items():
            parts.append(
                f"- {task_id}: status={envelope.status.value}, "
                f"observed_changes={list(envelope.observed_files_changed)[:10]}, "
                f"claimed_tests_passed={envelope.tests_passed}, "
                f"denied_actions={len(envelope.denied_actions)}"
            )
            if envelope.summary:
                parts.append(f"    summary: {envelope.summary[:500]}")
            if envelope.risks:
                parts.append(f"    risks: {list(envelope.risks)[:5]}")
        parts += ["", "## Independent verification by the controller"]
        for task_id, record in verification.items():
            for check in record.get("checks", []):
                parts.append(
                    f"- {task_id} {check['command_id']}: "
                    f"{'passed' if check['passed'] else 'FAILED'} (exit {check['returncode']})"
                )
        return "\n".join(parts)

    # -- dry run ------------------------------------------------------------

    def _dry_run(self, outcome: RunOutcome, models: dict, context: ContextPacket,
                 commands: "list[CommandSpec]", task: str, use_planner: bool,
                 packets_file: "Path | None" = None) -> RunOutcome:
        if packets_file is not None:
            plan = Plan(approach=f"Controller-authored decomposition from {packets_file}.",
                        packets=tuple(self.load_packets(packets_file)))
        else:
            plan = self._fallback_plan(task)
        packets = self.authorise(list(plan.packets), commands=commands)
        outcome.ok = True
        if packets_file is not None:
            note = f"No model was called. These are your own packets from {packets_file}."
        elif use_planner:
            note = ("No model was called. With the planner enabled the real packets come "
                    "from the planner's response; the packet below is the controller's own "
                    "fallback decomposition.")
        else:
            note = "The planner is disabled for this run; this is the packet that would run."
        outcome.plan = {
            "note": note,
            **plan.as_dict(),
        }
        outcome.packets = [p.as_dict() for p in packets]
        outcome.stats = {
            "would_call": {
                "planner": {
                    "model": models[Role.PLANNER].model_id,
                    "provider": models[Role.PLANNER].provider,
                    "context_chars": context.size,
                    "context_digest": context.digest(),
                    "isolation": "no MCP servers, no tools, no settings, neutral cwd",
                } if use_planner else None,
                "workers": {
                    "model": models[Role.WORKER].model_id,
                    "provider": models[Role.WORKER].provider,
                    "max_concurrent": self.config.limits.max_workers,
                },
            },
            "context": context.summary(),
            "commands_offered": [c.as_dict() for c in commands],
            "workspace_strategy": (
                "git worktree per writing worker"
                if supports_worktrees(self.repo_root) else
                "copy workspace per writing worker (target is not a git repository)"
            ),
            "limits": {
                "max_workers": self.config.limits.max_workers,
                "max_total_tasks": self.config.limits.max_total_tasks,
                "max_planner_rounds": self.config.limits.max_planner_rounds,
                "max_worker_turns": self.config.limits.max_worker_turns,
                "max_retries_per_task": self.config.limits.max_retries_per_task,
                "worker_nesting_enabled": self.config.limits.worker_nesting_enabled,
            },
        }
        self.ledger.write_text("dry-run-planner-context.txt", context.render())
        self.ledger.save_manifest(outcome.as_dict())
        self.logger.ok("controller", "dry run complete; nothing was called and nothing changed")
        return outcome

    # -- integration --------------------------------------------------------

    def verify(self, *, commands: "list[CommandSpec] | None" = None) -> dict:
        """Run the repository's validation commands against the working tree.

        Packets are verified in isolation, which is necessary but not
        sufficient: a test packet written in one worktree cannot see an
        implementation written in another, and two patches that each apply
        cleanly can still be wrong together. This is the check that covers the
        integrated result, and it is the one that actually decides whether the
        task is done.
        """
        commands = commands or detect_validation_commands(self.repo_root)
        wanted = {"pytest", "unittest", "npm_test", "make_test"}
        checks = []
        for spec in commands:
            if spec.id not in wanted:
                continue
            result = run_command(
                list(spec.argv), cwd=self.repo_root, env=build_child_env(),
                timeout_s=min(spec.timeout_s, self.config.limits.command_timeout_s),
            )
            checks.append({
                "command_id": spec.id, "argv": list(spec.argv),
                "returncode": result.returncode, "timed_out": result.timed_out,
                "passed": result.ok,
                "tail": result.stdout[-3000:] + ("\n" + result.stderr[-1500:] if result.stderr else ""),
            })
            self.logger.info(
                "controller",
                f"integrated check {spec.id}: {'passed' if result.ok else 'FAILED'}",
            )
        report = {"checks": checks,
                  "all_passed": all(c["passed"] for c in checks) if checks else None}
        if not checks:
            self.logger.warn("controller",
                             "no validation command was available; nothing was verified")
        self.ledger.write_json("verification.json", report)
        return report

    def integrate(self, task_ids: "list[str]", *, check_only: bool = False,
                  verify: bool = False) -> dict:
        """Apply staged patches to the repository. Explicit, never automatic."""
        report = {"applied": [], "rejected": [], "check_only": check_only}
        for task_id in task_ids:
            patch_path = self.ledger.patch_path(task_id)
            if not patch_path.is_file():
                report["rejected"].append({"task_id": task_id, "reason": "no staged patch"})
                continue
            argv = ["git", "-C", str(self.repo_root), "apply",
                    "--check" if check_only else "--index", "--whitespace=nowarn",
                    str(patch_path)]
            result = subprocess.run(  # noqa: S603 - argv list, shell=False
                argv, capture_output=True, text=True, timeout=300, check=False
            )
            if result.returncode == 0:
                report["applied"].append({"task_id": task_id, "patch": str(patch_path)})
                self.logger.ok("controller",
                               f"{'would apply' if check_only else 'applied'} patch for {task_id}")
            else:
                report["rejected"].append({
                    "task_id": task_id, "reason": result.stderr.strip()[:500],
                })
                self.logger.warn("controller",
                                 f"patch for {task_id} does not apply: {result.stderr.strip()[:200]}")
        if verify and not check_only and report["applied"]:
            report["verification"] = self.verify()
        self.ledger.write_json("integration.json", report)
        return report
