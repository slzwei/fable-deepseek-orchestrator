"""Command line interface.

Subcommands
-----------

``doctor``     verify the environment and the security properties
``resolve``    show which concrete models each role resolves to, with evidence
``plan``       one planning round, no workers, no file changes
``run``        the full orchestration; stages patches, never merges them
``integrate``  apply staged patches the controller has decided to accept
``runs``       list previous runs in this repository
``cache``      inspect, prune or purge the caches
``version``    print the version

Exit codes: ``0`` success, ``1`` the run failed, ``2`` a usage or configuration
error, ``3`` a required model could not be resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .cache import FileCache
from .config import load_config
from .errors import ConfigError, FabdsError, ModelResolutionError
from .ledger import RunLedger
from .logging import Level, RunLogger
from .models import ModelResolver, Role

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NO_MODEL = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabds",
        description="Fable plans, DeepSeek executes, the controller decides.",
    )
    parser.add_argument("--version", action="version", version=f"fabds {__version__}")
    parser.add_argument("-C", "--repo", default=".", metavar="PATH",
                        help="repository to operate on (default: the current directory)")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="count", default=0,
                           help="more detail; repeat for debug output")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="errors only")
    parser.add_argument("--json", action="store_true", help="machine-readable output on stdout")
    parser.add_argument("--no-cache", action="store_true", help="bypass the plan and analysis caches")

    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the environment and security properties")
    doctor.add_argument("--quick", action="store_true",
                        help="skip the live MCP isolation probe")

    resolve = sub.add_parser("resolve", help="show the resolved model for each role")
    resolve.add_argument("--refresh", action="store_true", help="ignore the resolution cache")

    for name, help_text in (("run", "plan, delegate, verify"), ("plan", "plan only")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("task", help="what you want done")
        command.add_argument("--constraints", default="", help="constraints the plan must respect")
        command.add_argument("--focus", action="append", default=[], metavar="PATH",
                             help="a file the planner should see (repeatable)")
        command.add_argument("--max-workers", type=int, default=None)
        command.add_argument("--max-tasks", type=int, default=None)
        command.add_argument("--base-rev", default="HEAD",
                             help="revision each worker workspace starts from")
        if name == "run":
            command.add_argument("--dry-run", action="store_true",
                                 help="show what would happen; call nothing, change nothing")
            command.add_argument("--no-planner", action="store_true",
                                 help="skip the planner; the controller decomposes directly")
            command.add_argument("--final-review", action="store_true",
                                 help="always run a closing adversarial review")

    integrate = sub.add_parser("integrate", help="apply staged patches")
    integrate.add_argument("task_ids", nargs="+", metavar="TASK_ID")
    integrate.add_argument("--run", dest="run_id", default=None, help="run id (default: the latest)")
    integrate.add_argument("--check", action="store_true", help="test whether patches apply")

    sub.add_parser("runs", help="list runs recorded in this repository")

    cache = sub.add_parser("cache", help="inspect or clear the caches")
    cache.add_argument("action", choices=("stats", "prune", "purge"))
    cache.add_argument("--namespace", choices=("plan", "analysis"), default=None)

    return parser


def _level(args) -> Level:
    if args.quiet:
        return Level.QUIET
    return {0: Level.NORMAL, 1: Level.VERBOSE}.get(args.verbose, Level.DEBUG)


def _config(args):
    repo = Path(args.repo).expanduser().resolve(strict=False)
    overrides: dict = {}
    if getattr(args, "no_cache", False):
        overrides["cache_enabled"] = False
    limits = {}
    if getattr(args, "max_workers", None):
        limits["max_workers"] = args.max_workers
    if getattr(args, "max_tasks", None):
        limits["max_total_tasks"] = args.max_tasks
    if limits:
        overrides["limits"] = limits
    config = load_config(repo, overrides)
    if config.deepseek_api_key_file is None:
        # Fall back to the DeepSeek MCP server's configured key file when the
        # user already has one; still a path, never a value.
        default = Path.home() / "quantlab" / "ds-api-key"
        if default.is_file():
            from dataclasses import replace

            config = replace(config, deepseek_api_key_file=default)
    return config, repo


def _emit(args, payload: dict, text_lines: "list[str]") -> None:
    if args.json:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        for line in text_lines:
            print(line)


# -- subcommands ------------------------------------------------------------

def cmd_doctor(args) -> int:
    from .doctor import run_doctor

    config, repo = _config(args)
    report = run_doctor(config, repo, quick=args.quick)
    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(report.render(verbose=args.verbose > 0))
    return EXIT_OK if report.ok else EXIT_FAILED


def cmd_resolve(args) -> int:
    config, _ = _config(args)
    resolver = ModelResolver(config)
    payload, lines, exit_code = {}, [], EXIT_OK
    for role in (Role.PLANNER, Role.WORKER):
        try:
            model = resolver.resolve(role, use_cache=not args.refresh)
            payload[role] = model.as_dict()
            lines.append(model.describe())
            lines.append(f"    evidence: {model.evidence}")
        except ModelResolutionError as exc:
            payload[role] = {"error": exc.as_dict()}
            lines.append(f"{role}: UNRESOLVED")
            lines.extend("    " + line for line in exc.message.splitlines())
            exit_code = EXIT_NO_MODEL
    _emit(args, payload, lines)
    return exit_code


def _run_common(args, *, plan_only: bool) -> int:
    from .orchestrator import Orchestrator

    config, repo = _config(args)
    logger = RunLogger(level=_level(args))
    orchestrator = Orchestrator(config, repo, logger=logger)
    logger.attach_jsonl(orchestrator.ledger.events_path)

    outcome = orchestrator.run(
        task=args.task,
        constraints=args.constraints,
        focus_paths=tuple(args.focus),
        use_planner=not getattr(args, "no_planner", False),
        dry_run=getattr(args, "dry_run", False) or plan_only,
        final_review=getattr(args, "final_review", False),
        base_rev=args.base_rev,
    ) if not plan_only else _plan_only(orchestrator, args)

    if args.json:
        json.dump(outcome.as_dict(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        for line in outcome.summary_lines():
            print(line)
        print(f"  ledger: {outcome.ledger_root}")
    if outcome.error and outcome.error.get("code") == "model_resolution_failed":
        return EXIT_NO_MODEL
    return EXIT_OK if outcome.ok else EXIT_FAILED


def _plan_only(orchestrator, args):
    """One planning round with no workers and no file changes."""
    from .models import Role as _Role
    from .orchestrator import RunOutcome, detect_validation_commands
    from .planner import Planner
    from .providers import get_provider
    from .context import git_state, repo_identity

    outcome = RunOutcome(run_id=orchestrator.run_id, ok=False, task=args.task,
                         ledger_root=str(orchestrator.ledger.root))
    try:
        models = orchestrator.resolve_models()
    except ModelResolutionError as exc:
        outcome.error = exc.as_dict()
        return outcome
    outcome.models = {role: model.as_dict() for role, model in models.items()}
    context = orchestrator.build_context(args.task, args.constraints, focus_paths=tuple(args.focus))
    planner = Planner(
        provider=get_provider(models[_Role.PLANNER].provider, orchestrator.config),
        model=models[_Role.PLANNER], logger=orchestrator.logger, cache=orchestrator.cache,
        version=__version__, timeout_s=orchestrator.config.limits.planner_timeout_s,
        max_rounds=orchestrator.config.limits.max_planner_rounds,
        plan_cache_ttl_s=orchestrator.config.plan_cache_ttl_s,
    )
    try:
        plan = planner.plan(
            task=args.task, constraints=args.constraints, context=context,
            repo_identity=repo_identity(orchestrator.repo_root),
            git_state=git_state(orchestrator.repo_root),
            max_packets=orchestrator.config.limits.max_total_tasks,
        )
        outcome.plan = plan.as_dict()
        outcome.stats = {"planner": planner.stats(), "context": context.summary()}
        outcome.ok = True
        orchestrator.ledger.write_json("plan.json", plan.as_dict())
    except FabdsError as exc:
        outcome.error = exc.as_dict()
        orchestrator.logger.error("planner", f"{exc.code}: {exc.message}")
    orchestrator.ledger.save_manifest(outcome.as_dict())
    return outcome


def cmd_run(args) -> int:
    return _run_common(args, plan_only=False)


def cmd_plan(args) -> int:
    return _run_common(args, plan_only=True)


def cmd_integrate(args) -> int:
    from .orchestrator import Orchestrator

    config, repo = _config(args)
    run_id = args.run_id or (RunLedger.list_runs(repo) or [None])[0]
    if run_id is None:
        print("no runs found in this repository", file=sys.stderr)
        return EXIT_USAGE
    orchestrator = Orchestrator(config, repo, logger=RunLogger(level=_level(args)), run_id=run_id)
    report = orchestrator.integrate(args.task_ids, check_only=args.check)
    lines = [f"run {run_id}"]
    lines += [f"  applied  {entry['task_id']}" for entry in report["applied"]]
    lines += [f"  rejected {entry['task_id']}: {entry['reason']}" for entry in report["rejected"]]
    _emit(args, report, lines)
    return EXIT_OK if not report["rejected"] else EXIT_FAILED


def cmd_runs(args) -> int:
    _, repo = _config(args)
    runs = RunLedger.list_runs(repo)
    payload = []
    lines = []
    for run_id in runs[:50]:
        manifest = RunLedger(repo, run_id).load_manifest()
        payload.append({"run_id": run_id, "ok": manifest.get("ok"),
                        "task": manifest.get("task", "")[:80]})
        lines.append(f"  {run_id}  {'ok    ' if manifest.get('ok') else 'failed'}  "
                     f"{manifest.get('task', '')[:70]}")
    _emit(args, {"runs": payload}, lines or ["  (no runs recorded)"])
    return EXIT_OK


def cmd_cache(args) -> int:
    config, _ = _config(args)
    cache = FileCache(Path(config.cache_dir), enabled=True)
    if args.action == "stats":
        _emit(args, cache.stats(), [f"  {k}: {v}" for k, v in cache.stats().items()])
    elif args.action == "prune":
        removed = cache.prune_expired()
        _emit(args, {"pruned": removed}, [f"  pruned {removed} expired entry(ies)"])
    else:
        removed = cache.purge(args.namespace)
        _emit(args, {"purged": removed}, [f"  purged {removed} entry(ies)"])
    return EXIT_OK


COMMANDS = {
    "doctor": cmd_doctor, "resolve": cmd_resolve, "run": cmd_run, "plan": cmd_plan,
    "integrate": cmd_integrate, "runs": cmd_runs, "cache": cmd_cache,
}


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except ConfigError as exc:
        print(f"configuration error: {exc.message}", file=sys.stderr)
        return EXIT_USAGE
    except ModelResolutionError as exc:
        print(exc.message, file=sys.stderr)
        return EXIT_NO_MODEL
    except FabdsError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
