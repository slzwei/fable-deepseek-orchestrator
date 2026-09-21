#!/usr/bin/env python3
"""Compare three orchestration architectures on the same task.

    A  one expensive generalist model does everything, no delegation
    B  the expensive model is consulted before every packet AND implements them
    C  one architecture call, then a swarm of cheap workers, controller verifies

Every arm runs the same task against a fresh copy of the same repository,
through the same worker action loop, with the same limits. The only variables
are how many expensive calls happen and who does the implementation.

Honesty notes, which the report repeats:

* The planner role is filled by whichever model ``--expensive-model`` names.
  When the Fable quota is exhausted this will not be Fable, and the run says so.
  That changes the quality of planning, not the shape of the architectures, so
  call counts, wall time, context volume and duplicated work remain comparable.
* Cost is taken from the provider where the provider reports it (the Claude CLI
  does) and left as token counts where it does not (DeepSeek). Nothing here
  invents a price.
* One sample per arm. These are indicative, not statistics.

Usage:
    python3 bench/benchmark.py --out bench/results.json
    python3 bench/benchmark.py --arms C --expensive-model claude-opus-5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabds import __version__                                    # noqa: E402
from fabds.config import load_config                             # noqa: E402
from fabds.logging import Level, RunLogger                       # noqa: E402
from fabds.models import ResolvedModel, Role                     # noqa: E402
from fabds.orchestrator import Orchestrator, detect_validation_commands  # noqa: E402
from fabds.packets import TaskKind                               # noqa: E402
from fabds.planner import Plan, PlannedPacket                    # noqa: E402
from fabds.providers import get_provider                         # noqa: E402
from fabds.providers.base import CompletionRequest               # noqa: E402

TASK = "Implement readability_score(text) per README.md and add tests for it."

PACKETS = [
    PlannedPacket(
        task_id="impl_readability", kind=TaskKind.IMPLEMENT,
        objective=("Implement readability_score(text) in src/textstats/core.py and export "
                   "it from src/textstats/__init__.py so src/textstats/cli.py imports."),
        owned_paths=("src/textstats/core.py", "src/textstats/__init__.py"),
        readonly_paths=("README.md", "src/**", "tests/**"),
        acceptance_criteria=(
            "readability_score returns the Flesch score from README.md",
            "empty text returns 0.0 and never raises ZeroDivisionError",
            "word_count and char_frequency are unchanged",
        ),
        rationale="Read README.md for the specification.",
    ),
    PlannedPacket(
        task_id="test_readability", kind=TaskKind.TEST,
        objective="Write adversarial unittest tests in tests/test_readability.py.",
        owned_paths=("tests/test_readability.py",),
        readonly_paths=("README.md", "src/**", "tests/**"),
        acceptance_criteria=(
            "runs under python3 -m unittest discover -s tests -t tests",
            "covers empty text, one word, several sentences, no terminator",
        ),
        rationale="Import lazily inside test methods; the implementation may not exist yet.",
    ),
    PlannedPacket(
        task_id="audit_spec", kind=TaskKind.AUDIT,
        objective="Review README.md and src/ for ambiguities in the readability spec.",
        readonly_paths=("**",),
        acceptance_criteria=("concrete ambiguities are listed under risks",),
        rationale="You are read-only.",
    ),
]

WHOLE_TASK_PACKET = PlannedPacket(
    task_id="everything", kind=TaskKind.IMPLEMENT,
    objective=(
        "Implement readability_score(text) in src/textstats/core.py per README.md, export "
        "it from src/textstats/__init__.py, and add adversarial tests in "
        "tests/test_readability.py that run under unittest discovery."
    ),
    owned_paths=("src/**", "tests/**"),
    readonly_paths=("**",),
    acceptance_criteria=(
        "readability_score returns the Flesch score from README.md",
        "empty text returns 0.0 and never raises",
        "tests/test_readability.py exists and passes",
        "the existing tests still pass",
    ),
    rationale="Do the whole task yourself. There is no one to delegate to.",
)


@dataclass
class ArmResult:
    arm: str
    label: str
    planner_calls: int = 0
    worker_calls: int = 0
    planner_model: str = ""
    worker_model: str = ""
    wall_clock_s: float = 0.0
    prompt_chars_sent: int = 0
    packets: int = 0
    packets_completed: int = 0
    files_touched: list = field(default_factory=list)
    duplicated_files: list = field(default_factory=list)
    cost_usd_reported: float = 0.0
    worker_tokens: dict = field(default_factory=dict)
    integrated_tests_passed: bool | None = None
    task_completed: bool = False
    notes: list = field(default_factory=list)


class CountingProvider:
    """Wraps a real provider and records what actually went over the wire."""

    def __init__(self, inner, sink: ArmResult, role: str):
        self.inner = inner
        self.name = inner.name
        self.sink = sink
        self.role = role

    def status(self):
        return self.inner.status()

    def discover_models(self):
        return self.inner.discover_models()

    def complete(self, request: CompletionRequest):
        self.sink.prompt_chars_sent += len(request.system_prompt) + len(request.user_prompt)
        if self.role == "planner":
            self.sink.planner_calls += 1
        else:
            self.sink.worker_calls += 1
        response = self.inner.complete(request)
        self.sink.cost_usd_reported += response.cost_usd or 0.0
        for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
            value = response.usage.get(key)
            if isinstance(value, int):
                self.sink.worker_tokens[f"{self.role}.{key}"] = (
                    self.sink.worker_tokens.get(f"{self.role}.{key}", 0) + value)
        return response


def fresh_repo(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        ".git", ".fabds", "__pycache__", "*.pyc"))
    for argv in (["git", "init", "-q", "-b", "main"],
                 ["git", "config", "user.email", "bench@example.invalid"],
                 ["git", "config", "user.name", "Bench"],
                 ["git", "add", "-A"],
                 ["git", "commit", "-q", "-m", "baseline"]):
        subprocess.run(argv, cwd=destination, check=True, capture_output=True)
    return destination


def run_arm(arm: str, label: str, repo: Path, *, expensive_model: str, cheap: bool,
            packets: list, plan_per_packet: bool, verbose: bool) -> ArmResult:
    result = ArmResult(arm=arm, label=label, packets=len(packets))
    config = load_config(repo)
    logger = RunLogger(level=Level.VERBOSE if verbose else Level.QUIET)
    orchestrator = Orchestrator(config, repo, logger=logger)

    planner = orchestrator.resolver.resolve(Role.PLANNER)
    if planner.model_id != expensive_model:
        planner = ResolvedModel(
            role="planner", display_name=f"expensive stand-in ({expensive_model})",
            model_id=expensive_model, provider="claude_cli",
            pattern=expensive_model.replace(".", r"\."), source="benchmark override",
            evidence=f"benchmark ran with --expensive-model {expensive_model}",
        )
        result.notes.append(
            f"planner role filled by {expensive_model}, not Fable 5.1 "
            "(quota exhausted at benchmark time)")
    result.planner_model = planner.model_id

    if cheap:
        worker = orchestrator.resolver.resolve(Role.WORKER)
    else:
        worker = ResolvedModel(
            role="worker", display_name=f"expensive worker ({expensive_model})",
            model_id=expensive_model, provider="claude_cli",
            pattern=expensive_model.replace(".", r"\."), source="benchmark override",
            evidence="arm uses the expensive model for implementation",
        )
    result.worker_model = worker.model_id

    planner_provider = CountingProvider(get_provider(planner.provider, config), result, "planner")
    worker_provider = CountingProvider(get_provider(worker.provider, config), result, "worker")

    original = None
    import fabds.providers as providers_module
    original = providers_module.get_provider
    providers_module.get_provider = lambda name, cfg: (
        worker_provider if name == worker.provider else planner_provider)

    started = time.monotonic()
    try:
        # Architecture calls: once for C, once per packet for B, never for A.
        rounds = len(packets) if plan_per_packet else (1 if arm != "A" else 0)
        for index in range(rounds):
            context = orchestrator.build_context(TASK, "")
            try:
                planner_provider.complete(CompletionRequest(
                    system_prompt="You are an architecture planner. Answer concisely in JSON.",
                    user_prompt=(f"Packet {index + 1} of {rounds}.\n" if plan_per_packet else "")
                                + context.render()[:20000]
                                + '\n\nRespond with {"approach": "...", "risks": ["..."]}',
                    model_id=planner.model_id, max_output_tokens=1500, timeout_s=300,
                    label="plan",
                ))
            except Exception as exc:  # a blocked planner is a real, reportable outcome
                result.notes.append(f"planner call failed: {type(exc).__name__}: {exc}")
                break

        plan = Plan(approach=f"benchmark arm {arm}", packets=tuple(packets))
        commands = detect_validation_commands(repo)
        authorised = orchestrator.authorise(list(plan.packets), commands=commands)
        envelopes = orchestrator.execute(authorised, worker)

        result.packets_completed = sum(1 for e in envelopes.values() if e.succeeded)
        seen: dict[str, int] = {}
        for envelope in envelopes.values():
            for path in envelope.observed_files_changed:
                seen[path] = seen.get(path, 0) + 1
        result.files_touched = sorted(seen)
        result.duplicated_files = sorted(p for p, n in seen.items() if n > 1)

        accepted = [e.task_id for e in envelopes.values() if e.succeeded]
        orchestrator.integrate(accepted)
        verification = orchestrator.verify(commands=commands)
        result.integrated_tests_passed = verification["all_passed"]
        result.task_completed = bool(verification["all_passed"]) and _cli_imports(repo)
    finally:
        providers_module.get_provider = original
        result.wall_clock_s = round(time.monotonic() - started, 1)
    return result


def _cli_imports(repo: Path) -> bool:
    """The real acceptance test: does the previously broken CLI work?"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src'); "
         "from textstats.cli import main; main(['a b. c!'])"],
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    return result.returncode == 0


def render(results: list[ArmResult], expensive_model: str) -> str:
    lines = [
        f"# Orchestration benchmark (fabds {__version__})", "",
        "Same task, same repository, same worker action loop, same limits.",
        f"Expensive-model stand-in: `{expensive_model}`.", "",
        "| | A: one expensive agent | B: expensive every step | C: architecture + cheap swarm |",
        "|---|---|---|---|",
    ]
    rows = [
        ("planner / architecture calls", lambda r: r.planner_calls),
        ("worker model calls", lambda r: r.worker_calls),
        ("worker model", lambda r: f"`{r.worker_model}`"),
        ("wall clock (s)", lambda r: r.wall_clock_s),
        ("prompt chars sent", lambda r: f"{r.prompt_chars_sent:,}"),
        ("packets completed", lambda r: f"{r.packets_completed}/{r.packets}"),
        ("files touched by >1 worker", lambda r: len(r.duplicated_files)),
        ("expensive cost reported (USD)", lambda r: f"{r.cost_usd_reported:.4f}"),
        ("integrated suite passes", lambda r: r.integrated_tests_passed),
        ("**task actually completed**", lambda r: "yes" if r.task_completed else "**NO**"),
    ]
    by_arm = {r.arm: r for r in results}
    for label, getter in rows:
        cells = []
        for arm in ("A", "B", "C"):
            cells.append(str(getter(by_arm[arm])) if arm in by_arm else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Notes", ""]
    for result in results:
        for note in result.notes:
            lines.append(f"- **{result.arm}**: {note}")
    lines += [
        "- \"integrated suite passes\" is necessary but not sufficient: a run that "
        "changes nothing also leaves the original suite green. \"task actually "
        "completed\" is the real measure - it imports the CLI that was broken at "
        "baseline and requires it to work.",
        "- One sample per arm. Indicative, not statistics.",
        "- Cost is what the provider reported. DeepSeek does not report a price, so its "
        "usage appears as token counts in the JSON rather than as dollars.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="source repository to copy per arm")
    parser.add_argument("--workdir", default="/tmp/fabds-bench")
    parser.add_argument("--out", default="bench/results.json")
    parser.add_argument("--expensive-model", default="claude-opus-5")
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.repo:
        parser.error("--repo is required: point it at a baseline repository to copy")
    source = Path(args.repo).expanduser().resolve()
    workdir = Path(args.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    arms = {
        "A": ("one expensive agent, no delegation", [WHOLE_TASK_PACKET], False, False),
        "B": ("expensive model plans and implements every packet", PACKETS, False, True),
        "C": ("architecture once, then a cheap worker swarm", PACKETS, True, False),
    }

    results: list[ArmResult] = []
    for arm in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
        label, packets, cheap, plan_per_packet = arms[arm]
        print(f"[bench] arm {arm}: {label}", file=sys.stderr)
        repo = fresh_repo(source, workdir / f"arm-{arm}")
        result = run_arm(arm, label, repo, expensive_model=args.expensive_model,
                         cheap=cheap, packets=list(packets),
                         plan_per_packet=plan_per_packet, verbose=args.verbose)
        results.append(result)
        print(f"[bench] arm {arm}: completed={result.task_completed} "
              f"planner={result.planner_calls} worker={result.worker_calls} "
              f"{result.wall_clock_s}s", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fabds_version": __version__,
        "expensive_model": args.expensive_model,
        "task": TASK,
        "arms": [asdict(r) for r in results],
    }, indent=2), encoding="utf-8")
    report = render(results, args.expensive_model)
    out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
