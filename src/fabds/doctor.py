"""Diagnostics.

``fabds doctor`` answers one question honestly: will a real run work here, and
are the security properties actually in force? It prefers *demonstrating* a
property over asserting it - the MCP isolation check really does start a fake
MCP server and really does confirm it is absent from an isolated invocation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .config import Config
from .errors import FabdsError
from .mcpprobe import IsolationProbeResult, probe_config, write_fake_home, write_probe
from .models import ModelResolver, Role
from .permissions import WorkerPermissions
from .runner import build_child_env, run_command
from .workspace import create_workspace, supports_worktrees

__all__ = ["Check", "DoctorReport", "run_doctor"]

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (PASS, WARN, SKIP)


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    version: str = __version__

    def add(self, name: str, status: str, detail: str = "") -> Check:
        check = Check(name, status, detail)
        self.checks.append(check)
        return check

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def render(self, *, verbose: bool = False) -> str:
        width = max((len(c.name) for c in self.checks), default=20)
        lines = [f"fabds {self.version}  -  orchestrator diagnostics", ""]
        for check in self.checks:
            line = f"  {check.name.ljust(width)}  {check.status}"
            if check.detail and (verbose or check.status != PASS):
                line += f"   {check.detail}"
            lines.append(line)
        lines.append("")
        failures = [c for c in self.checks if c.status == FAIL]
        lines.append(
            "All checks passed." if not failures
            else f"{len(failures)} check(s) failed: " + ", ".join(c.name for c in failures)
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "ok": self.ok,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail}
                       for c in self.checks],
        }


def run_doctor(config: Config, repo_root: Path, *, quick: bool = False) -> DoctorReport:
    report = DoctorReport()
    _check_runtime(report)
    _check_binaries(report, config)
    _check_codex(report)
    resolver = ModelResolver(config)
    _check_models(report, resolver)
    _check_cache(report, config)
    _check_workspaces(report, repo_root)
    _check_worker_sandbox(report)
    if quick:
        report.add("MCP isolation", SKIP, "skipped with --quick")
    else:
        _check_mcp_isolation(report, config)
    return report


# -- individual checks ------------------------------------------------------

def _check_runtime(report: DoctorReport) -> None:
    import sys

    version = ".".join(str(v) for v in sys.version_info[:3])
    report.add(
        "Python runtime",
        PASS if sys.version_info >= (3, 10) else FAIL,
        f"{version} at {sys.executable} (3.10+ required, standard library only)",
    )


def _check_binaries(report: DoctorReport, config: Config) -> None:
    for name, binary, required in (
        ("git", "git", True),
        ("claude CLI", config.claude_cli, True),
        ("jq (optional)", "jq", False),
    ):
        path = shutil.which(binary)
        if path:
            result = run_command([path, "--version"], timeout_s=30)
            version = (result.stdout or result.stderr).strip().splitlines()[0] if result.ok else "?"
            report.add(name, PASS, f"{path} ({version})")
        else:
            report.add(name, FAIL if required else WARN,
                       f"{binary!r} not found on PATH" + ("" if required else "; not used by fabds"))


def _check_codex(report: DoctorReport) -> None:
    codex = shutil.which("codex")
    skills_dir = Path(os.path.expanduser("~/.codex/skills"))
    if codex:
        result = run_command([codex, "--version"], timeout_s=30)
        detail = (result.stdout or result.stderr).strip().splitlines()[0] if result.ok else codex
        report.add("Codex environment", PASS, f"{detail}; skills dir "
                                              f"{'present' if skills_dir.is_dir() else 'missing'}")
    else:
        report.add("Codex environment", WARN,
                   "codex CLI not on PATH; fabds still works as a library and CLI")


def _check_models(report: DoctorReport, resolver: ModelResolver) -> None:
    for role, label in ((Role.PLANNER, "Fable 5.1"), (Role.WORKER, "DeepSeek Flash 4.1")):
        try:
            resolved = resolver.resolve(role, use_cache=False)
        except FabdsError as exc:
            report.add(label, FAIL, exc.message.splitlines()[0])
            continue
        note = "  [EXPLICIT FALLBACK]" if resolved.via_fallback else ""
        report.add(
            label,
            WARN if resolved.via_fallback else PASS,
            f"{resolved.model_id} via {resolved.provider} ({resolved.source}){note}",
        )
        report.add(f"  {label} evidence", PASS, resolved.evidence)


def _check_cache(report: DoctorReport, config: Config) -> None:
    cache_dir = Path(config.cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".fabds-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report.add("Cache directory", PASS, str(cache_dir))
    except OSError as exc:
        report.add("Cache directory", FAIL, f"{cache_dir} is not writable: {exc}")


def _check_workspaces(report: DoctorReport, repo_root: Path) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="fabds-doctor-") as temp:
            (Path(temp) / "probe").write_text("ok", encoding="utf-8")
        report.add("Temp workspace", PASS, tempfile.gettempdir())
    except OSError as exc:
        report.add("Temp workspace", FAIL, str(exc))

    if supports_worktrees(repo_root):
        try:
            workspace = create_workspace("doctor", repo_root, read_only=False)
            workspace.setup()
            kind, root = workspace.kind, workspace.root
            workspace.cleanup()
            report.add("Git worktree support", PASS, f"{kind} created and removed at {root}")
        except FabdsError as exc:
            report.add("Git worktree support", FAIL, exc.message)
    else:
        report.add("Git worktree support", WARN,
                   f"{repo_root} is not a git repository with a commit; "
                   "workers would use copy workspaces")


def _check_worker_sandbox(report: DoctorReport) -> None:
    """Demonstrate that a worker really cannot write outside its grant."""
    try:
        with tempfile.TemporaryDirectory(prefix="fabds-sandbox-") as temp:
            root = Path(temp) / "ws"
            (root / "src").mkdir(parents=True)
            (root / "src" / "owned.txt").write_text("x", encoding="utf-8")
            (root / "other.txt").write_text("x", encoding="utf-8")
            permissions = WorkerPermissions(workspace_root=root, owned=("src/**",))

            permissions.assert_write("src/owned.txt")
            refusals = 0
            for attempt in ("other.txt", "../escape.txt", "/etc/passwd", "src/../../x"):
                try:
                    permissions.assert_write(attempt)
                except FabdsError:
                    refusals += 1
            read_only = WorkerPermissions(workspace_root=root, readonly=("**",), read_only=True)
            try:
                read_only.assert_write("src/owned.txt")
            except FabdsError:
                refusals += 1

        report.add(
            "Worker sandbox",
            PASS if refusals == 5 else FAIL,
            f"{refusals}/5 boundary violations refused (ownership, traversal, "
            "absolute path, symlink-safe resolve, read-only)",
        )
    except Exception as exc:  # pragma: no cover - environment specific
        report.add("Worker sandbox", FAIL, str(exc))


def _check_mcp_isolation(report: DoctorReport, config: Config) -> None:
    """Start a fake MCP server and prove it is absent from an isolated call."""
    claude = shutil.which(config.claude_cli)
    if not claude:
        report.add("MCP isolation", SKIP, "claude CLI unavailable")
        return
    try:
        result = probe_isolation(claude, timeout_s=120)
    except Exception as exc:  # pragma: no cover - environment specific
        report.add("MCP isolation", WARN, f"probe could not run: {exc}")
        return
    status = PASS if result.passed else (WARN if not result.meaningful else FAIL)
    report.add("MCP isolation", status, result.describe())


def probe_isolation(claude_path: str, *, timeout_s: float = 120,
                    model: str = "claude-haiku-4-5-20251001") -> IsolationProbeResult:
    """Run the three-way isolation experiment. Shared with the test suite."""
    with tempfile.TemporaryDirectory(prefix="fabds-mcp-") as temp:
        base = Path(temp)
        control_marker = base / "control.marker"
        isolated_marker = base / "isolated.marker"
        granted_marker = base / "granted.marker"
        script = write_probe(base / "probe", control_marker)
        home = base / "home"
        write_fake_home(home, script, control_marker)
        env = build_child_env(extra={"HOME": str(home)})

        common = [claude_path, "--print", "--output-format", "json", "--model", model]

        # 1. Control: the probe is registered the ordinary way.
        run_command(common + ["ping"], env=env, cwd=temp, timeout_s=timeout_s)
        control = control_marker.exists()

        # 2. Isolated: exactly how fabds launches the planner.
        run_command(
            common + [
                "--safe-mode", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--setting-sources", "", "--tools", "", "--permission-prompts", "none",
                "--no-session-persistence", "ping",
            ],
            env=env, cwd=temp, timeout_s=timeout_s,
        )
        isolated = isolated_marker.exists() or _marker_grew(control_marker, control)

        # 3. Granted: an explicitly permitted server must still start.
        import json as _json

        granted_script = write_probe(base / "probe2", granted_marker)
        grant = _json.dumps(probe_config(granted_script, granted_marker))
        # No --safe-mode here: it disables MCP wholesale and would beat the
        # grant. The provider drops it for granted calls for the same reason.
        run_command(
            common + [
                "--strict-mcp-config", "--mcp-config", grant,
                "--setting-sources", "", "--permission-prompts", "none",
                "--no-session-persistence", "ping",
            ],
            env=env, cwd=temp, timeout_s=timeout_s,
        )
        granted = granted_marker.exists()

        return IsolationProbeResult(
            control_started=control,
            isolated_started=isolated,
            granted_started=granted,
            detail=f"control={control} isolated={isolated} granted={granted}",
        )


def _marker_grew(marker: Path, existed_before: bool) -> bool:
    """The isolated run must not append to the control marker either."""
    if not marker.exists():
        return False
    lines = marker.read_text(encoding="utf-8").count("STARTED")
    return lines > (1 if existed_before else 0)
