"""Dry run: show everything, call nothing, change nothing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FakeProvider
from fabds.models import ModelResolver
from fabds.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def dry_orchestrator(config, git_repo, null_logger):
    providers = {
        "claude_cli": FakeProvider("claude_cli", models=["claude-fable-5-1"]),
        "deepseek_http": FakeProvider("deepseek_http", models=["deepseek-flash"]),
    }
    resolver = ModelResolver(config, providers=providers,
                             cache_path=config.cache_dir / "r.json")
    orchestrator = Orchestrator(config, git_repo, logger=null_logger, resolver=resolver)
    return orchestrator, providers


def test_dry_run_calls_no_model(dry_orchestrator):
    orchestrator, providers = dry_orchestrator
    outcome = orchestrator.run(task="add retry logic", dry_run=True)
    assert outcome.ok is True
    assert outcome.dry_run is True
    for provider in providers.values():
        assert provider.requests == [], "a dry run must not call any model"


def test_dry_run_changes_no_file(dry_orchestrator, git_repo):
    orchestrator, _ = dry_orchestrator
    before = subprocess.run(["git", "status", "--porcelain"], cwd=git_repo,
                            capture_output=True, text=True).stdout
    orchestrator.run(task="add retry logic", dry_run=True)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=git_repo,
                           capture_output=True, text=True).stdout
    # Only the self-ignoring .fabds run directory may appear.
    new = set(after.splitlines()) - set(before.splitlines())
    assert all(".fabds" in line for line in new), f"dry run touched: {new}"
    assert (git_repo / ".fabds" / ".gitignore").is_file()


def test_dry_run_reports_what_would_be_called(dry_orchestrator):
    orchestrator, _ = dry_orchestrator
    outcome = orchestrator.run(task="add retry logic", dry_run=True)
    would = outcome.stats["would_call"]
    assert would["planner"]["model"] == "claude-fable-5-1"
    assert would["planner"]["provider"] == "claude_cli"
    assert would["planner"]["context_chars"] > 0
    assert "no MCP servers" in would["planner"]["isolation"]
    assert would["workers"]["model"] == "deepseek-flash"
    assert would["workers"]["max_concurrent"] >= 1


def test_dry_run_shows_packets_permissions_and_commands(dry_orchestrator):
    orchestrator, _ = dry_orchestrator
    outcome = orchestrator.run(task="add retry logic", dry_run=True)
    assert outcome.packets, "the packets that would run must be shown"
    packet = outcome.packets[0]
    assert "owned_paths" in packet and "read_only" in packet
    assert packet["commands"], "the command allowlist must be shown"
    for command in outcome.stats["commands_offered"]:
        assert isinstance(command["argv"], list), "commands are argv arrays, shown in full"
    assert "worktree" in outcome.stats["workspace_strategy"]


def test_dry_run_shows_every_limit(dry_orchestrator):
    orchestrator, _ = dry_orchestrator
    limits = orchestrator.run(task="x", dry_run=True).stats["limits"]
    for expected in ("max_workers", "max_total_tasks", "max_planner_rounds",
                     "max_worker_turns", "max_retries_per_task", "worker_nesting_enabled"):
        assert expected in limits
    assert limits["worker_nesting_enabled"] is False


def test_dry_run_writes_the_exact_planner_context(dry_orchestrator):
    orchestrator, _ = dry_orchestrator
    orchestrator.run(task="add retry logic to the client", dry_run=True)
    context = (orchestrator.ledger.root / "dry-run-planner-context.txt").read_text(
        encoding="utf-8")
    assert "add retry logic to the client" in context
    assert "Repository tree" in context


def test_dry_run_is_honest_about_the_fallback_decomposition(dry_orchestrator):
    """With the planner enabled, the real packets come from Fable. Say so."""
    orchestrator, _ = dry_orchestrator
    outcome = orchestrator.run(task="x", dry_run=True, use_planner=True)
    assert "fallback decomposition" in outcome.plan["note"]


def test_dry_run_still_fails_closed_on_an_unresolved_model(config, git_repo, null_logger):
    providers = {
        "claude_cli": FakeProvider("claude_cli", models=["claude-opus-5"]),
        "deepseek_http": FakeProvider("deepseek_http", models=["deepseek-flash"]),
    }
    resolver = ModelResolver(config, providers=providers, cache_path=config.cache_dir / "r.json")
    outcome = Orchestrator(config, git_repo, logger=null_logger, resolver=resolver).run(
        task="x", dry_run=True)
    assert outcome.ok is False
    assert outcome.error["code"] == "model_resolution_failed"


def test_cli_dry_run_end_to_end(git_repo, tmp_path):
    """Through the real CLI, with real providers available but never called."""
    import os

    env = {
        # The real PATH: this test is meant to exercise actual model resolution,
        # and skips only if that genuinely cannot resolve here.
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "PYTHONPATH": str(REPO / "src"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fabds", "-C", str(git_repo), "--json",
         "run", "a task", "--dry-run"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    if result.returncode == 3:
        pytest.skip("models are not resolvable in this sandboxed environment")
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["ok"] is True
