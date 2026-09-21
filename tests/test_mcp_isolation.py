"""MCP isolation, proved rather than asserted.

The structural tests below check the flags fabds passes. The probe test runs a
real fake MCP server three times and checks it starts when it should and does
not start when it must not. The control run is what makes the negative result
meaningful: without it, "no marker" could just mean a broken probe.

The probe does not need working credentials - MCP servers initialise during
session start-up, before authentication matters - so it runs anywhere the
``claude`` CLI is installed.
"""

from __future__ import annotations

import json
import shutil

import pytest

from fabds.doctor import probe_isolation
from fabds.mcpprobe import IsolationProbeResult, probe_config, write_fake_home, write_probe
from fabds.providers import claude_cli
from fabds.providers.base import CompletionRequest
from fabds.runner import CompletedCommand

ISOLATION_FLAGS = ["--strict-mcp-config", "--safe-mode", "--setting-sources",
                   "--tools", "--permission-prompts", "--no-session-persistence"]


@pytest.fixture
def provider(config, monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _name: "/fake/claude")
    return claude_cli.ClaudeCliProvider(config)


def capture_argv(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        return CompletedCommand(
            argv=list(argv), returncode=0, stderr="", duration_s=0.1,
            stdout=json.dumps({
                "result": "ok", "is_error": False,
                "modelUsage": {"claude-fable-5-1": {"canonicalModel": "claude-fable-5-1"}},
            }),
        )

    monkeypatch.setattr(claude_cli, "run_command", fake_run)
    return captured


def test_every_isolation_flag_is_passed(provider, monkeypatch):
    captured = capture_argv(monkeypatch)
    provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    argv = captured["argv"]
    for flag in ISOLATION_FLAGS:
        assert flag in argv, f"{flag} missing from the planner invocation"


def test_the_mcp_config_is_empty_by_default(provider, monkeypatch):
    captured = capture_argv(monkeypatch)
    provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    argv = captured["argv"]
    payload = json.loads(argv[argv.index("--mcp-config") + 1])
    assert payload == {"mcpServers": {}}, "the planner must start with no MCP servers"
    assert argv[argv.index("--setting-sources") + 1] == "", "no settings inheritance"
    assert argv[argv.index("--tools") + 1] == "", "the planner gets no tools"
    assert argv[argv.index("--permission-prompts") + 1] == "none"


def test_an_explicit_grant_is_the_only_way_in(provider, monkeypatch):
    """A grant is honoured for real, and the reduced isolation is recorded.

    ``--safe-mode`` disables MCP wholesale and beats ``--mcp-config``, so it is
    dropped exactly when a grant is present. Keeping it would silently ignore a
    capability the controller asked for, which is worse than the narrower
    isolation. Everything else stays in force.
    """
    captured = capture_argv(monkeypatch)
    grant = {"audit": {"command": "/bin/true", "args": []}}
    response = provider.complete(
        CompletionRequest("sys", "hi", "claude-fable-5-1", mcp_grants=grant))
    argv = captured["argv"]
    payload = json.loads(argv[argv.index("--mcp-config") + 1])
    assert payload == {"mcpServers": grant}
    assert "--strict-mcp-config" in argv, "even a grant stays strict"
    assert "--safe-mode" not in argv, "safe-mode would silently suppress the grant"
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--tools") + 1] == ""
    assert response.mcp_isolated is False, "a granted run is recorded as not isolated"


def test_safe_mode_is_used_whenever_nothing_is_granted(provider, monkeypatch):
    captured = capture_argv(monkeypatch)
    provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    assert "--safe-mode" in captured["argv"]


def test_the_planner_runs_in_a_neutral_directory(provider, monkeypatch):
    """Nothing of the user's repository is auto-discoverable from there."""
    captured = capture_argv(monkeypatch)
    provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    from pathlib import Path

    cwd = Path(captured["cwd"])
    assert "fabds-planner-" in cwd.name


def test_the_planner_environment_carries_no_bridge_or_credentials(provider, monkeypatch):
    captured = capture_argv(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "bridge")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    env = captured["env"]
    assert "CLAUDE_CODE_MESSAGING_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "HOME" in env, "HOME is needed for the CLI's own credential store"


def test_workers_have_no_mcp_surface_at_all(config, tmp_path):
    """DeepSeek workers use plain HTTPS; there is nothing to isolate."""
    from dataclasses import replace

    from fabds.providers.deepseek_http import DeepSeekHttpProvider

    key = tmp_path / "k"
    key.write_text("sk-x-0123456789", encoding="utf-8")
    provider = DeepSeekHttpProvider(replace(config, deepseek_api_key_file=key))
    assert "no MCP surface" in provider.status().isolation
    assert not hasattr(provider, "mcp_grants")


def test_probe_result_requires_a_working_control():
    """A negative result with a dead probe proves nothing, and says so."""
    dead = IsolationProbeResult(control_started=False, isolated_started=False)
    assert dead.meaningful is False
    assert dead.passed is False
    assert "inconclusive" in dead.describe()

    good = IsolationProbeResult(control_started=True, isolated_started=False,
                                granted_started=True)
    assert good.passed is True

    leaky = IsolationProbeResult(control_started=True, isolated_started=True)
    assert leaky.passed is False
    assert "FAILED" in leaky.describe()


def test_probe_script_records_its_own_startup(tmp_path):
    """The probe itself works: run it directly and check the marker."""
    import subprocess
    import sys

    marker = tmp_path / "m.marker"
    script = write_probe(tmp_path / "probe", marker)
    handshake = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n"
    result = subprocess.run([sys.executable, str(script)], input=handshake,
                            capture_output=True, text=True, timeout=30,
                            env={"FABDS_PROBE_MARKER": str(marker), "PATH": "/usr/bin"})
    assert marker.read_text(encoding="utf-8").strip() == "STARTED"
    assert "fabds-probe" in result.stdout


def test_probe_config_shape(tmp_path):
    script = write_probe(tmp_path / "p", tmp_path / "m")
    config = probe_config(script, tmp_path / "m", name="probe")
    entry = config["mcpServers"]["probe"]
    assert entry["env"]["FABDS_PROBE_MARKER"] == str(tmp_path / "m")
    assert str(script) in entry["args"]


@pytest.mark.skipif(not shutil.which("claude"), reason="claude CLI not installed")
def test_live_three_way_isolation_probe():
    """The real thing: registered -> starts; isolated -> absent; granted -> starts."""
    result = probe_isolation(shutil.which("claude"), timeout_s=180)
    assert result.control_started is True, (
        "control run did not start the probe, so this test proves nothing: " + result.detail
    )
    assert result.isolated_started is False, (
        "an unrelated MCP server initialised inside an isolated fabds call: " + result.detail
    )
    assert result.granted_started is True, (
        "an explicitly granted server failed to start, so isolation is not selective: "
        + result.detail
    )
