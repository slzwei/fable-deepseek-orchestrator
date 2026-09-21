"""Fable selection and attestation.

The reference failure mode this guards against: a Claude command succeeds, and
the output is labelled "Fable" because the command exited zero. Here, identity
comes from the ``modelUsage`` block and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from conftest import FakeProvider
from fabds.errors import (
    ModelAttestationError,
    ModelResolutionError,
    ProviderRateLimited,
    ProviderUnavailable,
)
from fabds.models import ModelResolver, ResolvedModel, Role, attest
from fabds.providers import claude_cli
from fabds.providers.base import CompletionRequest, ModelResponse
from fabds.runner import CompletedCommand

FABLE = ResolvedModel(
    role="planner", display_name="Fable 5.1", model_id="claude-fable-5-1",
    provider="claude_cli", pattern=r"claude-fable-5-1(?:\[[^\]]+\])?",
    source="registry", evidence="test",
)


def cli_payload(model_key: str, *, canonical: str | None = None, text: str = "ok",
                is_error: bool = False, status: int | None = None) -> str:
    return json.dumps({
        "result": text,
        "is_error": is_error,
        "api_error_status": status,
        "total_cost_usd": 0.01,
        "session_id": "s1",
        "modelUsage": {model_key: {
            "inputTokens": 10, "outputTokens": 5, "canonicalModel": canonical or model_key,
            "provider": "firstParty", "contextWindow": 1_000_000,
        }},
    })


@pytest.fixture
def provider(config, monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _name: "/fake/claude")
    return claude_cli.ClaudeCliProvider(config)


def stub_run(monkeypatch, stdout: str, *, stderr: str = "", returncode: int = 0):
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return CompletedCommand(argv=list(argv), returncode=returncode, stdout=stdout,
                                stderr=stderr, duration_s=0.1)

    monkeypatch.setattr(claude_cli, "run_command", fake_run)
    return captured


def test_invocation_names_fable_explicitly(provider, monkeypatch):
    captured = stub_run(monkeypatch, cli_payload("claude-fable-5-1"))
    provider.complete(CompletionRequest(
        system_prompt="sys", user_prompt="hello", model_id="claude-fable-5-1"))
    argv = captured["argv"]
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "claude-fable-5-1"
    # The prompt travels on stdin, never in argv, so it cannot appear in `ps`.
    assert "hello" not in argv
    assert captured["kwargs"]["stdin_text"] == "hello"


def test_metadata_reports_the_real_model(provider, monkeypatch):
    stub_run(monkeypatch, cli_payload("claude-fable-5-1"))
    response = provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    assert response.reported_model == "claude-fable-5-1"
    assert response.canonical_model == "claude-fable-5-1"
    assert response.attestation()["provider"] == "claude_cli"
    attest(response, FABLE)  # must not raise


def test_another_claude_model_is_never_relabelled_as_fable(provider, monkeypatch):
    """The call succeeded, but Opus answered. That is an attestation failure."""
    stub_run(monkeypatch, cli_payload("claude-opus-5"))
    response = provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    assert response.reported_model == "claude-opus-5"
    with pytest.raises(ModelAttestationError) as excinfo:
        attest(response, FABLE)
    assert "claude-opus-5" in str(excinfo.value)
    assert "Refusing to label" in str(excinfo.value)


def test_canonical_alias_of_fable_is_accepted(provider, monkeypatch):
    stub_run(monkeypatch, cli_payload("claude-fable-5-1[1m]", canonical="claude-fable-5-1"))
    response = provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    attest(response, FABLE)


def test_missing_model_usage_is_an_attestation_failure(provider, monkeypatch):
    stub_run(monkeypatch, json.dumps({"result": "ok", "is_error": False}))
    with pytest.raises(ModelAttestationError, match="cannot be verified"):
        provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))


def test_two_models_in_one_response_is_refused(provider, monkeypatch):
    stub_run(monkeypatch, json.dumps({
        "result": "ok", "is_error": False,
        "modelUsage": {"claude-fable-5-1": {}, "claude-sonnet-5": {}},
    }))
    with pytest.raises(ModelAttestationError, match="more than one model"):
        provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))


def test_unrecognised_model_is_detected(provider, monkeypatch):
    stub_run(monkeypatch, "", stderr="[claude-code:unrecognized_model] {\"model\":\"nope\"}")
    with pytest.raises(ProviderUnavailable, match="does not recognise"):
        provider.complete(CompletionRequest("sys", "hi", "nope"))


def test_quota_exhaustion_surfaces_as_rate_limited(provider, monkeypatch):
    """The real environment returns this; it must not degrade to another model."""
    stub_run(monkeypatch, json.dumps({
        "result": "You've reached your Fable limit. Switch to another model, or "
                  "manage usage credits at claude.ai/settings/usage to continue.",
        "is_error": True, "api_error_status": 429, "modelUsage": {},
    }))
    with pytest.raises(ProviderRateLimited) as excinfo:
        provider.complete(CompletionRequest("sys", "hi", "claude-fable-5-1"))
    assert "rate limited or out of quota" in str(excinfo.value)


def test_orchestration_aborts_when_fable_is_unavailable(config, tmp_path):
    """No Fable, no run. The controller does not continue with a substitute."""
    from fabds.orchestrator import Orchestrator

    providers = {
        "claude_cli": FakeProvider("claude_cli", models=["claude-opus-5", "claude-sonnet-5"]),
        "deepseek_http": FakeProvider("deepseek_http", models=["deepseek-flash"]),
    }
    resolver = ModelResolver(config, providers=providers,
                             cache_path=tmp_path / "resolution.json")
    repo = tmp_path / "repo"
    repo.mkdir()
    orchestrator = Orchestrator(config, repo, resolver=resolver)
    outcome = orchestrator.run(task="do something")
    assert outcome.ok is False
    assert outcome.error["code"] == "model_resolution_failed"
    assert not outcome.results, "no worker may run when the planner is unresolved"


def test_real_cli_binary_contains_the_fable_identifier():
    """Discovery reads ids out of the installed artefact, not a hard-coded list."""
    import shutil
    from pathlib import Path

    which = shutil.which("claude")
    if not which:
        pytest.skip("claude CLI not installed")
    ids = claude_cli.extract_model_ids(Path(which).resolve())
    if not ids:
        pytest.skip("installed CLI exposes no readable model registry")
    assert "claude-fable-5-1" in ids
    assert any(i.startswith("claude-opus") for i in ids), "sanity: other families present too"
