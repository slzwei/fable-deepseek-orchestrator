"""One failure must never take down a run, and retries must terminate."""

from __future__ import annotations

import json

import pytest

from conftest import FakeProvider
from fabds.errors import (
    ContextTooLarge,
    ModelAttestationError,
    ProviderCrashed,
    ProviderRateLimited,
    ProviderTimeout,
)
from fabds.models import ResolvedModel
from fabds.packets import TaskKind, TaskStatus, WorkPacket
from fabds.permissions import WorkerPermissions
from fabds.sanitizer import ContextSanitizer
from fabds.workers import PoolLimits, WorkerPool, WorkerRunner
from fabds.workspace import Workspace

DEEPSEEK = ResolvedModel("worker", "DeepSeek V4.1 Flash", "deepseek-flash",
                         "deepseek_http", r"deepseek-flash", "api", "test")

FINISH = json.dumps({"actions": [{"op": "finish", "summary": "done", "status": "completed",
                                  "files_changed": ["src/a.py"]}]})
WRITE = json.dumps({"actions": [{"op": "write_file", "path": "src/a.py", "content": "y = 2\n"}]})


def make_runner(tmp_path, provider, logger, *, max_retries=2, max_turns=6, name="w"):
    root = tmp_path / name
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    workspace = Workspace(task_id=name, source_root=root, root=root)
    workspace.setup()
    return WorkerRunner(
        provider=provider, model=DEEPSEEK,
        packet=WorkPacket(name, TaskKind.IMPLEMENT, "do it", owned_paths=("src/**",)),
        workspace=workspace, permissions=WorkerPermissions(root, owned=("src/**",)),
        sanitizer=ContextSanitizer(root), logger=logger,
        max_turns=max_turns, max_retries=max_retries, timeout_s=30,
    )


def test_retries_are_bounded_and_terminate(tmp_path, null_logger, monkeypatch):
    monkeypatch.setattr("fabds.workers.time.sleep", lambda _s: None)
    provider = FakeProvider("deepseek_http",
                            raises=[ProviderCrashed("boom")] * 20, responses=[FINISH] * 20)
    envelope = make_runner(tmp_path, provider, null_logger, max_retries=2).run()
    assert envelope.status is TaskStatus.FAILED
    assert envelope.error["code"] == "provider_crashed"
    # One initial attempt plus exactly two retries, then it stops.
    assert len(provider.requests) == 3


def test_a_transient_failure_is_recovered(tmp_path, null_logger, monkeypatch):
    monkeypatch.setattr("fabds.workers.time.sleep", lambda _s: None)
    provider = FakeProvider(
        "deepseek_http",
        raises=[ProviderRateLimited("slow down"), None, None],
        responses=[WRITE, FINISH],
    )
    envelope = make_runner(tmp_path, provider, null_logger).run()
    assert envelope.status is TaskStatus.COMPLETED
    assert "src/a.py" in envelope.observed_files_changed


def test_wrong_model_is_never_retried(tmp_path, null_logger):
    """Retrying an attestation failure would just burn money on the wrong model."""
    provider = FakeProvider("deepseek_http", responses=[FINISH] * 5,
                            reported_model="deepseek-v4-pro")
    envelope = make_runner(tmp_path, provider, null_logger, max_retries=3).run()
    assert envelope.status is TaskStatus.FAILED
    assert envelope.error["code"] == "model_attestation_failed"
    assert len(provider.requests) == 1


def test_context_too_large_is_not_retried(tmp_path, null_logger):
    provider = FakeProvider("deepseek_http", raises=[ContextTooLarge("too big")] * 5,
                            responses=[FINISH] * 5)
    envelope = make_runner(tmp_path, provider, null_logger, max_retries=3).run()
    assert envelope.status is TaskStatus.FAILED
    assert envelope.error["code"] == "context_too_large"
    assert len(provider.requests) == 1


def test_empty_response_is_handled(tmp_path, null_logger):
    from fabds.errors import EmptyResponse

    provider = FakeProvider("deepseek_http", raises=[EmptyResponse("nothing")] * 5,
                            responses=[FINISH] * 5)
    envelope = make_runner(tmp_path, provider, null_logger, max_retries=1).run()
    assert envelope.status is TaskStatus.FAILED


def test_one_failed_worker_preserves_the_others(tmp_path, null_logger):
    """B's work must survive A's failure."""
    pool = WorkerPool(PoolLimits(max_workers=2, implementation=2), null_logger)

    good = make_runner(tmp_path, FakeProvider("deepseek_http", responses=[WRITE, FINISH]),
                       null_logger, name="good")
    bad = make_runner(tmp_path, FakeProvider("deepseek_http",
                                             raises=[ProviderTimeout("hung")] * 5,
                                             responses=[FINISH] * 5),
                      null_logger, max_retries=0, name="bad")

    results = pool.run([
        (WorkPacket("good", TaskKind.IMPLEMENT, "ok", owned_paths=("src/good/**",)),
         lambda: good),
        (WorkPacket("bad", TaskKind.IMPLEMENT, "fails", owned_paths=("src/bad/**",)),
         lambda: bad),
    ])
    assert results["good"].status is TaskStatus.COMPLETED
    assert results["bad"].status is TaskStatus.FAILED
    assert (tmp_path / "good" / "src" / "a.py").read_text(encoding="utf-8") == "y = 2\n"


def test_a_crashing_runner_does_not_kill_the_pool(null_logger):
    from fabds.packets import ResultEnvelope

    class Exploding:
        def run(self):
            raise RuntimeError("unexpected internal error")

    class Fine:
        def run(self):
            return ResultEnvelope(task_id="fine", status=TaskStatus.COMPLETED,
                                  observed_files_changed=("x",))

    pool = WorkerPool(PoolLimits(max_workers=2, research=2), null_logger)
    results = pool.run([
        (WorkPacket("boom", TaskKind.RESEARCH, "x"), Exploding),
        (WorkPacket("fine", TaskKind.RESEARCH, "y"), Fine),
    ])
    assert results["boom"].status is TaskStatus.FAILED
    assert results["boom"].error["code"] == "worker_crashed"
    assert results["fine"].status is TaskStatus.COMPLETED


def test_planner_failure_does_not_lose_the_run_record(tmp_path, config, null_logger):
    """A planner crash is reported and recorded, not swallowed."""
    from fabds.errors import ProviderCrashed as Crash
    from fabds.models import ModelResolver
    from fabds.orchestrator import Orchestrator

    providers = {
        "claude_cli": FakeProvider("claude_cli", models=["claude-fable-5-1"],
                                   raises=[Crash("planner exploded")]),
        "deepseek_http": FakeProvider("deepseek_http", models=["deepseek-flash"]),
    }
    resolver = ModelResolver(config, providers=providers, cache_path=tmp_path / "r.json")
    repo = tmp_path / "repo"
    repo.mkdir()
    orchestrator = Orchestrator(config, repo, logger=null_logger, resolver=resolver)
    orchestrator.resolver._providers = providers

    from fabds import providers as providers_module

    original = providers_module.get_provider
    providers_module.get_provider = lambda name, cfg: providers[name]
    try:
        outcome = orchestrator.run(task="do a thing")
    finally:
        providers_module.get_provider = original

    assert outcome.ok is False
    assert outcome.error["code"] == "provider_crashed"
    assert orchestrator.ledger.manifest_path.is_file(), "the run must still be recorded"
    manifest = orchestrator.ledger.load_manifest()
    assert manifest["error"]["code"] == "provider_crashed"


def test_subprocess_timeouts_kill_the_process_group():
    from fabds.runner import run_command

    result = run_command(
        ["python3", "-c",
         "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); time.sleep(60)"],
        timeout_s=1.0,
    )
    assert result.timed_out is True
    assert result.duration_s < 20


def test_missing_executable_is_a_typed_failure():
    from fabds.runner import run_command

    with pytest.raises(ProviderCrashed, match="executable not found"):
        run_command(["definitely-not-a-real-binary-xyz"])
