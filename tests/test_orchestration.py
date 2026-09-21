"""The controller end to end, with scripted providers.

Covers the properties that only appear when the pieces run together: planner
proposals becoming authorised packets, workers running in real worktrees,
independent verification, escalation, patch staging and deliberate integration.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from conftest import FakeProvider
from fabds.models import ModelResolver
from fabds.orchestrator import Orchestrator
from fabds.packets import TaskKind, TaskStatus

PLAN = json.dumps({
    "approach": "Split the change from its tests.",
    "key_decisions": [{"decision": "keep the signature", "rationale": "callers depend on it"}],
    "packets": [
        {"task_id": "impl", "kind": "implement", "objective": "make add() handle floats",
         "owned_paths": ["src/**"], "readonly_paths": ["tests/**"],
         "acceptance_criteria": ["src/calc.py defines add"], "parallel_safe": True},
        {"task_id": "review", "kind": "audit", "objective": "look for edge cases",
         "readonly_paths": ["src/**"], "acceptance_criteria": ["risks listed"],
         "parallel_safe": True},
    ],
    "risks": ["floats change comparison semantics"],
    "verification_strategy": "run the suite",
})

CRITIQUE = json.dumps({
    "verdict": "needs_work",
    "findings": [{"severity": "high", "claim": "no float test", "evidence": "none seen",
                  "confidence": "likely", "suggested_fix": "add one"}],
    "unverified_areas": ["performance"],
    "summary": "Mostly fine, one gap.",
})


def worker_script(request):
    """Respond based on which packet is being driven."""
    prompt = request.user_prompt
    if "# Work packet impl" in prompt or "impl" in prompt.split("\n")[0]:
        if "wrote" in prompt or "OK" in prompt:
            return json.dumps({"actions": [{"op": "finish", "summary": "updated calc",
                                            "files_changed": ["src/calc.py"],
                                            "tests_passed": True, "status": "completed"}]})
        return json.dumps({"actions": [
            {"op": "write_file", "path": "src/calc.py",
             "content": "def add(a, b):\n    return float(a) + float(b)\n"}]})
    return json.dumps({"actions": [{"op": "finish", "summary": "looks fine",
                                    "risks": ["float precision"], "status": "completed"}]})


@pytest.fixture
def scripted(config, git_repo, recording_logger, monkeypatch):
    planner = FakeProvider("claude_cli", models=["claude-fable-5-1"],
                           responses=[PLAN, CRITIQUE])
    worker = FakeProvider("deepseek_http", models=["deepseek-flash"],
                          responses=[worker_script] * 40)
    providers = {"claude_cli": planner, "deepseek_http": worker}

    from fabds import providers as providers_module

    monkeypatch.setattr(providers_module, "get_provider",
                        lambda name, cfg: providers[name])
    resolver = ModelResolver(config, providers=providers, cache_path=config.cache_dir / "r.json")
    orchestrator = Orchestrator(config, git_repo, logger=recording_logger, resolver=resolver)
    recording_logger.attach_jsonl(orchestrator.ledger.events_path)
    return orchestrator, providers


def test_full_run_plans_delegates_and_stages(scripted, git_repo):
    orchestrator, providers = scripted
    outcome = orchestrator.run(task="make add() handle floats")

    assert outcome.models["planner"]["model_id"] == "claude-fable-5-1"
    assert outcome.models["worker"]["model_id"] == "deepseek-flash"
    assert set(outcome.results) == {"impl", "review"}
    assert outcome.results["impl"].status is TaskStatus.COMPLETED
    assert outcome.results["review"].status is TaskStatus.COMPLETED
    assert "src/calc.py" in outcome.results["impl"].observed_files_changed

    # The planner was called exactly once for planning.
    assert sum(1 for r in providers["claude_cli"].requests if r.label == "plan") == 1

    # Nothing was merged into the user's tree.
    assert "float" not in (git_repo / "src" / "calc.py").read_text(encoding="utf-8")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=git_repo,
                            capture_output=True, text=True).stdout
    assert "src/calc.py" not in status

    # But the patch is staged, ready for a deliberate decision.
    patch = orchestrator.ledger.patch_path("impl")
    assert patch.is_file()
    assert "float" in patch.read_text(encoding="utf-8")


def test_read_only_packet_can_inspect_but_not_write(scripted):
    """It may run the checks; it owns nothing and cannot write."""
    orchestrator, _ = scripted
    outcome = orchestrator.run(task="make add() handle floats")
    review = next(p for p in outcome.packets if p["task_id"] == "review")
    assert review["read_only"] is True
    assert review["owned_paths"] == []
    assert review["validation_command_ids"] == [], "reporting is the job, gating is not"

    # It may run the checks, but the filesystem boundary is unchanged.
    from fabds.permissions import WorkerPermissions

    permissions = WorkerPermissions(
        orchestrator.repo_root, readonly=tuple(review["readonly_paths"]), read_only=True)
    with pytest.raises(Exception):
        permissions.assert_write("src/calc.py")


def test_controller_verifies_independently(scripted):
    orchestrator, _ = scripted
    orchestrator.run(task="make add() handle floats")
    verification = orchestrator._verification
    assert "impl" in verification
    assert verification["impl"]["checks"], "the controller must run the checks itself"
    assert all("argv" in c for c in verification["impl"]["checks"])


def test_integration_is_explicit_and_selective(scripted, git_repo):
    orchestrator, _ = scripted
    orchestrator.run(task="make add() handle floats")

    check = orchestrator.integrate(["impl"], check_only=True)
    assert check["applied"] and not check["rejected"]
    assert "float" not in (git_repo / "src" / "calc.py").read_text(encoding="utf-8"), (
        "--check must not actually apply anything")

    report = orchestrator.integrate(["impl"])
    assert [entry["task_id"] for entry in report["applied"]] == ["impl"]
    assert "float" in (git_repo / "src" / "calc.py").read_text(encoding="utf-8")

    missing = orchestrator.integrate(["review"])
    assert missing["rejected"][0]["reason"] == "no staged patch"


def test_escalation_is_skipped_when_results_are_consistent(scripted):
    orchestrator, providers = scripted
    outcome = orchestrator.run(task="make add() handle floats")
    assert outcome.critique.get("skipped") is True
    assert "consistent" in outcome.critique["reason"]
    assert not any(r.label == "critique" for r in providers["claude_cli"].requests)


def test_escalation_happens_when_explicitly_requested(scripted):
    orchestrator, providers = scripted
    outcome = orchestrator.run(task="make add() handle floats", final_review=True)
    assert outcome.critique["verdict"] == "needs_work"
    assert outcome.critique["attestation"]["reported_model"] == "claude-fable-5-1"
    assert any(r.label == "critique" for r in providers["claude_cli"].requests)


def test_no_planner_mode_never_calls_the_planner(scripted):
    orchestrator, providers = scripted
    outcome = orchestrator.run(task="tweak something", use_planner=False)
    assert providers["claude_cli"].requests == []
    assert outcome.plan["approach"].startswith("Controller-directed")


def test_the_run_is_fully_recorded(scripted):
    orchestrator, _ = scripted
    outcome = orchestrator.run(task="make add() handle floats")
    root = orchestrator.ledger.root
    assert (root / "run.json").is_file()
    assert (root / "plan.json").is_file()
    assert (root / "packets" / "impl.json").is_file()
    assert (root / "results" / "impl.json").is_file()
    assert (root / "events.jsonl").is_file(), "the event stream must be recorded"
    events = [json.loads(line) for line in
              (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    components = {event["component"] for event in events}
    assert "controller" in components and "planner" in components
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert manifest["models"]["planner"]["model_id"] == "claude-fable-5-1"
    assert manifest["stats"]["workers"]["completed"] == 2


def test_planner_context_excludes_secrets(scripted, git_repo):
    orchestrator, providers = scripted
    # Assembled at runtime; see tests/test_secret_redaction.py::_synth for why.
    fake_aws = "".join(("wJalrXUtnFEMI", "K7MDENG", "bPxRfiCY"))
    (git_repo / ".env").write_text(f"AWS_SECRET_ACCESS_KEY={fake_aws}\n", encoding="utf-8")
    orchestrator.run(task="make add() handle floats")
    prompt = providers["claude_cli"].requests[0].user_prompt
    assert fake_aws not in prompt
    assert "excluded: credential path" in prompt


def test_planner_prompt_carries_the_trust_boundary(scripted):
    orchestrator, providers = scripted
    orchestrator.run(task="make add() handle floats")
    system = providers["claude_cli"].requests[0].system_prompt
    assert "TRUST BOUNDARY" in system
    assert "not from the controller" in system


def test_generated_artefacts_are_not_counted_as_worker_changes(git_repo):
    """Running tests creates __pycache__; that is not the worker's work."""
    from fabds.workspace import create_workspace, is_generated_artefact

    for relative, expected in [
        ("src/__pycache__/x.cpython-314.pyc", True),
        ("tests/__pycache__/test_a.cpython-311.pyc", True),
        (".pytest_cache/v/cache/lastfailed", True),
        ("node_modules/left-pad/index.js", True),
        (".DS_Store", True),
        ("src/calc.py", False),
        ("tests/test_calc.py", False),
        ("docs/pycache-notes.md", False),
    ]:
        assert is_generated_artefact(relative) is expected, relative

    workspace = create_workspace("art", git_repo, read_only=False)
    try:
        workspace.setup()
        (workspace.root / "src" / "__pycache__").mkdir(parents=True, exist_ok=True)
        (workspace.root / "src" / "__pycache__" / "calc.cpython-314.pyc").write_bytes(b"\x00\x01")
        (workspace.root / "src" / "calc.py").write_text("def add(a, b):\n    return a + b + 0\n",
                                                        encoding="utf-8")
        changed = workspace.changed_files()
        assert changed == ["src/calc.py"], changed
        patch = workspace.export_patch()
        assert "__pycache__" not in patch, "compiled output must never enter a patch"
        assert "src/calc.py" in patch
    finally:
        workspace.cleanup()


def test_unittest_command_is_discoverable_for_a_plain_tests_directory(tmp_path):
    """`unittest discover` alone finds nothing when tests/ is not a package."""
    import subprocess
    import sys

    from fabds.orchestrator import detect_validation_commands

    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "tests" / "test_demo.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n",
        encoding="utf-8")

    specs = {spec.id: spec for spec in detect_validation_commands(root)}
    runner = specs.get("pytest") or specs["unittest"]
    result = subprocess.run(  # noqa: S603
        [sys.executable, *runner.argv[1:]], cwd=root,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"{runner.display()} failed:\n{result.stderr}"
    assert "NO TESTS RAN" not in result.stderr, (
        f"{runner.display()} discovered nothing, so it would validate nothing")


def test_a_truncated_reply_costs_a_turn_not_the_worker(tmp_path, null_logger):
    """finish_reason=length gets a hint and a retry, not a dead worker."""
    import json as _json

    from conftest import FakeProvider
    from fabds.errors import ResponseTruncated
    from fabds.models import ResolvedModel
    from fabds.packets import TaskKind, WorkPacket
    from fabds.permissions import WorkerPermissions
    from fabds.sanitizer import ContextSanitizer
    from fabds.workers import WorkerRunner
    from fabds.workspace import Workspace

    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    workspace = Workspace(task_id="t", source_root=root, root=root)
    workspace.setup()

    provider = FakeProvider(
        "deepseek_http",
        raises=[ResponseTruncated("budget exhausted"), None, None],
        responses=[
            _json.dumps({"actions": [{"op": "write_file", "path": "src/a.py",
                                      "content": "y = 2\n"}]}),
            _json.dumps({"actions": [{"op": "finish", "summary": "done",
                                      "status": "completed"}]}),
        ],
    )
    envelope = WorkerRunner(
        provider=provider,
        model=ResolvedModel("worker", "DeepSeek V4.1 Flash", "deepseek-flash",
                            "deepseek_http", r"deepseek-flash", "api", "test"),
        packet=WorkPacket("t", TaskKind.IMPLEMENT, "edit", owned_paths=("src/**",)),
        workspace=workspace, permissions=WorkerPermissions(root, owned=("src/**",)),
        sanitizer=ContextSanitizer(root), logger=null_logger, max_turns=5, max_retries=0,
    ).run()

    assert envelope.status is TaskStatus.COMPLETED
    assert "src/a.py" in envelope.observed_files_changed
    hint = provider.requests[1].user_prompt
    assert "smaller pieces" in hint, "the model must be told why its reply was rejected"


def test_verify_runs_the_repository_checks_against_the_working_tree(scripted, git_repo):
    """The integrated check is separate from the per-packet checks."""
    orchestrator, _ = scripted
    report = orchestrator.verify()
    assert report["checks"], "a Python repo with tests/ must offer a validation command"
    assert report["all_passed"] is True
    assert (orchestrator.ledger.root / "verification.json").is_file()

    # Break the tree; verification must notice. Keep it unittest-shaped so the
    # same runner is still offered - otherwise this would test command
    # detection rather than verification.
    (git_repo / "tests" / "test_calc.py").write_text(
        "import sys\n"
        "import unittest\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\n"
        "from src.calc import add\n\n\n"
        "class TestAdd(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(1, 2), 99)\n",
        encoding="utf-8")
    broken = orchestrator.verify()
    assert broken["checks"], "the same runner must still be offered"
    assert broken["all_passed"] is False


def test_integrate_verify_reports_a_broken_merge(scripted, git_repo):
    orchestrator, _ = scripted
    orchestrator.run(task="make add() handle floats")
    report = orchestrator.integrate(["impl"], verify=True)
    assert report["applied"]
    assert "verification" in report, "--verify must actually verify"
    assert report["verification"]["checks"]


def test_controller_authored_packets_go_through_the_same_authorisation(scripted, tmp_path):
    """--packets is a convenience, not a way around the permission model."""
    import json as _json

    orchestrator, providers = scripted
    packets_file = tmp_path / "packets.json"
    packets_file.write_text(_json.dumps({"packets": [
        {"task_id": "mine", "kind": "audit", "objective": "look around",
         "owned_paths": ["src/**"],          # read-only kinds may not own paths
         "readonly_paths": ["**"]},
    ]}), encoding="utf-8")

    outcome = orchestrator.run(task="inspect", packets_file=packets_file)
    assert providers["claude_cli"].requests == [], "no planner call when packets are supplied"
    packet = outcome.packets[0]
    assert packet["read_only"] is True
    assert packet["owned_paths"] == [], "an audit packet cannot be granted write paths"


def test_a_malformed_packets_file_is_a_clear_error(scripted, tmp_path):
    from fabds.errors import ConfigError

    orchestrator, _ = scripted
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read packets file"):
        orchestrator.load_packets(bad)

    empty = tmp_path / "empty.json"
    empty.write_text('{"packets": []}', encoding="utf-8")
    with pytest.raises(ConfigError, match="no packets"):
        orchestrator.load_packets(empty)


def test_no_validation_command_is_offered_when_none_can_work(tmp_path, monkeypatch):
    """A command that discovers nothing is worse than an honest gap."""
    from fabds.orchestrator import _module_available, detect_validation_commands

    _module_available.cache_clear()
    monkeypatch.setattr("fabds.orchestrator._module_available", lambda _m: False)

    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    # pytest-style: a bare function, which `unittest discover` cannot collect.
    (root / "tests" / "test_demo.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")

    offered = {spec.id for spec in detect_validation_commands(root)}
    assert "pytest" not in offered, "pytest is not importable here"
    assert "unittest" not in offered, (
        "unittest cannot collect bare pytest functions; offering it would "
        "produce a command that validates nothing and exits non-zero")

    # Now make the tests unittest-compatible: the command reappears.
    (root / "tests" / "test_demo.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")
    assert "unittest" in {spec.id for spec in detect_validation_commands(root)}


def test_verify_is_honest_when_nothing_can_be_checked(tmp_path, config, null_logger, monkeypatch):
    from fabds.orchestrator import Orchestrator, _module_available

    _module_available.cache_clear()
    monkeypatch.setattr("fabds.orchestrator._module_available", lambda _m: False)
    root = tmp_path / "plain"
    root.mkdir()
    (root / "notes.txt").write_text("no code here\n", encoding="utf-8")
    report = Orchestrator(config, root, logger=null_logger).verify()
    assert report["checks"] == []
    assert report["all_passed"] is None, "unknown must not be reported as passing"


def test_a_dependent_packet_sees_its_dependency_result(git_repo, config, recording_logger,
                                                       monkeypatch):
    """An audit of work it cannot observe is worthless, so it must observe it."""
    import json as _json

    from conftest import FakeProvider
    from fabds.models import ModelResolver
    from fabds.orchestrator import Orchestrator

    def worker(request):
        prompt = request.user_prompt
        if "# Work packet impl" in prompt:
            if "wrote" in prompt:
                return _json.dumps({"actions": [{"op": "finish", "summary": "done",
                                                 "status": "completed"}]})
            return _json.dumps({"actions": [{"op": "write_file", "path": "src/marker.py",
                                             "content": "SENTINEL = 1\n"}]})
        # the auditor: navigate from the root, then read the dependency's file
        if "SENTINEL" in prompt:
            return _json.dumps({"actions": [{"op": "finish",
                                             "summary": "saw SENTINEL", "status": "completed"}]})
        if "list_dir" in prompt or "src/" in prompt:
            return _json.dumps({"actions": [{"op": "read_file", "path": "src/marker.py"}]})
        return _json.dumps({"actions": [{"op": "list_dir", "path": "."}]})

    providers = {
        "claude_cli": FakeProvider("claude_cli", models=["claude-fable-5-1"]),
        "deepseek_http": FakeProvider("deepseek_http", models=["deepseek-flash"],
                                      responses=[worker] * 40),
    }
    from fabds import providers as providers_module

    monkeypatch.setattr(providers_module, "get_provider", lambda name, cfg: providers[name])
    resolver = ModelResolver(config, providers=providers, cache_path=config.cache_dir / "r.json")
    orchestrator = Orchestrator(config, git_repo, logger=recording_logger, resolver=resolver)

    packets_file = git_repo / "packets.json"
    packets_file.write_text(_json.dumps({"packets": [
        {"task_id": "impl", "kind": "implement", "objective": "write a marker",
         "owned_paths": ["src/**"], "readonly_paths": ["**"]},
        {"task_id": "audit", "kind": "audit", "objective": "confirm the marker exists",
         "readonly_paths": ["src/**"], "depends_on": ["impl"]},
    ]}), encoding="utf-8")

    outcome = orchestrator.run(task="demo", packets_file=packets_file)
    audit = outcome.results["audit"]
    assert audit.status is TaskStatus.COMPLETED
    assert "SENTINEL" in audit.summary, (
        "the auditor could not observe its dependency's work; its report is worthless")
    assert audit.denied_actions == (), (
        f"the auditor was blocked navigating its own workspace: {audit.denied_actions}")


def test_an_auditor_can_navigate_to_its_granted_paths(tmp_path):
    """Listing an ancestor directory must not be refused."""
    from fabds.permissions import WorkerPermissions

    root = tmp_path / "ws"
    (root / "src" / "parser").mkdir(parents=True)
    (root / "src" / "parser" / "core.py").write_text("x = 1\n", encoding="utf-8")
    (root / "unrelated").mkdir()
    (root / "unrelated" / "other.py").write_text("y = 2\n", encoding="utf-8")

    permissions = WorkerPermissions(root, readonly=("src/parser/**",), read_only=True)
    for navigable in (".", "src", "src/parser"):
        permissions.assert_read(navigable)          # must not raise
    permissions.assert_read("src/parser/core.py")

    # Navigation does not become disclosure: a sibling subtree stays refused.
    with pytest.raises(Exception):
        permissions.assert_read("unrelated/other.py")
    # And an ancestor listing still marks credential files as excluded, not shown.
    assert permissions.may_read(root / "unrelated") is False


def test_read_only_packets_may_run_validation_commands(tmp_path, config, null_logger):
    """Read-only means cannot write files, not cannot run tests."""
    from fabds.orchestrator import Orchestrator, detect_validation_commands
    from fabds.planner import PlannedPacket

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo / "tests" / "test_a.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")

    orchestrator = Orchestrator(config, repo, logger=null_logger)
    commands = detect_validation_commands(repo)
    packets = orchestrator.authorise(
        [PlannedPacket("look", TaskKind.AUDIT, "review", readonly_paths=("**",))],
        commands=commands,
    )
    offered = {c.id for c in packets[0].commands}
    runner = {"pytest", "unittest"} & {c.id for c in commands}
    assert runner <= offered, (
        "an auditor that cannot run the tests can see a problem but never confirm it")
    # It is offered, not required: reporting is the job, gating is not.
    assert packets[0].validation_command_ids == ()
    # And it still cannot write anything.
    assert packets[0].read_only is True
    assert packets[0].owned_paths == ()


def test_a_seeded_worker_is_told_its_workspace_was_pre_populated(git_repo, config,
                                                                 recording_logger, monkeypatch):
    """Otherwise an empty `git status` reads as 'the work is missing'."""
    import json as _json

    from conftest import FakeProvider
    from fabds.models import ModelResolver
    from fabds.orchestrator import Orchestrator

    seen = {}

    def worker(request):
        if "# Work packet audit" in request.user_prompt:
            seen["prompt"] = request.user_prompt
            return _json.dumps({"actions": [{"op": "finish", "summary": "ok",
                                             "status": "completed"}]})
        if "wrote" in request.user_prompt:
            return _json.dumps({"actions": [{"op": "finish", "summary": "done",
                                             "status": "completed"}]})
        return _json.dumps({"actions": [{"op": "write_file", "path": "src/m.py",
                                         "content": "M = 1\n"}]})

    providers = {
        "claude_cli": FakeProvider("claude_cli", models=["claude-fable-5-1"]),
        "deepseek_http": FakeProvider("deepseek_http", models=["deepseek-flash"],
                                      responses=[worker] * 30),
    }
    from fabds import providers as providers_module

    monkeypatch.setattr(providers_module, "get_provider", lambda name, cfg: providers[name])
    resolver = ModelResolver(config, providers=providers, cache_path=config.cache_dir / "r.json")
    orchestrator = Orchestrator(config, git_repo, logger=recording_logger, resolver=resolver)

    packets_file = git_repo / "p.json"
    packets_file.write_text(_json.dumps({"packets": [
        {"task_id": "impl", "kind": "implement", "objective": "write",
         "owned_paths": ["src/**"], "readonly_paths": ["**"]},
        {"task_id": "audit", "kind": "audit", "objective": "review",
         "readonly_paths": ["**"], "depends_on": ["impl"]},
    ]}), encoding="utf-8")
    orchestrator.run(task="demo", packets_file=packets_file)

    assert "seeded with the completed results of: impl" in seen.get("prompt", "")
    assert "empty `git status` is expected" in seen["prompt"]
