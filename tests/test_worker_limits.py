"""Bounds: turns, tasks, retries, recursion, output size."""

from __future__ import annotations

import json

import pytest

from conftest import FakeProvider
from fabds.config import Limits
from fabds.errors import ConfigError, MalformedResponse
from fabds.models import ResolvedModel
from fabds.packets import TaskKind, TaskStatus, WorkPacket, parse_actions
from fabds.permissions import WorkerPermissions
from fabds.sanitizer import ContextSanitizer
from fabds.workers import MAX_WRITE_BYTES, WorkerRunner
from fabds.workspace import Workspace

DEEPSEEK = ResolvedModel("worker", "DeepSeek V4.1 Flash", "deepseek-flash",
                         "deepseek_http", r"deepseek-flash", "api", "test")


def make_runner(tmp_path, responses, logger, *, max_turns=4, packet=None, **kwargs):
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    workspace = Workspace(task_id="w", source_root=root, root=root)
    workspace.setup()
    packet = packet or WorkPacket("w", TaskKind.IMPLEMENT, "do a thing",
                                  owned_paths=("src/**",), max_turns=max_turns)
    return WorkerRunner(
        provider=FakeProvider("deepseek_http", responses=responses),
        model=DEEPSEEK, packet=packet, workspace=workspace,
        permissions=WorkerPermissions(root, owned=("src/**",)),
        sanitizer=ContextSanitizer(root), logger=logger, max_turns=max_turns, **kwargs,
    )


def test_turn_limit_terminates_a_looping_worker(tmp_path, null_logger):
    """A worker that never finishes stops at the cap instead of running forever."""
    looping = [json.dumps({"actions": [{"op": "list_dir", "path": "src"}]})] * 50
    runner = make_runner(tmp_path, looping, null_logger, max_turns=4)
    envelope = runner.run()
    assert envelope.turns_used == 4
    assert envelope.status is TaskStatus.FAILED
    assert envelope.error["code"] == "turn_limit"
    assert len(runner.provider.requests) == 4, "exactly the cap, not one more"


def test_malformed_responses_are_retried_then_give_up(tmp_path, null_logger):
    runner = make_runner(tmp_path, ["not json at all"] * 20, null_logger, max_turns=3)
    envelope = runner.run()
    assert envelope.status is TaskStatus.FAILED
    assert len(runner.provider.requests) == 3


def test_a_worker_recovers_from_one_malformed_turn(tmp_path, null_logger):
    responses = [
        "sorry, here is prose instead of JSON",
        json.dumps({"actions": [{"op": "write_file", "path": "src/a.py", "content": "y = 2\n"}]}),
        json.dumps({"actions": [{"op": "finish", "summary": "done", "status": "completed",
                                 "files_changed": ["src/a.py"]}]}),
    ]
    envelope = make_runner(tmp_path, responses, null_logger, max_turns=5).run()
    assert envelope.status is TaskStatus.COMPLETED
    assert "src/a.py" in envelope.observed_files_changed


def test_actions_per_turn_are_capped():
    payload = {"actions": [{"op": "list_dir", "path": "."} for _ in range(50)]}
    assert len(parse_actions(json.dumps(payload), max_actions=4)) == 4


def test_empty_action_list_is_rejected():
    with pytest.raises(MalformedResponse, match="empty"):
        parse_actions(json.dumps({"actions": []}))


def test_oversized_writes_are_refused(tmp_path, null_logger):
    huge = "x" * (MAX_WRITE_BYTES + 1)
    responses = [
        json.dumps({"actions": [{"op": "write_file", "path": "src/a.py", "content": huge}]}),
        json.dumps({"actions": [{"op": "finish", "summary": "s", "status": "blocked"}]}),
    ]
    envelope = make_runner(tmp_path, responses, null_logger, max_turns=3).run()
    assert envelope.status is TaskStatus.BLOCKED
    assert (tmp_path / "ws" / "src" / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_task_count_is_clamped_to_the_limit(tmp_path, config, null_logger):
    from dataclasses import replace

    from fabds.orchestrator import Orchestrator, detect_validation_commands
    from fabds.planner import PlannedPacket

    config = replace(config, limits=replace(config.limits, max_total_tasks=3))
    repo = tmp_path / "repo"
    repo.mkdir()
    orchestrator = Orchestrator(config, repo, logger=null_logger)
    proposals = [
        PlannedPacket(f"task_{i:02d}", TaskKind.IMPLEMENT, "x", owned_paths=(f"src/{i}/**",))
        for i in range(10)
    ]
    packets = orchestrator.authorise(proposals, commands=detect_validation_commands(repo))
    assert len(packets) == 3


def test_worker_nesting_cannot_be_enabled():
    with pytest.raises(ConfigError, match="only the controller may create workers"):
        Limits(worker_nesting_enabled=True).validate()


def test_no_action_can_spawn_a_worker():
    """Recursion is structurally impossible: there is no spawn action."""
    from fabds.packets import ActionKind

    names = {kind.value for kind in ActionKind}
    for forbidden in ("spawn", "spawn_worker", "delegate", "orchestrate", "subagent", "fabds"):
        assert forbidden not in names
    assert names == {"read_file", "list_dir", "search", "write_file",
                     "delete_file", "run_command", "finish"}


def test_max_workers_is_capped():
    with pytest.raises(ConfigError, match="capped at 16"):
        Limits(max_workers=64).validate()


def test_limits_reject_nonsense_values():
    for kwargs in ({"max_workers": 0}, {"max_retries_per_task": -1}, {"max_worker_turns": 0}):
        with pytest.raises(ConfigError):
            Limits(**kwargs).validate()


def test_search_is_bounded(tmp_path, null_logger):
    """A pathological pattern cannot stall the run."""
    from fabds.packets import Action, ActionKind
    from fabds.workers import ActionExecutor

    root = tmp_path / "ws"
    root.mkdir()
    for i in range(30):
        (root / f"f{i}.py").write_text(("a" * 500 + "\n") * 50, encoding="utf-8")
    workspace = Workspace(task_id="s", source_root=root, root=root)
    workspace.setup()
    executor = ActionExecutor(workspace, WorkerPermissions(root, readonly=("**",)),
                              ContextSanitizer(root))

    import time

    # A normal pattern works and is fast.
    ok = executor.execute(Action(ActionKind.SEARCH, {"pattern": "aaa", "path": "."}))
    assert ok.ok and ok.metadata["matches"] > 0

    # A catastrophically backtracking pattern is cancelled, not survived by luck.
    # re.search() cannot be interrupted in-process, so this is only bounded
    # because the scan runs in a child process the parent can kill.
    started = time.monotonic()
    redos = executor.execute(Action(ActionKind.SEARCH, {"pattern": "(a+)+b", "path": "."}))
    elapsed = time.monotonic() - started
    assert elapsed < 30, f"search ran for {elapsed:.1f}s; the time budget is not enforced"
    assert redos.ok is False
    assert "too expensive" in redos.error

    too_long = executor.execute(Action(ActionKind.SEARCH, {"pattern": "a" * 500}))
    assert too_long.ok is False
    assert "exceeds" in too_long.error
