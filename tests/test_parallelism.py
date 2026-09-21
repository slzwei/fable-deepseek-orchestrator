"""Concurrency is bounded, ordered by dependencies, and free of file races."""

from __future__ import annotations

import json
import threading
import time

from conftest import FakeProvider
from fabds.models import ResolvedModel
from fabds.packets import ResultEnvelope, TaskKind, TaskStatus, WorkPacket
from fabds.workers import PoolLimits, WorkerPool

DEEPSEEK = ResolvedModel("worker", "DeepSeek V4.1 Flash", "deepseek-flash",
                         "deepseek_http", r"deepseek-flash", "api", "test")


class TrackingRunner:
    """A stand-in worker that records overlap instead of calling a model."""

    active = 0
    peak = 0
    lock = threading.Lock()
    order: list[str] = []

    def __init__(self, task_id, *, duration=0.05, status=TaskStatus.COMPLETED):
        self.task_id = task_id
        self.duration = duration
        self.status = status

    def run(self) -> ResultEnvelope:
        with TrackingRunner.lock:
            TrackingRunner.active += 1
            TrackingRunner.peak = max(TrackingRunner.peak, TrackingRunner.active)
            TrackingRunner.order.append(f"start:{self.task_id}")
        time.sleep(self.duration)
        with TrackingRunner.lock:
            TrackingRunner.active -= 1
            TrackingRunner.order.append(f"end:{self.task_id}")
        return ResultEnvelope(task_id=self.task_id, status=self.status,
                              observed_files_changed=("x",))


def reset():
    TrackingRunner.active = 0
    TrackingRunner.peak = 0
    TrackingRunner.order = []


def packet(task_id, kind=TaskKind.RESEARCH, owned=(), depends=()):
    return WorkPacket(task_id, kind, "objective", owned_paths=owned, depends_on=depends)


def test_global_concurrency_is_capped(null_logger):
    reset()
    pool = WorkerPool(PoolLimits(max_workers=2, research=8), null_logger)
    jobs = [(packet(f"r{i}"), lambda i=i: TrackingRunner(f"r{i}")) for i in range(8)]
    results = pool.run(jobs)
    assert len(results) == 8
    assert all(r.succeeded for r in results.values())
    assert TrackingRunner.peak <= 2, f"peak was {TrackingRunner.peak}, limit was 2"
    assert pool.peak_concurrency <= 2


def test_per_kind_limits_apply_within_the_global_cap(null_logger):
    reset()
    pool = WorkerPool(PoolLimits(max_workers=8, research=4, implementation=1), null_logger)
    jobs = [
        (packet(f"i{i}", TaskKind.IMPLEMENT, owned=(f"src/{i}/**",)),
         lambda i=i: TrackingRunner(f"i{i}", duration=0.08))
        for i in range(4)
    ]
    pool.run(jobs)
    # Implementation is limited to 1, so starts and ends must strictly alternate.
    assert TrackingRunner.order == [
        item for i in range(4) for item in (f"start:i{i}", f"end:i{i}")
    ] or TrackingRunner.peak == 1


def test_packets_owning_the_same_path_are_serialised(null_logger):
    """Even if a planner marks them parallel, shared ownership is a lock."""
    reset()
    pool = WorkerPool(PoolLimits(max_workers=4, implementation=4), null_logger)
    jobs = [
        (packet(f"c{i}", TaskKind.IMPLEMENT, owned=("src/shared/**",)),
         lambda i=i: TrackingRunner(f"c{i}", duration=0.08))
        for i in range(3)
    ]
    pool.run(jobs)
    assert TrackingRunner.peak == 1, "contending packets must never overlap"


def test_disjoint_ownership_runs_in_parallel(null_logger):
    reset()
    pool = WorkerPool(PoolLimits(max_workers=3, implementation=3), null_logger)
    jobs = [
        (packet(f"d{i}", TaskKind.IMPLEMENT, owned=(f"src/mod{i}/**",)),
         lambda i=i: TrackingRunner(f"d{i}", duration=0.15))
        for i in range(3)
    ]
    pool.run(jobs)
    assert TrackingRunner.peak > 1, "independent packets should overlap"


def test_dependencies_are_respected(null_logger):
    reset()
    pool = WorkerPool(PoolLimits(max_workers=4, research=4), null_logger)
    jobs = [
        (packet("second", depends=("first",)), lambda: TrackingRunner("second")),
        (packet("first"), lambda: TrackingRunner("first")),
    ]
    pool.run(jobs)
    assert TrackingRunner.order.index("end:first") < TrackingRunner.order.index("start:second")


def test_dependents_are_skipped_when_a_dependency_fails(null_logger):
    reset()
    pool = WorkerPool(PoolLimits(max_workers=4, research=4), null_logger)
    jobs = [
        (packet("base"), lambda: TrackingRunner("base", status=TaskStatus.FAILED)),
        (packet("child", depends=("base",)), lambda: TrackingRunner("child")),
    ]
    results = pool.run(jobs)
    assert results["base"].status is TaskStatus.FAILED
    assert results["child"].status is TaskStatus.SKIPPED
    assert "start:child" not in TrackingRunner.order


def test_a_dependency_cycle_terminates(null_logger):
    """A cycle must be reported, not deadlock the pool."""
    reset()
    pool = WorkerPool(PoolLimits(max_workers=2), null_logger)
    jobs = [
        (packet("a", depends=("b",)), lambda: TrackingRunner("a")),
        (packet("b", depends=("a",)), lambda: TrackingRunner("b")),
    ]
    results = pool.run(jobs)
    assert {r.status for r in results.values()} == {TaskStatus.SKIPPED}


def test_workspaces_are_physically_separate(git_repo, null_logger):
    """Two writing workers cannot see each other's files."""
    from fabds.workspace import create_workspace

    first = create_workspace("w1", git_repo, read_only=False)
    second = create_workspace("w2", git_repo, read_only=False)
    try:
        first.setup()
        second.setup()
        assert first.root != second.root
        (first.root / "only-in-first.txt").write_text("a", encoding="utf-8")
        assert not (second.root / "only-in-first.txt").exists()
        assert not (git_repo / "only-in-first.txt").exists()
        assert "only-in-first.txt" in first.changed_files()
        assert second.changed_files() == []
    finally:
        first.cleanup()
        second.cleanup()
