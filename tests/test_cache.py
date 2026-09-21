"""Caches must be fast on a hit and impossible to poison with stale state."""

from __future__ import annotations

import json
import time

import pytest

from conftest import FakeProvider
from fabds.cache import FileCache, analysis_cache_key, plan_cache_key
from fabds.context import ContextPacket, ContextSection, git_state, repo_identity
from fabds.errors import ConfigError
from fabds.models import ResolvedModel
from fabds.planner import Planner

BASE = {"repo_identity": "git@example.invalid:me/app.git",
        "git_state": {"commit": "c1", "tree": "t1", "dirty_digest": "d1"},
        "task_fingerprint": "task-abc", "planner_model": "claude-fable-5-1",
        "context_digest": "ctx1", "version": "0.1.0"}

FABLE = ResolvedModel("planner", "Fable 5.1", "claude-fable-5-1", "claude_cli",
                      r"claude-fable-5-1", "registry", "test")

PLAN_JSON = json.dumps({
    "approach": "split the work",
    "packets": [{"task_id": "w1", "kind": "implement", "objective": "do it",
                 "owned_paths": ["src/**"], "acceptance_criteria": ["tests pass"]}],
})


@pytest.fixture
def cache(tmp_path):
    return FileCache(tmp_path / "cache")


def test_hit_and_miss(cache):
    key = plan_cache_key(**BASE)
    assert cache.get(key) is None
    cache.put(key, {"approach": "x"}, ttl_s=600)
    assert cache.get(key).payload == {"approach": "x"}
    assert cache.hits == 1


@pytest.mark.parametrize("field,value", [
    ("repo_identity", "git@example.invalid:me/other.git"),
    ("task_fingerprint", "task-different"),
    ("planner_model", "claude-opus-5"),
    ("context_digest", "ctx2"),
    ("version", "0.2.0"),
])
def test_any_input_change_invalidates(cache, field, value):
    cache.put(plan_cache_key(**BASE), {"approach": "x"}, ttl_s=600)
    assert cache.get(plan_cache_key(**{**BASE, field: value})) is None


@pytest.mark.parametrize("field,value", [
    ("commit", "c2"), ("tree", "t2"), ("dirty_digest", "d2"),
])
def test_repository_state_change_invalidates(cache, field, value):
    """An uncommitted edit changes dirty_digest, so a plan cannot outlive it."""
    cache.put(plan_cache_key(**BASE), {"approach": "x"}, ttl_s=600)
    moved = {**BASE, "git_state": {**BASE["git_state"], field: value}}
    assert cache.get(plan_cache_key(**moved)) is None


def test_entries_expire(cache):
    key = plan_cache_key(**BASE)
    cache.put(key, {"approach": "x"}, ttl_s=0)
    time.sleep(0.01)
    assert cache.get(key) is None
    assert not cache._path(key).exists(), "an expired entry is removed, not left behind"


def test_implementation_results_are_never_cached(cache):
    key = analysis_cache_key(repo_identity="r", git_state={}, packet_digest="p",
                             worker_model="deepseek-flash", version="1")
    with pytest.raises(ConfigError, match="only read-only analysis may be cached"):
        cache.put(key, {"files_changed": ["a.py"]}, ttl_s=600, metadata={"read_only": False})
    cache.put(key, {"findings": []}, ttl_s=600, metadata={"read_only": True})
    assert cache.get(key) is not None


def test_entries_are_private_on_disk(cache):
    key = plan_cache_key(**BASE)
    cache.put(key, {"approach": "x"}, ttl_s=600)
    mode = cache._path(key).stat().st_mode & 0o777
    assert mode == 0o600, f"cache entries must not be group or world readable (got {mode:o})"


def test_disabled_cache_never_reads_or_writes(tmp_path):
    cache = FileCache(tmp_path / "cache", enabled=False)
    key = plan_cache_key(**BASE)
    cache.put(key, {"approach": "x"}, ttl_s=600)
    assert cache.get(key) is None
    assert not (tmp_path / "cache").exists()


def test_corrupt_entries_are_treated_as_misses(cache):
    key = plan_cache_key(**BASE)
    cache.put(key, {"approach": "x"}, ttl_s=600)
    cache._path(key).write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


def test_schema_change_invalidates_everything(cache, monkeypatch):
    key = plan_cache_key(**BASE)
    cache.put(key, {"approach": "x"}, ttl_s=600)
    monkeypatch.setattr("fabds.cache.CACHE_SCHEMA", 999)
    assert cache.get(key) is None


def test_purge_and_prune(cache):
    cache.put(plan_cache_key(**BASE), {"a": 1}, ttl_s=600)
    cache.put(plan_cache_key(**{**BASE, "context_digest": "ctx9"}), {"a": 2}, ttl_s=0)
    assert cache.prune_expired() == 1
    assert cache.stats()["entries"] == 1
    assert cache.purge() >= 1
    assert cache.stats()["entries"] == 0


# -- planner integration ----------------------------------------------------

def _packet(digest_source="a"):
    packet = ContextPacket(budget_chars=1000)
    packet.sections.append(ContextSection("Task", digest_source))
    return packet


def test_planner_serves_a_second_identical_request_from_cache(cache, null_logger):
    provider = FakeProvider("claude_cli", responses=[PLAN_JSON, PLAN_JSON])
    planner = Planner(provider=provider, model=FABLE, logger=null_logger,
                      cache=cache, version="0.1.0", max_rounds=5)
    kwargs = dict(task="build a parser", constraints="", context=_packet(),
                  repo_identity="r", git_state=BASE["git_state"], max_packets=5)

    first = planner.plan(**kwargs)
    assert first.from_cache is False
    assert len(provider.requests) == 1

    planner.rounds_used = 0  # a later round with identical inputs
    second = planner.plan(**kwargs)
    assert second.from_cache is True
    assert len(provider.requests) == 1, "the model must not be called twice"
    assert second.packets[0].task_id == first.packets[0].task_id


def test_planner_calls_again_when_the_repository_moved(cache, null_logger):
    provider = FakeProvider("claude_cli", responses=[PLAN_JSON, PLAN_JSON])
    planner = Planner(provider=provider, model=FABLE, logger=null_logger,
                      cache=cache, version="0.1.0", max_rounds=5)
    base = dict(task="build", constraints="", context=_packet(),
                repo_identity="r", max_packets=5)
    planner.plan(**base, git_state={"commit": "c1", "tree": "t1", "dirty_digest": "d1"})
    planner.rounds_used = 0
    planner.plan(**base, git_state={"commit": "c1", "tree": "t1", "dirty_digest": "CHANGED"})
    assert len(provider.requests) == 2


def test_planner_round_cap_is_enforced(cache, null_logger):
    from fabds.errors import MalformedResponse

    provider = FakeProvider("claude_cli", responses=[PLAN_JSON] * 5)
    planner = Planner(provider=provider, model=FABLE, logger=null_logger,
                      cache=None, version="0.1.0", max_rounds=1)
    kwargs = dict(task="t", constraints="", context=_packet(), repo_identity="r",
                  git_state=BASE["git_state"], max_packets=5)
    planner.plan(**kwargs)
    with pytest.raises(MalformedResponse, match="round cap"):
        planner.plan(**kwargs)
    assert len(provider.requests) == 1


def test_repo_identity_strips_embedded_credentials(tmp_path, monkeypatch):
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin",
                    "https://user:s3cr3t@github.invalid/me/app.git"],
                   cwd=root, check=True, capture_output=True)
    identity = repo_identity(root)
    assert "s3cr3t" not in identity
    assert "github.invalid" in identity


def test_git_state_tracks_uncommitted_work(git_repo):
    before = git_state(git_repo)
    assert before["dirty"] is False
    (git_repo / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    after = git_state(git_repo)
    assert after["dirty"] is True
    assert after["dirty_digest"] != before["dirty_digest"]
    assert after["commit"] == before["commit"], "the commit alone would not have noticed"
