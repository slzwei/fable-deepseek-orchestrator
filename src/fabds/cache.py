"""Caches for planning and read-only analysis.

Two caches, both file backed, both short lived, neither able to return a stale
answer for a changed repository:

**Plan cache.** Keyed by repository identity, git commit *and* working-tree
digest, the task fingerprint, the resolved planner model id, the context digest
and the skill version. Any edit to a tracked or untracked file changes
``dirty_digest`` and therefore the key, so a plan can never survive the state it
was made for.

**Analysis cache.** Read-only worker results only. Implementation results are
never cached: replaying a diff onto a different tree is how you corrupt a
repository, so the API simply refuses.

Entries expire by TTL, are stored ``0600``, and hold sanitised payloads only.
``fabds cache purge`` removes them; nothing here is meant to be long lived.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

__all__ = ["CacheKey", "CacheEntry", "FileCache", "plan_cache_key", "analysis_cache_key"]

CACHE_SCHEMA = 2


@dataclass(frozen=True)
class CacheKey:
    namespace: str
    digest: str

    @property
    def filename(self) -> str:
        return f"{self.digest}.json"


@dataclass
class CacheEntry:
    key: str
    namespace: str
    created_at: float
    ttl_s: int
    payload: dict
    metadata: dict

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at

    @property
    def expired(self) -> bool:
        return self.age_s > self.ttl_s


def _digest(parts: dict) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def plan_cache_key(*, repo_identity: str, git_state: dict, task_fingerprint: str,
                   planner_model: str, context_digest: str, version: str,
                   round_index: int = 0) -> CacheKey:
    """Everything that could change the right answer goes into the key."""
    return CacheKey("plan", _digest({
        "schema": CACHE_SCHEMA,
        "repo": repo_identity,
        "commit": git_state.get("commit", ""),
        "tree": git_state.get("tree", ""),
        "dirty": git_state.get("dirty_digest", ""),
        "task": task_fingerprint,
        "model": planner_model,
        "context": context_digest,
        "round": round_index,
        "version": version,
    }))


def analysis_cache_key(*, repo_identity: str, git_state: dict, packet_digest: str,
                       worker_model: str, version: str) -> CacheKey:
    return CacheKey("analysis", _digest({
        "schema": CACHE_SCHEMA,
        "repo": repo_identity,
        "commit": git_state.get("commit", ""),
        "tree": git_state.get("tree", ""),
        "dirty": git_state.get("dirty_digest", ""),
        "packet": packet_digest,
        "model": worker_model,
        "version": version,
    }))


class FileCache:
    """A small, auditable JSON file cache. No database, no daemon."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, key: CacheKey) -> Path:
        return self.root / key.namespace / key.filename

    def get(self, key: CacheKey) -> CacheEntry | None:
        if not self.enabled:
            self.misses += 1
            return None
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        if data.get("schema") != CACHE_SCHEMA:
            self.misses += 1
            return None
        entry = CacheEntry(
            key=key.digest,
            namespace=key.namespace,
            created_at=float(data.get("created_at", 0)),
            ttl_s=int(data.get("ttl_s", 0)),
            payload=data.get("payload") or {},
            metadata=data.get("metadata") or {},
        )
        if entry.expired:
            self.invalidate(key)
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: CacheKey, payload: dict, *, ttl_s: int,
            metadata: dict | None = None) -> None:
        if not self.enabled:
            return
        if key.namespace == "analysis" and metadata and metadata.get("read_only") is False:
            raise ConfigError(
                "refusing to cache a result from a writing worker; only read-only "
                "analysis may be cached"
            )
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": CACHE_SCHEMA,
            "created_at": time.time(),
            "ttl_s": ttl_s,
            "payload": payload,
            "metadata": metadata or {},
        }
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, default=str)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, path)
        except OSError:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def invalidate(self, key: CacheKey) -> bool:
        try:
            self._path(key).unlink()
            return True
        except OSError:
            return False

    def purge(self, namespace: str | None = None) -> int:
        """Delete cached entries. Returns how many files were removed."""
        target = self.root / namespace if namespace else self.root
        if not target.exists():
            return 0
        removed = sum(1 for _ in target.rglob("*.json"))
        shutil.rmtree(target, ignore_errors=True)
        return removed

    def prune_expired(self) -> int:
        removed = 0
        for path in self.root.rglob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                age = time.time() - float(data.get("created_at", 0))
                if data.get("schema") != CACHE_SCHEMA or age > float(data.get("ttl_s", 0)):
                    path.unlink()
                    removed += 1
            except (OSError, json.JSONDecodeError, ValueError):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def stats(self) -> dict:
        entries = list(self.root.rglob("*.json")) if self.root.exists() else []
        return {
            "root": str(self.root),
            "enabled": self.enabled,
            "entries": len(entries),
            "hits": self.hits,
            "misses": self.misses,
            "bytes": sum(p.stat().st_size for p in entries if p.exists()),
        }
