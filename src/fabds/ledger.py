"""Run state on disk.

Each run gets a directory under ``<repo>/.fabds/runs/<run_id>/`` containing:

===================  =====================================================
``run.json``         the manifest: task, resolved models, limits, outcome
``events.jsonl``     every log event, redacted
``plan.json``        the planner's proposal and its attestation
``packets/``         the authorised packets, exactly as sent
``results/``         one result envelope per packet
``patches/``         one patch per writing worker, ready for integration
===================  =====================================================

Nothing here is applied to the repository. Patches sit in the run directory
until the controller explicitly integrates them, which is what keeps final
authority with Codex rather than with this program.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .redaction import REDACTOR

__all__ = ["RunLedger", "new_run_id"]


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid():05d}"


@dataclass
class RunLedger:
    repo_root: Path
    run_id: str
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve(strict=False)
        self.root = self.repo_root / ".fabds" / "runs" / self.run_id
        for sub in ("packets", "results", "patches"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._ensure_ignored()

    def _ensure_ignored(self) -> None:
        """Keep run state out of the user's commits without editing .gitignore."""
        marker = self.repo_root / ".fabds" / ".gitignore"
        if not marker.exists():
            try:
                marker.write_text("# Written by fabds. Run state is local only.\n*\n",
                                  encoding="utf-8")
            except OSError:
                pass

    # -- paths --------------------------------------------------------------

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "run.json"

    def patch_path(self, task_id: str) -> Path:
        return self.root / "patches" / f"{task_id}.patch"

    # -- writes -------------------------------------------------------------

    def write_json(self, relative: str, payload: dict) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
        path.write_text(REDACTOR.scrub(text), encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def write_text(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(REDACTOR.scrub(text), encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def save_manifest(self, manifest: dict) -> Path:
        return self.write_json("run.json", manifest)

    def save_packet(self, packet) -> Path:
        return self.write_json(f"packets/{packet.task_id}.json", packet.as_dict())

    def save_result(self, envelope) -> Path:
        return self.write_json(f"results/{envelope.task_id}.json", envelope.as_dict())

    def save_patch(self, task_id: str, patch: str) -> Path | None:
        if not patch.strip():
            return None
        return self.write_text(f"patches/{task_id}.patch", patch)

    # -- reads --------------------------------------------------------------

    def load_manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @classmethod
    def list_runs(cls, repo_root: Path) -> list[str]:
        runs_dir = Path(repo_root).resolve(strict=False) / ".fabds" / "runs"
        if not runs_dir.is_dir():
            return []
        return sorted((p.name for p in runs_dir.iterdir() if p.is_dir()), reverse=True)

    @classmethod
    def latest(cls, repo_root: Path) -> "RunLedger | None":
        runs = cls.list_runs(repo_root)
        return cls(repo_root, runs[0]) if runs else None
