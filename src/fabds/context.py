"""Context packets.

A context packet is a compact, sanitised, budgeted description of just enough
repository state for one model call. It is never a repository dump.

Budgeting is explicit and enforced: sections are added in priority order and
the packet stops growing when the character budget is reached, recording what
was dropped. That keeps planner calls cheap and makes "context too large"
a design-time decision rather than a runtime surprise.

Deduplication is per-run: once a section's content has been sent to the planner
in this run, later rounds reference it by digest instead of resending it.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import repository_content_block
from .sanitizer import ContextSanitizer

__all__ = ["ContextSection", "ContextPacket", "ContextBuilder", "repo_identity", "git_state"]


@dataclass
class ContextSection:
    title: str
    body: str
    priority: int = 50           # lower is more important
    is_repo_content: bool = False
    included: bool = True
    dropped_reason: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()[:16]

    def render(self) -> str:
        if self.is_repo_content:
            return repository_content_block(self.title, self.body)
        return f"\n## {self.title}\n{self.body}\n"

    def size(self) -> int:
        return len(self.render())


@dataclass
class ContextPacket:
    sections: list[ContextSection] = field(default_factory=list)
    budget_chars: int = 60_000
    sanitizer_report: dict = field(default_factory=dict)
    deduplicated: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [section.render() for section in self.sections if section.included]
        if self.deduplicated:
            parts.append(
                "\n## Previously sent context\n"
                "These sections were sent earlier in this run and have not changed; "
                "they are unchanged and deliberately not repeated:\n"
                + "\n".join(f"  - {title}" for title in self.deduplicated)
                + "\n"
            )
        dropped = [s for s in self.sections if not s.included]
        if dropped:
            parts.append(
                "\n## Omitted from this packet (context budget)\n"
                + "\n".join(f"  - {s.title}: {s.dropped_reason}" for s in dropped)
                + "\n"
            )
        return "".join(parts)

    @property
    def size(self) -> int:
        return len(self.render())

    def digest(self) -> str:
        digest = hashlib.sha256()
        for section in self.sections:
            if section.included:
                digest.update(section.title.encode("utf-8"))
                digest.update(section.digest.encode("utf-8"))
        return digest.hexdigest()

    def summary(self) -> dict:
        return {
            "sections": [
                {"title": s.title, "included": s.included, "chars": s.size(),
                 "reason": s.dropped_reason}
                for s in self.sections
            ],
            "total_chars": self.size,
            "budget_chars": self.budget_chars,
            "deduplicated": list(self.deduplicated),
            "sanitizer": self.sanitizer_report,
        }


def repo_identity(root: Path) -> str:
    """A stable identity for cache keys: the remote URL if there is one."""
    root = Path(root).resolve(strict=False)
    try:
        result = subprocess.run(  # noqa: S603 - argv list, no shell
            ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        url = result.stdout.strip()
        if url:
            # Strip any embedded credentials before this becomes a cache key.
            if "@" in url and "://" in url:
                scheme, _, rest = url.partition("://")
                url = f"{scheme}://{rest.rpartition('@')[2]}"
            return url
    except (OSError, subprocess.SubprocessError):
        pass
    return str(root)


def git_state(root: Path) -> dict:
    """Commit, tree hash and dirtiness. The cache key's invalidation source."""
    root = Path(root).resolve(strict=False)

    def git(*args: str) -> str:
        try:
            result = subprocess.run(  # noqa: S603 - argv list, no shell
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=20, check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    is_repo = git("rev-parse", "--is-inside-work-tree") == "true"
    if not is_repo:
        return {"is_git": False, "commit": "", "tree": "", "dirty": False, "dirty_digest": ""}

    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "is_git": True,
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        # Hash the working-tree status so an uncommitted edit invalidates caches.
        "dirty_digest": hashlib.sha256(status.encode("utf-8")).hexdigest()[:16],
    }


class ContextBuilder:
    """Assembles a budgeted context packet."""

    def __init__(self, root: Path, sanitizer: ContextSanitizer, *,
                 budget_chars: int = 60_000,
                 already_sent: "dict[str, str] | None" = None) -> None:
        self.root = Path(root).resolve(strict=False)
        self.sanitizer = sanitizer
        self.budget_chars = budget_chars
        self.already_sent = already_sent if already_sent is not None else {}
        self._sections: list[ContextSection] = []

    # -- section builders ---------------------------------------------------

    def add(self, title: str, body: str, *, priority: int = 50,
            is_repo_content: bool = False) -> "ContextBuilder":
        if body and body.strip():
            self._sections.append(ContextSection(title, body.strip(), priority, is_repo_content))
        return self

    def add_task(self, task: str, constraints: str = "") -> "ContextBuilder":
        body = task.strip()
        if constraints.strip():
            body += f"\n\nConstraints:\n{constraints.strip()}"
        return self.add("Task", body, priority=0)

    def add_tree(self, *, max_entries: int = 300) -> "ContextBuilder":
        entries = self.sanitizer.tree(max_entries=max_entries)
        if not entries:
            return self
        return self.add("Repository tree", "\n".join(entries), priority=20)

    def add_git_state(self) -> "ContextBuilder":
        state = git_state(self.root)
        if not state.get("is_git"):
            return self.add("Version control", "Not a git repository.", priority=25)
        body = (
            f"branch: {state['branch']}\n"
            f"commit: {state['commit'][:12]}\n"
            f"working tree: {'has uncommitted changes' if state['dirty'] else 'clean'}"
        )
        return self.add("Version control", body, priority=25)

    def add_files(self, paths, *, priority: int = 30, max_chars: int | None = None) -> "ContextBuilder":
        for raw in paths:
            path = (self.root / raw) if not Path(raw).is_absolute() else Path(raw)
            result = self.sanitizer.read(path, max_chars=max_chars)
            if not result.included:
                self.add(f"File {result.path}", f"[excluded: {result.reason}]", priority=priority + 10)
                continue
            self.add(f"File {result.path}", result.text, priority=priority, is_repo_content=True)
        return self

    def add_command_output(self, title: str, text: str, *, priority: int = 35) -> "ContextBuilder":
        clean, _ = self.sanitizer.sanitize_text(text)
        return self.add(title, clean, priority=priority, is_repo_content=True)

    def add_previous_attempts(self, attempts) -> "ContextBuilder":
        if not attempts:
            return self
        body = "\n".join(f"  - {attempt}" for attempt in attempts)
        return self.add("Previous attempts in this run", body, priority=15)

    # -- assembly -----------------------------------------------------------

    def build(self) -> ContextPacket:
        packet = ContextPacket(budget_chars=self.budget_chars)
        used = 0
        for section in sorted(self._sections, key=lambda s: (s.priority, s.title)):
            previous = self.already_sent.get(section.title)
            if previous == section.digest:
                packet.deduplicated.append(section.title)
                continue
            size = section.size()
            if used + size > self.budget_chars:
                section.included = False
                section.dropped_reason = (
                    f"would exceed the {self.budget_chars} char budget "
                    f"({size} chars, {self.budget_chars - used} remaining)"
                )
            else:
                used += size
                self.already_sent[section.title] = section.digest
            packet.sections.append(section)
        packet.sanitizer_report = self.sanitizer.report()
        return packet
