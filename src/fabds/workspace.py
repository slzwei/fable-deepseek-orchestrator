"""Worker workspaces.

Each writing worker gets its own filesystem, so concurrent workers physically
cannot collide and a bad worker cannot damage the user's working tree.

Two implementations:

``GitWorktreeWorkspace``
    ``git worktree add --detach`` from a base revision into a temp directory.
    The worktree has its own index, so staging and diffing inside it never
    touches the user's index or working tree. This is the default whenever the
    target is a git repository.

``CopyWorkspace``
    For non-git targets. Copies only the paths the packet grants, so the worker
    sees exactly its own slice of the repository and nothing else.

Read-only workers get no workspace of their own: they read the repository in
place and :class:`~fabds.permissions.WorkerPermissions` refuses every write.

Uncommitted user work is never moved or reverted. A worktree starts from a
commit; if the controller needs dirty files present it asks for them
explicitly with ``include_uncommitted``, which *copies* them in.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .errors import WorkspaceError
from .pathsafety import resolve_within

__all__ = [
    "Workspace", "GitWorktreeWorkspace", "CopyWorkspace", "ReadOnlyWorkspace",
    "create_workspace", "supports_worktrees",
]


def _git(root: Path, *args: str, timeout: int = 120, check: bool = True) -> str:
    result = subprocess.run(  # noqa: S603 - argv list, shell=False
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if check and result.returncode != 0:
        raise WorkspaceError(
            f"git {' '.join(args[:2])} failed in {root}: {result.stderr.strip()[:300]}"
        )
    return result.stdout


def supports_worktrees(root: Path) -> bool:
    """True when ``root`` is a git repository with at least one commit."""
    try:
        inside = _git(root, "rev-parse", "--is-inside-work-tree", check=False).strip()
        if inside != "true":
            return False
        return bool(_git(root, "rev-parse", "--verify", "HEAD", check=False).strip())
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass
class Workspace:
    """Base workspace: a root directory plus change accounting."""

    task_id: str
    source_root: Path
    root: Path
    kind: str = "base"
    read_only: bool = False
    _baseline: dict = field(default_factory=dict, repr=False)
    _temp_parent: Path | None = field(default=None, repr=False)

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> "Workspace":
        return self

    def cleanup(self) -> None:
        if self._temp_parent is not None and self._temp_parent.exists():
            shutil.rmtree(self._temp_parent, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self.setup()

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    # -- change accounting --------------------------------------------------

    def changed_files(self) -> list[str]:
        """Paths modified since setup, observed by the controller."""
        current = self._snapshot()
        changed = sorted(
            set(current) ^ set(self._baseline)
            | {p for p in current if p in self._baseline and current[p] != self._baseline[p]}
        )
        return changed

    def diff(self) -> str:
        return ""

    def export_patch(self) -> str:
        """A patch the controller can apply to the source repository."""
        return ""

    def read(self, rel: str) -> str:
        return resolve_within(self.root, rel, must_exist=True).read_text(
            encoding="utf-8", errors="replace"
        )

    # -- internals ----------------------------------------------------------

    def _snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in self.root.rglob("*"):
            if path.is_dir() or ".git" in path.parts:
                continue
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                snapshot[path.relative_to(self.root).as_posix()] = digest
            except OSError:
                continue
        return snapshot

    def describe(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "root": str(self.root),
            "source_root": str(self.source_root),
            "read_only": self.read_only,
        }


class GitWorktreeWorkspace(Workspace):
    """An isolated git worktree, detached at a base revision."""

    def __init__(self, task_id: str, source_root: Path, *, base_rev: str = "HEAD",
                 include_uncommitted: "tuple[str, ...]" = ()) -> None:
        temp_parent = Path(tempfile.mkdtemp(prefix=f"fabds-ws-{task_id}-"))
        super().__init__(
            task_id=task_id,
            source_root=Path(source_root).resolve(strict=False),
            root=temp_parent / "tree",
            kind="git-worktree",
        )
        self._temp_parent = temp_parent
        self.base_rev = base_rev
        self.include_uncommitted = include_uncommitted

    def setup(self) -> "GitWorktreeWorkspace":
        if not supports_worktrees(self.source_root):
            raise WorkspaceError(
                f"{self.source_root} is not a git repository with a commit; "
                "use a copy workspace instead"
            )
        resolved = _git(self.source_root, "rev-parse", self.base_rev).strip()
        if not resolved:
            raise WorkspaceError(f"cannot resolve base revision {self.base_rev!r}")
        self.base_rev = resolved
        _git(self.source_root, "worktree", "add", "--detach", str(self.root), resolved)

        for rel in self.include_uncommitted:
            source = resolve_within(self.source_root, rel)
            if not source.is_file():
                continue
            destination = resolve_within(self.root, rel, follow_final_symlink=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        self._baseline = self._snapshot()
        return self

    def changed_files(self) -> list[str]:
        output = _git(self.root, "status", "--porcelain=v1", "--untracked-files=all", check=False)
        changed = []
        for line in output.splitlines():
            if len(line) > 3:
                path = line[3:].strip().strip('"')
                if " -> " in path:          # a rename reports both sides
                    changed.extend(part.strip() for part in path.split(" -> "))
                else:
                    changed.append(path)
        return sorted(set(changed))

    def diff(self) -> str:
        _git(self.root, "add", "--all", check=False)   # the worktree has its own index
        return _git(self.root, "diff", "--cached", check=False)

    def export_patch(self) -> str:
        _git(self.root, "add", "--all", check=False)
        return _git(self.root, "diff", "--cached", "--binary", check=False)

    def cleanup(self) -> None:
        try:
            _git(self.source_root, "worktree", "remove", "--force", str(self.root), check=False)
            _git(self.source_root, "worktree", "prune", check=False)
        except (OSError, subprocess.SubprocessError, WorkspaceError):
            pass
        super().cleanup()


class CopyWorkspace(Workspace):
    """A temp directory holding copies of only the granted paths."""

    def __init__(self, task_id: str, source_root: Path, *,
                 include_globs: "tuple[str, ...]" = ("**/*",),
                 sanitizer=None, max_files: int = 2000) -> None:
        temp_parent = Path(tempfile.mkdtemp(prefix=f"fabds-ws-{task_id}-"))
        super().__init__(
            task_id=task_id,
            source_root=Path(source_root).resolve(strict=False),
            root=temp_parent / "tree",
            kind="copy",
        )
        self._temp_parent = temp_parent
        self.include_globs = include_globs
        self.sanitizer = sanitizer
        self.max_files = max_files

    def setup(self) -> "CopyWorkspace":
        self.root.mkdir(parents=True, exist_ok=True)
        copied = 0
        for pattern in self.include_globs:
            for source in sorted(self.source_root.glob(pattern)):
                if copied >= self.max_files:
                    raise WorkspaceError(
                        f"copy workspace for {self.task_id} exceeded {self.max_files} files; "
                        "narrow the packet's granted paths"
                    )
                if source.is_dir() or source.is_symlink():
                    continue
                try:
                    rel = source.resolve(strict=False).relative_to(self.source_root).as_posix()
                except ValueError:
                    continue  # a symlink escaping the source tree
                if self.sanitizer is not None and self.sanitizer.is_secret_path(rel):
                    continue  # credential files never enter a worker workspace
                if self.sanitizer is not None and any(
                    self.sanitizer.should_skip_dir(part) for part in Path(rel).parts[:-1]
                ):
                    continue
                destination = resolve_within(self.root, rel, follow_final_symlink=False)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied += 1
        self._baseline = self._snapshot()
        return self

    def diff(self) -> str:
        lines = []
        for rel in self.changed_files():
            source = self.source_root / rel
            target = self.root / rel
            if not target.exists():
                lines.append(f"--- deleted: {rel}")
            elif not source.exists():
                lines.append(f"+++ added: {rel}")
            else:
                lines.append(f"~~~ modified: {rel}")
        return "\n".join(lines)


class ReadOnlyWorkspace(Workspace):
    """The repository itself, with writes refused by the permission layer."""

    def __init__(self, task_id: str, source_root: Path) -> None:
        root = Path(source_root).resolve(strict=False)
        super().__init__(task_id=task_id, source_root=root, root=root,
                         kind="read-only", read_only=True)

    def setup(self) -> "ReadOnlyWorkspace":
        self._baseline = {}
        return self

    def changed_files(self) -> list[str]:
        return []

    def cleanup(self) -> None:
        return  # never delete the user's repository


def create_workspace(task_id: str, source_root: Path, *, read_only: bool,
                     base_rev: str = "HEAD", include_globs: "tuple[str, ...]" = ("**/*",),
                     sanitizer=None, include_uncommitted: "tuple[str, ...]" = ()) -> Workspace:
    """Pick the strongest isolation the environment supports."""
    source_root = Path(source_root).resolve(strict=False)
    if read_only:
        return ReadOnlyWorkspace(task_id, source_root)
    if supports_worktrees(source_root):
        return GitWorktreeWorkspace(
            task_id, source_root, base_rev=base_rev, include_uncommitted=include_uncommitted
        )
    return CopyWorkspace(task_id, source_root, include_globs=include_globs, sanitizer=sanitizer)
