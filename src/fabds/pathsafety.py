"""Path containment.

Every filesystem access a model asks for goes through :func:`resolve_within`.
The rule is simple and has no exceptions: the *fully resolved* path, and every
one of its existing ancestors, must stay inside the resolved workspace root.

This defeats, in one place:

* ``../../etc/passwd`` traversal,
* absolute paths pointing anywhere else,
* a symlink inside the repo pointing at ``~/.ssh`` (repository-controlled
  symlinks are attacker-controlled data),
* creating a *new* file underneath a symlinked directory that escapes,
* NUL bytes and other path smuggling.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from .errors import PathSafetyError

__all__ = ["resolve_within", "is_within", "relative_to_root", "assert_relative_path"]


def _resolved_root(root: os.PathLike | str) -> Path:
    resolved = Path(root).expanduser().resolve(strict=False)
    if not resolved.is_absolute():  # pragma: no cover - resolve() always absolutises
        raise PathSafetyError(f"workspace root is not absolute: {root!r}")
    return resolved


def is_within(root: os.PathLike | str, candidate: os.PathLike | str) -> bool:
    """True when ``candidate`` resolves inside ``root``. Never raises."""
    try:
        resolve_within(root, candidate)
    except PathSafetyError:
        return False
    return True


def assert_relative_path(raw: str) -> PurePosixPath:
    """Validate the *shape* of a model-supplied path before touching the disk."""
    if not isinstance(raw, str) or not raw.strip():
        raise PathSafetyError("path must be a non-empty string")
    if "\x00" in raw:
        raise PathSafetyError("path contains a NUL byte")
    if raw.startswith(("/", "\\")):
        raise PathSafetyError(f"absolute paths are not accepted: {raw!r}")
    if raw.startswith("~"):
        raise PathSafetyError(f"home-relative paths are not accepted: {raw!r}")
    if len(raw) > 1 and raw[1] == ":":
        raise PathSafetyError(f"drive-qualified paths are not accepted: {raw!r}")
    pure = PurePosixPath(raw.replace("\\", "/"))
    if any(part == ".." for part in pure.parts):
        raise PathSafetyError(f"parent traversal is not accepted: {raw!r}")
    return pure


def resolve_within(
    root: os.PathLike | str,
    candidate: os.PathLike | str,
    *,
    must_exist: bool = False,
    follow_final_symlink: bool = True,
) -> Path:
    """Resolve ``candidate`` relative to ``root`` and assert containment.

    Args:
        root: workspace root. Resolved first, so a symlinked root is fine.
        candidate: relative path supplied by a model, or an absolute path.
        must_exist: require the final path to exist.
        follow_final_symlink: when False, a symlinked *final* component is
            rejected outright rather than followed. Used for writes, so a
            worker cannot overwrite through a symlink it just created.

    Returns:
        The resolved absolute path, guaranteed inside ``root``.
    """
    resolved_root = _resolved_root(root)
    raw = os.fspath(candidate)
    if "\x00" in raw:
        raise PathSafetyError("path contains a NUL byte")

    joined = Path(raw)
    if joined.is_absolute():
        target = joined
    else:
        assert_relative_path(raw)
        target = resolved_root / joined

    # Resolve symlinks and ".." for real. strict=False so not-yet-created files
    # still resolve through their existing parents.
    final = target.resolve(strict=False)

    if final != resolved_root and resolved_root not in final.parents:
        raise PathSafetyError(
            f"path escapes the workspace: {raw!r} resolves to {final} "
            f"which is outside {resolved_root}"
        )

    # A symlink whose *parent chain* resolves inside the root is already safe by
    # the check above, because resolve() followed every link. The remaining case
    # is a final component that is itself a link: for writes we refuse it so the
    # write cannot be redirected after validation.
    if not follow_final_symlink and target.is_symlink():
        raise PathSafetyError(f"refusing to write through a symlink: {raw!r}")

    if must_exist and not final.exists():
        raise PathSafetyError(f"path does not exist: {raw!r}")

    return final


def relative_to_root(root: os.PathLike | str, path: os.PathLike | str) -> str:
    """POSIX-style path of ``path`` relative to ``root``. Asserts containment."""
    resolved_root = _resolved_root(root)
    resolved = resolve_within(resolved_root, path)
    if resolved == resolved_root:
        return "."
    return resolved.relative_to(resolved_root).as_posix()
