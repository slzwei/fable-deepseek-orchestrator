"""The permission model.

Three principals, decreasing authority:

===============  ==========================================================
Controller       Codex/Astra. Writes packets, accepts or rejects results,
                 performs integration. Not constrained by this module.
Planner          Fable. No filesystem access, no tools, no commands. Its
                 output is advisory text that the controller may ignore.
Worker           DeepSeek. Acts only through the typed action protocol,
                 inside one workspace, on paths the controller granted.
===============  ==========================================================

A worker's authority is the *intersection* of:

* its packet's ``owned`` globs (write) and ``readonly`` globs (read),
* the workspace root containment check in :mod:`fabds.pathsafety`,
* the global forbidden list below, which no packet can override,
* the packet's command allowlist, which is a list of pre-built argv arrays
  selected by id - a worker names a command, it never composes one.

No path in :data:`GLOBAL_FORBIDDEN` is reachable, because workers are confined
to a workspace root that is never one of those directories. The list is kept
anyway as a second, explicit gate: a controller that mistakenly roots a
workspace at ``$HOME`` still cannot hand out ``.ssh``.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .errors import ConfigError, PermissionDeniedError
from .pathsafety import relative_to_root, resolve_within

__all__ = [
    "GLOBAL_FORBIDDEN",
    "FORBIDDEN_EXECUTABLES",
    "CommandSpec",
    "WorkerPermissions",
    "check_command_spec",
]

#: Paths a worker may never touch, whatever a packet says. Expanded per call so
#: tests can relocate ``$HOME``.
GLOBAL_FORBIDDEN: tuple[str, ...] = (
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.azure",
    "~/.config/gcloud",
    "~/.kube",
    "~/.docker",
    "~/.claude",
    "~/.codex",
    "~/.config/gh",
    "~/.netrc",
    "~/.npmrc",
    "~/.pypirc",
    "~/.git-credentials",
    "~/.password-store",
    "~/Library/Keychains",
    "~/Library/LaunchAgents",
    "~/Library/Application Support/Google/Chrome",
    "~/Library/Application Support/Firefox",
    "/etc",
    "/System",
    "/Library/LaunchAgents",
    "/Library/LaunchDaemons",
    "/usr/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/bin",
    "/sbin",
    "/var/root",
)

#: Executables a worker command may never invoke. Checked on argv[0]'s basename
#: so ``/usr/bin/sudo`` and ``sudo`` are both refused.
FORBIDDEN_EXECUTABLES: frozenset[str] = frozenset({
    "sudo", "su", "doas", "sudoedit",
    "ssh", "scp", "sftp", "rsync", "ssh-add", "ssh-keygen", "ssh-agent",
    "curl", "wget", "nc", "ncat", "netcat", "socat", "telnet", "ftp",
    "launchctl", "systemctl", "service", "crontab", "at",
    "shutdown", "reboot", "halt", "kextload", "csrutil",
    "security", "defaults", "dscl", "diskutil", "mount", "umount",
    "aws", "gcloud", "az", "kubectl", "helm", "terraform", "pulumi",
    "docker", "podman", "colima",
    "sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh",
    "eval", "exec", "env", "xargs", "nohup", "setsid", "script",
    "osascript", "open", "pbcopy", "pbpaste",
    "chown", "chmod", "chgrp", "dd", "mkfs", "fdisk",
    "brew", "apt", "apt-get", "yum", "dnf", "pacman", "port",
})

#: ``git`` is allowed, but only for read-only inspection. Anything that can
#: reach a remote, rewrite history or touch global config is refused.
FORBIDDEN_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "push", "remote", "clone", "fetch", "pull", "submodule",
    "config", "credential", "daemon", "request-pull", "send-email",
    "filter-branch", "filter-repo", "gc", "prune", "reflog",
})

#: Package managers may run, but never with a global-install flag.
GLOBAL_INSTALL_FLAGS: frozenset[str] = frozenset({"-g", "--global", "--location=global"})
PACKAGE_MANAGERS: frozenset[str] = frozenset({"npm", "pnpm", "yarn", "bun", "pip", "pip3", "uv", "gem", "cargo"})


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve(strict=False)


def forbidden_roots() -> list[Path]:
    """Resolve :data:`GLOBAL_FORBIDDEN` against the current ``$HOME``."""
    return [_expand(p) for p in GLOBAL_FORBIDDEN]


def assert_not_globally_forbidden(path: os.PathLike | str) -> None:
    """Refuse a path that lands on or inside a globally forbidden root."""
    resolved = Path(path).expanduser().resolve(strict=False)
    for root in forbidden_roots():
        if resolved == root or root in resolved.parents:
            raise PermissionDeniedError(
                f"path is globally forbidden to workers: {resolved} (under {root})"
            )


@dataclass(frozen=True)
class CommandSpec:
    """A pre-authorised command.

    The controller builds the whole argv. A worker selects it by ``id`` and may
    only append paths when ``max_extra_paths`` is positive - and every appended
    value is validated as a workspace-relative path before it is used. There is
    no code path anywhere in fabds that turns model text into a shell string.
    """

    id: str
    argv: tuple[str, ...]
    description: str = ""
    timeout_s: int = 300
    max_extra_paths: int = 0

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(f"command id must be alphanumeric/_/-: {self.id!r}")
        if not self.argv:
            raise ConfigError(f"command {self.id!r} has an empty argv")
        check_command_spec(self.argv)

    def build_argv(self, extra_paths: list[str], workspace_root: os.PathLike | str) -> list[str]:
        """Return the final argv for execution, validating any extra paths."""
        if len(extra_paths) > self.max_extra_paths:
            raise PermissionDeniedError(
                f"command {self.id!r} accepts at most {self.max_extra_paths} extra "
                f"path(s), got {len(extra_paths)}"
            )
        argv = list(self.argv)
        for raw in extra_paths:
            resolve_within(workspace_root, raw)  # raises PathSafetyError on escape
            argv.append(relative_to_root(workspace_root, raw))
        return argv

    def display(self) -> str:
        return shlex.join(self.argv)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "argv": list(self.argv),
            "description": self.description,
            "timeout_s": self.timeout_s,
            "max_extra_paths": self.max_extra_paths,
        }


def check_command_spec(argv: "tuple[str, ...] | list[str]") -> None:
    """Static policy check on a controller-authored argv array."""
    if not argv:
        raise ConfigError("empty command")
    for part in argv:
        if not isinstance(part, str):
            raise ConfigError(f"command arguments must be strings, got {type(part).__name__}")
        if "\x00" in part:
            raise ConfigError("command argument contains a NUL byte")

    executable = PurePosixPath(argv[0]).name
    if executable in FORBIDDEN_EXECUTABLES:
        raise PermissionDeniedError(f"executable is not permitted for workers: {executable!r}")

    if executable == "git":
        subcommands = [a for a in argv[1:] if not a.startswith("-")]
        if subcommands and subcommands[0] in FORBIDDEN_GIT_SUBCOMMANDS:
            raise PermissionDeniedError(
                f"git subcommand is not permitted for workers: {subcommands[0]!r}"
            )
    if executable in PACKAGE_MANAGERS:
        for arg in argv[1:]:
            if arg in GLOBAL_INSTALL_FLAGS:
                raise PermissionDeniedError(
                    f"global installs are not permitted for workers: {executable} {arg}"
                )
    # Shell metacharacters in argv are inert (we never use a shell), but their
    # presence means the controller probably meant to build a pipeline. Refuse
    # so the mistake is loud rather than silently ineffective.
    for arg in argv:
        if any(ch in arg for ch in ("|", ";", "&&", "||", "`", "$(")) and not arg.startswith("-"):
            raise ConfigError(
                f"command argument looks like shell syntax and would not be interpreted: {arg!r}"
            )


@dataclass
class WorkerPermissions:
    """Effective authority for one worker inside one workspace."""

    workspace_root: Path
    owned: tuple[str, ...] = ()
    readonly: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    read_only: bool = False
    commands: dict[str, CommandSpec] = field(default_factory=dict)
    network: bool = False

    #: Patterns denied for reads in every workspace, on top of the sanitizer.
    default_forbidden: tuple[str, ...] = (
        ".git/config",
        ".git/credentials",
        "**/.env",
        "**/.env.*",
        "**/*.pem",
        "**/*.key",
        "**/id_rsa*",
        "**/id_ed25519*",
        "**/.netrc",
        "**/.npmrc",
        "**/.git-credentials",
    )

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve(strict=False)
        assert_not_globally_forbidden(self.workspace_root)
        if self.read_only and self.owned:
            raise ConfigError("a read-only worker cannot own writable paths")

    # -- matching -----------------------------------------------------------

    @staticmethod
    def _matches(rel: str, patterns: "tuple[str, ...]") -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(rel, pattern):
                return True
            # Treat "src/parser/**" as also matching "src/parser" itself and,
            # because fnmatch's * crosses separators, keep an explicit prefix
            # test for directory-style patterns.
            if pattern.endswith("/**"):
                prefix = pattern[:-3]
                if rel == prefix or rel.startswith(prefix + "/"):
                    return True
            if pattern.endswith("/"):
                if rel.startswith(pattern):
                    return True
        return False

    def _relative(self, path: os.PathLike | str) -> str:
        return relative_to_root(self.workspace_root, path)

    def is_forbidden(self, rel: str) -> bool:
        return self._matches(rel, tuple(self.forbidden) + self.default_forbidden)

    def _is_ancestor_of_grant(self, rel: str) -> bool:
        """True when ``rel`` is a directory containing something we may read.

        A worker granted ``src/parser/**`` must still be able to list ``.`` and
        ``src`` to find it. Listing an ancestor reveals only names, never
        contents, and credential paths are marked excluded in a listing rather
        than shown - so this widens navigation without widening disclosure.
        """
        prefix = "" if rel == "." else rel.rstrip("/") + "/"
        for pattern in tuple(self.owned) + tuple(self.readonly):
            root = pattern.split("*", 1)[0]
            if rel == "." or root.startswith(prefix):
                return True
        return False

    def may_read(self, path: os.PathLike | str) -> bool:
        try:
            rel = self._relative(path)
        except Exception:
            return False
        if self.is_forbidden(rel):
            return False
        if not self.owned and not self.readonly:
            return True  # whole-workspace read grant
        if self._matches(rel, self.owned) or self._matches(rel, self.readonly):
            return True
        # Directories on the way to a granted path are navigable.
        try:
            if Path(path).is_dir() and self._is_ancestor_of_grant(rel):
                return True
        except OSError:
            pass
        return False

    def may_write(self, path: os.PathLike | str) -> bool:
        if self.read_only:
            return False
        try:
            rel = self._relative(path)
        except Exception:
            return False
        if self.is_forbidden(rel):
            return False
        return self._matches(rel, self.owned)

    # -- enforcement --------------------------------------------------------

    def assert_read(self, path: os.PathLike | str) -> Path:
        resolved = resolve_within(self.workspace_root, path)
        if not self.may_read(resolved):
            raise PermissionDeniedError(
                f"read denied: {self._safe_rel(path)} is outside this worker's granted paths"
            )
        return resolved

    def assert_write(self, path: os.PathLike | str) -> Path:
        if self.read_only:
            raise PermissionDeniedError(
                f"write denied: this worker is read-only (attempted {self._safe_rel(path)})"
            )
        resolved = resolve_within(self.workspace_root, path, follow_final_symlink=False)
        if not self.may_write(resolved):
            raise PermissionDeniedError(
                f"write denied: {self._safe_rel(path)} is not owned by this worker "
                f"(owned: {', '.join(self.owned) or 'none'})"
            )
        return resolved

    def assert_command(self, command_id: str) -> CommandSpec:
        spec = self.commands.get(command_id)
        if spec is None:
            available = ", ".join(sorted(self.commands)) or "none"
            raise PermissionDeniedError(
                f"command {command_id!r} is not in this worker's allowlist (available: {available})"
            )
        return spec

    def _safe_rel(self, path: os.PathLike | str) -> str:
        try:
            return self._relative(path)
        except Exception:
            return os.fspath(path)

    def as_dict(self) -> dict:
        return {
            "workspace_root": str(self.workspace_root),
            "owned": list(self.owned),
            "readonly": list(self.readonly),
            "forbidden": list(self.forbidden),
            "read_only": self.read_only,
            "network": self.network,
            "commands": [spec.as_dict() for spec in self.commands.values()],
        }
