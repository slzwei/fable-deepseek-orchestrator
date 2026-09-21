"""Context sanitisation.

Nothing reaches a model without passing through here. Two independent gates:

**Path exclusion.** Files whose *name or location* marks them as credential
bearing are never opened, so their contents cannot leak even through a
redaction bug. This covers dotenv files, key material, cloud and package
manager credentials, browser profile data, keychains, password stores and
Terraform state.

**Content redaction.** Whatever does get included is run through
:mod:`fabds.redaction`, so a key pasted into an otherwise innocent source file
is still removed.

An operator can override an exclusion with an explicit allowlist entry. That is
deliberate: sometimes you really do need to show a model a config file. It has
to be asked for by path, it is logged, and content redaction still applies.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .redaction import REDACTOR, RedactionReport

__all__ = [
    "SECRET_PATH_PATTERNS",
    "SKIP_DIRECTORIES",
    "ContextSanitizer",
    "SanitizedFile",
    "ExclusionReason",
]


class ExclusionReason:
    SECRET_PATH = "secret_path"
    BINARY = "binary"
    TOO_LARGE = "too_large"
    SKIPPED_DIR = "skipped_directory"
    UNREADABLE = "unreadable"
    OUTSIDE_ROOT = "outside_root"


#: Matched against the workspace-relative POSIX path *and* the bare filename.
SECRET_PATH_PATTERNS: tuple[str, ...] = (
    # dotenv and friends
    ".env", ".env.*", "*.env", "env.local", ".envrc",
    # generic credential names
    "*secret*", "*secrets*", "credentials", "credentials.*", "*.credentials",
    "*password*", "*passwd*", "*.token", "token.txt", "*apikey*", "*api_key*",
    "*-api-key", "*_api_key", "ds-api-key",
    # key material
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.asc", "*.gpg",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "*.ppk",
    # tool-specific credential stores
    ".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials", ".vault-token",
    ".htpasswd", ".dockercfg", "*service-account*.json", "*serviceaccount*.json",
    "gha-creds-*.json", ".terraformrc", "terraform.tfstate", "terraform.tfstate.*",
    "*.tfvars", "*.kdbx", "*.agilekeychain", "*.keychain", "*.keychain-db",
    # per-directory stores
    ".ssh/*", "**/.ssh/*", ".aws/*", "**/.aws/*", ".gnupg/*", "**/.gnupg/*",
    ".password-store/**", "**/.password-store/**",
    ".kube/config", "**/.kube/config", ".docker/config.json", "**/.docker/config.json",
    ".config/gcloud/**", "**/.config/gcloud/**", ".config/gh/hosts.yml",
    # browser data
    "Cookies", "Cookies-journal", "Login Data", "Login Data-journal",
    "key3.db", "key4.db", "logins.json", "cert9.db",
    # git internals that carry credentials
    ".git/config", "**/.git/config", ".git-credentials",
)

#: Directories never walked when building a repository tree.
SKIP_DIRECTORIES: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv", "env", ".env.d",
    "dist", "build", "target", ".next", ".nuxt", ".parcel-cache", ".cache",
    "coverage", ".nyc_output", ".gradle", "Pods", "DerivedData", ".terraform",
    ".idea", ".vscode", ".fabds", ".DS_Store", ".ssh", ".gnupg", ".aws",
    ".password-store", "site-packages", ".git-crypt",
})

_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff", ".icns",
    ".pdf", ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar", ".dmg", ".pkg",
    ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".class", ".jar",
    ".pyc", ".pyo", ".wasm", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac", ".webm",
    ".sqlite", ".sqlite3", ".db", ".mdb", ".pack", ".idx",
})


@dataclass
class SanitizedFile:
    """One file considered for inclusion in a context packet."""

    path: str
    included: bool
    reason: str = ""
    text: str = ""
    redactions: RedactionReport = field(default_factory=RedactionReport)
    truncated: bool = False
    size_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "included": self.included,
            "reason": self.reason,
            "redactions": self.redactions.as_dict(),
            "truncated": self.truncated,
            "size_bytes": self.size_bytes,
        }


class ContextSanitizer:
    """Decides what a model may see, and scrubs what it does."""

    def __init__(
        self,
        root: Path,
        *,
        extra_secret_patterns: "tuple[str, ...]" = (),
        allowlist: "tuple[str, ...]" = (),
        max_file_chars: int = 8_000,
        max_file_bytes: int = 512_000,
    ) -> None:
        self.root = Path(root).resolve(strict=False)
        self.secret_patterns = SECRET_PATH_PATTERNS + tuple(extra_secret_patterns)
        self.allowlist = tuple(allowlist)
        self.max_file_chars = max_file_chars
        self.max_file_bytes = max_file_bytes
        self.decisions: list[SanitizedFile] = []

    # -- classification -----------------------------------------------------

    def is_allowlisted(self, rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pattern) for pattern in self.allowlist)

    def is_secret_path(self, rel: str) -> bool:
        """True when a path is credential bearing by name or location."""
        pure = PurePosixPath(rel)
        name = pure.name
        for pattern in self.secret_patterns:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                return True
            if pattern.endswith("/**") and rel.startswith(pattern[:-3] + "/"):
                return True
        # Any component that is itself a credential directory.
        for part in pure.parts[:-1]:
            if part in {".ssh", ".gnupg", ".aws", ".password-store", ".git-crypt"}:
                return True
        return False

    def should_skip_dir(self, name: str) -> bool:
        return name in SKIP_DIRECTORIES

    # -- reading ------------------------------------------------------------

    def read(self, path: Path, *, max_chars: int | None = None) -> SanitizedFile:
        """Read one file for inclusion, or explain why it was excluded."""
        max_chars = self.max_file_chars if max_chars is None else max_chars
        try:
            rel = path.resolve(strict=False).relative_to(self.root).as_posix()
        except ValueError:
            return self._record(SanitizedFile(str(path), False, ExclusionReason.OUTSIDE_ROOT))

        if self.is_secret_path(rel) and not self.is_allowlisted(rel):
            return self._record(SanitizedFile(rel, False, ExclusionReason.SECRET_PATH))
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return self._record(SanitizedFile(rel, False, ExclusionReason.BINARY))
        if any(self.should_skip_dir(part) for part in PurePosixPath(rel).parts[:-1]):
            return self._record(SanitizedFile(rel, False, ExclusionReason.SKIPPED_DIR))

        try:
            size = path.stat().st_size
        except OSError as exc:
            return self._record(SanitizedFile(rel, False, f"{ExclusionReason.UNREADABLE}: {exc.strerror}"))
        if size > self.max_file_bytes:
            return self._record(SanitizedFile(rel, False, ExclusionReason.TOO_LARGE, size_bytes=size))

        try:
            raw = path.read_bytes()
        except OSError as exc:
            return self._record(SanitizedFile(rel, False, f"{ExclusionReason.UNREADABLE}: {exc.strerror}"))
        if b"\x00" in raw[:8192]:
            return self._record(SanitizedFile(rel, False, ExclusionReason.BINARY, size_bytes=size))

        text = raw.decode("utf-8", errors="replace")
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated by fabds at {max_chars} chars] ..."
            truncated = True

        text, report = REDACTOR.redact(text)
        return self._record(SanitizedFile(
            rel, True, "allowlisted" if self.is_allowlisted(rel) else "",
            text=text, redactions=report, truncated=truncated, size_bytes=size,
        ))

    def sanitize_text(self, text: str) -> tuple[str, RedactionReport]:
        """Redact an arbitrary blob (command output, a diff, a log)."""
        return REDACTOR.redact(text)

    # -- tree ---------------------------------------------------------------

    def tree(self, *, max_entries: int = 400, max_depth: int = 6) -> list[str]:
        """A compact, secret-free listing of the repository."""
        entries: list[str] = []
        root = self.root

        def walk(directory: Path, depth: int) -> None:
            if depth > max_depth or len(entries) >= max_entries:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
            except OSError:
                return
            for child in children:
                if len(entries) >= max_entries:
                    return
                if child.name.startswith(".") and child.name not in {".github", ".fabds"}:
                    if child.is_dir():
                        continue
                if child.is_dir():
                    if self.should_skip_dir(child.name):
                        continue
                    rel = child.resolve(strict=False)
                    try:
                        rel_str = rel.relative_to(root).as_posix()
                    except ValueError:
                        continue  # symlink pointing outside the tree
                    entries.append(rel_str + "/")
                    walk(child, depth + 1)
                else:
                    try:
                        rel_str = child.resolve(strict=False).relative_to(root).as_posix()
                    except ValueError:
                        continue
                    if self.is_secret_path(rel_str) and not self.is_allowlisted(rel_str):
                        entries.append(f"{rel_str}  [excluded: credential path]")
                        continue
                    entries.append(rel_str)

        walk(root, 0)
        return entries

    # -- reporting ----------------------------------------------------------

    def _record(self, decision: SanitizedFile) -> SanitizedFile:
        self.decisions.append(decision)
        return decision

    def report(self) -> dict:
        excluded = [d for d in self.decisions if not d.included]
        redactions = RedactionReport()
        for decision in self.decisions:
            redactions.merge(decision.redactions)
        return {
            "files_considered": len(self.decisions),
            "files_included": len(self.decisions) - len(excluded),
            "files_excluded": len(excluded),
            "excluded_paths": [
                {"path": d.path, "reason": d.reason} for d in excluded
            ][:100],
            "redactions": redactions.as_dict(),
        }
