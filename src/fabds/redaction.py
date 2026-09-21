"""Secret redaction applied to everything that leaves the process.

Two layers:

1. **Pattern redaction** - well-known credential shapes (provider keys, PEM
   blocks, JWTs, ``NAME=secret`` assignments, credential URLs).
2. **Literal redaction** - exact secret values the process has learned at
   runtime, e.g. the DeepSeek API key read off disk. A literal can never be
   printed even if it does not match any pattern.

Both layers run on model prompts, model responses, log records and ledger
entries. Redaction is deliberately biased toward false positives: losing a
hex blob from a prompt is cheaper than leaking a key.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Iterable, Pattern

PLACEHOLDER = "[REDACTED:{label}]"

# Ordered most-specific first; the first match wins for a given span.
_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("private-key-block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL)),
    ("openssh-key-block", re.compile(
        r"-----BEGIN OPENSSH PRIVATE KEY-----.*?-----END OPENSSH PRIVATE KEY-----",
        re.DOTALL)),
    ("pgp-key-block", re.compile(
        r"-----BEGIN PGP PRIVATE KEY BLOCK-----.*?-----END PGP PRIVATE KEY BLOCK-----",
        re.DOTALL)),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{12,}")),
    ("openai-project-key", re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{12,}")),
    ("generic-sk-key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("stripe-key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("github-token", re.compile(
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("slack-webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/+_-]{10,}")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("google-oauth-id", re.compile(r"\b[0-9]{10,}-[a-z0-9]{32}\.apps\.googleusercontent\.com\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{30,}")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("credential-url", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s/:@]+:[^\s/@]+@[^\s]+")),
    ("authorization-header", re.compile(
        r"(?i)\b(authorization|proxy-authorization)\s*[:=]\s*[\"']?(?:bearer|basic|token)\s+[A-Za-z0-9._\-+/=]{8,}")),
    # NAME=value where NAME smells like a credential. Value may be quoted.
    ("secret-assignment", re.compile(
        r"(?im)^[ \t]*(?:export[ \t]+)?"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*"
        r"(?:SECRET|PASSWORD|PASSWD|TOKEN|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|CREDENTIALS?|SESSION[_-]?KEY|CLIENT[_-]?SECRET|DSN))"
        r"[ \t]*[:=][ \t]*(?P<value>\"[^\"\n]*\"|'[^'\n]*'|[^\s#]+)")),
]

_MIN_LITERAL_LEN = 8


@dataclass
class RedactionReport:
    """What redaction did to one blob of text."""

    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def bump(self, label: str, n: int = 1) -> None:
        if n:
            self.counts[label] = self.counts.get(label, 0) + n

    def merge(self, other: "RedactionReport") -> None:
        for label, n in other.counts.items():
            self.bump(label, n)

    def as_dict(self) -> dict:
        return {"total": self.total, "by_kind": dict(sorted(self.counts.items()))}


class Redactor:
    """Thread-safe redactor with a runtime-extensible literal set."""

    def __init__(self, literals: Iterable[str] = ()) -> None:
        self._lock = threading.Lock()
        self._literals: set[str] = set()
        for literal in literals:
            self.register_literal(literal)

    def register_literal(self, value: str | None, *, label: str = "runtime-literal") -> None:
        """Register an exact secret value that must never be emitted."""
        if not value:
            return
        value = value.strip()
        if len(value) < _MIN_LITERAL_LEN:
            return
        with self._lock:
            self._literals.add(value)
        self._label_for_literal = label

    def register_file_contents(self, path) -> None:
        """Register the contents of a credential file (best effort, never raises)."""
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as fh:
                self.register_literal(fh.read())
        except (OSError, UnicodeDecodeError):
            return

    def redact(self, text: str) -> tuple[str, RedactionReport]:
        report = RedactionReport()
        if not text:
            return text, report

        with self._lock:
            literals = sorted(self._literals, key=len, reverse=True)
        for literal in literals:
            if literal in text:
                count = text.count(literal)
                text = text.replace(literal, PLACEHOLDER.format(label="literal"))
                report.bump("literal", count)

        for label, pattern in _PATTERNS:
            def _sub(match: re.Match, _label: str = label) -> str:
                if _label == "secret-assignment":
                    # Preserve the variable name; only the value is secret.
                    return f"{match.group('name')}={PLACEHOLDER.format(label=_label)}"
                return PLACEHOLDER.format(label=_label)

            text, n = pattern.subn(_sub, text)
            report.bump(label, n)

        return text, report

    def scrub(self, text: str) -> str:
        """Redact and discard the report. For log formatting."""
        return self.redact(text)[0]

    def contains_literal(self, text: str) -> bool:
        with self._lock:
            return any(lit in text for lit in self._literals)


#: Process-wide redactor. Providers register their credentials here on load so
#: that no later log line, prompt or ledger entry can echo them.
REDACTOR = Redactor()


def redact(text: str) -> tuple[str, RedactionReport]:
    return REDACTOR.redact(text)


def scrub(text: str) -> str:
    return REDACTOR.scrub(text)
