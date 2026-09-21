"""Model resolution: the part the reference project got wrong.

The rule
--------

A response is attributed to Fable only when the provider's own transport
metadata names a Fable model. Never because a command exited zero, never
because a CLI was on PATH, never because a default happened to be configured.

Resolution steps, in order:

1. Check the provider is installed and usable at all.
2. Ask it which model identifiers actually exist on this machine. The Claude
   CLI's ids are read out of the installed binary's embedded registry; the
   DeepSeek ids come from its ``/models`` endpoint. Neither is guessed.
3. Select the ids matching the role's pattern, preferring the configured
   order. If nothing matches: **fail closed**, with a message naming the role.
4. Record the resolved id together with the evidence that produced it.

At call time :func:`attest` re-checks the response metadata against the same
pattern, so a provider that quietly served a different model is caught after
the fact as well as before it.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .errors import ModelAttestationError, ModelResolutionError, ProviderUnavailable
from .providers import DiscoveredModel, ModelResponse, Provider, get_provider

__all__ = ["Role", "ResolvedModel", "ModelResolver", "attest"]

RESOLUTION_SCHEMA = 1


class Role:
    PLANNER = "planner"
    WORKER = "worker"


@dataclass(frozen=True)
class ResolvedModel:
    role: str
    display_name: str
    model_id: str
    provider: str
    pattern: str
    source: str
    evidence: str
    resolved_at: float = field(default_factory=time.time)
    #: True when this id was reached through an explicitly enabled fallback.
    via_fallback: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        suffix = "  [EXPLICIT FALLBACK]" if self.via_fallback else ""
        return f"{self.display_name}: {self.model_id} via {self.provider} ({self.source}){suffix}"


class ModelResolver:
    """Resolves each role to a concrete, discovered model identifier."""

    def __init__(self, config: Config, *, providers: "dict[str, Provider] | None" = None,
                 cache_path: Path | None = None) -> None:
        self.config = config
        self._providers: dict[str, Provider] = providers or {}
        self.cache_path = cache_path if cache_path is not None else (
            Path(config.cache_dir) / "resolution.json"
        )

    # -- providers ----------------------------------------------------------

    def provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = get_provider(name, self.config)
        return self._providers[name]

    # -- resolution ---------------------------------------------------------

    def resolve(self, role: str, *, use_cache: bool = True) -> ResolvedModel:
        spec = self._spec(role)
        if use_cache:
            cached = self._read_cache(role, spec)
            if cached is not None:
                return cached
        resolved = self._resolve_uncached(role, spec)
        self._write_cache(role, resolved)
        return resolved

    def resolve_all(self, roles=(Role.PLANNER, Role.WORKER), *, use_cache: bool = True) -> dict:
        return {role: self.resolve(role, use_cache=use_cache) for role in roles}

    def _spec(self, role: str) -> dict:
        models = self.config.models
        if role == Role.PLANNER:
            return {
                "provider": models.planner_provider,
                "pattern": models.planner_pattern,
                "display": models.planner_display,
                "preferred": models.planner_preferred,
                "fallback": models.planner_fallback,
            }
        if role == Role.WORKER:
            return {
                "provider": models.worker_provider,
                "pattern": models.worker_pattern,
                "display": models.worker_display,
                "preferred": models.worker_preferred,
                "fallback": models.worker_fallback,
            }
        raise ModelResolutionError(f"unknown role: {role!r}")

    def _resolve_uncached(self, role: str, spec: dict) -> ResolvedModel:
        provider_name = spec["provider"]
        try:
            provider = self.provider(provider_name)
        except Exception as exc:
            raise ModelResolutionError(
                _fail_message(role, spec, f"provider {provider_name!r} could not be constructed: {exc}")
            ) from exc

        status = provider.status()
        if not status.available:
            raise ModelResolutionError(
                _fail_message(role, spec, f"provider {provider_name!r} is unavailable: {status.detail}")
            )

        try:
            discovered = provider.discover_models()
        except ProviderUnavailable as exc:
            raise ModelResolutionError(
                _fail_message(role, spec, f"model discovery failed: {exc.message}")
            ) from exc

        match = _select(discovered, spec["pattern"], spec["preferred"])
        if match is not None:
            return ResolvedModel(
                role=role,
                display_name=spec["display"],
                model_id=match.id,
                provider=provider_name,
                pattern=spec["pattern"],
                source=match.source,
                evidence=match.evidence,
            )

        fallback = self._try_fallback(role, spec, discovered, provider_name)
        if fallback is not None:
            return fallback

        available = ", ".join(sorted(m.id for m in discovered)) or "none discovered"
        raise ModelResolutionError(
            _fail_message(
                role, spec,
                f"no installed model matches /{spec['pattern']}/. Discovered: {available}"
            )
        )

    def _try_fallback(self, role: str, spec: dict, discovered: "list[DiscoveredModel]",
                      provider_name: str) -> "ResolvedModel | None":
        if not self.config.models.allow_model_fallback:
            return None
        available = {m.id: m for m in discovered}
        for candidate in spec["fallback"]:
            if candidate in available:
                match = available[candidate]
                return ResolvedModel(
                    role=role,
                    display_name=f"{spec['display']} (fallback)",
                    model_id=match.id,
                    provider=provider_name,
                    pattern=re.escape(match.id),
                    source=match.source,
                    evidence=f"explicit user-enabled fallback; {match.evidence}",
                    via_fallback=True,
                )
        return None

    # -- persistence --------------------------------------------------------

    def _cache_fingerprint(self, spec: dict) -> str:
        """Invalidate the cache when the provider or the requirement changes."""
        provider_version = ""
        try:
            provider_version = self.provider(spec["provider"]).status().version or ""
        except Exception:
            provider_version = "unknown"
        return "|".join([
            str(RESOLUTION_SCHEMA), spec["provider"], spec["pattern"],
            ",".join(spec["preferred"]), provider_version,
            "fallback" if self.config.models.allow_model_fallback else "strict",
        ])

    def _read_cache(self, role: str, spec: dict) -> "ResolvedModel | None":
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        entry = (data.get("roles") or {}).get(role)
        if not isinstance(entry, dict):
            return None
        if entry.get("fingerprint") != self._cache_fingerprint(spec):
            return None
        record = entry.get("resolved") or {}
        try:
            return ResolvedModel(**record)
        except TypeError:
            return None

    def _write_cache(self, role: str, resolved: ResolvedModel) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            data.setdefault("schema", RESOLUTION_SCHEMA)
            roles = data.setdefault("roles", {})
            roles[role] = {
                "fingerprint": self._cache_fingerprint(self._spec(role)),
                "resolved": resolved.as_dict(),
            }
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.cache_path)
            os.chmod(self.cache_path, 0o600)
        except OSError:
            return  # a cache miss is never fatal


def _select(discovered: "list[DiscoveredModel]", pattern: str,
            preferred: "tuple[str, ...]") -> "DiscoveredModel | None":
    compiled = re.compile(f"^(?:{pattern})$")
    matches = [m for m in discovered if compiled.match(m.id)]
    if not matches:
        return None
    by_id = {m.id: m for m in matches}
    for candidate in preferred:
        if candidate in by_id:
            return by_id[candidate]
    return sorted(matches, key=lambda m: m.id)[0]


def _fail_message(role: str, spec: dict, reason: str) -> str:
    return (
        f"Required {role} model {spec['display']} could not be resolved.\n"
        f"  reason: {reason}\n"
        f"  fabds will not substitute another model. To allow a specific "
        f"substitute, set models.allow_model_fallback = true and list the exact "
        f"identifier in models.{role}_fallback."
    )


def attest(response: ModelResponse, resolved: ResolvedModel) -> None:
    """Verify after the fact that the expected model served the request."""
    compiled = re.compile(f"^(?:{resolved.pattern})$")
    candidates = [response.reported_model, response.canonical_model]
    if any(c and compiled.match(c) for c in candidates):
        return
    raise ModelAttestationError(
        f"{resolved.display_name} was requested as {resolved.model_id!r}, but the "
        f"provider reported {response.reported_model!r} "
        f"(canonical {response.canonical_model!r}). Refusing to label this output "
        f"as {resolved.display_name}.",
        detail={
            "expected_pattern": resolved.pattern,
            "reported_model": response.reported_model,
            "canonical_model": response.canonical_model,
            "provider": response.provider,
        },
    )
