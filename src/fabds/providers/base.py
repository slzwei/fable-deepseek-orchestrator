"""Provider interface and the value objects that cross it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "ProviderStatus",
    "DiscoveredModel",
    "CompletionRequest",
    "ModelResponse",
    "Provider",
]


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    available: bool
    version: str | None = None
    detail: str = ""
    #: Human-readable description of how isolation is achieved for this provider.
    isolation: str = ""


@dataclass(frozen=True)
class DiscoveredModel:
    """One model identifier found on this machine.

    ``source`` records *how* we learned about it, so ``fabds resolve`` can show
    evidence rather than a bare claim:

    ``registry``  read out of the installed CLI's own embedded model table
    ``api``       returned by the provider's list-models endpoint
    ``probe``     accepted by a local validation probe
    """

    id: str
    source: str
    evidence: str = ""


@dataclass
class CompletionRequest:
    """One bounded request to one model."""

    system_prompt: str
    user_prompt: str
    model_id: str
    max_output_tokens: int = 4096
    temperature: float | None = None
    timeout_s: float = 600.0
    #: Provider-specific reasoning knob, normalised to low/high/max.
    reasoning_effort: str = "high"
    #: MCP servers to deliberately grant. Empty means: none, strictly.
    mcp_grants: dict = field(default_factory=dict)
    #: Optional label used only in logs.
    label: str = "request"


@dataclass
class ModelResponse:
    """A completion plus the evidence of who produced it."""

    text: str
    #: The identifier the *provider* reported, from transport metadata.
    reported_model: str
    #: Normalised form of the above (e.g. a dated id mapped to its family id).
    canonical_model: str
    provider: str
    duration_s: float
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    raw_metadata: dict = field(default_factory=dict)
    #: True when the process that produced this had no MCP servers configured.
    mcp_isolated: bool = True

    def attestation(self) -> dict:
        return {
            "provider": self.provider,
            "reported_model": self.reported_model,
            "canonical_model": self.canonical_model,
            "mcp_isolated": self.mcp_isolated,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "duration_s": round(self.duration_s, 3),
        }


@runtime_checkable
class Provider(Protocol):
    name: str

    def status(self) -> ProviderStatus: ...

    def discover_models(self) -> list[DiscoveredModel]: ...

    def complete(self, request: CompletionRequest) -> ModelResponse: ...
