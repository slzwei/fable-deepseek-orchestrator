"""Deterministic provider for tests and benchmarks.

Backed by a directory of JSON fixtures. In ``record`` mode it delegates to a
real provider and writes what came back; in ``replay`` mode it serves fixtures
and never touches the network. Benchmarks use replay so their numbers are
reproducible, and the report says plainly which mode produced them.

This provider is never selected by ``fabds run`` unless the caller asks for it
by name, so it cannot be mistaken for a silent fallback.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..errors import ProviderUnavailable
from .base import CompletionRequest, DiscoveredModel, ModelResponse, ProviderStatus

__all__ = ["ReplayProvider", "fixture_key"]


def fixture_key(request: CompletionRequest) -> str:
    digest = hashlib.sha256()
    for part in (request.model_id, request.system_prompt, request.user_prompt,
                 str(request.max_output_tokens), request.reasoning_effort):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


class ReplayProvider:
    name = "replay"

    def __init__(self, config, fixture_dir: Path | None = None, *,
                 delegate=None, mode: str = "replay",
                 declared_models: "list[str] | None" = None) -> None:
        self.config = config
        self.fixture_dir = Path(
            fixture_dir or getattr(config, "replay_fixture_dir", None) or "fixtures"
        )
        self.delegate = delegate
        self.mode = mode
        self.declared_models = declared_models or []
        self.calls: list[CompletionRequest] = []

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            self.name,
            self.fixture_dir.is_dir() or self.mode == "record",
            detail=f"{self.mode} from {self.fixture_dir}",
            isolation="no external process and no network in replay mode",
        )

    def discover_models(self) -> list[DiscoveredModel]:
        manifest = self.fixture_dir / "models.json"
        ids = list(self.declared_models)
        if manifest.is_file():
            ids += json.loads(manifest.read_text(encoding="utf-8"))
        return [
            DiscoveredModel(id=model_id, source="probe", evidence=f"replay fixture {manifest}")
            for model_id in dict.fromkeys(ids)
        ]

    def complete(self, request: CompletionRequest) -> ModelResponse:
        self.calls.append(request)
        path = self.fixture_dir / f"{fixture_key(request)}.json"

        if self.mode == "record":
            if self.delegate is None:
                raise ProviderUnavailable("record mode needs a delegate provider")
            response = self.delegate.complete(request)
            self.fixture_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "text": response.text,
                "reported_model": response.reported_model,
                "canonical_model": response.canonical_model,
                "provider": response.provider,
                "usage": response.usage,
                "cost_usd": response.cost_usd,
                "duration_s": response.duration_s,
            }, indent=2), encoding="utf-8")
            return response

        if not path.is_file():
            raise ProviderUnavailable(
                f"no replay fixture for {request.label!r} (model {request.model_id}); "
                f"expected {path}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0)  # replay is instantaneous by design
        return ModelResponse(
            text=data["text"],
            reported_model=data["reported_model"],
            canonical_model=data.get("canonical_model", data["reported_model"]),
            provider=data.get("provider", self.name),
            duration_s=float(data.get("duration_s", 0.0)),
            usage=data.get("usage", {}),
            cost_usd=data.get("cost_usd"),
            raw_metadata={"replayed_from": str(path)},
            mcp_isolated=True,
        )
