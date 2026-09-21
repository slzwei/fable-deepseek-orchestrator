"""Shared fixtures.

Nothing in the default test run touches a network or spends money. Tests that
need a real provider are marked ``live`` and deselected unless asked for.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fabds.config import Config, load_config  # noqa: E402
from fabds.logging import NullLogger  # noqa: E402
from fabds.providers.base import (  # noqa: E402
    CompletionRequest,
    DiscoveredModel,
    ModelResponse,
    ProviderStatus,
)


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip_live = pytest.mark.skip(reason="live provider test; run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


class FakeProvider:
    """A scriptable provider. Records every request it is given."""

    def __init__(self, name="fake", *, models=(), responses=(), available=True,
                 raises=None, reported_model=None):
        self.name = name
        self._models = list(models)
        self._responses = list(responses)
        self._available = available
        self._raises = list(raises or [])
        self.reported_model = reported_model
        self.requests: list[CompletionRequest] = []

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, self._available, version="test",
                              detail="fake provider", isolation="none needed")

    def discover_models(self) -> list[DiscoveredModel]:
        return [DiscoveredModel(m, "probe", "fake fixture") for m in self._models]

    def complete(self, request: CompletionRequest) -> ModelResponse:
        self.requests.append(request)
        if self._raises:
            error = self._raises.pop(0)
            if error is not None:
                raise error
        text = self._responses.pop(0) if self._responses else "{}"
        if callable(text):
            text = text(request)
        reported = self.reported_model or request.model_id
        return ModelResponse(
            text=text, reported_model=reported, canonical_model=reported,
            provider=self.name, duration_s=0.01, usage={"input_tokens": 1, "output_tokens": 1},
            cost_usd=0.0, mcp_isolated=True,
        )


@pytest.fixture
def null_logger():
    return NullLogger()


@pytest.fixture
def config(tmp_path) -> Config:
    return replace(
        load_config(),
        cache_dir=tmp_path / "cache",
        deepseek_api_key_file=tmp_path / "ds-key",
    )


@pytest.fixture
def git_repo(tmp_path) -> Path:
    """A real git repository with one commit."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "tests" / "test_calc.py").write_text(
        "from src.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    for argv in (
        ["git", "init", "-b", "main", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Test"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "initial"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


@pytest.fixture
def worker_script():
    """Build a canned worker transcript: a list of JSON turn strings."""

    def build(*turns: dict) -> list[str]:
        return [json.dumps(turn) for turn in turns]

    return build
