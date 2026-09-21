"""Model resolution must be real, evidenced and fail closed."""

from __future__ import annotations

import pytest
from dataclasses import replace

from conftest import FakeProvider
from fabds.errors import ModelResolutionError
from fabds.models import ModelResolver, Role


def resolver_with(config, planner_models=(), worker_models=(), **model_kwargs):
    config = replace(config, models=replace(config.models, **model_kwargs))
    providers = {
        config.models.planner_provider: FakeProvider("claude_cli", models=planner_models),
        config.models.worker_provider: FakeProvider("deepseek_http", models=worker_models),
    }
    return ModelResolver(config, providers=providers,
                         cache_path=config.cache_dir / "resolution.json")


def test_resolves_fable_from_discovered_ids(config):
    resolver = resolver_with(
        config,
        planner_models=["claude-opus-5", "claude-fable-5-1", "claude-sonnet-5"],
        worker_models=["deepseek-flash", "deepseek-v4-pro"],
    )
    planner = resolver.resolve(Role.PLANNER)
    assert planner.model_id == "claude-fable-5-1"
    assert planner.display_name == "Fable 5.1"
    assert planner.evidence, "a resolution must carry evidence of how it was found"
    assert planner.via_fallback is False

    worker = resolver.resolve(Role.WORKER)
    assert worker.model_id == "deepseek-flash"


def test_never_substitutes_another_claude_model(config):
    """Opus and Sonnet present, Fable absent -> fail, do not pick a neighbour."""
    resolver = resolver_with(
        config,
        planner_models=["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
        worker_models=["deepseek-flash"],
    )
    with pytest.raises(ModelResolutionError) as excinfo:
        resolver.resolve(Role.PLANNER)
    message = str(excinfo.value)
    assert "Fable 5.1" in message
    assert "will not substitute" in message
    assert "claude-opus-5" in message  # it reports what it did find


def test_never_substitutes_another_deepseek_model(config):
    resolver = resolver_with(
        config,
        planner_models=["claude-fable-5-1"],
        worker_models=["deepseek-v4-pro", "deepseek-chat"],
    )
    with pytest.raises(ModelResolutionError) as excinfo:
        resolver.resolve(Role.WORKER)
    assert "DeepSeek V4.1 Flash" in str(excinfo.value)


def test_unavailable_provider_fails_closed(config):
    config = replace(config, models=replace(config.models))
    providers = {"claude_cli": FakeProvider("claude_cli", available=False),
                 "deepseek_http": FakeProvider("deepseek_http", models=["deepseek-flash"])}
    resolver = ModelResolver(config, providers=providers,
                             cache_path=config.cache_dir / "r.json")
    with pytest.raises(ModelResolutionError, match="unavailable"):
        resolver.resolve(Role.PLANNER)


def test_fallback_is_off_by_default_and_requires_explicit_ids(config):
    resolver = resolver_with(
        config, planner_models=["claude-opus-5"], worker_models=["deepseek-flash"],
        planner_fallback=("claude-opus-5",),
    )
    with pytest.raises(ModelResolutionError):
        resolver.resolve(Role.PLANNER)  # allow_model_fallback is still False


def test_explicit_fallback_is_marked_when_enabled(config):
    resolver = resolver_with(
        config, planner_models=["claude-opus-5"], worker_models=["deepseek-flash"],
        allow_model_fallback=True, planner_fallback=("claude-opus-5",),
    )
    planner = resolver.resolve(Role.PLANNER)
    assert planner.model_id == "claude-opus-5"
    assert planner.via_fallback is True
    assert "fallback" in planner.display_name.lower()
    assert "EXPLICIT FALLBACK" in planner.describe()


def test_preference_order_is_honoured(config):
    resolver = resolver_with(
        config,
        planner_models=["claude-fable-5-1", "claude-fable-5-1[1m]"],
        worker_models=["deepseek-flash"],
        planner_preferred=("claude-fable-5-1[1m]", "claude-fable-5-1"),
    )
    assert resolver.resolve(Role.PLANNER).model_id == "claude-fable-5-1[1m]"


def test_resolution_cache_invalidates_when_requirements_change(config):
    resolver = resolver_with(config, planner_models=["claude-fable-5-1"],
                             worker_models=["deepseek-flash"])
    assert resolver.resolve(Role.PLANNER).model_id == "claude-fable-5-1"

    stricter = resolver_with(config, planner_models=["claude-fable-5-1"],
                             worker_models=["deepseek-flash"],
                             planner_pattern=r"claude-fable-9")
    stricter.cache_path = resolver.cache_path
    with pytest.raises(ModelResolutionError):
        stricter.resolve(Role.PLANNER)  # the cached answer must not be reused
