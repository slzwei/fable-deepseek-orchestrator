"""Configuration and run limits.

Precedence, lowest to highest: built-in defaults, then a TOML file at
``$FABDS_CONFIG`` or ``~/.config/fabds/config.toml``, then a per-repository
``.fabds/config.toml``, then explicit CLI flags.

No configuration value may loosen a security invariant. Specifically, MCP
isolation, the shell-free execution model, path containment and the global
forbidden list are not configurable. ``allow_model_fallback`` is the single
switch that changes fail-closed behaviour, it defaults to off, and turning it
on is recorded in every run's metadata.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .errors import ConfigError

__all__ = ["Limits", "ModelConfig", "Config", "load_config", "default_max_workers"]


def default_max_workers() -> int:
    """Conservative default: a quarter of the CPUs, clamped to 2..4."""
    cpus = os.cpu_count() or 4
    return max(2, min(4, cpus // 4 or 2))


@dataclass(frozen=True)
class Limits:
    """Hard stops. Every one of these terminates a run rather than degrading."""

    max_planner_rounds: int = 2
    max_worker_rounds: int = 3
    max_workers: int = field(default_factory=default_max_workers)
    max_research_workers: int = 3
    max_implementation_workers: int = 2
    max_test_workers: int = 2
    max_total_tasks: int = 12
    max_retries_per_task: int = 2
    max_worker_turns: int = 12
    worker_timeout_s: int = 600
    planner_timeout_s: int = 900
    command_timeout_s: int = 300
    max_context_chars: int = 120_000
    max_planner_context_chars: int = 60_000
    max_file_excerpt_chars: int = 8_000
    max_response_chars: int = 200_000
    max_run_wall_clock_s: int = 7200
    worker_nesting_enabled: bool = False

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value < 1:
                raise ConfigError(f"limit {name} must be >= 1, got {value}")
        if self.worker_nesting_enabled:
            raise ConfigError(
                "worker nesting is not implemented; only the controller may create workers"
            )
        if self.max_workers > 16:
            raise ConfigError(f"max_workers is capped at 16, got {self.max_workers}")


@dataclass(frozen=True)
class ModelConfig:
    """What each role must resolve to. Patterns, never invented ids."""

    planner_provider: str = "claude_cli"
    #: Regex the resolved planner id must match. Anchored at both ends.
    planner_pattern: str = r"claude-fable-5-1(?:\[[^\]]+\])?"
    planner_display: str = "Fable 5.1"
    #: Preference order used when several installed ids match the pattern.
    planner_preferred: tuple[str, ...] = ("claude-fable-5-1",)

    worker_provider: str = "deepseek_http"
    worker_pattern: str = r"deepseek-flash"
    worker_display: str = "DeepSeek V4.1 Flash"
    worker_preferred: tuple[str, ...] = ("deepseek-flash",)

    #: Off by default. When off, an unavailable required model aborts the run.
    allow_model_fallback: bool = False
    #: Only consulted when allow_model_fallback is true; must be set explicitly.
    planner_fallback: tuple[str, ...] = ()
    worker_fallback: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    limits: Limits = field(default_factory=Limits)
    models: ModelConfig = field(default_factory=ModelConfig)

    #: Cache
    cache_dir: Path = field(default_factory=lambda: _xdg_cache() / "fabds")
    plan_cache_ttl_s: int = 6 * 3600
    analysis_cache_ttl_s: int = 24 * 3600
    cache_enabled: bool = True

    #: Provider endpoints and credentials (paths, never values)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key_file: Path | None = None
    claude_cli: str = "claude"

    #: Context
    extra_secret_paths: tuple[str, ...] = ()
    context_allowlist: tuple[str, ...] = ()

    #: Controller-authored command allowlist, offered to workers by id.
    #: Every entry becomes a CommandSpec; models select ids, never argv.
    commands: tuple[dict, ...] = ()

    source_files: tuple[Path, ...] = ()

    def with_limits(self, **kwargs) -> "Config":
        return replace(self, limits=replace(self.limits, **kwargs))

    def with_models(self, **kwargs) -> "Config":
        return replace(self, models=replace(self.models, **kwargs))

    def command_specs(self):
        """Materialise the configured commands, validating each against policy."""
        from .permissions import CommandSpec

        specs = []
        for entry in self.commands:
            if not isinstance(entry, dict) or "id" not in entry or "argv" not in entry:
                raise ConfigError("each command needs at least `id` and `argv`")
            specs.append(CommandSpec(
                id=str(entry["id"]),
                argv=tuple(str(a) for a in entry["argv"]),
                description=str(entry.get("description", "")),
                timeout_s=int(entry.get("timeout_s", 300)),
                max_extra_paths=int(entry.get("max_extra_paths", 0)),
            ))
        return specs

    def validate(self) -> None:
        self.limits.validate()
        self.command_specs()
        if self.models.allow_model_fallback and not (
            self.models.planner_fallback or self.models.worker_fallback
        ):
            raise ConfigError(
                "allow_model_fallback is on but no fallback ids were configured; "
                "fabds refuses to pick a substitute model for you"
            )
        if not self.deepseek_base_url.startswith("https://"):
            raise ConfigError(
                f"deepseek_base_url must be https, got {self.deepseek_base_url!r}"
            )

    def as_dict(self) -> dict:
        out = asdict(self)
        out["cache_dir"] = str(self.cache_dir)
        out["deepseek_api_key_file"] = str(self.deepseek_api_key_file) if self.deepseek_api_key_file else None
        out["source_files"] = [str(p) for p in self.source_files]
        return out


def _xdg_cache() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))


def _config_candidates(repo_root: Path | None) -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("FABDS_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    else:
        xdg_config = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
        candidates.append(xdg_config / "fabds" / "config.toml")
    if repo_root is not None:
        candidates.append(Path(repo_root) / ".fabds" / "config.toml")
    return candidates


_SCALAR_FIELDS = {
    "cache_enabled", "plan_cache_ttl_s", "analysis_cache_ttl_s",
    "deepseek_base_url", "claude_cli",
}


def load_config(repo_root: Path | None = None, overrides: dict | None = None) -> Config:
    """Build the effective configuration."""
    config = Config()
    loaded: list[Path] = []

    for path in _config_candidates(repo_root):
        if not path.is_file():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        config = _apply(config, data, path)
        loaded.append(path)

    if overrides:
        config = _apply(config, overrides, Path("<cli>"))

    config = replace(config, source_files=tuple(loaded))

    if config.deepseek_api_key_file is None:
        env_file = os.environ.get("DEEPSEEK_API_KEY_FILE")
        if env_file:
            config = replace(config, deepseek_api_key_file=Path(env_file).expanduser())

    config.validate()
    return config


def _apply(config: Config, data: dict, source: Path) -> Config:
    unknown: list[str] = []

    limits_data = data.get("limits", {})
    if limits_data:
        known = {f for f in asdict(config.limits)}
        unknown += [f"limits.{k}" for k in limits_data if k not in known]
        config = replace(config, limits=replace(
            config.limits, **{k: v for k, v in limits_data.items() if k in known}))

    models_data = data.get("models", {})
    if models_data:
        known = {f for f in asdict(config.models)}
        unknown += [f"models.{k}" for k in models_data if k not in known]
        coerced = {}
        for key, value in models_data.items():
            if key not in known:
                continue
            coerced[key] = tuple(value) if isinstance(value, list) else value
        config = replace(config, models=replace(config.models, **coerced))

    top = {k: v for k, v in data.items() if k not in ("limits", "models")}
    updates: dict = {}
    for key, value in top.items():
        if key in _SCALAR_FIELDS:
            updates[key] = value
        elif key == "cache_dir":
            updates[key] = Path(str(value)).expanduser()
        elif key == "deepseek_api_key_file":
            updates[key] = Path(str(value)).expanduser()
        elif key in ("extra_secret_paths", "context_allowlist"):
            updates[key] = tuple(value)
        elif key == "commands":
            if not isinstance(value, list):
                raise ConfigError("`commands` must be a list of tables")
            updates[key] = tuple(value)
        else:
            unknown.append(key)
    if updates:
        config = replace(config, **updates)

    if unknown:
        raise ConfigError(
            f"unknown configuration key(s) in {source}: {', '.join(sorted(unknown))}"
        )
    return config
