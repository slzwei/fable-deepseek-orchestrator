"""Model providers.

Each provider knows three things and nothing else:

* whether it is installed and usable (:meth:`Provider.status`),
* which model identifiers are *actually* present on this machine
  (:meth:`Provider.discover_models`) - discovered, never assumed,
* how to run one completion and report, from transport metadata, which model
  actually served it (:meth:`Provider.complete`).
"""

from .base import (
    CompletionRequest,
    DiscoveredModel,
    ModelResponse,
    Provider,
    ProviderStatus,
)

__all__ = [
    "CompletionRequest",
    "DiscoveredModel",
    "ModelResponse",
    "Provider",
    "ProviderStatus",
    "get_provider",
]


def get_provider(name: str, config):
    """Instantiate a provider by configuration name."""
    from ..errors import ConfigError

    if name == "claude_cli":
        from .claude_cli import ClaudeCliProvider

        return ClaudeCliProvider(config)
    if name == "deepseek_http":
        from .deepseek_http import DeepSeekHttpProvider

        return DeepSeekHttpProvider(config)
    if name == "replay":
        from .replay import ReplayProvider

        return ReplayProvider(config)
    raise ConfigError(f"unknown provider: {name!r}")
