"""Typed failure modes.

Every error carries a stable ``code`` so the CLI, the ledger and the tests can
agree on what happened without string matching on messages.
"""

from __future__ import annotations


class FabdsError(Exception):
    """Base class for every fabds failure."""

    code = "fabds_error"
    #: Whether retrying the identical operation could plausibly succeed.
    retryable = False

    def __init__(self, message: str, *, detail: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigError(FabdsError):
    code = "config_error"


class ModelResolutionError(FabdsError):
    """A required model could not be resolved. Always fail closed."""

    code = "model_resolution_failed"


class ModelAttestationError(FabdsError):
    """A response arrived, but provider metadata did not name the expected model."""

    code = "model_attestation_failed"


class ProviderError(FabdsError):
    code = "provider_error"


class ProviderUnavailable(ProviderError):
    code = "provider_unavailable"


class ProviderRateLimited(ProviderError):
    code = "provider_rate_limited"
    retryable = True


class ProviderTimeout(ProviderError):
    code = "provider_timeout"
    retryable = True


class ProviderCrashed(ProviderError):
    code = "provider_crashed"
    retryable = True


class ContextTooLarge(FabdsError):
    code = "context_too_large"


class MalformedResponse(FabdsError):
    code = "malformed_response"
    retryable = True


class EmptyResponse(MalformedResponse):
    code = "empty_response"
    retryable = True


class ResponseTruncated(MalformedResponse):
    """The model hit its output budget mid-reply."""

    code = "response_truncated"
    retryable = True


class PathSafetyError(FabdsError):
    code = "path_safety_violation"


class PermissionDeniedError(FabdsError):
    code = "permission_denied"


class LimitExceeded(FabdsError):
    code = "limit_exceeded"


class WorkspaceError(FabdsError):
    code = "workspace_error"


class PeakHoursBlocked(FabdsError):
    """A call was refused because it would have been billed at peak rates."""

    code = "peak_hours_blocked"
