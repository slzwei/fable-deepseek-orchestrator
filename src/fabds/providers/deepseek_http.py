"""DeepSeek V4.1 Flash over HTTPS.

Workers talk to the DeepSeek API directly rather than through an agentic CLI.
That is a deliberate security decision: an HTTP completion endpoint has no
filesystem, no shell and no MCP surface, so there is nothing for a worker to
escape *from*. Every effect a worker has on the world goes through the typed
action protocol in :mod:`fabds.workers`, which the controller validates.

Hardening applied here:

* TLS certificate verification is on and cannot be disabled by configuration.
* The URL opener ignores ``*_proxy`` environment variables, so a hostile
  environment cannot silently redirect prompts through a logging proxy.
* Only ``https://`` base URLs are accepted (enforced in :mod:`fabds.config`).
* The API key is read from a file at call time, registered with the process
  redactor the moment it is read, and never placed in argv, logs or caches.
* Model identity comes from the response body's ``model`` field.
"""

from __future__ import annotations

import json
import random
import ssl
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path

from ..errors import (
    ContextTooLarge,
    PeakHoursBlocked,
    EmptyResponse,
    MalformedResponse,
    ModelAttestationError,
    ProviderCrashed,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ResponseTruncated,
)
from ..pricing import describe, parse_holidays, status
from ..redaction import REDACTOR
from .base import CompletionRequest, DiscoveredModel, ModelResponse, ProviderStatus

__all__ = ["DeepSeekHttpProvider"]

_USER_AGENT = "fabds/orchestrator (+stdlib urllib)"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

#: Wait-loop slice bounds. Short enough to notice a clock change, long
#: enough not to busy-spin through a multi-hour window.
_MIN_SLICE_S = 1.0
_MAX_SLICE_S = 60.0


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """An opener that ignores environment proxies and verifies certificates.

    Passing an explicit ``ProxyHandler({})`` suppresses the default handler that
    would otherwise read ``HTTPS_PROXY`` and friends from the environment. Note
    that the empty handler defines no ``*_open`` methods and so does not appear
    in ``opener.handlers`` - that absence is the point, not a bug. Do not
    "simplify" this to ``build_opener()``: that reinstates env proxy support and
    lets a hostile environment variable intercept every prompt.
    """
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),           # {} == no proxies, ignore env
        urllib.request.HTTPSHandler(context=context),
    )


class DeepSeekHttpProvider:
    """Worker provider. Bounded, stateless, one completion per call."""

    name = "deepseek_http"

    def __init__(self, config, *, sleep=time.sleep, clock=None) -> None:
        self.config = config
        self.base_url = config.deepseek_base_url.rstrip("/")
        self.key_file = Path(config.deepseek_api_key_file) if config.deepseek_api_key_file else None
        self._opener = _no_proxy_opener()
        self._sleep = sleep
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._holidays = parse_holidays(getattr(config, "offpeak_extra_dates", ()))
        #: Set by the orchestrator so waits are visible rather than silent.
        self.on_wait = None

    # -- off-peak gate ------------------------------------------------------

    def _await_offpeak(self) -> None:
        """Block until the request would be billed off-peak.

        The gate lives here, at the transport boundary, rather than in the
        orchestrator. That is deliberate: the guarantee the operator asked for
        is that the prompt does not go out at peak rates, and a check in the
        orchestrator only holds for callers who go through the orchestrator. A
        library caller, a test, or a future code path would bypass it. Nothing
        reaches DeepSeek without passing this function.
        """
        if not getattr(self.config, "deepseek_offpeak_only", False):
            return
        if getattr(self.config, "deepseek_allow_peak", False):
            return  # explicit operator override; logged by the caller

        state = status(self._clock(), holidays=self._holidays)
        if not state.peak:
            return

        limit = getattr(self.config, "deepseek_offpeak_max_wait_s", 5 * 3600)
        if state.wait_seconds > limit:
            raise PeakHoursBlocked(
                f"DeepSeek is in a peak window until {state.next_offpeak:%H:%M}Z "
                f"({state.wait_seconds / 3600:.1f}h away), which exceeds the "
                f"{limit / 3600:.1f}h wait limit. Raise "
                f"deepseek_offpeak_max_wait_s, or pass --peak-ok to accept "
                f"full-rate billing.",
                detail=state.as_dict(),
            )

        if self.on_wait is not None:
            self.on_wait(state)

        # Sleep in short slices, re-evaluating the window each time, so a clock
        # adjustment during the wait is noticed rather than ignored.
        #
        # Termination is guaranteed by three independent bounds, because any one
        # of them alone can be defeated:
        #
        #   * the window itself - the normal exit;
        #   * real elapsed time, from a monotonic clock, so a wall clock that is
        #     frozen or stepped backwards cannot strand the loop;
        #   * an iteration cap, so even a no-op sleep (a test double, a stubbed
        #     scheduler) still terminates.
        #
        # Spinning forever would be a worse failure than paying peak rates,
        # because it is silent. All three paths raise instead.
        # Wait until the *margin-adjusted* target, not merely until is_peak()
        # flips. Exiting the instant the window closes means firing a request at
        # exactly 04:00:00, and how the server classifies that on arrival is not
        # ours to decide. The margin is the whole point of computing a target.
        target = state.next_offpeak
        started_monotonic = time.monotonic()
        max_iterations = int(limit // _MIN_SLICE_S) + 60
        iterations = 0

        while True:
            now = self._clock()
            state = status(now, holidays=self._holidays)
            if not state.peak and now >= target:
                return

            iterations += 1
            elapsed = time.monotonic() - started_monotonic
            if elapsed > limit or iterations > max_iterations:
                raise PeakHoursBlocked(
                    f"gave up waiting for an off-peak window after "
                    f"{elapsed / 3600:.1f}h of real time and {iterations} checks; "
                    f"it is still peak at {now:%H:%M}Z. Either the clock is not "
                    f"advancing or the window never opened. Pass --peak-ok to "
                    f"proceed at full rate.",
                    detail=state.as_dict(),
                )

            remaining = max(state.wait_seconds, (target - now).total_seconds())
            self._sleep(max(_MIN_SLICE_S, min(_MAX_SLICE_S, remaining)))

    # -- credentials --------------------------------------------------------

    def _api_key(self) -> str:
        if self.key_file is None:
            raise ProviderUnavailable(
                "no DeepSeek key file configured; set deepseek_api_key_file or "
                "the DEEPSEEK_API_KEY_FILE environment variable"
            )
        try:
            key = self.key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ProviderUnavailable(
                f"cannot read DeepSeek key file {self.key_file}: {exc.strerror}"
            ) from exc
        if not key:
            raise ProviderUnavailable(f"DeepSeek key file {self.key_file} is empty")
        # From here on, this exact value can never appear in a log or a cache.
        REDACTOR.register_literal(key)
        return key

    # -- discovery ----------------------------------------------------------

    def status(self) -> ProviderStatus:
        if self.key_file is None:
            return ProviderStatus(self.name, False, detail="no key file configured")
        if not self.key_file.is_file():
            return ProviderStatus(self.name, False, detail=f"key file missing: {self.key_file}")
        mode = self.key_file.stat().st_mode & 0o777
        warning = "" if mode & 0o077 == 0 else f" (warning: key file mode {mode:o} is group/world readable)"
        gate = ""
        if getattr(self.config, "deepseek_offpeak_only", False):
            gate = (" | off-peak gate: OVERRIDDEN (--peak-ok)"
                    if getattr(self.config, "deepseek_allow_peak", False)
                    else f" | off-peak gate on: {describe(holidays=self._holidays)}")
        return ProviderStatus(
            self.name,
            True,
            detail=f"{self.base_url}, key file {self.key_file}{warning}{gate}",
            isolation="direct HTTPS: no shell, no filesystem, no MCP surface; "
                      "proxies from the environment are ignored",
        )

    def discover_models(self) -> list[DiscoveredModel]:
        payload = self._request("GET", "/models", None, timeout_s=30)
        data = payload.get("data")
        if not isinstance(data, list):
            raise MalformedResponse("DeepSeek /models did not return a list")
        models = []
        for entry in data:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                models.append(
                    DiscoveredModel(
                        id=entry["id"],
                        source="api",
                        evidence=f"listed by {self.base_url}/models"
                                 f" (owned_by={entry.get('owned_by', 'unknown')})",
                    )
                )
        return models

    # -- invocation ---------------------------------------------------------

    def complete(self, request: CompletionRequest) -> ModelResponse:
        self._await_offpeak()
        thinking = request.reasoning_effort not in ("none", "off")
        body = {
            "model": request.model_id,
            "messages": self._messages(request),
            "max_tokens": max(1, min(int(request.max_output_tokens), 8192)),
            "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"},
            "reasoning_effort": request.reasoning_effort if thinking else "none",
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature

        started = time.monotonic()
        payload = self._request("POST", "/chat/completions", body, timeout_s=request.timeout_s)
        duration = time.monotonic() - started

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise EmptyResponse("DeepSeek returned no choices", detail=str(payload)[:500])
        message = choices[0].get("message") or {}
        text = _extract_text(message)
        finish = choices[0].get("finish_reason")
        if not text.strip():
            if finish == "length":
                # The whole budget went on reasoning, leaving no answer. Worth a
                # retry with a hint; a flat "empty response" would hide the cause.
                raise ResponseTruncated(
                    "DeepSeek used its entire output budget before producing an "
                    "answer (finish_reason=length)"
                )
            raise EmptyResponse(
                f"DeepSeek returned empty content (finish_reason={finish})"
            )

        reported = payload.get("model")
        if not isinstance(reported, str) or not reported:
            raise ModelAttestationError(
                "DeepSeek response carried no model field, so the model that "
                "served this request cannot be verified"
            )
        return ModelResponse(
            text=text,
            reported_model=reported,
            canonical_model=reported,
            provider=self.name,
            duration_s=duration,
            usage=payload.get("usage") or {},
            cost_usd=None,
            raw_metadata={
                "id": payload.get("id"),
                "finish_reason": finish,
                "endpoint": self.base_url,
            },
            mcp_isolated=True,  # structurally: this transport has no MCP
        )

    @staticmethod
    def _messages(request: CompletionRequest) -> list[dict]:
        messages = []
        if request.system_prompt.strip():
            messages.append({"role": "system", "content": request.system_prompt.strip()})
        messages.append({"role": "user", "content": request.user_prompt})
        return messages

    # -- transport ----------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None, *, timeout_s: float,
                 max_attempts: int = 3) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            headers = {
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            }
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self._opener.open(req, timeout=timeout_s) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return _parse_json(raw)
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                error = self._classify_http(exc.code, raw)
                if exc.code in _RETRY_STATUS and attempt < max_attempts:
                    last_error = error
                    time.sleep(_backoff(attempt))
                    continue
                raise error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, (TimeoutError,)) or "timed out" in str(reason).lower():
                    error: Exception = ProviderTimeout(
                        f"DeepSeek request timed out after {timeout_s:.0f}s"
                    )
                else:
                    error = ProviderUnavailable(f"cannot reach {url}: {reason}")
                if attempt < max_attempts:
                    last_error = error
                    time.sleep(_backoff(attempt))
                    continue
                raise error from exc

        raise last_error or ProviderCrashed("DeepSeek request failed with no error recorded")

    @staticmethod
    def _classify_http(code: int, raw: str) -> Exception:
        message = raw.strip()
        try:
            parsed = json.loads(raw)
            message = str(parsed.get("error", {}).get("message") or message)
        except (json.JSONDecodeError, AttributeError):
            pass
        message = REDACTOR.scrub(message)[:400]
        if code == 429:
            return ProviderRateLimited(f"DeepSeek rate limited (HTTP 429): {message}")
        if code in (401, 403):
            return ProviderUnavailable(f"DeepSeek rejected the credentials (HTTP {code}): {message}")
        if code == 404:
            return ProviderUnavailable(f"DeepSeek endpoint or model not found (HTTP 404): {message}")
        if code == 413 or "context" in message.lower() or "too long" in message.lower():
            return ContextTooLarge(f"DeepSeek rejected the prompt size (HTTP {code}): {message}")
        if 500 <= code < 600:
            return ProviderCrashed(f"DeepSeek server error (HTTP {code}): {message}")
        return ProviderCrashed(f"DeepSeek returned HTTP {code}: {message}")


def _backoff(attempt: int) -> float:
    """Bounded exponential backoff with jitter. Never unbounded, never zero."""
    return min(8.0, (2 ** (attempt - 1)) * 1.0) * (0.7 + 0.6 * random.random())


def _parse_json(raw: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedResponse(
            "DeepSeek returned a non-JSON body", detail=REDACTOR.scrub(raw)[:500]
        ) from exc
    if not isinstance(payload, dict):
        raise MalformedResponse("DeepSeek returned a non-object JSON body")
    return payload


def _extract_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "output_text")
        ]
        return "".join(parts)
    return ""
