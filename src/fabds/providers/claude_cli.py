"""Fable via the installed Claude Code CLI.

Isolation contract for every call made here
-------------------------------------------

=========================  ==================================================
``--strict-mcp-config``    ignore every MCP configuration except the one we
                           pass explicitly
``--mcp-config {}``        and the one we pass is empty unless the controller
                           deliberately granted a server
``--safe-mode``            no CLAUDE.md, skills, plugins, hooks, custom agents,
                           output styles or workflows
``--setting-sources ""``   no user, project or local settings files
``--tools ""``             the planner gets no tools at all: it reads nothing,
                           writes nothing and runs nothing
``--permission-prompts``   ``none`` - nothing can be approved, by anyone
``--no-session-persistence`` the planning conversation is not written to disk
scrubbed environment       no ``*KEY*``/``*TOKEN*``/``*SECRET*`` variables, no
                           ``CLAUDE_CODE_*`` bridge back to the parent session
neutral working directory  an empty temp dir, so nothing in the user's repo is
                           auto-discovered
=========================  ==================================================

Model identity is taken from the CLI's ``modelUsage`` block, which reports the
model the request was actually billed against. It is never inferred from the
fact that the command exited zero.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from ..errors import (
    EmptyResponse,
    MalformedResponse,
    ModelAttestationError,
    ProviderCrashed,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from ..runner import build_child_env, run_command
from .base import CompletionRequest, DiscoveredModel, ModelResponse, ProviderStatus

__all__ = ["ClaudeCliProvider", "extract_model_ids", "parse_cli_result"]

#: Model ids embedded in the CLI binary look exactly like this.
_MODEL_ID_RE = re.compile(rb"claude-(?:opus|sonnet|haiku|fable)-[0-9][0-9a-zA-Z._-]{0,40}")
_SCAN_CHUNK = 8 * 1024 * 1024
_SCAN_OVERLAP = 64

#: Errors the CLI reports before it ever reaches the API.
_UNRECOGNISED_RE = re.compile(r"unrecognized_model|unknown model|invalid model", re.I)
_RATE_LIMIT_RE = re.compile(r"reached your .* limit|rate.?limit|usage credits|quota", re.I)
_CONTEXT_RE = re.compile(r"context (?:window|length)|too (?:long|large)|prompt is too", re.I)


def extract_model_ids(binary_path: Path, *, limit_bytes: int | None = None) -> set[str]:
    """Scan an installed CLI binary for the model identifiers it knows about.

    This is the discovery step the spec asks for: the ids come out of the
    artifact that is actually installed, so fabds never has to guess or
    hard-code one. Pure standard library so it works without ``strings(1)``.
    """
    found: set[str] = set()
    try:
        size = binary_path.stat().st_size
    except OSError as exc:
        raise ProviderUnavailable(f"cannot stat {binary_path}: {exc}") from exc

    budget = size if limit_bytes is None else min(size, limit_bytes)
    read = 0
    tail = b""
    try:
        with binary_path.open("rb") as fh:
            while read < budget:
                chunk = fh.read(min(_SCAN_CHUNK, budget - read))
                if not chunk:
                    break
                read += len(chunk)
                for match in _MODEL_ID_RE.finditer(tail + chunk):
                    found.add(match.group(0).decode("ascii"))
                tail = chunk[-_SCAN_OVERLAP:]
    except OSError as exc:
        raise ProviderUnavailable(f"cannot read {binary_path}: {exc}") from exc
    return found


def parse_cli_result(stdout: str) -> dict:
    """Parse ``--output-format json``. The CLI may prefix diagnostic lines."""
    stdout = stdout.strip()
    if not stdout:
        raise EmptyResponse("claude CLI produced no output")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    # Recover the last complete JSON object on any line.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise MalformedResponse(
        "claude CLI output was not JSON", detail=stdout[-2000:]
    )


class ClaudeCliProvider:
    """Planner provider. Advisory text only - this provider grants no tools."""

    name = "claude_cli"

    def __init__(self, config) -> None:
        self.config = config
        self.executable = config.claude_cli
        self._model_cache: set[str] | None = None

    # -- discovery ----------------------------------------------------------

    def _resolved_binary(self) -> Path | None:
        which = shutil.which(self.executable)
        if not which:
            return None
        return Path(which).resolve(strict=False)

    def status(self) -> ProviderStatus:
        which = shutil.which(self.executable)
        if not which:
            return ProviderStatus(
                self.name, False, detail=f"{self.executable!r} is not on PATH"
            )
        result = run_command([which, "--version"], timeout_s=30)
        if not result.ok:
            return ProviderStatus(
                self.name, False, detail=f"`{self.executable} --version` failed: "
                f"{result.stderr.strip()[:200] or result.stdout.strip()[:200]}"
            )
        return ProviderStatus(
            self.name,
            True,
            version=result.stdout.strip().splitlines()[0] if result.stdout.strip() else None,
            detail=str(which),
            isolation="--strict-mcp-config with an empty server set, --safe-mode, "
                      "--setting-sources '', --tools '', scrubbed env, neutral cwd",
        )

    def discover_models(self) -> list[DiscoveredModel]:
        binary = self._resolved_binary()
        if binary is None:
            raise ProviderUnavailable(f"{self.executable!r} is not on PATH")

        models: list[DiscoveredModel] = []
        try:
            ids = extract_model_ids(binary)
        except ProviderUnavailable:
            ids = set()
        self._model_cache = ids
        for model_id in sorted(ids):
            models.append(
                DiscoveredModel(
                    id=model_id,
                    source="registry",
                    evidence=f"found in the installed CLI binary at {binary}",
                )
            )
        return models

    def probe_model(self, model_id: str) -> bool:
        """Local validity probe: does this CLI recognise the identifier?

        The CLI rejects unknown model names before any network request, so this
        costs nothing. A recognised name still has to pass attestation at call
        time; this only rules ids out.
        """
        which = shutil.which(self.executable)
        if not which:
            return False
        argv = self._base_argv(which, model_id) + ["--max-budget-usd", "0.01"]
        result = run_command(
            argv, stdin_text="ping", timeout_s=120, env=self._child_env(), cwd=tempfile.gettempdir()
        )
        blob = f"{result.stdout}\n{result.stderr}"
        return not _UNRECOGNISED_RE.search(blob)

    # -- invocation ---------------------------------------------------------

    def _child_env(self) -> dict:
        # HOME is required for the CLI's OAuth credentials; everything that
        # smells like a credential is dropped by build_child_env itself.
        return build_child_env()

    def _base_argv(self, executable: str, model_id: str, *,
                   mcp_grants: "dict | None" = None) -> list[str]:
        """Build the isolated invocation.

        ``--safe-mode`` disables *all* customisation, MCP servers included, and
        that override beats ``--mcp-config``: with it on, even an explicitly
        granted server does not start (verified against the installed CLI). So
        when the controller deliberately grants a server, ``--safe-mode`` is
        dropped, because silently ignoring a requested capability would be worse
        than the narrower isolation. Everything else - strict MCP config, no
        settings inheritance, no tools, neutral cwd, scrubbed environment -
        stays in force either way, and the response records
        ``mcp_isolated=False`` so the reduction is visible downstream.
        """
        argv = [
            executable,
            "--print",
            "--output-format", "json",
            "--model", model_id,
        ]
        if not mcp_grants:
            argv.append("--safe-mode")
        argv += [
            "--strict-mcp-config",
            "--setting-sources", "",
            "--tools", "",
            "--permission-mode", "manual",
            "--permission-prompts", "none",
            "--no-session-persistence",
        ]
        return argv

    def complete(self, request: CompletionRequest) -> ModelResponse:
        which = shutil.which(self.executable)
        if not which:
            raise ProviderUnavailable(f"{self.executable!r} is not on PATH")

        argv = self._base_argv(which, request.model_id, mcp_grants=request.mcp_grants)
        if request.system_prompt:
            argv += ["--system-prompt", request.system_prompt]

        with tempfile.TemporaryDirectory(prefix="fabds-planner-") as neutral_cwd:
            mcp_payload = json.dumps({"mcpServers": request.mcp_grants or {}})
            argv += ["--mcp-config", mcp_payload]

            started = time.monotonic()
            result = run_command(
                argv,
                cwd=neutral_cwd,       # empty dir: nothing of the user's is discoverable
                env=self._child_env(),
                stdin_text=request.user_prompt,
                timeout_s=request.timeout_s,
            )

        duration = time.monotonic() - started
        blob = f"{result.stdout}\n{result.stderr}"

        if result.timed_out:
            raise ProviderTimeout(
                f"{self.name} exceeded {request.timeout_s:.0f}s for {request.label}"
            )
        if _UNRECOGNISED_RE.search(blob):
            raise ProviderUnavailable(
                f"the installed Claude CLI does not recognise model {request.model_id!r}",
                detail=blob[-800:],
            )

        payload = parse_cli_result(result.stdout)
        self._raise_for_payload_error(payload, request, blob)

        text = payload.get("result")
        if not isinstance(text, str) or not text.strip():
            raise EmptyResponse(
                f"{request.model_id} returned no text", detail=str(payload)[:800]
            )

        reported, canonical, usage = self._attest(payload, request.model_id)
        return ModelResponse(
            text=text,
            reported_model=reported,
            canonical_model=canonical,
            provider=self.name,
            duration_s=duration,
            usage=usage,
            cost_usd=payload.get("total_cost_usd"),
            raw_metadata={
                "session_id": payload.get("session_id"),
                "stop_reason": payload.get("stop_reason"),
                "num_turns": payload.get("num_turns"),
                "terminal_reason": payload.get("terminal_reason"),
                "permission_denials": payload.get("permission_denials"),
            },
            mcp_isolated=not request.mcp_grants,
        )

    def _raise_for_payload_error(self, payload: dict, request: CompletionRequest, blob: str) -> None:
        if not payload.get("is_error"):
            return
        message = str(payload.get("result") or payload.get("error") or "")
        status = payload.get("api_error_status")
        if status == 429 or _RATE_LIMIT_RE.search(message):
            raise ProviderRateLimited(
                f"{request.model_id} is rate limited or out of quota: {message.strip()}",
                detail={"api_error_status": status},
            )
        if _CONTEXT_RE.search(message):
            from ..errors import ContextTooLarge

            raise ContextTooLarge(
                f"{request.model_id} rejected the prompt size: {message.strip()}"
            )
        if _UNRECOGNISED_RE.search(message) or _UNRECOGNISED_RE.search(blob):
            raise ProviderUnavailable(
                f"model {request.model_id!r} is not available: {message.strip()}"
            )
        raise ProviderCrashed(
            f"{request.model_id} call failed: {message.strip() or 'unknown error'}",
            detail={"api_error_status": status, "terminal_reason": payload.get("terminal_reason")},
        )

    @staticmethod
    def _attest(payload: dict, requested: str) -> tuple[str, str, dict]:
        """Read the billed model out of ``modelUsage``.

        A successful exit code proves nothing about which model answered. This
        block is written by the CLI from the API response, so it is the only
        thing we trust.
        """
        usage_block = payload.get("modelUsage")
        if not isinstance(usage_block, dict) or not usage_block:
            raise ModelAttestationError(
                f"the CLI returned no modelUsage block, so the model that served "
                f"this request cannot be verified (requested {requested!r})",
                detail={"keys": sorted(payload.keys())},
            )
        if len(usage_block) > 1:
            raise ModelAttestationError(
                f"more than one model served this request: {sorted(usage_block)}; "
                "fabds requires a single attested planner model",
                detail={"models": sorted(usage_block)},
            )
        reported = next(iter(usage_block))
        entry = usage_block[reported] or {}
        canonical = entry.get("canonicalModel") or reported
        usage = {
            "input_tokens": entry.get("inputTokens"),
            "output_tokens": entry.get("outputTokens"),
            "thinking_tokens": entry.get("thinkingTokens"),
            "cache_read_input_tokens": entry.get("cacheReadInputTokens"),
            "cache_creation_input_tokens": entry.get("cacheCreationInputTokens"),
            "context_window": entry.get("contextWindow"),
            "api_provider": entry.get("provider"),
        }
        return reported, canonical, usage
