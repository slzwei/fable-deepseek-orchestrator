"""DeepSeek selection and attestation.

Note the lesson that shaped this file: asked to identify itself, the real
``deepseek-flash`` endpoint replies "I'm Claude 3.5 Sonnet, made by Anthropic."
Model self-report is worthless as evidence. Identity comes from the endpoint
that was called and the ``model`` field the API returns.
"""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace

import pytest

from fabds.errors import (
    ContextTooLarge,
    ModelAttestationError,
    ProviderRateLimited,
    ProviderUnavailable,
)
from fabds.models import ResolvedModel, attest
from fabds.providers import deepseek_http
from fabds.providers.base import CompletionRequest

def _synth(*parts: str) -> str:
    """Assemble a credential-shaped fixture at runtime.

    Every value built here is fake. The reason they are not written as literals
    is that a string shaped like a real key trips secret scanners - GitHub push
    protection, GitGuardian, and whatever a person who clones this runs - on a
    repository whose entire purpose is preventing credential leaks. Joining the
    parts produces the identical string at runtime, so the redaction paths are
    exercised exactly as before while the file on disk matches nothing.
    """
    return "".join(parts)


#: A fake key, assembled so the file contains no key-shaped literal.
TEST_KEY = _synth("sk", "-testkey-", "0123456789abcdef")

DEEPSEEK = ResolvedModel(
    role="worker", display_name="DeepSeek V4.1 Flash", model_id="deepseek-flash",
    provider="deepseek_http", pattern=r"deepseek-flash", source="api", evidence="test",
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def provider(config, tmp_path):
    key_file = tmp_path / "ds-key"
    key_file.write_text(TEST_KEY, encoding="utf-8")
    return deepseek_http.DeepSeekHttpProvider(replace(config, deepseek_api_key_file=key_file))


def stub_open(provider, monkeypatch, payload: dict):
    captured: dict = {}

    def fake_open(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode()) if request.data else None
        captured["timeout"] = timeout
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(provider._opener, "open", fake_open)
    return captured


def chat_payload(model: str, text: str = "done") -> dict:
    return {
        "id": "x", "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def test_request_names_deepseek_flash_explicitly(provider, monkeypatch):
    captured = stub_open(provider, monkeypatch, chat_payload("deepseek-flash"))
    provider.complete(CompletionRequest("sys", "hi", "deepseek-flash"))
    assert captured["body"]["model"] == "deepseek-flash"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"].startswith("Bearer ")


def test_metadata_reports_the_real_model(provider, monkeypatch):
    stub_open(provider, monkeypatch, chat_payload("deepseek-flash"))
    response = provider.complete(CompletionRequest("sys", "hi", "deepseek-flash"))
    assert response.reported_model == "deepseek-flash"
    assert response.provider == "deepseek_http"
    attest(response, DEEPSEEK)


def test_another_deepseek_model_is_never_relabelled(provider, monkeypatch):
    stub_open(provider, monkeypatch, chat_payload("deepseek-v4-pro"))
    response = provider.complete(CompletionRequest("sys", "hi", "deepseek-flash"))
    with pytest.raises(ModelAttestationError, match="deepseek-v4-pro"):
        attest(response, DEEPSEEK)


def test_model_self_report_is_not_identity(provider, monkeypatch):
    """The body claims to be Claude. The attestation still reads the API field."""
    stub_open(provider, monkeypatch,
              chat_payload("deepseek-flash", "I'm Claude 3.5 Sonnet, made by Anthropic."))
    response = provider.complete(CompletionRequest("sys", "who are you?", "deepseek-flash"))
    assert "Claude" in response.text
    assert response.reported_model == "deepseek-flash"
    attest(response, DEEPSEEK)  # identity is transport metadata, not prose


def test_missing_model_field_is_an_attestation_failure(provider, monkeypatch):
    payload = chat_payload("deepseek-flash")
    payload.pop("model")
    stub_open(provider, monkeypatch, payload)
    with pytest.raises(ModelAttestationError, match="cannot be verified"):
        provider.complete(CompletionRequest("sys", "hi", "deepseek-flash"))


def test_missing_key_file_fails_closed(config, tmp_path):
    provider = deepseek_http.DeepSeekHttpProvider(
        replace(config, deepseek_api_key_file=tmp_path / "absent"))
    assert provider.status().available is False
    with pytest.raises(ProviderUnavailable, match="cannot read DeepSeek key file"):
        provider.complete(CompletionRequest("sys", "hi", "deepseek-flash"))


def test_api_key_is_registered_for_redaction(provider, monkeypatch):
    from fabds.redaction import REDACTOR

    stub_open(provider, monkeypatch, chat_payload("deepseek-flash"))
    provider.complete(CompletionRequest("sys", "hi", "deepseek-flash"))
    leaked = f"the key is {TEST_KEY} right here"
    assert TEST_KEY not in REDACTOR.scrub(leaked)


def test_proxy_environment_is_ignored(monkeypatch):
    """A hostile HTTPS_PROXY must not be able to intercept prompts.

    Passing ``ProxyHandler({})`` to ``build_opener`` suppresses the default,
    environment-reading ProxyHandler. An empty handler defines no ``*_open``
    methods, so it does not appear in ``handlers`` at all - the correct
    assertion is that *no* proxy mapping survives, not that a handler is
    present. The control below shows a default opener would have been hijacked.
    """
    import urllib.request

    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid:8080")

    control = urllib.request.build_opener()
    control_proxies = [h.proxies for h in control.handlers
                       if type(h).__name__ == "ProxyHandler"]
    assert any("attacker.invalid" in str(p) for p in control_proxies), (
        "control: a default opener should have picked the proxy up")

    hardened = deepseek_http._no_proxy_opener()
    hardened_proxies = [h.proxies for h in hardened.handlers
                        if type(h).__name__ == "ProxyHandler"]
    assert not any(hardened_proxies), "the hardened opener must carry no proxies"
    assert all("attacker.invalid" not in str(p) for p in hardened_proxies)


def test_http_errors_map_to_typed_failures(provider, monkeypatch):
    cases = [
        (429, '{"error":{"message":"too many"}}', ProviderRateLimited),
        (401, '{"error":{"message":"bad key"}}', ProviderUnavailable),
        (413, '{"error":{"message":"context too long"}}', ContextTooLarge),
    ]
    for status, body, expected in cases:
        def fake_open(request, timeout=None, _status=status, _body=body):
            raise urllib.error.HTTPError(
                request.full_url, _status, "err", {}, io.BytesIO(_body.encode()))

        monkeypatch.setattr(provider._opener, "open", fake_open)
        with pytest.raises(expected):
            provider._request("POST", "/chat/completions", {}, timeout_s=1, max_attempts=1)


@pytest.mark.live
def test_live_deepseek_reports_its_model(config):
    """Spends a fraction of a cent. Run with: pytest -m live"""
    from fabds.config import load_config as _load

    key_file = _load().deepseek_api_key_file
    if key_file is None or not key_file.is_file():
        pytest.skip("no DeepSeek key file configured")
    provider = deepseek_http.DeepSeekHttpProvider(
        replace(config, deepseek_api_key_file=key_file))
    ids = {m.id for m in provider.discover_models()}
    assert "deepseek-flash" in ids
    response = provider.complete(CompletionRequest(
        "Answer in one word.", "What is 2+2?", "deepseek-flash",
        max_output_tokens=32, reasoning_effort="none", timeout_s=60))
    assert response.reported_model == "deepseek-flash"
    attest(response, DEEPSEEK)
