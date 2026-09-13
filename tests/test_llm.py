"""The request payload must stay valid for the model each provider talks to."""

import pytest

from pr_agent.config import Settings
from pr_agent.llm import build_chat_model

GOOGLE_KEY = "test-key-no-request-is-made"
ANTHROPIC_KEY = "sk-ant-dummy-no-request-is-made"


def _gemini(**overrides):
    return build_chat_model(Settings(google_api_key=GOOGLE_KEY, **overrides))


def _claude_payload(**overrides):
    model = build_chat_model(
        Settings(provider="anthropic", anthropic_api_key=ANTHROPIC_KEY, **overrides)
    )
    return model, model._get_request_payload([{"role": "user", "content": "hi"}])


# --- defaults --------------------------------------------------------------


def test_gemini_is_the_default_provider():
    model = _gemini()
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model == "gemini-3.8-flash"


def test_each_provider_has_its_own_default_model():
    model, _payload = _claude_payload()
    assert model.model == "claude-opus-5"


def test_a_model_from_the_other_provider_is_rejected():
    with pytest.raises(ValueError, match="does not belong to provider"):
        Settings(model="claude-opus-5")


# --- gemini ----------------------------------------------------------------


def test_gemini_lets_the_model_choose_its_thinking_level():
    # No budget and no level: a fixed budget would be rejected by Gemini 3.
    model = _gemini()
    assert model.thinking_config == {"include_thoughts": True}
    assert model.thinking_budget is None


def test_gemini_thinking_can_be_turned_off():
    assert _gemini(thinking=False).thinking_config == {"thinking_budget": 0}


def test_gemini_model_and_limits_come_from_settings():
    model = _gemini(model="gemini-3.5-flash", max_tokens=4096, request_timeout=30.0)
    assert model.model == "gemini-3.5-flash"
    assert model.max_output_tokens == 4096
    assert model.timeout == 30.0


def test_gemini_key_is_taken_from_the_gemini_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini-var")
    assert Settings().google_api_key == "from-gemini-var"


# --- claude ----------------------------------------------------------------


def test_claude_sampling_parameters_are_never_sent():
    # temperature / top_p / top_k are rejected with a 400 by current models.
    _model, payload = _claude_payload()
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload


def test_claude_thinking_is_adaptive_not_a_fixed_budget():
    # The fixed budget_tokens form is rejected by current models.
    _model, payload = _claude_payload()
    assert payload["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in payload.get("thinking", {})


def test_claude_thinking_can_be_turned_off():
    _model, payload = _claude_payload(thinking=False)
    assert "thinking" not in payload


def test_claude_model_and_max_tokens_come_from_settings():
    model, payload = _claude_payload(model="claude-sonnet-5", max_tokens=4096)
    assert model.model == "claude-sonnet-5"
    assert payload["max_tokens"] == 4096
