"""The request payload must stay valid for current Claude models."""

from pr_agent.config import Settings
from pr_agent.llm import build_chat_model

KEY = "sk-ant-dummy-no-request-is-made"


def _payload(**overrides):
    model = build_chat_model(Settings(anthropic_api_key=KEY, **overrides))
    return model, model._get_request_payload([{"role": "user", "content": "hi"}])


def test_sampling_parameters_are_never_sent():
    # temperature / top_p / top_k are rejected with a 400 by current models.
    _model, payload = _payload()
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload


def test_thinking_is_adaptive_not_a_fixed_budget():
    # The fixed budget_tokens form is rejected by current models.
    _model, payload = _payload()
    assert payload["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in payload.get("thinking", {})


def test_thinking_can_be_turned_off():
    _model, payload = _payload(thinking=False)
    assert "thinking" not in payload


def test_model_and_max_tokens_come_from_settings():
    model, payload = _payload(model="claude-sonnet-5", max_tokens=4096)
    assert model.model == "claude-sonnet-5"
    assert payload["max_tokens"] == 4096


def test_default_model_is_opus_5():
    model, _payload_ = _payload()
    assert model.model == "claude-opus-5"
