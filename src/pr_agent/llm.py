"""Chat model construction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .config import Settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

log = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    """Raised when the configured provider's optional dependency is missing."""


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build the chat model used by every node in the graph.

    Both providers are driven through the standard LangChain interface —
    ``with_structured_output`` and ``bind_tools`` — so the graph itself does
    not care which one is configured.
    """
    if settings.provider == "google":
        return _build_gemini(settings)
    return _build_claude(settings)


def _build_gemini(settings: Settings) -> BaseChatModel:
    """Gemini via the Generative Language API."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ProviderUnavailable(
            "The 'google' provider needs langchain-google-genai: "
            "pip install 'pr-review-implementor'"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "request_timeout": settings.request_timeout,
    }
    if settings.google_api_key:
        kwargs["api_key"] = settings.google_api_key

    if settings.thinking:
        # No explicit level or budget: Gemini picks how much to think per
        # request, and the thoughts come back as blocks the nodes filter out.
        kwargs["thinking_config"] = {"include_thoughts": True}
    else:
        # Gemini 2.x honours a zero budget; Gemini 3 models always think and
        # reject it, which is why thinking is left on by default.
        kwargs["thinking_config"] = {"thinking_budget": 0}

    return ChatGoogleGenerativeAI(**kwargs)


def _build_claude(settings: Settings) -> BaseChatModel:
    """Claude via the Anthropic API.

    Note two current-model constraints: sampling parameters (``temperature``,
    ``top_p``, ``top_k``) are rejected, and thinking is configured as adaptive
    rather than with a fixed token budget.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ProviderUnavailable(
            "The 'anthropic' provider needs langchain-anthropic: "
            "pip install 'pr-review-implementor[anthropic]'"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "timeout": settings.request_timeout,
    }
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key

    if settings.thinking:
        try:
            return ChatAnthropic(thinking={"type": "adaptive"}, **kwargs)
        except (TypeError, ValueError) as exc:
            # Older langchain-anthropic builds do not expose `thinking`.
            log.warning("Adaptive thinking unavailable in this langchain-anthropic build: %s", exc)

    return ChatAnthropic(**kwargs)
