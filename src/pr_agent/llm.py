"""Chat model construction."""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic

from .config import Settings

log = logging.getLogger(__name__)


def build_chat_model(settings: Settings) -> ChatAnthropic:
    """Build the Claude chat model used by every node in the graph.

    Note two current-model constraints: sampling parameters (``temperature``,
    ``top_p``, ``top_k``) are rejected, and thinking is configured as adaptive
    rather than with a fixed token budget.
    """
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
