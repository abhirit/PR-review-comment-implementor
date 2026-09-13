"""Embedding backends.

The embedding model is chosen independently of the chat model: Anthropic
serves no embeddings endpoint at all, and Gemini's is a separate model, so
neither provider decides this. Three backends are supported plus an explicit
opt-out that leaves retrieval to BM25 alone.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from ..config import Settings

log = logging.getLogger(__name__)


class EmbeddingsUnavailable(RuntimeError):
    """Raised when the requested backend's optional dependency is missing."""


def build_embeddings(settings: Settings) -> Embeddings | None:
    """Instantiate the configured embedding backend.

    Returns ``None`` when embeddings are disabled, in which case the index
    falls back to keyword-only retrieval.
    """
    backend = (settings.embedding_backend or "none").lower()

    if backend in {"none", "off", "bm25"}:
        log.info("Embeddings disabled; retrieval will use BM25 only.")
        return None

    if backend in {"google", "gemini"}:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise EmbeddingsUnavailable(
                "The 'google' embedding backend needs langchain-google-genai: "
                "pip install 'pr-review-implementor'"
            ) from exc
        if not settings.google_api_key:
            raise EmbeddingsUnavailable("GOOGLE_API_KEY is not set.")
        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    if backend == "voyage":
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except ImportError as exc:
            raise EmbeddingsUnavailable(
                "The 'voyage' embedding backend needs langchain-voyageai: "
                "pip install 'pr-review-implementor[voyage]'"
            ) from exc
        if not settings.voyage_api_key:
            raise EmbeddingsUnavailable("VOYAGE_API_KEY is not set.")
        return VoyageAIEmbeddings(
            model=settings.voyage_model,
            voyage_api_key=settings.voyage_api_key,
        )

    if backend in {"local", "huggingface", "hf"}:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise EmbeddingsUnavailable(
                "The 'local' embedding backend needs langchain-huggingface: "
                "pip install 'pr-review-implementor[local-embeddings]' "
                "(or set PR_AGENT_EMBEDDINGS=none for BM25-only retrieval)"
            ) from exc
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    raise EmbeddingsUnavailable(
        f"Unknown embedding backend '{backend}'. "
        "Use 'google', 'local', 'voyage' or 'none'."
    )
