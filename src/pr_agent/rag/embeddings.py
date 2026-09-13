"""Embedding backends.

Anthropic does not serve an embeddings endpoint, so the embedding model is
chosen independently of the chat model. Two backends are supported plus an
explicit opt-out that leaves retrieval to BM25 alone.
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
        f"Unknown embedding backend '{backend}'. Use 'local', 'voyage' or 'none'."
    )
