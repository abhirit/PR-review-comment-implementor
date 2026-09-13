"""Incremental indexing: only changed files should be re-embedded."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pr_agent.config import Settings
from pr_agent.rag.index import CodeIndex
from pr_agent.workspace import Workspace


class FakeEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), float(text.count("e")), 1.0]


class RecordingStore:
    """A persistent-looking vector store that records what it is asked to do."""

    def __init__(self) -> None:
        self.docs: dict[str, Document] = {}
        self.added: list[list[str]] = []
        self.deleted: list[str] = []

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
        ids = ids or [d.metadata["id"] for d in documents]
        self.added.append(list(ids))
        for doc_id, doc in zip(ids, documents, strict=True):
            self.docs[doc_id] = doc

    def delete(self, ids: list[str]) -> None:
        self.deleted.extend(ids)
        for doc_id in ids:
            self.docs.pop(doc_id, None)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return list(self.docs.values())[:k]


@pytest.fixture
def indexed(repo, monkeypatch):
    """A CodeIndex wired to fake embeddings and a persistent recording store."""
    store = RecordingStore()
    settings = Settings(repo_path=repo, embedding_backend="fake", index_dir=repo / ".idx")
    monkeypatch.setattr(
        "pr_agent.rag.index.build_embeddings", lambda _s: FakeEmbeddings()
    )
    monkeypatch.setattr(
        CodeIndex, "_open_store", lambda self, _e: (store, "recording")
    )
    return CodeIndex(settings, Workspace(root=repo)), store, settings


def _embedded_paths(store: RecordingStore, batch: int = -1) -> set[str]:
    return {doc_id.rsplit(":", 1)[0] for doc_id in store.added[batch]}


def test_first_refresh_embeds_every_file(indexed):
    index, store, _ = indexed
    stats = index.refresh()
    assert stats.chunks > 0
    assert stats.vector_backend == "recording"
    assert "src/calc.py" in _embedded_paths(store)


def test_second_refresh_embeds_nothing_when_nothing_changed(indexed):
    index, store, _ = indexed
    index.refresh()
    calls_before = len(store.added)

    stats = index.refresh()

    assert len(store.added) == calls_before  # no new embedding batches
    assert stats.embedded_files == 0
    assert stats.chunks > 0  # but chunks are still available for BM25


def test_only_the_changed_file_is_re_embedded(indexed, repo):
    index, store, _ = indexed
    index.refresh()
    (repo / "src" / "calc.py").write_text("def divide(a, b):\n    return 0\n", encoding="utf-8")

    stats = index.refresh()

    assert stats.embedded_files == 1
    assert _embedded_paths(store) == {"src/calc.py"}
    # The superseded chunks are dropped so retrieval cannot return stale code.
    assert any(doc_id.startswith("src/calc.py") for doc_id in store.deleted)


def test_a_deleted_file_is_removed_from_the_store(indexed, repo):
    index, store, _ = indexed
    index.refresh()
    (repo / "README.md").unlink()

    stats = index.refresh()

    assert stats.removed_files == 1
    assert any(doc_id.startswith("README.md") for doc_id in store.deleted)


def test_changing_the_chunk_size_rebuilds_the_whole_index(indexed, repo):
    index, store, settings = indexed
    index.refresh()

    # A different chunk size invalidates every stored vector, so the manifest
    # fingerprint must force a full rebuild rather than a partial update.
    settings.chunk_size = 300
    stats = index.refresh()

    assert stats.embedded_files > 0
    assert "src/calc.py" in _embedded_paths(store)


def test_retrieval_fuses_vector_and_keyword_hits(indexed):
    index, _store, _ = indexed
    index.refresh()
    hits = index.retriever.retrieve("divide", k=5)
    assert hits
    assert any("vector" in hit.source for hit in hits)
    assert any("bm25" in hit.source for hit in hits)


def test_index_works_without_embeddings(repo):
    settings = Settings(repo_path=repo, embedding_backend="none", index_dir=repo / ".idx")
    index = CodeIndex(settings, Workspace(root=repo))

    stats = index.refresh()

    assert stats.vector_backend == "none"
    assert stats.chunks > 0
    assert index.retriever.retrieve("divide", k=3)


def test_missing_embedding_dependency_degrades_to_bm25(repo, monkeypatch):
    from pr_agent.rag.embeddings import EmbeddingsUnavailable

    def boom(_settings):
        raise EmbeddingsUnavailable("sentence-transformers is not installed")

    monkeypatch.setattr("pr_agent.rag.index.build_embeddings", boom)
    settings = Settings(repo_path=repo, embedding_backend="local", index_dir=repo / ".idx")
    index = CodeIndex(settings, Workspace(root=repo))

    stats = index.refresh()

    assert stats.vector_backend == "none"
    assert index.retriever.retrieve("divide", k=3)
