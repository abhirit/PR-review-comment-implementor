"""Building and persisting the repository index."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..config import Settings
from ..workspace import Workspace, WorkspaceError
from .embeddings import EmbeddingsUnavailable, build_embeddings
from .retriever import HybridRetriever
from .splitter import chunk_file

log = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"


@dataclass
class IndexStats:
    """What the last refresh did, for logging and the CLI."""

    files: int = 0
    chunks: int = 0
    embedded_files: int = 0
    removed_files: int = 0
    vector_backend: str = "none"

    def describe(self) -> str:
        return (
            f"{self.chunks} chunks from {self.files} files "
            f"(vector backend: {self.vector_backend}, "
            f"{self.embedded_files} re-embedded, {self.removed_files} removed)"
        )


class CodeIndex:
    """A hybrid index over a repository checkout.

    Chunking is cheap and always redone in memory; embedding is not, so only
    files whose content hash changed are re-embedded, using a manifest kept
    next to the vector store.

    The agent rewrites files while it runs, so the chunks captured by a full
    scan go stale mid-run. :meth:`reindex` brings individual files back in sync
    without repeating the scan.
    """

    def __init__(self, settings: Settings, workspace: Workspace) -> None:
        self.settings = settings
        self.workspace = workspace
        self.index_dir = settings.resolved_index_dir()
        self._exclude_own_storage()
        self.documents: list[Document] = []
        self.vectorstore: VectorStore | None = None
        self.stats = IndexStats()
        self._retriever: HybridRetriever | None = None
        # Per-file chunks and content hashes, kept so a single file can be
        # re-chunked in place rather than rescanning the repository.
        self._per_file: dict[str, list[Document]] = {}
        self._hashes: dict[str, str] = {}
        self._embeddings: Embeddings | None = None
        self._embeddings_ready = False
        self._vector_known: dict[str, dict] | None = None
        self._vector_fingerprint: str | None = None
        self._vector_backend = "none"

    def _exclude_own_storage(self) -> None:
        """Keep the index out of its own corpus.

        The manifest is rewritten on every refresh, so indexing it would mean
        one file always looks changed and gets re-embedded forever.
        """
        try:
            rel = self.index_dir.relative_to(self.workspace.root).as_posix()
        except ValueError:
            return  # stored outside the repository, nothing to exclude
        patterns = (f"{rel}", f"{rel}/*", f"{rel}/**/*")
        missing = tuple(p for p in patterns if p not in self.workspace.excludes)
        if missing:
            self.workspace.excludes = self.workspace.excludes + missing

    # -- public API -------------------------------------------------------

    def refresh(self) -> IndexStats:
        """Re-chunk the whole repository and bring the vector store up to date."""
        self._per_file = {}
        self._hashes = {}
        for rel in self.workspace.iter_files():
            chunks, digest = self._chunk_one(rel)
            if chunks is None:
                continue
            self._per_file[rel] = chunks
            self._hashes[rel] = digest

        self._rebuild()
        log.info("Index ready: %s", self.stats.describe())
        return self.stats

    def reindex(self, paths: Iterable[str]) -> IndexStats:
        """Bring specific files back in sync after they were edited.

        Retrieval that returns superseded code is worse than returning nothing,
        because the prompt then contradicts the file the agent is about to edit.
        """
        if self._retriever is None:
            return self.refresh()

        targets = [p for p in dict.fromkeys(paths) if not self.workspace.is_excluded(p)]
        dirty = False
        for rel in targets:
            chunks, digest = self._chunk_one(rel)
            if chunks is None:
                # Deleted, emptied, or no longer indexable.
                if self._per_file.pop(rel, None) is not None:
                    self._hashes.pop(rel, None)
                    dirty = True
                continue
            if self._hashes.get(rel) == digest:
                continue
            self._per_file[rel] = chunks
            self._hashes[rel] = digest
            dirty = True

        if not dirty:
            return self.stats

        self._rebuild()
        log.info("Reindexed %d file(s): %s", len(targets), self.stats.describe())
        return self.stats

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self.refresh()
        assert self._retriever is not None
        return self._retriever

    # -- internals --------------------------------------------------------

    def _rebuild(self) -> None:
        """Rebuild the document list, vectors and retriever from ``_per_file``."""
        self.documents = [doc for chunks in self._per_file.values() for doc in chunks]
        self.stats = IndexStats(files=len(self._per_file), chunks=len(self.documents))

        embeddings = self._embeddings_or_none()
        if embeddings is not None:
            self._sync_vectors(embeddings)

        self._retriever = HybridRetriever(self.documents, self.vectorstore)

    def _chunk_one(self, rel: str) -> tuple[list[Document] | None, str]:
        """Chunk one file, or ``(None, "")`` if it is gone or not indexable."""
        try:
            path = self.workspace.resolve(rel)
            if not path.is_file():
                return None, ""
            if path.stat().st_size > self.settings.max_index_file_bytes:
                return None, ""
            content = self.workspace.read(rel)
        except (WorkspaceError, OSError) as exc:
            log.debug("skipping %s: %s", rel, exc)
            return None, ""

        chunks = chunk_file(
            rel,
            content,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            return None, ""
        return chunks, _hash(content)

    def _embeddings_or_none(self) -> Embeddings | None:
        """The embedding backend, built once and reused across every reindex."""
        if not self._embeddings_ready:
            self._embeddings_ready = True
            try:
                self._embeddings = build_embeddings(self.settings)
            except EmbeddingsUnavailable as exc:
                # Degrading to keyword-only retrieval is far better than refusing
                # to run, so this is a warning rather than an error.
                log.warning(
                    "Embeddings unavailable (%s). Falling back to BM25-only retrieval.",
                    exc,
                )
                self._embeddings = None
        return self._embeddings

    def _sync_vectors(self, embeddings: Embeddings) -> None:
        known = self._open_vector_state(embeddings)
        self.stats.vector_backend = self._vector_backend
        assert self.vectorstore is not None
        hashes = self._hashes

        stale_ids: list[str] = []
        removed = 0
        for rel, entry in list(known.items()):
            if hashes.get(rel) != entry.get("hash"):
                stale_ids.extend(entry.get("ids", []))
                known.pop(rel, None)
                removed += 1

        to_add: list[Document] = []
        for rel, chunks in self._per_file.items():
            if known.get(rel, {}).get("hash") == hashes[rel]:
                continue
            to_add.extend(chunks)
            known[rel] = {"hash": hashes[rel], "ids": [c.metadata["id"] for c in chunks]}

        if stale_ids:
            try:
                self.vectorstore.delete(ids=stale_ids)
            except (NotImplementedError, ValueError, TypeError) as exc:
                log.warning("Vector store could not delete stale chunks: %s", exc)

        if to_add:
            ids = [doc.metadata["id"] for doc in to_add]
            for start in range(0, len(to_add), 200):
                batch = to_add[start : start + 200]
                self.vectorstore.add_documents(batch, ids=ids[start : start + 200])

        self.stats.embedded_files = len({d.metadata["path"] for d in to_add})
        self.stats.removed_files = removed

        if self._vector_backend != "memory":
            self._save_manifest(
                {"fingerprint": self._vector_fingerprint, "files": known}
            )

    def _open_vector_state(self, embeddings: Embeddings) -> dict[str, dict]:
        """Open the store once per run, rebuilding it if the configuration changed."""
        fingerprint = self._fingerprint()
        if self._vector_known is not None and self._vector_fingerprint == fingerprint:
            return self._vector_known

        manifest = self._load_manifest()
        if manifest.get("fingerprint") != fingerprint:
            # A different embedding model or chunk size invalidates every vector.
            log.info("Index fingerprint changed; rebuilding the vector store from scratch.")
            manifest = {"fingerprint": fingerprint, "files": {}}
            self._reset_store_dir()

        store, backend = self._open_store(embeddings)
        self.vectorstore = store
        self._vector_backend = backend
        self._vector_fingerprint = fingerprint
        # An in-memory store keeps nothing between runs, so everything must be
        # embedded now; within a run this map carries forward what was added.
        self._vector_known = {} if backend == "memory" else manifest.get("files", {})
        return self._vector_known

    def _open_store(self, embeddings: Embeddings) -> tuple[VectorStore, str]:
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_core.vectorstores import InMemoryVectorStore

            log.warning(
                "langchain-chroma is not installed; using a non-persistent in-memory "
                "vector store. Install 'pr-review-implementor[vector]' to cache embeddings."
            )
            return InMemoryVectorStore(embeddings), "memory"

        self.index_dir.mkdir(parents=True, exist_ok=True)
        store = Chroma(
            collection_name="repo",
            embedding_function=embeddings,
            persist_directory=str(self.index_dir / "chroma"),
        )
        return store, "chroma"

    def _fingerprint(self) -> str:
        """Identifies the embedding configuration a persisted store was built with."""
        parts = [
            self.settings.embedding_backend,
            self.settings.embedding_model,
            self.settings.voyage_model,
            str(self.settings.chunk_size),
            str(self.settings.chunk_overlap),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _manifest_path(self) -> Path:
        return self.index_dir / _MANIFEST_NAME

    def _load_manifest(self) -> dict:
        path = self._manifest_path()
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Index manifest is unreadable; rebuilding.")
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(json.dumps(manifest), encoding="utf-8")

    def _reset_store_dir(self) -> None:
        import shutil

        chroma_dir = self.index_dir / "chroma"
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir, ignore_errors=True)
        manifest = self._manifest_path()
        if manifest.exists():
            manifest.unlink()


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
