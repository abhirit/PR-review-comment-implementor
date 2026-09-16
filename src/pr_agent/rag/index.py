"""Building the repository index."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document

from ..config import Settings
from ..workspace import Workspace, WorkspaceError
from .retriever import CodeRetriever
from .splitter import chunk_file

log = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """What the last refresh did, for logging and the CLI."""

    files: int = 0
    chunks: int = 0

    def describe(self) -> str:
        return f"{self.chunks} chunks from {self.files} files"


class CodeIndex:
    """A BM25 index over a repository checkout.

    Chunking is cheap, so the index lives entirely in memory and is rebuilt
    from the working tree rather than persisted between runs.

    The agent rewrites files while it runs, so the chunks captured by a full
    scan go stale mid-run. :meth:`reindex` brings individual files back in sync
    without repeating the scan.
    """

    def __init__(self, settings: Settings, workspace: Workspace) -> None:
        self.settings = settings
        self.workspace = workspace
        self.documents: list[Document] = []
        self.stats = IndexStats()
        self._retriever: CodeRetriever | None = None
        # Per-file chunks and content hashes, kept so a single file can be
        # re-chunked in place rather than rescanning the repository.
        self._per_file: dict[str, list[Document]] = {}
        self._hashes: dict[str, str] = {}

    # -- public API -------------------------------------------------------

    def refresh(self) -> IndexStats:
        """Re-chunk the whole repository and rebuild the retriever."""
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
    def retriever(self) -> CodeRetriever:
        if self._retriever is None:
            self.refresh()
        assert self._retriever is not None
        return self._retriever

    # -- internals --------------------------------------------------------

    def _rebuild(self) -> None:
        """Rebuild the document list and retriever from ``_per_file``."""
        self.documents = [doc for chunks in self._per_file.values() for doc in chunks]
        self.stats = IndexStats(files=len(self._per_file), chunks=len(self.documents))
        self._retriever = CodeRetriever(self.documents)

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


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
