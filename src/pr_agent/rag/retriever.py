"""Code-aware BM25 retrieval.

Keyword search matters a lot for code — a reviewer writing "rename
`parse_lines`" wants the chunk containing that exact identifier, and the
tokenizer below makes sure the query matches however the reviewer spelled it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

# Rank smoothing constant. Scores are reported as 1/(RANK_BIAS + rank) rather
# than raw BM25 magnitudes, so retrieve_many can compare hits from queries of
# different lengths without the longest query dominating.
_RANK_BIAS = 60

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def tokenize_code(text: str) -> list[str]:
    """Tokenize identifiers the way a developer searches for them.

    ``parseHTTPResponse`` yields ``parsehttpresponse``, ``parse``, ``http``,
    ``response`` so that a query in any of those forms still matches.
    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        tokens.append(lowered)
        parts = [p for chunk in word.split("_") for p in _CAMEL_RE.findall(chunk)]
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts if len(part) > 1)
    return tokens


@dataclass
class RetrievedChunk:
    """One retrieved chunk with provenance, ready to paste into a prompt."""

    path: str
    start_line: int
    end_line: int
    content: str
    score: float = 0.0
    source: str = ""

    @classmethod
    def from_document(cls, doc: Document, score: float = 0.0, source: str = "") -> RetrievedChunk:
        meta = doc.metadata or {}
        return cls(
            path=str(meta.get("path", "<unknown>")),
            start_line=int(meta.get("start_line", 1)),
            end_line=int(meta.get("end_line", 1)),
            content=doc.page_content,
            score=score,
            source=source,
        )

    def render(self) -> str:
        return (
            f"--- {self.path} (lines {self.start_line}-{self.end_line}) ---\n{self.content}"
        )


class BM25Index:
    """A compact Okapi BM25 index over chunk documents."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        doc_freq: Counter[str] = Counter()

        for doc in documents:
            text = f"{(doc.metadata or {}).get('path', '')}\n{doc.page_content}"
            tokens = tokenize_code(text)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._lengths.append(len(tokens))
            doc_freq.update(counts.keys())

        total = len(documents)
        self._avg_len = (sum(self._lengths) / total) if total else 0.0
        # BM25+ style idf floor keeps very common tokens from going negative.
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self.documents:
            return []
        terms = tokenize_code(query)
        if not terms:
            return []

        scores: list[tuple[int, float]] = []
        for idx, counts in enumerate(self._tf):
            length = self._lengths[idx] or 1
            score = 0.0
            for term in terms:
                freq = counts.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1)
                )
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scores[:k]]


class CodeRetriever:
    """Ranks chunks for a query with BM25 over code-aware tokens."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.bm25 = BM25Index(documents)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        """Return the top ``k`` chunks for a query."""
        if not query.strip():
            return []

        return [
            RetrievedChunk.from_document(doc, 1.0 / (_RANK_BIAS + rank + 1), "bm25")
            for rank, (doc, _) in enumerate(self.bm25.search(query, k))
        ]

    def retrieve_many(self, queries: list[str], k: int = 8) -> list[RetrievedChunk]:
        """Retrieve for several queries and merge, preserving the best rank."""
        merged: dict[str, RetrievedChunk] = {}
        for query in queries:
            for chunk in self.retrieve(query, k=k):
                key = f"{chunk.path}:{chunk.start_line}"
                existing = merged.get(key)
                if existing is None or chunk.score > existing.score:
                    merged[key] = chunk
        ordered = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        return ordered[:k]


def render_chunks(chunks: list[RetrievedChunk], max_chars: int = 20000) -> str:
    """Render retrieved chunks into a prompt block, within a character budget."""
    if not chunks:
        return "(no relevant code found in the index)"
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        rendered = chunk.render()
        if used + len(rendered) > max_chars:
            break
        parts.append(rendered)
        used += len(rendered)
    return "\n\n".join(parts) if parts else "(retrieved context exceeded the size budget)"
