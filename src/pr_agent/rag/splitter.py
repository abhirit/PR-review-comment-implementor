"""Language-aware chunking of source files."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Map file extensions onto the language-specific separator sets that
# langchain-text-splitters ships, so chunks break on function and class
# boundaries instead of mid-expression.
_EXT_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".cs": Language.CSHARP,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".hpp": Language.CPP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".rst": Language.RST,
    ".sol": Language.SOL,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".hs": Language.HASKELL,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
}


def language_for(rel_path: str) -> Language | None:
    return _EXT_TO_LANGUAGE.get(Path(rel_path).suffix.lower())


def splitter_for(rel_path: str, chunk_size: int, chunk_overlap: int):
    """Return a splitter tuned to the file's language, or a generic fallback."""
    language = language_for(rel_path)
    if language is None:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    try:
        return RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    except (ValueError, KeyError):  # pragma: no cover - unsupported language build
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )


def chunk_file(
    rel_path: str,
    content: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split one file into documents carrying path and line-range metadata."""
    if not content.strip():
        return []

    splitter = splitter_for(rel_path, chunk_size, chunk_overlap)
    pieces = splitter.split_text(content)

    documents: list[Document] = []
    cursor = 0
    for position, piece in enumerate(pieces):
        # Locate each chunk in the original text so we can attach line numbers,
        # which is what makes a retrieved chunk actionable for an edit.
        found = content.find(piece, cursor)
        if found == -1:
            found = content.find(piece)
        if found == -1:
            start_line = 1
        else:
            start_line = content.count("\n", 0, found) + 1
            cursor = found + len(piece)
        end_line = start_line + piece.count("\n")

        documents.append(
            Document(
                page_content=piece,
                metadata={
                    "path": rel_path,
                    "chunk": position,
                    "start_line": start_line,
                    "end_line": end_line,
                    "id": f"{rel_path}:{position}",
                },
            )
        )
    return documents
