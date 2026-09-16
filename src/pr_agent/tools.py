"""File tools exposed to the model during the implement step.

The tools are built per run so they close over a single :class:`Workspace`,
which keeps every edit inside the repository and records what was touched.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .models import truncate
from .rag.retriever import CodeRetriever
from .workspace import Workspace, WorkspaceError

log = logging.getLogger(__name__)

_MAX_READ_CHARS = 60_000


class ReadFileArgs(BaseModel):
    path: str = Field(description="Repository-relative path, e.g. src/app/api.py")
    start_line: int = Field(default=1, description="First line to read (1-indexed).")
    end_line: int | None = Field(
        default=None, description="Last line to read, inclusive. Omit to read to the end."
    )


class EditFileArgs(BaseModel):
    path: str = Field(description="Repository-relative path of the file to edit.")
    old_text: str = Field(
        description=(
            "Exact text to replace, copied verbatim from the file including "
            "indentation. Must appear exactly once unless replace_all is true."
        )
    )
    new_text: str = Field(description="Replacement text. Use an empty string to delete.")
    replace_all: bool = Field(
        default=False, description="Replace every occurrence instead of requiring uniqueness."
    )


class WriteFileArgs(BaseModel):
    path: str = Field(description="Repository-relative path to write.")
    content: str = Field(description="Full file content. Overwrites any existing file.")


class SearchArgs(BaseModel):
    query: str = Field(description="Literal substring to search for across the repository.")


class SemanticSearchArgs(BaseModel):
    query: str = Field(
        description="Natural-language or identifier query for semantically similar code."
    )


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Repository-relative directory to list.")


def build_file_tools(
    workspace: Workspace,
    retriever: CodeRetriever | None = None,
    allow_edits: bool = True,
) -> list[StructuredTool]:
    """Build the tool set the implement node binds to the model.

    With ``allow_edits`` off the writing tools are left out entirely, so a
    read-only caller cannot change the checkout even if the model tries.
    """

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        try:
            if start_line == 1 and end_line is None:
                text = workspace.read(path)
                if len(text) > _MAX_READ_CHARS:
                    return (
                        f"{path} is {len(text)} characters, too large to read at once. "
                        "Read it in ranges with start_line/end_line.\n\n"
                        + workspace.read_lines(path, 1, 400)
                    )
                return workspace.read_lines(path, 1, None)
            return workspace.read_lines(path, start_line, end_line)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"

    def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        if old_text == new_text:
            return "ERROR: old_text and new_text are identical; nothing to do."
        try:
            return workspace.replace(path, old_text, new_text, replace_all=replace_all)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"

    def write_file(path: str, content: str) -> str:
        try:
            existed = workspace.exists(path)
            rel = workspace.write(path, content)
            verb = "Overwrote" if existed else "Created"
            return f"{verb} {rel} ({len(content)} characters)."
        except WorkspaceError as exc:
            return f"ERROR: {exc}"

    def search_repository(query: str) -> str:
        try:
            hits = workspace.grep(query)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        if not hits:
            return f"No matches for {query!r}."
        return "\n".join(f"{path}:{line}: {text}" for path, line, text in hits)

    def semantic_search(query: str) -> str:
        if retriever is None:
            return "ERROR: the semantic index is not available; use search_repository instead."
        chunks = retriever.retrieve(query, k=5)
        if not chunks:
            return f"No semantically similar code found for {query!r}."
        return "\n\n".join(chunk.render() for chunk in chunks)

    def list_directory(path: str = ".") -> str:
        try:
            target = workspace.resolve(path)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        if not target.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            rel = workspace.relative(child)
            if workspace.is_excluded(rel) or child.name == ".git":
                continue
            entries.append(f"{rel}/" if child.is_dir() else rel)
        return "\n".join(entries) if entries else f"{path} is empty."

    tools = [
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=(
                "Read a file from the repository with line numbers. Always read a file "
                "before editing it."
            ),
            args_schema=ReadFileArgs,
        ),
        StructuredTool.from_function(
            func=search_repository,
            name="search_repository",
            description="Literal substring search across the repository. Returns path:line: text.",
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=semantic_search,
            name="semantic_search",
            description=(
                "Search the repository index for code related to a description or "
                "identifier. Use when you do not know the exact string to grep for."
            ),
            args_schema=SemanticSearchArgs,
        ),
        StructuredTool.from_function(
            func=list_directory,
            name="list_directory",
            description="List the files and subdirectories of a directory in the repository.",
            args_schema=ListDirArgs,
        ),
    ]

    if allow_edits:
        tools[1:1] = [
            StructuredTool.from_function(
                func=edit_file,
                name="edit_file",
                description=(
                    "Replace an exact block of text in a file. The preferred way to make "
                    "a change. old_text must match the file byte for byte."
                ),
                args_schema=EditFileArgs,
            ),
            StructuredTool.from_function(
                func=write_file,
                name="write_file",
                description=(
                    "Write a file in full, creating it if needed. Use only for new files "
                    "or a complete rewrite; prefer edit_file otherwise."
                ),
                args_schema=WriteFileArgs,
            ),
        ]
    return tools


# ---------------------------------------------------------------------------
# Driving the model's tool calls
#
# Shared by the graph's implement and fix nodes and by the follow-up chat, so
# all three execute tool calls the same way.
# ---------------------------------------------------------------------------


def run_tool_loop(
    model,
    tools: list[StructuredTool],
    messages: list[BaseMessage],
    max_iterations: int,
) -> tuple[str, str]:
    """Drive the model's tool calls until it answers with text.

    ``messages`` is appended to in place, so a caller keeping a conversation
    across turns gets the tool calls and their results in its own history.
    Returns ``(summary, error)``; ``error`` is empty on success.
    """
    by_name = {tool.name: tool for tool in tools}

    for _ in range(max_iterations):
        try:
            response = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - surface, do not crash the run
            log.error("Model call failed: %s", exc)
            return "", f"model call failed: {exc}"

        assert isinstance(response, AIMessage)
        messages.append(response)

        tool_calls = response.tool_calls or []
        if not tool_calls:
            return text_of(response), ""

        for call in tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                result = f"ERROR: unknown tool {call['name']!r}"
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001 - report back to the model
                    result = f"ERROR: {exc}"
            log.debug("tool %s -> %s", call["name"], str(result)[:200])
            messages.append(
                ToolMessage(content=truncate(str(result), 30000), tool_call_id=call["id"])
            )

    return "", f"gave up after {max_iterations} tool iterations without a final answer"


def text_of(message: AIMessage) -> str:
    """Extract plain text from a response that may contain thinking blocks."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts).strip()
