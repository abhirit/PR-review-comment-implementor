"""Follow-up chat about a run.

A run ends with a report, a set of diffs and a pile of decisions the agent made
on its own. This is where you get to ask about any of it — "why did you skip
the second comment?", "show me what changed in retriever.py" — and to ask for
further changes, which are made with the same sandboxed file tools the run
itself used.

The session holds the conversation, so each turn sees the ones before it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .config import Settings
from .models import truncate
from .rag.index import CodeIndex
from .tools import build_file_tools, run_tool_loop
from .workspace import Workspace

log = logging.getLogger(__name__)

CHAT_SYSTEM = """\
You are the agent that just finished implementing the review comments on a \
pull request. The user is now asking you about that run, in the same \
repository you worked in.

Two kinds of request arrive here:

- A question about the run — what you changed, why you skipped a comment, what \
a diff does, whether something is covered by a test. Answer it directly from \
the run summary below and from the files themselves. Read the file rather than \
answering from the summary when the detail matters.
- A further change. Make it with the file tools, the same way you made the \
original ones: read_file before you edit, keep the edit minimal and in the \
style of the surrounding code, and do not touch anything the request does not \
call for.

Rules:
- Never claim to have changed a file you did not actually edit with a tool.
- If a request is ambiguous enough that guessing would waste the user's time, \
ask which of the readings they meant instead of picking one.
- If you cannot do something — the code is not there, the request conflicts \
with the codebase — say so plainly and explain what you found.
- Answer in prose, not a diff: the diff of your edits is shown alongside your \
reply already.
- Be brief. This is a terminal-width chat pane, not a report.
"""

_NO_EDIT_NOTE = """\

File edits are turned off for this message, so you have read-only tools. If the \
user is asking for a change, explain what you would change and where, and tell \
them to re-send with edits enabled.
"""


@dataclass
class ChatTurn:
    """One exchange: what was asked, what came back, and what it changed."""

    role: str
    text: str
    files_changed: list[str] = field(default_factory=list)
    diff: str = ""
    error: str = ""
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "files_changed": self.files_changed,
            "diff": self.diff,
            "error": self.error,
            "ts": self.ts,
        }


class ChatSession:
    """One conversation about one run.

    Cheap to construct: the retrieval index is only refreshed on the first
    message, so opening the chat tab costs nothing.
    """

    def __init__(
        self,
        settings: Settings,
        workspace: Workspace,
        index: CodeIndex,
        llm: Any,
        context: str,
    ) -> None:
        self.settings = settings
        self.workspace = workspace
        self.index = index
        self.llm = llm
        self.context = context
        self.turns: list[ChatTurn] = []
        self._messages: list[BaseMessage] = [
            SystemMessage(f"{CHAT_SYSTEM}\n\n---\n\n{context}")
        ]
        self._indexed = False
        # Two messages in flight would interleave their workspace transactions
        # and hand each other the wrong diff.
        self._lock = threading.Lock()

    # -- history ----------------------------------------------------------

    def history(self) -> list[dict[str, Any]]:
        return [turn.as_dict() for turn in self.turns]

    # -- the exchange -----------------------------------------------------

    def ask(self, message: str, allow_edits: bool = True) -> ChatTurn:
        """Answer one message, optionally letting the model edit files."""
        with self._lock:
            self.turns.append(ChatTurn(role="user", text=message))
            turn = self._answer(message, allow_edits)
            self.turns.append(turn)
            return turn

    def _answer(self, message: str, allow_edits: bool) -> ChatTurn:
        try:
            self._ensure_index()
        except Exception as exc:  # noqa: BLE001 - retrieval is a convenience
            log.warning("Could not refresh the index for chat: %s", exc)

        tools = build_file_tools(
            self.workspace, self.index.retriever, allow_edits=allow_edits
        )
        model = self.llm.bind_tools(tools)

        prompt = message if allow_edits else f"{message}\n{_NO_EDIT_NOTE}"
        self._messages.append(HumanMessage(prompt))

        # Snapshot so this turn's edits can be shown as a diff of their own.
        # The changed list comes from the snapshot rather than from
        # workspace.touched, which accumulates: a second edit to a file the
        # first message already changed would otherwise look like no edit.
        self.workspace.begin()
        try:
            reply, error = run_tool_loop(
                model, tools, self._messages, self.settings.max_tool_iterations
            )
        finally:
            diff = self.workspace.pending_diff()
            changed = self.workspace.commit_changes()

        if changed:
            # Keep retrieval honest for the next message in this conversation.
            self.index.reindex(changed)
            log.info("chat changed %d file(s)", len(changed))

        return ChatTurn(
            role="agent",
            text=reply,
            files_changed=changed,
            diff=truncate(diff, 40000),
            error=error,
        )

    def _ensure_index(self) -> None:
        if self._indexed:
            return
        self.index.refresh()
        self._indexed = True


def build_run_context(
    pr_slug: str,
    pr_title: str,
    branch_detail: str,
    report: str,
    outcomes: list[dict[str, Any]],
) -> str:
    """Render what the agent knows about a finished run, for the system prompt.

    Diffs are included but capped: the whole point is that the model can go and
    read the current file when it needs more than the summary.
    """
    lines = ["# The run you are being asked about", ""]
    lines.append(f"Pull request: {pr_slug or '(unknown)'}")
    if pr_title:
        lines.append(f"Title: {pr_title}")
    if branch_detail:
        lines.append(f"Branch: {branch_detail}")
    lines.append("")

    if report:
        lines.append("## Report")
        lines.append(truncate(report, 4000))
        lines.append("")

    if not outcomes:
        lines.append("No review threads were acted on.")
        return "\n".join(lines)

    lines.append("## What happened to each review comment")
    for outcome in outcomes:
        location = outcome.get("path") or "general comment"
        status = "implemented" if outcome.get("files_changed") else outcome.get("action", "skip")
        if outcome.get("error"):
            status = "failed"
        lines.append("")
        lines.append(f"### Thread {outcome.get('thread_id')} on {location} — {status}")
        if outcome.get("reason"):
            lines.append(f"Decision: {outcome['reason']}")
        if outcome.get("summary"):
            lines.append(f"What was done: {truncate(outcome['summary'], 1500)}")
        if outcome.get("files_changed"):
            lines.append(f"Files changed: {', '.join(outcome['files_changed'])}")
        validation = outcome.get("validation") or {}
        if validation.get("command"):
            verdict = "passed" if validation.get("ok") else "failed"
            lines.append(f"Validation: `{validation['command']}` {verdict}")
        if outcome.get("error"):
            lines.append(f"Error: {outcome['error']}")
        if outcome.get("diff"):
            lines.append("Diff:")
            lines.append("```diff")
            lines.append(truncate(outcome["diff"], 6000))
            lines.append("```")

    return truncate("\n".join(lines), 120000)
