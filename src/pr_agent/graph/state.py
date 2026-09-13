"""Graph state.

Only plain, serializable data lives in the state so a run can be checkpointed
and resumed. Live resources (the GitHub client, the index, the chat model) are
held in :class:`~pr_agent.graph.build.AgentDeps` and closed over by the nodes.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from ..models import PullRequest, ReviewThread, ThreadOutcome, Triage


def _replace(_current: Any, new: Any) -> Any:
    """Last write wins — the default, spelled out for clarity."""
    return new


class AgentState(TypedDict, total=False):
    """State threaded through every node."""

    # -- populated once, at load ------------------------------------------
    pull_request: PullRequest | None
    threads: list[ReviewThread]
    index_summary: str

    # -- the work queue ---------------------------------------------------
    pending: list[int]
    """Indices into ``threads`` that are still to be processed."""

    current: int | None
    """Index of the thread being worked on, or ``None`` between items."""

    # -- per-thread scratch space -----------------------------------------
    triage: Triage | None
    retrieved: str
    plan: str
    implementation_summary: str
    files_changed: list[str]
    attempts: int
    validation_ok: bool
    validation_command: str
    validation_output: str
    last_error: str

    # -- results ----------------------------------------------------------
    outcomes: Annotated[list[ThreadOutcome], _replace]
    commit_sha: str | None
    pushed: bool
    replies_posted: int
    report: str


def initial_state() -> AgentState:
    return AgentState(
        pull_request=None,
        threads=[],
        index_summary="",
        pending=[],
        current=None,
        triage=None,
        retrieved="",
        plan="",
        implementation_summary="",
        files_changed=[],
        attempts=0,
        validation_ok=True,
        validation_command="",
        validation_output="",
        last_error="",
        outcomes=[],
        commit_sha=None,
        pushed=False,
        replies_posted=0,
        report="",
    )
