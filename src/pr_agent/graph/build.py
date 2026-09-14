"""Wiring the nodes into a compiled LangGraph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..github_client import GitHubClient
from ..llm import build_chat_model
from ..models import PRRef
from ..rag.index import CodeIndex
from ..workspace import Workspace
from . import nodes
from .state import AgentState


@dataclass
class AgentDeps:
    """Live resources shared by every node, bound when the graph is built."""

    settings: Settings
    pr_ref: PRRef
    github: GitHubClient
    workspace: Workspace
    index: CodeIndex
    llm: Any

    # behaviour switches
    dry_run: bool = False
    commit: bool = True
    push: bool = False
    checkout_branch: bool = True
    """Move the checkout onto the PR's head branch before working."""
    restore_branch: bool = False
    """Put the checkout back on the original branch once the run is done."""
    push_branch: str | None = None
    write_replies: bool = False
    resolve_threads: bool = False
    include_review_bodies: bool = False
    only_comment_ids: list[int] = field(default_factory=list)
    ignore_authors: set[str] = field(default_factory=set)
    self_login: str | None = None


def build_deps(
    settings: Settings,
    pr_ref: PRRef,
    github: GitHubClient | None = None,
    **behaviour: Any,
) -> AgentDeps:
    """Construct the dependency bundle from settings."""
    workspace = Workspace(
        root=settings.repo_path, max_file_bytes=settings.max_index_file_bytes
    )
    return AgentDeps(
        settings=settings,
        pr_ref=pr_ref,
        github=github
        or GitHubClient(settings.github_token, api_url=settings.github_api_url),
        workspace=workspace,
        index=CodeIndex(settings, workspace),
        llm=build_chat_model(settings),
        **behaviour,
    )


def build_agent_graph(deps: AgentDeps, checkpointer: Any | None = None):
    """Compile the agent graph.

    The shape is a queue-driven loop: threads are processed one at a time so
    that edits never race, with an inner validate/fix cycle per thread.

        load_pr -> prepare_branch -> index_repo -> next_thread
                                                        |
                                    (queue empty) ------+--> finalize -> END
                                                        |
                                                     triage --(not actionable)--> record
                                                        |
                                                     retrieve -> plan -> implement -> validate
                                                                                         |
                                                             (fails, attempts left)      |
                                                                     +--> fix <----------+
                                                                     |                   |
                                                                     +--> validate       |
                                                                                         |
                                                                         record <--------+
                                                                            |
                                                                            +--> next_thread
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_pr", nodes.make_load_pr(deps))
    graph.add_node("prepare_branch", nodes.make_prepare_branch(deps))
    graph.add_node("index_repo", nodes.make_index_repo(deps))
    graph.add_node("next_thread", nodes.make_next_thread(deps))
    graph.add_node("triage", nodes.make_triage(deps))
    graph.add_node("retrieve", nodes.make_retrieve(deps))
    graph.add_node("plan", nodes.make_plan(deps))
    graph.add_node("implement", nodes.make_implement(deps))
    graph.add_node("validate", nodes.make_validate(deps))
    graph.add_node("fix", nodes.make_fix(deps))
    graph.add_node("record", nodes.make_record(deps))
    graph.add_node("finalize", nodes.make_finalize(deps))

    graph.add_edge(START, "load_pr")
    graph.add_edge("load_pr", "prepare_branch")
    # Indexing has to follow the checkout: the branch decides the file contents.
    graph.add_edge("prepare_branch", "index_repo")
    graph.add_edge("index_repo", "next_thread")

    graph.add_conditional_edges(
        "next_thread",
        nodes.route_after_next,
        {"triage": "triage", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "triage",
        nodes.route_after_triage,
        {"retrieve": "retrieve", "record": "record"},
    )
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "validate")
    graph.add_conditional_edges(
        "validate",
        nodes.make_route_after_validate(deps),
        {"fix": "fix", "record": "record"},
    )
    graph.add_edge("fix", "validate")
    graph.add_edge("record", "next_thread")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
