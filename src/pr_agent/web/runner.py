"""Background run management and live event streaming.

The graph is synchronous, so each run executes on a worker thread and pushes
events onto an asyncio queue per subscriber. Every event is also retained so a
browser that connects late, or reconnects, replays the whole run.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..chat import ChatSession, build_run_context
from ..config import load_settings
from ..github_client import GitHubClient
from ..graph.build import build_agent_graph, build_deps
from ..graph.state import initial_state
from ..llm import build_chat_model
from ..models import PRRef
from ..rag.index import CodeIndex
from ..workspace import Workspace
from .schemas import RunRequest

log = logging.getLogger(__name__)

# Nodes the UI shows as pipeline stages, in order.
STAGE_LABELS: dict[str, str] = {
    "load_pr": "Loading pull request",
    "prepare_branch": "Checking out the branch",
    "index_repo": "Indexing repository",
    "next_thread": "Selecting comment",
    "triage": "Triaging",
    "retrieve": "Retrieving context",
    "plan": "Planning",
    "implement": "Implementing",
    "validate": "Validating",
    "fix": "Fixing",
    "record": "Recording",
    "finalize": "Finalising",
}


class RunError(RuntimeError):
    """A run could not be started."""


@dataclass
class Run:
    """One agent run and everything the UI needs to render it."""

    id: str
    request: RunRequest
    status: str = "starting"  # starting | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    report: str = ""
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    threads: list[dict[str, Any]] = field(default_factory=list)
    pr: dict[str, Any] | None = None
    branch: dict[str, Any] | None = None
    _subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "report": self.report,
            "outcomes": self.outcomes,
            "threads": self.threads,
            "pr": self.pr,
            "branch": self.branch,
            "request": self.request.model_dump(),
            "events": self.events,
        }


class _QueueLogHandler(logging.Handler):
    """Forwards the agent's own log records into a run's event stream."""

    def __init__(self, emit_event) -> None:
        super().__init__(level=logging.INFO)
        self._emit = emit_event

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("pr_agent"):
            return
        try:
            self._emit({"type": "log", "level": record.levelname, "message": record.getMessage()})
        except Exception:  # noqa: BLE001 - logging must never break a run
            pass


class RunManager:
    """Owns every run in this server process."""

    def __init__(self, max_history: int = 25) -> None:
        self._runs: dict[str, Run] = {}
        self._order: list[str] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._active_repos: dict[str, str] = {}  # resolved repo path -> run id
        self._chats: dict[str, ChatSession] = {}  # run id -> follow-up conversation

    # -- accessors --------------------------------------------------------

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))
        out = []
        for run_id in ids:
            run = self._runs.get(run_id)
            if run is None:
                continue
            out.append(
                {
                    "id": run.id,
                    "status": run.status,
                    "pr": run.request.pr,
                    "created_at": run.created_at,
                    "finished_at": run.finished_at,
                    "implemented": sum(
                        1 for o in run.outcomes if o.get("files_changed")
                    ),
                }
            )
        return out

    # -- lifecycle --------------------------------------------------------

    def start(self, request: RunRequest, loop: asyncio.AbstractEventLoop) -> Run:
        """Validate the request, then run the graph on a worker thread."""
        try:
            pr_ref = PRRef.parse(request.pr)
        except ValueError as exc:
            raise RunError(str(exc)) from exc

        repo_key = str(Path(request.repo_path).expanduser().resolve())
        with self._lock:
            active = self._active_repos.get(repo_key)
            if active and self._runs.get(active) and self._runs[active].status in {
                "starting",
                "running",
            }:
                raise RunError(
                    f"A run is already in progress for {repo_key}. "
                    "Two runs editing one checkout would corrupt each other."
                )

            run = Run(id=uuid.uuid4().hex[:12], request=request)
            self._runs[run.id] = run
            self._order.append(run.id)
            self._active_repos[repo_key] = run.id
            self._prune_locked()

        thread = threading.Thread(
            target=self._execute,
            args=(run, pr_ref, loop, repo_key),
            name=f"pr-agent-run-{run.id}",
            daemon=True,
        )
        thread.start()
        return run

    def cancel(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.status not in {"starting", "running"}:
            return False
        run._cancel.set()
        return True

    def _prune_locked(self) -> None:
        while len(self._order) > self._max_history:
            oldest = self._order[0]
            candidate = self._runs.get(oldest)
            if candidate and candidate.status in {"starting", "running"}:
                break  # never evict a live run
            self._order.pop(0)
            self._runs.pop(oldest, None)
            self._chats.pop(oldest, None)

    # -- follow-up chat ---------------------------------------------------

    def peek_chat(self, run_id: str) -> ChatSession | None:
        """The existing conversation for a run, without building one."""
        return self._chats.get(run_id)

    def chat_session(self, run: Run) -> ChatSession:
        """The conversation about ``run``, built on first use.

        One session per run, so a question three messages in still knows what
        the first two established.
        """
        existing = self._chats.get(run.id)
        if existing is not None:
            return existing

        settings = load_settings(
            repo_path=Path(run.request.repo_path),
            model=run.request.model,
            embedding_backend=run.request.embeddings,
        )
        workspace = Workspace(
            root=settings.repo_path, max_file_bytes=settings.max_index_file_bytes
        )
        session = ChatSession(
            settings=settings,
            workspace=workspace,
            index=CodeIndex(settings, workspace),
            llm=build_chat_model(settings),
            context=_chat_context(run),
        )
        with self._lock:
            # Another request may have built one while this was constructing;
            # the first to land wins so both sides share a conversation.
            return self._chats.setdefault(run.id, session)

    # -- events -----------------------------------------------------------

    async def subscribe(self, run: Run) -> asyncio.Queue:
        """Get a queue pre-filled with the run's history, then live events."""
        queue: asyncio.Queue = asyncio.Queue()
        for event in list(run.events):
            queue.put_nowait(event)
        if run.status in {"done", "failed", "cancelled"}:
            queue.put_nowait({"type": "end", "status": run.status})
        else:
            run._subscribers.add(queue)
        return queue

    def unsubscribe(self, run: Run, queue: asyncio.Queue) -> None:
        run._subscribers.discard(queue)

    def _make_emitter(self, run: Run, loop: asyncio.AbstractEventLoop):
        """Build a thread-safe event emitter for this run."""

        def emit(event: dict[str, Any]) -> None:
            event.setdefault("ts", time.time())
            event.setdefault("seq", len(run.events))
            run.events.append(event)
            for queue in list(run._subscribers):
                # The worker thread cannot touch the loop's queues directly.
                loop.call_soon_threadsafe(queue.put_nowait, event)

        return emit

    # -- the run itself ---------------------------------------------------

    def _execute(
        self,
        run: Run,
        pr_ref: PRRef,
        loop: asyncio.AbstractEventLoop,
        repo_key: str,
    ) -> None:
        emit = self._make_emitter(run, loop)
        handler = _QueueLogHandler(emit)
        agent_log = logging.getLogger("pr_agent")
        agent_log.addHandler(handler)

        deps = None
        try:
            run.status = "running"
            emit({"type": "status", "status": "running"})

            request = run.request
            settings = load_settings(
                repo_path=Path(request.repo_path),
                model=request.model,
                embedding_backend=request.embeddings,
                max_fix_attempts=request.max_fix_attempts,
            )
            if request.validate_commands:
                settings.validate_commands = request.validate_commands

            deps = build_deps(
                settings,
                pr_ref,
                dry_run=request.dry_run,
                commit=request.commit,
                push=request.push,
                checkout_branch=request.checkout_branch,
                restore_branch=request.restore_branch,
                write_replies=request.reply or request.resolve,
                resolve_threads=request.resolve,
                include_review_bodies=request.include_review_bodies,
                only_comment_ids=list(request.comment_ids),
                ignore_authors={a.lower() for a in request.ignore_authors},
                self_login=request.self_login,
            )

            graph = build_agent_graph(deps)
            config = {
                "configurable": {"thread_id": run.id},
                "recursion_limit": 500,
            }

            final: dict[str, Any] = {}
            for chunk in graph.stream(initial_state(), config=config, stream_mode="updates"):
                for node, update in chunk.items():
                    final.update(update or {})
                    self._emit_node(emit, run, node, update or {})
                if run._cancel.is_set():
                    # The graph cannot be interrupted mid-node, so stop at the
                    # next boundary and say so plainly.
                    run.status = "cancelled"
                    emit({"type": "status", "status": "cancelled"})
                    break

            if run.status != "cancelled":
                run.status = "done"
                run.report = str(final.get("report", ""))
                emit({"type": "report", "report": run.report})
                emit({"type": "status", "status": "done"})

        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            log.exception("Run %s failed", run.id)
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            emit({"type": "error", "message": run.error})
            emit({"type": "status", "status": "failed"})
        finally:
            agent_log.removeHandler(handler)
            if deps is not None:
                try:
                    deps.github.close()
                except Exception:  # noqa: BLE001 - best effort
                    pass
            run.finished_at = time.time()
            emit({"type": "end", "status": run.status})
            with self._lock:
                if self._active_repos.get(repo_key) == run.id:
                    self._active_repos.pop(repo_key, None)

    def _emit_node(self, emit, run: Run, node: str, update: dict[str, Any]) -> None:
        """Translate one graph state update into UI events."""
        emit(
            {
                "type": "node",
                "node": node,
                "label": STAGE_LABELS.get(node, node.replace("_", " ").title()),
            }
        )

        if node == "load_pr":
            pr = update.get("pull_request")
            threads = update.get("threads") or []
            run.threads = [_thread_summary(t) for t in threads]
            run.pr = pr.model_dump(mode="json") if pr is not None else None
            emit({"type": "pr", "pr": run.pr, "threads": run.threads})

        elif node == "prepare_branch":
            switch = update.get("branch_switch")
            if switch is not None:
                run.branch = switch.model_dump(mode="json")
                emit({"type": "branch", "branch": run.branch})

        elif node == "index_repo":
            emit({"type": "index", "summary": update.get("index_summary", "")})

        elif node == "next_thread":
            current = update.get("current")
            if current is not None and current < len(run.threads):
                emit({"type": "thread_start", "thread": run.threads[current]})

        elif node == "triage":
            triage = update.get("triage")
            if triage is not None:
                emit(
                    {
                        "type": "triage",
                        "action": triage.action.value,
                        "reason": triage.reason,
                        "queries": triage.search_queries,
                    }
                )

        elif node == "plan":
            emit({"type": "plan", "plan": update.get("plan", "")})

        elif node in {"implement", "fix"}:
            emit(
                {
                    "type": "implement",
                    "phase": node,
                    "summary": update.get("implementation_summary", ""),
                    "files": update.get("files_changed", []),
                    "error": update.get("last_error", ""),
                }
            )

        elif node == "validate":
            emit(
                {
                    "type": "validation",
                    "ok": update.get("validation_ok", True),
                    "command": update.get("validation_command", ""),
                    "output": update.get("validation_output", ""),
                }
            )

        elif node == "record":
            outcomes = update.get("outcomes") or []
            if outcomes:
                latest = outcomes[-1].model_dump(mode="json")
                run.outcomes = [o.model_dump(mode="json") for o in outcomes]
                emit({"type": "outcome", "outcome": latest})

        elif node == "finalize":
            switch = update.get("branch_switch")
            if switch is not None:
                run.branch = switch.model_dump(mode="json")
                emit({"type": "branch", "branch": run.branch})
            emit(
                {
                    "type": "finalize",
                    "commit_sha": update.get("commit_sha"),
                    "pushed": bool(update.get("pushed")),
                    "replies_posted": update.get("replies_posted", 0),
                }
            )


def _chat_context(run: Run) -> str:
    """Everything the chat needs to know about a run, as prompt text."""
    pr = run.pr or {}
    branch = run.branch or {}
    branch_detail = " ".join(
        part for part in (branch.get("detail"), branch.get("restore_detail")) if part
    )
    return build_run_context(
        pr_slug=run.request.pr,
        pr_title=pr.get("title", ""),
        branch_detail=branch_detail,
        report=run.report or run.error,
        outcomes=run.outcomes,
    )


def _thread_summary(thread) -> dict[str, Any]:
    return {
        "id": thread.id,
        "author": thread.root.author,
        "path": thread.path,
        "line": thread.line,
        "body": thread.root.body,
        "replies": len(thread.replies),
        "html_url": thread.root.html_url,
    }


def make_github_client(settings) -> GitHubClient:
    return GitHubClient(settings.github_token, api_url=settings.github_api_url)
