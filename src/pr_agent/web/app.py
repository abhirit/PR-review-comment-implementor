"""The FastAPI application.

This is a local development tool: it edits files on disk and runs the
validation commands you configure, so it binds to localhost and ships no
authentication. Do not expose it to a network you do not control.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, git_ops
from ..config import load_settings
from ..github_client import GitHubClient, GitHubError, build_threads
from ..models import PRRef
from ..rag.index import CodeIndex
from ..validation import detect_validation_commands
from ..workspace import Workspace, WorkspaceError
from .runner import RunError, RunManager
from .schemas import (
    ChatHistory,
    ChatMessage,
    ChatRequest,
    ConfigStatus,
    RunCreated,
    RunRequest,
    SearchHit,
    ThreadSummary,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="PR Review Comment Implementor",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    manager = RunManager()
    app.state.manager = manager

    # -- config -----------------------------------------------------------

    @app.get("/api/config", response_model=ConfigStatus)
    def get_config(repo_path: str = Query(default=".")) -> ConfigStatus:
        settings = load_settings(repo_path=Path(repo_path))
        resolved = Path(repo_path).expanduser()
        branch: str | None = None
        dirty = False
        has_stash = False
        is_git = git_ops.is_git_repo(resolved)
        if is_git:
            try:
                branch = git_ops.current_branch(resolved)
                dirty = git_ops.is_dirty(resolved)
                has_stash = git_ops.stash_count(resolved) > 0
            except git_ops.GitError:
                pass  # detached HEAD or an unreadable index

        detected: list[str] = []
        if resolved.is_dir():
            detected = detect_validation_commands(resolved)

        return ConfigStatus(
            llm_key_set=bool(settings.llm_api_key),
            github_token_set=bool(settings.github_token),
            voyage_key_set=bool(settings.voyage_api_key),
            provider=settings.provider,
            model=settings.model,
            embedding_backend=settings.embedding_backend,
            repo_path=str(resolved.resolve()) if resolved.exists() else str(resolved),
            repo_is_git=is_git,
            repo_branch=branch,
            repo_dirty=dirty,
            repo_has_stash=has_stash,
            detected_validate_commands=detected,
            version=__version__,
        )

    # -- review comments --------------------------------------------------

    @app.get("/api/comments", response_model=list[ThreadSummary])
    def get_comments(pr: str = Query(...)) -> list[ThreadSummary]:
        settings = load_settings()
        try:
            pr_ref = PRRef.parse(pr)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            with GitHubClient(settings.github_token, api_url=settings.github_api_url) as client:
                threads = build_threads(client.list_review_comments(pr_ref))
        except GitHubError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return [
            ThreadSummary(
                id=t.id,
                author=t.root.author,
                path=t.path,
                line=t.line,
                body=t.root.body,
                replies=len(t.replies),
                html_url=t.root.html_url,
            )
            for t in threads
        ]

    # -- retrieval --------------------------------------------------------

    @app.get("/api/search", response_model=list[SearchHit])
    def search(
        q: str = Query(..., min_length=1),
        repo_path: str = Query(default="."),
        k: int = Query(default=6, ge=1, le=30),
    ) -> list[SearchHit]:
        settings = load_settings(repo_path=Path(repo_path))
        try:
            workspace = Workspace(
                root=settings.repo_path, max_file_bytes=settings.max_index_file_bytes
            )
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        index = _index_for(app, settings, workspace)
        return [
            SearchHit(
                path=c.path,
                start_line=c.start_line,
                end_line=c.end_line,
                score=c.score,
                source=c.source,
                content=c.content,
            )
            for c in index.retriever.retrieve(q, k=k)
        ]

    # -- runs -------------------------------------------------------------

    @app.post("/api/runs", response_model=RunCreated, status_code=201)
    async def create_run(request: RunRequest) -> RunCreated:
        loop = asyncio.get_running_loop()
        try:
            run = manager.start(request, loop)
        except RunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RunCreated(run_id=run.id)

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return manager.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="No such run")
        return run.snapshot()

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        if manager.get(run_id) is None:
            raise HTTPException(status_code=404, detail="No such run")
        stopped = manager.cancel(run_id)
        return {
            "cancelled": stopped,
            # Being honest about the granularity matters: the graph cannot be
            # interrupted in the middle of a node.
            "detail": (
                "The run will stop after the current step finishes."
                if stopped
                else "The run has already finished."
            ),
        }

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="No such run")

        async def stream():
            queue = await manager.subscribe(run)
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"  # hold the connection open
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "end":
                        break
            finally:
                manager.unsubscribe(run, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- follow-up chat ---------------------------------------------------

    def _chattable_run(run_id: str):
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="No such run")
        return run

    @app.get("/api/runs/{run_id}/chat", response_model=ChatHistory)
    def get_chat(run_id: str) -> ChatHistory:
        run = _chattable_run(run_id)
        live = run.status in {"starting", "running"}
        session = manager.peek_chat(run_id)
        # A run that handed the checkout back is no longer on the PR branch, so
        # say which branch a further change would land on.
        branch = None
        try:
            checkout = Path(run.request.repo_path).expanduser()
            if git_ops.is_git_repo(checkout):
                branch = git_ops.current_branch(checkout)
        except git_ops.GitError:
            pass
        return ChatHistory(
            run_id=run_id,
            messages=[ChatMessage(**turn) for turn in (session.history() if session else [])],
            available=not live,
            branch=branch,
            detail=(
                "The run is still going. Chat opens when it finishes, so the two "
                "cannot edit the checkout at the same time."
                if live
                else ""
            ),
        )

    @app.post("/api/runs/{run_id}/chat", response_model=ChatMessage)
    def post_chat(run_id: str, request: ChatRequest) -> ChatMessage:
        run = _chattable_run(run_id)
        # One writer at a time: the run and the chat share a checkout.
        if run.status in {"starting", "running"}:
            raise HTTPException(
                status_code=409,
                detail="This run is still going. Wait for it to finish before chatting about it.",
            )
        try:
            session = manager.chat_session(run)
        except Exception as exc:  # noqa: BLE001 - config and model errors alike
            raise HTTPException(
                status_code=400, detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        turn = session.ask(request.message, allow_edits=request.allow_edits)
        return ChatMessage(**turn.as_dict())

    # -- static UI --------------------------------------------------------

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index_page() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _index_for(app: FastAPI, settings, workspace: Workspace) -> CodeIndex:
    """Reuse one index per repository path, refreshed on every request.

    The CodeIndex is kept so the vector store and its manifest survive between
    searches; refresh is still called each time because files change under the
    server, and re-chunking is cheap while re-embedding is incremental.
    """
    cache: dict[str, CodeIndex] = getattr(app.state, "index_cache", None) or {}
    key = f"{workspace.root}|{settings.embedding_backend}|{settings.chunk_size}"
    index = cache.get(key)
    if index is None:
        index = CodeIndex(settings, workspace)
        cache[key] = index
        app.state.index_cache = cache
    index.refresh()
    return index
