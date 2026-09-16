"""Request and response bodies for the web API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Everything the UI can configure for one run."""

    pr: str = Field(description="owner/repo#123 or a GitHub PR URL.")
    repo_path: str = Field(default=".", description="Local checkout to edit.")

    dry_run: bool = False
    commit: bool = True
    push: bool = False
    reply: bool = False
    resolve: bool = False

    checkout_branch: bool = Field(
        default=True,
        description="Check the PR's head branch out first, stashing uncommitted work.",
    )
    restore_branch: bool = Field(
        default=False,
        description="Afterwards, return to the original branch and unstash.",
    )

    comment_ids: list[int] = Field(default_factory=list)
    ignore_authors: list[str] = Field(default_factory=list)
    self_login: str | None = None
    include_review_bodies: bool = False

    model: str | None = None
    validate_commands: list[str] = Field(default_factory=list)
    max_fix_attempts: int | None = None


class RunCreated(BaseModel):
    run_id: str


class ChatRequest(BaseModel):
    """A question or follow-up instruction about a finished run."""

    message: str = Field(min_length=1, description="What to ask the agent.")
    allow_edits: bool = Field(
        default=True,
        description="Let the reply change files. Off makes the chat read-only.",
    )


class ChatMessage(BaseModel):
    """One turn of a run's chat, as replayed to the UI."""

    role: str
    """'user' or 'agent'."""

    text: str
    files_changed: list[str] = Field(default_factory=list)
    diff: str = ""
    error: str = ""
    ts: float = 0.0


class ChatHistory(BaseModel):
    run_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    available: bool = True
    """False while the run is still going, when the checkout is not free to edit."""

    branch: str | None = None
    """The branch the checkout is on now — which is what a further change edits."""

    detail: str = ""


class ThreadSummary(BaseModel):
    """One review thread, as shown in the preview list."""

    id: int
    author: str
    path: str | None = None
    line: int | None = None
    body: str
    replies: int = 0
    html_url: str = ""


class SearchHit(BaseModel):
    path: str
    start_line: int
    end_line: int
    score: float
    source: str
    content: str


class ConfigStatus(BaseModel):
    """What the server knows about its own configuration."""

    llm_key_set: bool
    github_token_set: bool
    provider: str
    model: str
    repo_path: str
    repo_is_git: bool
    repo_branch: str | None = None
    repo_dirty: bool = False
    repo_has_stash: bool = False
    detected_validate_commands: list[str] = Field(default_factory=list)
    version: str
