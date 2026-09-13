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

    comment_ids: list[int] = Field(default_factory=list)
    ignore_authors: list[str] = Field(default_factory=list)
    self_login: str | None = None
    include_review_bodies: bool = False

    model: str | None = None
    embeddings: str | None = None
    validate_commands: list[str] = Field(default_factory=list)
    max_fix_attempts: int | None = None


class RunCreated(BaseModel):
    run_id: str


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

    anthropic_key_set: bool
    github_token_set: bool
    voyage_key_set: bool
    model: str
    embedding_backend: str
    repo_path: str
    repo_is_git: bool
    repo_branch: str | None = None
    repo_dirty: bool = False
    detected_validate_commands: list[str] = Field(default_factory=list)
    version: str
