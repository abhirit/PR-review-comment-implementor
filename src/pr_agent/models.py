"""Domain models shared across the GitHub client, the graph and the CLI."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PRRef(BaseModel):
    """Identifies a pull request."""

    owner: str
    repo: str
    number: int

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}#{self.number}"

    @classmethod
    def parse(cls, value: str) -> PRRef:
        """Parse ``owner/repo#123`` or a full GitHub PR URL."""
        raw = value.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            parts = [p for p in raw.split("/") if p]
            # https://github.com/<owner>/<repo>/pull/<number>
            try:
                idx = parts.index("pull")
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Not a pull request URL: {value}") from exc
            if idx < 2:
                raise ValueError(f"Not a pull request URL: {value}")
            number = parts[idx + 1].split("?")[0].split("#")[0]
            return cls(owner=parts[idx - 2], repo=parts[idx - 1], number=int(number))

        if "#" not in raw or "/" not in raw:
            raise ValueError(f"Expected 'owner/repo#number' or a PR URL, got: {value}")
        repo_part, number_part = raw.rsplit("#", 1)
        owner, repo = repo_part.split("/", 1)
        return cls(owner=owner, repo=repo, number=int(number_part))


class PullRequest(BaseModel):
    """The subset of PR metadata the agent needs."""

    ref: PRRef
    title: str
    body: str = ""
    head_ref: str
    head_sha: str
    base_ref: str
    state: str = "open"
    html_url: str = ""


class ReviewComment(BaseModel):
    """A single inline review comment from GitHub."""

    id: int
    body: str
    author: str
    path: str | None = None
    line: int | None = None
    start_line: int | None = None
    side: str | None = None
    diff_hunk: str = ""
    in_reply_to_id: int | None = None
    html_url: str = ""
    created_at: str = ""


class ReviewThread(BaseModel):
    """A root review comment plus its replies, treated as one unit of work."""

    root: ReviewComment
    replies: list[ReviewComment] = Field(default_factory=list)

    @property
    def id(self) -> int:
        return self.root.id

    @property
    def path(self) -> str | None:
        return self.root.path

    @property
    def line(self) -> int | None:
        return self.root.line or self.root.start_line

    def transcript(self) -> str:
        """The whole conversation, oldest first, as plain text."""
        lines = [f"@{self.root.author}: {self.root.body.strip()}"]
        for reply in self.replies:
            lines.append(f"@{reply.author}: {reply.body.strip()}")
        return "\n\n".join(lines)


class CommentAction(str, Enum):
    """What the agent decided to do with a thread."""

    IMPLEMENT = "implement"
    ANSWER_ONLY = "answer_only"
    SKIP = "skip"


class Triage(BaseModel):
    """Structured triage decision for one review thread."""

    action: CommentAction = Field(
        description=(
            "implement: the comment asks for a concrete code change. "
            "answer_only: it asks a question or requests clarification but no code change. "
            "skip: praise, acknowledgement, already-resolved, or not actionable."
        )
    )
    reason: str = Field(description="One or two sentences justifying the action.")
    target_files: list[str] = Field(
        default_factory=list,
        description="Repo-relative paths likely to need editing, best guess, may be empty.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="2-4 natural-language or identifier queries for retrieving relevant code.",
    )


class ChangePlan(BaseModel):
    """The plan the model commits to before touching any file."""

    summary: str = Field(description="One-sentence summary of the change to make.")
    steps: list[str] = Field(description="Ordered, concrete edit steps.")
    files_to_edit: list[str] = Field(
        default_factory=list, description="Repo-relative paths the plan will modify."
    )
    risks: str = Field(default="", description="Anything that could break, or empty.")


class ValidationResult(BaseModel):
    """Outcome of running the configured validation commands."""

    ok: bool
    command: str = ""
    exit_code: int = 0
    output: str = ""

    @property
    def short_output(self) -> str:
        return truncate(self.output, 6000)


class ThreadOutcome(BaseModel):
    """Everything the agent concluded about one thread, for the final report."""

    thread_id: int
    path: str | None = None
    action: CommentAction = CommentAction.SKIP
    reason: str = ""
    summary: str = ""
    files_changed: list[str] = Field(default_factory=list)
    diff: str = ""
    reply: str = ""
    attempts: int = 0
    validation: ValidationResult | None = None
    error: str = ""

    @property
    def implemented(self) -> bool:
        return self.action is CommentAction.IMPLEMENT and bool(self.files_changed)


def truncate(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters, keeping the head and the tail."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n\n...[{len(text) - limit} characters truncated]...\n\n{text[-tail:]}"
