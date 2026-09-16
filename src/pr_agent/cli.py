"""Command line interface."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from . import git_ops
from .config import load_settings
from .github_client import GitHubClient, GitHubError
from .graph import build_agent_graph, build_deps, run_config
from .graph.state import initial_state
from .models import PRRef
from .rag.index import CodeIndex
from .tracing import configure_tracing, flush_traces, traced_run
from .validation import detect_validation_commands
from .workspace import Workspace

app = typer.Typer(
    add_completion=False,
    help="Read GitHub pull request review comments and implement the requested changes.",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
    )
    # These are chatty at DEBUG and drown out the agent's own logs.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def run(
    pr: str = typer.Argument(..., help="Pull request as owner/repo#123 or a GitHub URL."),
    repo_path: Path = typer.Option(
        Path("."), "--repo", "-r", help="Local checkout of the repository to edit."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan and edit locally, then roll nothing back but commit nothing."
    ),
    commit: bool = typer.Option(True, "--commit/--no-commit", help="Commit the changes."),
    push: bool = typer.Option(False, "--push", help="Push the commit to the PR's head branch."),
    checkout: bool = typer.Option(
        True,
        "--checkout/--no-checkout",
        help="Check the PR's head branch out first, stashing any uncommitted work.",
    ),
    restore_branch: bool = typer.Option(
        False,
        "--restore-branch",
        help="When the run is done, return to the original branch and unstash.",
    ),
    reply: bool = typer.Option(
        False, "--reply", help="Post a reply on each review thread that was handled."
    ),
    resolve: bool = typer.Option(
        False, "--resolve", help="Resolve review threads that were implemented (implies --reply)."
    ),
    comment_id: list[int] = typer.Option(
        None, "--comment-id", help="Only handle these review comment ids. Repeatable."
    ),
    include_review_bodies: bool = typer.Option(
        False,
        "--include-review-bodies",
        help="Also act on review summary text, not just inline comments.",
    ),
    ignore_author: list[str] = typer.Option(
        None, "--ignore-author", help="Skip comments by this login. Repeatable."
    ),
    self_login: str = typer.Option(
        None,
        "--self-login",
        help="The agent's own GitHub login, so threads it already answered are skipped.",
    ),
    provider: str = typer.Option(
        None, "--provider", help="Chat provider: google (Gemini) or anthropic (Claude)."
    ),
    model: str = typer.Option(None, "--model", help="Override the model id."),
    validate: list[str] = typer.Option(
        None, "--validate", help="Validation command to run after each change. Repeatable."
    ),
    auto_validate: bool = typer.Option(
        False, "--auto-validate", help="Detect validation commands from the repository."
    ),
    max_fix_attempts: int = typer.Option(
        None, "--max-fix-attempts", help="How many times to retry after a failed check."
    ),
    trace: bool = typer.Option(
        None,
        "--trace/--no-trace",
        help="Send the run to LangSmith. On by default when LANGSMITH_API_KEY is set.",
    ),
    trace_project: str = typer.Option(
        None, "--trace-project", help="LangSmith project the run is filed under."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Implement the review comments on a pull request."""
    _setup_logging(verbose)

    try:
        pr_ref = PRRef.parse(pr)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    settings = load_settings(
        repo_path=repo_path,
        provider=provider,
        model=model,
        max_fix_attempts=max_fix_attempts,
        langsmith_tracing=trace,
        langsmith_project=trace_project,
    )
    tracing = configure_tracing(settings)
    if tracing:
        console.print(f"[dim]Tracing to LangSmith project '{settings.langsmith_project}'.[/dim]")
    elif trace:
        # Asked for explicitly, so say why it is not happening rather than
        # running quietly untraced.
        console.print("[yellow]--trace needs LANGSMITH_API_KEY; running untraced.[/yellow]")

    commands = list(validate or [])
    if auto_validate:
        detected = detect_validation_commands(settings.repo_path)
        commands.extend(c for c in detected if c not in commands)
        if detected:
            console.print(f"[dim]Detected checks: {', '.join(detected)}[/dim]")
    if commands:
        settings.validate_commands = commands
    if not settings.validate_commands:
        console.print(
            "[yellow]No validation commands configured — changes will not be verified. "
            "Pass --validate or --auto-validate.[/yellow]"
        )

    try:
        if git_ops.is_dirty(settings.repo_path):
            if not checkout:
                fate = (
                    "The agent stages only the files it edits, so your work will not be "
                    "swept into its commit."
                )
            elif restore_branch:
                fate = "They will be stashed before the PR branch is checked out, and restored after."
            else:
                fate = (
                    "They will be stashed before the PR branch is checked out. "
                    "Pass --restore-branch to get them back automatically."
                )
            console.print(f"[yellow]The working tree has uncommitted changes. {fate}[/yellow]")
    except git_ops.GitError:
        pass  # not a git checkout, or git is unavailable

    try:
        deps = build_deps(
            settings,
            pr_ref,
            dry_run=dry_run,
            commit=commit,
            push=push,
            checkout_branch=checkout,
            restore_branch=restore_branch,
            write_replies=reply or resolve,
            resolve_threads=resolve,
            include_review_bodies=include_review_bodies,
            only_comment_ids=list(comment_id or []),
            ignore_authors={a.lower() for a in (ignore_author or [])},
            self_login=self_login,
        )
    except GitHubError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    graph = build_agent_graph(deps)
    config = run_config(deps, thread_id=f"{pr_ref.owner}/{pr_ref.repo}#{pr_ref.number}")

    try:
        with traced_run(tracing) as trace_link:
            final = graph.invoke(initial_state(), config=config)
    except git_ops.GitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    finally:
        deps.github.close()
        if tracing:
            # The tracer uploads in batches from its own thread; this process
            # is about to exit, so wait for the tail of the trace.
            flush_traces()

    console.print()
    console.print(final.get("report", "(no report)"), markup=False, highlight=False)

    url = trace_link()
    if url:
        console.print()
        console.print(f"[dim]Trace: {url}[/dim]")

    failed = [o for o in (final.get("outcomes") or []) if o.error]
    raise typer.Exit(code=1 if failed else 0)


@app.command()
def index(
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository to index."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build or refresh the retrieval index without running the agent."""
    _setup_logging(verbose)
    settings = load_settings(repo_path=repo_path)
    workspace = Workspace(root=settings.repo_path, max_file_bytes=settings.max_index_file_bytes)
    stats = CodeIndex(settings, workspace).refresh()
    console.print(f"[green]{stats.describe()}[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    repo_path: Path = typer.Option(Path("."), "--repo", "-r"),
    k: int = typer.Option(5, "--k", help="How many chunks to return."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Query the index directly — useful for checking retrieval quality."""
    _setup_logging(verbose)
    settings = load_settings(repo_path=repo_path)
    workspace = Workspace(root=settings.repo_path, max_file_bytes=settings.max_index_file_bytes)
    code_index = CodeIndex(settings, workspace)
    code_index.refresh()
    for chunk in code_index.retriever.retrieve(query, k=k):
        console.print(
            f"[cyan]{chunk.path}:{chunk.start_line}-{chunk.end_line}[/cyan] "
            f"[dim](score {chunk.score:.4f} via {chunk.source})[/dim]"
        )
        console.print(chunk.content[:800], markup=False, highlight=False)
        console.print()


@app.command()
def comments(
    pr: str = typer.Argument(..., help="Pull request as owner/repo#123 or a GitHub URL."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List the review threads the agent would consider, without acting."""
    _setup_logging(verbose)
    from .github_client import build_threads

    settings = load_settings()
    try:
        pr_ref = PRRef.parse(pr)
        with GitHubClient(settings.github_token, api_url=settings.github_api_url) as client:
            threads = build_threads(client.list_review_comments(pr_ref))
    except (ValueError, GitHubError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if not threads:
        console.print("No review comments found.")
        return
    for thread in threads:
        location = f"{thread.path}:{thread.line}" if thread.path else "(general)"
        console.print(
            f"[cyan]{thread.id}[/cyan] [dim]{location}[/dim] @{thread.root.author} "
            f"({len(thread.replies)} repl{'y' if len(thread.replies) == 1 else 'ies'})"
        )
        console.print(f"  {thread.root.body.strip()[:300]}", markup=False, highlight=False)
        console.print()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind."),
    port: int = typer.Option(8765, "--port", "-p", help="Port to listen on."),
    repo_path: Path = typer.Option(
        Path("."), "--repo", "-r", help="Repository the UI defaults to."
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Serve the web UI."""
    _setup_logging(verbose)
    try:
        import uvicorn
    except ImportError as exc:
        console.print(
            "[red]The web UI needs extra packages: "
            "pip install 'pr-review-implementor[web]'[/red]"
        )
        raise typer.Exit(code=2) from exc

    # The server edits files and runs your validation commands, so binding it
    # beyond localhost exposes both to anyone who can reach the port.
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            f"[yellow]Binding to {host}. This server has no authentication and can "
            "edit files and run commands — only do this on a network you trust.[/yellow]"
        )

    # The app reads the repo path from each request; seed the default here so
    # the UI opens on the right checkout.
    import os

    os.environ.setdefault("PR_AGENT_REPO_PATH", str(repo_path))

    # Each run configures tracing again from its own settings; this is so the
    # server says up front whether it is on.
    settings = load_settings(repo_path=repo_path)
    if configure_tracing(settings):
        console.print(
            f"[dim]Tracing to LangSmith project '{settings.langsmith_project}'.[/dim]"
        )

    console.print(f"[green]UI on http://{host}:{port}[/green]")
    uvicorn.run(
        "pr_agent.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="debug" if verbose else "info",
    )


def main() -> None:  # pragma: no cover - console-script shim
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
