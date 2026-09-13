"""Git operations: staging, committing and pushing the agent's work."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class GitError(RuntimeError):
    """A git command failed."""


def _git(repo_path: Path, *args: str, check: bool = True, strip: bool = True) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"Could not run git in {repo_path}: {exc}") from exc
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip() if strip else result.stdout


def is_git_repo(repo_path: Path) -> bool:
    try:
        return _git(repo_path, "rev-parse", "--is-inside-work-tree") == "true"
    except GitError:
        return False


def current_branch(repo_path: Path) -> str:
    """The checked-out branch name.

    ``rev-parse HEAD`` fails on a repository with no commits yet, so fall back
    to the symbolic ref, which resolves an unborn branch fine.
    """
    try:
        return _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    except GitError:
        return _git(repo_path, "symbolic-ref", "--short", "HEAD")


def is_dirty(repo_path: Path) -> bool:
    return bool(_git(repo_path, "status", "--porcelain"))


def changed_files(repo_path: Path) -> list[str]:
    """Files modified in the working tree, staged or not, plus untracked ones."""
    # Do not strip: porcelain status codes are column-significant, and a
    # leading space (" M path") is part of the format.
    out = _git(repo_path, "status", "--porcelain", strip=False)
    files: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[2:].strip()
        # Renames are reported as "old -> new"; keep the new path.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path.strip('"'))
    return files


def commit(repo_path: Path, message: str, paths: list[str] | None = None) -> str | None:
    """Stage and commit. Returns the commit SHA, or ``None`` if nothing changed."""
    if paths:
        _git(repo_path, "add", "--", *paths)
    else:
        _git(repo_path, "add", "-A")

    if not _git(repo_path, "diff", "--cached", "--name-only"):
        log.info("Nothing staged; skipping commit.")
        return None

    _git(repo_path, "commit", "-m", message)
    return _git(repo_path, "rev-parse", "HEAD")


def push(repo_path: Path, branch: str, remote: str = "origin", retries: int = 4) -> None:
    """Push with exponential backoff, since transient network errors are common."""
    import time

    delay = 2
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            _git(repo_path, "push", "-u", remote, branch)
            return
        except GitError as exc:
            last_error = exc
            if not _is_transient(str(exc)) or attempt == retries:
                raise
            log.warning("Push failed (attempt %d), retrying in %ds: %s", attempt + 1, delay, exc)
            time.sleep(delay)
            delay *= 2
    if last_error:  # pragma: no cover - loop always returns or raises
        raise last_error


def _is_transient(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "could not resolve host",
            "connection reset",
            "connection timed out",
            "timed out",
            "rpc failed",
            "unexpected disconnect",
            "early eof",
            "network",
            "503",
            "502",
        )
    )
