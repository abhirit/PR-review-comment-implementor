"""Git operations: staging, committing and pushing the agent's work."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .models import BranchSwitch

log = logging.getLogger(__name__)


class GitError(RuntimeError):
    """A git command failed."""


def _run(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"Could not run git in {repo_path}: {exc}") from exc


def _git(repo_path: Path, *args: str, check: bool = True, strip: bool = True) -> str:
    result = _run(repo_path, *args)
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip() if strip else result.stdout


def _succeeds(repo_path: Path, *args: str) -> bool:
    """Run a git command purely for its exit status."""
    return _run(repo_path, *args).returncode == 0


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


# ---------------------------------------------------------------------------
# Branch switching
#
# The agent must edit the code the reviewer was looking at, so a run starts by
# putting the checkout on the pull request's head branch. Whatever the user had
# in flight is stashed first and handed back afterwards, so borrowing the
# checkout leaves no trace.
# ---------------------------------------------------------------------------


def local_branch_exists(repo_path: Path, branch: str) -> bool:
    return _succeeds(repo_path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")


def remote_branch_exists(repo_path: Path, branch: str, remote: str = "origin") -> bool:
    return _succeeds(
        repo_path, "rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{branch}"
    )


def has_remote(repo_path: Path, remote: str = "origin") -> bool:
    return remote in _git(repo_path, "remote").splitlines()


def checkout(repo_path: Path, ref: str) -> None:
    _git(repo_path, "checkout", ref)


def stash_push(repo_path: Path, message: str) -> str | None:
    """Stash the working tree, including untracked files.

    Returns the stash entry's commit sha, or ``None`` if there was nothing to
    stash. Ignored files are deliberately left alone: they are usually build
    output and the retrieval index the agent itself maintains.
    """
    before = _stash_head(repo_path)
    _git(repo_path, "stash", "push", "--include-untracked", "--message", message)
    after = _stash_head(repo_path)
    if after is None or after == before:
        return None
    log.info("stashed the working tree as %s", after[:10])
    return after


def stash_count(repo_path: Path) -> int:
    """How many entries are on the stash, for reporting what is parked."""
    listing = _git(repo_path, "stash", "list", "--format=%H", check=False)
    return len([line for line in listing.splitlines() if line.strip()])


def stash_pop(repo_path: Path, stash_sha: str) -> bool:
    """Restore a stash entry by its commit sha.

    The entry is looked up by sha rather than by ``stash@{0}``: anything the
    run did could have pushed another entry on top, and popping the wrong one
    would hand back the wrong work.
    """
    ref = _find_stash_ref(repo_path, stash_sha)
    if ref is None:
        log.warning("Stash entry %s is no longer in the stash list", stash_sha[:10])
        return False
    _git(repo_path, "stash", "pop", ref)
    log.info("restored the stashed changes from %s", stash_sha[:10])
    return True


def _stash_head(repo_path: Path) -> str | None:
    result = _run(repo_path, "rev-parse", "--verify", "--quiet", "refs/stash")
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _find_stash_ref(repo_path: Path, stash_sha: str) -> str | None:
    """Map a stash commit sha back to the ``stash@{n}`` selector git needs."""
    listing = _git(repo_path, "stash", "list", "--format=%H", check=False)
    for position, sha in enumerate(listing.splitlines()):
        if sha.strip() == stash_sha:
            return "stash@{" + str(position) + "}"
    return None


def prepare_branch(
    repo_path: Path,
    branch: str,
    pr_number: int | None = None,
    remote: str = "origin",
) -> BranchSwitch:
    """Put the checkout on ``branch``, stashing any uncommitted work first.

    Returns a :class:`BranchSwitch` describing what changed, which
    :func:`restore_branch` can undo.
    """
    if not is_git_repo(repo_path):
        return BranchSwitch(
            branch=branch, detail="Not a git checkout; leaving the files as they are."
        )

    previous = current_branch(repo_path)
    previous_sha = _git(repo_path, "rev-parse", "HEAD", check=False) or None

    if previous == branch:
        return BranchSwitch(
            branch=branch,
            previous_branch=previous,
            previous_sha=previous_sha,
            detail=f"Already on {branch}.",
        )

    stash_sha = None
    if is_dirty(repo_path):
        stash_sha = stash_push(repo_path, f"pr-agent: work in progress on {previous}")

    try:
        created = _switch_to(repo_path, branch, pr_number, remote)
    except GitError:
        # The switch failed, so put the user's work back before giving up:
        # leaving it stashed with no branch change is only confusing.
        if stash_sha:
            try:
                stash_pop(repo_path, stash_sha)
            except GitError as pop_error:  # pragma: no cover - defensive
                log.error("Could not restore the stash after a failed checkout: %s", pop_error)
        raise

    detail = f"Switched from {previous} to {branch}"
    if created:
        detail += " (created from the remote)"
    if stash_sha:
        detail += f", stashing uncommitted work on {previous}"
    return BranchSwitch(
        branch=branch,
        previous_branch=previous,
        previous_sha=previous_sha,
        switched=True,
        created=created,
        stash_sha=stash_sha,
        detail=f"{detail}.",
    )


def _switch_to(repo_path: Path, branch: str, pr_number: int | None, remote: str) -> bool:
    """Check ``branch`` out, fetching it if it is not here yet.

    Returns True if the local branch had to be created.
    """
    if local_branch_exists(repo_path, branch):
        checkout(repo_path, branch)
        return False

    if not has_remote(repo_path, remote):
        raise GitError(
            f"Branch '{branch}' is not in this checkout and there is no '{remote}' remote "
            "to fetch it from."
        )

    # Spell the refspec out rather than relying on `git fetch origin <branch>`:
    # a single-branch clone's configured refspec does not cover other branches,
    # so the shorthand would land in FETCH_HEAD and create no tracking ref.
    # A failed fetch is not fatal here — the pull/N/head fallback below still
    # covers a pull request opened from a fork.
    if not _succeeds(
        repo_path,
        "fetch",
        remote,
        f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
    ):
        log.debug("Could not fetch %s from %s; trying the pull request ref", branch, remote)

    if remote_branch_exists(repo_path, branch, remote):
        _git(repo_path, "checkout", "-b", branch, f"{remote}/{branch}")
        # Tracking is a convenience, not a requirement: push() always names the
        # remote and branch, and a single-branch clone refuses to set it at all.
        _succeeds(repo_path, "branch", "--set-upstream-to", f"{remote}/{branch}", branch)
        return True

    # A fork's head branch does not exist on this remote, but GitHub publishes
    # every pull request as pull/<number>/head on the base repository.
    if pr_number is not None and _succeeds(
        repo_path, "fetch", remote, f"pull/{pr_number}/head:{branch}"
    ):
        checkout(repo_path, branch)
        return True

    raise GitError(
        f"Could not find branch '{branch}' locally or on '{remote}'. "
        "Check it out yourself, or turn the automatic checkout off."
    )


def restore_branch(repo_path: Path, switch: BranchSwitch) -> BranchSwitch:
    """Undo :func:`prepare_branch`: return to the old branch and pop the stash.

    Never raises. A checkout that cannot be undone is reported through
    ``restore_detail`` rather than failing a run whose work is already
    committed and pushed.
    """
    if not switch.switched and not switch.stash_sha:
        return switch

    notes: list[str] = []
    target = switch.previous_branch
    if switch.switched and target:
        # A detached HEAD has no branch name to go back to, only the commit.
        ref = switch.previous_sha if target == "HEAD" else target
        try:
            checkout(repo_path, ref or target)
            notes.append(f"back on {target}")
        except GitError as exc:
            stash_note = (
                f", and your changes are still stashed ({switch.stash_sha[:10]})"
                if switch.stash_sha
                else ""
            )
            return switch.model_copy(
                update={
                    "restore_detail": (
                        f"Could not return to {target}: {exc}. The checkout is still on "
                        f"{switch.branch}{stash_note}."
                    )
                }
            )

    if switch.stash_sha:
        try:
            popped = stash_pop(repo_path, switch.stash_sha)
            notes.append(
                "restored the stashed changes"
                if popped
                else f"the stash entry {switch.stash_sha[:10]} was already gone"
            )
        except GitError as exc:
            notes.append(
                f"could not pop the stash automatically ({exc}); "
                "it is still listed by `git stash list`"
            )

    detail = ", ".join(notes)
    return switch.model_copy(
        update={
            "restored": True,
            "restore_detail": f"{detail[:1].upper()}{detail[1:]}." if detail else "",
        }
    )
