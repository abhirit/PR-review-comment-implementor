"""Node implementations.

Each node is a small function over :class:`AgentState`; the shared, non-
serializable resources arrive via ``deps`` bound at graph-construction time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .. import prompts
from ..github_client import build_threads
from ..models import (
    BranchSwitch,
    ChangePlan,
    CommentAction,
    ReviewThread,
    ThreadOutcome,
    Triage,
    ValidationResult,
    truncate,
)
from ..rag.retriever import render_chunks
from ..tools import build_file_tools, run_tool_loop, text_of
from ..validation import run_validation
from .state import AgentState

if TYPE_CHECKING:  # pragma: no cover
    from .build import AgentDeps

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load the pull request and its review threads
# ---------------------------------------------------------------------------


def make_load_pr(deps: AgentDeps):
    def load_pr(state: AgentState) -> AgentState:
        pr = deps.github.get_pull_request(deps.pr_ref)
        log.info("Loaded %s: %s", deps.pr_ref.slug, pr.title)

        comments = deps.github.list_review_comments(deps.pr_ref)
        threads = build_threads(comments)

        if deps.include_review_bodies:
            bodies = deps.github.list_review_bodies(deps.pr_ref)
            threads.extend(ReviewThread(root=body) for body in bodies)

        threads = [t for t in threads if _is_candidate(t, deps)]

        if deps.only_comment_ids:
            wanted = set(deps.only_comment_ids)
            threads = [t for t in threads if t.id in wanted]

        log.info("%d review thread(s) to consider", len(threads))
        return {
            "pull_request": pr,
            "threads": threads,
            "pending": list(range(len(threads))),
            "outcomes": [],
        }

    return load_pr


def _is_candidate(thread: ReviewThread, deps: AgentDeps) -> bool:
    """Filter out threads the agent should not act on."""
    author = thread.root.author.lower()
    if author in deps.ignore_authors:
        return False
    if not thread.root.body.strip():
        return False
    # A thread whose most recent message is from the agent itself has already
    # been handled in an earlier run.
    last = thread.replies[-1] if thread.replies else thread.root
    if deps.self_login and last.author.lower() == deps.self_login.lower():
        return False
    return True


# ---------------------------------------------------------------------------
# 2. Put the checkout on the pull request's branch
# ---------------------------------------------------------------------------


def make_prepare_branch(deps: AgentDeps):
    def prepare_branch(state: AgentState) -> AgentState:
        from .. import git_ops

        pr = state.get("pull_request")
        if pr is None or not deps.checkout_branch:
            return {}

        # Editing the wrong branch produces plausible-looking changes against
        # code the reviewer never saw, so a failure here stops the run rather
        # than carrying on with whatever happens to be checked out.
        switch = git_ops.prepare_branch(
            deps.workspace.root, pr.head_ref, pr_number=deps.pr_ref.number
        )
        log.info("branch: %s", switch.detail)
        return {"branch_switch": switch}

    return prepare_branch


# ---------------------------------------------------------------------------
# 3. Index the repository for retrieval
# ---------------------------------------------------------------------------


def make_index_repo(deps: AgentDeps):
    def index_repo(state: AgentState) -> AgentState:
        stats = deps.index.refresh()
        return {"index_summary": stats.describe()}

    return index_repo


# ---------------------------------------------------------------------------
# 4. Pick the next thread off the queue
# ---------------------------------------------------------------------------


def make_next_thread(deps: AgentDeps):
    def next_thread(state: AgentState) -> AgentState:
        pending = list(state.get("pending") or [])
        if not pending:
            return {"current": None}
        current = pending.pop(0)
        thread = state["threads"][current]
        log.info(
            "--- thread %s (%s) by @%s ---",
            thread.id,
            thread.path or "no file",
            thread.root.author,
        )
        # Isolate this thread's edits so they can be rolled back on failure.
        deps.workspace.begin()
        return {
            "pending": pending,
            "current": current,
            "triage": None,
            "retrieved": "",
            "plan": "",
            "implementation_summary": "",
            "files_changed": [],
            "attempts": 0,
            "validation_ok": True,
            "validation_command": "",
            "validation_output": "",
            "last_error": "",
        }

    return next_thread


# ---------------------------------------------------------------------------
# 5. Triage
# ---------------------------------------------------------------------------


def make_triage(deps: AgentDeps):
    model = deps.llm.with_structured_output(Triage)

    def triage(state: AgentState) -> AgentState:
        thread = _current_thread(state)
        task = prompts.triage_task(
            thread.transcript(), thread.path, thread.line, thread.root.diff_hunk
        )
        try:
            result = model.invoke(
                [SystemMessage(prompts.TRIAGE_SYSTEM), HumanMessage(task)]
            )
        except Exception as exc:  # noqa: BLE001 - one bad thread must not kill the run
            log.error("Triage failed for thread %s: %s", thread.id, exc)
            return {
                "triage": Triage(
                    action=CommentAction.SKIP, reason=f"Triage failed: {exc}"
                ),
                "last_error": str(exc),
            }
        assert isinstance(result, Triage)
        log.info("triage: %s — %s", result.action.value, result.reason)
        return {"triage": result}

    return triage


# ---------------------------------------------------------------------------
# 6. Retrieve context (RAG)
# ---------------------------------------------------------------------------


def make_retrieve(deps: AgentDeps):
    def retrieve(state: AgentState) -> AgentState:
        thread = _current_thread(state)
        triage_result = state.get("triage")

        queries: list[str] = []
        if triage_result and triage_result.search_queries:
            queries.extend(triage_result.search_queries)
        # The comment text itself is always a query: reviewers quote identifiers.
        queries.append(thread.root.body)
        if thread.path:
            queries.append(thread.path)

        chunks = deps.index.retriever.retrieve_many(queries, k=deps.settings.retrieval_k)

        # The file the comment is attached to is the single most relevant
        # context there is, so pin it ahead of whatever retrieval returned.
        pinned = _pinned_context(deps, thread)
        rendered = render_chunks(chunks)
        retrieved = f"{pinned}\n\n{rendered}" if pinned else rendered

        log.info(
            "retrieved %d chunk(s) from %d quer(y|ies)", len(chunks), len(queries)
        )
        return {"retrieved": retrieved}

    return retrieve


def _pinned_context(deps: AgentDeps, thread: ReviewThread) -> str:
    """The lines around the comment's anchor in its own file, if it has one."""
    if not thread.path or not deps.workspace.exists(thread.path):
        return ""
    line = thread.line or 1
    start = max(1, line - 60)
    end = line + 60
    try:
        excerpt = deps.workspace.read_lines(thread.path, start, end)
    except Exception as exc:  # noqa: BLE001 - context is best-effort
        log.debug("Could not read anchor context for %s: %s", thread.path, exc)
        return ""
    return (
        f"--- {thread.path} (lines {start}-{end}, the file the comment is on) ---\n"
        f"{truncate(excerpt, 12000)}"
    )


# ---------------------------------------------------------------------------
# 7. Plan
# ---------------------------------------------------------------------------


def make_plan(deps: AgentDeps):
    model = deps.llm.with_structured_output(ChangePlan)

    def plan(state: AgentState) -> AgentState:
        thread = _current_thread(state)
        pr = state.get("pull_request")
        location = _location(thread)
        task = prompts.plan_task(
            thread.transcript(),
            location,
            thread.root.diff_hunk,
            state.get("retrieved", ""),
            pr.title if pr else "",
        )
        try:
            result = model.invoke([SystemMessage(prompts.PLAN_SYSTEM), HumanMessage(task)])
        except Exception as exc:  # noqa: BLE001
            log.error("Planning failed for thread %s: %s", thread.id, exc)
            return {"last_error": f"planning failed: {exc}"}
        assert isinstance(result, ChangePlan)

        steps = "\n".join(f"{i}. {step}" for i, step in enumerate(result.steps, 1))
        text = f"{result.summary}\n\n{steps}"
        if result.files_to_edit:
            text += f"\n\nExpected files: {', '.join(result.files_to_edit)}"
        if result.risks:
            text += f"\n\nRisks: {result.risks}"
        log.info("plan: %s", result.summary)
        return {"plan": text}

    return plan


# ---------------------------------------------------------------------------
# 8. Implement
# ---------------------------------------------------------------------------


def make_implement(deps: AgentDeps):
    def implement(state: AgentState) -> AgentState:
        tools = build_file_tools(deps.workspace, deps.index.retriever)
        model = deps.llm.bind_tools(tools)
        thread = _current_thread(state)
        if state.get("last_error", "").startswith("planning failed"):
            return {"implementation_summary": "", "files_changed": []}

        task = prompts.implement_task(
            thread.transcript(),
            _location(thread),
            state.get("plan", ""),
            state.get("retrieved", ""),
            thread.root.diff_hunk,
        )
        messages: list[BaseMessage] = [
            SystemMessage(prompts.IMPLEMENT_SYSTEM),
            HumanMessage(task),
        ]
        summary, error = run_tool_loop(
            model, tools, messages, deps.settings.max_tool_iterations
        )
        # Ask the open transaction, not `touched`: the latter accumulates across
        # threads, so a file an earlier thread edited would look unchanged here.
        changed = deps.workspace.pending_paths()
        log.info("implement: %d file(s) changed", len(changed))
        return {
            "implementation_summary": summary,
            "files_changed": changed,
            "last_error": error,
        }

    return implement


def make_fix(deps: AgentDeps):
    def fix(state: AgentState) -> AgentState:
        tools = build_file_tools(deps.workspace, deps.index.retriever)
        model = deps.llm.bind_tools(tools)
        attempts = state.get("attempts", 0) + 1
        log.info("fix attempt %d/%d", attempts, deps.settings.max_fix_attempts)
        task = prompts.fix_task(
            state.get("validation_command", ""),
            state.get("validation_output", ""),
            state.get("files_changed", []),
        )
        messages: list[BaseMessage] = [SystemMessage(prompts.FIX_SYSTEM), HumanMessage(task)]
        summary, error = run_tool_loop(
            model, tools, messages, deps.settings.max_tool_iterations
        )
        changed = sorted(
            set(state.get("files_changed", [])) | set(deps.workspace.pending_paths())
        )
        return {
            "attempts": attempts,
            "implementation_summary": (
                f"{state.get('implementation_summary', '')}\n\nFix: {summary}".strip()
            ),
            "files_changed": changed,
            "last_error": error,
        }

    return fix


# ---------------------------------------------------------------------------
# 9. Validate
# ---------------------------------------------------------------------------


def make_validate(deps: AgentDeps):
    def validate(state: AgentState) -> AgentState:
        if not state.get("files_changed"):
            return {"validation_ok": True, "validation_output": "no files changed"}
        result = run_validation(
            deps.workspace.root, deps.settings.validate_commands, deps.settings.validate_timeout
        )
        if result.ok:
            log.info("validation passed")
        else:
            log.warning("validation failed (%s): exit %d", result.command, result.exit_code)
        return {
            "validation_ok": result.ok,
            "validation_command": result.command,
            "validation_output": result.short_output,
        }

    return validate


# ---------------------------------------------------------------------------
# 10. Record the outcome for this thread
# ---------------------------------------------------------------------------


def make_record(deps: AgentDeps):
    def record(state: AgentState) -> AgentState:
        thread = _current_thread(state)
        triage_result = state.get("triage") or Triage(
            action=CommentAction.SKIP, reason="not triaged"
        )
        validation_ok = state.get("validation_ok", True)
        files = state.get("files_changed", [])
        error = state.get("last_error", "")

        keep = triage_result.action is CommentAction.IMPLEMENT and validation_ok and not error
        # Capture the diff before either branch clears the snapshots.
        diff = deps.workspace.pending_diff() if keep else ""
        if keep:
            committed = deps.workspace.commit_changes()
            # Without this the next thread retrieves the superseded version of
            # what we just wrote. A rollback needs none: restoring the snapshot
            # puts back exactly the content the index already holds.
            deps.index.reindex(committed)
        else:
            rolled_back = deps.workspace.rollback()
            if rolled_back:
                log.warning(
                    "Rolled back %d file(s) for thread %s", len(rolled_back), thread.id
                )
                files = []

        validation = ValidationResult(
            ok=validation_ok,
            command=state.get("validation_command", ""),
            output=state.get("validation_output", ""),
        )

        # Reply when the reviewer is owed an answer: a change was made, a
        # question was asked, or the agent tried and could not finish. Silence
        # is right for "skip" (praise, already-resolved, not actionable).
        owed_reply = (
            triage_result.action in (CommentAction.IMPLEMENT, CommentAction.ANSWER_ONLY)
            or bool(error)
            or not validation_ok
        )
        reply = ""
        if owed_reply and (deps.write_replies or deps.dry_run):
            reply = _compose_reply(deps, thread, state, keep, error)

        outcome = ThreadOutcome(
            thread_id=thread.id,
            path=thread.path,
            action=triage_result.action,
            reason=triage_result.reason,
            summary=state.get("implementation_summary", ""),
            files_changed=files,
            diff=diff,
            reply=reply,
            attempts=state.get("attempts", 0),
            validation=validation,
            error=error if not keep else "",
        )
        outcomes = list(state.get("outcomes") or [])
        outcomes.append(outcome)
        return {"outcomes": outcomes, "current": None}

    return record


def _compose_reply(
    deps: AgentDeps, thread: ReviewThread, state: AgentState, kept: bool, error: str
) -> str:
    triage_result = state.get("triage")
    if kept:
        reason = ""
        summary = state.get("implementation_summary", "") or "Change applied."
    else:
        summary = ""
        if error:
            reason = f"the agent could not complete the change: {error}"
        elif not state.get("validation_ok", True):
            reason = (
                f"the change did not pass `{state.get('validation_command', '')}` "
                "and was rolled back"
            )
        elif triage_result:
            reason = triage_result.reason
        else:
            reason = "no change was required"

    task = prompts.reply_task(
        thread.transcript(), summary, state.get("files_changed", []), reason
    )
    try:
        response = deps.llm.invoke([SystemMessage(prompts.REPLY_SYSTEM), HumanMessage(task)])
    except Exception as exc:  # noqa: BLE001 - a missing reply is not fatal
        log.error("Could not compose a reply for thread %s: %s", thread.id, exc)
        return ""
    assert isinstance(response, AIMessage)
    return text_of(response)


# ---------------------------------------------------------------------------
# 11. Finalize: commit, push, reply
# ---------------------------------------------------------------------------


def make_finalize(deps: AgentDeps):
    def finalize(state: AgentState) -> AgentState:
        from .. import git_ops

        outcomes = state.get("outcomes") or []
        implemented = [o for o in outcomes if o.implemented]
        updates: AgentState = {}

        if deps.dry_run:
            log.info("Dry run: not committing, pushing or replying.")
            _restore_branch(deps, state, updates)
            return {**updates, "report": build_report({**state, **updates}, dry_run=True)}

        if implemented and deps.commit:
            message = _commit_message(state, implemented)
            # Stage only what the agent wrote. Staging everything would sweep
            # up unrelated work already in the user's tree.
            paths = sorted({path for o in implemented for path in o.files_changed})
            try:
                sha = git_ops.commit(deps.workspace.root, message, paths=paths)
                updates["commit_sha"] = sha
                if sha:
                    log.info("Committed %s", sha[:10])
            except git_ops.GitError as exc:
                log.error("Commit failed: %s", exc)
                updates["last_error"] = f"commit failed: {exc}"

            if updates.get("commit_sha") and deps.push:
                pr = state.get("pull_request")
                branch = deps.push_branch or (pr.head_ref if pr else None)
                if branch:
                    try:
                        git_ops.push(deps.workspace.root, branch)
                        updates["pushed"] = True
                        log.info("Pushed to %s", branch)
                    except git_ops.GitError as exc:
                        log.error("Push failed: %s", exc)
                        updates["last_error"] = f"push failed: {exc}"

        if deps.write_replies:
            posted = 0
            for outcome in outcomes:
                if not outcome.reply:
                    continue
                try:
                    deps.github.reply_to_review_comment(
                        deps.pr_ref, outcome.thread_id, outcome.reply
                    )
                    posted += 1
                    if deps.resolve_threads and outcome.implemented:
                        deps.github.resolve_review_thread(deps.pr_ref, outcome.thread_id)
                except Exception as exc:  # noqa: BLE001 - a failed reply is not fatal
                    log.error("Could not reply to thread %s: %s", outcome.thread_id, exc)
            updates["replies_posted"] = posted
            log.info("Posted %d repl(y|ies)", posted)

        _restore_branch(deps, state, updates)

        merged = {**state, **updates}
        updates["report"] = build_report(merged, dry_run=False)
        return updates

    return finalize


def _restore_branch(deps: AgentDeps, state: AgentState, updates: AgentState) -> None:
    """Hand the checkout back the way it was found, if the run asked for it.

    Skipped when a commit was made but could not be pushed: the work only
    exists on this branch, and walking away from it would hide that.
    """
    from .. import git_ops

    switch = state.get("branch_switch")
    if not deps.restore_branch or not isinstance(switch, BranchSwitch):
        return
    if not switch.switched and not switch.stash_sha:
        return
    if deps.push and updates.get("commit_sha") and not updates.get("pushed"):
        log.warning(
            "Staying on %s: the commit was not pushed, so the work is only on this branch.",
            switch.branch,
        )
        return

    restored = git_ops.restore_branch(deps.workspace.root, switch)
    if restored.restore_detail:
        log.info("branch: %s", restored.restore_detail)
    updates["branch_switch"] = restored


def _commit_message(state: AgentState, implemented: list[ThreadOutcome]) -> str:
    pr = state.get("pull_request")
    header = f"Address review comments on {pr.ref.slug}" if pr else "Address review comments"
    if len(implemented) == 1 and implemented[0].summary:
        first_line = implemented[0].summary.strip().splitlines()[0]
        header = truncate(first_line, 72)

    bullets = "\n".join(
        f"- [{o.path or 'general'}] {truncate((o.summary or o.reason).strip().splitlines()[0], 100)}"
        for o in implemented
    )
    return f"{header}\n\n{bullets}"


def build_report(state: AgentState, dry_run: bool) -> str:
    """A human-readable summary of the run, printed by the CLI."""
    outcomes = state.get("outcomes") or []
    pr = state.get("pull_request")
    lines: list[str] = []
    header = f"Review comment run for {pr.ref.slug}" if pr else "Review comment run"
    lines.append(header)
    if dry_run:
        lines.append("(dry run — nothing was committed, pushed or posted)")
    switch = state.get("branch_switch")
    if isinstance(switch, BranchSwitch) and switch.detail:
        lines.append(f"Branch: {switch.detail}")
        if switch.restore_detail:
            lines.append(f"        {switch.restore_detail}")
    if state.get("index_summary"):
        lines.append(f"Index: {state['index_summary']}")
    lines.append("")

    if not outcomes:
        lines.append("No review threads to act on.")
        return "\n".join(lines)

    for outcome in outcomes:
        status = "implemented" if outcome.implemented else outcome.action.value
        if outcome.error:
            status = "failed"
        location = outcome.path or "general"
        lines.append(f"[{status}] thread {outcome.thread_id} ({location})")
        if outcome.reason:
            lines.append(f"    why: {outcome.reason}")
        if outcome.files_changed:
            lines.append(f"    files: {', '.join(outcome.files_changed)}")
        if outcome.error:
            lines.append(f"    error: {outcome.error}")
        if outcome.attempts:
            lines.append(f"    fix attempts: {outcome.attempts}")

    lines.append("")
    implemented = sum(1 for o in outcomes if o.implemented)
    lines.append(
        f"{implemented}/{len(outcomes)} thread(s) implemented"
        + (f", commit {state['commit_sha'][:10]}" if state.get("commit_sha") else "")
        + (", pushed" if state.get("pushed") else "")
        + (f", {state['replies_posted']} repl(y|ies) posted" if state.get("replies_posted") else "")
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_next(state: AgentState) -> str:
    return "finalize" if state.get("current") is None else "triage"


def route_after_triage(state: AgentState) -> str:
    triage_result = state.get("triage")
    if triage_result and triage_result.action is CommentAction.IMPLEMENT:
        return "retrieve"
    return "record"


def make_route_after_validate(deps: AgentDeps):
    def route_after_validate(state: AgentState) -> str:
        if state.get("validation_ok", True):
            return "record"
        if state.get("attempts", 0) >= deps.settings.max_fix_attempts:
            log.warning("Out of fix attempts; abandoning this thread's changes.")
            return "record"
        return "fix"

    return route_after_validate


def _current_thread(state: AgentState) -> ReviewThread:
    index = state.get("current")
    if index is None:  # pragma: no cover - routing prevents this
        raise RuntimeError("No current thread selected")
    return state["threads"][index]


def _location(thread: ReviewThread) -> str:
    if thread.path and thread.line:
        return f"{thread.path}:{thread.line}"
    return thread.path or "(not attached to a file)"
