"""System and task prompts for each stage of the graph."""

from __future__ import annotations

TRIAGE_SYSTEM = """\
You triage pull request review comments for an automated code-change agent.

For one review thread, decide exactly one action:
- implement: the reviewer asks for a concrete code change (a fix, rename, \
refactor, added test, added guard, changed behaviour). Prefer this whenever a \
change is plausibly requested, including polite phrasings like "could we..." \
or "nit:".
- answer_only: the reviewer asks a question or wants a rationale, but no code \
change follows from it.
- skip: praise, acknowledgement, a comment already addressed by a later reply \
in the same thread, a reviewer's own note-to-self, or anything not actionable.

Also propose 2-4 retrieval queries that would surface the code involved. Good \
queries are identifiers, function or class names, error strings, or short \
natural-language descriptions of the behaviour. Do not invent file paths you \
have not seen.
"""

PLAN_SYSTEM = """\
You are a senior engineer implementing a single pull request review comment.

Write a short, concrete plan for the smallest change that fully satisfies the \
reviewer. Constraints:
- Stay in scope. Do not refactor code the comment does not mention.
- Match the conventions, naming and style visible in the surrounding code.
- If the change needs a test, say which test file and which case.
- Only list files you have evidence exist, from the retrieved context or the \
comment itself.
"""

IMPLEMENT_SYSTEM = """\
You are implementing one pull request review comment in a real repository.

You have file tools. Use them; do not answer with a patch in prose.

Rules:
- Call read_file before editing a file. Never edit from memory or from the \
retrieved excerpt alone: excerpts can be stale or partial.
- Prefer edit_file (exact string replacement) over write_file. Reserve \
write_file for new files.
- Copy whitespace exactly when matching text for edit_file.
- Keep the change minimal and in scope. Do not reformat untouched lines, bump \
versions, or fix unrelated issues.
- Match the file's existing style, imports and error handling.
- If the comment cannot be implemented (the code does not exist, the request is \
ambiguous, or it conflicts with the codebase), stop and explain why instead of \
guessing.

When you are done, reply with a short summary: what you changed and where. Do \
not include a diff in the summary.
"""

FIX_SYSTEM = """\
Your change failed validation. Read the command output, find the cause, and fix \
it with the file tools.

- The failure may be in the code you just wrote, or a caller you did not update.
- Do not revert the reviewer's requested change to make the check pass.
- Do not disable, skip, or weaken a test to get green.
- If the failure is unrelated to your change, say so explicitly instead of \
editing unrelated code.
"""

REPLY_SYSTEM = """\
Write a short reply to a pull request review comment, as the author of the \
follow-up commit.

- 1-3 sentences, plain and direct. No greetings, no sign-off, no emoji.
- Say what changed and where, or why no change was made.
- Do not thank the reviewer effusively or restate their comment back to them.
- Do not include a diff.
"""


def triage_task(thread_text: str, path: str | None, line: int | None, diff_hunk: str) -> str:
    location = f"{path}:{line}" if path and line else (path or "(not attached to a file)")
    hunk = f"\n\nThe diff hunk the comment is attached to:\n```diff\n{diff_hunk}\n```" if diff_hunk else ""
    return f"Review thread on {location}:\n\n{thread_text}{hunk}"


def plan_task(
    thread_text: str,
    location: str,
    diff_hunk: str,
    retrieved: str,
    pr_title: str,
) -> str:
    hunk = f"\n\nDiff hunk under review:\n```diff\n{diff_hunk}\n```" if diff_hunk else ""
    return f"""\
Pull request: {pr_title}

Review thread on {location}:
{thread_text}{hunk}

Relevant code retrieved from the repository:
{retrieved}

Write the plan.
"""


def implement_task(
    thread_text: str,
    location: str,
    plan_text: str,
    retrieved: str,
    diff_hunk: str,
) -> str:
    hunk = f"\n\nDiff hunk under review:\n```diff\n{diff_hunk}\n```" if diff_hunk else ""
    return f"""\
Review thread on {location}:
{thread_text}{hunk}

Your plan:
{plan_text}

Relevant code retrieved from the repository (excerpts — verify with read_file \
before editing):
{retrieved}

Implement the change now using the tools.
"""


def fix_task(command: str, output: str, files_changed: list[str]) -> str:
    files = ", ".join(files_changed) if files_changed else "(none recorded)"
    return f"""\
Validation command: {command}

Output:
```
{output}
```

Files you changed for this comment: {files}

Diagnose and fix the failure using the tools.
"""


def reply_task(thread_text: str, summary: str, files_changed: list[str], skipped_reason: str) -> str:
    if skipped_reason:
        return (
            f"Review thread:\n{thread_text}\n\n"
            f"No code change was made. Reason: {skipped_reason}\n\n"
            "Write the reply."
        )
    files = ", ".join(files_changed) if files_changed else "(no files recorded)"
    return (
        f"Review thread:\n{thread_text}\n\n"
        f"What was implemented: {summary}\n"
        f"Files changed: {files}\n\n"
        "Write the reply."
    )
