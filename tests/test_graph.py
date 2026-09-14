"""End-to-end runs of the compiled graph against a fake GitHub and a fake model."""

from __future__ import annotations

import pytest
from fakes import FakeChatModel
from langchain_core.messages import AIMessage

from pr_agent.config import Settings
from pr_agent.graph.build import AgentDeps, build_agent_graph
from pr_agent.graph.state import initial_state
from pr_agent.models import (
    ChangePlan,
    CommentAction,
    PRRef,
    PullRequest,
    ReviewComment,
    Triage,
)
from pr_agent.rag.index import CodeIndex
from pr_agent.workspace import Workspace

PR_REF = PRRef(owner="octo", repo="demo", number=1)

GUARD = '''"""Arithmetic helpers."""


def divide(a, b):
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b


def add(a, b):
    return a + b
'''


class FakeGitHub:
    """Implements only the client surface the nodes touch."""

    def __init__(self, comments: list[ReviewComment]) -> None:
        self._comments = comments
        self.replies: list[tuple[int, str]] = []
        self.resolved: list[int] = []
        self.closed = False

    def get_pull_request(self, ref):
        return PullRequest(
            ref=ref,
            title="Add arithmetic helpers",
            body="",
            head_ref="feature/calc",
            head_sha="abc123",
            base_ref="main",
        )

    def list_review_comments(self, _ref):
        return list(self._comments)

    def list_review_bodies(self, _ref):
        return []

    def reply_to_review_comment(self, _ref, comment_id, body):
        self.replies.append((comment_id, body))
        return {}

    def resolve_review_thread(self, _ref, comment_id):
        self.resolved.append(comment_id)
        return True

    def close(self):
        self.closed = True


def make_deps(repo, llm, github, **behaviour) -> AgentDeps:
    settings = Settings(
        repo_path=repo,
        embedding_backend="none",  # keep tests offline: BM25-only retrieval
        validate_commands=behaviour.pop("validate_commands", []),
        max_fix_attempts=behaviour.pop("max_fix_attempts", 0),
    )
    workspace = Workspace(root=repo)
    # Most of these tests run against a fixture with no remote to fetch the
    # PR branch from, so the checkout step is opt-in here; it has its own tests.
    behaviour.setdefault("checkout_branch", False)
    return AgentDeps(
        settings=settings,
        pr_ref=PR_REF,
        github=github,
        workspace=workspace,
        index=CodeIndex(settings, workspace),
        llm=llm,
        **behaviour,
    )


def run(deps) -> dict:
    graph = build_agent_graph(deps)
    return graph.invoke(
        initial_state(),
        config={"configurable": {"thread_id": "t"}, "recursion_limit": 200},
    )


def comment(body="Please raise on division by zero.", cid=101, path="src/calc.py", line=5):
    return ReviewComment(
        id=cid,
        body=body,
        author="reviewer",
        path=path,
        line=line,
        diff_hunk="@@ -1,4 +1,6 @@\n+def divide(a, b):\n+    return a / b",
    )


def implementing_llm(new_content=GUARD, summary="Added a zero check to divide()."):
    return FakeChatModel(
        structured={
            "Triage": Triage(
                action=CommentAction.IMPLEMENT,
                reason="The reviewer asks for a guard clause.",
                target_files=["src/calc.py"],
                search_queries=["divide", "division by zero"],
            ),
            "ChangePlan": ChangePlan(
                summary="Raise ValueError when b is zero.",
                steps=["Add an explicit check at the top of divide()."],
                files_to_edit=["src/calc.py"],
            ),
        },
        tool_script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "src/calc.py", "content": new_content},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content=summary),
        ],
        text="Added a zero check in src/calc.py.",
    )


# ---------------------------------------------------------------------------


def test_happy_path_implements_commits_nothing_and_records_the_change(repo):
    github = FakeGitHub([comment()])
    deps = make_deps(repo, implementing_llm(), github, commit=False)

    final = run(deps)

    assert "raise ValueError" in (repo / "src" / "calc.py").read_text()
    outcomes = final["outcomes"]
    assert len(outcomes) == 1
    assert outcomes[0].implemented
    assert outcomes[0].files_changed == ["src/calc.py"]
    assert "implemented" in final["report"]


def test_skipped_comment_makes_no_edit_and_no_reply(repo):
    original = (repo / "src" / "calc.py").read_text()
    llm = FakeChatModel(
        structured={
            "Triage": Triage(action=CommentAction.SKIP, reason="Praise, not a request."),
        }
    )
    github = FakeGitHub([comment(body="Nice, this reads well!")])
    deps = make_deps(repo, llm, github, commit=False, write_replies=True)

    final = run(deps)

    assert (repo / "src" / "calc.py").read_text() == original
    assert final["outcomes"][0].action is CommentAction.SKIP
    assert github.replies == []  # silence is the right answer to praise


def test_answer_only_comment_replies_without_editing(repo):
    original = (repo / "src" / "calc.py").read_text()
    llm = FakeChatModel(
        structured={"Triage": Triage(action=CommentAction.ANSWER_ONLY, reason="A question.")},
        text="It uses true division deliberately.",
    )
    github = FakeGitHub([comment(body="Why float division here?")])
    deps = make_deps(repo, llm, github, commit=False, write_replies=True)

    final = run(deps)

    assert (repo / "src" / "calc.py").read_text() == original
    assert len(github.replies) == 1
    assert github.replies[0][0] == 101
    assert final["outcomes"][0].action is CommentAction.ANSWER_ONLY


def test_failed_validation_rolls_the_change_back(repo):
    original = (repo / "src" / "calc.py").read_text()
    github = FakeGitHub([comment()])
    deps = make_deps(
        repo,
        implementing_llm(),
        github,
        commit=False,
        validate_commands=["exit 1"],
        max_fix_attempts=0,
    )

    final = run(deps)

    # The edit must not survive a failing check.
    assert (repo / "src" / "calc.py").read_text() == original
    outcome = final["outcomes"][0]
    assert not outcome.implemented
    assert outcome.files_changed == []
    assert outcome.validation is not None and not outcome.validation.ok


def test_failed_validation_retries_then_gives_up(repo):
    github = FakeGitHub([comment()])
    deps = make_deps(
        repo,
        implementing_llm(),
        github,
        commit=False,
        validate_commands=["exit 1"],
        max_fix_attempts=2,
    )

    final = run(deps)

    assert final["outcomes"][0].attempts == 2


def test_validation_that_passes_keeps_the_change(repo):
    github = FakeGitHub([comment()])
    deps = make_deps(
        repo, implementing_llm(), github, commit=False, validate_commands=["exit 0"]
    )

    final = run(deps)

    assert "raise ValueError" in (repo / "src" / "calc.py").read_text()
    assert final["outcomes"][0].implemented


def test_multiple_threads_are_each_processed(repo):
    comments = [comment(cid=1), comment(cid=2, body="Same again on add().", line=9)]
    github = FakeGitHub(comments)
    deps = make_deps(repo, implementing_llm(), github, commit=False)

    final = run(deps)

    assert [o.thread_id for o in final["outcomes"]] == [1, 2]


def test_dry_run_neither_commits_nor_replies(repo):
    github = FakeGitHub([comment()])
    deps = make_deps(repo, implementing_llm(), github, dry_run=True, write_replies=True, push=True)

    final = run(deps)

    assert github.replies == []
    assert final["commit_sha"] is None
    assert "dry run" in final["report"]
    # A dry run still shows what the reply would have said.
    assert final["outcomes"][0].reply


def test_replies_and_resolution_are_posted_when_enabled(repo):
    github = FakeGitHub([comment()])
    deps = make_deps(
        repo, implementing_llm(), github, commit=False, write_replies=True, resolve_threads=True
    )

    final = run(deps)

    assert len(github.replies) == 1
    assert github.resolved == [101]
    assert final["replies_posted"] == 1


def test_ignored_authors_are_filtered_out(repo):
    github = FakeGitHub([comment()])
    deps = make_deps(
        repo, implementing_llm(), github, commit=False, ignore_authors={"reviewer"}
    )

    final = run(deps)

    assert final["outcomes"] == []
    assert "No review threads" in final["report"]


def test_threads_already_answered_by_the_agent_are_skipped(repo):
    root = comment()
    reply = ReviewComment(id=102, body="Done.", author="my-bot", in_reply_to_id=101)
    github = FakeGitHub([root, reply])
    deps = make_deps(repo, implementing_llm(), github, commit=False, self_login="my-bot")

    final = run(deps)

    assert final["outcomes"] == []


def test_only_comment_ids_narrows_the_queue(repo):
    github = FakeGitHub([comment(cid=1), comment(cid=2)])
    deps = make_deps(repo, implementing_llm(), github, commit=False, only_comment_ids=[2])

    final = run(deps)

    assert [o.thread_id for o in final["outcomes"]] == [2]


def test_a_model_that_never_finishes_is_cut_off(repo):
    """The tool loop must terminate even if the model keeps calling tools."""
    loop_call = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "src/calc.py"}, "id": "c"}],
    )
    llm = FakeChatModel(
        structured={
            "Triage": Triage(action=CommentAction.IMPLEMENT, reason="fix it"),
            "ChangePlan": ChangePlan(summary="s", steps=["a"]),
        },
        tool_script=[loop_call] * 100,
    )
    github = FakeGitHub([comment()])
    deps = make_deps(repo, llm, github, commit=False)
    deps.settings.max_tool_iterations = 3

    final = run(deps)

    assert llm.tool_runnable is not None
    assert llm.tool_runnable.invocations == 3
    assert "gave up" in final["outcomes"][0].error


def test_a_tool_error_is_reported_to_the_model_not_raised(repo):
    """An edit that does not match must come back as a tool result, not a crash."""
    llm = FakeChatModel(
        structured={
            "Triage": Triage(action=CommentAction.IMPLEMENT, reason="fix it"),
            "ChangePlan": ChangePlan(summary="s", steps=["a"]),
        },
        tool_script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_file",
                        "args": {
                            "path": "src/calc.py",
                            "old_text": "text that is not there",
                            "new_text": "x",
                        },
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="I could not find that text, so I made no change."),
        ],
    )
    github = FakeGitHub([comment()])
    deps = make_deps(repo, llm, github, commit=False)

    final = run(deps)

    outcome = final["outcomes"][0]
    assert outcome.files_changed == []
    assert not outcome.implemented


def test_edits_outside_the_repository_are_refused(repo, tmp_path):
    outside = tmp_path.parent / "escape.txt"
    llm = FakeChatModel(
        structured={
            "Triage": Triage(action=CommentAction.IMPLEMENT, reason="fix it"),
            "ChangePlan": ChangePlan(summary="s", steps=["a"]),
        },
        tool_script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "../../escape.txt", "content": "pwned"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="blocked"),
        ],
    )
    github = FakeGitHub([comment()])
    deps = make_deps(repo, llm, github, commit=False)

    final = run(deps)

    assert not outside.exists()
    assert final["outcomes"][0].files_changed == []


def test_commit_stages_only_the_agents_files(repo):
    """A user's unrelated work must not be swept into the agent's commit."""
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    # Unrelated work in progress, sitting in the tree before the agent runs.
    (repo / "WIP.txt").write_text("my half-finished notes\n", encoding="utf-8")

    deps = make_deps(repo, implementing_llm(), FakeGitHub([comment()]), commit=True)
    final = run(deps)

    assert final["commit_sha"]
    committed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert committed == ["src/calc.py"]
    assert "WIP.txt" in subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout


@pytest.mark.parametrize("action", [CommentAction.IMPLEMENT, CommentAction.SKIP])
def test_report_is_always_produced(repo, action):
    llm = FakeChatModel(
        structured={
            "Triage": Triage(action=action, reason="r"),
            "ChangePlan": ChangePlan(summary="s", steps=["a"]),
        },
        tool_script=[AIMessage(content="done")],
    )
    deps = make_deps(repo, llm, FakeGitHub([comment()]), commit=False)
    final = run(deps)
    assert final["report"]
    assert "octo/demo#1" in final["report"]
# -- branch handling ------------------------------------------------------


def _git_repo_with_pr_branch(repo):
    """The fixture repo, committed on main with the PR's head branch beside it."""
    import subprocess

    for args in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@e.com"],
        ["git", "config", "user.name", "T"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
        ["git", "branch", "feature/calc"],
    ):
        subprocess.run(args, cwd=repo, check=True)
    return repo


def test_the_run_checks_out_the_pr_branch(repo):
    from pr_agent import git_ops

    _git_repo_with_pr_branch(repo)

    deps = make_deps(repo, implementing_llm(), FakeGitHub([comment()]), checkout_branch=True)
    final = run(deps)

    assert git_ops.current_branch(repo) == "feature/calc"
    assert final["branch_switch"].switched
    assert "Branch:" in final["report"]


def test_uncommitted_work_is_stashed_and_handed_back(repo):
    from pr_agent import git_ops

    _git_repo_with_pr_branch(repo)
    (repo / "WIP.txt").write_text("my half-finished notes\n", encoding="utf-8")

    deps = make_deps(
        repo,
        implementing_llm(),
        FakeGitHub([comment()]),
        checkout_branch=True,
        restore_branch=True,
        commit=True,
    )
    final = run(deps)

    # The change was committed on the PR branch...
    assert final["commit_sha"]
    # ...and the checkout was handed back exactly as it was found.
    assert git_ops.current_branch(repo) == "main"
    assert (repo / "WIP.txt").read_text(encoding="utf-8") == "my half-finished notes\n"
    assert git_ops.stash_count(repo) == 0
    assert final["branch_switch"].restored


def test_an_unpushed_commit_keeps_the_checkout_on_the_pr_branch(repo):
    """Walking away from work that only exists locally would hide it."""
    from pr_agent import git_ops

    _git_repo_with_pr_branch(repo)

    deps = make_deps(
        repo,
        implementing_llm(),
        FakeGitHub([comment()]),
        checkout_branch=True,
        restore_branch=True,
        commit=True,
        push=True,  # there is no remote, so the push fails
    )
    final = run(deps)

    assert final["commit_sha"]
    assert not final.get("pushed")
    assert git_ops.current_branch(repo) == "feature/calc"
    assert not final["branch_switch"].restored


def test_a_dry_run_still_hands_the_checkout_back(repo):
    from pr_agent import git_ops

    _git_repo_with_pr_branch(repo)
    (repo / "WIP.txt").write_text("notes\n", encoding="utf-8")

    deps = make_deps(
        repo,
        implementing_llm(),
        FakeGitHub([comment()]),
        checkout_branch=True,
        restore_branch=True,
        dry_run=True,
    )
    run(deps)

    assert git_ops.current_branch(repo) == "main"
    assert (repo / "WIP.txt").read_text(encoding="utf-8") == "notes\n"


def test_a_branch_that_cannot_be_checked_out_stops_the_run(repo):
    """Editing the wrong branch is worse than not running at all."""
    import subprocess

    from pr_agent import git_ops

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    deps = make_deps(repo, implementing_llm(), FakeGitHub([comment()]), checkout_branch=True)

    with pytest.raises(git_ops.GitError):
        run(deps)


def test_the_checkout_can_be_turned_off(repo):
    from pr_agent import git_ops

    _git_repo_with_pr_branch(repo)

    deps = make_deps(repo, implementing_llm(), FakeGitHub([comment()]), checkout_branch=False)
    final = run(deps)

    assert git_ops.current_branch(repo) == "main"
    assert final.get("branch_switch") is None
