"""The follow-up chat: answering about a run, and changing things afterwards."""

from __future__ import annotations

import pytest
from fakes import FakeChatModel
from langchain_core.messages import AIMessage

from pr_agent.chat import ChatSession, build_run_context
from pr_agent.config import Settings
from pr_agent.rag.index import CodeIndex
from pr_agent.workspace import Workspace

GUARDED = '''"""Arithmetic helpers."""


def divide(a, b):
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b


def add(a, b):
    return a + b
'''


def make_session(repo, llm, context="# The run\n\nNothing happened.") -> ChatSession:
    settings = Settings(repo_path=repo, embedding_backend="none")
    workspace = Workspace(root=repo)
    return ChatSession(
        settings=settings,
        workspace=workspace,
        index=CodeIndex(settings, workspace),
        llm=llm,
        context=context,
    )


# -- answering ------------------------------------------------------------


def test_a_question_is_answered_without_touching_the_repository(repo):
    original = (repo / "src" / "calc.py").read_text(encoding="utf-8")
    session = make_session(
        repo, FakeChatModel(tool_script=[AIMessage(content="It guards against b == 0.")])
    )

    turn = session.ask("Why did you change divide()?")

    assert turn.role == "agent"
    assert turn.text == "It guards against b == 0."
    assert turn.files_changed == []
    assert turn.diff == ""
    assert (repo / "src" / "calc.py").read_text(encoding="utf-8") == original


def test_the_run_context_reaches_the_model(repo):
    llm = FakeChatModel(tool_script=[AIMessage(content="ok")])
    session = make_session(repo, llm, context="# The run\n\nThread 7 was skipped: it was praise.")

    session.ask("What happened to thread 7?")

    system_text = session._messages[0].content
    assert "Thread 7 was skipped" in system_text
    assert "You are the agent that just finished" in system_text


def test_history_records_both_sides(repo):
    session = make_session(repo, FakeChatModel(tool_script=[AIMessage(content="Sure.")]))

    session.ask("Anything else to do?")

    history = session.history()
    assert [turn["role"] for turn in history] == ["user", "agent"]
    assert history[0]["text"] == "Anything else to do?"
    assert history[1]["text"] == "Sure."


def test_earlier_turns_stay_in_the_conversation(repo):
    session = make_session(repo, FakeChatModel(tool_script=[AIMessage(content="First.")]))

    session.ask("One.")
    session.ask("Two.")

    texts = [str(m.content) for m in session._messages]
    assert "One." in texts
    assert "Two." in texts


# -- changing things ------------------------------------------------------


def editing_llm(content=GUARDED, summary="Added the guard to divide()."):
    return FakeChatModel(
        tool_script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "src/calc.py", "content": content},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content=summary),
        ]
    )


def test_a_follow_up_change_is_applied_and_reported(repo):
    session = make_session(repo, editing_llm())

    turn = session.ask("Also raise on zero.")

    assert "raise ValueError" in (repo / "src" / "calc.py").read_text(encoding="utf-8")
    assert turn.files_changed == ["src/calc.py"]
    assert "+    if b == 0:" in turn.diff
    assert turn.error == ""


def test_read_only_mode_withholds_the_writing_tools(repo):
    original = (repo / "src" / "calc.py").read_text(encoding="utf-8")
    llm = editing_llm()
    session = make_session(repo, llm)

    turn = session.ask("Also raise on zero.", allow_edits=False)

    # The model asked for write_file; without it the call comes back as an
    # error to the model rather than as an edit to the file.
    assert (repo / "src" / "calc.py").read_text(encoding="utf-8") == original
    assert turn.files_changed == []


def test_read_only_mode_tells_the_model_it_cannot_edit(repo):
    session = make_session(repo, FakeChatModel(tool_script=[AIMessage(content="I would…")]))

    session.ask("Change it.", allow_edits=False)

    assert "read-only tools" in str(session._messages[1].content)


def test_each_turn_only_shows_its_own_diff(repo):
    session = make_session(repo, editing_llm())
    session.ask("Add the guard.")

    session.llm.tool_script = [AIMessage(content="Nothing more to do.")]
    second = session.ask("Anything else?")

    assert second.diff == ""
    assert second.files_changed == []


def test_a_second_edit_to_the_same_file_is_still_reported(repo):
    """Changed files come from the per-turn snapshot, not a cumulative set."""
    session = make_session(repo, editing_llm())
    assert session.ask("Add the guard.").files_changed == ["src/calc.py"]

    session.llm.tool_script = editing_llm(
        content=GUARDED.replace("b must not be zero", "b cannot be zero")
    ).tool_script
    second = session.ask("Reword the message.")

    assert second.files_changed == ["src/calc.py"]
    assert "cannot be zero" in second.diff


def test_a_model_failure_is_reported_not_raised(repo):
    class Exploding:
        def bind_tools(self, _tools, **_kwargs):
            return self

        def invoke(self, _messages, **_kwargs):
            raise RuntimeError("the API is down")

    turn = make_session(repo, Exploding()).ask("Hello?")

    assert "the API is down" in turn.error
    assert turn.text == ""


def test_an_edit_outside_the_repository_is_refused(repo):
    """The chat runs through the same sandbox the graph does."""
    llm = FakeChatModel(
        tool_script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "../escaped.txt", "content": "nope"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="I could not write there."),
        ]
    )
    turn = make_session(repo, llm).ask("Write to the parent directory.")

    assert not (repo.parent / "escaped.txt").exists()
    assert turn.files_changed == []


# -- the context the chat is given ----------------------------------------


def test_context_summarises_each_outcome():
    context = build_run_context(
        pr_slug="octo/demo#1",
        pr_title="Add arithmetic helpers",
        branch_detail="Switched from main to feature/calc.",
        report="1/2 thread(s) implemented",
        outcomes=[
            {
                "thread_id": 101,
                "path": "src/calc.py",
                "action": "implement",
                "reason": "Guard requested.",
                "summary": "Added a zero check.",
                "files_changed": ["src/calc.py"],
                "diff": "--- a/src/calc.py\n+++ b/src/calc.py\n+    raise ValueError",
                "validation": {"ok": True, "command": "pytest -q"},
            },
            {
                "thread_id": 102,
                "path": None,
                "action": "skip",
                "reason": "Praise.",
                "files_changed": [],
            },
        ],
    )

    assert "octo/demo#1" in context
    assert "Switched from main to feature/calc." in context
    assert "Thread 101 on src/calc.py — implemented" in context
    assert "Thread 102 on general comment — skip" in context
    assert "`pytest -q` passed" in context
    assert "raise ValueError" in context


def test_context_survives_a_run_that_did_nothing():
    context = build_run_context("octo/demo#1", "", "", "", [])
    assert "No review threads were acted on." in context


@pytest.mark.parametrize("field", ["summary", "reason", "error"])
def test_context_tolerates_missing_outcome_fields(field):
    outcome = {"thread_id": 1, "path": "a.py", "action": "implement", field: "x"}
    assert "Thread 1 on a.py" in build_run_context("s", "", "", "", [outcome])
