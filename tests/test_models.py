import pytest

from pr_agent.models import PRRef, ReviewComment, ReviewThread, truncate


@pytest.mark.parametrize(
    "value,expected",
    [
        ("octo/hello#7", ("octo", "hello", 7)),
        ("https://github.com/octo/hello/pull/7", ("octo", "hello", 7)),
        ("https://github.com/octo/hello/pull/7#discussion_r123", ("octo", "hello", 7)),
        ("https://github.com/octo/hello/pull/7/files", ("octo", "hello", 7)),
        ("  octo/hello#7  ", ("octo", "hello", 7)),
    ],
)
def test_prref_parse(value, expected):
    ref = PRRef.parse(value)
    assert (ref.owner, ref.repo, ref.number) == expected


@pytest.mark.parametrize("value", ["nonsense", "octo/hello", "#12", "https://github.com/octo/hello"])
def test_prref_parse_rejects_garbage(value):
    with pytest.raises(ValueError):
        PRRef.parse(value)


def test_thread_transcript_orders_root_first():
    thread = ReviewThread(
        root=ReviewComment(id=1, body="Use a guard clause.", author="alice"),
        replies=[ReviewComment(id=2, body="Agreed.", author="bob", in_reply_to_id=1)],
    )
    transcript = thread.transcript()
    assert transcript.index("@alice") < transcript.index("@bob")
    assert "Use a guard clause." in transcript


def test_truncate_keeps_both_ends():
    text = "A" * 100 + "B" * 100
    out = truncate(text, 60)
    assert out.startswith("A")
    assert out.endswith("B")
    assert "truncated" in out
    assert truncate("short", 100) == "short"
