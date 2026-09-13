from pr_agent.github_client import _next_link, _parse_comment, build_threads
from pr_agent.models import ReviewComment


def _c(cid, reply_to=None, body="x"):
    return ReviewComment(id=cid, body=body, author="a", in_reply_to_id=reply_to)


def test_build_threads_groups_replies_under_root():
    threads = build_threads([_c(1), _c(2, 1), _c(3, 1), _c(10)])
    assert [(t.id, [r.id for r in t.replies]) for t in threads] == [(1, [2, 3]), (10, [])]


def test_build_threads_handles_replies_arriving_before_their_root():
    threads = build_threads([_c(5, 1), _c(1)])
    assert [(t.id, [r.id for r in t.replies]) for t in threads] == [(1, [5])]


def test_build_threads_follows_reply_chains_to_the_root():
    # A reply that points at another reply still belongs to the same thread.
    threads = build_threads([_c(1), _c(2, 1), _c(3, 2)])
    assert [(t.id, [r.id for r in t.replies]) for t in threads] == [(1, [2, 3])]


def test_build_threads_promotes_orphan_replies():
    # The root is missing (deleted), so the reply becomes its own thread
    # rather than being silently dropped.
    threads = build_threads([_c(9, 4)])
    assert [t.id for t in threads] == [9]


def test_build_threads_ignores_cycles():
    threads = build_threads([_c(1, 2), _c(2, 1)])
    assert len(threads) == 2


def test_parse_comment_falls_back_to_original_line():
    comment = _parse_comment(
        {
            "id": 1,
            "body": "b",
            "user": {"login": "alice"},
            "path": "a.py",
            "line": None,
            "original_line": 42,
            "diff_hunk": "@@ -1 +1 @@",
        }
    )
    assert comment.line == 42
    assert comment.author == "alice"


def test_next_link_extracts_the_next_page():
    header = '<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=9>; rel="last"'
    assert _next_link(header) == "https://api.github.com/x?page=2"
    assert _next_link('<https://api.github.com/x?page=9>; rel="last"') is None
    assert _next_link("") is None
