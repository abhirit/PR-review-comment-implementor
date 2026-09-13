"""Minimal GitHub REST/GraphQL client for the bits of the API the agent needs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import httpx

from .models import PRRef, PullRequest, ReviewComment, ReviewThread

log = logging.getLogger(__name__)

_API_VERSION = "2022-11-28"


class GitHubError(RuntimeError):
    """Raised when the GitHub API returns an error we cannot recover from."""


class GitHubClient:
    """Thin wrapper over the GitHub API.

    Only the endpoints the agent actually uses are implemented; everything is
    synchronous because the graph runs one thread at a time.
    """

    def __init__(
        self,
        token: str | None,
        api_url: str = "https://api.github.com",
        timeout: float = 30.0,
    ) -> None:
        if not token:
            raise GitHubError(
                "A GitHub token is required. Set GITHUB_TOKEN to a token with "
                "'pull_request' read access (and write access if you want replies posted)."
            )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _API_VERSION,
            "Authorization": f"Bearer {token}",
            "User-Agent": "pr-review-implementor",
        }
        self._api_url = api_url.rstrip("/")
        self._client = httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = path if path.startswith("http") else f"{self._api_url}{path}"
        response = self._client.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise GitHubError(
                f"{method} {url} failed with {response.status_code}: {response.text[:500]}"
            )
        return response

    def _paginate(self, path: str, **params: Any) -> Iterator[dict[str, Any]]:
        """Follow GitHub's Link-header pagination and yield every item."""
        url: str | None = f"{self._api_url}{path}"
        query: dict[str, Any] | None = {"per_page": 100, **params}
        while url:
            response = self._request("GET", url, params=query)
            payload = response.json()
            if not isinstance(payload, list):  # pragma: no cover - defensive
                raise GitHubError(f"Expected a list from {url}, got {type(payload).__name__}")
            yield from payload
            url = _next_link(response.headers.get("link", ""))
            query = None  # the next link already carries the query string

    # -- pull requests ----------------------------------------------------

    def get_pull_request(self, ref: PRRef) -> PullRequest:
        data = self._request(
            "GET", f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}"
        ).json()
        return PullRequest(
            ref=ref,
            title=data.get("title") or "",
            body=data.get("body") or "",
            head_ref=data["head"]["ref"],
            head_sha=data["head"]["sha"],
            base_ref=data["base"]["ref"],
            state=data.get("state", "open"),
            html_url=data.get("html_url", ""),
        )

    def get_pull_request_diff(self, ref: PRRef) -> str:
        response = self._request(
            "GET",
            f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        return response.text

    def list_review_comments(self, ref: PRRef) -> list[ReviewComment]:
        """All inline review comments on the PR, oldest first."""
        comments = [
            _parse_comment(item)
            for item in self._paginate(
                f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/comments"
            )
        ]
        comments.sort(key=lambda c: c.id)
        return comments

    def list_review_bodies(self, ref: PRRef) -> list[ReviewComment]:
        """Review summary bodies (the text submitted with an approval/request-changes)."""
        out: list[ReviewComment] = []
        for item in self._paginate(f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/reviews"):
            body = (item.get("body") or "").strip()
            if not body:
                continue
            out.append(
                ReviewComment(
                    id=int(item["id"]),
                    body=body,
                    author=(item.get("user") or {}).get("login", "unknown"),
                    html_url=item.get("html_url", ""),
                    created_at=item.get("submitted_at", "") or "",
                )
            )
        return out

    # -- writes -----------------------------------------------------------

    def reply_to_review_comment(self, ref: PRRef, comment_id: int, body: str) -> dict[str, Any]:
        """Post a threaded reply under an existing review comment."""
        return self._request(
            "POST",
            f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/comments/{comment_id}/replies",
            json={"body": body},
        ).json()

    def create_issue_comment(self, ref: PRRef, body: str) -> dict[str, Any]:
        """Post a comment on the PR conversation."""
        return self._request(
            "POST",
            f"/repos/{ref.owner}/{ref.repo}/issues/{ref.number}/comments",
            json={"body": body},
        ).json()

    def resolve_review_thread(self, ref: PRRef, comment_id: int) -> bool:
        """Resolve the review thread containing ``comment_id`` (GraphQL only)."""
        thread_id = self._find_thread_node_id(ref, comment_id)
        if not thread_id:
            log.warning("Could not find a GraphQL thread node for comment %s", comment_id)
            return False
        mutation = """
        mutation($threadId: ID!) {
          resolveReviewThread(input: {threadId: $threadId}) {
            thread { isResolved }
          }
        }
        """
        data = self._graphql(mutation, {"threadId": thread_id})
        return bool(data["resolveReviewThread"]["thread"]["isResolved"])

    def _find_thread_node_id(self, ref: PRRef, comment_id: int) -> str | None:
        query = """
        query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              reviewThreads(first: 100, after: $cursor) {
                pageInfo { hasNextPage endCursor }
                nodes {
                  id
                  isResolved
                  comments(first: 100) { nodes { databaseId } }
                }
              }
            }
          }
        }
        """
        cursor: str | None = None
        while True:
            data = self._graphql(
                query,
                {
                    "owner": ref.owner,
                    "repo": ref.repo,
                    "number": ref.number,
                    "cursor": cursor,
                },
            )
            threads = data["repository"]["pullRequest"]["reviewThreads"]
            for node in threads["nodes"]:
                ids = {c["databaseId"] for c in node["comments"]["nodes"]}
                if comment_id in ids:
                    return str(node["id"])
            if not threads["pageInfo"]["hasNextPage"]:
                return None
            cursor = threads["pageInfo"]["endCursor"]

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"{self._api_url}/graphql",
            json={"query": query, "variables": variables},
        )
        payload = response.json()
        if payload.get("errors"):
            raise GitHubError(f"GraphQL error: {payload['errors']}")
        return payload["data"]


# -- pure helpers ---------------------------------------------------------


def _parse_comment(item: dict[str, Any]) -> ReviewComment:
    return ReviewComment(
        id=int(item["id"]),
        body=item.get("body") or "",
        author=(item.get("user") or {}).get("login", "unknown"),
        path=item.get("path"),
        line=item.get("line") or item.get("original_line"),
        start_line=item.get("start_line") or item.get("original_start_line"),
        side=item.get("side"),
        diff_hunk=item.get("diff_hunk") or "",
        in_reply_to_id=item.get("in_reply_to_id"),
        html_url=item.get("html_url", ""),
        created_at=item.get("created_at", "") or "",
    )


def _next_link(link_header: str) -> str | None:
    """Extract the ``rel="next"`` URL from a Link header."""
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        url = section[0].strip().strip("<>")
        for attr in section[1:]:
            if attr.strip() == 'rel="next"':
                return url
    return None


def build_threads(comments: list[ReviewComment]) -> list[ReviewThread]:
    """Group flat review comments into threads keyed by their root comment.

    GitHub sets ``in_reply_to_id`` on every reply to the id of the thread's
    first comment, but a reply can still arrive before its root in an unsorted
    list, so roots are collected in a first pass.
    """
    by_id = {c.id: c for c in comments}
    roots: dict[int, ReviewThread] = {}
    orphans: list[ReviewComment] = []

    for comment in comments:
        if comment.in_reply_to_id is None:
            roots[comment.id] = ReviewThread(root=comment)

    for comment in comments:
        if comment.in_reply_to_id is None:
            continue
        root_id = comment.in_reply_to_id
        # Follow the chain in case a reply points at another reply.
        seen: set[int] = set()
        while root_id in by_id and by_id[root_id].in_reply_to_id is not None:
            if root_id in seen:  # pragma: no cover - cycle guard
                break
            seen.add(root_id)
            root_id = by_id[root_id].in_reply_to_id  # type: ignore[assignment]
        if root_id in roots:
            roots[root_id].replies.append(comment)
        else:
            orphans.append(comment)

    # A reply whose root is missing (deleted, or outside the page) still deserves
    # to be worked on, so promote it to its own thread.
    for orphan in orphans:
        roots[orphan.id] = ReviewThread(root=orphan)

    threads = list(roots.values())
    for thread in threads:
        thread.replies.sort(key=lambda c: c.id)
    threads.sort(key=lambda t: t.root.id)
    return threads
