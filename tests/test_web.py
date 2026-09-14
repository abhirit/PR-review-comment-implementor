"""API-level tests for the web UI, driven through FastAPI's test client."""

from __future__ import annotations

import json
import time

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

from pr_agent.web.app import create_app  # noqa: E402
from pr_agent.web.schemas import RunRequest  # noqa: E402


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


# -- static + config -------------------------------------------------------


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "PR Review Comment Implementor" in response.text


def test_static_assets_are_served(client):
    for path in ("/static/app.js", "/static/styles.css"):
        assert client.get(path).status_code == 200


def test_config_reports_credential_presence(client, repo, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    body = client.get("/api/config", params={"repo_path": str(repo)}).json()

    assert body["llm_key_set"] is True
    assert body["github_token_set"] is False
    assert body["provider"] == "google"
    assert body["model"] == "gemini-3.8-flash"
    # A plain directory is not a git checkout.
    assert body["repo_is_git"] is False


def test_config_reports_git_state(client, repo):
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)

    body = client.get("/api/config", params={"repo_path": str(repo)}).json()

    assert body["repo_is_git"] is True
    assert body["repo_branch"] == "main"
    assert body["repo_dirty"] is True  # untracked files


# -- validation of input ---------------------------------------------------


def test_a_malformed_pr_reference_is_rejected(client):
    response = client.get("/api/comments", params={"pr": "not-a-pr"})
    assert response.status_code == 400
    assert "owner/repo" in response.json()["detail"]


def test_starting_a_run_with_a_bad_pr_reference_is_rejected(client):
    response = client.post("/api/runs", json={"pr": "garbage", "repo_path": "."})
    assert response.status_code == 409


def test_search_requires_a_query(client, repo):
    response = client.get("/api/search", params={"q": "", "repo_path": str(repo)})
    assert response.status_code == 422


def test_search_returns_ranked_hits(client, repo, monkeypatch):
    monkeypatch.setenv("PR_AGENT_EMBEDDINGS", "none")
    hits = client.get(
        "/api/search", params={"q": "divide", "repo_path": str(repo), "k": 3}
    ).json()
    assert hits
    assert hits[0]["path"] == "src/calc.py"
    assert "bm25" in hits[0]["source"]


def test_search_rejects_a_missing_repository(client):
    response = client.get("/api/search", params={"q": "x", "repo_path": "/nope/missing"})
    assert response.status_code == 400


# -- runs ------------------------------------------------------------------


def test_unknown_run_is_a_404(client):
    assert client.get("/api/runs/deadbeef").status_code == 404
    assert client.post("/api/runs/deadbeef/cancel").status_code == 404
    assert client.get("/api/runs/deadbeef/events").status_code == 404


def test_run_list_is_empty_at_first(client):
    assert client.get("/api/runs").json() == []


def _wait_for_end(client, run_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = client.get(f"/api/runs/{run_id}").json()
        if snapshot["status"] in {"done", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last status {snapshot['status']}")


def test_a_run_that_cannot_reach_github_fails_cleanly(client, repo, monkeypatch):
    """No GitHub token: the run must fail with a message, not hang or crash."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    created = client.post(
        "/api/runs",
        json={"pr": "octo/demo#1", "repo_path": str(repo), "dry_run": True},
    )
    assert created.status_code == 201
    run_id = created.json()["run_id"]

    snapshot = _wait_for_end(client, run_id)

    assert snapshot["status"] == "failed"
    assert "GitHub token" in snapshot["error"]
    assert any(e["type"] == "error" for e in snapshot["events"])
    assert snapshot["events"][-1]["type"] == "end"


def test_events_replay_the_whole_run_for_a_late_subscriber(client, repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    run_id = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo), "dry_run": True}
    ).json()["run_id"]
    _wait_for_end(client, run_id)

    # Subscribing after the run ended must still yield its full history.
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        types = []
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            types.append(event["type"])
            if event["type"] == "end":
                break

    assert "status" in types
    assert types[-1] == "end"


def test_two_runs_on_one_checkout_are_refused(client, repo, monkeypatch):
    """Concurrent runs would edit the same files, so the second must be rejected."""
    from pr_agent.web.runner import Run, RunManager

    manager = client.app.state.manager
    assert isinstance(manager, RunManager)

    # Simulate a live run holding the checkout.
    from pathlib import Path

    key = str(Path(repo).resolve())
    live = Run(id="live", request=RunRequest(pr="octo/demo#1", repo_path=str(repo)))
    live.status = "running"
    manager._runs["live"] = live
    manager._order.append("live")
    manager._active_repos[key] = "live"

    response = client.post(
        "/api/runs", json={"pr": "octo/demo#2", "repo_path": str(repo)}
    )

    assert response.status_code == 409
    assert "already in progress" in response.json()["detail"]


def test_a_finished_run_releases_its_checkout(client, repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    first = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo)}
    ).json()["run_id"]
    _wait_for_end(client, first)

    second = client.post("/api/runs", json={"pr": "octo/demo#2", "repo_path": str(repo)})

    assert second.status_code == 201


def test_cancelling_a_finished_run_says_so(client, repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    run_id = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo)}
    ).json()["run_id"]
    _wait_for_end(client, run_id)

    body = client.post(f"/api/runs/{run_id}/cancel").json()

    assert body["cancelled"] is False
    assert "already finished" in body["detail"]


def test_run_history_lists_finished_runs(client, repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    run_id = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo)}
    ).json()["run_id"]
    _wait_for_end(client, run_id)

    runs = client.get("/api/runs").json()

    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["pr"] == "octo/demo#1"


def test_search_reflects_files_changed_since_the_last_search(client, repo, monkeypatch):
    """The index is refreshed per request, so edits show up without a restart."""
    monkeypatch.setenv("PR_AGENT_EMBEDDINGS", "none")
    params = {"q": "quantum_flux_capacitor", "repo_path": str(repo)}

    assert client.get("/api/search", params=params).json() == []

    (repo / "src" / "new_module.py").write_text(
        "def quantum_flux_capacitor():\n    return 42\n", encoding="utf-8"
    )
    hits = client.get("/api/search", params=params).json()

    assert [h["path"] for h in hits] == ["src/new_module.py"]
# -- branch options --------------------------------------------------------


def test_the_checkout_is_on_by_default(client):
    assert RunRequest(pr="octo/demo#1").checkout_branch is True
    assert RunRequest(pr="octo/demo#1").restore_branch is False


def test_config_reports_a_stash(client, repo):
    import subprocess

    for args in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@e.com"],
        ["git", "config", "user.name", "T"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(args, cwd=repo, check=True)

    assert client.get("/api/config", params={"repo_path": str(repo)}).json()[
        "repo_has_stash"
    ] is False

    (repo / "README.md").write_text("# changed\n", encoding="utf-8")
    subprocess.run(["git", "stash", "push", "-qm", "wip"], cwd=repo, check=True)

    assert client.get("/api/config", params={"repo_path": str(repo)}).json()[
        "repo_has_stash"
    ] is True


# -- chat ------------------------------------------------------------------


def test_chat_on_an_unknown_run_is_a_404(client):
    assert client.get("/api/runs/deadbeef/chat").status_code == 404
    assert client.post("/api/runs/deadbeef/chat", json={"message": "hi"}).status_code == 404


def test_chat_starts_empty(client, repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    run_id = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo)}
    ).json()["run_id"]
    _wait_for_end(client, run_id)

    body = client.get(f"/api/runs/{run_id}/chat").json()

    assert body["run_id"] == run_id
    assert body["messages"] == []
    assert body["available"] is True


def test_chat_is_closed_while_the_run_is_going(client, repo):
    """The run owns the checkout until it finishes; two writers would collide."""
    from pathlib import Path as _Path

    from pr_agent.web.runner import Run

    manager = client.app.state.manager
    live = Run(id="live", request=RunRequest(pr="octo/demo#1", repo_path=str(repo)))
    live.status = "running"
    manager._runs["live"] = live
    manager._order.append("live")
    manager._active_repos[str(_Path(repo).resolve())] = "live"

    history = client.get("/api/runs/live/chat").json()
    assert history["available"] is False
    assert "still going" in history["detail"]

    refused = client.post("/api/runs/live/chat", json={"message": "hello"})
    assert refused.status_code == 409
    assert "still going" in refused.json()["detail"]


def test_chat_names_the_branch_a_change_would_land_on(client, repo, monkeypatch):
    """After a restore the checkout is back on your branch, not the PR's."""
    import subprocess

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    for args in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@e.com"],
        ["git", "config", "user.name", "T"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(args, cwd=repo, check=True)

    run_id = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo)}
    ).json()["run_id"]
    _wait_for_end(client, run_id)

    assert client.get(f"/api/runs/{run_id}/chat").json()["branch"] == "main"


def test_an_empty_chat_message_is_rejected(client, repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    run_id = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo)}
    ).json()["run_id"]
    _wait_for_end(client, run_id)

    assert client.post(f"/api/runs/{run_id}/chat", json={"message": ""}).status_code == 422


def test_a_message_is_answered_and_kept_in_the_history(client, repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("PR_AGENT_EMBEDDINGS", "none")
    run_id = client.post(
        "/api/runs", json={"pr": "octo/demo#1", "repo_path": str(repo)}
    ).json()["run_id"]
    _wait_for_end(client, run_id)

    # Swap in a scripted model rather than calling a real provider.
    from fakes import FakeChatModel
    from langchain_core.messages import AIMessage

    from pr_agent.web import runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "build_chat_model",
        lambda _settings: FakeChatModel(
            tool_script=[AIMessage(content="Nothing was changed: the run could not start.")]
        ),
    )

    reply = client.post(
        f"/api/runs/{run_id}/chat", json={"message": "What did you change?"}
    ).json()

    assert reply["role"] == "agent"
    assert "Nothing was changed" in reply["text"]
    assert reply["files_changed"] == []

    history = client.get(f"/api/runs/{run_id}/chat").json()["messages"]
    assert [m["role"] for m in history] == ["user", "agent"]
    assert history[0]["text"] == "What did you change?"
