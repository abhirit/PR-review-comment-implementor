import subprocess

import pytest

from pr_agent import git_ops
from pr_agent.config import Settings
from pr_agent.validation import detect_validation_commands, run_validation

# -- validation -----------------------------------------------------------


def test_no_commands_is_a_pass(tmp_path):
    result = run_validation(tmp_path, [])
    assert result.ok
    assert "no validation commands" in result.output


def test_all_commands_must_pass(tmp_path):
    assert run_validation(tmp_path, ["exit 0", "exit 0"]).ok


def test_the_first_failure_stops_the_run(tmp_path):
    result = run_validation(tmp_path, ["echo one", "exit 3", "echo never"])
    assert not result.ok
    assert result.exit_code == 3
    assert result.command == "exit 3"


def test_failure_output_captures_stdout_and_stderr(tmp_path):
    result = run_validation(tmp_path, ["echo to-out; echo to-err 1>&2; exit 1"])
    assert "to-out" in result.output
    assert "to-err" in result.output


def test_a_hanging_command_times_out(tmp_path):
    result = run_validation(tmp_path, ["sleep 30"], timeout=1)
    assert not result.ok
    assert result.exit_code == 124
    assert "timed out" in result.output


def test_commands_run_inside_the_repository(tmp_path):
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    assert run_validation(tmp_path, ["test -f marker.txt"]).ok


def test_detect_finds_go_build(tmp_path):
    (tmp_path / "go.mod").write_text("module demo\n", encoding="utf-8")
    assert "go build ./..." in detect_validation_commands(tmp_path)


def test_detect_reads_npm_scripts(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"scripts": {"lint": "eslint .", "test": "jest"}}', encoding="utf-8"
    )
    commands = detect_validation_commands(tmp_path)
    assert any("lint" in c for c in commands)
    assert any("test" in c for c in commands)


def test_detect_survives_a_malformed_package_json(tmp_path):
    (tmp_path / "package.json").write_text("{not json", encoding="utf-8")
    assert detect_validation_commands(tmp_path) == []


def test_settings_parse_validate_commands_from_the_environment(monkeypatch):
    monkeypatch.setenv("PR_AGENT_VALIDATE", "ruff check .;;pytest -q")
    assert Settings().validate_commands == ["ruff check .", "pytest -q"]


# -- git ------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def test_current_branch(git_repo):
    assert git_ops.current_branch(git_repo) == "main"


def test_clean_tree_reports_no_changes(git_repo):
    assert not git_ops.is_dirty(git_repo)
    assert git_ops.changed_files(git_repo) == []


def test_changed_files_lists_modified_and_untracked(git_repo):
    (git_repo / "a.txt").write_text("two\n", encoding="utf-8")
    (git_repo / "b.txt").write_text("new\n", encoding="utf-8")
    assert sorted(git_ops.changed_files(git_repo)) == ["a.txt", "b.txt"]
    assert git_ops.is_dirty(git_repo)


def test_commit_returns_a_sha_and_cleans_the_tree(git_repo):
    (git_repo / "a.txt").write_text("two\n", encoding="utf-8")
    sha = git_ops.commit(git_repo, "update a")
    assert sha and len(sha) == 40
    assert not git_ops.is_dirty(git_repo)


def test_commit_with_nothing_staged_returns_none(git_repo):
    assert git_ops.commit(git_repo, "empty") is None


def test_commit_can_stage_specific_paths(git_repo):
    (git_repo / "a.txt").write_text("two\n", encoding="utf-8")
    (git_repo / "b.txt").write_text("new\n", encoding="utf-8")
    git_ops.commit(git_repo, "only a", paths=["a.txt"])
    assert git_ops.changed_files(git_repo) == ["b.txt"]


def test_a_failing_git_command_raises(tmp_path):
    with pytest.raises(git_ops.GitError):
        git_ops.current_branch(tmp_path / "not-a-repo")


@pytest.mark.parametrize(
    "message,transient",
    [
        ("fatal: unable to access: Could not resolve host: github.com", True),
        ("error: RPC failed; curl 92", True),
        ("fatal: connection reset by peer", True),
        ("! [rejected] main -> main (non-fast-forward)", False),
        ("Permission denied (publickey)", False),
    ],
)
def test_transient_push_errors_are_recognised(message, transient):
    assert git_ops._is_transient(message) is transient


def test_is_git_repo_recognises_a_checkout(git_repo):
    assert git_ops.is_git_repo(git_repo)


def test_is_git_repo_rejects_a_plain_directory(tmp_path_factory):
    # A fresh root, so it cannot sit inside another test's repository.
    plain = tmp_path_factory.mktemp("plain")
    assert not git_ops.is_git_repo(plain)


def test_current_branch_works_before_the_first_commit(tmp_path):
    # rev-parse HEAD fails on an unborn branch; the symbolic ref still resolves.
    subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=tmp_path, check=True)
    assert git_ops.current_branch(tmp_path) == "trunk"
