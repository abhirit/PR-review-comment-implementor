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
# -- branch switching -----------------------------------------------------


def _branch(repo, name):
    subprocess.run(["git", "branch", name], cwd=repo, check=True)


def test_prepare_branch_switches_and_stashes(git_repo):
    _branch(git_repo, "feature")
    (git_repo / "a.txt").write_text("work in progress\n", encoding="utf-8")

    switch = git_ops.prepare_branch(git_repo, "feature")

    assert switch.switched
    assert switch.previous_branch == "main"
    assert switch.stash_sha
    assert git_ops.current_branch(git_repo) == "feature"
    # The work in progress went with the stash, not onto the PR branch.
    assert not git_ops.is_dirty(git_repo)
    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "one\n"


def test_restore_branch_puts_everything_back(git_repo):
    _branch(git_repo, "feature")
    (git_repo / "a.txt").write_text("work in progress\n", encoding="utf-8")
    switch = git_ops.prepare_branch(git_repo, "feature")

    restored = git_ops.restore_branch(git_repo, switch)

    assert restored.restored
    assert git_ops.current_branch(git_repo) == "main"
    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "work in progress\n"
    assert git_ops.stash_count(git_repo) == 0


def test_untracked_files_are_stashed_and_restored_too(git_repo):
    _branch(git_repo, "feature")
    (git_repo / "notes.txt").write_text("scratch\n", encoding="utf-8")

    switch = git_ops.prepare_branch(git_repo, "feature")
    assert not (git_repo / "notes.txt").exists()

    git_ops.restore_branch(git_repo, switch)
    assert (git_repo / "notes.txt").read_text(encoding="utf-8") == "scratch\n"


def test_a_clean_tree_is_switched_without_a_stash(git_repo):
    _branch(git_repo, "feature")

    switch = git_ops.prepare_branch(git_repo, "feature")

    assert switch.switched
    assert switch.stash_sha is None
    assert git_ops.stash_count(git_repo) == 0


def test_already_on_the_branch_is_a_no_op(git_repo):
    (git_repo / "a.txt").write_text("work in progress\n", encoding="utf-8")

    switch = git_ops.prepare_branch(git_repo, "main")

    assert not switch.switched
    assert switch.stash_sha is None
    assert "Already on main" in switch.detail
    # Nothing was stashed, so the user's work is untouched.
    assert git_ops.is_dirty(git_repo)


def test_restoring_a_no_op_switch_does_nothing(git_repo):
    switch = git_ops.prepare_branch(git_repo, "main")
    assert git_ops.restore_branch(git_repo, switch) is switch


def test_a_plain_directory_is_left_alone(tmp_path_factory):
    plain = tmp_path_factory.mktemp("plain")
    switch = git_ops.prepare_branch(plain, "feature")
    assert not switch.switched
    assert "Not a git checkout" in switch.detail


def test_an_unreachable_branch_raises_and_restores_the_stash(git_repo):
    (git_repo / "a.txt").write_text("work in progress\n", encoding="utf-8")

    with pytest.raises(git_ops.GitError, match="no 'origin' remote"):
        git_ops.prepare_branch(git_repo, "never-existed")

    # The failure must not leave the user's work parked in the stash.
    assert git_ops.stash_count(git_repo) == 0
    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "work in progress\n"


def test_the_right_stash_entry_is_popped(git_repo):
    """Another stash pushed on top must not be mistaken for ours."""
    _branch(git_repo, "feature")
    (git_repo / "a.txt").write_text("ours\n", encoding="utf-8")
    switch = git_ops.prepare_branch(git_repo, "feature")

    # Something else stashes while we are on the PR branch.
    (git_repo / "a.txt").write_text("theirs\n", encoding="utf-8")
    subprocess.run(["git", "stash", "push", "-qm", "someone else"], cwd=git_repo, check=True)

    git_ops.restore_branch(git_repo, switch)

    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "ours\n"
    assert git_ops.stash_count(git_repo) == 1  # theirs, still parked


def test_restore_reports_a_stash_that_has_gone(git_repo):
    _branch(git_repo, "feature")
    (git_repo / "a.txt").write_text("work in progress\n", encoding="utf-8")
    switch = git_ops.prepare_branch(git_repo, "feature")

    subprocess.run(["git", "stash", "drop", "-q"], cwd=git_repo, check=True)
    restored = git_ops.restore_branch(git_repo, switch)

    assert restored.restored
    assert "already gone" in restored.restore_detail


def test_prepare_branch_creates_a_local_branch_from_the_remote(tmp_path):
    origin = tmp_path / "origin"
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=origin, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=origin, check=True)
    (origin / "a.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=origin, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=origin, check=True)
    (origin / "b.txt").write_text("two\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(["git", "commit", "-qm", "feature work"], cwd=origin, check=True)
    subprocess.run(["git", "checkout", "-q", "main"], cwd=origin, check=True)

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--single-branch", "-b", "main", str(origin), str(clone)],
        check=True,
    )

    switch = git_ops.prepare_branch(clone, "feature")

    assert switch.switched and switch.created
    assert git_ops.current_branch(clone) == "feature"
    assert (clone / "b.txt").exists()
