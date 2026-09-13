import pytest

from pr_agent.workspace import Workspace, WorkspaceError


@pytest.fixture
def ws(repo):
    return Workspace(root=repo)


def test_resolve_rejects_paths_outside_the_repo(ws):
    for bad in ["../secrets.txt", "src/../../etc/passwd", "/etc/passwd"]:
        with pytest.raises(WorkspaceError):
            ws.resolve(bad)


def test_resolve_allows_paths_inside_the_repo(ws, repo):
    assert ws.resolve("src/calc.py") == (repo / "src" / "calc.py").resolve()
    assert ws.resolve("src/../README.md") == (repo / "README.md").resolve()


def test_read_lines_is_one_indexed_and_inclusive(ws):
    out = ws.read_lines("src/calc.py", 4, 5)
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("4\t")
    assert "def divide" in lines[0]


def test_replace_requires_a_unique_match(ws):
    ws.write("dup.py", "x = 1\nx = 1\n")
    with pytest.raises(WorkspaceError, match="appears 2 times"):
        ws.replace("dup.py", "x = 1", "x = 2")
    ws.replace("dup.py", "x = 1", "x = 2", replace_all=True)
    assert ws.read("dup.py") == "x = 2\nx = 2\n"


def test_replace_reports_a_missing_match(ws):
    with pytest.raises(WorkspaceError, match="not found"):
        ws.replace("src/calc.py", "def multiply", "def divide")


def test_touched_records_every_write(ws):
    ws.replace("src/calc.py", "def add(a, b):", "def add(a, b, c=0):")
    assert "src/calc.py" in ws.touched


def test_rollback_restores_edits_and_removes_new_files(ws):
    original = ws.read("src/calc.py")
    ws.begin()
    ws.replace("src/calc.py", "return a / b", "return a // b")
    ws.write("src/new_file.py", "print('hi')\n")
    assert ws.read("src/calc.py") != original
    assert ws.exists("src/new_file.py")

    restored = ws.rollback()

    assert sorted(restored) == ["src/calc.py", "src/new_file.py"]
    assert ws.read("src/calc.py") == original
    assert not ws.exists("src/new_file.py")
    assert "src/calc.py" not in ws.touched


def test_commit_changes_keeps_edits(ws):
    ws.begin()
    ws.replace("src/calc.py", "return a + b", "return a + b  # noqa")
    changed = ws.commit_changes()
    assert changed == ["src/calc.py"]
    assert "# noqa" in ws.read("src/calc.py")
    # A later rollback must not undo already-committed work.
    assert ws.rollback() == []
    assert "# noqa" in ws.read("src/calc.py")


def test_iter_files_skips_excluded_directories(ws, repo):
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    (repo / ".git").mkdir(exist_ok=True)
    (repo / ".git" / "config").write_text("x", encoding="utf-8")

    files = ws.iter_files()

    assert "src/calc.py" in files
    assert not any(f.startswith("node_modules") for f in files)
    assert not any(f.startswith(".git") for f in files)


def test_iter_files_skips_binary_and_oversized_files(repo):
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
    (repo / "big.txt").write_text("x" * 5000, encoding="utf-8")
    ws = Workspace(root=repo, max_file_bytes=1000)
    files = ws.iter_files()
    assert "blob.bin" not in files
    assert "big.txt" not in files


def test_grep_finds_matches_case_insensitively(ws):
    hits = ws.grep("DEF DIVIDE")
    assert hits and hits[0][0] == "src/calc.py"
