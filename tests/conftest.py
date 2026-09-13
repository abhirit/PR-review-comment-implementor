"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path shim
    sys.path.insert(0, str(SRC))


# Every variable the agent reads from the environment, so a developer's own
# shell or .env cannot change what the suite sees.
_SETTINGS_ENV = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_API_URL",
    "VOYAGE_API_KEY",
    "PR_AGENT_PROVIDER",
    "PR_AGENT_MODEL",
    "PR_AGENT_EMBEDDINGS",
    "PR_AGENT_REPO_PATH",
    "PR_AGENT_VALIDATE",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run against default settings, not the checkout's .env."""
    from pr_agent import config

    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny repository checkout to run the agent against."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(
        '''"""Arithmetic helpers."""


def divide(a, b):
    return a / b


def add(a, b):
    return a + b
''',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return tmp_path
