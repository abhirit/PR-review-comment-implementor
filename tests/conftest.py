"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path shim
    sys.path.insert(0, str(SRC))


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
