"""Running the project's own checks after an edit."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from .models import ValidationResult, truncate

log = logging.getLogger(__name__)


def run_validation(
    repo_path: Path, commands: list[str], timeout: int = 900
) -> ValidationResult:
    """Run each configured command in order; stop at the first failure.

    With no commands configured this is a no-op that reports success, so an
    unconfigured repository still completes a run.
    """
    if not commands:
        return ValidationResult(ok=True, command="", output="no validation commands configured")

    for command in commands:
        log.info("validating: %s", command)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                ok=False,
                command=command,
                exit_code=124,
                output=f"Command timed out after {timeout}s.",
            )
        except OSError as exc:
            return ValidationResult(
                ok=False, command=command, exit_code=127, output=f"Could not run command: {exc}"
            )

        output = f"{completed.stdout}\n{completed.stderr}".strip()
        if completed.returncode != 0:
            return ValidationResult(
                ok=False,
                command=command,
                exit_code=completed.returncode,
                output=truncate(output, 12000),
            )

    return ValidationResult(ok=True, command=commands[-1], output="all checks passed")


def detect_validation_commands(repo_path: Path) -> list[str]:
    """Guess sensible checks for a repository that has not configured any.

    Deliberately conservative: only commands whose config file is present, and
    only fast ones, because these run after every implemented comment.
    """
    commands: list[str] = []
    if (repo_path / "pyproject.toml").is_file() or (repo_path / "setup.cfg").is_file():
        if _has_tool(repo_path, "ruff"):
            commands.append("ruff check .")
        if (repo_path / "tests").is_dir() or (repo_path / "test").is_dir():
            commands.append("python -m pytest -q")
    package_json = repo_path / "package.json"
    if package_json.is_file():
        import json

        try:
            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if "lint" in scripts:
            commands.append("npm run lint --if-present")
        if "test" in scripts:
            commands.append("npm test --if-present")
    if (repo_path / "go.mod").is_file():
        commands.append("go build ./...")
    return commands


def _has_tool(repo_path: Path, name: str) -> bool:
    try:
        result = subprocess.run(
            shlex.split(f"{name} --version"),
            cwd=str(repo_path),
            capture_output=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
