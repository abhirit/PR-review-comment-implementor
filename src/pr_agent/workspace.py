"""Sandboxed file access for the repository under work.

Every read and write the agent performs goes through :class:`Workspace`, which
keeps paths inside the repository root and records which files were touched.
"""

from __future__ import annotations

import difflib
import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git/*",
    "*/.git/*",
    ".pr_agent/*",
    "*/node_modules/*",
    "node_modules/*",
    "*/.venv/*",
    ".venv/*",
    "venv/*",
    "*/__pycache__/*",
    "__pycache__/*",
    "*/dist/*",
    "dist/*",
    "*/build/*",
    "build/*",
    "*.min.js",
    "*.lock",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.zip",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.class",
    "*.pyc",
)


class WorkspaceError(RuntimeError):
    """Raised for a path outside the repo, or an edit that cannot be applied."""


@dataclass
class Workspace:
    """A repository checkout the agent may read and edit."""

    root: Path
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES
    max_file_bytes: int = 400_000
    touched: set[str] = field(default_factory=set)
    _snapshots: dict[str, str | None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"Repository path does not exist: {self.root}")

    # -- path handling ----------------------------------------------------

    def resolve(self, rel_path: str) -> Path:
        """Resolve a repo-relative path, refusing anything outside the root."""
        if not rel_path or rel_path.strip() != rel_path:
            rel_path = rel_path.strip()
        candidate = Path(rel_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(
                f"Refusing to access '{rel_path}': outside the repository root {self.root}"
            ) from exc
        return resolved

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def is_excluded(self, rel_path: str) -> bool:
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in self.excludes)

    # -- reads ------------------------------------------------------------

    def exists(self, rel_path: str) -> bool:
        return self.resolve(rel_path).is_file()

    def read(self, rel_path: str) -> str:
        path = self.resolve(rel_path)
        if not path.is_file():
            raise WorkspaceError(f"No such file: {rel_path}")
        return path.read_text(encoding="utf-8", errors="replace")

    def read_lines(self, rel_path: str, start: int = 1, end: int | None = None) -> str:
        """Read a 1-indexed, inclusive line range, prefixed with line numbers."""
        lines = self.read(rel_path).splitlines()
        start = max(1, start)
        end = len(lines) if end is None else min(end, len(lines))
        width = len(str(end))
        return "\n".join(
            f"{idx:>{width}}\t{lines[idx - 1]}" for idx in range(start, end + 1)
        )

    def iter_files(self) -> list[str]:
        """Every indexable, repo-relative file path, sorted."""
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune excluded directories in place so os.walk does not descend.
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            dirnames[:] = [
                d
                for d in dirnames
                if not self.is_excluded(f"{rel_dir}/{d}/x" if rel_dir != "." else f"{d}/x")
            ]
            for name in filenames:
                rel = (
                    name if rel_dir == "." else f"{rel_dir}/{name}"
                )
                if self.is_excluded(rel):
                    continue
                full = self.root / rel
                try:
                    if full.stat().st_size > self.max_file_bytes:
                        continue
                except OSError:  # pragma: no cover - race with external edits
                    continue
                if _looks_binary(full):
                    continue
                out.append(rel)
        out.sort()
        return out

    def grep(self, pattern: str, limit: int = 40) -> list[tuple[str, int, str]]:
        """Plain substring search across the workspace. Returns (path, line, text)."""
        needle = pattern.lower()
        hits: list[tuple[str, int, str]] = []
        for rel in self.iter_files():
            try:
                text = self.read(rel)
            except WorkspaceError:  # pragma: no cover - race with external edits
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    hits.append((rel, lineno, line.strip()[:200]))
                    if len(hits) >= limit:
                        return hits
        return hits

    # -- writes -----------------------------------------------------------

    def write(self, rel_path: str, content: str) -> str:
        path = self.resolve(rel_path)
        self._snapshot(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rel = self.relative(path)
        self.touched.add(rel)
        log.info("wrote %s (%d bytes)", rel, len(content))
        return rel

    def replace(self, rel_path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Exact string replacement, mirroring the semantics of an editor tool."""
        content = self.read(rel_path)
        count = content.count(old)
        if count == 0:
            raise WorkspaceError(
                f"The text to replace was not found in {rel_path}. "
                "Read the file again and copy the exact text, including indentation."
            )
        if count > 1 and not replace_all:
            raise WorkspaceError(
                f"The text to replace appears {count} times in {rel_path}. "
                "Include more surrounding context to make it unique, or set replace_all."
            )
        updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        self.write(rel_path, updated)
        return f"Replaced {count if replace_all else 1} occurrence(s) in {rel_path}."

    # -- transactions -----------------------------------------------------

    def begin(self) -> None:
        """Start recording original file contents so edits can be undone.

        Used to isolate one review comment's changes: if the agent cannot get
        them to pass validation, they are rolled back rather than committed on
        top of the changes that did pass.
        """
        self._snapshots = {}

    def _snapshot(self, path: Path) -> None:
        if self._snapshots is None:
            return
        rel = path.resolve().relative_to(self.root).as_posix()
        if rel in self._snapshots:
            return
        self._snapshots[rel] = (
            path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        )

    def rollback(self) -> list[str]:
        """Undo every write since :meth:`begin`. Returns the paths restored."""
        if not self._snapshots:
            self._snapshots = None
            return []
        restored: list[str] = []
        for rel, original in self._snapshots.items():
            path = self.root / rel
            try:
                if original is None:
                    if path.is_file():
                        path.unlink()
                else:
                    path.write_text(original, encoding="utf-8")
                restored.append(rel)
                self.touched.discard(rel)
            except OSError as exc:  # pragma: no cover - filesystem failure
                log.error("Could not roll back %s: %s", rel, exc)
        self._snapshots = None
        log.info("rolled back %d file(s)", len(restored))
        return restored

    def pending_diff(self, context: int = 3) -> str:
        """A unified diff of everything written since :meth:`begin`.

        Call before :meth:`commit_changes` or :meth:`rollback`, which both
        clear the snapshots.
        """
        if not self._snapshots:
            return ""
        parts: list[str] = []
        for rel in sorted(self._snapshots):
            original = self._snapshots[rel]
            path = self.root / rel
            current = (
                path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
            )
            if original == current:
                continue
            diff = difflib.unified_diff(
                (original or "").splitlines(keepends=True),
                (current or "").splitlines(keepends=True),
                fromfile=f"a/{rel}" if original is not None else "/dev/null",
                tofile=f"b/{rel}" if current is not None else "/dev/null",
                n=context,
            )
            parts.append("".join(diff))
        return "".join(parts)

    def commit_changes(self) -> list[str]:
        """Keep the writes made since :meth:`begin`. Returns the paths changed."""
        changed = sorted(self._snapshots.keys()) if self._snapshots else []
        self._snapshots = None
        return changed

def _looks_binary(path: Path) -> bool:
    """Cheap binary sniff: a NUL byte in the first 8 KiB."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(8192)
    except OSError:  # pragma: no cover - race with external edits
        return True


