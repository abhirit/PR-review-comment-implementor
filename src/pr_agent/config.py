"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Anthropic does not ship an embeddings endpoint, so the embedding backend is a
# separate, pluggable choice from the chat model.
EmbeddingBackend = str  # one of: "local", "voyage", "none"


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be set via environment variable (or a .env file) using the
    upper-cased field name, e.g. ``ANTHROPIC_API_KEY``, ``PR_AGENT_MODEL``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- credentials -----------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")

    # --- model -----------------------------------------------------------
    model: str = Field(default="claude-opus-5", alias="PR_AGENT_MODEL")
    max_tokens: int = Field(default=16000, alias="PR_AGENT_MAX_TOKENS")
    # Adaptive thinking: Claude decides when and how deeply to think. The older
    # fixed `budget_tokens` form is rejected by current models.
    thinking: bool = Field(default=True, alias="PR_AGENT_THINKING")
    request_timeout: float = Field(default=600.0, alias="PR_AGENT_REQUEST_TIMEOUT")

    # --- github ----------------------------------------------------------
    github_api_url: str = Field(default="https://api.github.com", alias="GITHUB_API_URL")

    # --- repository / workspace -----------------------------------------
    repo_path: Path = Field(default=Path("."), alias="PR_AGENT_REPO_PATH")

    # --- retrieval -------------------------------------------------------
    # BM25 over code-aware tokens already matches the identifiers reviewers
    # quote, and the model can grep for the rest, so embeddings are opt-in
    # rather than a required model download on first run.
    embedding_backend: EmbeddingBackend = Field(default="none", alias="PR_AGENT_EMBEDDINGS")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="PR_AGENT_EMBEDDING_MODEL",
    )
    voyage_model: str = Field(default="voyage-code-3", alias="PR_AGENT_VOYAGE_MODEL")
    index_dir: Path = Field(default=Path(".pr_agent/index"), alias="PR_AGENT_INDEX_DIR")
    chunk_size: int = Field(default=1200, alias="PR_AGENT_CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="PR_AGENT_CHUNK_OVERLAP")
    retrieval_k: int = Field(default=8, alias="PR_AGENT_RETRIEVAL_K")
    max_index_file_bytes: int = Field(default=400_000, alias="PR_AGENT_MAX_FILE_BYTES")

    # --- agent behaviour -------------------------------------------------
    max_fix_attempts: int = Field(default=3, alias="PR_AGENT_MAX_FIX_ATTEMPTS")
    max_tool_iterations: int = Field(default=25, alias="PR_AGENT_MAX_TOOL_ITERATIONS")
    # NoDecode stops pydantic-settings from JSON-decoding the env value, so
    # the validator below can accept a plain ';;'-separated string.
    validate_commands: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="PR_AGENT_VALIDATE"
    )
    validate_timeout: int = Field(default=900, alias="PR_AGENT_VALIDATE_TIMEOUT")

    @field_validator("validate_commands", mode="before")
    @classmethod
    def _split_commands(cls, value: object) -> object:
        """Allow ``PR_AGENT_VALIDATE="ruff check .;;pytest -q"`` in the environment."""
        if isinstance(value, str):
            return [part.strip() for part in value.split(";;") if part.strip()]
        return value

    @field_validator("repo_path", "index_dir", mode="before")
    @classmethod
    def _expand(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    def resolved_index_dir(self) -> Path:
        """Index directory, resolved relative to the repository when relative."""
        if self.index_dir.is_absolute():
            return self.index_dir
        return (self.repo_path / self.index_dir).resolve()


def load_settings(**overrides: object) -> Settings:
    """Build settings, applying explicit overrides on top of the environment."""
    clean = {key: val for key, val in overrides.items() if val is not None}
    return Settings(**clean)  # type: ignore[arg-type]
