"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Chat provider. Gemini is the default; Claude stays supported so an existing
# ANTHROPIC_API_KEY setup keeps working by setting PR_AGENT_PROVIDER=anthropic.
Provider = str  # one of: "google", "anthropic"

# The chat provider does not decide the embedding backend: Anthropic serves no
# embeddings endpoint at all, and Gemini's is a separate model and API call, so
# the backend stays a pluggable choice of its own.
EmbeddingBackend = str  # one of: "local", "google", "voyage", "none"

DEFAULT_MODELS = {
    # Pro-tier Gemini models are not served on the free API tier, so the
    # default is the newest flash model, which is.
    "google": "gemini-3.8-flash",
    "anthropic": "claude-opus-5",
}

_PROVIDER_ALIASES = {
    "google": "google",
    "gemini": "google",
    "google-genai": "google",
    "googlegenai": "google",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be set via environment variable (or a .env file) using the
    upper-cased field name, e.g. ``GOOGLE_API_KEY``, ``PR_AGENT_MODEL``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- credentials -----------------------------------------------------
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        serialization_alias="GOOGLE_API_KEY",
    )
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")

    # --- model -----------------------------------------------------------
    provider: Provider = Field(default="google", alias="PR_AGENT_PROVIDER")
    # Left empty, the model id defaults to DEFAULT_MODELS[provider].
    model: str = Field(default="", alias="PR_AGENT_MODEL")
    max_tokens: int = Field(default=16000, alias="PR_AGENT_MAX_TOKENS")
    # Let the model decide how much to think per request: adaptive thinking on
    # Claude, the model's own default thinking level on Gemini. Turning this
    # off asks for no thinking at all, which Gemini 3 models do not allow.
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
    google_embedding_model: str = Field(
        default="models/gemini-embedding-001", alias="PR_AGENT_GOOGLE_EMBEDDING_MODEL"
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

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        """Accept the names people actually type: 'gemini', 'claude', ..."""
        if isinstance(value, str):
            key = value.strip().lower()
            if key not in _PROVIDER_ALIASES:
                raise ValueError(
                    f"Unknown provider '{value}'. Use 'google' (Gemini) or 'anthropic' (Claude)."
                )
            return _PROVIDER_ALIASES[key]
        return value

    @model_validator(mode="after")
    def _default_model_for_provider(self) -> Settings:
        if not self.model:
            # Assign through __dict__ so this does not re-trigger validation.
            self.__dict__["model"] = DEFAULT_MODELS[self.provider]
            return self
        # A model id from the other provider is a config mistake worth naming
        # here rather than leaving to a 404 from the API.
        name = self.model.lower().removeprefix("models/")
        wrong = {"google": "claude", "anthropic": "gemini"}[self.provider]
        if name.startswith(wrong):
            other = "anthropic" if self.provider == "google" else "google"
            raise ValueError(
                f"Model '{self.model}' does not belong to provider '{self.provider}'. "
                f"Set PR_AGENT_PROVIDER={other} to use it."
            )
        return self

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

    @property
    def llm_api_key(self) -> str | None:
        """The API key for the configured chat provider, if one is set."""
        if self.provider == "google":
            return self.google_api_key
        return self.anthropic_api_key

    def resolved_index_dir(self) -> Path:
        """Index directory, resolved relative to the repository when relative."""
        if self.index_dir.is_absolute():
            return self.index_dir
        return (self.repo_path / self.index_dir).resolve()


def load_settings(**overrides: object) -> Settings:
    """Build settings, applying explicit overrides on top of the environment."""
    clean = {key: val for key, val in overrides.items() if val is not None}
    return Settings(**clean)  # type: ignore[arg-type]
