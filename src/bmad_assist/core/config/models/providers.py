"""Provider configuration models for Master, Multi, and Helper LLMs."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MasterProviderConfig(BaseModel):
    """Configuration for Master LLM provider.

    The Master provider is the primary LLM that can modify files and
    synthesize validation reports.

    Attributes:
        provider: Provider name (e.g., "claude-subprocess", "codex", "gemini").
        model: Model identifier for CLI invocation (e.g., "opus", "sonnet").
        model_name: Display name for the model (e.g., "glm-4.7"). If set,
            used in logs/reports instead of model. Useful when "tricking"
            a CLI to use alternative models.
        settings: Optional path to provider settings JSON file (tilde expanded).

    """

    model_config = ConfigDict(frozen=True)

    provider: str = Field(
        ...,
        description="Provider name: claude-subprocess, codex, gemini",
        json_schema_extra={"security": "risky", "ui_widget": "dropdown"},
    )
    model: str = Field(
        ...,
        description="Model identifier for CLI: opus, sonnet, etc.",
        json_schema_extra={"security": "risky", "ui_widget": "dropdown"},
    )
    model_name: str | None = Field(
        None,
        description="Display name for model (used in logs/reports instead of model)",
        json_schema_extra={"security": "safe", "ui_widget": "text"},
    )
    settings: str | None = Field(
        None,
        description="Path to provider settings JSON (tilde expanded)",
        json_schema_extra={"security": "dangerous"},
    )

    @property
    def display_model(self) -> str:
        """Return model name for display (model_name if set, else model)."""
        return self.model_name or self.model

    @property
    def settings_path(self) -> Path | None:
        """Return expanded settings path, or None if not set."""
        if self.settings is None:
            return None
        return Path(self.settings).expanduser()


class MultiProviderConfig(BaseModel):
    """Configuration for Multi LLM validator.

    Multi providers are used for parallel validation during
    VALIDATE_STORY and CODE_REVIEW phases.

    Attributes:
        provider: Provider name (e.g., "claude-subprocess", "codex", "gemini").
        model: Model identifier for CLI invocation (e.g., "opus", "sonnet").
        model_name: Display name for the model (e.g., "glm-4.7"). If set,
            used in logs/reports instead of model.
        settings: Optional path to provider settings JSON file (tilde expanded).

    """

    model_config = ConfigDict(frozen=True)

    provider: str = Field(
        ...,
        description="Provider name: claude-subprocess, codex, gemini",
        json_schema_extra={"security": "risky", "ui_widget": "dropdown"},
    )
    model: str = Field(
        ...,
        description="Model identifier for CLI: opus, sonnet, etc.",
        json_schema_extra={"security": "risky", "ui_widget": "dropdown"},
    )
    model_name: str | None = Field(
        None,
        description="Display name for model (used in logs/reports instead of model)",
        json_schema_extra={"security": "safe", "ui_widget": "text"},
    )
    settings: str | None = Field(
        None,
        description="Path to provider settings JSON (tilde expanded)",
        json_schema_extra={"security": "dangerous"},
    )

    @property
    def display_model(self) -> str:
        """Return model name for display (model_name if set, else model)."""
        return self.model_name or self.model

    @property
    def settings_path(self) -> Path | None:
        """Return expanded settings path, or None if not set."""
        if self.settings is None:
            return None
        return Path(self.settings).expanduser()


class HelperProviderConfig(BaseModel):
    """Configuration for Helper LLM provider.

    The Helper provider is used for secondary tasks like metrics extraction,
    summarization, and eligibility assessment. Typically a fast/cheap model.

    Attributes:
        provider: Provider name (e.g., "claude-subprocess", "codex", "gemini").
        model: Model identifier for CLI invocation (e.g., "haiku", "sonnet").
        model_name: Display name for the model (e.g., "glm-4.7"). If set,
            used in logs/reports instead of model. Useful when "tricking"
            a CLI to use alternative models.
        settings: Optional path to provider settings JSON file (tilde expanded).

    """

    model_config = ConfigDict(frozen=True)

    provider: str = Field(
        default="claude",
        description="Provider name: claude-subprocess, codex, gemini",
        json_schema_extra={"security": "risky", "ui_widget": "dropdown"},
    )
    model: str = Field(
        default="haiku",
        description="Model identifier for CLI: haiku, sonnet, etc.",
        json_schema_extra={"security": "risky", "ui_widget": "dropdown"},
    )
    model_name: str | None = Field(
        None,
        description="Display name for model (used in logs/reports instead of model)",
        json_schema_extra={"security": "safe", "ui_widget": "text"},
    )
    settings: str | None = Field(
        None,
        description="Path to provider settings JSON (tilde expanded)",
        json_schema_extra={"security": "dangerous"},
    )

    @property
    def display_model(self) -> str:
        """Return model name for display (model_name if set, else model)."""
        return self.model_name or self.model

    @property
    def settings_path(self) -> Path | None:
        """Return expanded settings path, or None if not set."""
        if self.settings is None:
            return None
        return Path(self.settings).expanduser()


class ProviderConfig(BaseModel):
    """Provider configuration section.

    Contains configuration for Master, Multi, and Helper LLM providers.

    Attributes:
        master: Configuration for the Master LLM provider.
        multi: List of Multi LLM validator configurations.
        helper: Configuration for the Helper LLM provider (metrics extraction, summarization, etc.).

    """

    model_config = ConfigDict(frozen=True)

    master: MasterProviderConfig
    multi: list[MultiProviderConfig] = Field(default_factory=list)
    helper: HelperProviderConfig = Field(default_factory=HelperProviderConfig)
