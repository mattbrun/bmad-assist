"""Main Config model that aggregates all configuration sections."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bmad_assist.core.config.models.features import (
    AntipatternConfig,
    BenchmarkingConfig,
    CompilerConfig,
    QAConfig,
    TimeoutsConfig,
)
from bmad_assist.core.config.models.loop import LoopConfig, SprintConfig, WarningsConfig
from bmad_assist.core.config.models.paths import (
    BmadPathsConfig,
    PowerPromptConfig,
    ProjectPathsConfig,
)
from bmad_assist.core.config.models.providers import ProviderConfig
from bmad_assist.notifications.config import NotificationConfig
from bmad_assist.testarch.config import TestarchConfig


class Config(BaseModel):
    """Main bmad-assist configuration model.

    This is the root configuration model that composes all nested
    configuration sections.

    Migration Notes (v6.0.0+):
        The `providers.helper` section is now the single source of truth for
        secondary LLM tasks (metrics extraction, summarization, eligibility).
        Old config paths are deprecated but still work:
        - `benchmarking.extraction_provider`/`extraction_model` -> use `providers.helper`
        - `testarch.eligibility.provider`/`model` -> use `providers.helper`

    Attributes:
        providers: Provider configuration section.
        power_prompts: Power-prompt configuration section.
        state_path: Path to state file, or None to use default (~/.bmad-assist/state.yaml).
            Tilde (~) is expanded to home directory. Use get_state_path() for resolved path.
        timeout: Global timeout for provider operations in seconds.
        bmad_paths: Paths to BMAD documentation files.
        paths: Project paths configuration for artifact organization.
        compiler: Compiler configuration options.
        benchmarking: Benchmarking/metrics extraction configuration.
        notifications: Notification system configuration (optional).
        testarch: Testarch module configuration (optional).
        sprint: Sprint-status management configuration (optional).
        workflow_variant: Workflow variant identifier for A/B testing.

    """

    model_config = ConfigDict(frozen=True)

    providers: ProviderConfig
    power_prompts: PowerPromptConfig = Field(default_factory=PowerPromptConfig)
    state_path: str | None = Field(
        default=None,
        description="Path to state file. If None, defaults to ~/.bmad-assist/state.yaml. "
        "Supports tilde (~) expansion and relative paths.",
        json_schema_extra={"security": "dangerous"},
    )
    timeout: int = Field(
        default=300,
        description="Global timeout for providers in seconds (legacy, prefer timeouts)",
        json_schema_extra={"security": "safe", "ui_widget": "number"},
    )
    timeouts: TimeoutsConfig | None = Field(
        default=None,
        description="Per-phase timeout configuration (optional, overrides timeout)",
    )
    bmad_paths: BmadPathsConfig = Field(default_factory=BmadPathsConfig)
    paths: ProjectPathsConfig = Field(default_factory=ProjectPathsConfig)
    compiler: CompilerConfig = Field(default_factory=CompilerConfig)
    benchmarking: BenchmarkingConfig = Field(default_factory=BenchmarkingConfig)
    antipatterns: AntipatternConfig = Field(
        default_factory=AntipatternConfig,
        description="Antipatterns extraction and loading configuration",
    )
    notifications: NotificationConfig | None = Field(
        default=None,
        description="Notification system configuration (optional)",
    )
    testarch: TestarchConfig | None = Field(
        default=None,
        description="Testarch module configuration (optional)",
    )
    sprint: SprintConfig | None = Field(
        default=None,
        description="Sprint-status management configuration (optional)",
    )
    qa: QAConfig | None = Field(
        default=None,
        description="QA execution configuration (optional)",
    )
    loop: LoopConfig | None = Field(
        default=None,
        description="Loop phase configuration (optional, uses DEFAULT_LOOP_CONFIG if not set)",
    )
    warnings: WarningsConfig | None = Field(
        default=None,
        description="Warning suppression configuration (optional)",
    )
    workflow_variant: str = Field(
        default="default",
        description="Workflow variant identifier for A/B testing",
        json_schema_extra={"security": "safe", "ui_widget": "text"},
    )

    @model_validator(mode="after")
    def expand_state_path(self) -> Self:
        """Expand ~ to user home directory in state_path.

        Only expands user home (~) - does not modify relative paths.
        Uses model_validator to ensure default values are also processed.
        Skips expansion if state_path is None (uses default via get_state_path).
        """
        if self.state_path is not None and self.state_path.startswith("~"):
            # Use object.__setattr__ since model is frozen
            object.__setattr__(self, "state_path", str(Path(self.state_path).expanduser()))
        return self
