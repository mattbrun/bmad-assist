"""Loop and sprint configuration models."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoopConfig(BaseModel):
    """Development loop phase configuration.

    Defines the phases that run at different scopes in the development loop.
    This is the declarative configuration for workflow ordering.

    All phase names use snake_case to match Phase enum values exactly
    (e.g., "create_story", NOT "create-story").

    Attributes:
        epic_setup: Phases to run once at the start of each epic
            (before first story's CREATE_STORY). Example: ["testarch_setup"]
        story: Phases to run for each story in sequence.
            Standard: ["create_story", "validate_story", "validate_story_synthesis",
                      "dev_story", "code_review", "code_review_synthesis"]
        epic_teardown: Phases to run once at the end of each epic
            (after last story's CODE_REVIEW_SYNTHESIS). Example: ["retrospective"]

    Example:
        >>> config = LoopConfig(
        ...     epic_setup=[],
        ...     story=["create_story", "dev_story"],
        ...     epic_teardown=["retrospective"]
        ... )
        >>> "create_story" in config.story
        True

    Raises:
        ValueError: If story list is empty (must have at least one phase).

    """

    model_config = ConfigDict(frozen=True)

    epic_setup: list[str] = Field(
        default_factory=list,
        description="Phases to run once at the start of each epic",
    )
    story: list[str] = Field(
        default_factory=list,
        description="Phases to run for each story in sequence",
    )
    epic_teardown: list[str] = Field(
        default_factory=list,
        description="Phases to run once at the end of each epic",
    )

    @model_validator(mode="after")
    def validate_non_empty_story(self) -> Self:
        """Validate that story list is non-empty.

        An empty story list would cause the loop to have nothing to execute,
        which is always an error in the loop configuration.
        """
        if not self.story:
            raise ValueError("LoopConfig.story must contain at least one phase")
        return self


# Default loop configuration when no loop config found in any config file
DEFAULT_LOOP_CONFIG: LoopConfig = LoopConfig(
    epic_setup=[],
    story=[
        "create_story",
        "validate_story",
        "validate_story_synthesis",
        "dev_story",
        "code_review",
        "code_review_synthesis",
    ],
    epic_teardown=["retrospective"],
)


class SprintConfig(BaseModel):
    """Sprint-status management configuration.

    Controls sprint-status repair behavior including divergence thresholds,
    dialog timeouts, and module story prefixes.

    Attributes:
        divergence_threshold: Threshold for interactive repair dialog (0.3 = 30%).
            When divergence exceeds this, INTERACTIVE mode shows confirmation dialog.
        dialog_timeout_seconds: Timeout for repair dialog before auto-cancel.
            Range: 5-300 seconds. Default: 60 seconds.
        module_prefixes: Prefixes for module story classification.
            Stories with these prefixes are treated as MODULE_STORY entries.
        auto_repair: Enable silent auto-repair after phase completions.
        preserve_unknown: Never delete unknown entries during reconciliation.

    Example:
        >>> config = SprintConfig(
        ...     divergence_threshold=0.25,
        ...     dialog_timeout_seconds=30,
        ... )
        >>> config.divergence_threshold
        0.25

    """

    model_config = ConfigDict(frozen=True)

    divergence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for interactive repair dialog (0.3 = 30%)",
        json_schema_extra={"security": "safe", "ui_widget": "number"},
    )
    dialog_timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Timeout for repair dialog before auto-cancel (5-300 seconds)",
        json_schema_extra={"security": "safe", "ui_widget": "number", "unit": "s"},
    )
    module_prefixes: list[str] = Field(
        default_factory=lambda: ["testarch", "guardian"],
        description="Prefixes for module story classification",
        json_schema_extra={"security": "safe", "ui_widget": "text"},
    )
    auto_repair: bool = Field(
        default=True,
        description="Enable silent auto-repair after phase completions",
        json_schema_extra={"security": "safe", "ui_widget": "toggle"},
    )
    preserve_unknown: bool = Field(
        default=True,
        description="Never delete unknown entries during reconciliation",
        json_schema_extra={"security": "safe", "ui_widget": "toggle"},
    )


class WarningsConfig(BaseModel):
    """Warning suppression configuration."""

    model_config = ConfigDict(frozen=True)

    suppress_gitignore: bool = Field(
        default=False,
        description="Suppress gitignore configuration warnings during run/init",
        json_schema_extra={"security": "safe", "ui_widget": "toggle"},
    )
