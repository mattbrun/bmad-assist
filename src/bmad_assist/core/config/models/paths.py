"""Path configuration models for BMAD documentation and project artifacts."""

from pydantic import BaseModel, ConfigDict, Field


class PowerPromptConfig(BaseModel):
    """Power-prompt configuration section.

    Power-prompts are enhanced prompts tailored to specific tech stacks
    and project types.

    Attributes:
        set_name: Name of power-prompt set to use (e.g., "python-cli").
        variables: Custom variables to inject into prompts.

    """

    model_config = ConfigDict(frozen=True)

    set_name: str | None = Field(
        default=None,
        description="Name of power-prompt set to use",
        json_schema_extra={"security": "safe", "ui_widget": "text"},
    )
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Custom variables to inject into prompts",
        json_schema_extra={"security": "safe"},
    )


class BmadPathsConfig(BaseModel):
    """Paths to BMAD documentation files.

    These paths point to project documentation that provides context
    for LLM operations.

    All path fields are marked as dangerous and excluded from dashboard schema.

    Attributes:
        prd: Path to Product Requirements Document.
        architecture: Path to Architecture document.
        epics: Path to Epics document.
        stories: Path to Stories directory.

    """

    model_config = ConfigDict(frozen=True)

    prd: str | None = Field(
        default=None,
        description="Path to Product Requirements Document",
        json_schema_extra={"security": "dangerous"},
    )
    architecture: str | None = Field(
        default=None,
        description="Path to Architecture document",
        json_schema_extra={"security": "dangerous"},
    )
    epics: str | None = Field(
        default=None,
        description="Path to Epics document",
        json_schema_extra={"security": "dangerous"},
    )
    stories: str | None = Field(
        default=None,
        description="Path to Stories directory",
        json_schema_extra={"security": "dangerous"},
    )


class ProjectPathsConfig(BaseModel):
    """Project paths configuration for artifact organization.

    Defines the folder structure for planning and implementation artifacts.
    All paths support {project-root} placeholder which is resolved at runtime.

    All path fields are marked as dangerous and excluded from dashboard schema.

    Attributes:
        output_folder: Base output folder for all generated artifacts.
        planning_artifacts: Folder for planning phase artifacts (epics, stories).
        implementation_artifacts: Folder for implementation artifacts (validations, reviews).
        project_knowledge: Folder for project documentation (PRD, architecture).

    """

    model_config = ConfigDict(frozen=True)

    output_folder: str = Field(
        default="{project-root}/_bmad-output",
        description="Base output folder for all generated artifacts",
        json_schema_extra={"security": "dangerous"},
    )
    planning_artifacts: str = Field(
        default="{project-root}/_bmad-output/planning-artifacts",
        description="Folder for planning phase artifacts (epics, stories)",
        json_schema_extra={"security": "dangerous"},
    )
    implementation_artifacts: str = Field(
        default="{project-root}/_bmad-output/implementation-artifacts",
        description="Folder for implementation artifacts (validations, reviews, benchmarks)",
        json_schema_extra={"security": "dangerous"},
    )
    project_knowledge: str = Field(
        default="{project-root}/docs",
        description="Folder for project documentation (PRD, architecture)",
        json_schema_extra={"security": "dangerous"},
    )
