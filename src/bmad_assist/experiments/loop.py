"""Loop template system for experiment framework.

This module provides loop templates for experiment runs,
defining workflow sequences in a type-safe, reusable way.

Usage:
    from bmad_assist.experiments import LoopTemplate, LoopRegistry

    # Load a single template
    template = load_loop_template(Path("experiments/loops/standard.yaml"))

    # Use registry for discovery and access
    registry = LoopRegistry(Path("experiments/loops"))
    available = registry.list()
    template = registry.get("standard")
"""

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bmad_assist.core.config import MAX_CONFIG_SIZE
from bmad_assist.core.exceptions import ConfigError

# Reuse name pattern from config module
from bmad_assist.experiments.config import NAME_PATTERN

logger = logging.getLogger(__name__)

# Known workflows - supports both snake_case (LoopConfig convention) and kebab-case (legacy)
# Note: test-design is a CUSTOM workflow not in Phase enum - used for experimental ATDD loops
KNOWN_WORKFLOWS: frozenset[str] = frozenset(
    {
        # snake_case (matches LoopConfig and Phase enum values)
        "create_story",
        "validate_story",
        "validate_story_synthesis",
        "atdd",
        "dev_story",
        "code_review",
        "code_review_synthesis",
        "test_review",
        "retrospective",
        "qa_plan_generate",
        "qa_plan_execute",
        # kebab-case (legacy, for backwards compatibility)
        "create-story",
        "validate-story",
        "validate-story-synthesis",
        "dev-story",
        "code-review",
        "code-review-synthesis",
        "test-review",
        "qa-plan-generate",
        "qa-plan-execute",
        # CUSTOM workflows (not a Phase enum value)
        "test-design",  # ATDD test planning
        "test_design",  # snake_case variant
    }
)


def normalize_workflow_name(workflow: str) -> str:
    """Normalize workflow name to snake_case.

    Converts kebab-case to snake_case for consistency with LoopConfig
    and Phase enum values.

    Args:
        workflow: Workflow name in either convention.

    Returns:
        Normalized snake_case workflow name.

    Example:
        >>> normalize_workflow_name("create-story")
        'create_story'
        >>> normalize_workflow_name("dev_story")
        'dev_story'

    """
    return workflow.replace("-", "_")


class LoopStep(BaseModel):
    """Single step in a loop template sequence.

    Attributes:
        workflow: Workflow name (maps to Phase enum or custom workflow).
        required: If True, failure stops the experiment run.

    """

    model_config = ConfigDict(frozen=True)

    workflow: str = Field(
        ...,
        description="Workflow name to execute",
        json_schema_extra={"security": "safe"},
    )
    required: bool = Field(
        default=True,
        description="If True, failure stops the experiment",
    )

    @field_validator("workflow", mode="after")
    @classmethod
    def validate_workflow_non_empty(cls, v: str) -> str:
        """Validate workflow name is non-empty and strip whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("workflow cannot be empty")
        return v


class LoopTemplate(BaseModel):
    """Loop template defining workflow sequence for experiments.

    Defines which workflows run and in what order during an experiment.
    Templates are stored as YAML files in experiments/loops/.

    Attributes:
        name: Unique identifier for this template.
        description: Human-readable description.
        sequence: Ordered list of workflow steps to execute.

    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description="Unique identifier for this template",
        json_schema_extra={"security": "safe"},
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of this loop",
        json_schema_extra={"security": "safe"},
    )
    sequence: list[LoopStep] = Field(
        ...,
        description="Ordered sequence of workflow steps",
        min_length=1,
    )

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty, strip whitespace, and follow naming rules."""
        v = v.strip()
        if not v:
            raise ValueError("name cannot be empty")
        if not NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid name '{v}': must start with letter/underscore, "
                "contain only alphanumeric, hyphens, underscores"
            )
        return v


def _validate_workflow_name(workflow: str, template_name: str) -> None:
    """Warn if workflow name is not in known list.

    Args:
        workflow: Workflow name to validate.
        template_name: Template name for logging context.

    """
    if workflow not in KNOWN_WORKFLOWS:
        logger.warning(
            "Unknown workflow '%s' in template '%s'. Known workflows: %s. "
            "This may be a typo or a new workflow not yet added to the known list.",
            workflow,
            template_name,
            ", ".join(sorted(KNOWN_WORKFLOWS)),
        )


def load_loop_template(path: Path) -> LoopTemplate:
    """Load and validate a loop template from YAML file.

    Args:
        path: Path to the YAML loop template file.

    Returns:
        Validated LoopTemplate instance.

    Raises:
        ConfigError: If file not found, invalid YAML, or validation fails.

    """
    # Check file exists
    if not path.exists():
        raise ConfigError(f"Loop template not found: {path}")

    if not path.is_file():
        raise ConfigError(f"Loop template path is not a file: {path}")

    # Read file with size limit
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read(MAX_CONFIG_SIZE + 1)
    except PermissionError as e:
        raise ConfigError(f"Permission denied reading {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read loop template {path}: {e}") from e

    if len(content) > MAX_CONFIG_SIZE:
        raise ConfigError(f"Loop template {path} exceeds 1MB limit")

    # Parse YAML
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        raise ConfigError(f"Loop template {path} is empty")

    if not isinstance(data, dict):
        raise ConfigError(
            f"Loop template {path} must contain a YAML mapping, got {type(data).__name__}"
        )

    # Validate with Pydantic
    try:
        template = LoopTemplate.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Loop template validation failed for {path}: {e}") from e

    # Validate workflow names (warn only, don't fail for forward compatibility)
    for step in template.sequence:
        _validate_workflow_name(step.workflow, template.name)

    return template


class LoopRegistry:
    """Registry for discovering and accessing loop templates.

    Provides discovery of loop templates in a directory and
    cached access to template instances.

    Usage:
        registry = LoopRegistry(Path("experiments/loops"))
        names = registry.list()
        template = registry.get("standard")

    """

    def __init__(self, loops_dir: Path) -> None:
        """Initialize the registry.

        Args:
            loops_dir: Directory containing loop template YAML files.

        """
        self._loops_dir = loops_dir
        self._templates: dict[str, Path] = {}
        self._discovered = False

    def _ensure_discovered(self) -> None:
        """Ensure templates have been discovered."""
        if not self._discovered:
            self._templates = self.discover(self._loops_dir)
            self._discovered = True

    def discover(self, loops_dir: Path) -> dict[str, Path]:
        """Discover loop templates in a directory.

        Scans for *.yaml files (not *.yml) and validates that each file's
        internal 'name' field matches the filename stem.

        Args:
            loops_dir: Directory to scan for templates.

        Returns:
            Mapping of template name to file path.

        """
        if not loops_dir.exists():
            logger.info(
                "Loop templates directory does not exist: %s",
                loops_dir,
            )
            return {}

        if not loops_dir.is_dir():
            logger.warning(
                "Loop templates path is not a directory: %s",
                loops_dir,
            )
            return {}

        templates: dict[str, Path] = {}

        for yaml_file in loops_dir.glob("*.yaml"):
            # Skip hidden files
            if yaml_file.name.startswith("."):
                continue

            expected_name = yaml_file.stem

            # Try to read and validate the file
            try:
                with yaml_file.open("r", encoding="utf-8") as f:
                    content = f.read(MAX_CONFIG_SIZE + 1)

                if len(content) > MAX_CONFIG_SIZE:
                    logger.warning(
                        "Skipping %s: file exceeds 1MB limit",
                        yaml_file,
                    )
                    continue

                data = yaml.safe_load(content)

                if not isinstance(data, dict):
                    logger.warning(
                        "Skipping %s: not a YAML mapping",
                        yaml_file,
                    )
                    continue

                # Check name field matches filename
                name = data.get("name")
                if name != expected_name:
                    logger.warning(
                        "Skipping %s: internal name '%s' does not match filename stem '%s'",
                        yaml_file,
                        name,
                        expected_name,
                    )
                    continue

                templates[expected_name] = yaml_file

            except yaml.YAMLError as e:
                logger.warning("Skipping %s: invalid YAML: %s", yaml_file, e)
                continue
            except OSError as e:
                logger.warning("Skipping %s: cannot read: %s", yaml_file, e)
                continue

        return templates

    def list(self) -> list[str]:
        """Return list of available loop template names.

        Returns:
            Sorted list of template names.

        """
        self._ensure_discovered()
        return sorted(self._templates.keys())

    def get(self, name: str) -> LoopTemplate:
        """Get a loop template by name.

        Args:
            name: Template name (filename stem without .yaml).

        Returns:
            Loaded and validated LoopTemplate.

        Raises:
            ConfigError: If template not found.

        """
        self._ensure_discovered()

        if name not in self._templates:
            available = ", ".join(sorted(self._templates.keys())) or "(none)"
            raise ConfigError(f"Loop template '{name}' not found. Available: {available}")

        return self._load_template(name)

    @lru_cache(maxsize=32)  # noqa: B019
    def _load_template(self, name: str) -> LoopTemplate:
        """Load a template with caching.

        Args:
            name: Template name.

        Returns:
            Loaded LoopTemplate.

        """
        path = self._templates[name]
        return load_loop_template(path)
