"""Config template system for experiment framework.

This module provides configuration templates for experiment runs,
defining provider/model combinations in a type-safe, reusable way.

Usage:
    from bmad_assist.experiments import ConfigTemplate, ConfigRegistry

    # Load a single template
    template = load_config_template(Path("experiments/configs/opus-solo.yaml"))

    # Use registry for discovery and access
    registry = ConfigRegistry(Path("experiments/configs"))
    available = registry.list()
    template = registry.get("opus-solo")
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bmad_assist.core.config import (
    MAX_CONFIG_SIZE,
    MasterProviderConfig,
    MultiProviderConfig,
)
from bmad_assist.core.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Valid name pattern: starts with letter or underscore, contains only alphanumeric, hyphens, underscores # noqa: E501
NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")

# Variable pattern for resolution: ${var_name}
_VAR_PATTERN = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Known providers for validation warnings (not errors - forward compatibility)
KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {
        "claude",
        "claude-subprocess",
        "codex",
        "gemini",
    }
)


class ConfigTemplateProviders(BaseModel):
    """Provider configuration for a config template.

    Reuses the existing provider config structures from core/config.py.

    Attributes:
        master: Master LLM provider configuration.
        multi: List of Multi LLM validator configurations.

    """

    model_config = ConfigDict(frozen=True)

    master: MasterProviderConfig
    multi: list[MultiProviderConfig] = Field(default_factory=list)


class ConfigTemplate(BaseModel):
    """Configuration template for experiment runs.

    Defines which LLM providers and models to use for an experiment.
    Templates are stored as YAML files in experiments/configs/.

    Attributes:
        name: Unique identifier for this template (alphanumeric, hyphens, underscores).
        description: Human-readable description of the template.
        providers: Provider configuration section.

    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description="Unique identifier for this template",
        json_schema_extra={"security": "safe"},
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of this configuration",
        json_schema_extra={"security": "safe"},
    )
    providers: ConfigTemplateProviders = Field(
        ...,
        description="Provider configuration for this template",
    )

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty and follows naming rules."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        if not NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid name '{v}': must start with letter/underscore, "
                "contain only alphanumeric, hyphens, underscores"
            )
        return v


def _resolve_variables(content: str, context: dict[str, str | None]) -> str:
    """Resolve ${var_name} variables in content.

    Args:
        content: String content with variable placeholders.
        context: Mapping of variable names to values. None values indicate
            the variable is not available.

    Returns:
        Content with variables resolved.

    Raises:
        ConfigError: If a variable is referenced but not in context,
            or if a variable is in context but has None value.

    """

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name not in context:
            raise ConfigError(f"Unknown variable: ${{{var_name}}}")
        value = context[var_name]
        if value is None:
            raise ConfigError(
                f"Variable ${{{var_name}}} cannot be resolved: required context not provided"
            )
        return value

    return _VAR_PATTERN.sub(replace, content)


def _validate_provider_name(provider: str, source: str) -> None:
    """Warn if provider name is not in known list.

    Args:
        provider: Provider name to validate.
        source: Source location for logging (e.g., "master" or "multi[0]").

    """
    if provider not in KNOWN_PROVIDERS:
        logger.warning(
            "Unknown provider '%s' in %s. Known providers: %s. "
            "This may be a typo or a new provider not yet added to the known list.",
            provider,
            source,
            ", ".join(sorted(KNOWN_PROVIDERS)),
        )


def _validate_model_non_empty(model: str, source: str) -> None:
    """Validate that model is a non-empty string.

    Args:
        model: Model string to validate.
        source: Source location for error messages.

    Raises:
        ConfigError: If model is empty or whitespace-only.

    """
    if not model or not model.strip():
        raise ConfigError(f"Model in {source} cannot be empty or whitespace-only")


def _validate_settings_path(
    settings: str | None,
    source: str,
    project_root: Path | None,
) -> None:
    """Validate that settings path exists if provided.

    Args:
        settings: Settings path to validate (may be None).
        source: Source location for error messages.
        project_root: Project root for relative path resolution.

    Raises:
        ConfigError: If settings path is provided but doesn't exist.

    """
    if settings is None:
        return

    settings_path = Path(settings).expanduser()
    if not settings_path.is_absolute() and project_root is not None:
        settings_path = project_root / settings_path

    if not settings_path.exists():
        raise ConfigError(f"Settings file in {source} does not exist: {settings_path}")


def load_config_template(
    path: Path,
    project_root: Path | None = None,
) -> ConfigTemplate:
    """Load and validate a config template from YAML file.

    Args:
        path: Path to the YAML config template file.
        project_root: Project root for ${project} variable resolution.
            Required if the template uses ${project} variable.

    Returns:
        Validated ConfigTemplate instance.

    Raises:
        ConfigError: If file not found, invalid YAML, validation fails,
            unknown variable, or ${project} used without project_root.

    """
    # Check file exists
    if not path.exists():
        raise ConfigError(f"Config template not found: {path}")

    if not path.is_file():
        raise ConfigError(f"Config template path is not a file: {path}")

    # Read file with size limit
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read(MAX_CONFIG_SIZE + 1)
    except PermissionError as e:
        raise ConfigError(f"Permission denied reading {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read config template {path}: {e}") from e

    if len(content) > MAX_CONFIG_SIZE:
        raise ConfigError(f"Config template {path} exceeds 1MB limit")

    # Build variable context
    var_context: dict[str, str | None] = {
        "home": os.path.expanduser("~"),
    }

    # ${project} requires project_root parameter
    if "${project}" in content:
        if project_root is None:
            raise ConfigError("project_root parameter required for ${project} variable resolution")
        var_context["project"] = str(project_root)

    # Resolve variables before YAML parsing
    try:
        resolved_content = _resolve_variables(content, var_context)
    except ConfigError:
        raise

    # Parse YAML
    try:
        data = yaml.safe_load(resolved_content)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        raise ConfigError(f"Config template {path} is empty")

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config template {path} must contain a YAML mapping, got {type(data).__name__}"
        )

    # Validate with Pydantic
    try:
        template = ConfigTemplate.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Config template validation failed for {path}: {e}") from e

    # Additional validation: warn on unknown providers
    _validate_provider_name(template.providers.master.provider, "master")
    for i, multi in enumerate(template.providers.multi):
        _validate_provider_name(multi.provider, f"multi[{i}]")

    # Validate model is non-empty
    _validate_model_non_empty(template.providers.master.model, "master")
    for i, multi in enumerate(template.providers.multi):
        _validate_model_non_empty(multi.model, f"multi[{i}]")

    # Validate settings paths exist
    _validate_settings_path(
        template.providers.master.settings,
        "master",
        project_root,
    )
    for i, multi in enumerate(template.providers.multi):
        _validate_settings_path(multi.settings, f"multi[{i}]", project_root)

    return template


class ConfigRegistry:
    """Registry for discovering and accessing config templates.

    Provides discovery of config templates in a directory and
    cached access to template instances.

    Usage:
        registry = ConfigRegistry(Path("experiments/configs"))
        names = registry.list()
        template = registry.get("opus-solo")

    """

    def __init__(
        self,
        configs_dir: Path,
        project_root: Path | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            configs_dir: Directory containing config template YAML files.
            project_root: Project root for ${project} variable resolution.

        """
        self._configs_dir = configs_dir
        self._project_root = project_root
        self._templates: dict[str, Path] = {}
        self._discovered = False

    def _ensure_discovered(self) -> None:
        """Ensure templates have been discovered."""
        if not self._discovered:
            self._templates = self.discover(self._configs_dir)
            self._discovered = True

    def discover(self, configs_dir: Path) -> dict[str, Path]:
        """Discover config templates in a directory.

        Scans for *.yaml files (not *.yml) and validates that each file's
        internal 'name' field matches the filename stem.

        Args:
            configs_dir: Directory to scan for templates.

        Returns:
            Mapping of template name to file path.

        """
        if not configs_dir.exists():
            logger.info(
                "Config templates directory does not exist: %s",
                configs_dir,
            )
            return {}

        if not configs_dir.is_dir():
            logger.warning(
                "Config templates path is not a directory: %s",
                configs_dir,
            )
            return {}

        templates: dict[str, Path] = {}

        for yaml_file in configs_dir.glob("*.yaml"):
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
        """Return list of available config template names.

        Returns:
            Sorted list of template names.

        """
        self._ensure_discovered()
        return sorted(self._templates.keys())

    def get(self, name: str) -> ConfigTemplate:
        """Get a config template by name.

        Args:
            name: Template name (filename stem without .yaml).

        Returns:
            Loaded and validated ConfigTemplate.

        Raises:
            ConfigError: If template not found.

        """
        self._ensure_discovered()

        if name not in self._templates:
            available = ", ".join(sorted(self._templates.keys())) or "(none)"
            raise ConfigError(f"Config template '{name}' not found. Available: {available}")

        return self._load_template(name)

    @lru_cache(maxsize=32)  # noqa: B019
    def _load_template(self, name: str) -> ConfigTemplate:
        """Load a template with caching.

        Args:
            name: Template name.

        Returns:
            Loaded ConfigTemplate.

        """
        path = self._templates[name]
        return load_config_template(path, self._project_root)
