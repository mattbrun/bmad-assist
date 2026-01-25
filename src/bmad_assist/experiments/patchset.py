"""Patch-set system for experiment framework.

This module provides patch-set manifests for experiment runs,
defining which workflow patches and overrides to use in a type-safe, reusable way.

Usage:
    from bmad_assist.experiments import PatchSetManifest, PatchSetRegistry

    # Load a single manifest
    manifest = load_patchset_manifest(Path("experiments/patch-sets/baseline.yaml"))

    # Use registry for discovery and access
    registry = PatchSetRegistry(Path("experiments/patch-sets"))
    available = registry.list()
    manifest = registry.get("baseline")
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from bmad_assist.core.config import MAX_CONFIG_SIZE
from bmad_assist.core.exceptions import ConfigError

# Reuse patterns from config and loop modules
from bmad_assist.experiments.config import _VAR_PATTERN, NAME_PATTERN
from bmad_assist.experiments.loop import KNOWN_WORKFLOWS

logger = logging.getLogger(__name__)


class PatchSetManifest(BaseModel):
    """Patch-set manifest defining workflow patches for experiments.

    Defines which patch files or alternative workflows to use during an experiment.
    Enables A/B testing of patch versions and workflow implementations.

    Attributes:
        name: Unique identifier for this patch-set.
        description: Human-readable description.
        patches: Mapping of workflow name to patch file path (None = no patch).
        workflow_overrides: Mapping of workflow name to alternative workflow directory.

    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description="Unique identifier for this patch-set",
        json_schema_extra={"security": "safe"},
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of this patch-set",
        json_schema_extra={"security": "safe"},
    )
    patches: dict[str, str | None] = Field(
        default_factory=dict,
        description="Workflow name → patch file path (None = no patch)",
    )
    workflow_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Workflow name → alternative workflow directory",
    )

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty and follows naming rules."""
        v = v.strip()
        if not v:
            raise ValueError("name cannot be empty")
        if not NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid name '{v}': must start with letter/underscore, "
                "contain only alphanumeric, hyphens, underscores"
            )
        return v

    @field_validator("patches", mode="after")
    @classmethod
    def validate_patches(cls, v: dict[str, str | None]) -> dict[str, str | None]:
        """Validate patch entries."""
        for workflow, path in v.items():
            if not workflow or not workflow.strip():
                raise ValueError("Workflow name in patches cannot be empty")
            # path can be None (no patch) or non-empty string
            if path is not None and not path.strip():
                raise ValueError(f"Patch path for '{workflow}' cannot be empty string")
        return v

    @field_validator("workflow_overrides", mode="after")
    @classmethod
    def validate_workflow_overrides(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate workflow override entries."""
        for workflow, path in v.items():
            if not workflow or not workflow.strip():
                raise ValueError("Workflow name in workflow_overrides cannot be empty")
            if not path or not path.strip():
                raise ValueError(f"Workflow override path for '{workflow}' cannot be empty")
        return v

    @model_validator(mode="after")
    def warn_patch_override_conflict(self) -> "PatchSetManifest":
        """Warn if workflow appears in both patches (non-null) and workflow_overrides."""
        for wf in self.workflow_overrides:
            if wf in self.patches and self.patches[wf] is not None:
                logger.warning(
                    "Workflow '%s' has both patch and workflow_override; "
                    "workflow_override takes precedence, patch ignored",
                    wf,
                )
        return self


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

    def replace(match: "re.Match[str]") -> str:
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


def _resolve_patch_path(path: str, manifest_dir: Path) -> Path:
    """Resolve a patch file path after variable substitution.

    Args:
        path: Path string with variables already resolved.
        manifest_dir: Parent directory of the manifest file.

    Returns:
        Normalized absolute path.

    """
    # Handle tilde paths (e.g., ~/patches/file.yaml)
    if path.startswith("~"):
        return Path(path).expanduser().resolve()

    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    # Relative path - resolve against manifest directory
    return (manifest_dir / path_obj).resolve()


def _validate_workflow_name(workflow: str, manifest_name: str, source: str) -> None:
    """Warn if workflow name is not in known list.

    Args:
        workflow: Workflow name to validate.
        manifest_name: Manifest name for logging context.
        source: Source section (patches or workflow_overrides).

    """
    if workflow not in KNOWN_WORKFLOWS:
        logger.warning(
            "Unknown workflow '%s' in %s of manifest '%s'. Known workflows: %s. "
            "This may be a typo or a new workflow not yet added to the known list.",
            workflow,
            source,
            manifest_name,
            ", ".join(sorted(KNOWN_WORKFLOWS)),
        )


def load_patchset_manifest(
    path: Path,
    project_root: Path | None = None,
    validate_paths: bool = True,
) -> PatchSetManifest:
    """Load and validate a patch-set manifest from YAML file.

    Args:
        path: Path to the YAML patch-set manifest file.
        project_root: Project root for ${project} variable resolution.
            Required if the manifest uses ${project} variable.
        validate_paths: If True, validate that patch files and workflow_override
            directories exist. Set to False for discovery mode.

    Returns:
        Validated PatchSetManifest instance.

    Raises:
        ConfigError: If file not found, invalid YAML, validation fails,
            unknown variable, or ${project} used without project_root.

    """
    # Check file exists
    if not path.exists():
        raise ConfigError(f"Patch-set manifest not found: {path}")

    if not path.is_file():
        raise ConfigError(f"Patch-set manifest path is not a file: {path}")

    # Read file with size limit
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read(MAX_CONFIG_SIZE + 1)
    except PermissionError as e:
        raise ConfigError(f"Permission denied reading {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read patch-set manifest {path}: {e}") from e

    if len(content) > MAX_CONFIG_SIZE:
        raise ConfigError(f"Patch-set manifest {path} exceeds 1MB limit")

    # Build variable context
    var_context: dict[str, str | None] = {
        "home": str(Path.home()),
    }

    # ${project} requires project_root parameter
    if "${project}" in content:
        if project_root is None:
            raise ConfigError("project_root parameter required for ${project} variable resolution")
        var_context["project"] = str(project_root.resolve())

    # Resolve variables before YAML parsing
    resolved_content = _resolve_variables(content, var_context)

    # Parse YAML
    try:
        data = yaml.safe_load(resolved_content)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        raise ConfigError(f"Patch-set manifest {path} is empty")

    if not isinstance(data, dict):
        raise ConfigError(
            f"Patch-set manifest {path} must contain a YAML mapping, got {type(data).__name__}"
        )

    # Validate with Pydantic
    try:
        manifest = PatchSetManifest.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Patch-set manifest validation failed for {path}: {e}") from e

    # Validate workflow names (warn only, don't fail for forward compatibility)
    for workflow in manifest.patches:
        _validate_workflow_name(workflow, manifest.name, "patches")
    for workflow in manifest.workflow_overrides:
        _validate_workflow_name(workflow, manifest.name, "workflow_overrides")

    # Validate paths if requested
    if validate_paths:
        manifest_dir = path.parent

        # Validate patch file paths
        for workflow, patch_path in manifest.patches.items():
            if patch_path is None:
                continue  # null = no patch, skip validation
            resolved_path = _resolve_patch_path(patch_path, manifest_dir)
            if not resolved_path.exists():
                raise ConfigError(
                    f"Patch file for workflow '{workflow}' does not exist: "
                    f"{resolved_path} (manifest: {path})"
                )
            if not resolved_path.is_file():
                raise ConfigError(
                    f"Patch path for workflow '{workflow}' is not a file: "
                    f"{resolved_path} (manifest: {path})"
                )

        # Validate workflow_override directory paths
        for workflow, override_path in manifest.workflow_overrides.items():
            resolved_path = _resolve_patch_path(override_path, manifest_dir)
            if not resolved_path.exists():
                raise ConfigError(
                    f"Workflow override directory for '{workflow}' does not exist: "
                    f"{resolved_path} (manifest: {path})"
                )
            if not resolved_path.is_dir():
                raise ConfigError(
                    f"Workflow override path for '{workflow}' is not a directory: "
                    f"{resolved_path} (manifest: {path})"
                )

    return manifest


class PatchSetRegistry:
    """Registry for discovering and accessing patch-set manifests.

    Provides discovery of patch-set manifests in a directory and
    cached access to manifest instances.

    Usage:
        registry = PatchSetRegistry(Path("experiments/patch-sets"))
        names = registry.list()
        manifest = registry.get("baseline")

    """

    def __init__(
        self,
        patchsets_dir: Path,
        project_root: Path | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            patchsets_dir: Directory containing patch-set manifest YAML files.
            project_root: Project root for ${project} variable resolution.

        """
        self._patchsets_dir = patchsets_dir
        self._project_root = project_root
        self._manifests: dict[str, Path] = {}
        self._discovered = False

    def _ensure_discovered(self) -> None:
        """Ensure manifests have been discovered."""
        if not self._discovered:
            self._manifests = self.discover(self._patchsets_dir)
            self._discovered = True

    def discover(self, patchsets_dir: Path) -> dict[str, Path]:
        """Discover patch-set manifests in a directory.

        Scans for *.yaml files (not *.yml) and validates that each file's
        internal 'name' field matches the filename stem.

        Args:
            patchsets_dir: Directory to scan for manifests.

        Returns:
            Mapping of manifest name to file path.

        """
        if not patchsets_dir.exists():
            logger.info(
                "Patch-set manifests directory does not exist: %s",
                patchsets_dir,
            )
            return {}

        if not patchsets_dir.is_dir():
            logger.warning(
                "Patch-set manifests path is not a directory: %s",
                patchsets_dir,
            )
            return {}

        manifests: dict[str, Path] = {}

        for yaml_file in patchsets_dir.glob("*.yaml"):
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

                manifests[expected_name] = yaml_file

            except yaml.YAMLError as e:
                logger.warning("Skipping %s: invalid YAML: %s", yaml_file, e)
                continue
            except OSError as e:
                logger.warning("Skipping %s: cannot read: %s", yaml_file, e)
                continue

        return manifests

    def list(self) -> list[str]:
        """Return list of available patch-set manifest names.

        Returns:
            Sorted list of manifest names.

        """
        self._ensure_discovered()
        return sorted(self._manifests.keys())

    def get(self, name: str, validate_paths: bool = True) -> PatchSetManifest:
        """Get a patch-set manifest by name.

        Args:
            name: Manifest name (filename stem without .yaml).
            validate_paths: If True, validate that patch files exist.

        Returns:
            Loaded and validated PatchSetManifest.

        Raises:
            ConfigError: If manifest not found.

        """
        self._ensure_discovered()

        if name not in self._manifests:
            available = ", ".join(sorted(self._manifests.keys())) or "(none)"
            raise ConfigError(f"Patch-set manifest '{name}' not found. Available: {available}")

        return self._load_manifest(name, validate_paths)

    @lru_cache(maxsize=32)  # noqa: B019
    def _load_manifest(
        self,
        name: str,
        validate_paths: bool,
    ) -> PatchSetManifest:
        """Load a manifest with caching.

        Args:
            name: Manifest name.
            validate_paths: Whether to validate paths.

        Returns:
            Loaded PatchSetManifest.

        """
        path = self._manifests[name]
        return load_patchset_manifest(path, self._project_root, validate_paths)
