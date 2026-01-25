"""Configuration editor with provenance tracking and backup management.

This module provides infrastructure for editing configuration files through
the dashboard UI, tracking where each value comes from (default/global/project),
and maintaining backup history.

Comment Preservation:
    ConfigEditor preserves YAML comments during save operations by updating
    values in-place in the original CommentedMap structure using ruamel.yaml
    (required dependency).

    Use `has_ruamel_yaml()` to check availability (always True since required),
    and `editor.comments_preserved(scope)` to check if comments will be
    preserved for a specific save operation.

Usage:
    from bmad_assist.core.config_editor import ConfigEditor, has_ruamel_yaml

    # Context manager usage (recommended)
    with ConfigEditor(global_path, project_path) as editor:
        merged = editor.get_merged_with_provenance()
        editor.update("global", "benchmarking.enabled", False)
        editor.validate()  # Raises ConfigError if invalid
        editor.save("global")  # Comments preserved via ruamel.yaml

    # Manual usage
    editor = ConfigEditor(global_path, project_path)
    editor.load()
    # ... use editor
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefinedType

from bmad_assist.core.exceptions import ConfigError, ConfigValidationError

if TYPE_CHECKING:
    from ruamel.yaml.comments import CommentedMap

    from bmad_assist.core.config import Config

logger = logging.getLogger(__name__)

# Maximum number of backup versions to keep
MAX_BACKUPS = 5

# =============================================================================
# ruamel.yaml Verification (required dependency)
# =============================================================================


def _verify_ruamel() -> None:
    """Verify ruamel.yaml is installed (required dependency).

    Raises:
        ImportError: If ruamel.yaml is not available.

    """
    try:
        from ruamel.yaml import YAML  # noqa: F401
        from ruamel.yaml.comments import CommentedMap  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ruamel.yaml is required for YAML comment preservation. "
            "Install with: pip install 'ruamel.yaml>=0.18.0,<1.0.0'"
        ) from e


# Verify at module load time - fail fast if not available
_verify_ruamel()


def has_ruamel_yaml() -> bool:
    """Check if ruamel.yaml is available for comment preservation.

    This is a public API function that can be used by REST API endpoints
    to inform the UI whether comments will be preserved during save.

    Returns:
        True (always, since ruamel.yaml is now a required dependency).

    Note:
        This function is kept for backwards compatibility.
        Since ruamel.yaml is now required, it always returns True.

    """
    return True


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dictionary.

    Rules:
    - Dicts are merged recursively
    - Lists are replaced (NOT merged/appended)
    - Scalar values are replaced by override
    - Keys only in base are preserved
    - Keys only in override are added

    Args:
        base: Base configuration dictionary.
        override: Override dictionary with higher priority.

    Returns:
        Merged dictionary (new dict, does not modify inputs).

    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get_pydantic_defaults(model: type[BaseModel]) -> dict[str, Any]:
    """Extract default values from a Pydantic model recursively.

    Args:
        model: Pydantic model class.

    Returns:
        Dictionary of default values with nested models expanded.

    """
    defaults: dict[str, Any] = {}
    for name, field_info in model.model_fields.items():
        # Handle default value
        default_val = field_info.default

        if isinstance(default_val, PydanticUndefinedType):
            # Required field, no default - skip
            continue

        # Check if this is a nested model
        annotation = field_info.annotation

        # Handle Optional types (e.g., NotificationConfig | None)
        if hasattr(annotation, "__origin__"):
            from typing import get_args, get_origin

            origin = get_origin(annotation)
            # For Union/Optional types, get the non-None type
            if origin is not None:
                args = get_args(annotation)
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

        # Check if annotation is a BaseModel subclass
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            # Nested model - recurse to get its defaults
            if default_val is None:
                defaults[name] = None
            else:
                defaults[name] = _get_pydantic_defaults(annotation)
        elif field_info.default_factory is not None:
            # Default factory - call it to get default value
            try:
                factory = field_info.default_factory
                # factory is a callable that returns the default value
                defaults[name] = factory()  # type: ignore[call-arg]
            except Exception:
                # If factory fails, skip this default
                pass
        else:
            # Regular scalar default (including None)
            defaults[name] = default_val

    return defaults


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dictionary to dot-notation keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys (used in recursion).
        sep: Separator for nested keys.

    Returns:
        Flattened dictionary with dot-notation keys.

    Example:
        >>> _flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}

    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and v:
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_nested_value(d: dict[str, Any], path: str) -> tuple[Any, bool]:
    """Get value at dot-notation path from nested dict.

    Args:
        d: Dictionary to search.
        path: Dot-notation path (e.g., "a.b.c").

    Returns:
        Tuple of (value, found). If path not found, returns (None, False).

    """
    keys = path.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return None, False
        if key not in current:
            return None, False
        current = current[key]
    return current, True


def _set_nested_value(d: dict[str, Any], path: str, value: Any) -> None:
    """Set value at dot-notation path in nested dict, creating intermediate dicts.

    Args:
        d: Dictionary to modify.
        path: Dot-notation path (e.g., "a.b.c").
        value: Value to set.

    Raises:
        ValueError: If path is empty, has empty segments, or intermediate
            value exists but is not a dict.

    """
    if not path:
        raise ValueError("Path cannot be empty")

    keys = path.split(".")

    # Validate no empty segments
    for key in keys:
        if not key:
            raise ValueError(f"Invalid path '{path}': contains empty segment")

    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise ValueError(
                f"Cannot set '{path}': intermediate value at '{key}' "
                f"is {type(current[key]).__name__}, not dict"
            )
        current = current[key]

    current[keys[-1]] = value


def _delete_nested_value(d: dict[str, Any], path: str) -> bool:
    """Delete value at dot-notation path from nested dict.

    Args:
        d: Dictionary to modify.
        path: Dot-notation path (e.g., "a.b.c").

    Returns:
        True if value was deleted, False if path didn't exist.

    Raises:
        ValueError: If path is empty or has empty segments.

    """
    if not path:
        raise ValueError("Path cannot be empty")

    keys = path.split(".")

    # Validate no empty segments
    for key in keys:
        if not key:
            raise ValueError(f"Invalid path '{path}': contains empty segment")

    current = d
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]

    if not isinstance(current, dict) or keys[-1] not in current:
        return False

    del current[keys[-1]]
    return True


@dataclass
class ProvenanceTracker:
    """Tracks the source of each configuration value.

    Source can be:
    - "default": Value comes from Pydantic model defaults
    - "global": Value defined in global config file
    - "project": Value defined in project config file

    Attributes:
        _provenance: Internal mapping of paths to sources.

    """

    _provenance: dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        defaults: dict[str, Any],
        global_cfg: dict[str, Any],
        project_cfg: dict[str, Any],
    ) -> None:
        """Build provenance tracking from config layers.

        Args:
            defaults: Pydantic model defaults.
            global_cfg: Global config raw data.
            project_cfg: Project config raw data.

        """
        self._provenance = {}
        self._build_provenance(defaults, global_cfg, project_cfg)

    def _build_provenance(
        self,
        defaults: dict[str, Any],
        global_cfg: dict[str, Any],
        project_cfg: dict[str, Any],
    ) -> None:
        """Build provenance map from config layers.

        Priority (highest wins): project > global > default
        """
        # Flatten all configs to dot-notation
        flat_defaults = _flatten_dict(defaults)
        flat_global = _flatten_dict(global_cfg)
        flat_project = _flatten_dict(project_cfg)

        # Start with defaults
        for path in flat_defaults:
            self._provenance[path] = "default"

        # Override with global
        for path in flat_global:
            self._provenance[path] = "global"

        # Override with project
        for path in flat_project:
            self._provenance[path] = "project"

    def get_provenance(self, path: str) -> str:
        """Get source of a specific config path.

        Args:
            path: Dot-notation path (e.g., "benchmarking.enabled").

        Returns:
            Source string: "default", "global", or "project".

        """
        return self._provenance.get(path, "default")

    def get_all_provenance(self) -> dict[str, str]:
        """Get flat dict of all paths to their sources.

        Returns:
            Dictionary mapping all known paths to their sources.

        """
        return dict(self._provenance)


class ConfigEditor:
    """Edit configuration YAML files with provenance tracking.

    This class provides:
    - Loading raw YAML configs (not Pydantic models until validation)
    - Tracking provenance (where each value comes from)
    - Backup rotation before writes
    - Atomic file writes
    - Comment preservation when ruamel.yaml is available

    Thread Safety:
        ConfigEditor is NOT thread-safe. Dashboard API routes must use
        external locking for concurrent access.

    Attributes:
        global_path: Path to global config file.
        project_path: Path to project config file (or None).
        MAX_BACKUPS: Maximum backup versions to keep (class constant).

    """

    MAX_BACKUPS = MAX_BACKUPS

    def __init__(
        self,
        global_path: Path,
        project_path: Path | None = None,
    ) -> None:
        """Initialize ConfigEditor.

        Args:
            global_path: Path to global configuration file.
            project_path: Path to project configuration file (or None).

        """
        self.global_path = global_path
        self.project_path = project_path

        self._global_data: dict[str, Any] = {}
        self._project_data: dict[str, Any] = {}
        self._provenance: ProvenanceTracker | None = None

        # Comment preservation: store original CommentedMaps when ruamel available
        self._global_commented: CommentedMap | None = None
        self._project_commented: CommentedMap | None = None

    def __enter__(self) -> ConfigEditor:
        """Context manager entry: load configs."""
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit: cleanup (no auto-save)."""
        # No auto-save on exit - explicit save() call required
        pass

    def _get_max_config_size(self) -> int:
        """Get the maximum config file size from config module.

        Returns:
            Maximum config file size in bytes.

        """
        # Import here to avoid circular imports
        from bmad_assist.core.config import MAX_CONFIG_SIZE

        return MAX_CONFIG_SIZE

    def load(self) -> None:
        """Load raw YAML configs and build provenance tracker.

        When ruamel.yaml is available, also stores the original CommentedMap
        structures to enable comment preservation during save.

        Raises:
            ConfigError: If global config is missing or invalid.

        """
        max_size = self._get_max_config_size()

        # Load global config (required) with optional comment preservation
        self._global_data, self._global_commented = self._load_yaml_with_comments(
            self.global_path, max_size
        )

        # Load project config (optional) with optional comment preservation
        if self.project_path is not None and self.project_path.exists():
            self._project_data, self._project_commented = self._load_yaml_with_comments(
                self.project_path, max_size
            )
        else:
            self._project_data = {}
            self._project_commented = None

        # Build provenance tracker
        self._rebuild_provenance()

    def _load_yaml_file(self, path: Path, max_size: int) -> dict[str, Any]:
        """Load and parse a YAML file with safety checks.

        Args:
            path: Path to YAML file.
            max_size: Maximum file size in bytes.

        Returns:
            Parsed YAML content as dictionary.

        Raises:
            ConfigError: If file cannot be read, is too large, is empty,
                is a directory, or YAML is invalid.

        """
        try:
            # Read with size limit
            with path.open("r", encoding="utf-8") as f:
                content = f.read(max_size + 1)

            if len(content) > max_size:
                raise ConfigError(
                    f"Config file {path} exceeds size limit "
                    f"(read {len(content):,} bytes before stopping)."
                )

            parsed = yaml.safe_load(content)

            # Empty file detection
            if parsed is None:
                raise ConfigError(f"Config file {path} is empty or contains only whitespace.")

            if not isinstance(parsed, dict):
                raise ConfigError(
                    f"Config file {path} must contain a YAML mapping, got {type(parsed).__name__}."
                )

            return parsed
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}") from e
        except IsADirectoryError as e:
            raise ConfigError(f"{path} is a directory, not a config file.") from e
        except PermissionError as e:
            raise ConfigError(f"Permission denied reading {path}: {e}") from e
        except FileNotFoundError as e:
            raise ConfigError(f"Config file not found: {path}") from e
        except OSError as e:
            raise ConfigError(f"Cannot read config file {path}: {e}") from e

    def _load_yaml_with_comments(
        self, path: Path, max_size: int
    ) -> tuple[dict[str, Any], CommentedMap | None]:
        """Load YAML file with optional comment preservation using ruamel.yaml.

        When ruamel.yaml is available, loads in round-trip mode to preserve
        comments. Falls back to PyYAML when ruamel unavailable or parse fails.

        Args:
            path: Path to YAML file.
            max_size: Maximum file size in bytes.

        Returns:
            Tuple of (data_dict, commented_map).
            - data_dict: Parsed YAML content as regular dict
            - commented_map: Original CommentedMap if ruamel available, else None

        Raises:
            ConfigError: If file cannot be read, is too large, is empty,
                is a directory, or YAML is invalid.

        """
        # First check if file exists and is not empty for edge cases
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        if path.is_dir():
            raise ConfigError(f"{path} is a directory, not a config file.")

        # Check file size
        try:
            file_size = path.stat().st_size
        except OSError as e:
            raise ConfigError(f"Cannot stat config file {path}: {e}") from e

        if file_size == 0:
            raise ConfigError(f"Config file {path} is empty or contains only whitespace.")

        if file_size > max_size:
            raise ConfigError(
                f"Config file {path} exceeds size limit ({file_size:,} bytes > {max_size:,} limit)."
            )

        # Try ruamel.yaml first for comment preservation
        if has_ruamel_yaml():
            try:
                from ruamel.yaml import YAML
                from ruamel.yaml.comments import CommentedMap

                yaml_rt = YAML(typ="rt")
                yaml_rt.preserve_quotes = True

                with path.open("r", encoding="utf-8") as f:
                    data = yaml_rt.load(f)

                # Handle empty file (ruamel returns None)
                if data is None:
                    raise ConfigError(f"Config file {path} is empty or contains only whitespace.")

                if not isinstance(data, (dict, CommentedMap)):
                    raise ConfigError(
                        f"Config file {path} must contain a YAML mapping, "
                        f"got {type(data).__name__}."
                    )

                # Convert CommentedMap to dict for provenance tracking
                # (provenance uses regular dicts)
                data_dict = dict(data)
                # Deep convert nested CommentedMaps
                data_dict = self._deep_convert_to_dict(data_dict)

                # Return original CommentedMap for comment preservation during save
                if isinstance(data, CommentedMap):
                    return data_dict, data
                return data_dict, None

            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML in {path}: {e}") from e
            except Exception as e:
                # ruamel parse failed - fall back to PyYAML with warning
                logger.warning(
                    "ruamel.yaml failed to parse %s, falling back to PyYAML: %s",
                    path,
                    e,
                )
                # Continue to PyYAML fallback below

        # PyYAML fallback (or ruamel unavailable)
        data_dict = self._load_yaml_file(path, max_size)
        return data_dict, None

    def _deep_convert_to_dict(self, obj: Any) -> Any:
        """Recursively convert CommentedMaps to regular dicts.

        Args:
            obj: Object to convert (may be dict, list, or scalar).

        Returns:
            Converted object with all CommentedMaps as regular dicts.

        """
        if isinstance(obj, dict):
            return {k: self._deep_convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_convert_to_dict(item) for item in obj]
        return obj

    def _rebuild_provenance(self) -> None:
        """Rebuild provenance tracker from current data."""
        # Import here to avoid circular imports
        from bmad_assist.core.config import Config

        defaults = _get_pydantic_defaults(Config)
        self._provenance = ProvenanceTracker(
            defaults=defaults,
            global_cfg=self._global_data,
            project_cfg=self._project_data,
        )

    def get_provenance(self, path: str) -> str:
        """Get source of a specific config path.

        Args:
            path: Dot-notation path (e.g., "benchmarking.enabled").

        Returns:
            Source string: "default", "global", or "project".

        """
        if self._provenance is None:
            return "default"
        return self._provenance.get_provenance(path)

    def get_all_provenance(self) -> dict[str, str]:
        """Get flat dict of all paths to their sources.

        Returns:
            Dictionary mapping all known paths to their sources.

        """
        if self._provenance is None:
            return {}
        return self._provenance.get_all_provenance()

    def get_merged_with_provenance(self) -> dict[str, Any]:
        """Get merged config with source info for each value.

        Returns:
            Nested structure where scalar values are wrapped in
            {"value": X, "source": Y}.

        Example:
            {
                "benchmarking": {
                    "enabled": {"value": True, "source": "default"}
                }
            }

        """
        # Import here to avoid circular imports
        from bmad_assist.core.config import Config

        defaults = _get_pydantic_defaults(Config)
        merged = _deep_merge(defaults, self._global_data)
        merged = _deep_merge(merged, self._project_data)

        return self._add_provenance_to_dict(merged, "")

    def _add_provenance_to_dict(
        self,
        d: dict[str, Any],
        prefix: str,
    ) -> dict[str, Any]:
        """Recursively add provenance info to dict values.

        Args:
            d: Dictionary to process.
            prefix: Current path prefix for provenance lookup.

        Returns:
            Dictionary with values wrapped in {"value": X, "source": Y}.

        """
        result: dict[str, Any] = {}

        for key, value in d.items():
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict) and value:
                # Recurse into nested dict
                result[key] = self._add_provenance_to_dict(value, path)
            else:
                # Wrap scalar/list value with provenance
                source = self.get_provenance(path)
                result[key] = {"value": value, "source": source}

        return result

    def get_global_raw(self) -> dict[str, Any]:
        """Get copy of raw global config data.

        Returns:
            Deep copy of global configuration dictionary.

        """
        return copy.deepcopy(self._global_data)

    def get_project_raw(self) -> dict[str, Any]:
        """Get copy of raw project config data.

        Returns:
            Deep copy of project configuration dictionary.

        """
        return copy.deepcopy(self._project_data)

    def comments_preserved(self, scope: str = "project") -> bool:
        """Check if comments will be preserved when saving the specified scope.

        This method is intended for UI indicators, allowing the dashboard to
        display "Comments will be preserved" or "Comments will be lost".

        Returns True if:
        - ruamel.yaml is available AND
        - CommentedMap exists for the specified scope (file was loaded)

        Returns False if:
        - ruamel.yaml is unavailable OR
        - File was loaded via PyYAML fallback OR
        - File doesn't exist (new file)

        Note: Returns True even if the file has no comments - this indicates
        the capability to preserve comments, not whether comments exist.

        Args:
            scope: "global" or "project" (default: "project").

        Returns:
            True if comments will be preserved during save for this scope.

        Example:
            >>> editor = ConfigEditor(global_path, project_path)
            >>> editor.load()
            >>> if editor.comments_preserved("global"):
            ...     print("Global config comments will be preserved")

        """
        if not has_ruamel_yaml():
            return False

        if scope == "global":
            return self._global_commented is not None
        elif scope == "project":
            return self._project_commented is not None
        else:
            return False

    def update(self, scope: str, path: str, value: Any) -> None:
        """Update a config value in the specified scope.

        Args:
            scope: "global" or "project".
            path: Dot-notation path (e.g., "benchmarking.enabled").
            value: Value to set.

        Raises:
            ValueError: If scope is invalid, path format is invalid,
                or scope=project but no project path configured.

        """
        if scope not in ("global", "project"):
            raise ValueError(f"scope must be 'global' or 'project', got '{scope}'")

        if scope == "project" and self.project_path is None:
            raise ValueError("Cannot update project config: no project path configured")

        data = self._global_data if scope == "global" else self._project_data
        _set_nested_value(data, path, value)

    def remove(self, scope: str, path: str) -> None:
        """Remove a config override at the specified path.

        This enables "Reset to global" or "Reset to default" operations.

        Args:
            scope: "global" or "project".
            path: Dot-notation path (e.g., "benchmarking.enabled").

        Raises:
            ValueError: If scope is invalid or scope=project but no project path.

        """
        if scope not in ("global", "project"):
            raise ValueError(f"scope must be 'global' or 'project', got '{scope}'")

        if scope == "project" and self.project_path is None:
            raise ValueError("Cannot remove from project config: no project path configured")

        data = self._global_data if scope == "global" else self._project_data
        _delete_nested_value(data, path)

    def validate(self) -> Config:
        """Validate merged config against Pydantic models.

        Returns:
            Validated Config instance.

        Raises:
            ConfigValidationError: If Pydantic validation fails (with structured details).
            ConfigError: If other config errors occur.

        """
        # Import here to avoid circular imports
        from pydantic import ValidationError

        from bmad_assist.core.config import Config

        # Get defaults and merge with current data
        defaults = _get_pydantic_defaults(Config)
        merged = _deep_merge(defaults, self._global_data)
        merged = _deep_merge(merged, self._project_data)

        try:
            return Config.model_validate(merged)
        except ValidationError as e:
            # Extract structured error details from Pydantic
            errors = [
                {"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in e.errors()
            ]
            raise ConfigValidationError("Configuration validation failed", errors) from e

    def _rotate_backups(self, path: Path) -> None:
        """Rotate backup files before writing.

        Creates backups as: config.yaml.1 (newest) to config.yaml.5 (oldest).

        Args:
            path: Path to the file being backed up.

        """
        if not path.exists():
            return

        try:
            # Delete oldest backup
            oldest = Path(f"{path}.{self.MAX_BACKUPS}")
            if oldest.exists():
                oldest.unlink()

            # Shift existing backups
            for i in range(self.MAX_BACKUPS - 1, 0, -1):
                current = Path(f"{path}.{i}")
                next_backup = Path(f"{path}.{i + 1}")
                if current.exists():
                    current.rename(next_backup)

            # Copy current file to .1
            newest = Path(f"{path}.1")
            shutil.copy2(path, newest)

            # Preserve permissions
            try:
                mode = path.stat().st_mode
                os.chmod(newest, mode)
            except OSError as e:
                logger.warning("Failed to preserve permissions on backup %s: %s", newest, e)

        except OSError as e:
            logger.warning("Backup rotation failed for %s: %s. Continuing with save.", path, e)

    def save(self, scope: str) -> None:
        """Save config with validation, backup rotation, and comment preservation.

        When ruamel.yaml is available and we have the original CommentedMap,
        updates are made in-place to preserve YAML comments. Otherwise, falls
        back to PyYAML (comments will be lost).

        Args:
            scope: "global" or "project".

        Raises:
            ValueError: If scope is invalid or scope=project but no project path.
            ConfigError: If validation fails.

        """
        if scope not in ("global", "project"):
            raise ValueError(f"scope must be 'global' or 'project', got '{scope}'")

        if scope == "project" and self.project_path is None:
            raise ValueError("Cannot save project config: no project path configured")

        # Validate before writing
        self.validate()

        # Determine target path
        path = self.global_path if scope == "global" else self.project_path
        assert path is not None  # For type checker

        # Rotate backups
        self._rotate_backups(path)

        # Get data to write
        data = self._global_data if scope == "global" else self._project_data

        # Get original CommentedMap if available for comment preservation
        commented_map = self._global_commented if scope == "global" else self._project_commented

        # Try to save with ruamel.yaml for comment preservation
        if has_ruamel_yaml() and commented_map is not None:
            self._save_with_ruamel(path, data, commented_map)
        else:
            self._save_with_pyyaml(path, data)

        # Rebuild provenance after successful save
        self._rebuild_provenance()

        logger.info("Saved %s config to %s", scope, path)

    def _save_with_ruamel(
        self,
        path: Path,
        data: dict[str, Any],
        commented_map: CommentedMap,
    ) -> None:
        """Save config using ruamel.yaml with comment preservation.

        Updates values in-place in the original CommentedMap to preserve
        comment associations.

        Args:
            path: Target file path.
            data: Updated config data as regular dict.
            commented_map: Original CommentedMap from load.

        Raises:
            OSError: If write fails.

        """
        from ruamel.yaml import YAML

        # Update CommentedMap in-place to preserve comments
        self._update_commented_map_recursive(commented_map, data)

        # Atomic write using temp file
        temp_path = Path(f"{path}.tmp.{os.getpid()}")
        try:
            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Capture original permissions if file exists, else use secure default
            orig_mode = path.stat().st_mode if path.exists() else 0o600

            # Configure ruamel.yaml for round-trip mode
            yaml_rt = YAML(typ="rt")
            yaml_rt.preserve_quotes = True
            yaml_rt.default_flow_style = False

            # Write to temp file
            with temp_path.open("w", encoding="utf-8") as f:
                yaml_rt.dump(commented_map, f)

            # Set permissions before rename
            os.chmod(temp_path, orig_mode & 0o777)

            # Atomic rename
            temp_path.rename(path)

            logger.debug("Saved config with ruamel.yaml (comments preserved)")

        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            raise

    def _save_with_pyyaml(self, path: Path, data: dict[str, Any]) -> None:
        """Save config using PyYAML (no comment preservation).

        Note: This is used for new files where no CommentedMap exists.
        Comments cannot be preserved for new files.

        Args:
            path: Target file path.
            data: Config data to write.

        Raises:
            OSError: If write fails.

        """
        # Atomic write using temp file
        temp_path = Path(f"{path}.tmp.{os.getpid()}")
        try:
            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Capture original permissions if file exists, else use secure default
            orig_mode = path.stat().st_mode if path.exists() else 0o600

            # Write to temp file
            with temp_path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            # Set permissions before rename
            os.chmod(temp_path, orig_mode & 0o777)

            # Atomic rename
            temp_path.rename(path)

            logger.debug("Saved config with PyYAML")

        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            raise

    def _update_commented_map_recursive(
        self,
        commented_map: CommentedMap,
        data: dict[str, Any],
    ) -> None:
        """Recursively update CommentedMap in-place to preserve comments.

        For existing keys: updates value in-place (preserves inline comments).
        For new keys: appends to end of parent dict (no comments to preserve).
        For removed keys: deletes from CommentedMap.
        For nested dicts: recursively updates in-place.
        For lists: replaces entire list (list item comments not preserved).

        Args:
            commented_map: Original CommentedMap to update in-place.
            data: New data as regular dict.

        """
        from ruamel.yaml.comments import CommentedMap

        # Get current and new keys
        old_keys = set(commented_map.keys())
        new_keys = set(data.keys())

        # Remove keys that are not in new data
        for key in old_keys - new_keys:
            del commented_map[key]

        # Update/add entries from new data
        for key, value in data.items():
            if key in commented_map:
                old_value = commented_map[key]
                # If both old and new are dicts, recurse to preserve nested comments
                if isinstance(old_value, (dict, CommentedMap)) and isinstance(value, dict):
                    if isinstance(old_value, CommentedMap):
                        self._update_commented_map_recursive(old_value, value)
                    else:
                        # Old value was regular dict, replace it
                        commented_map[key] = value
                else:
                    # Replace value (preserves inline comments for this key)
                    commented_map[key] = value
            else:
                # New key - just add it (no comments to preserve)
                commented_map[key] = value

    def list_backups(self, scope: str) -> list[dict[str, Any]]:
        """List available backup versions.

        Args:
            scope: "global" or "project".

        Returns:
            List of backup metadata dicts, sorted by version (1 = newest).
            Each dict has: version, path, modified (ISO timestamp).

        """
        if scope not in ("global", "project"):
            raise ValueError(f"scope must be 'global' or 'project', got '{scope}'")

        if scope == "project" and self.project_path is None:
            return []

        path = self.global_path if scope == "global" else self.project_path
        assert path is not None

        backups: list[dict[str, Any]] = []

        for i in range(1, self.MAX_BACKUPS + 1):
            backup_path = Path(f"{path}.{i}")
            if backup_path.exists():
                try:
                    mtime = backup_path.stat().st_mtime
                    modified = datetime.fromtimestamp(mtime).isoformat()
                except OSError:
                    modified = "unknown"

                backups.append(
                    {
                        "version": i,
                        "path": str(backup_path),
                        "modified": modified,
                    }
                )

        return backups

    def restore_backup(self, scope: str, version: int) -> None:
        """Restore config from a backup version.

        Safe restore flow:
        1. Read backup file content
        2. Validate content with Pydantic
        3. If invalid: raise ConfigError without modifying main config
        4. If valid: backup current state, write backup content

        Args:
            scope: "global" or "project".
            version: Backup version to restore (1 = newest).

        Raises:
            ValueError: If scope is invalid or scope=project but no project path.
            ConfigError: If version doesn't exist or backup content is invalid.

        """
        if scope not in ("global", "project"):
            raise ValueError(f"scope must be 'global' or 'project', got '{scope}'")

        if scope == "project" and self.project_path is None:
            raise ValueError("Cannot restore project config: no project path configured")

        path = self.global_path if scope == "global" else self.project_path
        assert path is not None

        backup_path = Path(f"{path}.{version}")
        if not backup_path.exists():
            raise ConfigError(f"Backup version {version} does not exist: {backup_path}")

        # Import here to avoid circular imports
        from bmad_assist.core.config import MAX_CONFIG_SIZE

        # Read backup content - we read the file content NOW because _rotate_backups
        # will delete the oldest backup (version 5) before we can copy it.
        # Story 17.10 synthesis fix: read file content into memory before rotation.
        backup_data = self._load_yaml_file(backup_path, MAX_CONFIG_SIZE)
        backup_content = backup_path.read_text(encoding="utf-8")

        # Validate backup content before applying
        # Temporarily swap in backup data to validate
        original_data = self._global_data if scope == "global" else self._project_data

        if scope == "global":
            self._global_data = backup_data
        else:
            self._project_data = backup_data

        try:
            self.validate()
        except ConfigError:
            # Restore original data and re-raise
            if scope == "global":
                self._global_data = original_data
            else:
                self._project_data = original_data
            raise

        # Validation passed - create backup of current state before overwriting
        # Note: _rotate_backups shifts all backup versions up by 1, and may delete
        # the oldest backup (version 5), so we use the pre-read backup_content.
        self._rotate_backups(path)

        # Capture original permissions if file exists, else use secure default
        orig_mode = path.stat().st_mode if path.exists() else 0o600

        # Restore backup using in-memory content to preserve comments
        # Story 17.10 synthesis fix: write from memory instead of copying file
        # because _rotate_backups may have deleted the oldest backup.
        temp_path = Path(f"{path}.tmp.{os.getpid()}")
        try:
            temp_path.write_text(backup_content, encoding="utf-8")

            # Set permissions before rename
            os.chmod(temp_path, orig_mode & 0o777)

            temp_path.rename(path)

            # Reload the restored scope to pick up any comments from backup
            max_size = self._get_max_config_size()
            if scope == "global":
                self._global_data, self._global_commented = self._load_yaml_with_comments(
                    self.global_path, max_size
                )
            else:
                # scope == "project"
                assert self.project_path is not None
                self._project_data, self._project_commented = self._load_yaml_with_comments(
                    self.project_path, max_size
                )

            # Rebuild provenance after successful restore
            self._rebuild_provenance()

            logger.info("Restored %s config from backup version %d", scope, version)

        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            raise
