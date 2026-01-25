"""Config route utilities - security, provenance, and diff helpers.

This package provides utilities for config route handlers:
- security: Dangerous field filtering and schema introspection
- provenance: Config source tracking and editor creation
- diff: Config difference calculation
"""

import asyncio
from typing import Any

# Re-export all utilities for backward compatibility
from .diff import _calculate_diff, _find_risky_fields_in_diff
from .provenance import (
    _add_provenance_to_raw,
    _create_config_editor,
    _strip_provenance,
    _validate_path_exists,
)
from .security import (
    _build_full_schema,
    _check_risky_fields,
    _filter_dangerous_fields,
    _filter_dangerous_fields_for_export,
    _filter_dangerous_for_export_recursive,
    _filter_dangerous_recursive,
    _find_dangerous_fields,
    _get_field_security,
    _get_full_schema,
)

# Shared state (singleton per process - Python caches module imports)
_config_editor_lock = asyncio.Lock()
_full_schema_cache: dict[str, Any] | None = None

# Maximum import content size (100KB) - used by import_export.py
IMPORT_MAX_SIZE = 100 * 1024

__all__ = [
    # Shared state
    "_config_editor_lock",
    "_full_schema_cache",
    "IMPORT_MAX_SIZE",
    # Security
    "_get_full_schema",
    "_build_full_schema",
    "_filter_dangerous_fields",
    "_filter_dangerous_recursive",
    "_check_risky_fields",
    "_get_field_security",
    "_find_dangerous_fields",
    "_filter_dangerous_fields_for_export",
    "_filter_dangerous_for_export_recursive",
    # Provenance
    "_validate_path_exists",
    "_create_config_editor",
    "_add_provenance_to_raw",
    "_strip_provenance",
    # Diff
    "_calculate_diff",
    "_find_risky_fields_in_diff",
]
