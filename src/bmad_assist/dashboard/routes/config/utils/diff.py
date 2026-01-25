"""Diff utilities for config route handlers.

Provides config difference calculation.
"""

from typing import Any

from bmad_assist.core.config_editor import _flatten_dict

from .security import _get_field_security


def _calculate_diff(current: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Calculate diff between current and new config data.

    Uses _flatten_dict for comparison. Arrays are compared as atomic values
    (not element-by-element).

    Args:
        current: Current scope config data.
        new: New config data to import.

    Returns:
        Dict with added, modified, and removed fields.

    """
    flat_current = _flatten_dict(current) if current else {}
    flat_new = _flatten_dict(new)

    current_keys = set(flat_current.keys())
    new_keys = set(flat_new.keys())

    added = {k: flat_new[k] for k in (new_keys - current_keys)}
    removed = list(current_keys - new_keys)
    modified = {}

    for key in current_keys & new_keys:
        if flat_current[key] != flat_new[key]:
            modified[key] = {"old": flat_current[key], "new": flat_new[key]}

    return {"added": added, "modified": modified, "removed": removed}


def _find_risky_fields_in_diff(diff: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Find fields with security='risky' in diff result.

    Checks both added and modified fields (removed fields don't need confirmation).

    Args:
        diff: Diff result from _calculate_diff.
        schema: Full schema with security metadata.

    Returns:
        List of dot-notation paths to risky fields in the diff.

    """
    risky = []

    for path in diff.get("added", {}):
        if _get_field_security(path, schema) == "risky":
            risky.append(path)

    for path in diff.get("modified", {}):
        if _get_field_security(path, schema) == "risky":
            risky.append(path)

    return risky
