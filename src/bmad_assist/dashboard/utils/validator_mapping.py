"""Validator mapping resolution utilities.

Story 23.8: Resolves anonymous validator IDs (Validator A, B, C...) to actual
model names by loading and parsing mapping files from .bmad-assist/cache/.

This module adapts logic from experiments/prepare.py for dashboard use.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


def load_all_mappings(project_root: Path) -> dict[str, dict[str, Any]]:
    """Load all mapping files (validation and code-review) from cache.

    Loads both validation-mapping-*.json and code-review-mapping-*.json files,
    indexing them by session_id for fast lookup.

    Args:
        project_root: Project root directory.

    Returns:
        Dict mapping session_id to mapping data. Each entry contains:
        - session_id: str
        - timestamp: str
        - mapping: dict[str, dict] (Validator A -> entry data)
        - type: str ("validation" or "code-review")

    """
    cache_dir = project_root / ".bmad-assist" / "cache"
    if not cache_dir.exists():
        return {}

    mappings: dict[str, dict[str, Any]] = {}

    # Load validation mappings
    for path in cache_dir.glob("validation-mapping-*.json"):
        data = _load_mapping_file(path)
        if data:
            session_id = data.get("session_id")
            if session_id:
                data["type"] = "validation"
                mappings[session_id] = data

    # Load code-review mappings
    for path in cache_dir.glob("code-review-mapping-*.json"):
        data = _load_mapping_file(path)
        if data:
            session_id = data.get("session_id")
            if session_id:
                data["type"] = "code-review"
                mappings[session_id] = data

    return mappings


def _load_mapping_file(path: Path) -> dict[str, Any] | None:
    """Load a single mapping JSON file.

    Args:
        path: Path to mapping file.

    Returns:
        Parsed JSON data or None on error.

    """
    try:
        with open(path) as f:
            return cast("dict[str, Any]", json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load mapping file %s: %s", path, e)
        return None


def resolve_model_name(mapping_entry: dict[str, Any]) -> str:
    """Extract human-readable model name from mapping entry.

    Priority: provider string parsing > model field.

    Provider string formats:
    - "gemini-gemini-3-flash-preview" -> "gemini-3-flash-preview"
    - "master-opus" -> "opus"
    - "claude-subprocess-glm-4.7" -> "glm-4.7"
    - "codex-gpt-4" -> "gpt-4"

    Args:
        mapping_entry: Single validator mapping entry with provider/model fields.

    Returns:
        Human-readable model name.

    """
    provider = mapping_entry.get("provider", "")
    model = mapping_entry.get("model", "unknown")

    if "-" in provider:
        parts = provider.split("-")
        if parts[0] in ("claude", "gemini", "codex", "master"):
            # Handle claude-subprocess-{model} format
            if parts[0] == "claude" and len(parts) > 1 and parts[1] == "subprocess":
                return "-".join(parts[2:]) if len(parts) > 2 else str(model)
            else:
                # For gemini-gemini-3-flash, master-opus, codex-gpt-4
                return "-".join(parts[1:])

    return str(model)


def find_mapping_by_session_id(
    session_id: str,
    mappings: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find mapping data by session_id.

    Args:
        session_id: Session ID from synthesis report frontmatter.
        mappings: Dict of all loaded mappings indexed by session_id.

    Returns:
        Mapping data dict or None if not found.

    """
    return mappings.get(session_id)


def build_validator_display_map(
    mapping: dict[str, Any],
) -> dict[str, str]:
    """Build validator ID to display name lookup.

    Creates a dictionary mapping "Validator A", "Validator B", etc. to
    actual model names with disambiguation for duplicate models.

    When multiple validators use the same model (e.g., two gemini instances),
    appends validator letter suffix: "gemini-3-flash (C)", "gemini-3-flash (E)".

    Args:
        mapping: Full mapping data with "mapping" field containing validator entries.

    Returns:
        Dict mapping validator IDs to display names.
        Example: {"Validator A": "gemini-3-flash-preview", "Validator C": "glm-4.7 (C)"}

    """
    validator_mapping = mapping.get("mapping", {})
    if not validator_mapping:
        return {}

    # First pass: resolve all model names
    resolved: dict[str, str] = {}
    for validator_id, entry in validator_mapping.items():
        resolved[validator_id] = resolve_model_name(entry)

    # Second pass: count model occurrences to detect duplicates
    model_counts: Counter[str] = Counter(resolved.values())

    # Third pass: add disambiguation suffix for duplicate models
    result: dict[str, str] = {}
    for validator_id, model_name in resolved.items():
        if model_counts[model_name] > 1:
            # Extract letter from "Validator X" format
            letter = validator_id.split()[-1] if " " in validator_id else validator_id
            result[validator_id] = f"{model_name} ({letter})"
        else:
            result[validator_id] = model_name

    return result


def get_mapping_for_story(
    project_root: Path,
    mapping_type: str,
    epic: str,
    story: str,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Get validator mapping for a specific story.

    If session_id is provided, looks up directly. Otherwise, searches for
    the most recent mapping matching the story pattern.

    Args:
        project_root: Project root directory.
        mapping_type: Type of mapping - "validation" or "code-review".
        epic: Epic identifier.
        story: Story number.
        session_id: Optional session_id for direct lookup.

    Returns:
        Dict with validators mapping and metadata, or None if not found.

    """
    mappings = load_all_mappings(project_root)

    if session_id:
        # Direct lookup by session_id
        mapping = find_mapping_by_session_id(session_id, mappings)
        if mapping:
            return {
                "session_id": mapping.get("session_id"),
                "timestamp": mapping.get("timestamp"),
                "validators": build_validator_display_map(mapping),
            }
        return None

    # Find most recent mapping of the specified type
    # This is a fallback when session_id is not available
    matching: list[dict[str, Any]] = []
    for data in mappings.values():
        if data.get("type") == mapping_type:
            matching.append(data)

    if not matching:
        return None

    # Sort by timestamp descending and return first
    matching.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    best = matching[0]

    return {
        "session_id": best.get("session_id"),
        "timestamp": best.get("timestamp"),
        "validators": build_validator_display_map(best),
    }
