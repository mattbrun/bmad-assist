"""Small utility functions for the loop runner.

These are pure functions with no side effects, easy to test in isolation.
Extracted from runner.py as part of the runner refactoring (Story standalone-03).

"""

from __future__ import annotations

from pathlib import Path

from bmad_assist.core.state import State

__all__ = ["_count_epic_stories", "_get_story_title"]

# Type alias for state parameter
LoopState = State


def _count_epic_stories(state: LoopState) -> int:
    """Count completed stories belonging to the current epic only.

    Stories in completed_stories have format like "1.1", "2.5", "testarch.3".
    This function filters to count only those matching current_epic.

    Args:
        state: Current loop state with completed_stories and current_epic.

    Returns:
        Count of stories completed in the current epic (0 if none).

    """
    if not state.completed_stories or state.current_epic is None:
        return 0

    epic_prefix = f"{state.current_epic}."
    return sum(1 for story in state.completed_stories if story.startswith(epic_prefix))


def _get_story_title(project_path: Path, story_id: str) -> str | None:  # noqa: ARG001
    """Get human-readable story title from sprint-status or story key.

    Tries to extract story title from:
    1. Sprint-status entries (e.g., "2-1-css-design-tokens" -> "CSS Design Tokens")
    2. Story key slug (fallback)

    Args:
        project_path: Project root path (reserved for future use).
        story_id: Story identifier (e.g., "2.1").

    Returns:
        Story title if found, None otherwise.

    """
    try:
        from bmad_assist.core.paths import get_paths
        from bmad_assist.sprint.parser import parse_sprint_status

        # Load sprint-status to find story entry with title
        sprint_path = get_paths().sprint_status_file
        if not sprint_path.exists():
            return None

        sprint_data = parse_sprint_status(sprint_path)

        # Find entry matching this story ID (e.g., "2.1" matches "2-1-css-design-tokens")
        # Story ID format: "X.Y" -> key prefix "X-Y-"
        story_parts = story_id.split(".")
        if len(story_parts) == 2:
            key_prefix = f"{story_parts[0]}-{story_parts[1]}-"
            for entry in sprint_data.entries.values():
                if entry.key.startswith(key_prefix):
                    # Extract title from key: "2-1-css-design-tokens" -> "css design tokens" -> "CSS Design Tokens" # noqa: E501
                    title_slug = entry.key[len(key_prefix) :]
                    if title_slug:
                        # Convert slug to title: kebab-case -> Title Case
                        return title_slug.replace("-", " ").title()
    except Exception:
        pass

    return None
