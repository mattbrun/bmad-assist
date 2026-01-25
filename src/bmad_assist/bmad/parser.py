"""BMAD markdown file parser with YAML frontmatter support.

This module provides deterministic parsing of BMAD markdown files,
extracting YAML frontmatter metadata and markdown content without LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter  # Library lacks type stubs

from bmad_assist.core.exceptions import ParserError
from bmad_assist.core.types import EpicId

logger = logging.getLogger(__name__)

# Regex patterns for epic file parsing
# Pattern: ## Story X.Y: Title (2+ hashes - supports ##, ###, ####, etc.)
STORY_HEADER_PATTERN = re.compile(
    r"^#{2,}\s+Story\s+(\d+)\.(\d+):\s+(.+)$",
    re.MULTILINE,
)

# Pattern: **Estimate:** 3 SP or **Story Points:** 3
ESTIMATE_PATTERN = re.compile(
    r"\*\*(?:Estimate|Story Points):\*\*\s*(\d+)",
    re.IGNORECASE,
)

# Pattern: **Status:** done OR **Status:** Ready for Review (captures multi-word)
STATUS_PATTERN = re.compile(
    r"\*\*Status:\*\*\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

# Pattern: **Dependencies:** Story 3.2, Story 3.4
DEPENDENCIES_PATTERN = re.compile(
    r"\*\*Dependencies:\*\*\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

# Pattern to extract story numbers from dependencies
STORY_NUMBER_PATTERN = re.compile(r"(\d+\.\d+)")

# Patterns for checkbox criteria
CHECKBOX_CHECKED_PATTERN = re.compile(r"-\s*\[x\]", re.IGNORECASE)
CHECKBOX_UNCHECKED_PATTERN = re.compile(r"-\s*\[\s*\]")

# Pattern for multiple epics in a file: # Epic N: Title or ## Epic N: Title
EPIC_HEADER_PATTERN = re.compile(
    r"^#{1,2}\s+Epic\s+\d+:",
    re.MULTILINE,
)

# Pattern to extract epic number and title from header: # Epic 16: Real-time Dashboard
EPIC_TITLE_PATTERN = re.compile(
    r"^#\s+Epic\s+(\d+|[a-zA-Z][\w-]*):\s*(.+)$",
    re.MULTILINE,
)


@dataclass
class BmadDocument:
    """Parsed BMAD document with frontmatter and content.

    Attributes:
        frontmatter: Dictionary of YAML frontmatter metadata.
        content: Markdown content after frontmatter.
        path: Original file path as string.

    """

    frontmatter: dict[str, Any]
    content: str
    path: str

    def __eq__(self, other: object) -> bool:
        """Compare BmadDocument objects based on their path."""
        if not isinstance(other, BmadDocument):
            return NotImplemented
        return self.path == other.path

    def __hash__(self) -> int:
        """Return the hash of the BmadDocument object based on its path."""
        return hash(self.path)


@dataclass
class EpicStory:
    """Represents a story extracted from an epic file.

    Attributes:
        number: Story number in format "X.Y" (e.g., "2.1").
        title: Story title.
        estimate: Story point estimate, or None if not specified.
        status: Story status from explicit **Status:** field, or None.
        dependencies: List of dependent story numbers (e.g., ["3.2", "3.4"]).
        completed_criteria: Count of checked acceptance criteria checkboxes.
        total_criteria: Total count of acceptance criteria checkboxes.

    """

    number: str
    title: str
    estimate: int | None = None
    status: str | None = None
    dependencies: list[str] = field(default_factory=list)
    completed_criteria: int | None = None
    total_criteria: int | None = None


@dataclass
class EpicDocument:
    """Represents a parsed epic file with stories.

    Attributes:
        epic_num: Epic ID from frontmatter (int or str), or None for multi-epic files.
        title: Epic title from frontmatter, or None for multi-epic files.
        status: Epic status from frontmatter, or None for multi-epic files.
        stories: List of stories extracted from the epic file.
        path: Original file path as string.

    """

    epic_num: EpicId | None
    title: str | None
    status: str | None
    stories: list[EpicStory]
    path: str


def parse_bmad_file(path: str | Path) -> BmadDocument:
    """Parse a BMAD markdown file with YAML frontmatter.

    Reads the file, extracts YAML frontmatter metadata, and separates
    the markdown content. Uses python-frontmatter library for robust
    parsing.

    Args:
        path: Path to the BMAD markdown file (string or Path object).

    Returns:
        BmadDocument with frontmatter dict, content string, and path.

    Raises:
        FileNotFoundError: If the file does not exist.
        ParserError: If YAML frontmatter is malformed or parsing fails.

    Examples:
        >>> doc = parse_bmad_file("docs/prd.md")
        >>> doc.frontmatter["status"]
        'complete'
        >>> doc.content[:20]
        '# Product Requirements'

    """
    path = Path(path)

    try:
        post = frontmatter.load(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ParserError(f"Failed to parse {path}: {e}") from e

    return BmadDocument(
        frontmatter=dict(post.metadata),
        content=post.content,
        path=str(path),
    )


def _extract_estimate(section: str) -> int | None:
    """Extract story point estimate from a story section.

    Args:
        section: The markdown content of a story section.

    Returns:
        The story point estimate as an integer, or None if not found.

    """
    match = ESTIMATE_PATTERN.search(section)
    if match:
        return int(match.group(1))
    return None


def _extract_status(section: str) -> str | None:
    """Extract explicit status from a story section.

    Args:
        section: The markdown content of a story section.

    Returns:
        The status string, or None if no explicit status is found.

    """
    match = STATUS_PATTERN.search(section)
    if match:
        return match.group(1).strip()
    return None


def _extract_dependencies(section: str) -> list[str]:
    """Extract dependency story numbers from a story section.

    Args:
        section: The markdown content of a story section.

    Returns:
        List of story numbers that this story depends on.

    """
    deps_match = DEPENDENCIES_PATTERN.search(section)
    if not deps_match:
        return []

    deps_text = deps_match.group(1)
    numbers = STORY_NUMBER_PATTERN.findall(deps_text)
    return numbers


def _count_criteria(section: str) -> tuple[int | None, int | None]:
    """Count checked and total acceptance criteria checkboxes.

    Args:
        section: The markdown content of a story section.

    Returns:
        Tuple of (completed_criteria, total_criteria), or (None, None)
        if no checkboxes are found.

    """
    checked = len(CHECKBOX_CHECKED_PATTERN.findall(section))
    unchecked = len(CHECKBOX_UNCHECKED_PATTERN.findall(section))
    total = checked + unchecked

    if total == 0:
        return None, None

    return checked, total


def _is_multi_epic_file(content: str) -> bool:
    """Check if content contains multiple epic headers.

    Args:
        content: The markdown content to check.

    Returns:
        True if multiple "# Epic N:" headers are found.

    """
    matches = EPIC_HEADER_PATTERN.findall(content)
    return len(matches) > 1


def _parse_story_sections(content: str) -> list[EpicStory]:
    """Extract stories from epic content.

    Splits content by story headers and parses each section.

    Args:
        content: The markdown content of an epic file.

    Returns:
        List of EpicStory objects extracted from the content.

    """
    matches = list(STORY_HEADER_PATTERN.finditer(content))

    if not matches:
        return []

    stories = []
    for i, match in enumerate(matches):
        try:
            # Extract section content (from this header to next or end)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section = content[start:end]

            epic_num, story_num, title = match.groups()
            number = f"{epic_num}.{story_num}"

            # Extract details from section
            estimate = _extract_estimate(section)
            status = _extract_status(section)
            dependencies = _extract_dependencies(section)
            completed, total = _count_criteria(section)

            stories.append(
                EpicStory(
                    number=number,
                    title=title.strip(),
                    estimate=estimate,
                    status=status,
                    dependencies=dependencies,
                    completed_criteria=completed,
                    total_criteria=total,
                )
            )
        except Exception:
            # Log warning for malformed headers and continue
            malformed_text = match.group(0) if match else "unknown"
            logger.warning("Skipping malformed story header: %s", malformed_text)
            continue

    return stories


def parse_epic_file(path: str | Path) -> EpicDocument:
    """Parse a BMAD epic file to extract story list.

    Uses parse_bmad_file() for frontmatter/content parsing, then
    applies epic-specific parsing to extract story information.

    Args:
        path: Path to the epic markdown file (string or Path object).

    Returns:
        EpicDocument with epic metadata and list of stories.

    Raises:
        FileNotFoundError: If the file does not exist.
        ParserError: If file parsing fails.

    Examples:
        >>> epic = parse_epic_file("docs/epics.md")
        >>> len(epic.stories)
        60
        >>> epic.stories[0].number
        '1.1'

    """
    doc = parse_bmad_file(path)

    # Check if this is a multi-epic file
    is_multi_epic = _is_multi_epic_file(doc.content)

    if is_multi_epic:
        # For consolidated epics file, metadata is None
        epic_num = None
        title = None
        status = None
    else:
        # Extract epic metadata from frontmatter
        epic_num = doc.frontmatter.get("epic_num")
        title = doc.frontmatter.get("title")
        status = doc.frontmatter.get("status")

        # Fallback: parse from markdown header if frontmatter is empty
        # Handles files like "# Epic 16: Real-time Dashboard"
        if epic_num is None or title is None:
            header_match = EPIC_TITLE_PATTERN.search(doc.content)
            if header_match:
                if epic_num is None:
                    raw_num = header_match.group(1)
                    # Try to convert to int, keep as string if not numeric
                    try:
                        epic_num = int(raw_num)
                    except ValueError:
                        epic_num = raw_num
                if title is None:
                    title = header_match.group(2).strip()

    # Parse story sections from content
    stories = _parse_story_sections(doc.content)

    return EpicDocument(
        epic_num=epic_num,
        title=title,
        status=status,
        stories=stories,
        path=doc.path,
    )
