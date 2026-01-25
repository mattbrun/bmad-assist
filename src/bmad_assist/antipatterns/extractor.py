"""Antipatterns extraction from synthesis reports.

This module extracts verified issues from synthesis reports using regex patterns,
then appends them to epic-scoped antipatterns files for use by subsequent workflows.

Public API:
    extract_antipatterns: Extract issues list from synthesis content
    append_to_antipatterns_file: Append issues to antipatterns file
    extract_and_append_antipatterns: Combined convenience function
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from bmad_assist.core.io import atomic_write
from bmad_assist.core.paths import get_paths

if TYPE_CHECKING:
    from bmad_assist.core.config import Config
    from bmad_assist.core.types import EpicId

logger = logging.getLogger(__name__)

# Regex patterns for extraction
ISSUES_SECTION_PATTERN = re.compile(
    r"## Issues Verified.*?(?=^## |\Z)", re.MULTILINE | re.DOTALL
)
SEVERITY_HEADER_PATTERN = re.compile(
    r"^###\s+(Critical|High|Medium|Low)", re.IGNORECASE | re.MULTILINE
)
# Issue with Fix - handles BOTH formats:
# Format A: - **Issue desc** | ... | **Fix**: action (whole desc is bold)
# Format B: - **Issue**: desc | ... | **Fix**: action (only "Issue" is bold)
# Must have **Fix**: to be extracted (skip DEFERRED)
# Captures: group(1) = raw issue text (needs cleanup), group(2) = fix action
ISSUE_WITH_FIX_PATTERN = re.compile(
    r"^\s*-\s+\*\*([^|]+?)\s*\|.*?\*\*Fix\*\*:\s*(.+?)$", re.MULTILINE
)


# Warning header for antipatterns files
STORY_ANTIPATTERNS_HEADER = """# Epic {epic_id} - Story Antipatterns

> **WARNING: ANTI-PATTERNS**
> The issues below were MISTAKES found during validation of previous stories.
> DO NOT repeat these patterns. Learn from them and avoid similar errors.
> These represent story-writing mistakes (unclear AC, missing Notes, unrealistic scope).

"""

CODE_ANTIPATTERNS_HEADER = """# Epic {epic_id} - Code Antipatterns

> **WARNING: ANTI-PATTERNS**
> The issues below were MISTAKES found during code review of previous stories.
> DO NOT repeat these patterns. Learn from them and avoid similar errors.
> These represent implementation mistakes (race conditions, missing tests, weak assertions, etc.)

"""


def extract_antipatterns(
    synthesis_content: str,
    epic_id: "EpicId",
    story_id: str,
    config: "Config",
) -> list[dict[str, str]]:
    """Extract verified issues from synthesis content using regex patterns.

    Args:
        synthesis_content: Raw synthesis report content.
        epic_id: Epic identifier (numeric or string like "testarch").
        story_id: Story identifier (e.g., "24-11").
        config: Application configuration with antipatterns settings.

    Returns:
        List of issue dictionaries with keys: severity, issue, fix.
        Returns empty list on any failure (best-effort, non-blocking).

    """
    logger.info("Starting antipatterns extraction for story %s", story_id)

    # Check config
    try:
        if not config.antipatterns.enabled:
            logger.debug("Antipatterns extraction disabled in config")
            return []
    except AttributeError:
        pass  # Config doesn't have antipatterns yet, proceed with default enabled

    # Input validation - early exit
    if not synthesis_content or not synthesis_content.strip():
        logger.debug("Empty synthesis content, skipping antipatterns extraction")
        return []

    # Find Issues Verified section
    section_match = ISSUES_SECTION_PATTERN.search(synthesis_content)
    if not section_match:
        logger.debug("No 'Issues Verified' section found, skipping extraction")
        return []

    section_content = section_match.group(0)
    issues: list[dict[str, str]] = []

    # Split by severity headers and extract issues
    current_severity = "unknown"
    lines = section_content.split("\n")

    for line in lines:
        # Check for severity header
        header_match = SEVERITY_HEADER_PATTERN.match(line)
        if header_match:
            current_severity = header_match.group(1).lower()
            continue

        # Check for issue with fix
        issue_match = ISSUE_WITH_FIX_PATTERN.match(line)
        if issue_match:
            issue_desc = issue_match.group(1).strip()
            fix_desc = issue_match.group(2).strip()

            # Clean up issue description:
            # - Remove trailing ** (Format A: **desc**)
            # - Remove leading Issue**: prefix (Format B: Issue**: desc)
            issue_desc = issue_desc.rstrip("*").strip()
            if issue_desc.startswith("Issue"):
                # Remove "Issue**:" or "Issue:" prefix
                issue_desc = re.sub(r"^Issue\*{0,2}:?\s*", "", issue_desc)

            issues.append(
                {
                    "severity": current_severity,
                    "issue": issue_desc,
                    "fix": fix_desc,
                }
            )

    logger.info(
        "Extracted %d antipatterns from story %s (epic %s)",
        len(issues),
        story_id,
        epic_id,
    )
    return issues


def append_to_antipatterns_file(
    issues: list[dict[str, str]],
    epic_id: "EpicId",
    story_id: str,
    antipattern_type: Literal["story", "code"],
    project_path: Path,
) -> None:
    """Append extracted issues to antipatterns file.

    Creates file with warning header if it doesn't exist.
    Appends story section with issues table in markdown format.

    Args:
        issues: List of issue dictionaries to append.
        epic_id: Epic identifier (numeric or string).
        story_id: Story identifier (e.g., "24-11").
        antipattern_type: Either "story" (for validation) or "code" (for code review).
        project_path: Project root path for path resolution.

    """
    if not issues:
        logger.debug("No issues to append, skipping file write")
        return

    try:
        paths = get_paths()
        impl_artifacts = paths.implementation_artifacts
    except RuntimeError:
        # Paths not initialized - use fallback
        impl_artifacts = project_path / "_bmad-output" / "implementation-artifacts"

    # Create antipatterns subdirectory
    antipatterns_dir = impl_artifacts / "antipatterns"
    antipatterns_dir.mkdir(parents=True, exist_ok=True)
    antipatterns_path = antipatterns_dir / f"epic-{epic_id}-{antipattern_type}-antipatterns.md"

    # Determine header based on type
    if antipattern_type == "story":
        header = STORY_ANTIPATTERNS_HEADER.format(epic_id=epic_id)
    else:
        header = CODE_ANTIPATTERNS_HEADER.format(epic_id=epic_id)

    # Read existing content or start with header
    if antipatterns_path.exists():
        existing_content = antipatterns_path.read_text(encoding="utf-8")
    else:
        existing_content = header

    # Build story section with 3-column table (no file column)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    story_section = f"\n## Story {story_id} ({date_str})\n\n"
    story_section += "| Severity | Issue | Fix |\n"
    story_section += "|----------|-------|-----|\n"

    for issue in issues:
        severity = issue.get("severity", "unknown")
        issue_desc = issue.get("issue", "").replace("|", "\\|").replace("\n", " ")
        fix_desc = issue.get("fix", "-").replace("|", "\\|").replace("\n", " ")
        story_section += f"| {severity} | {issue_desc} | {fix_desc} |\n"

    # Append to content
    full_content = existing_content.rstrip() + "\n" + story_section

    # Atomic write
    atomic_write(antipatterns_path, full_content)
    logger.info("Appended %d antipatterns to %s", len(issues), antipatterns_path)


def extract_and_append_antipatterns(
    synthesis_content: str,
    epic_id: "EpicId",
    story_id: str,
    antipattern_type: Literal["story", "code"],
    project_path: Path,
    config: "Config",
) -> None:
    """Extract antipatterns and append to file (convenience function).

    Combines extract_antipatterns() and append_to_antipatterns_file() into
    a single call. Handles all errors gracefully (best-effort, non-blocking).

    Args:
        synthesis_content: Raw synthesis report content.
        epic_id: Epic identifier (numeric or string).
        story_id: Story identifier (e.g., "24-11").
        antipattern_type: Either "story" (for validation) or "code" (for code review).
        project_path: Project root path for path resolution.
        config: Application configuration with helper provider settings.

    """
    issues = extract_antipatterns(synthesis_content, epic_id, story_id, config)
    if issues:
        append_to_antipatterns_file(issues, epic_id, story_id, antipattern_type, project_path)
