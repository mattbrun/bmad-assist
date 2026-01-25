"""Git commit automation for bmad-assist phases.

Handles automatic commits after successful phase execution.
"""

import logging
import os
import subprocess
from pathlib import Path

from bmad_assist.core.state import Phase

logger = logging.getLogger(__name__)

# Phases that trigger auto-commit
# CREATE_STORY: Creates story documentation (commit after creation)
# DEV_STORY: Modifies code (commit after implementation)
# CODE_REVIEW_SYNTHESIS: Final story completion (commit after review)
# RETROSPECTIVE: Epic retrospective report (commit after retrospective)
# Validation phases excluded - their reports are outputs, not code changes
COMMIT_PHASES: frozenset[Phase] = frozenset(
    {
        Phase.CREATE_STORY,
        Phase.DEV_STORY,
        Phase.CODE_REVIEW_SYNTHESIS,
        Phase.RETROSPECTIVE,
    }
)

# Phase to conventional commit type mapping
PHASE_COMMIT_TYPES: dict[Phase, str] = {
    Phase.CREATE_STORY: "docs",
    Phase.DEV_STORY: "feat",
    Phase.CODE_REVIEW_SYNTHESIS: "refactor",
    Phase.RETROSPECTIVE: "chore",
}


def is_git_enabled() -> bool:
    """Check if git auto-commit is enabled via environment variable."""
    return os.environ.get("BMAD_GIT_COMMIT") == "1"


def should_commit_phase(phase: Phase | None) -> bool:
    """Check if the given phase should trigger a commit."""
    return phase in COMMIT_PHASES


def _run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a git command and return exit code, stdout, stderr.

    Args:
        args: Git command arguments (without 'git' prefix).
        cwd: Working directory for git command.

    Returns:
        Tuple of (exit_code, stdout, stderr).

    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Git command timed out"
    except FileNotFoundError:
        return 1, "", "Git not found in PATH"


def get_modified_files(project_path: Path) -> list[str]:
    """Get list of modified files in the repository.

    Returns both staged and unstaged changes, EXCLUDING output directories
    that contain generated artifacts (validation reports, benchmarks, etc.).

    Args:
        project_path: Path to git repository.

    Returns:
        List of modified file paths relative to repo root.

    """
    # Get both staged and unstaged changes
    exit_code, stdout, _ = _run_git(
        ["status", "--porcelain", "-uall"],
        project_path,
    )

    if exit_code != 0:
        return []

    # Directories to exclude from auto-commit (generated artifacts)
    exclude_prefixes = (
        "_bmad-output/",
        ".bmad-assist/prompts/",
        ".bmad-assist/cache/",
        ".bmad-assist/debug/",
    )

    files = []
    for line in stdout.strip().split("\n"):
        if line:
            # porcelain format: XY filename
            # Skip the status prefix (2 chars + space)
            filename = line[3:].strip()
            # Handle renamed files (old -> new)
            if " -> " in filename:
                filename = filename.split(" -> ")[1]

            # Skip files in excluded directories
            if any(filename.startswith(prefix) for prefix in exclude_prefixes):
                continue

            files.append(filename)

    return files


def check_for_deleted_story_files(project_path: Path) -> list[str]:
    """Check if any story files are marked for deletion.

    Story files are critical artifacts that should never be auto-deleted.
    This function detects deleted story files before they get committed.

    Args:
        project_path: Path to git repository.

    Returns:
        List of deleted story file paths. Empty if none found.

    """
    exit_code, stdout, _ = _run_git(
        ["status", "--porcelain", "-uall"],
        project_path,
    )

    if exit_code != 0:
        return []

    import re

    deleted_files = []

    for line in stdout.strip().split("\n"):
        if not line:
            continue

        # porcelain format: XY filename
        # X = staged status, Y = unstaged status
        # D in either position means deleted
        status = line[:2]
        filename = line[3:].strip()
        if " -> " in filename:
            filename = filename.split(" -> ")[1]

        # Check if file is deleted (D in staged or unstaged status)
        if "D" not in status:
            continue

        # Check if it's a story file: docs/sprint-artifacts/{epic}-{story}-{slug}.md
        # or stories/{epic}-{story}-{slug}.md
        story_pattern = re.compile(r"^(docs/sprint-artifacts/|stories/)?\d+-\d+-[\w-]+\.md$")
        if story_pattern.match(filename):
            deleted_files.append(filename)

    return deleted_files


def generate_commit_message(
    phase: Phase,
    story_id: str | None,
    modified_files: list[str],
) -> str:
    """Generate deterministic conventional commit message for the phase.

    Args:
        phase: The completed phase.
        story_id: Current story ID (e.g., "1.2").
        modified_files: List of files that were modified.

    Returns:
        Commit message string.

    """
    return _generate_conventional_message(phase, story_id, modified_files)


def _generate_conventional_message(
    phase: Phase,
    story_id: str | None,
    modified_files: list[str],
) -> str:
    """Generate conventional commit message.

    Format: <type>(story-X.Y): <description>

    Args:
        phase: The completed phase.
        story_id: Current story ID.
        modified_files: List of modified files.

    Returns:
        Conventional commit message.

    """
    commit_type = PHASE_COMMIT_TYPES.get(phase, "chore")

    # RETROSPECTIVE uses epic-based scope; other phases use story-based scope
    if phase == Phase.RETROSPECTIVE:
        # Extract epic from story_id (e.g., "22.11" -> "22", "testarch.1" -> "testarch")
        epic_id = story_id.split(".")[0] if story_id and "." in story_id else (story_id or "unknown") # noqa: E501
        scope = f"epic-{epic_id}"
        description = f"archive epic {epic_id} retrospective"
    else:
        scope = f"story-{story_id}" if story_id else "bmad"
        if phase == Phase.CREATE_STORY:
            description = "create story file"
        elif phase == Phase.DEV_STORY:
            description = "implement story"
        elif phase == Phase.CODE_REVIEW_SYNTHESIS:
            description = "apply code review changes"
        else:
            description = f"complete {phase.value.replace('_', ' ')}"

    message = f"{commit_type}({scope}): {description}"

    # Add file count in body if many files changed
    if len(modified_files) > 3:
        message += f"\n\nModified {len(modified_files)} files"

    return message


def stage_all_changes(project_path: Path) -> bool:
    """Stage all modified files for commit.

    Args:
        project_path: Path to git repository.

    Returns:
        True if staging succeeded.

    """
    exit_code, _, stderr = _run_git(["add", "-A"], project_path)
    if exit_code != 0:
        logger.error("Failed to stage changes: %s", stderr)
        return False
    return True


def commit_changes(project_path: Path, message: str) -> bool:
    """Create a git commit with the given message.

    Args:
        project_path: Path to git repository.
        message: Commit message.

    Returns:
        True if commit succeeded.

    """
    exit_code, stdout, stderr = _run_git(
        ["commit", "-m", message],
        project_path,
    )

    if exit_code != 0:
        # "nothing to commit" is not an error
        if "nothing to commit" in stdout or "nothing to commit" in stderr:
            logger.info("Nothing to commit")
            return True
        logger.error("Failed to commit: %s", stderr)
        return False

    logger.info("Created commit: %s", message.split("\n")[0])
    return True


def auto_commit_phase(
    phase: Phase | None,
    story_id: str | None,
    project_path: Path,
) -> bool:
    """Automatically commit changes after a successful phase.

    This is the main entry point called from runner.py.

    Args:
        phase: The completed phase.
        story_id: Current story ID.
        project_path: Path to project root.

    Returns:
        True if commit was successful or not needed, False on error.

    """
    if not is_git_enabled():
        return True

    if not should_commit_phase(phase):
        return True

    # Check for modified files
    modified_files = get_modified_files(project_path)
    if not modified_files:
        logger.debug("No changes to commit for phase %s", phase)
        return True

    logger.info(
        "Auto-committing %d files after %s",
        len(modified_files),
        phase.value if phase else "unknown",
    )

    # CRITICAL: Check if any story files are marked for deletion
    # This prevents accidental permanent deletion of story artifacts
    deleted_story_files = check_for_deleted_story_files(project_path)
    if deleted_story_files:
        logger.error(
            "CRITICAL: Detected story file deletion(s): %s. "
            "Story files should NEVER be auto-deleted. "
            "Aborting commit to prevent data loss. "
            "Please investigate what caused these deletions.",
            deleted_story_files,
        )
        # Return False to halt the workflow - this is a critical error
        return False

    # Stage all changes
    if not stage_all_changes(project_path):
        return False

    # Generate commit message
    message = generate_commit_message(
        phase,  # type: ignore[arg-type]
        story_id,
        modified_files,
    )

    # Commit
    return commit_changes(project_path, message)
