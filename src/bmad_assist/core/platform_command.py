"""Cross-platform command building utilities for large prompts.

This module provides utilities for building CLI commands that handle large prompts
without hitting ARG_MAX limits on POSIX systems (~128KB-2MB depending on platform).

Strategy:
    - POSIX with large prompt (>=100KB): Use temp file + shell command
    - POSIX with small prompt (<100KB): Direct command
    - Windows: Always direct command (no temp file support)

Used by: copilot.py, cursor_agent.py

Example:
    >>> from bmad_assist.core.platform_command import (
    ...     build_cross_platform_command,
    ...     cleanup_temp_file,
    ... )
    >>> cmd, temp_file = build_cross_platform_command("copilot", ["-p"], prompt)
    >>> try:
    ...     subprocess.run(cmd, ...)
    ... finally:
    ...     cleanup_temp_file(temp_file)

"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import sys
import tempfile

logger = logging.getLogger(__name__)

# Platform detection
IS_WINDOWS = sys.platform == "win32"
IS_POSIX = not IS_WINDOWS

# Threshold for using temp file approach on POSIX (100KB in bytes)
# Safe margin under ARG_MAX which is typically 128KB-2MB
TEMP_FILE_THRESHOLD = 100_000


def build_cross_platform_command(
    executable: str,
    args: list[str],
    prompt: str,
) -> tuple[list[str], str | None]:
    """Build command list for cross-platform execution with large prompt support.

    On POSIX systems with large prompts (>=100KB bytes), creates a temp file
    containing the prompt and returns a shell command that reads from it.
    This avoids ARG_MAX limits.

    On Windows or with small prompts, returns a direct command list.

    Args:
        executable: The CLI executable name (e.g., "copilot", "cursor-agent").
        args: Arguments before the prompt (e.g., ["-p"] or ["--print", "--model", "x"]).
        prompt: The prompt text to pass to the CLI.

    Returns:
        Tuple of (command_list, temp_file_path_or_none).
        If temp_file_path is not None, caller MUST clean it up in a finally block.

    Example:
        >>> cmd, temp = build_cross_platform_command("copilot", ["-p"], "short prompt")
        >>> cmd
        ['copilot', '-p', 'short prompt']
        >>> temp is None
        True

    """
    # Measure prompt size in bytes (unicode chars can be multi-byte)
    prompt_bytes = len(prompt.encode("utf-8"))

    # Windows: always direct command (temp file not supported)
    # POSIX with small prompt: direct command
    if IS_WINDOWS or prompt_bytes < TEMP_FILE_THRESHOLD:
        return _build_direct_command(executable, args, prompt), None

    # POSIX with large prompt: use temp file
    return _build_shell_command(executable, args, prompt)


def _build_direct_command(
    executable: str,
    args: list[str],
    prompt: str,
) -> list[str]:
    """Build direct command list with prompt as argument.

    Args:
        executable: The CLI executable name.
        args: Arguments before the prompt.
        prompt: The prompt text.

    Returns:
        Command list: [executable, *args, prompt]

    """
    return [executable, *args, prompt]


def _build_shell_command(
    executable: str,
    args: list[str],
    prompt: str,
) -> tuple[list[str], str]:
    """Build shell command using temp file for large prompt.

    Creates a temp file containing the prompt and builds a shell command
    that uses command substitution to read the prompt from the file.

    Args:
        executable: The CLI executable name.
        args: Arguments before the prompt.
        prompt: The prompt text.

    Returns:
        Tuple of (command_list, temp_file_path).
        Command list is ["/bin/sh", "-c", "..."].

    """
    # Create temp file with prompt content
    fd, temp_path = tempfile.mkstemp(prefix="bmad_prompt_", suffix=".txt")
    try:
        os.write(fd, prompt.encode("utf-8"))
    finally:
        os.close(fd)

    logger.debug(
        "Created temp file for large prompt: path=%s, size=%d bytes",
        temp_path,
        len(prompt.encode("utf-8")),
    )

    # Build shell command with command substitution
    # Format: executable arg1 arg2 "$(cat /tmp/prompt.txt)"
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    quoted_temp = shlex.quote(temp_path)

    if quoted_args:
        shell_cmd = f'{executable} {quoted_args} "$(cat {quoted_temp})"'
    else:
        shell_cmd = f'{executable} "$(cat {quoted_temp})"'

    return ["/bin/sh", "-c", shell_cmd], temp_path


def cleanup_temp_file(temp_file_path: str | None) -> None:
    """Clean up temp file created by build_cross_platform_command.

    Safe to call with None - no-op in that case.
    Suppresses all errors (file already deleted, permissions, etc.).

    Args:
        temp_file_path: Path to temp file, or None.

    """
    if temp_file_path is None:
        return

    with contextlib.suppress(OSError):
        os.unlink(temp_file_path)
        logger.debug("Cleaned up temp file: %s", temp_file_path)
