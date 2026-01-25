"""Interactive repair dialog for sprint-status management.

This module provides the interactive dialog infrastructure for confirming
sprint-status repairs when high divergence is detected. The dialog system
supports CLI prompts with timeout and placeholder for future dashboard modal.

Two dialog implementations:
- CLIRepairDialog: Rich-based CLI prompt with timeout (threading-based)
- DashboardRepairDialog: Auto-cancel stub until dashboard modal implemented

Key Design Decisions:
- RepairDialogResult is frozen for immutability and hashability
- RepairSummary provides formatted summary for display
- Timeout uses threading for cross-platform compatibility (Linux/WSL2/Windows)
- Non-TTY and CI environments auto-cancel immediately (fire-and-forget safety)
- Dashboard dialog auto-cancels for safety (prevents silent mass updates)

Public API:
    - RepairSummary: Summary of proposed repair changes for dialog display
    - RepairDialogResult: Result of repair confirmation dialog
    - RepairDialog: Protocol for dialog implementations
    - CLIRepairDialog: CLI implementation with Rich and timeout
    - DashboardRepairDialog: Safety-first auto-cancel stub
    - get_repair_dialog: Factory function to get appropriate dialog
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)

__all__ = [
    "RepairSummary",
    "RepairDialogResult",
    "RepairDialog",
    "CLIRepairDialog",
    "DashboardRepairDialog",
    "get_repair_dialog",
]

# Default timeout in seconds (configurable via SprintConfig)
DEFAULT_DIALOG_TIMEOUT = 60


# =============================================================================
# RepairSummary Dataclass (Task 1)
# =============================================================================


@dataclass(frozen=True)
class RepairSummary:
    """Summary of proposed repair changes for dialog display.

    Frozen dataclass providing a structured view of repair changes
    for user confirmation. Used by both CLI and dashboard dialogs.

    Attributes:
        stories_to_update: Number of stories with status changes.
        epics_to_update: Number of epics with status changes.
        new_entries: Number of new entries to be added.
        removed_entries: Number of entries removed/deferred.
        divergence_pct: Percentage of entries that diverged.

    Example:
        >>> summary = RepairSummary(
        ...     stories_to_update=15,
        ...     epics_to_update=3,
        ...     new_entries=5,
        ...     removed_entries=2,
        ...     divergence_pct=45.0,
        ... )
        >>> summary.format_summary()
        '15 stories to update, 3 epics to update, 5 new, 2 removed (45.0% divergence)'

    """

    stories_to_update: int
    epics_to_update: int
    new_entries: int
    removed_entries: int
    divergence_pct: float

    def format_summary(self) -> str:
        """Format summary for display.

        Returns:
            Human-readable summary string with all counts and divergence.

        """
        parts: list[str] = []
        if self.stories_to_update:
            parts.append(f"{self.stories_to_update} stories to update")
        if self.epics_to_update:
            parts.append(f"{self.epics_to_update} epics to update")
        if self.new_entries:
            parts.append(f"{self.new_entries} new entries")
        if self.removed_entries:
            parts.append(f"{self.removed_entries} removed/deferred")

        if not parts:
            return "No changes (0.0% divergence)"

        return f"{', '.join(parts)} ({self.divergence_pct:.1f}% divergence)"

    @property
    def total_changes(self) -> int:
        """Return total number of changes."""
        return (
            self.stories_to_update + self.epics_to_update + self.new_entries + self.removed_entries
        )


# =============================================================================
# RepairDialogResult Dataclass (Task 1)
# =============================================================================


@dataclass(frozen=True)
class RepairDialogResult:
    """Result of repair confirmation dialog.

    Frozen dataclass for immutability. Captures user decision and
    timing information for logging and metrics.

    Attributes:
        approved: True if user chose Update, False if Cancel or timeout.
        timed_out: True if dialog timed out (default 60s).
        elapsed_seconds: Time spent in dialog before response.

    Example:
        >>> result = RepairDialogResult(approved=True, timed_out=False, elapsed_seconds=2.5)
        >>> result.approved
        True

    """

    approved: bool
    timed_out: bool = False
    elapsed_seconds: float = 0.0

    def __repr__(self) -> str:
        """Return debug-friendly representation."""
        status = "approved" if self.approved else ("timed_out" if self.timed_out else "cancelled")
        return f"RepairDialogResult({status}, elapsed={self.elapsed_seconds:.1f}s)"


# =============================================================================
# RepairDialog Protocol (Task 1)
# =============================================================================


class RepairDialog(Protocol):
    """Protocol for repair confirmation dialog implementations.

    Defines the interface that both CLI and dashboard dialogs must implement.
    Uses typing.Protocol for duck-typing flexibility.

    """

    def show(self, summary: RepairSummary) -> RepairDialogResult:
        """Show dialog and return user decision.

        Args:
            summary: Summary of proposed repair changes.

        Returns:
            RepairDialogResult with user decision.

        """
        ...


# =============================================================================
# Timeout Helper (Task 3)
# =============================================================================


def _is_interactive_terminal() -> bool:
    """Check if running in an interactive terminal.

    Returns False for:
    - Piped input (echo "y" | bmad-assist ...)
    - CI environments (all major CI systems)
    - Non-TTY stdin

    Returns:
        True if running in interactive terminal, False otherwise.

    """
    if not sys.stdin.isatty():
        return False
    # Comprehensive CI environment detection
    ci_indicators = [
        "CI",  # Generic CI indicator
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "JENKINS_HOME",  # Jenkins
        "TRAVIS",  # Travis CI
        "CIRCLECI",  # CircleCI
        "BUILDKITE",  # Buildkite
        "TF_BUILD",  # Azure Pipelines
        "CODEBUILD_BUILD_ID",  # AWS CodeBuild
    ]
    return not any(os.environ.get(key) for key in ci_indicators)


def _flush_stdin() -> None:
    """Flush stdin buffer to prevent partial input affecting next prompt.

    Cross-platform implementation:
    - Unix: Uses termios.tcflush() to clear input buffer
    - Windows: Best effort - stdin.flush() may not clear input buffer

    This is called after timeout to ensure leftover keystrokes don't
    affect subsequent prompts.
    """
    try:
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (ImportError, AttributeError):
        # Windows or non-TTY - termios not available
        pass
    except OSError:
        # stdin not a terminal or other OS error
        pass


def _prompt_with_timeout(
    prompt: str,
    timeout_seconds: int,
    default: bool,
    console: Console | None = None,
) -> tuple[bool, bool]:
    """Prompt with timeout using threading (cross-platform).

    Uses ThreadPoolExecutor for cross-platform timeout support on
    Linux, WSL2, and Windows.

    Args:
        prompt: Prompt text to display.
        timeout_seconds: Seconds before auto-cancel.
        default: Default value if timeout or non-interactive.
        console: Optional Rich console for output.

    Returns:
        Tuple of (user_choice, timed_out).

    """
    from rich.console import Console

    if console is None:
        console = Console()

    # Check non-interactive environments (AC9)
    if not sys.stdin.isatty():
        logger.info("Non-TTY environment detected, auto-cancelling repair dialog")
        return (default, True)

    # Comprehensive CI environment detection (same as _is_interactive_terminal)
    ci_indicators = [
        "CI",  # Generic CI indicator
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "JENKINS_HOME",  # Jenkins
        "TRAVIS",  # Travis CI
        "CIRCLECI",  # CircleCI
        "BUILDKITE",  # Buildkite
        "TF_BUILD",  # Azure Pipelines
        "CODEBUILD_BUILD_ID",  # AWS CodeBuild
    ]
    if any(os.environ.get(key) for key in ci_indicators):
        logger.info("CI environment detected, auto-cancelling repair dialog")
        return (default, True)

    # Thread-safe containers for results
    result: list[bool] = [default]
    timed_out: list[bool] = [True]

    def get_input() -> None:
        """Get user input in background thread."""
        try:
            # Use print for prompt since input() blocks
            console.print(
                f"\n{prompt} [[bold green]y[/bold green]/[bold red]N[/bold red]] "
                f"(timeout in {timeout_seconds}s): ",
                end="",
            )
            response = input().strip().lower()
            result[0] = response in ("y", "yes")
            timed_out[0] = False
        except (EOFError, KeyboardInterrupt):
            # User cancelled or EOF
            timed_out[0] = False

    # Use executor without context manager to avoid hang on Ctrl+C
    # The wait=True default in __exit__ would block on the input() thread
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(get_input)
    try:
        future.result(timeout=timeout_seconds)
    except FuturesTimeout:
        console.print("\n[yellow]Timeout - auto-cancelling repair[/yellow]")
        # Flush stdin buffer to prevent partial input affecting next prompt (AC7)
        _flush_stdin()
    finally:
        # Shutdown without waiting - input() thread will be abandoned
        # This prevents hang on Ctrl+C since we don't wait for blocked thread
        executor.shutdown(wait=False)

    return (result[0], timed_out[0])


# =============================================================================
# CLIRepairDialog (Task 2)
# =============================================================================


class CLIRepairDialog:
    """CLI implementation of repair confirmation dialog.

    Uses Rich console for formatted output and threading-based timeout
    for cross-platform compatibility.

    Attributes:
        console: Rich Console instance for output.
        timeout_seconds: Timeout before auto-cancel.

    Example:
        >>> dialog = CLIRepairDialog(timeout_seconds=30)
        >>> result = dialog.show(summary)
        >>> result.approved
        False  # User cancelled or timed out

    """

    def __init__(
        self,
        console: Console | None = None,
        timeout_seconds: int = DEFAULT_DIALOG_TIMEOUT,
    ) -> None:
        """Initialize CLI repair dialog.

        Args:
            console: Optional Rich Console. Creates new one if not provided.
            timeout_seconds: Timeout in seconds (default 60).

        """
        from rich.console import Console

        self.console = console or Console()
        self.timeout_seconds = timeout_seconds

    def show(self, summary: RepairSummary) -> RepairDialogResult:
        """Show interactive repair confirmation dialog.

        Displays summary of proposed changes and prompts for confirmation.
        Handles timeout, keyboard interrupt, and non-interactive environments.

        Args:
            summary: Summary of proposed repair changes.

        Returns:
            RepairDialogResult with user decision.

        """
        start_time = time.monotonic()

        # Display summary with Rich formatting (AC1)
        self.console.print("\n[bold yellow]Sprint Status Repair Required[/bold yellow]")
        self.console.print(f"Detected [bold]{summary.divergence_pct:.1f}%[/bold] divergence\n")
        self.console.print(f"  • [cyan]{summary.stories_to_update}[/cyan] stories to update")
        self.console.print(f"  • [cyan]{summary.epics_to_update}[/cyan] epics to update")
        self.console.print(f"  • [cyan]{summary.new_entries}[/cyan] new entries")
        if summary.removed_entries:
            self.console.print(f"  • [cyan]{summary.removed_entries}[/cyan] removed/deferred")
        self.console.print()

        # Handle keyboard interrupt gracefully (AC8)
        try:
            approved, timed_out = _prompt_with_timeout(
                "Apply these repairs?",
                self.timeout_seconds,
                default=False,
                console=self.console,
            )
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Cancelled by user[/yellow]")
            return RepairDialogResult(
                approved=False,
                timed_out=False,
                elapsed_seconds=time.monotonic() - start_time,
            )

        elapsed = time.monotonic() - start_time

        if approved:
            self.console.print("[green]Applying repairs...[/green]")
        elif timed_out:
            logger.info("Repair dialog timed out after %.1fs, auto-cancelling", elapsed)
        else:
            self.console.print("[yellow]Repair cancelled[/yellow]")

        return RepairDialogResult(
            approved=approved,
            timed_out=timed_out,
            elapsed_seconds=elapsed,
        )


# =============================================================================
# DashboardRepairDialog (Task 5)
# =============================================================================


class DashboardRepairDialog:
    """Dashboard implementation - safety-first placeholder until UI modal.

    Auto-cancels for safety to prevent silent mass updates via dashboard.
    When Epic 16 UI modal is integrated, replace with actual modal confirmation.

    SAFETY: Auto-cancelling prevents silent mass updates via dashboard.
    This is intentional defensive design - users should use CLI for
    interactive repairs until dashboard modal is properly implemented.

    """

    def show(self, summary: RepairSummary) -> RepairDialogResult:
        """Auto-cancel for safety - dashboard modal not yet implemented.

        Args:
            summary: Summary of proposed repair changes.

        Returns:
            RepairDialogResult with approved=False.

        """
        logger.warning(
            "Dashboard repair dialog: auto-cancelling for safety (%s). "
            "Use CLI for interactive repair until dashboard modal is implemented.",
            summary.format_summary(),
        )
        return RepairDialogResult(
            approved=False,
            timed_out=False,
            elapsed_seconds=0.0,
        )


# =============================================================================
# Factory Function (Task 5)
# =============================================================================


def get_repair_dialog(
    context: str = "cli",
    timeout_seconds: int | None = None,
) -> RepairDialog:
    """Get appropriate repair dialog for context.

    Factory function that returns CLI or dashboard dialog based on context.
    Supports configuration of timeout through SprintConfig.

    Args:
        context: Dialog context ("cli" or "dashboard").
        timeout_seconds: Optional timeout override. Uses config or default if None.

    Returns:
        RepairDialog implementation for the context.

    Example:
        >>> dialog = get_repair_dialog("cli", timeout_seconds=30)
        >>> isinstance(dialog, CLIRepairDialog)
        True

    """
    # Get timeout from config if not overridden
    if timeout_seconds is None:
        try:
            from bmad_assist.core.config import get_config

            config = get_config()
            if config.sprint is not None:
                timeout_seconds = config.sprint.dialog_timeout_seconds
            else:
                timeout_seconds = DEFAULT_DIALOG_TIMEOUT
        except Exception:
            # Config not loaded or missing sprint section
            timeout_seconds = DEFAULT_DIALOG_TIMEOUT

    if context == "dashboard":
        return DashboardRepairDialog()

    # Default to CLI dialog
    return CLIRepairDialog(timeout_seconds=timeout_seconds)
