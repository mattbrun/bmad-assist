"""Sprint subcommand group for bmad-assist CLI.

Commands for sprint-status management and validation.
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from bmad_assist.cli_utils import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    _error,
    _success,
    _validate_project_path,
    _warning,
    console,
)
from bmad_assist.core.config import load_config_with_project
from bmad_assist.core.exceptions import BmadAssistError, ConfigError

if TYPE_CHECKING:
    from bmad_assist.sprint.reconciler import StatusChange

sprint_app = typer.Typer(
    name="sprint",
    help="Sprint-status management commands",
    no_args_is_help=True,
)


# --------------------------------------------------------------------------
# Sprint CLI Types and Helpers
# --------------------------------------------------------------------------


class DiscrepancySeverity(Enum):
    """Severity level for sprint-status discrepancies."""

    ERROR = "error"  # Requires fixing - causes exit code 1
    WARN = "warn"  # Advisory only - exit code 0


@dataclass(frozen=True)
class Discrepancy:
    """A discrepancy between sprint-status and artifact evidence."""

    key: str
    sprint_status: str
    inferred_status: str
    severity: DiscrepancySeverity
    reason: str
    evidence: str  # Description of artifact evidence


def _get_sprint_status_path(project_root: Path) -> Path:
    """Get sprint-status.yaml path using paths singleton.

    Args:
        project_root: Project root directory (kept for backward compatibility).

    Returns:
        Path to sprint-status.yaml in implementation artifacts.

    """
    from bmad_assist.core.paths import get_paths

    return get_paths().sprint_status_file


def _setup_sprint_context(project: str) -> tuple[Path, Path, bool]:
    """Validate paths and return (project_root, sprint_path, is_legacy_only).

    If only legacy location exists (docs/sprint-artifacts/sprint-status.yaml),
    uses that path and signals that auto_exclude_legacy should be disabled.

    Args:
        project: Project path string.

    Returns:
        Tuple of (project_root, sprint_path, is_legacy_only).
        is_legacy_only=True means auto_exclude_legacy should be False.

    Raises:
        typer.Exit: If project path is invalid.

    """
    from bmad_assist.core.paths import get_paths, init_paths

    project_root = _validate_project_path(project)

    # Initialize paths singleton if not already done
    try:
        get_paths()
    except RuntimeError:
        # Load config to get external paths if configured
        try:
            config = load_config_with_project(project_path=project_root)
            paths_config = {}
            if config.paths:
                if config.paths.output_folder:
                    paths_config["output_folder"] = config.paths.output_folder
                if config.paths.project_knowledge:
                    paths_config["project_knowledge"] = config.paths.project_knowledge
            # Add bmad_paths.epics if configured (supports custom epic locations)
            if config.bmad_paths and config.bmad_paths.epics:
                paths_config["epics"] = config.bmad_paths.epics
            init_paths(project_root, paths_config)
        except ConfigError:
            # No config - use defaults
            init_paths(project_root, {})

    paths = get_paths()
    sprint_path = paths.sprint_status_file
    legacy_path = paths.legacy_sprint_artifacts / "sprint-status.yaml"

    # If only legacy location exists, use it
    is_legacy_only = False
    if not sprint_path.exists() and legacy_path.exists():
        sprint_path = legacy_path
        is_legacy_only = True

    return project_root, sprint_path, is_legacy_only


def _display_changes_table(changes: "list[StatusChange]") -> None:
    """Display changes in a Rich table.

    Args:
        changes: List of StatusChange objects from reconciler.

    """
    from rich.table import Table

    table = Table(title="Changes Applied")
    table.add_column("Key", style="cyan")
    table.add_column("Old Status", style="yellow")
    table.add_column("New Status", style="green")
    table.add_column("Reason")
    table.add_column("Confidence", style="dim")

    for change in changes:
        old = change.old_status or "(new)"
        conf = change.confidence.name if change.confidence else "-"
        table.add_row(
            change.key,
            old,
            change.new_status,
            change.reason,
            conf,
        )

    console.print(table)


def _display_discrepancies_table(discrepancies: list[Discrepancy]) -> None:
    """Display discrepancies in a Rich table.

    Args:
        discrepancies: List of Discrepancy objects.

    """
    from rich.table import Table

    table = Table(title="Discrepancies Found")
    table.add_column("Key", style="cyan")
    table.add_column("Severity")
    table.add_column("Sprint Status", style="yellow")
    table.add_column("Inferred", style="green")
    table.add_column("Reason")

    for d in discrepancies:
        severity_str = (
            "[red]ERROR[/red]"
            if d.severity == DiscrepancySeverity.ERROR
            else "[yellow]WARN[/yellow]"
        )
        table.add_row(
            d.key,
            severity_str,
            d.sprint_status,
            d.inferred_status,
            d.reason,
        )

    console.print(table)


# --------------------------------------------------------------------------
# Sprint Commands
# --------------------------------------------------------------------------


@sprint_app.command("generate")
def sprint_generate(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    include_legacy: bool = typer.Option(
        False,
        "--include-legacy",
        help="Include legacy epics (normally auto-excluded if tracked in docs/sprint-artifacts/)",
    ),
) -> None:
    """Generate sprint-status entries from epic files.

    Scans epic files in docs/epics/ and _bmad-output/planning-artifacts/epics/,
    extracts story definitions, and merges with existing sprint-status.

    By default, auto-excludes epics tracked in docs/sprint-artifacts/sprint-status.yaml
    and skips epics in archive/ directories.
    """
    from bmad_assist.sprint import (
        ArtifactIndex,
        SprintStatus,
        generate_from_epics,
        parse_sprint_status,
        reconcile,
        write_sprint_status,
    )

    project_root, sprint_path, is_legacy_only = _setup_sprint_context(project)

    # Load existing sprint-status or create empty
    if sprint_path.exists():
        existing = parse_sprint_status(sprint_path)
        console.print(
            f"[dim]Loaded existing sprint-status with {len(existing.entries)} entries[/dim]"
        )
    else:
        existing = SprintStatus.empty(project=project_root.name)
        console.print("[dim]Creating new sprint-status file[/dim]")

    # Generate entries from epic files
    # Disable auto_exclude_legacy if only legacy location exists or user requests include
    effective_auto_exclude = not include_legacy and not is_legacy_only
    generated = generate_from_epics(project_root, auto_exclude_legacy=effective_auto_exclude)
    console.print(
        f"[dim]Generated {len(generated.entries)} entries from {generated.files_processed} files[/dim]" # noqa: E501
    )

    if generated.duplicates_skipped > 0:
        _warning(f"{generated.duplicates_skipped} duplicate entries skipped")
    if generated.files_failed > 0:
        _warning(f"{generated.files_failed} files failed to parse")

    # Create empty artifact index (merge-only, no evidence inference)
    index = ArtifactIndex()

    # Reconcile (merge without evidence-based inference)
    reconciliation = reconcile(existing, generated, index)

    # Write result
    write_sprint_status(reconciliation.status, sprint_path, preserve_comments=True)

    # Summary
    console.print()
    _success(f"Generated {len(reconciliation.status.entries)} entries")
    console.print(f"  Output: {sprint_path}")
    console.print(f"  {reconciliation.summary()}")

    if verbose and reconciliation.changes:
        console.print()
        _display_changes_table(reconciliation.changes)


@sprint_app.command("repair")
def sprint_repair(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show changes without applying",
    ),
    include_legacy: bool = typer.Option(
        False,
        "--include-legacy",
        help="Include legacy epics (normally auto-excluded if tracked in docs/sprint-artifacts/)",
    ),
) -> None:
    """Repair sprint-status from artifact evidence.

    Scans project artifacts (stories, code reviews, validations, retrospectives)
    and repairs sprint-status using evidence-based inference.

    By default, auto-excludes epics tracked in docs/sprint-artifacts/sprint-status.yaml
    and skips epics in archive/ directories.
    """
    from bmad_assist.sprint import (
        ArtifactIndex,
        RepairMode,
        SprintStatus,
        generate_from_epics,
        parse_sprint_status,
        reconcile,
        repair_sprint_status,
    )

    project_root, sprint_path, is_legacy_only = _setup_sprint_context(project)

    if dry_run:
        # Generate what would change without writing
        try:
            if sprint_path.exists():
                existing = parse_sprint_status(sprint_path)
            else:
                existing = SprintStatus.empty(project=project_root.name)

            # Disable auto_exclude_legacy if only legacy location exists
            effective_auto_exclude = not include_legacy and not is_legacy_only
            generated = generate_from_epics(
                project_root, auto_exclude_legacy=effective_auto_exclude
            )
            index = ArtifactIndex.scan(project_root)
            reconciliation = reconcile(existing, generated, index)
        except BmadAssistError as e:
            _error(f"Failed to analyze sprint status: {e}")
            raise typer.Exit(code=EXIT_ERROR) from None

        console.print("[yellow]Dry run - no changes written[/yellow]")
        console.print()
        console.print(f"Would apply {len(reconciliation.changes)} changes")
        console.print(f"  {reconciliation.summary()}")

        if verbose and reconciliation.changes:
            console.print()
            _display_changes_table(reconciliation.changes)
    else:
        # Check if we need confirmation before overwriting
        from bmad_assist.core.loop.interactive import is_non_interactive

        if sprint_path.exists() and not is_non_interactive():
            console.print(f"\n[yellow]Warning:[/yellow] This will overwrite {sprint_path}")
            console.print("[dim]You can restore using BMAD workflow: /bmad:bmm:workflows:sprint-planning[/dim]") # noqa: E501
            if not typer.confirm("Continue?", default=False):
                console.print("[dim]Aborted.[/dim]")
                raise typer.Exit(code=EXIT_SUCCESS)

        # Actually perform repair
        # Note: repair_sprint_status also detects legacy-only internally, but we pass the flag
        effective_auto_exclude = not include_legacy and not is_legacy_only
        result = repair_sprint_status(
            project_root, RepairMode.SILENT, auto_exclude_legacy=effective_auto_exclude
        )

        if result.errors:
            _error(f"Repair completed with errors: {', '.join(result.errors)}")
            raise typer.Exit(code=EXIT_ERROR)

        console.print()
        _success(result.summary())
        console.print(f"  Output: {sprint_path}")

        if verbose:
            console.print(f"  Divergence: {result.divergence_pct:.1f}%")


@sprint_app.command("validate")
def sprint_validate(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output with evidence",
    ),
    format_: str = typer.Option(
        "plain",
        "--format",
        "-f",
        help="Output format: plain or json",
    ),
) -> None:
    """Validate sprint-status against artifact evidence.

    Compares sprint-status entries with evidence from code reviews,
    validations, and story files. Reports discrepancies with severity.

    Exit code 0 if no ERROR discrepancies (WARN only is OK).
    Exit code 1 if any ERROR discrepancies found.
    """
    import json

    from bmad_assist.sprint import (
        ArtifactIndex,
        InferenceConfidence,
        infer_story_status_detailed,
        parse_sprint_status,
    )
    from bmad_assist.sprint.classifier import EntryType

    project_root, sprint_path, _is_legacy_only = _setup_sprint_context(project)

    # Validate format
    format_lower = format_.lower()
    if format_lower not in ("plain", "json"):
        _error(f"Invalid --format: {format_}. Use 'plain' or 'json'.")
        raise typer.Exit(code=EXIT_ERROR)

    # Load sprint-status
    if not sprint_path.exists():
        _error(f"Sprint-status not found: {sprint_path}")
        raise typer.Exit(code=EXIT_ERROR)

    sprint_status = parse_sprint_status(sprint_path)

    # Scan artifacts
    index = ArtifactIndex.scan(project_root)

    # Compare each entry with inferred status
    discrepancies: list[Discrepancy] = []

    for key, entry in sprint_status.entries.items():
        # Skip non-story entries (epic meta, retrospectives)
        if entry.entry_type in (EntryType.EPIC_META, EntryType.RETROSPECTIVE):
            continue

        # Get inferred status
        result = infer_story_status_detailed(key, index)
        inferred = result.status
        confidence = result.confidence

        # Compare
        if entry.status == inferred:
            continue  # No discrepancy

        # Determine severity based on rules from story
        severity = DiscrepancySeverity.WARN
        reason = ""
        evidence = ""

        # Build evidence description
        if result.evidence_sources:
            evidence = f"Found: {result.evidence_sources[0].name}"
            if len(result.evidence_sources) > 1:
                evidence += f" (+{len(result.evidence_sources) - 1} more)"
        else:
            evidence = "No artifacts found"

        # Classification rules from AC4
        if entry.status == "done" and not index.has_master_review(key):
            severity = DiscrepancySeverity.ERROR
            reason = "Sprint says 'done' but no master code review exists"
        elif entry.status == "backlog" and index.has_master_review(key):
            severity = DiscrepancySeverity.ERROR
            reason = "Sprint says 'backlog' but master code review exists (missed update)"
        elif entry.status == "in-progress" and index.has_any_review(key):
            severity = DiscrepancySeverity.WARN
            reason = "Sprint says 'in-progress' but code reviews exist (should be 'review')"
        elif entry.status == "review" and index.has_master_review(key):
            severity = DiscrepancySeverity.WARN
            reason = "Sprint says 'review' but master review exists (should be 'done')"
        elif confidence == InferenceConfidence.EXPLICIT:
            severity = DiscrepancySeverity.WARN
            reason = "Story file Status differs from sprint-status (possible manual override)"
        else:
            reason = f"Status mismatch: sprint={entry.status}, inferred={inferred}"

        discrepancies.append(
            Discrepancy(
                key=key,
                sprint_status=entry.status,
                inferred_status=inferred,
                severity=severity,
                reason=reason,
                evidence=evidence,
            )
        )

    # Count by severity
    error_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.ERROR)
    warn_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.WARN)

    # Output
    if format_lower == "json":
        output = {
            "success": error_count == 0,
            "exit_code": 1 if error_count > 0 else 0,
            "summary": {
                "total": len(discrepancies),
                "error_count": error_count,
                "warn_count": warn_count,
            },
            "discrepancies": [
                {
                    "key": d.key,
                    "sprint_status": d.sprint_status,
                    "inferred_status": d.inferred_status,
                    "severity": d.severity.value,
                    "reason": d.reason,
                    "evidence": d.evidence,
                }
                for d in discrepancies
            ],
        }
        # JSON to stdout
        sys.stdout.write(json.dumps(output, indent=2))
        sys.stdout.write("\n")
    else:
        # Plain output
        if not discrepancies:
            _success("No discrepancies found")
            console.print(f"  Validated {len(sprint_status.entries)} entries")
        else:
            console.print()
            console.print(f"Found {len(discrepancies)} discrepancies:")
            console.print(f"  [red]ERROR:[/red] {error_count}")
            console.print(f"  [yellow]WARN:[/yellow] {warn_count}")
            console.print()
            _display_discrepancies_table(discrepancies)

            if verbose:
                console.print()
                console.print("[bold]Evidence Details:[/bold]")
                for d in discrepancies:
                    console.print(f"  {d.key}: {d.evidence}")

    # Exit code based on ERROR count
    if error_count > 0:
        raise typer.Exit(code=EXIT_ERROR)
    raise typer.Exit(code=EXIT_SUCCESS)


@sprint_app.command("sync")
def sprint_sync(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """Sync sprint-status from state.yaml.

    One-way sync: state.yaml (runtime authority) -> sprint-status.yaml (BMAD view).
    Updates current story status based on phase and marks completed stories/epics.
    """
    from bmad_assist.core.state import get_state_path, load_state
    from bmad_assist.sprint import trigger_sync

    project_root, sprint_path, _is_legacy_only = _setup_sprint_context(project)

    # Try to load config for state path resolution
    try:
        loaded_config = load_config_with_project(project_path=project_root)
        state_path = get_state_path(loaded_config, project_root=project_root)
    except ConfigError:
        # No config - use default state path
        state_path = project_root / ".bmad-assist" / "state.yaml"

    # Check state.yaml exists
    if not state_path.exists():
        _error(f"state.yaml not found: {state_path}")
        _error("Cannot sync without state file. Run the development loop first.")
        raise typer.Exit(code=EXIT_ERROR)

    # Load state
    try:
        state = load_state(state_path)
    except Exception as e:
        _error(f"Failed to load state: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    # Perform sync
    try:
        result = trigger_sync(state, project_root)
    except Exception as e:
        _error(f"Sync failed: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    # Summary
    console.print()
    _success(result.summary())
    console.print(f"  Output: {sprint_path}")

    if verbose and result.skipped_keys:
        console.print()
        _warning(f"Skipped {len(result.skipped_keys)} missing keys:")
        for key in result.skipped_keys:
            console.print(f"    - {key}")
