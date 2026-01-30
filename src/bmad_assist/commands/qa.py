"""QA subcommand group for bmad-assist CLI.

Commands for QA plan generation and test execution.
"""

import typer

from bmad_assist.cli_utils import (
    EXIT_CONFIG_ERROR,
    EXIT_ERROR,
    _error,
    _setup_logging,
    _success,
    _validate_project_path,
    _warning,
    console,
)
from bmad_assist.core.config import load_config_with_project
from bmad_assist.core.exceptions import ConfigError
from bmad_assist.core.paths import init_paths
from bmad_assist.core.types import parse_epic_id

qa_app = typer.Typer(
    name="qa",
    help="QA plan generation and test execution commands",
    no_args_is_help=True,
)


@qa_app.command("generate")
def qa_generate(
    epic: str = typer.Option(
        ...,
        "--epic",
        "-e",
        help="Epic ID to generate QA plan for (e.g., '17')",
    ),
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
        help="Enable verbose output",
    ),
) -> None:
    """Generate E2E test plan for an epic.

    Creates a comprehensive QA plan with test categories A/B/C.
    If a plan already exists, creates a timestamped backup first.

    The generator loads ux-elements.md (if available) to use correct
    data-testid selectors for Category B (Playwright) tests.

    Examples:
        bmad-assist qa generate -e 17         # Generate plan for epic 17
        bmad-assist qa generate -e 19 -v      # Verbose output

    Backups are stored as:
        epic-{id}-e2e-plan.backup-{YYYYMMDD}T{HHMMSS}.md

    """
    from bmad_assist.qa.checker import backup_qa_plan, get_qa_plan_path
    from bmad_assist.qa.generator import generate_qa_plan

    # Setup logging
    _setup_logging(verbose=verbose, quiet=False)

    # Validate project path
    project_path = _validate_project_path(project)

    # Parse epic ID
    epic_id = parse_epic_id(epic.strip())

    # Load configuration
    try:
        loaded_config = load_config_with_project(project_path=project_path)
        paths_config = {
            "output_folder": loaded_config.paths.output_folder,
            "planning_artifacts": loaded_config.paths.planning_artifacts,
            "implementation_artifacts": loaded_config.paths.implementation_artifacts,
            "project_knowledge": loaded_config.paths.project_knowledge,
        }
        # Add bmad_paths.epics if configured (supports custom epic locations)
        if loaded_config.bmad_paths and loaded_config.bmad_paths.epics:
            paths_config["epics"] = loaded_config.bmad_paths.epics
        init_paths(project_path, paths_config)
    except ConfigError as e:
        _error(f"Config error: {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from None

    # Check for existing plan and backup if needed
    qa_plan_path = get_qa_plan_path(loaded_config, project_path, epic_id)
    if qa_plan_path.exists():
        console.print("[yellow]QA plan exists, creating backup...[/yellow]")
        backup_path = backup_qa_plan(loaded_config, project_path, epic_id)
        if backup_path:
            console.print(f"  Backup: {backup_path}")
            # Remove original after backup so LLM doesn't see stale content
            qa_plan_path.unlink()
            console.print("  [dim]Old plan removed[/dim]")

    # Generate new plan
    console.print(f"Generating QA plan for epic {epic_id}...")

    try:
        result = generate_qa_plan(
            loaded_config,
            project_path,
            epic_id,
            force=True,  # Always regenerate (we already backed up)
        )

        if result.success:
            _success(f"QA plan generated: {result.qa_plan_path}")
            if result.trace_path:
                console.print(f"  Trace used: {result.trace_path}")
        else:
            _error(f"QA plan generation failed: {result.error}")
            raise typer.Exit(code=EXIT_ERROR)

    except Exception as e:
        _error(f"Failed to generate QA plan: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=EXIT_ERROR) from None


@qa_app.command("execute")
def qa_execute(
    epic: str = typer.Option(
        ...,
        "--epic",
        "-e",
        help="Epic ID to execute tests for (e.g., '17')",
    ),
    category: str = typer.Option(
        "A",
        "--category",
        "-c",
        help="Test category to run (A, B, or all)",
    ),
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
        help="Enable verbose output (show LLM streaming)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output (more detailed than verbose)",
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        help="Tests per batch for incremental execution (default: 10)",
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        help="Force batch mode (recommended for >10 tests)",
    ),
    no_batch: bool = typer.Option(
        False,
        "--no-batch",
        help="Force single run mode (all tests at once)",
    ),
    retry: bool = typer.Option(
        False,
        "--retry",
        help="Retry failed/error tests from last run (also runs new tests)",
    ),
    include_skipped: bool = typer.Option(
        False,
        "--include-skipped",
        help="When retrying, also include SKIP tests",
    ),
    retry_run: str = typer.Option(
        None,
        "--retry-run",
        help="Specific run ID to retry from (default: latest run)",
    ),
) -> None:
    """Execute E2E tests from a generated QA plan.

    Runs the qa-plan-execute workflow to execute tests for an epic.
    The QA plan must exist (run qa generate first if needed).

    For test sets >10 tests, batch mode is auto-enabled to prevent
    context overflow and enable crash recovery. Use --batch or --no-batch
    to override.

    RETRY MODE:
    Use --retry to re-run only failed/error tests from the last run.
    Automatically finds the latest run for the epic and retries failed tests.
    Also executes any new tests not in the previous run.

    Examples:
        bmad-assist qa execute -e 17 --retry           # Retry from latest run
        bmad-assist qa execute -e 17 --retry --include-skipped  # Include skips
        bmad-assist qa execute -e 17 --retry-run run-20260112-115110  # Specific run

    Use -v/--verbose to see LLM output streaming in real-time.
    Use --debug for even more detailed output.

    """
    from bmad_assist.qa.executor import execute_qa_plan

    # Setup logging based on verbosity
    # debug implies verbose
    effective_verbose = verbose or debug
    _setup_logging(verbose=effective_verbose, quiet=False)

    # Validate project path
    project_path = _validate_project_path(project)

    # Parse epic ID
    epic_id = parse_epic_id(epic.strip())

    # Load configuration
    try:
        loaded_config = load_config_with_project(project_path=project_path)
        paths_config = {
            "output_folder": loaded_config.paths.output_folder,
            "planning_artifacts": loaded_config.paths.planning_artifacts,
            "implementation_artifacts": loaded_config.paths.implementation_artifacts,
            "project_knowledge": loaded_config.paths.project_knowledge,
        }
        # Add bmad_paths.epics if configured (supports custom epic locations)
        if loaded_config.bmad_paths and loaded_config.bmad_paths.epics:
            paths_config["epics"] = loaded_config.bmad_paths.epics
        init_paths(project_path, paths_config)
    except ConfigError as e:
        _error(f"Config error: {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from None

    # Determine batch mode
    if batch and no_batch:
        _warning("Both --batch and --no-batch specified, --batch takes precedence")
    batch_mode_value: str | None = None
    if batch:
        batch_mode_value = "batch"
    elif no_batch:
        batch_mode_value = "all"

    # Execute tests
    mode_info = ""
    if batch_mode_value == "batch":
        mode_info = f" [batch mode, size={batch_size}]"
    elif batch_mode_value == "all":
        mode_info = " [single run mode]"
    console.print(f"Executing QA tests for epic {epic_id} (category {category}){mode_info}...")

    try:
        result = execute_qa_plan(
            loaded_config,
            project_path,
            epic_id,
            category=category,
            batch_size=batch_size,
            batch_mode=batch_mode_value,
            retry=retry,
            include_skipped=include_skipped,
            retry_run=retry_run,
        )

        if result.success:
            mode_str = ""
            if result.batch_mode:
                mode_str = f" ({result.batches_completed} batches)"
            _success(
                f"QA execution complete{mode_str}: {result.tests_passed}/{result.tests_run} "
                f"passed ({result.pass_rate:.1f}%)"
            )
            if result.results_path:
                console.print(f"  Results: {result.results_path}")
            if result.summary_path:
                console.print(f"  Summary: {result.summary_path}")
            if result.tests_failed > 0:
                _warning(f"{result.tests_failed} tests failed")
        else:
            _error(f"QA execution failed: {result.error}")
            raise typer.Exit(code=EXIT_ERROR)

    except Exception as e:
        _error(f"Failed to execute QA plan: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None
