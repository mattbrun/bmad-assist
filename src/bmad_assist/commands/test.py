"""Test subcommand group for bmad-assist CLI.

Commands for testing framework scorecard generation and comparison.
"""

import logging
import subprocess
import sys
from pathlib import Path

import typer

from bmad_assist.cli_utils import (
    EXIT_CONFIG_ERROR,
    EXIT_ERROR,
    _error,
    _setup_logging,
    _success,
    _validate_project_path,
    console,
)

logger = logging.getLogger(__name__)

test_app = typer.Typer(
    name="test",
    help="Testing framework commands",
    no_args_is_help=True,
)

# Hardcoded paths per tech-spec
FIXTURES_DIR = "experiments/fixtures"
SCORECARD_SCRIPT = "experiments/testing-framework/common/scorecard.py"


def _validate_fixture(project_path: Path, fixture: str) -> Path:
    """Validate that a fixture directory exists.

    Args:
        project_path: Project root path.
        fixture: Fixture name.

    Returns:
        Path to the fixture directory.

    Raises:
        typer.Exit: If fixture does not exist.

    """
    fixture_path = project_path / FIXTURES_DIR / fixture

    if not fixture_path.exists():
        _error(f"Fixture not found: {FIXTURES_DIR}/{fixture}")
        raise typer.Exit(code=EXIT_ERROR)

    if not fixture_path.is_dir():
        _error(f"Fixture must be a directory: {FIXTURES_DIR}/{fixture}")
        raise typer.Exit(code=EXIT_ERROR)

    return fixture_path


@test_app.command("scorecard")
def test_scorecard(
    fixture: str = typer.Argument(..., help="Fixture name (e.g., webhook-relay-001)"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: experiments/analysis/scorecards/{fixture}.yaml)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show subprocess output",
    ),
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
) -> None:
    """Generate quality scorecard for a fixture.

    Analyzes fixture completeness, functionality, code quality,
    and documentation to produce a quality score.

    Examples:
        bmad-assist test scorecard webhook-relay-001
        bmad-assist test scorecard webhook-relay-002 -o custom-output.yaml
        bmad-assist test scorecard webhook-relay-001 -v

    """
    _setup_logging(verbose=verbose, quiet=False)
    project_path = _validate_project_path(project)

    # Pre-validate fixture exists before spawning subprocess
    _validate_fixture(project_path, fixture)

    # Build command using direct script path (not -m, as dir uses hyphens)
    scorecard_path = project_path / SCORECARD_SCRIPT
    if not scorecard_path.exists():
        _error(f"Scorecard script not found: {SCORECARD_SCRIPT}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    cmd = [sys.executable, str(scorecard_path), fixture]

    if output:
        cmd.extend(["--output", output])

    console.print(f"Generating scorecard for: [bold]{fixture}[/bold]")

    try:
        result = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=not verbose,
            text=True,
            timeout=600,  # 10 min - security scans can be slow
        )

        if result.returncode == 0:
            if not verbose and result.stdout:
                # Show summary from scorecard output
                for line in result.stdout.split("\n"):
                    if line.strip():
                        console.print(line)
            _success("Scorecard generated")
        else:
            if result.stderr:
                _error(result.stderr.strip())
            elif result.stdout:
                _error(result.stdout.strip())
            raise typer.Exit(code=EXIT_ERROR)

    except subprocess.TimeoutExpired:
        _error("Scorecard generation timed out after 10 minutes")
        raise typer.Exit(code=EXIT_ERROR) from None

    except FileNotFoundError:
        _error(f"Python executable not found: {sys.executable}")
        raise typer.Exit(code=EXIT_ERROR) from None


@test_app.command("compare")
def test_compare(
    fixture1: str = typer.Argument(..., help="First fixture name"),
    fixture2: str = typer.Argument(..., help="Second fixture to compare against"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: experiments/analysis/scorecards/{fixture1}.yaml)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show subprocess output",
    ),
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
) -> None:
    """Compare two fixtures and generate scorecard with delta.

    Generates a scorecard for fixture1 that includes comparison
    delta against fixture2's scores.

    Examples:
        bmad-assist test compare webhook-relay-001 webhook-relay-002
        bmad-assist test compare new-version old-version -o comparison.yaml

    """
    _setup_logging(verbose=verbose, quiet=False)
    project_path = _validate_project_path(project)

    # Pre-validate both fixtures exist
    _validate_fixture(project_path, fixture1)
    _validate_fixture(project_path, fixture2)

    # Build command using direct script path (not -m, as dir uses hyphens)
    scorecard_path = project_path / SCORECARD_SCRIPT
    if not scorecard_path.exists():
        _error(f"Scorecard script not found: {SCORECARD_SCRIPT}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    cmd = [sys.executable, str(scorecard_path), fixture1, "--compare", fixture2]

    if output:
        cmd.extend(["--output", output])

    console.print(f"Comparing: [bold]{fixture1}[/bold] vs [bold]{fixture2}[/bold]")

    try:
        result = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=not verbose,
            text=True,
            timeout=600,  # Longer timeout for comparison (two scorecards)
        )

        if result.returncode == 0:
            if not verbose and result.stdout:
                for line in result.stdout.split("\n"):
                    if line.strip():
                        console.print(line)
            _success("Comparison scorecard generated")
        else:
            if result.stderr:
                _error(result.stderr.strip())
            elif result.stdout:
                _error(result.stdout.strip())
            raise typer.Exit(code=EXIT_ERROR)

    except subprocess.TimeoutExpired:
        _error("Comparison timed out after 10 minutes")
        raise typer.Exit(code=EXIT_ERROR) from None

    except FileNotFoundError:
        _error(f"Python executable not found: {sys.executable}")
        raise typer.Exit(code=EXIT_ERROR) from None
