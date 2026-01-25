"""Benchmark subcommand group for bmad-assist CLI.

Commands for workflow benchmarking and model comparison.
"""

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

from bmad_assist.cli_utils import (
    EXIT_ERROR,
    _error,
    _get_benchmarks_dir,
    _validate_project_path,
)

logger = logging.getLogger(__name__)

benchmark_app = typer.Typer(
    name="benchmark",
    help="Workflow benchmarking comparison commands",
    no_args_is_help=True,
)


@benchmark_app.command("compare")
def benchmark_compare(
    variant_a: str = typer.Option(
        ...,
        "--variant-a",
        "-a",
        help="First variant name to compare",
    ),
    variant_b: str = typer.Option(
        ...,
        "--variant-b",
        "-b",
        help="Second variant name to compare",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    ),
    date_from: str | None = typer.Option(
        None,
        "--from",
        help="Filter records from date (ISO 8601)",
    ),
    date_to: str | None = typer.Option(
        None,
        "--to",
        help="Filter records to date (ISO 8601)",
    ),
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
) -> None:
    """Compare workflow variants by aggregating benchmarking metrics.

    Loads all evaluation records for the specified variants, aggregates
    metrics, performs statistical significance testing (if scipy is installed),
    and generates a markdown comparison report.
    """
    from datetime import datetime as dt

    from bmad_assist.benchmarking.reports import (
        compare_workflow_variants,
        generate_comparison_report,
    )

    # Validate project path
    project_path = _validate_project_path(project)
    base_dir = _get_benchmarks_dir(project_path)

    # Parse dates if provided (ensure timezone-aware for comparison with UTC records)
    from datetime import UTC

    from_dt: dt | None = None
    to_dt: dt | None = None

    if date_from:
        try:
            from_dt = dt.fromisoformat(date_from)
            # Ensure timezone-aware (assume UTC if naive)
            if from_dt.tzinfo is None:
                from_dt = from_dt.replace(tzinfo=UTC)
        except ValueError:
            _error(f"Invalid --from date format: {date_from}")
            raise typer.Exit(code=EXIT_ERROR) from None

    if date_to:
        try:
            to_dt = dt.fromisoformat(date_to)
            # Ensure timezone-aware (assume UTC if naive)
            if to_dt.tzinfo is None:
                to_dt = to_dt.replace(tzinfo=UTC)
        except ValueError:
            _error(f"Invalid --to date format: {date_to}")
            raise typer.Exit(code=EXIT_ERROR) from None

    try:
        result = compare_workflow_variants(variant_a, variant_b, base_dir, from_dt, to_dt)
        report = generate_comparison_report(result)
    except Exception as e:
        _error(f"Comparison failed: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    # Separate console for stderr confirmations
    stderr_console = Console(stderr=True)

    if output:
        import os
        import tempfile

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename (project standard)
        fd, temp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(report)
            os.replace(temp_path, output_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        stderr_console.print(f"[green]Report written to:[/green] {output_path}")
    else:
        # Raw markdown to stdout (no Rich formatting for piping)
        sys.stdout.write(report)


@benchmark_app.command("models")
def benchmark_models(
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    ),
    date_from: str | None = typer.Option(
        None,
        "--from",
        help="Filter records from date (ISO 8601)",
    ),
    date_to: str | None = typer.Option(
        None,
        "--to",
        help="Filter records to date (ISO 8601)",
    ),
    format_: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown or json",
    ),
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
) -> None:
    """Compare all LLM models by aggregating benchmarking metrics.

    Loads all evaluation records, groups by provider/model, aggregates
    metrics, analyzes behavioral tendencies, and generates a comparison report.
    """
    from datetime import datetime as dt

    from bmad_assist.benchmarking.reports import (
        compare_models,
        generate_model_report_json,
        generate_model_report_markdown,
    )

    # Validate project path
    project_path = _validate_project_path(project)
    base_dir = _get_benchmarks_dir(project_path)

    # Parse dates if provided (ensure timezone-aware for comparison with UTC records)
    from datetime import UTC

    from_dt: dt | None = None
    to_dt: dt | None = None

    if date_from:
        try:
            from_dt = dt.fromisoformat(date_from)
            # Ensure timezone-aware (assume UTC if naive)
            if from_dt.tzinfo is None:
                from_dt = from_dt.replace(tzinfo=UTC)
        except ValueError:
            _error(f"Invalid --from date format: {date_from}")
            raise typer.Exit(code=EXIT_ERROR) from None

    if date_to:
        try:
            to_dt = dt.fromisoformat(date_to)
            # Ensure timezone-aware (assume UTC if naive)
            if to_dt.tzinfo is None:
                to_dt = to_dt.replace(tzinfo=UTC)
        except ValueError:
            _error(f"Invalid --to date format: {date_to}")
            raise typer.Exit(code=EXIT_ERROR) from None

    # Validate format
    format_lower = format_.lower()
    if format_lower not in ("markdown", "json"):
        _error(f"Invalid --format: {format_}. Use 'markdown' or 'json'.")
        raise typer.Exit(code=EXIT_ERROR) from None

    try:
        result = compare_models(base_dir, from_dt, to_dt)

        if format_lower == "json":
            report = generate_model_report_json(result)
        else:
            report = generate_model_report_markdown(result)

    except Exception as e:
        _error(f"Comparison failed: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    # Separate console for stderr confirmations
    stderr_console = Console(stderr=True)

    if output:
        import os
        import tempfile

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename (project standard)
        fd, temp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(report)
            os.replace(temp_path, output_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        stderr_console.print(f"[green]Report written to:[/green] {output_path}")
    else:
        # Raw output to stdout (no Rich formatting for piping)
        sys.stdout.write(report)
