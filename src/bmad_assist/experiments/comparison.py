"""Comparison report generator for experiment framework.

This module provides the comparison infrastructure for analyzing multiple
experiment runs, detecting configuration differences across four axes,
computing metric comparisons, and generating detailed Markdown reports.

Usage:
    from bmad_assist.experiments import ComparisonGenerator, ComparisonReport

    generator = ComparisonGenerator(Path("experiments/runs"))
    report = generator.compare(["run-001", "run-002"])
    markdown = generator.generate_markdown(report)
    generator.save(report, Path("comparison.md"))

"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
)

from bmad_assist.core.exceptions import ConfigError
from bmad_assist.experiments.manifest import (
    ManifestInput,
    ManifestManager,
    ManifestResolved,
)
from bmad_assist.experiments.metrics import MetricsCollector, RunMetrics
from bmad_assist.experiments.runner import ExperimentStatus

logger = logging.getLogger(__name__)

# Maximum runs for comparison (NFR-E05)
MAX_COMPARISON_RUNS = 10

# Metrics to compare with their comparison direction (name, lower_is_better)
COMPARISON_METRICS: list[tuple[str, bool]] = [
    ("total_cost", True),  # lower is better
    ("total_tokens", True),  # lower is better
    ("total_duration_seconds", True),  # lower is better
    ("avg_tokens_per_phase", True),  # lower is better
    ("avg_cost_per_phase", True),  # lower is better
    ("stories_completed", False),  # higher is better
    ("stories_failed", True),  # lower is better
    ("success_rate", False),  # higher is better (calculated)
]

# Display names for Markdown report
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "total_cost": "Total Cost",
    "total_tokens": "Total Tokens",
    "total_duration_seconds": "Duration",
    "avg_tokens_per_phase": "Avg Tokens/Phase",
    "avg_cost_per_phase": "Avg Cost/Phase",
    "stories_completed": "Stories Completed",
    "stories_failed": "Stories Failed",
    "success_rate": "Success Rate",
}

__all__ = [
    # Data models
    "RunComparison",
    "ConfigDiff",
    "ComparisonDiff",
    "MetricComparison",
    "ComparisonReport",
    # Generator
    "ComparisonGenerator",
    # Constants
    "MAX_COMPARISON_RUNS",
    "COMPARISON_METRICS",
    "METRIC_DISPLAY_NAMES",
]


# =============================================================================
# Data Models
# =============================================================================


class RunComparison(BaseModel):
    """Comparison data for a single run.

    Combines manifest input/resolved with metrics summary.

    Attributes:
        run_id: Run identifier.
        input: Input configuration (fixture, config, patch_set, loop).
        resolved: Resolved configuration details.
        metrics: Summary metrics (None if not available).
        status: Run completion status.

    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Run identifier")
    input: ManifestInput = Field(..., description="Input configuration")
    resolved: ManifestResolved = Field(..., description="Resolved configuration")
    metrics: RunMetrics | None = Field(None, description="Summary metrics (None if not available)")
    status: ExperimentStatus = Field(..., description="Run completion status")


class ConfigDiff(BaseModel):
    """Difference for a single configuration axis.

    Attributes:
        axis: Which axis this diff represents.
        values: run_id → value mapping for this axis.
        is_same: True if all values identical.

    """

    model_config = ConfigDict(frozen=True)

    axis: Literal["fixture", "config", "patch_set", "loop"] = Field(
        ..., description="Which axis this diff represents"
    )
    values: dict[str, str] = Field(..., description="run_id → value mapping for this axis")
    is_same: bool = Field(..., description="True if all values identical")


class ComparisonDiff(BaseModel):
    """Complete configuration difference across all axes.

    Attributes:
        fixture: Fixture axis diff.
        config: Config axis diff.
        patch_set: Patch-set axis diff.
        loop: Loop axis diff.

    """

    model_config = ConfigDict(frozen=True)

    fixture: ConfigDiff
    config: ConfigDiff
    patch_set: ConfigDiff
    loop: ConfigDiff

    @computed_field  # type: ignore[prop-decorator]
    @property
    def varying_axes(self) -> list[str]:
        """List of axis names that differ across runs."""
        varying = []
        if not self.fixture.is_same:
            varying.append("fixture")
        if not self.config.is_same:
            varying.append("config")
        if not self.patch_set.is_same:
            varying.append("patch_set")
        if not self.loop.is_same:
            varying.append("loop")
        return varying


class MetricComparison(BaseModel):
    """Comparison of a single metric across runs.

    Attributes:
        metric_name: Metric name (e.g., "total_cost", "total_tokens").
        values: run_id → metric value mapping.
        deltas: run_id → percentage change from baseline.
        winner: run_id with best value (None if tie or N/A).
        lower_is_better: Whether lower values win.

    """

    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(..., description="Metric name")
    values: dict[str, float | int | None] = Field(..., description="run_id → metric value")
    deltas: dict[str, float | None] = Field(
        ..., description="run_id → percentage change from baseline"
    )
    winner: str | None = Field(None, description="run_id with best value (None if tie)")
    lower_is_better: bool = Field(..., description="Whether lower values win")


class ComparisonReport(BaseModel):
    """Complete comparison report for multiple runs.

    Attributes:
        generated_at: Report generation timestamp.
        run_ids: Ordered list of run IDs.
        runs: Full run comparison data.
        config_diff: Configuration differences across all axes.
        metrics: Per-metric comparisons.
        conclusion: Auto-generated conclusion (optional).

    """

    generated_at: datetime = Field(..., description="Report generation timestamp")
    run_ids: list[str] = Field(..., description="Ordered list of run IDs")
    runs: list[RunComparison] = Field(..., description="Full run data")
    config_diff: ComparisonDiff = Field(..., description="Configuration differences")
    metrics: list[MetricComparison] = Field(..., description="Per-metric comparisons")
    conclusion: str | None = Field(None, description="Auto-generated conclusion")

    @field_serializer("generated_at")
    def serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO 8601 with UTC timezone."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()


# =============================================================================
# Helper Functions
# =============================================================================


def _calculate_delta(value: float | int | None, baseline: float | int | None) -> float | None:
    """Calculate percentage change from baseline.

    Args:
        value: Current value to compare.
        baseline: Baseline value (first run).

    Returns:
        Percentage change, or None if comparison not possible.

    """
    if value is None or baseline is None:
        return None
    if baseline == 0:
        return 0.0 if value == 0 else None
    return ((value - baseline) / baseline) * 100.0


def _determine_winner(
    values: dict[str, float | int | None],
    lower_is_better: bool,
) -> str | None:
    """Determine the winner for a metric.

    Args:
        values: run_id → metric value mapping.
        lower_is_better: Whether lower values are better.

    Returns:
        run_id of winner, or None if tie or no valid values.

    """
    # Filter out None values
    valid = {k: v for k, v in values.items() if v is not None}
    if not valid:
        return None

    # Find best value
    best_value = min(valid.values()) if lower_is_better else max(valid.values())

    # Find run(s) with best value
    winners = [k for k, v in valid.items() if v == best_value]

    # Return winner only if unique
    return winners[0] if len(winners) == 1 else None


def _calculate_success_rate(completed: int, failed: int) -> float | None:
    """Calculate success rate percentage.

    Args:
        completed: Number of stories completed.
        failed: Number of stories failed.

    Returns:
        Success rate as percentage, or None if no stories.

    """
    total = completed + failed
    if total == 0:
        return None
    return (completed / total) * 100.0


def _format_cost(value: float | None) -> str:
    """Format cost as $X.XX or N/A."""
    if value is None:
        return "N/A"
    return f"${value:.2f}"


def _format_tokens(value: int | float | None) -> str:
    """Format tokens with comma separators."""
    if value is None:
        return "N/A"
    return f"{int(value):,}"


def _format_duration(seconds: float | None) -> str:
    """Format duration as HH:MM:SS (if >= 1h) or MM:SS."""
    if seconds is None:
        return "N/A"
    total_seconds = int(seconds)
    if total_seconds >= 3600:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"{minutes}:{secs:02d}"


def _format_delta(delta: float | None) -> str:
    """Format delta as signed percentage with 1 decimal."""
    if delta is None:
        return "N/A"
    # Round first to handle floating point edge cases like -0.04 -> -0.0%
    rounded = round(delta, 1)
    if rounded == 0.0:
        return "0.0%"
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded:.1f}%"


def _format_percentage(value: float | None) -> str:
    """Format percentage with 1 decimal."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _format_metric_value(metric_name: str, value: float | int | None) -> str:
    """Format a metric value based on its type.

    Args:
        metric_name: Name of the metric.
        value: Value to format.

    Returns:
        Formatted string representation.

    """
    if value is None:
        return "N/A"

    if metric_name in ("total_cost", "avg_cost_per_phase"):
        return _format_cost(value)
    elif metric_name in ("total_tokens", "avg_tokens_per_phase"):
        return _format_tokens(value)
    elif metric_name == "total_duration_seconds":
        return _format_duration(value)
    elif metric_name == "success_rate":
        return _format_percentage(value)
    elif metric_name in ("stories_completed", "stories_failed"):
        return str(int(value))
    else:
        return str(value)


# =============================================================================
# ComparisonGenerator Class
# =============================================================================


class ComparisonGenerator:
    """Generates comparison reports for experiment runs.

    Usage:
        generator = ComparisonGenerator(Path("experiments/runs"))
        report = generator.compare(["run-001", "run-002"])
        markdown = generator.generate_markdown(report)
        generator.save(report, Path("comparison.md"))

    """

    def __init__(self, runs_dir: Path) -> None:
        """Initialize the generator.

        Args:
            runs_dir: Base directory containing run subdirectories.

        """
        self._runs_dir = runs_dir

    def load_run(self, run_id: str) -> RunComparison:
        """Load comparison data for a single run.

        Args:
            run_id: Run identifier.

        Returns:
            RunComparison with manifest and metrics data.

        Raises:
            ConfigError: If manifest not found.

        """
        run_dir = self._runs_dir / run_id

        # Load manifest (required)
        manifest_manager = ManifestManager(run_dir)
        manifest = manifest_manager.load()

        # Load metrics (optional)
        metrics: RunMetrics | None = None
        try:
            collector = MetricsCollector(run_dir)
            metrics_file = collector.load()
            metrics = metrics_file.summary
        except ConfigError:
            logger.debug("Metrics not available for run %s", run_id)

        return RunComparison(
            run_id=run_id,
            input=manifest.input,
            resolved=manifest.resolved,
            metrics=metrics,
            status=manifest.status,
        )

    def compare(self, run_ids: list[str]) -> ComparisonReport:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare.

        Returns:
            ComparisonReport with full comparison data.

        Raises:
            ValueError: If less than 2 or more than 10 runs.
            ConfigError: If any run not found.

        """
        # Validation
        if len(run_ids) < 2:
            raise ValueError("At least 2 runs required for comparison")
        if len(run_ids) > MAX_COMPARISON_RUNS:
            raise ValueError(f"Maximum {MAX_COMPARISON_RUNS} runs allowed for comparison")

        # Load all runs
        runs = [self.load_run(run_id) for run_id in run_ids]

        # Warn if any run has non-completed status
        for run in runs:
            if run.status != ExperimentStatus.COMPLETED:
                logger.warning(
                    "Run %s has status %s (not completed) - metrics may be incomplete",
                    run.run_id,
                    run.status.value,
                )

        # Compute diffs and metrics
        config_diff = self._compute_config_diff(runs)
        metric_comparisons = self._compute_metric_comparisons(runs)

        # Generate conclusion
        conclusion = self._generate_conclusion(config_diff, metric_comparisons, runs)

        return ComparisonReport(
            generated_at=datetime.now(UTC),
            run_ids=run_ids,
            runs=runs,
            config_diff=config_diff,
            metrics=metric_comparisons,
            conclusion=conclusion,
        )

    def _compute_config_diff(self, runs: list[RunComparison]) -> ComparisonDiff:
        """Compute configuration differences across runs.

        Args:
            runs: List of run comparisons.

        Returns:
            ComparisonDiff with differences for all four axes.

        """

        def make_axis_diff(
            axis: Literal["fixture", "config", "patch_set", "loop"],
        ) -> ConfigDiff:
            values = {run.run_id: getattr(run.input, axis) for run in runs}
            unique_values = set(values.values())
            return ConfigDiff(
                axis=axis,
                values=values,
                is_same=len(unique_values) == 1,
            )

        return ComparisonDiff(
            fixture=make_axis_diff("fixture"),
            config=make_axis_diff("config"),
            patch_set=make_axis_diff("patch_set"),
            loop=make_axis_diff("loop"),
        )

    def _compute_metric_comparisons(self, runs: list[RunComparison]) -> list[MetricComparison]:
        """Compute metric comparisons across runs.

        Args:
            runs: List of run comparisons.

        Returns:
            List of MetricComparison for each metric.

        """
        comparisons = []

        for metric_name, lower_is_better in COMPARISON_METRICS:
            # Extract values for this metric
            values: dict[str, float | int | None] = {}
            for run in runs:
                if run.metrics is None:
                    values[run.run_id] = None
                elif metric_name == "success_rate":
                    values[run.run_id] = _calculate_success_rate(
                        run.metrics.stories_completed,
                        run.metrics.stories_failed,
                    )
                else:
                    values[run.run_id] = getattr(run.metrics, metric_name, None)

            # Skip metrics where ALL values are None
            if all(v is None for v in values.values()):
                continue

            # Calculate deltas (first run is baseline)
            baseline_run_id = runs[0].run_id
            baseline_value = values.get(baseline_run_id)
            deltas: dict[str, float | None] = {}
            for run_id, value in values.items():
                if run_id == baseline_run_id:
                    deltas[run_id] = 0.0 if value is not None else None
                else:
                    deltas[run_id] = _calculate_delta(value, baseline_value)

            # Determine winner
            winner = _determine_winner(values, lower_is_better)

            comparisons.append(
                MetricComparison(
                    metric_name=metric_name,
                    values=values,
                    deltas=deltas,
                    winner=winner,
                    lower_is_better=lower_is_better,
                )
            )

        return comparisons

    def _generate_conclusion(
        self,
        config_diff: ComparisonDiff,
        metrics: list[MetricComparison],
        runs: list[RunComparison],
    ) -> str | None:
        """Generate auto-conclusion based on comparison results.

        Heuristics:
        - Single axis varies + clear winner: Recommend adoption
        - Multiple axes vary: Flag confounded results
        - All ties: No significant difference
        - Non-completed runs: Append warning note

        Returns None if fewer than 2 metrics have valid comparisons.

        """
        # Count valid metric comparisons
        valid_metrics = [m for m in metrics if any(v is not None for v in m.values.values())]
        if len(valid_metrics) < 2:
            return None

        # Check if all runs have non-completed status
        all_incomplete = all(r.status != ExperimentStatus.COMPLETED for r in runs)
        if all_incomplete:
            return None

        # Count wins per run (only for efficiency metrics)
        efficiency_metrics = [
            "total_cost",
            "total_tokens",
            "total_duration_seconds",
        ]
        win_counts: dict[str, int] = {}
        for metric in metrics:
            if metric.winner and metric.metric_name in efficiency_metrics:
                win_counts[metric.winner] = win_counts.get(metric.winner, 0) + 1

        # Check for non-completed runs
        incomplete_runs = [r for r in runs if r.status != ExperimentStatus.COMPLETED]
        incomplete_note = ""
        if incomplete_runs:
            incomplete_note = (
                " Note: Comparison includes non-completed runs - results may be incomplete."
            )

        varying = config_diff.varying_axes

        if len(varying) > 1:
            axes_str = ", ".join(varying)
            return (
                f"Multiple configuration axes differ ({axes_str}); results may be "
                f"confounded. Consider isolating individual axis changes for clearer "
                f"attribution.{incomplete_note}"
            )

        if not win_counts:
            return f"No significant difference found between compared runs.{incomplete_note}"

        # Find overall winner - check for ties
        max_wins = max(win_counts.values())
        winners_with_max = [k for k, v in win_counts.items() if v == max_wins]

        if len(winners_with_max) > 1:
            # Tie in win counts
            return (
                f"No significant difference found between compared runs "
                f"(tie in wins).{incomplete_note}"
            )

        top_winner = winners_with_max[0]
        top_wins = max_wins
        baseline_run_id = runs[0].run_id

        # Calculate average improvement for efficiency metrics
        total_delta = 0.0
        delta_count = 0
        for metric in metrics:
            if metric.metric_name in efficiency_metrics and metric.winner == top_winner:
                delta = metric.deltas.get(top_winner)
                if delta is not None and delta != 0.0:
                    # For lower-is-better metrics, negative delta is improvement
                    total_delta += abs(delta)
                    delta_count += 1

        avg_improvement = total_delta / delta_count if delta_count > 0 else 0.0

        # Special handling when baseline wins (no improvement from experimental changes)
        if top_winner == baseline_run_id:
            if len(varying) == 1:
                axis = varying[0]
                return (
                    f"Baseline ({top_winner}) remains the most efficient configuration. "
                    f"The {axis} change did not yield performance improvements.{incomplete_note}"
                )
            return (
                f"Baseline ({top_winner}) shows the best performance across "
                f"{top_wins} efficiency metrics.{incomplete_note}"
            )

        if len(varying) == 1:
            axis = varying[0]
            return (
                f"{top_winner} shows ~{avg_improvement:.0f}% improvement in efficiency "
                f"metrics. The {axis} change appears beneficial. Recommendation: "
                f"Consider adopting this configuration.{incomplete_note}"
            )

        return (
            f"{top_winner} shows overall better performance across {top_wins} "
            f"efficiency metrics.{incomplete_note}"
        )

    def generate_markdown(self, report: ComparisonReport) -> str:
        """Generate Markdown report.

        Args:
            report: Comparison report to format.

        Returns:
            Formatted Markdown string.

        """
        lines: list[str] = []

        # Header
        lines.append("# Experiment Comparison Report")
        lines.append("")
        lines.append(f"Generated: {report.generated_at.isoformat()}")
        lines.append("")

        # Runs Compared table
        lines.append("## Runs Compared")
        lines.append("")
        lines.append("| Run ID | Fixture | Config | Patch-Set | Loop |")
        lines.append("|--------|---------|--------|-----------|------|")
        for run in report.runs:
            lines.append(
                f"| {run.run_id} | {run.input.fixture} | {run.input.config} | "
                f"{run.input.patch_set} | {run.input.loop} |"
            )
        lines.append("")

        # Configuration Diff table
        lines.append("## Configuration Diff")
        lines.append("")
        header = "| Axis |"
        separator = "|------|"
        for run_id in report.run_ids:
            header += f" {run_id} |"
            separator += "---------|"
        header += " Same? |"
        separator += "-------|"
        lines.append(header)
        lines.append(separator)

        for axis_name in ["fixture", "config", "patch_set", "loop"]:
            axis_diff = getattr(report.config_diff, axis_name)
            display_name = axis_name.replace("_", "-").title()
            if axis_name == "patch_set":
                display_name = "Patch-Set"
            row = f"| {display_name} |"
            for run_id in report.run_ids:
                row += f" {axis_diff.values[run_id]} |"
            same_indicator = "✓" if axis_diff.is_same else "**DIFFERENT**"
            row += f" {same_indicator} |"
            lines.append(row)
        lines.append("")

        # Patch-Set Differences (if patch_set varies)
        if not report.config_diff.patch_set.is_same:
            lines.append("### Patch-Set Differences")
            lines.append("")
            # Create header with run IDs and their patch-set names
            header = "| Workflow |"
            separator = "|----------|"
            for run in report.runs:
                patchset_name = run.input.patch_set
                header += f" {run.run_id} ({patchset_name}) |"
                separator += "---------------------------|"
            lines.append(header)
            lines.append(separator)

            # Collect all workflows across all runs
            all_workflows: set[str] = set()
            for run in report.runs:
                all_workflows.update(run.resolved.patch_set.patches.keys())

            # Sort workflows for consistent output
            for workflow in sorted(all_workflows):
                row = f"| {workflow} |"
                # Get patch path for first run as baseline
                baseline_patch = report.runs[0].resolved.patch_set.patches.get(workflow)

                for run in report.runs:
                    patch_path = run.resolved.patch_set.patches.get(workflow)
                    if run == report.runs[0]:
                        # First run shows actual path (simplified)
                        display = Path(patch_path).name if patch_path else "none"
                    elif patch_path == baseline_patch:
                        display = "same"
                    else:
                        # Different - show bold
                        display = f"**{Path(patch_path).name if patch_path else 'none'}**"
                    row += f" {display} |"
                lines.append(row)
            lines.append("")

        # Results Comparison table
        lines.append("## Results Comparison")
        lines.append("")
        header = "| Metric |"
        separator = "|--------|"
        for run_id in report.run_ids:
            header += f" {run_id} |"
            separator += "---------|"
        header += " Delta | Winner |"
        separator += "-------|--------|"
        lines.append(header)
        lines.append(separator)

        for metric in report.metrics:
            display_name = METRIC_DISPLAY_NAMES.get(metric.metric_name, metric.metric_name)
            row = f"| {display_name} |"

            # Values for each run
            for run_id in report.run_ids:
                value = metric.values.get(run_id)
                formatted = _format_metric_value(metric.metric_name, value)
                row += f" {formatted} |"

            # Delta (show for second run onwards, use second run's delta)
            if len(report.run_ids) >= 2:
                second_run_id = report.run_ids[1]
                delta = metric.deltas.get(second_run_id)
                row += f" {_format_delta(delta)} |"
            else:
                row += " N/A |"

            # Winner
            if metric.winner:
                row += f" {metric.winner} |"
            elif all(v is None for v in metric.values.values()):
                row += " N/A |"
            else:
                row += " TIE |"

            lines.append(row)
        lines.append("")

        # Conclusion
        if report.conclusion:
            lines.append("## Conclusion")
            lines.append("")
            lines.append(report.conclusion)
            lines.append("")

        return "\n".join(lines)

    def save(self, report: ComparisonReport, output_path: Path) -> Path:
        """Save report to file.

        Args:
            report: Comparison report to save.
            output_path: Path to output file.

        Returns:
            Path to saved file.

        Raises:
            ConfigError: If save operation fails.

        """
        markdown = self.generate_markdown(report)
        temp_path = output_path.with_suffix(".md.tmp")

        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file
            with temp_path.open("w", encoding="utf-8") as f:
                f.write(markdown)

            # Atomic rename
            os.replace(temp_path, output_path)

        except Exception as e:
            # Clean up temp file on any failure
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temp file: %s", temp_path)
            raise ConfigError(f"Failed to save comparison report: {e}") from e

        logger.info("Saved comparison report to %s", output_path)
        return output_path
