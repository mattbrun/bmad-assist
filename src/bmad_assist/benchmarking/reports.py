"""Workflow comparison report module for benchmarking.

Provides comparison of workflow variants by aggregating metrics across
evaluation records and performing statistical significance testing.

Public API:
    VariantMetrics: Aggregated metrics for a single workflow variant
    MetricComparison: Comparison of a single metric between variants
    ComparisonResult: Full comparison result between two variants
    compare_workflow_variants: Main entry point for variant comparison
    generate_comparison_report: Generate markdown report from comparison result
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bmad_assist.benchmarking.schema import EvaluatorRole, LLMEvaluationRecord
from bmad_assist.benchmarking.storage import (
    RecordFilters,
    StorageError,
    list_evaluation_records,
    load_evaluation_record,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Story 13.8: Workflow Comparison
    "VariantMetrics",
    "MetricComparison",
    "ComparisonResult",
    "compare_workflow_variants",
    "generate_comparison_report",
    # Story 13.9: Model Comparison
    "SeverityDistribution",
    "ModelTendencies",
    "ModelMetrics",
    "ModelComparisonResult",
    "compare_models",
    "generate_model_report_markdown",
    "generate_model_report_json",
]

# Constants for statistical analysis (Story 13.8)
MIN_SAMPLES_FOR_DISPLAY = 5  # Minimum records to show comparison
MIN_SAMPLES_FOR_STDEV = 2  # Minimum samples for standard deviation
MIN_SAMPLES_FOR_SIGNIFICANCE = 10  # Minimum samples per variant per metric for t-test
SIGNIFICANCE_THRESHOLD = 0.05  # p-value threshold for significance

# Constants for model comparison (Story 13.9)
MIN_SAMPLES_FOR_INCLUSION = 5  # Minimum evals for model (otherwise low_confidence)
MIN_MODELS_FOR_TENDENCY = 3  # Minimum models for meaningful tendency analysis
VERBOSE_THRESHOLD = 1.4  # char_count > 1.4x median = verbose
TERSE_THRESHOLD = 0.6  # char_count < 0.6x median = terse
OVER_REPORT_THRESHOLD = 1.5  # severity % > 1.5x avg = over-reporting
UNDER_REPORT_THRESHOLD = 0.5  # severity % < 0.5x avg = under-reporting


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class VariantMetrics:
    """Aggregated metrics for a single workflow variant.

    Tracks both synthesizer and validator metrics separately based on
    record.evaluator.role to ensure correct aggregation.
    """

    variant: str
    count: int
    date_range: tuple[datetime, datetime] | None

    # Agreement metrics (from synthesizer records)
    mean_agreement_score: float | None
    std_agreement_score: float | None

    # Findings metrics (from validator records)
    mean_findings_count: float | None
    std_findings_count: float | None

    # Quality metrics (from synthesizer records)
    mean_actionable_ratio: float | None
    std_actionable_ratio: float | None
    mean_specificity: float | None
    std_specificity: float | None

    # Per-role sample counts for statistical validity
    synthesizer_count: int
    validator_count: int


@dataclass(frozen=True)
class MetricComparison:
    """Comparison of a single metric between variants.

    Includes values from both variants, the delta, p-value from t-test,
    and significance flag.
    """

    name: str
    value_a: float | None
    std_a: float | None
    count_a: int
    value_b: float | None
    std_b: float | None
    count_b: int
    delta: float | None  # value_a - value_b
    p_value: float | None
    significant: bool | None  # None if insufficient data
    format_spec: str = ".2f"  # Format for display


@dataclass(frozen=True)
class ComparisonResult:
    """Full comparison result between two variants.

    Contains aggregated metrics for each variant, per-metric comparisons
    with significance testing, and metadata about the comparison.
    """

    variant_a: VariantMetrics
    variant_b: VariantMetrics
    metrics: list[MetricComparison]
    generated_at: datetime
    notes: list[str]
    scipy_available: bool


# =============================================================================
# Story 13.9: Model Comparison Dataclasses
# =============================================================================


@dataclass(frozen=True)
class SeverityDistribution:
    """Percentage breakdown of findings by severity.

    All percentages are expressed as floats (0.0-1.0).
    None indicates no data available for calculation.
    """

    critical_pct: float | None
    major_pct: float | None
    minor_pct: float | None
    nit_pct: float | None
    other_pct: float | None  # Unknown severity levels grouped here


@dataclass(frozen=True)
class ModelTendencies:
    """Behavioral patterns detected for a model.

    Each field contains a natural language description of the pattern,
    or None if not detected.
    """

    strength: str | None  # e.g., "High evidence citation (1.35x average)"
    tendency: str | None  # e.g., "Verbose reports (+40% avg length)"
    bias: str | None  # e.g., "Over-reports critical issues"


@dataclass(frozen=True)
class ModelMetrics:
    """Aggregated metrics for a single provider/model.

    Tracks both validator and synthesizer metrics, with separate counts
    for statistical validity. Tendencies are embedded (not a separate dict)
    to ensure JSON serialization works (JSON requires string keys).
    """

    provider: str
    model: str
    total_evaluations: int
    validator_count: int
    synthesizer_count: int

    # Findings metrics (validator records only)
    mean_findings_count: float | None
    std_findings_count: float | None
    severity_distribution: SeverityDistribution | None

    # Ground truth metrics (when available)
    false_positive_rate: float | None
    ground_truth_count: int  # How many records have ground_truth.populated == True

    # For tendency analysis (validator records only)
    mean_char_count: float | None

    # Low confidence flag
    low_confidence: bool  # True if total_evaluations < MIN_SAMPLES_FOR_INCLUSION

    # Tendencies embedded (not separate dict - avoids JSON tuple key issues)
    tendencies: ModelTendencies | None


@dataclass(frozen=True)
class ModelComparisonResult:
    """Full model comparison result.

    Contains aggregated metrics for each provider/model combination,
    with tendency analysis and metadata about the comparison.
    """

    models: list[ModelMetrics]
    generated_at: datetime
    total_records: int
    date_range: tuple[datetime, datetime] | None
    notes: list[str]


# =============================================================================
# Record Loading
# =============================================================================


def _load_records_by_variant(
    base_dir: Path,
    variant_a: str,
    variant_b: str,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> dict[str, list[LLMEvaluationRecord]]:
    """Load records grouped by variant.

    Uses list_evaluation_records for efficient metadata filtering,
    then loads only matching records. Filters by variant in memory
    since RecordFilters doesn't support variant filtering.

    Args:
        base_dir: Base directory for storage (docs/sprint-artifacts/).
        variant_a: First variant name.
        variant_b: Second variant name.
        date_from: Optional start date filter.
        date_to: Optional end date filter.

    Returns:
        Dict mapping variant name to list of matching records.

    """
    filters = RecordFilters(
        date_from=date_from,
        date_to=date_to,
    )

    summaries = list_evaluation_records(base_dir, filters)

    result: dict[str, list[LLMEvaluationRecord]] = {
        variant_a: [],
        variant_b: [],
    }

    for summary in summaries:
        try:
            record = load_evaluation_record(summary.path)
            if record.workflow.variant == variant_a:
                result[variant_a].append(record)
            elif record.workflow.variant == variant_b:
                result[variant_b].append(record)
        except StorageError as e:
            logger.warning("Failed to load record %s: %s", summary.path, e)

    logger.debug(
        "Loaded %d records for variant %s, %d for variant %s",
        len(result[variant_a]),
        variant_a,
        len(result[variant_b]),
        variant_b,
    )

    return result


def _load_records_by_model(
    base_dir: Path,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    workflow_id: str | None = None,
) -> dict[tuple[str, str], list[LLMEvaluationRecord]]:
    """Load records grouped by (provider, model) tuple.

    Uses list_evaluation_records for efficient metadata filtering,
    then loads full records and groups by provider/model.

    Args:
        base_dir: Base directory for storage (docs/sprint-artifacts/).
        date_from: Optional start date filter.
        date_to: Optional end date filter.
        workflow_id: Optional workflow ID filter (e.g., "code-review").
            Story 13.10: Added for cross-phase filtering.

    Returns:
        Dict mapping (provider, model) tuple to list of matching records.

    """
    # Story 13.10: Support workflow_id filtering
    filters = RecordFilters(
        date_from=date_from,
        date_to=date_to,
        workflow_id=workflow_id,
    )

    summaries = list_evaluation_records(base_dir, filters)

    result: dict[tuple[str, str], list[LLMEvaluationRecord]] = {}

    for summary in summaries:
        try:
            record = load_evaluation_record(summary.path)
            # AC2: Skip records with missing provider/model, log warning
            provider = record.evaluator.provider
            model = record.evaluator.model
            if not provider or not model:
                logger.warning(
                    "Skipping record %s: missing provider=%r or model=%r",
                    summary.path,
                    provider,
                    model,
                )
                continue
            key = (provider, model)
            if key not in result:
                result[key] = []
            result[key].append(record)
        except StorageError as e:
            logger.warning("Failed to load record %s: %s", summary.path, e)

    logger.debug("Loaded records for %d unique models", len(result))
    return result


# =============================================================================
# Metrics Aggregation
# =============================================================================


def _calculate_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Calculate mean and standard deviation for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Tuple of (mean, std). Returns (None, None) if empty,
        (mean, None) if only one value.

    """
    if not values:
        return None, None

    mean = sum(values) / len(values)

    if len(values) < MIN_SAMPLES_FOR_STDEV:
        return mean, None

    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std = variance**0.5

    return mean, std


def _aggregate_variant_metrics(
    records: list[LLMEvaluationRecord],
    variant: str,
) -> VariantMetrics:
    """Aggregate metrics for a single variant.

    Separates records by evaluator role before aggregation:
    - SYNTHESIZER records: agreement_score, actionable_ratio, specificity_score
    - VALIDATOR records: findings_count

    Args:
        records: List of evaluation records for this variant.
        variant: Variant name for identification.

    Returns:
        VariantMetrics with aggregated statistics.

    """
    synthesizer_records = [r for r in records if r.evaluator.role == EvaluatorRole.SYNTHESIZER]
    validator_records = [r for r in records if r.evaluator.role == EvaluatorRole.VALIDATOR]

    # Calculate date range
    if records:
        dates = [r.created_at for r in records]
        date_range = (min(dates), max(dates))
    else:
        date_range = None

    # Agreement score (synthesizer only)
    agreement_values = [
        r.consensus.agreement_score for r in synthesizer_records if r.consensus is not None
    ]
    mean_agreement, std_agreement = _calculate_mean_std(agreement_values)

    # Findings count (validator only)
    findings_values = [
        float(r.findings.total_count) for r in validator_records if r.findings is not None
    ]
    mean_findings, std_findings = _calculate_mean_std(findings_values)

    # Actionable ratio (synthesizer only)
    actionable_values = [
        r.quality.actionable_ratio for r in synthesizer_records if r.quality is not None
    ]
    mean_actionable, std_actionable = _calculate_mean_std(actionable_values)

    # Specificity score (synthesizer only)
    specificity_values = [
        r.quality.specificity_score for r in synthesizer_records if r.quality is not None
    ]
    mean_specificity, std_specificity = _calculate_mean_std(specificity_values)

    return VariantMetrics(
        variant=variant,
        count=len(records),
        date_range=date_range,
        mean_agreement_score=mean_agreement,
        std_agreement_score=std_agreement,
        mean_findings_count=mean_findings,
        std_findings_count=std_findings,
        mean_actionable_ratio=mean_actionable,
        std_actionable_ratio=std_actionable,
        mean_specificity=mean_specificity,
        std_specificity=std_specificity,
        synthesizer_count=len(synthesizer_records),
        validator_count=len(validator_records),
    )


# =============================================================================
# Statistical Significance Testing
# =============================================================================


def _check_scipy_available() -> bool:
    """Check if scipy is available for statistical testing."""
    try:
        from scipy.stats import ttest_ind  # noqa: F401

        return True
    except ImportError:
        return False


def _calculate_significance(
    values_a: list[float],
    values_b: list[float],
    min_samples: int = MIN_SAMPLES_FOR_SIGNIFICANCE,
) -> tuple[float | None, bool | None]:
    """Calculate t-test significance between two value sets.

    Uses scipy.stats.ttest_ind for independent samples t-test.

    Args:
        values_a: Values for first variant.
        values_b: Values for second variant.
        min_samples: Minimum samples per variant (default: 10).

    Returns:
        Tuple of (p_value, is_significant).
        Both None if insufficient samples or scipy unavailable.

    """
    if len(values_a) < min_samples or len(values_b) < min_samples:
        return None, None

    try:
        import math

        from scipy.stats import ttest_ind

        _, p_value = ttest_ind(values_a, values_b)
        # Handle NaN p-value (occurs when both samples have zero variance)
        if math.isnan(p_value):
            return None, None
        return float(p_value), p_value < SIGNIFICANCE_THRESHOLD
    except ImportError:
        logger.warning("scipy not installed, skipping significance testing")
        return None, None


# =============================================================================
# Markdown Report Generation
# =============================================================================


def _format_metric_value(
    value: float | None,
    std: float | None,
    count: int,
    format_spec: str,
) -> str:
    """Format a metric value with standard deviation and count.

    Args:
        value: Mean value (or None).
        std: Standard deviation (or None).
        count: Sample count.
        format_spec: Format specifier for the value.

    Returns:
        Formatted string like "0.85 ± 0.05 (12)" or "-" if no data.

    """
    if value is None:
        return "-"

    if format_spec == ".2%":
        # Percentage format
        val_str = f"{value:.2%}"
        std_str = f" ± {std:.2%}" if std is not None else ""
    else:
        # Numeric format
        val_str = f"{value:{format_spec}}"
        std_str = f" ± {std:{format_spec}}" if std is not None else ""

    return f"{val_str}{std_str} ({count})"


def _format_delta(delta: float | None, format_spec: str) -> str:
    """Format delta value with sign.

    Args:
        delta: Delta value (or None).
        format_spec: Format specifier.

    Returns:
        Formatted string with +/- prefix or "N/A".

    """
    if delta is None:
        return "N/A"

    if format_spec == ".2%":
        return f"{delta:+.2%}"

    return f"{delta:+{format_spec}}"


def _generate_interpretation(
    result: ComparisonResult,
) -> list[str]:
    """Generate interpretation bullet points.

    Args:
        result: Comparison result with metrics.

    Returns:
        List of interpretation strings.

    """
    lines: list[str] = []
    var_a = result.variant_a.variant
    var_b = result.variant_b.variant

    for metric in result.metrics:
        if metric.delta is None:
            continue

        # Determine which variant is better based on metric
        if metric.delta > 0:
            better = var_a
            direction = "higher" if metric.name != "Findings Count" else "more"
        else:
            better = var_b
            direction = "higher" if metric.name != "Findings Count" else "more"

        # Adjust direction for metrics where lower is better (findings count)
        if metric.name == "Findings Count":
            direction = "more" if metric.delta > 0 else "fewer"

        sig_note = " (statistically significant)" if metric.significant else ""
        lines.append(
            f"- **{better}** produces {direction} {metric.name.lower()}"
            f" (Δ = {_format_delta(metric.delta, metric.format_spec)}){sig_note}"
        )

    if not lines:
        lines.append("- No metrics available for comparison")

    return lines


def generate_comparison_report(result: ComparisonResult) -> str:
    """Generate markdown comparison report.

    Args:
        result: ComparisonResult with all data.

    Returns:
        Complete markdown report string.

    """
    lines: list[str] = []

    # Header
    lines.append("# Workflow Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {result.generated_at.isoformat()}")
    lines.append(f"**Variants:** {result.variant_a.variant} vs {result.variant_b.variant}")
    lines.append("")

    # Sample Sizes
    lines.append("## Sample Sizes")
    lines.append("")
    lines.append("| Variant | Count | Date Range |")
    lines.append("|---------|-------|------------|")

    for variant in [result.variant_a, result.variant_b]:
        date_str = (
            f"{variant.date_range[0].date()} - {variant.date_range[1].date()}"
            if variant.date_range
            else "-"
        )
        lines.append(f"| {variant.variant} | {variant.count} | {date_str} |")

    lines.append("")

    # Metrics Comparison
    lines.append("## Metrics Comparison")
    lines.append("")
    lines.append(
        f"| Metric | {result.variant_a.variant} (n) | {result.variant_b.variant} (n) "
        f"| Delta | p-value | Significant? |"
    )
    lines.append("|--------|-----------------|-----------------|-------|---------|--------------|")

    for metric in result.metrics:
        value_a_str = _format_metric_value(
            metric.value_a, metric.std_a, metric.count_a, metric.format_spec
        )
        value_b_str = _format_metric_value(
            metric.value_b, metric.std_b, metric.count_b, metric.format_spec
        )
        delta_str = _format_delta(metric.delta, metric.format_spec)

        p_str = f"{metric.p_value:.3f}" if metric.p_value is not None else "N/A"

        if metric.significant is True:
            sig_str = "✓"
        elif metric.significant is False:
            sig_str = "✗"
        else:
            sig_str = "-"

        lines.append(
            f"| {metric.name} | {value_a_str} | {value_b_str} | {delta_str} | {p_str} | {sig_str} |"
        )

    lines.append("")
    lines.append(
        'Note: n = sample count per metric, "-" = insufficient samples for significance test'
    )
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    for interp_line in _generate_interpretation(result):
        lines.append(interp_line)
    lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Minimum sample size: {MIN_SAMPLES_FOR_DISPLAY} records per variant")
    lines.append(f"- Statistical significance: p < {SIGNIFICANCE_THRESHOLD} (two-tailed t-test)")

    for note in result.notes:
        lines.append(f"- {note}")

    if not result.scipy_available:
        lines.append("- scipy not installed - significance testing unavailable")

    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Story 13.9: Model Comparison Functions
# =============================================================================


def _calculate_severity_distribution(
    records: list[LLMEvaluationRecord],
) -> SeverityDistribution | None:
    """Calculate severity distribution across records.

    Aggregates findings by severity across all records, returning
    percentages for known severity levels.

    Args:
        records: List of evaluation records.

    Returns:
        SeverityDistribution with percentages, or None if no findings.

    """
    total_counts: dict[str, int] = {
        "critical": 0,
        "major": 0,
        "minor": 0,
        "nit": 0,
        "other": 0,
    }

    total_findings = 0

    for record in records:
        if record.findings is None:
            continue
        for severity, count in record.findings.by_severity.items():
            severity_lower = severity.lower()
            if severity_lower in ("critical", "major", "minor", "nit"):
                total_counts[severity_lower] += count
            else:
                total_counts["other"] += count
            total_findings += count

    if total_findings == 0:
        return None

    return SeverityDistribution(
        critical_pct=total_counts["critical"] / total_findings,
        major_pct=total_counts["major"] / total_findings,
        minor_pct=total_counts["minor"] / total_findings,
        nit_pct=total_counts["nit"] / total_findings,
        other_pct=total_counts["other"] / total_findings,
    )


def _aggregate_model_metrics(
    records: list[LLMEvaluationRecord],
    provider: str,
    model: str,
) -> ModelMetrics:
    """Aggregate metrics for a single provider/model.

    Separates records by evaluator role:
    - VALIDATOR records: findings_count, char_count, source_references
    - SYNTHESIZER records: counted but not used for validator-specific metrics

    Args:
        records: List of evaluation records for this model.
        provider: Provider name.
        model: Model name.

    Returns:
        ModelMetrics with aggregated statistics.

    """
    validator_records = [r for r in records if r.evaluator.role == EvaluatorRole.VALIDATOR]
    synthesizer_records = [r for r in records if r.evaluator.role == EvaluatorRole.SYNTHESIZER]

    total_evaluations = len(records)
    low_confidence = total_evaluations < MIN_SAMPLES_FOR_INCLUSION

    # Findings count (validator records only)
    findings_counts = [
        float(r.findings.total_count) for r in validator_records if r.findings is not None
    ]
    mean_findings, std_findings = _calculate_mean_std(findings_counts)

    # Char count from output analysis (all records have this)
    char_counts = [float(r.output.char_count) for r in validator_records]
    mean_char, _ = _calculate_mean_std(char_counts)

    # Severity distribution
    severity_dist = _calculate_severity_distribution(validator_records)

    # Ground truth metrics
    gt_records = [
        r for r in validator_records if r.ground_truth is not None and r.ground_truth.populated
    ]
    ground_truth_count = len(gt_records)

    # False positive rate = total_false_alarm / (total_false_alarm + total_confirmed)
    false_positive_rate: float | None = None
    if ground_truth_count > 0:
        total_confirmed = sum(
            r.ground_truth.findings_confirmed for r in gt_records if r.ground_truth
        )
        total_false_alarm = sum(
            r.ground_truth.findings_false_alarm for r in gt_records if r.ground_truth
        )
        total_evaluated = total_confirmed + total_false_alarm
        if total_evaluated > 0:
            false_positive_rate = total_false_alarm / total_evaluated

    return ModelMetrics(
        provider=provider,
        model=model,
        total_evaluations=total_evaluations,
        validator_count=len(validator_records),
        synthesizer_count=len(synthesizer_records),
        mean_findings_count=mean_findings,
        std_findings_count=std_findings,
        severity_distribution=severity_dist,
        false_positive_rate=false_positive_rate,
        ground_truth_count=ground_truth_count,
        mean_char_count=mean_char,
        low_confidence=low_confidence,
        tendencies=None,  # Added later by _analyze_model_tendencies
    )


def _calculate_median(values: list[float]) -> float | None:
    """Calculate median of a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Median value or None if list is empty.

    """
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def _detect_category_bias(
    metrics: ModelMetrics,
    all_metrics: list[ModelMetrics],
) -> str | None:
    """Detect severity category over/under-reporting bias.

    Compares model's severity distribution against average across all models.

    Args:
        metrics: Model to analyze.
        all_metrics: All models for comparison.

    Returns:
        Bias description string or None.

    """
    if metrics.severity_distribution is None:
        return None

    # Get all models with severity distributions
    with_dist = [m for m in all_metrics if m.severity_distribution is not None]
    if len(with_dist) < MIN_MODELS_FOR_TENDENCY:
        return None

    # Calculate average percentages across models
    severity_fields = [
        ("critical", "critical_pct"),
        ("major", "major_pct"),
        ("minor", "minor_pct"),
        ("nit", "nit_pct"),
    ]

    for severity_name, field_name in severity_fields:
        values = [getattr(m.severity_distribution, field_name) or 0.0 for m in with_dist]
        avg = sum(values) / len(values) if values else 0.0
        model_val = getattr(metrics.severity_distribution, field_name) or 0.0

        # Check over-reporting (>1.5x average)
        if avg > 0 and model_val > avg * OVER_REPORT_THRESHOLD:
            ratio = model_val / avg
            return f"Over-reports {severity_name} ({ratio:.1f}x average)"

        # Check under-reporting (<0.5x average)
        if avg > 0 and model_val < avg * UNDER_REPORT_THRESHOLD:
            ratio = model_val / avg if avg > 0 else 0
            return f"Under-reports {severity_name} ({ratio:.1f}x average)"

    return None


def _analyze_model_tendencies(
    metrics: ModelMetrics,
    all_metrics: list[ModelMetrics],
) -> ModelTendencies | None:
    """Analyze behavioral tendencies for a model.

    Compares model's metrics against median of all models to detect
    patterns like verbosity, terseness, high evidence citation.

    Args:
        metrics: Model to analyze.
        all_metrics: All models for comparison.

    Returns:
        ModelTendencies with detected patterns, or None if insufficient data.

    """
    if len(all_metrics) < MIN_MODELS_FOR_TENDENCY:
        return None

    # Calculate medians
    char_counts = [m.mean_char_count for m in all_metrics if m.mean_char_count is not None]

    median_char = _calculate_median(char_counts)

    tendency: str | None = None
    bias: str | None = None

    # Detect verbosity/terseness (AC5: require median > 0 before division)
    if median_char is not None and median_char > 0 and metrics.mean_char_count is not None:
        ratio = metrics.mean_char_count / median_char
        if ratio > VERBOSE_THRESHOLD:
            pct = (ratio - 1) * 100
            tendency = f"Verbose reports (+{pct:.0f}% avg length)"
        elif ratio < TERSE_THRESHOLD:
            pct = (1 - ratio) * 100
            tendency = f"Concise reports (-{pct:.0f}% avg length)"

    # Detect category bias
    bias = _detect_category_bias(metrics, all_metrics)

    # Return None if no tendencies detected
    if tendency is None and bias is None:
        return None

    return ModelTendencies(
        strength=None,
        tendency=tendency,
        bias=bias,
    )


def compare_models(
    base_dir: Path,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    workflow_id: str | None = None,
) -> ModelComparisonResult:
    """Compare all models by aggregating per-model metrics.

    Loads all evaluation records, groups by provider/model, aggregates
    metrics, performs tendency analysis, and returns comprehensive result.

    Args:
        base_dir: Base directory (docs/sprint-artifacts/).
        date_from: Optional start date filter.
        date_to: Optional end date filter.
        workflow_id: Optional workflow ID filter (e.g., "code-review").
            Story 13.10: Added for cross-phase filtering.

    Returns:
        ModelComparisonResult with all model data.

    """
    from datetime import UTC

    # Load and group records by model
    # Story 13.10: Pass workflow_id for phase-specific filtering
    records_by_model = _load_records_by_model(base_dir, date_from, date_to, workflow_id)

    notes: list[str] = []
    models: list[ModelMetrics] = []

    if not records_by_model:
        notes.append("No evaluation records found")
        return ModelComparisonResult(
            models=[],
            generated_at=datetime.now(UTC),
            total_records=0,
            date_range=None,
            notes=notes,
        )

    # First pass: aggregate base metrics
    for (provider, model), records in sorted(records_by_model.items()):
        metrics = _aggregate_model_metrics(records, provider, model)
        models.append(metrics)

    # Second pass: analyze tendencies (needs all models for comparison)
    updated_models: list[ModelMetrics] = []
    for m in models:
        tendencies = _analyze_model_tendencies(m, models)
        # Create new ModelMetrics with tendencies (frozen dataclass)
        updated = ModelMetrics(
            provider=m.provider,
            model=m.model,
            total_evaluations=m.total_evaluations,
            validator_count=m.validator_count,
            synthesizer_count=m.synthesizer_count,
            mean_findings_count=m.mean_findings_count,
            std_findings_count=m.std_findings_count,
            severity_distribution=m.severity_distribution,
            false_positive_rate=m.false_positive_rate,
            ground_truth_count=m.ground_truth_count,
            mean_char_count=m.mean_char_count,
            low_confidence=m.low_confidence,
            tendencies=tendencies,
        )
        updated_models.append(updated)

    # Calculate totals and date range
    all_records = [r for records in records_by_model.values() for r in records]
    total_records = len(all_records)

    date_range: tuple[datetime, datetime] | None = None
    if all_records:
        timestamps = [r.execution.start_time for r in all_records]
        date_range = (min(timestamps), max(timestamps))

    # Story 13.10 AC8: Note when records include multiple phases
    if workflow_id is None and all_records:
        unique_workflow_ids = {r.workflow.id for r in all_records}
        if len(unique_workflow_ids) > 1:
            phase_list = ", ".join(sorted(unique_workflow_ids))
            notes.append(f"Records include multiple phases: {phase_list}")

    # Note low confidence models
    low_conf_models = [m for m in updated_models if m.low_confidence]
    if low_conf_models:
        model_names = ", ".join(f"{m.provider}/{m.model}" for m in low_conf_models)
        notes.append(f"Low confidence (<{MIN_SAMPLES_FOR_INCLUSION} evals): {model_names}")

    # Note ground truth availability
    gt_models = [m for m in updated_models if m.ground_truth_count > 0]
    if gt_models:
        notes.append(f"FP Rate available for {len(gt_models)}/{len(updated_models)} models")
    else:
        notes.append("No ground truth data available (FP Rate not calculated)")

    return ModelComparisonResult(
        models=updated_models,
        generated_at=datetime.now(UTC),
        total_records=total_records,
        date_range=date_range,
        notes=notes,
    )


def generate_model_report_markdown(result: ModelComparisonResult) -> str:
    """Generate markdown model comparison report.

    Args:
        result: ModelComparisonResult with all data.

    Returns:
        Complete markdown report string.

    """
    lines: list[str] = []

    # Header
    lines.append("# Model Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {result.generated_at.isoformat()}")
    lines.append(f"**Total Records:** {result.total_records}")
    if result.date_range:
        lines.append(
            f"**Date Range:** {result.date_range[0].date()} - {result.date_range[1].date()}"
        )
    lines.append(f"**Models Analyzed:** {len(result.models)}")
    lines.append("")

    if not result.models:
        lines.append("No models to compare.")
        return "\n".join(lines)

    # Summary Table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Evals | FP Rate | Findings | Severity Focus | Low Conf? |")
    lines.append("|-------|-------|---------|----------|----------------|-----------|")

    for m in sorted(result.models, key=lambda x: x.total_evaluations, reverse=True):
        model_name = f"{m.provider}/{m.model}"
        fp_str = f"{m.false_positive_rate:.0%}" if m.false_positive_rate is not None else "-"
        findings_str = f"{m.mean_findings_count:.1f}" if m.mean_findings_count is not None else "-"

        # Determine severity focus (highest severity percentage)
        severity_focus = "-"
        if m.severity_distribution:
            dist = m.severity_distribution
            severities = [
                ("Critical", dist.critical_pct or 0),
                ("Major", dist.major_pct or 0),
                ("Minor", dist.minor_pct or 0),
                ("Nit", dist.nit_pct or 0),
            ]
            top = max(severities, key=lambda x: x[1])
            if top[1] > 0:
                severity_focus = f"{top[0]} ({top[1]:.0%})"

        low_conf_str = "⚠️" if m.low_confidence else ""

        lines.append(
            f"| {model_name} | {m.total_evaluations} | {fp_str} | "
            f"{findings_str} | {severity_focus} | {low_conf_str} |"
        )

    lines.append("")

    # Tendencies Section
    lines.append("## Model Tendencies")
    lines.append("")

    tendencies_found = False
    for m in result.models:
        if m.tendencies:
            tendencies_found = True
            model_name = f"{m.provider}/{m.model}"
            lines.append(f"### {model_name}")
            if m.tendencies.strength:
                lines.append(f"- **Strength:** {m.tendencies.strength}")
            if m.tendencies.tendency:
                lines.append(f"- **Tendency:** {m.tendencies.tendency}")
            if m.tendencies.bias:
                lines.append(f"- **Bias:** {m.tendencies.bias}")
            lines.append("")

    if not tendencies_found:
        lines.append("No significant tendencies detected across models.")
        lines.append("")

    # Severity Distribution
    lines.append("## Severity Distribution")
    lines.append("")
    lines.append("| Model | Critical | Major | Minor | Nit | Other |")
    lines.append("|-------|----------|-------|-------|-----|-------|")

    for m in result.models:
        model_name = f"{m.provider}/{m.model}"
        if m.severity_distribution:
            d = m.severity_distribution
            lines.append(
                f"| {model_name} | {d.critical_pct or 0:.0%} | {d.major_pct or 0:.0%} | "
                f"{d.minor_pct or 0:.0%} | {d.nit_pct or 0:.0%} | {d.other_pct or 0:.0%} |"
            )
        else:
            lines.append(f"| {model_name} | - | - | - | - | - |")

    lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("")
    for note in result.notes:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def generate_model_report_json(result: ModelComparisonResult) -> str:
    """Generate JSON model comparison report.

    Args:
        result: ModelComparisonResult with all data.

    Returns:
        JSON string with full data.

    """
    import json
    from typing import Any

    def serialize_model(m: ModelMetrics) -> dict[str, Any]:
        """Serialize a ModelMetrics to a JSON-compatible dict."""
        data: dict[str, Any] = {
            "provider": m.provider,
            "model": m.model,
            "total_evaluations": m.total_evaluations,
            "validator_count": m.validator_count,
            "synthesizer_count": m.synthesizer_count,
            "mean_findings_count": m.mean_findings_count,
            "std_findings_count": m.std_findings_count,
            "false_positive_rate": m.false_positive_rate,
            "ground_truth_count": m.ground_truth_count,
            "mean_char_count": m.mean_char_count,
            "low_confidence": m.low_confidence,
        }

        # Serialize severity distribution
        if m.severity_distribution:
            data["severity_distribution"] = {
                "critical_pct": m.severity_distribution.critical_pct,
                "major_pct": m.severity_distribution.major_pct,
                "minor_pct": m.severity_distribution.minor_pct,
                "nit_pct": m.severity_distribution.nit_pct,
                "other_pct": m.severity_distribution.other_pct,
            }
        else:
            data["severity_distribution"] = None

        # Serialize tendencies
        if m.tendencies:
            data["tendencies"] = {
                "strength": m.tendencies.strength,
                "tendency": m.tendencies.tendency,
                "bias": m.tendencies.bias,
            }
        else:
            data["tendencies"] = None

        return data

    output: dict[str, Any] = {
        "generated_at": result.generated_at.isoformat(),
        "total_records": result.total_records,
        "date_range": None,
        "notes": result.notes,
        "models": [serialize_model(m) for m in result.models],
    }

    if result.date_range:
        output["date_range"] = {
            "from": result.date_range[0].isoformat(),
            "to": result.date_range[1].isoformat(),
        }

    return json.dumps(output, indent=2)


# =============================================================================
# Main Comparison Function
# =============================================================================


def compare_workflow_variants(
    variant_a: str,
    variant_b: str,
    base_dir: Path,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> ComparisonResult:
    """Compare two workflow variants by aggregating metrics.

    Loads all evaluation records for the specified variants, aggregates
    metrics per variant, performs statistical significance testing, and
    returns a comprehensive comparison result.

    Args:
        variant_a: First variant name.
        variant_b: Second variant name.
        base_dir: Base directory (docs/sprint-artifacts/).
        date_from: Optional start date filter.
        date_to: Optional end date filter.

    Returns:
        ComparisonResult with all aggregated data and statistics.

    """
    from datetime import UTC

    notes: list[str] = []
    scipy_available = _check_scipy_available()

    if not scipy_available:
        logger.warning("scipy not installed, skipping significance testing")

    # Load records by variant
    records_by_variant = _load_records_by_variant(
        base_dir, variant_a, variant_b, date_from, date_to
    )

    records_a = records_by_variant[variant_a]
    records_b = records_by_variant[variant_b]

    # Check sample sizes
    if len(records_a) < MIN_SAMPLES_FOR_DISPLAY:
        logger.warning(
            "Insufficient samples for variant %s: %d (minimum: %d)",
            variant_a,
            len(records_a),
            MIN_SAMPLES_FOR_DISPLAY,
        )
        notes.append(
            f"Insufficient samples for variant {variant_a}: "
            f"{len(records_a)} (minimum: {MIN_SAMPLES_FOR_DISPLAY})"
        )

    if len(records_b) < MIN_SAMPLES_FOR_DISPLAY:
        logger.warning(
            "Insufficient samples for variant %s: %d (minimum: %d)",
            variant_b,
            len(records_b),
            MIN_SAMPLES_FOR_DISPLAY,
        )
        notes.append(
            f"Insufficient samples for variant {variant_b}: "
            f"{len(records_b)} (minimum: {MIN_SAMPLES_FOR_DISPLAY})"
        )

    if len(records_a) == 0:
        notes.append(f"No data for variant {variant_a}")

    if len(records_b) == 0:
        notes.append(f"No data for variant {variant_b}")

    # Aggregate metrics
    metrics_a = _aggregate_variant_metrics(records_a, variant_a)
    metrics_b = _aggregate_variant_metrics(records_b, variant_b)

    # Build metric comparisons with significance tests
    metric_comparisons: list[MetricComparison] = []

    # Agreement Score (synthesizer records)
    synth_a = [r for r in records_a if r.evaluator.role == EvaluatorRole.SYNTHESIZER]
    synth_b = [r for r in records_b if r.evaluator.role == EvaluatorRole.SYNTHESIZER]

    agreement_a = [r.consensus.agreement_score for r in synth_a if r.consensus is not None]
    agreement_b = [r.consensus.agreement_score for r in synth_b if r.consensus is not None]

    if scipy_available:
        p_val, sig = _calculate_significance(agreement_a, agreement_b)
    else:
        p_val, sig = None, None
    delta = (
        metrics_a.mean_agreement_score - metrics_b.mean_agreement_score
        if metrics_a.mean_agreement_score is not None and metrics_b.mean_agreement_score is not None
        else None
    )

    metric_comparisons.append(
        MetricComparison(
            name="Agreement Score",
            value_a=metrics_a.mean_agreement_score,
            std_a=metrics_a.std_agreement_score,
            count_a=len(agreement_a),
            value_b=metrics_b.mean_agreement_score,
            std_b=metrics_b.std_agreement_score,
            count_b=len(agreement_b),
            delta=delta,
            p_value=p_val,
            significant=sig,
            format_spec=".2f",
        )
    )

    # Findings Count (validator records)
    val_a = [r for r in records_a if r.evaluator.role == EvaluatorRole.VALIDATOR]
    val_b = [r for r in records_b if r.evaluator.role == EvaluatorRole.VALIDATOR]

    findings_a = [float(r.findings.total_count) for r in val_a if r.findings is not None]
    findings_b = [float(r.findings.total_count) for r in val_b if r.findings is not None]

    if scipy_available:
        p_val, sig = _calculate_significance(findings_a, findings_b)
    else:
        p_val, sig = None, None
    delta = (
        metrics_a.mean_findings_count - metrics_b.mean_findings_count
        if metrics_a.mean_findings_count is not None and metrics_b.mean_findings_count is not None
        else None
    )

    metric_comparisons.append(
        MetricComparison(
            name="Findings Count",
            value_a=metrics_a.mean_findings_count,
            std_a=metrics_a.std_findings_count,
            count_a=len(findings_a),
            value_b=metrics_b.mean_findings_count,
            std_b=metrics_b.std_findings_count,
            count_b=len(findings_b),
            delta=delta,
            p_value=p_val,
            significant=sig,
            format_spec=".1f",
        )
    )

    # Actionable Ratio (synthesizer records)
    actionable_a = [r.quality.actionable_ratio for r in synth_a if r.quality is not None]
    actionable_b = [r.quality.actionable_ratio for r in synth_b if r.quality is not None]

    if scipy_available:
        p_val, sig = _calculate_significance(actionable_a, actionable_b)
    else:
        p_val, sig = None, None
    delta = (
        metrics_a.mean_actionable_ratio - metrics_b.mean_actionable_ratio
        if metrics_a.mean_actionable_ratio is not None
        and metrics_b.mean_actionable_ratio is not None
        else None
    )

    metric_comparisons.append(
        MetricComparison(
            name="Actionable Ratio",
            value_a=metrics_a.mean_actionable_ratio,
            std_a=metrics_a.std_actionable_ratio,
            count_a=len(actionable_a),
            value_b=metrics_b.mean_actionable_ratio,
            std_b=metrics_b.std_actionable_ratio,
            count_b=len(actionable_b),
            delta=delta,
            p_value=p_val,
            significant=sig,
            format_spec=".2%",
        )
    )

    # Specificity Score (synthesizer records)
    specificity_a = [r.quality.specificity_score for r in synth_a if r.quality is not None]
    specificity_b = [r.quality.specificity_score for r in synth_b if r.quality is not None]

    if scipy_available:
        p_val, sig = _calculate_significance(specificity_a, specificity_b)
    else:
        p_val, sig = None, None
    delta = (
        metrics_a.mean_specificity - metrics_b.mean_specificity
        if metrics_a.mean_specificity is not None and metrics_b.mean_specificity is not None
        else None
    )

    metric_comparisons.append(
        MetricComparison(
            name="Specificity Score",
            value_a=metrics_a.mean_specificity,
            std_a=metrics_a.std_specificity,
            count_a=len(specificity_a),
            value_b=metrics_b.mean_specificity,
            std_b=metrics_b.std_specificity,
            count_b=len(specificity_b),
            delta=delta,
            p_value=p_val,
            significant=sig,
            format_spec=".2f",
        )
    )

    logger.info("Comparison complete: %d vs %d records", len(records_a), len(records_b))

    return ComparisonResult(
        variant_a=metrics_a,
        variant_b=metrics_b,
        metrics=metric_comparisons,
        generated_at=datetime.now(UTC),
        notes=notes,
        scipy_available=scipy_available,
    )


# =============================================================================
# Multi-Phase Comparison (Story 13.10)
# =============================================================================


@dataclass(frozen=True)
class MultiPhaseResult:
    """Result of multi-phase model comparison.

    Story 13.10: Segments model metrics by workflow phase.

    Attributes:
        phases: Dict mapping workflow_id to ModelComparisonResult.
        total_records: Total records across all phases.
        generated_at: Report generation timestamp.
        notes: Any warnings or info notes.

    """

    phases: dict[str, ModelComparisonResult]
    total_records: int
    generated_at: datetime
    notes: list[str]


def compare_models_by_phase(
    base_dir: Path,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    workflow_ids: list[str] | None = None,
) -> MultiPhaseResult:
    """Compare models segmented by workflow phase.

    Story 13.10: Generates separate model comparison results for each
    workflow phase (e.g., "validate-story" vs "code-review").

    Args:
        base_dir: Base directory (docs/sprint-artifacts/).
        date_from: Optional start date filter.
        date_to: Optional end date filter.
        workflow_ids: Optional list of workflow IDs to include.
            If None, discovers all unique workflow IDs from records.

    Returns:
        MultiPhaseResult with per-phase model comparisons.

    """
    from datetime import UTC

    notes: list[str] = []
    phases: dict[str, ModelComparisonResult] = {}
    total_records = 0

    # If no workflow_ids specified, discover them from all records
    if workflow_ids is None:
        filters = RecordFilters(date_from=date_from, date_to=date_to)
        summaries = list_evaluation_records(base_dir, filters)

        # Extract unique workflow IDs
        discovered_ids: set[str] = set()
        for summary in summaries:
            if summary.workflow_id:
                discovered_ids.add(summary.workflow_id)

        if not discovered_ids:
            notes.append("No workflow IDs found in records")
            return MultiPhaseResult(
                phases={},
                total_records=0,
                generated_at=datetime.now(UTC),
                notes=notes,
            )

        workflow_ids = sorted(discovered_ids)
        logger.info("Discovered workflow phases: %s", workflow_ids)

    # Generate comparison for each phase
    for wf_id in workflow_ids:
        logger.debug("Generating comparison for workflow: %s", wf_id)
        result = compare_models(
            base_dir=base_dir,
            date_from=date_from,
            date_to=date_to,
            workflow_id=wf_id,
        )
        phases[wf_id] = result
        total_records += result.total_records

    if not phases:
        notes.append("No records found for specified workflow phases")

    return MultiPhaseResult(
        phases=phases,
        total_records=total_records,
        generated_at=datetime.now(UTC),
        notes=notes,
    )


def generate_multi_phase_report_markdown(result: MultiPhaseResult) -> str:
    """Generate Markdown report for multi-phase model comparison.

    Story 13.10: Creates a report with separate sections for each workflow
    phase, allowing easy comparison between validation and code review metrics.

    Args:
        result: MultiPhaseResult from compare_models_by_phase().

    Returns:
        Markdown-formatted report string.

    """
    lines: list[str] = []

    # Header
    lines.append("# Multi-Phase Model Comparison Report")
    lines.append("")
    lines.append(f"Generated: {result.generated_at.isoformat()}")
    lines.append(f"Total Records: {result.total_records}")
    lines.append(f"Phases Analyzed: {len(result.phases)}")
    lines.append("")

    # Notes
    if result.notes:
        lines.append("## Notes")
        lines.append("")
        for note in result.notes:
            lines.append(f"- {note}")
        lines.append("")

    # Per-phase sections
    for workflow_id, phase_result in sorted(result.phases.items()):
        lines.append(f"## Phase: {workflow_id}")
        lines.append("")
        lines.append(f"Records: {phase_result.total_records}")

        if phase_result.date_range:
            from_dt, to_dt = phase_result.date_range
            lines.append(
                f"Date Range: {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}"
            )
        lines.append("")

        if not phase_result.models:
            lines.append("*No model data available for this phase*")
            lines.append("")
            continue

        # Model summary table
        lines.append("### Model Summary")
        lines.append("")
        lines.append("| Model | Evaluations | Findings (μ±σ) | Severity | Tendencies |")
        lines.append("|-------|-------------|----------------|----------|------------|")

        for m in sorted(phase_result.models, key=lambda x: (x.provider, x.model)):
            # Format findings
            if m.mean_findings_count is not None:
                std = m.std_findings_count or 0
                findings = f"{m.mean_findings_count:.1f}±{std:.1f}"
            else:
                findings = "N/A"

            # Format severity distribution
            if m.severity_distribution:
                sev = m.severity_distribution
                severity = f"C:{sev.critical_pct:.0%} M:{sev.major_pct:.0%}"
            else:
                severity = "N/A"

            # Format tendencies
            if m.tendencies:
                tendencies = f"{m.tendencies.tendency} ({m.tendencies.strength})"
            else:
                tendencies = "-"

            conf_marker = "*" if m.low_confidence else ""
            lines.append(
                f"| {m.provider}/{m.model}{conf_marker} | {m.total_evaluations} | "
                f"{findings} | {severity} | {tendencies} |"
            )

        lines.append("")

        # Phase notes
        if phase_result.notes:
            lines.append("**Phase Notes:**")
            for note in phase_result.notes:
                lines.append(f"- {note}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
