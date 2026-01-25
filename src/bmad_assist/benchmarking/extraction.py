"""LLM-based metrics extraction from validator output.

This module provides asynchronous extraction of qualitative metrics
from validation outputs using an LLM session. Approximately 45% of
benchmarking metrics require LLM judgment: findings classification,
formality assessment, anomaly detection, etc.

Public API:
    extract_metrics_async: Primary async API for parallel execution
    extract_metrics: Sync wrapper using asyncio.run()
    ExtractionContext: Context dataclass for extraction
    ExtractedMetrics: Result dataclass with all extracted fields
    MetricsExtractionError: Exception for extraction failures
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bmad_assist.benchmarking.schema import (
    BenchmarkingError,
    FindingsExtracted,
    LinguisticFingerprint,
    QualitySignals,
)
from bmad_assist.core.exceptions import ProviderError
from bmad_assist.core.types import EpicId

if TYPE_CHECKING:
    from bmad_assist.benchmarking.collector import LinguisticMetrics

logger = logging.getLogger(__name__)

# =============================================================================
# Extraction Prompt Template Loading (BMAD-agnostic)
# =============================================================================

# Cache for loaded prompt template
_extraction_prompt_cache: str | None = None


def _load_extraction_prompt_template() -> str:
    """Load extraction prompt template from package resource.

    The prompt is loaded from bmad_assist/benchmarking/prompts/extraction.xml.
    This is a package resource bundled with bmad-assist, NOT a BMAD workflow.

    Returns:
        Prompt template string with {validator_output}, {story_epic}, {story_num}
        placeholders.

    Raises:
        FileNotFoundError: If prompt file is missing from package.

    """
    global _extraction_prompt_cache

    if _extraction_prompt_cache is not None:
        return _extraction_prompt_cache

    # Use importlib.resources for Python 3.9+ compatible resource loading
    from importlib import resources

    try:
        # Python 3.11+ preferred API
        prompt_file = resources.files("bmad_assist.benchmarking.prompts").joinpath("extraction.xml")
        _extraction_prompt_cache = prompt_file.read_text(encoding="utf-8")
    except (TypeError, AttributeError):
        # Fallback for older Python versions
        import importlib.resources as pkg_resources

        with pkg_resources.open_text("bmad_assist.benchmarking.prompts", "extraction.xml") as f:
            _extraction_prompt_cache = f.read()

    logger.debug("Loaded extraction prompt template from package resource")
    return _extraction_prompt_cache


# Valid severity and category keys for validation
VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "major", "minor", "nit"})
VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "security",
        "performance",
        "correctness",
        "completeness",
        "clarity",
        "testability",
    }
)


# =============================================================================
# Exceptions
# =============================================================================


class MetricsExtractionError(BenchmarkingError):
    """Metrics extraction from LLM output failed.

    Raised when:
    - LLM returns invalid JSON after max retries
    - Required fields are missing from response
    - Provider invocation fails

    Attributes:
        attempts: Number of extraction attempts made.
        last_error: Last error message before failure.

    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_error: str | None = None,
    ) -> None:
        """Initialize MetricsExtractionError with attempts and last error."""
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# =============================================================================
# Dataclasses - Context and Intermediate Results
# =============================================================================


@dataclass(frozen=True)
class ExtractionContext:
    """Context for metrics extraction.

    Attributes:
        story_epic: Epic number for story being analyzed.
        story_num: Story number.
        timestamp: UTC-aware timestamp.
        project_root: Project root path (required, from caller).
        max_retries: Maximum retry attempts for failed extraction.
        timeout_seconds: Timeout for LLM invocation.
        provider: LLM provider to use (default: claude).
        model: Model for extraction (default: haiku - fast/cheap).
        settings_file: Optional settings file path for custom provider config.

    """

    story_epic: EpicId
    story_num: int | str
    timestamp: datetime
    project_root: Path
    max_retries: int = 3
    timeout_seconds: int = 120
    provider: str = "claude"
    model: str = "haiku"
    settings_file: str | None = None


@dataclass(frozen=True)
class FindingsData:
    """Extracted findings data from LLM analysis."""

    total_count: int
    by_severity: dict[str, int]
    by_category: dict[str, int]
    has_fix_count: int
    has_location_count: int
    has_evidence_count: int


@dataclass(frozen=True)
class LinguisticData:
    """LLM-assessed linguistic characteristics."""

    formality_score: float
    sentiment: str


@dataclass(frozen=True)
class QualityData:
    """LLM-assessed quality signals."""

    actionable_ratio: float
    specificity_score: float
    evidence_quality: float
    internal_consistency: float


@dataclass(frozen=True)
class ExtractedMetrics:
    """All LLM-extracted metrics from a validation output.

    Provides conversion methods to schema models for integration
    with the benchmarking storage layer.

    """

    findings: FindingsData
    complexity_flags: dict[str, bool]
    linguistic: LinguisticData
    quality: QualityData
    anomalies: tuple[str, ...]
    extracted_at: datetime

    def to_findings_extracted(self) -> FindingsExtracted:
        """Convert to FindingsExtracted schema model."""
        return FindingsExtracted(
            total_count=self.findings.total_count,
            by_severity=self.findings.by_severity,
            by_category=self.findings.by_category,
            has_fix_count=self.findings.has_fix_count,
            has_location_count=self.findings.has_location_count,
            has_evidence_count=self.findings.has_evidence_count,
        )

    def to_linguistic_fingerprint(
        self,
        deterministic: LinguisticMetrics,
    ) -> LinguisticFingerprint:
        """Merge LLM-assessed values with deterministic values.

        Args:
            deterministic: Deterministic metrics from collector.py

        Returns:
            Complete LinguisticFingerprint with all fields populated.

        """
        return LinguisticFingerprint(
            avg_sentence_length=deterministic.avg_sentence_length,
            vocabulary_richness=deterministic.vocabulary_richness,
            flesch_reading_ease=deterministic.flesch_reading_ease,
            vague_terms_count=deterministic.vague_terms_count,
            formality_score=self.linguistic.formality_score,
            sentiment=self.linguistic.sentiment,
        )

    def to_quality_signals(self) -> QualitySignals:
        """Convert to QualitySignals schema model.

        Note: follows_template is deterministic and set separately
        by the orchestrator based on marker detection.

        """
        return QualitySignals(
            actionable_ratio=self.quality.actionable_ratio,
            specificity_score=self.quality.specificity_score,
            evidence_quality=self.quality.evidence_quality,
            follows_template=True,  # Set by orchestrator based on marker detection
            internal_consistency=self.quality.internal_consistency,
        )

    def to_complexity_flags(self) -> dict[str, bool]:
        """Return complexity flags dictionary."""
        return dict(self.complexity_flags)


# =============================================================================
# JSON Parsing and Validation
# =============================================================================


def _parse_extraction_response(
    raw_json: str,
    timestamp: datetime,
) -> ExtractedMetrics:
    """Parse LLM response into ExtractedMetrics.

    Args:
        raw_json: Raw JSON string from LLM.
        timestamp: Timestamp for extracted_at field.

    Returns:
        ExtractedMetrics dataclass.

    Raises:
        json.JSONDecodeError: If JSON is invalid.
        KeyError: If required fields are missing.
        ValueError: If field values are out of range.

    """
    # Strip any markdown code block wrappers and extract JSON
    json_str = raw_json.strip()

    # Handle multiple possible formats:
    # 1. Raw JSON: {"findings": ...}
    # 2. Markdown wrapped: ```json\n{...}\n```
    # 3. Markdown with text before: "Here's the analysis:\n```json\n{...}\n```"
    # 4. Markdown with text after: "```json\n{...}\n```\nLet me know if..."
    # 5. Multiple assistant messages concatenated

    # Try to find JSON block between ``` markers
    if "```" in json_str:
        # Find the start of the JSON block
        start_marker = json_str.find("```")
        if start_marker != -1:
            # Find the content after the opening ```
            content_start = json_str.find("\n", start_marker)
            if content_start != -1:
                content_start += 1  # Skip the newline
                # Find the closing ```
                end_marker = json_str.find("\n```", content_start)
                if end_marker != -1:
                    json_str = json_str[content_start:end_marker]
                else:
                    # No closing ```, try to find just ``` at end
                    end_marker = json_str.rfind("```")
                    if end_marker > content_start:
                        json_str = json_str[content_start:end_marker].rstrip()

    # If still starts with ```, use old line-based stripping
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        # Find the closing ```
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        # Remove first line (```json) and last line (```)
        json_str = "\n".join(lines[1:end_idx])

    # Strip whitespace
    json_str = json_str.strip()

    # Final fallback: extract JSON object by finding balanced braces
    # This handles cases where there's extra text before/after the JSON
    if not json_str.startswith("{"):
        # Find the first { that could start our JSON
        brace_start = json_str.find('{"findings"')
        if brace_start == -1:
            brace_start = json_str.find("{")
        if brace_start != -1:
            json_str = json_str[brace_start:]

    # Find the matching closing brace
    if json_str.startswith("{"):
        brace_count = 0
        end_pos = -1
        for i, char in enumerate(json_str):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break
        if end_pos != -1 and end_pos < len(json_str) - 1:
            # There's extra content after the JSON - trim it
            json_str = json_str[: end_pos + 1]

    # Log for debugging if parsing fails
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.debug(
            "JSON parse failed. Raw length: %d, Stripped length: %d, First 200: %s, Last 200: %s",
            len(raw_json),
            len(json_str),
            repr(json_str[:200]),
            repr(json_str[-200:]),
        )
        raise

    # Extract findings (required section)
    findings_raw = data["findings"]

    # Validate severity keys per AC3 (log warnings for invalid keys)
    by_severity = dict(findings_raw["by_severity"])
    unknown_severities = set(by_severity.keys()) - VALID_SEVERITIES
    if unknown_severities:
        logger.warning("Unknown severity keys in extraction response: %s", unknown_severities)

    # Validate category keys per AC3 (log warnings for invalid keys)
    by_category = dict(findings_raw["by_category"])
    unknown_categories = set(by_category.keys()) - VALID_CATEGORIES
    if unknown_categories:
        logger.warning("Unknown category keys in extraction response: %s", unknown_categories)

    findings = FindingsData(
        total_count=int(findings_raw["total_count"]),
        by_severity=by_severity,
        by_category=by_category,
        has_fix_count=int(findings_raw["has_fix_count"]),
        has_location_count=int(findings_raw["has_location_count"]),
        has_evidence_count=int(findings_raw["has_evidence_count"]),
    )

    # Extract complexity flags (required section)
    complexity_flags = dict(data["complexity_flags"])

    # Extract linguistic with semantic validation
    linguistic_raw = data["linguistic"]
    formality_score = float(linguistic_raw["formality_score"])
    if not 0.0 <= formality_score <= 1.0:
        raise ValueError(f"formality_score must be 0.0-1.0, got {formality_score}")

    sentiment = str(linguistic_raw["sentiment"])
    valid_sentiments = {"positive", "neutral", "negative", "mixed"}
    if sentiment not in valid_sentiments:
        raise ValueError(f"sentiment must be one of {valid_sentiments}, got {sentiment!r}")

    linguistic = LinguisticData(formality_score=formality_score, sentiment=sentiment)

    # Extract quality signals with semantic validation
    quality_raw = data["quality_signals"]
    quality_fields = [
        "actionable_ratio",
        "specificity_score",
        "evidence_quality",
        "internal_consistency",
    ]
    for field in quality_fields:
        value = float(quality_raw[field])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be 0.0-1.0, got {value}")

    quality = QualityData(
        actionable_ratio=float(quality_raw["actionable_ratio"]),
        specificity_score=float(quality_raw["specificity_score"]),
        evidence_quality=float(quality_raw["evidence_quality"]),
        internal_consistency=float(quality_raw["internal_consistency"]),
    )

    # Extract anomalies (may be empty)
    anomalies = tuple(str(a) for a in data.get("anomalies", []))

    # Log unmapped fields as warnings
    expected_keys = {"findings", "complexity_flags", "linguistic", "quality_signals", "anomalies"}
    unmapped = set(data.keys()) - expected_keys
    if unmapped:
        logger.warning("Unmapped fields in extraction response: %s", unmapped)

    logger.debug("Parsed extraction response: %d findings", findings.total_count)

    return ExtractedMetrics(
        findings=findings,
        complexity_flags=complexity_flags,
        linguistic=linguistic,
        quality=quality,
        anomalies=anomalies,
        extracted_at=timestamp,
    )


# =============================================================================
# Provider Invocation
# =============================================================================


def _build_extraction_prompt(
    validator_output: str,
    context: ExtractionContext,
) -> str:
    """Build extraction prompt using package resource template.

    Loads template from bmad_assist/benchmarking/prompts/extraction.xml.
    This is BMAD-agnostic - no external workflow dependencies.

    Args:
        validator_output: Raw validator output to analyze.
        context: ExtractionContext with story info.

    Returns:
        Complete prompt for metrics extraction.

    """
    template = _load_extraction_prompt_template()
    return template.format(
        validator_output=validator_output,
        story_epic=context.story_epic,
        story_num=context.story_num,
    )


async def extract_metrics_async(
    raw_output: str,
    context: ExtractionContext,
) -> ExtractedMetrics:
    """Extract LLM-assessed metrics from validator output (async).

    Primary async API for parallel execution with synthesis.
    Use this when calling from async context (e.g., orchestrator).

    Args:
        raw_output: Raw validator output text to analyze.
        context: ExtractionContext with story info and config.

    Returns:
        ExtractedMetrics with all LLM-extracted fields.

    Raises:
        MetricsExtractionError: If extraction fails after max retries.

    """
    from bmad_assist.providers import get_provider

    # Build extraction prompt
    prompt = _build_extraction_prompt(raw_output, context)

    # Get provider
    provider = get_provider(context.provider)

    # Retry loop
    last_error: str | None = None
    for attempt in range(context.max_retries):
        try:
            # Add clarification on retry
            retry_prompt = prompt
            if attempt > 0 and last_error:
                retry_hint = (
                    f"Previous attempt failed: {last_error}. "
                    "Please output ONLY valid JSON matching the schema."
                )
                retry_prompt = f"{prompt}\n\n<error>{retry_hint}</error>"

            # Invoke LLM with allowed_tools=[] to prevent file modification
            settings_path = Path(context.settings_file) if context.settings_file else None
            result = await asyncio.to_thread(
                provider.invoke,
                retry_prompt,
                model=context.model,
                timeout=context.timeout_seconds,
                settings_file=settings_path,
                allowed_tools=[],  # Extraction is read-only
            )

            if result.exit_code != 0:
                last_error = f"Provider returned exit code {result.exit_code}: {result.stderr}"
                logger.warning("Extraction attempt %d failed: %s", attempt + 1, last_error)
                continue

            # Parse and validate JSON response
            return _parse_extraction_response(result.stdout, context.timestamp)

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, last_error)
        except KeyError as e:
            last_error = f"Missing required field: {e}"
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, last_error)
        except ValueError as e:
            last_error = f"Validation error: {e}"
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, last_error)
        except ProviderError as e:
            last_error = f"Provider error: {e}"
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, last_error)

    raise MetricsExtractionError(
        f"Extraction failed after {context.max_retries} attempts",
        attempts=context.max_retries,
        last_error=last_error,
    )


def extract_metrics(
    raw_output: str,
    context: ExtractionContext,
) -> ExtractedMetrics:
    """Extract LLM-assessed metrics from validator output (sync wrapper).

    Sync wrapper for CLI or non-async contexts. Uses asyncio.run().
    For parallel execution, use extract_metrics_async() instead.

    Args:
        raw_output: Raw validator output text to analyze.
        context: ExtractionContext with story info and config.

    Returns:
        ExtractedMetrics with all LLM-extracted fields.

    Raises:
        MetricsExtractionError: If extraction fails after max retries.

    """
    return asyncio.run(extract_metrics_async(raw_output, context))
