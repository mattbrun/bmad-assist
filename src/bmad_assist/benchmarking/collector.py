"""Deterministic metrics collector for LLM output analysis.

This module provides functions for calculating reproducible metrics from
raw LLM output text without requiring LLM calls. Approximately 55% of
benchmarking metrics are deterministic and can be computed with regex
patterns, character counts, and readability scores.

Public API:
    collect_deterministic_metrics: Main entry point - calculates all metrics
    calculate_structure_metrics: Structural analysis (headings, code blocks, lists)
    calculate_linguistic_metrics: Linguistic analysis (sentence length, vocabulary)
    calculate_reasoning_signals: Reasoning pattern detection (citations, conditionals)
    CollectorContext: Context dataclass for collection
    DeterministicMetrics: Result dataclass composing all metrics
    StructureMetrics: Structural metrics dataclass
    LinguisticMetrics: Linguistic metrics dataclass
    ReasoningSignals: Reasoning signals dataclass
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

import textstat

from bmad_assist.benchmarking.schema import (
    LinguisticFingerprint,
    OutputAnalysis,
    ReasoningPatterns,
)
from bmad_assist.core.types import EpicId

logger = logging.getLogger(__name__)

# =============================================================================
# Regex Patterns - Module-level compiled constants
# =============================================================================

# Structure patterns
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```")
LIST_ITEM_PATTERN = re.compile(r"^(\s*)([-*+]|\d+\.)\s", re.MULTILINE)

# Linguistic patterns
SENTENCE_END_PATTERN = re.compile(r"[.!?]+(?:\s+|\n)")
WORD_PATTERN = re.compile(r"\b\w+\b")
VAGUE_TERMS_PATTERN = re.compile(
    r"\b(some|various|etc|many|few|several|often|sometimes)\b",
    re.IGNORECASE,
)

# Reasoning patterns
PRD_PATTERN = re.compile(r"\b(PRD|prd)\b|docs/prd\.md")
ARCH_PATTERN = re.compile(r"\b[Aa]rchitecture\b|docs/architecture\.md")
STORY_SECTION_PATTERN = re.compile(r"AC-?\d+|Task\s+\d+|#\d+\.\d+")
CONDITIONAL_PATTERN = re.compile(
    r"\b(if|when|unless|je[sś]li|gdy|kiedy)\b",
    re.IGNORECASE,
)
UNCERTAINTY_PATTERN = re.compile(
    r"\b(mo[zż]e|perhaps|possibly|unclear|might|could|may)\b",
    re.IGNORECASE,
)
CONFIDENCE_PATTERN = re.compile(
    r"\b(definitely|clearly|must|always|certainly|zawsze|obviously)\b",
    re.IGNORECASE,
)


# =============================================================================
# Dataclasses - Intermediate result containers
# =============================================================================


@dataclass(frozen=True)
class CollectorContext:
    """Context for metrics collection.

    Provides story identification and timestamp for the collection session.
    """

    story_epic: EpicId
    story_num: int | str
    timestamp: datetime


@dataclass(frozen=True)
class StructureMetrics:
    """Structural analysis of LLM output - deterministic metrics.

    Attributes:
        char_count: Total character count.
        heading_count: Number of markdown headings (# to ######).
        list_depth_max: Maximum nesting depth of lists (0 if no lists).
        code_block_count: Number of code blocks (paired ```).
        sections_detected: Tuple of heading text from level 1-3 headings.

    """

    char_count: int
    heading_count: int
    list_depth_max: int
    code_block_count: int
    sections_detected: tuple[str, ...]


@dataclass(frozen=True)
class LinguisticMetrics:
    """Linguistic analysis - deterministic metrics.

    Attributes:
        avg_sentence_length: Average words per sentence.
        vocabulary_richness: Unique words / total words (type-token ratio).
        flesch_reading_ease: Flesch readability score.
        vague_terms_count: Count of vague terms (some, various, etc.).

    """

    avg_sentence_length: float
    vocabulary_richness: float
    flesch_reading_ease: float
    vague_terms_count: int


@dataclass(frozen=True)
class ReasoningSignals:
    """Reasoning pattern detection - deterministic regex-based signals.

    These are regex-based signals that Story 13.3's LLM extraction can
    use or override. Boolean values indicate text pattern matches,
    not semantic correctness.

    Attributes:
        cites_prd: True if PRD pattern found in text.
        cites_architecture: True if architecture pattern found.
        cites_story_sections: True if AC/Task patterns found.
        uses_conditionals: True if conditional patterns found.
        uncertainty_phrases_count: Count of uncertainty phrases.
        confidence_phrases_count: Count of confidence phrases.

    """

    cites_prd: bool
    cites_architecture: bool
    cites_story_sections: bool
    uses_conditionals: bool
    uncertainty_phrases_count: int
    confidence_phrases_count: int


@dataclass(frozen=True)
class DeterministicMetrics:
    """Composite of all deterministic metrics.

    Combines structure, linguistic, and reasoning metrics collected
    from a single LLM output.
    """

    structure: StructureMetrics
    linguistic: LinguisticMetrics
    reasoning: ReasoningSignals
    collected_at: datetime

    def to_output_analysis(self) -> OutputAnalysis:
        """Convert structure metrics to OutputAnalysis schema model."""
        return OutputAnalysis(
            char_count=self.structure.char_count,
            heading_count=self.structure.heading_count,
            list_depth_max=self.structure.list_depth_max,
            code_block_count=self.structure.code_block_count,
            sections_detected=list(self.structure.sections_detected),
            anomalies=[],
        )

    def to_linguistic_fingerprint(self) -> LinguisticFingerprint:
        """Convert linguistic metrics to LinguisticFingerprint schema model.

        Note: formality_score and sentiment are LLM-assessed fields,
        set to placeholder values here for Story 13.3 to populate.
        """
        return LinguisticFingerprint(
            avg_sentence_length=self.linguistic.avg_sentence_length,
            vocabulary_richness=self.linguistic.vocabulary_richness,
            flesch_reading_ease=self.linguistic.flesch_reading_ease,
            vague_terms_count=self.linguistic.vague_terms_count,
            formality_score=0.0,
            sentiment="neutral",
        )

    def to_reasoning_patterns(self) -> ReasoningPatterns:
        """Convert reasoning signals to ReasoningPatterns schema model."""
        return ReasoningPatterns(
            cites_prd=self.reasoning.cites_prd,
            cites_architecture=self.reasoning.cites_architecture,
            cites_story_sections=self.reasoning.cites_story_sections,
            uses_conditionals=self.reasoning.uses_conditionals,
            uncertainty_phrases_count=self.reasoning.uncertainty_phrases_count,
            confidence_phrases_count=self.reasoning.confidence_phrases_count,
        )


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _calculate_list_depth(content: str) -> int:
    """Calculate maximum nesting depth of lists.

    Uses relative depth detection to handle mixed indent styles (2-space,
    4-space, tabs). Each increase in indent = +1 level. Tabs are treated
    as 4 spaces.

    Args:
        content: Markdown content to analyze.

    Returns:
        Maximum list depth (0 if no lists found).

    """
    max_depth = 0
    indent_stack: list[int] = []

    for match in LIST_ITEM_PATTERN.finditer(content):
        indent = len(match.group(1).replace("\t", "    "))

        # Pop stack until we find a smaller indent
        while indent_stack and indent <= indent_stack[-1]:
            indent_stack.pop()

        indent_stack.append(indent)
        max_depth = max(max_depth, len(indent_stack))

    return max_depth


def _extract_sections(content: str) -> tuple[str, ...]:
    """Extract heading text from level 1-3 headings.

    Args:
        content: Markdown content to analyze.

    Returns:
        Tuple of heading texts from # to ### headings.

    """
    sections: list[str] = []
    for match in HEADING_PATTERN.finditer(content):
        level = len(match.group(1))
        if level <= 3:
            sections.append(match.group(2).strip())
    return tuple(sections)


# =============================================================================
# Public API Functions
# =============================================================================


def calculate_structure_metrics(content: str) -> StructureMetrics:
    """Calculate structural metrics from content.

    Analyzes markdown structure: character count, headings, lists, code blocks.

    Args:
        content: Raw LLM output text.

    Returns:
        StructureMetrics dataclass with all structure-related fields.

    """
    char_count = len(content)

    # Count all headings (# to ######)
    heading_matches = HEADING_PATTERN.findall(content)
    heading_count = len(heading_matches)

    # Count code blocks (paired ```)
    code_block_matches = CODE_BLOCK_PATTERN.findall(content)
    code_block_count = len(code_block_matches) // 2

    # Calculate list depth
    list_depth_max = _calculate_list_depth(content)

    # Extract sections from level 1-3 headings
    sections_detected = _extract_sections(content)

    logger.debug(
        "Calculated structure metrics: %d chars, %d headings, %d code blocks, depth %d",
        char_count,
        heading_count,
        code_block_count,
        list_depth_max,
    )

    return StructureMetrics(
        char_count=char_count,
        heading_count=heading_count,
        list_depth_max=list_depth_max,
        code_block_count=code_block_count,
        sections_detected=sections_detected,
    )


def calculate_linguistic_metrics(content: str) -> LinguisticMetrics:
    """Calculate linguistic metrics from content.

    Analyzes sentence structure, vocabulary, readability, and vague terms.

    Args:
        content: Raw LLM output text.

    Returns:
        LinguisticMetrics dataclass with all linguistic fields.

    Note:
        Returns zeroed metrics for empty content.
        Catches textstat exceptions and returns flesch_reading_ease=0.0.

    """
    # Handle empty content
    if not content.strip():
        return LinguisticMetrics(
            avg_sentence_length=0.0,
            vocabulary_richness=0.0,
            flesch_reading_ease=0.0,
            vague_terms_count=0,
        )

    # Extract words (case-insensitive, alphanumeric only)
    words = WORD_PATTERN.findall(content.lower())
    total_words = len(words)

    # Handle single word or no words
    if total_words == 0:
        return LinguisticMetrics(
            avg_sentence_length=0.0,
            vocabulary_richness=0.0,
            flesch_reading_ease=0.0,
            vague_terms_count=0,
        )

    # Calculate vocabulary richness
    unique_words = len(set(words))
    vocabulary_richness = unique_words / total_words

    # Calculate average sentence length
    # Split on sentence-ending punctuation followed by whitespace or newline
    sentences = SENTENCE_END_PATTERN.split(content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if sentences:
        sentence_word_counts = [len(WORD_PATTERN.findall(s)) for s in sentences]
        total_sentence_words = sum(sentence_word_counts)
        avg_sentence_length = total_sentence_words / len(sentences)
    else:
        # No sentences detected - use word count as fallback
        avg_sentence_length = float(total_words)

    # Flesch Reading Ease - external library with specific exception handling
    # textstat raises ValueError for empty/invalid input, ZeroDivisionError for
    # text with no syllables
    try:
        flesch_reading_ease = float(textstat.flesch_reading_ease(content))
    except (ValueError, ZeroDivisionError, TypeError) as e:
        logger.warning("textstat.flesch_reading_ease() failed: %s", e)
        flesch_reading_ease = 0.0

    # Count vague terms
    vague_matches = VAGUE_TERMS_PATTERN.findall(content)
    vague_terms_count = len(vague_matches)

    logger.debug(
        "Calculated linguistic metrics: %.1f avg sentence, %.2f richness, %.1f flesch",
        avg_sentence_length,
        vocabulary_richness,
        flesch_reading_ease,
    )

    return LinguisticMetrics(
        avg_sentence_length=avg_sentence_length,
        vocabulary_richness=vocabulary_richness,
        flesch_reading_ease=flesch_reading_ease,
        vague_terms_count=vague_terms_count,
    )


def calculate_reasoning_signals(content: str) -> ReasoningSignals:
    """Calculate reasoning pattern signals from content.

    Detects citations, conditionals, and uncertainty/confidence phrases
    using regex patterns. These are signals for LLM extraction to refine.

    Args:
        content: Raw LLM output text.

    Returns:
        ReasoningSignals dataclass with all reasoning-related fields.

    """
    # Citation patterns
    cites_prd = bool(PRD_PATTERN.search(content))
    cites_architecture = bool(ARCH_PATTERN.search(content))
    cites_story_sections = bool(STORY_SECTION_PATTERN.search(content))

    # Conditional patterns
    uses_conditionals = bool(CONDITIONAL_PATTERN.search(content))

    # Uncertainty and confidence phrase counts
    uncertainty_phrases_count = len(UNCERTAINTY_PATTERN.findall(content))
    confidence_phrases_count = len(CONFIDENCE_PATTERN.findall(content))

    logger.debug(
        "Calculated reasoning signals: PRD=%s, arch=%s, story=%s",
        cites_prd,
        cites_architecture,
        cites_story_sections,
    )

    return ReasoningSignals(
        cites_prd=cites_prd,
        cites_architecture=cites_architecture,
        cites_story_sections=cites_story_sections,
        uses_conditionals=uses_conditionals,
        uncertainty_phrases_count=uncertainty_phrases_count,
        confidence_phrases_count=confidence_phrases_count,
    )


def collect_deterministic_metrics(
    raw_output: str,
    context: CollectorContext,
) -> DeterministicMetrics:
    """Collect all deterministic metrics from raw LLM output.

    Main entry point for metrics collection. Calculates structure,
    linguistic, and reasoning metrics in a single call.

    Args:
        raw_output: Raw LLM output text to analyze.
        context: CollectorContext with story info and timestamp.

    Returns:
        DeterministicMetrics composing all calculated metrics.

    """
    structure = calculate_structure_metrics(raw_output)
    linguistic = calculate_linguistic_metrics(raw_output)
    reasoning = calculate_reasoning_signals(raw_output)

    return DeterministicMetrics(
        structure=structure,
        linguistic=linguistic,
        reasoning=reasoning,
        collected_at=context.timestamp,
    )
