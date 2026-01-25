"""Experiments API module for dashboard.

This module provides the REST API infrastructure for browsing experiment runs,
including discovery, filtering, sorting, and pagination.

Public API:
    discover_runs: Discover and load all run manifests with caching
    filter_runs: Filter runs by various criteria
    sort_runs: Sort runs by field and direction
    format_duration: Format duration seconds for display
    ExperimentRunSummary: Summary model for API response
    ExperimentsListResponse: Response model with pagination
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from anyio import to_thread
from pydantic import BaseModel, ConfigDict

from bmad_assist.core.exceptions import ConfigError
from bmad_assist.experiments import (
    ExperimentStatus,
    FixtureEntry,
    FixtureManager,
    ManifestManager,
    RunManifest,
)

logger = logging.getLogger(__name__)

# Thread-safe TTL cache with lock
_runs_cache: dict[str, tuple[datetime, list[RunManifest]]] = {}
_cache_lock = asyncio.Lock()
CACHE_TTL_SECONDS = 30.0

# Thread-safe TTL cache for fixtures (60 seconds - changes rarely)
_fixtures_cache: dict[str, tuple[datetime, list[FixtureEntry]]] = {}
_fixtures_cache_lock = asyncio.Lock()
FIXTURES_CACHE_TTL_SECONDS = 60.0

# Pattern for valid run_id: alphanumeric, underscores, hyphens only
RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

__all__ = [
    "CACHE_TTL_SECONDS",
    "CONFIGS_CACHE_TTL_SECONDS",
    "ConfigDetails",
    "ConfigSummary",
    "ExperimentCancelResponse",
    "ExperimentProgressEvent",
    "ExperimentRunDetails",
    "ExperimentRunRequest",
    "ExperimentRunResponse",
    "ExperimentRunSummary",
    "ExperimentStatusEvent",
    "ExperimentsListResponse",
    "FIXTURES_CACHE_TTL_SECONDS",
    "FixtureDetails",
    "FixtureRunInfo",
    "FixtureSummary",
    "FixturesListResponse",
    "LOOPS_CACHE_TTL_SECONDS",
    "LoopDetails",
    "LoopSummary",
    "MAX_YAML_CONTENT_SIZE",
    "PATCHSETS_CACHE_TTL_SECONDS",
    "PaginationInfo",
    "PatchSetDetails",
    "PatchSetSummary",
    "PhaseDetails",
    "ResolvedDetails",
    "TemplateRunInfo",
    "TemplateStats",
    "clear_configs_cache",
    "clear_fixtures_cache",
    "clear_loops_cache",
    "clear_patchsets_cache",
    "discover_configs",
    "discover_fixtures",
    "discover_loops",
    "discover_patchsets",
    "discover_runs",
    "filter_runs",
    "format_duration",
    "get_config_run_stats",
    "get_fixture_run_stats",
    "get_loop_run_stats",
    "get_patchset_run_stats",
    "get_run_by_id",
    "get_yaml_content",
    "manifest_to_details",
    "sort_runs",
    "validate_run_id",
]


# =============================================================================
# Response Models
# =============================================================================


class ExperimentRunSummary(BaseModel):
    """Summary of an experiment run for list view."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str  # ExperimentStatus.value
    started: datetime
    completed: datetime | None
    duration_seconds: float | None
    input: dict[str, str]  # fixture, config, patch_set, loop
    results: dict[str, int] | None  # stories_attempted, completed, failed
    metrics: dict[str, Any] | None  # total_cost, total_tokens


class PaginationInfo(BaseModel):
    """Pagination metadata."""

    model_config = ConfigDict(frozen=True)

    total: int
    offset: int
    limit: int
    has_more: bool


class ExperimentsListResponse(BaseModel):
    """Response for experiments list endpoint."""

    model_config = ConfigDict(frozen=True)

    runs: list[ExperimentRunSummary]
    pagination: PaginationInfo


# =============================================================================
# Experiment Run Trigger Models (Story 19.6)
# =============================================================================


class ExperimentRunRequest(BaseModel):
    """Request body for POST /api/experiments/run."""

    model_config = ConfigDict(frozen=True)

    fixture: str
    config: str
    patch_set: str
    loop: str


class ExperimentRunResponse(BaseModel):
    """Response for POST /api/experiments/run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    message: str


class ExperimentCancelResponse(BaseModel):
    """Response for POST /api/experiments/run/{run_id}/cancel."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    message: str


class ExperimentProgressEvent(BaseModel):
    """SSE progress event data."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    percent: int
    stories_completed: int
    stories_total: int


class ExperimentStatusEvent(BaseModel):
    """SSE status event data."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    phase: str | None = None
    story: str | None = None
    position: int | None = None  # Current step/phase number


# =============================================================================
# Run Details Models (Story 19.2)
# =============================================================================


class PhaseDetails(BaseModel):
    """Details of a single phase execution."""

    model_config = ConfigDict(frozen=True)

    phase: str
    story: str | None
    status: str  # "completed", "failed", "skipped"
    duration_seconds: float
    duration_formatted: str
    tokens: int | None
    cost: float | None
    error: str | None


class ResolvedDetails(BaseModel):
    """Resolved configuration details."""

    model_config = ConfigDict(frozen=True)

    fixture: dict[str, Any]
    config: dict[str, Any]
    patch_set: dict[str, Any]
    loop: dict[str, Any]


class ExperimentRunDetails(BaseModel):
    """Full details of an experiment run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    started: datetime
    completed: datetime | None
    duration_seconds: float | None
    duration_formatted: str
    input: dict[str, str]
    resolved: ResolvedDetails
    results: dict[str, int] | None
    metrics: dict[str, Any] | None
    phases: list[PhaseDetails]


# =============================================================================
# Fixture Response Models (Story 19.3)
# =============================================================================


class FixtureRunInfo(BaseModel):
    """Brief info about a run using this fixture."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    started: datetime
    config: str  # Config template used


class FixtureSummary(BaseModel):
    """Summary of a fixture for list view."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None
    path: str
    tags: list[str]
    difficulty: str  # "easy", "medium", "hard"
    estimated_cost: str  # "$X.XX" format
    estimated_cost_value: float  # Parsed float for sorting
    run_count: int
    last_run: datetime | None


class FixtureDetails(BaseModel):
    """Full details of a fixture."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None
    path: str
    resolved_path: str  # Absolute path
    tags: list[str]
    difficulty: str
    estimated_cost: str
    estimated_cost_value: float
    run_count: int
    last_run: datetime | None
    recent_runs: list[FixtureRunInfo]


class FixturesListResponse(BaseModel):
    """Response for fixtures list endpoint."""

    model_config = ConfigDict(frozen=True)

    fixtures: list[FixtureSummary]
    total: int


# =============================================================================
# Run Discovery
# =============================================================================


def _scan_runs_sync(experiments_dir: Path) -> list[RunManifest]:
    """Scan manifests synchronously in thread pool.

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of loaded RunManifest objects.

    """
    runs_dir = experiments_dir / "runs"
    if not runs_dir.exists():
        return []

    manifests: list[RunManifest] = []

    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "manifest.yaml"
        if not manifest_path.exists():
            continue

        try:
            manager = ManifestManager(run_dir)
            manifest = manager.load()
            manifests.append(manifest)
        except (ConfigError, Exception) as e:
            logger.warning("Failed to load manifest %s: %s", manifest_path, e)
            continue

    return manifests


async def discover_runs(experiments_dir: Path) -> list[RunManifest]:
    """Discover and load all run manifests (thread-safe with caching).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of loaded RunManifest objects, sorted by started desc.

    """
    cache_key = str(experiments_dir / "runs")
    now = datetime.now(UTC)

    # Check cache with lock
    async with _cache_lock:
        if cache_key in _runs_cache:
            cached_time, cached_runs = _runs_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_runs

    # Load outside lock to avoid blocking other requests
    manifests = await to_thread.run_sync(lambda: _scan_runs_sync(experiments_dir))

    # Update cache with lock
    async with _cache_lock:
        _runs_cache[cache_key] = (datetime.now(UTC), manifests)

    return manifests


# =============================================================================
# Filtering
# =============================================================================


def filter_runs(
    runs: list[RunManifest],
    *,
    status: str | None = None,
    fixture: str | None = None,
    config: str | None = None,
    patch_set: str | None = None,
    loop: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[RunManifest]:
    """Filter runs by criteria.

    All filters combine with AND logic.
    String filters are case-insensitive exact matches.
    Date inputs are interpreted as UTC (naive dates assumed to be UTC midnight).

    Args:
        runs: List of runs to filter.
        status: Filter by ExperimentStatus value.
        fixture: Filter by fixture ID (case-insensitive).
        config: Filter by config template name (case-insensitive).
        patch_set: Filter by patch-set name (case-insensitive).
        loop: Filter by loop template name (case-insensitive).
        start_date: ISO date for runs started on or after.
        end_date: ISO date for runs started on or before (includes entire day).

    Returns:
        Filtered list of runs.

    Raises:
        ValueError: For invalid status enum or date format, or if start_date > end_date.

    """
    result = runs

    if status:
        try:
            status_enum = ExperimentStatus(status)
            result = [r for r in result if r.status == status_enum]
        except ValueError as e:
            raise ValueError(f"Invalid status: {status}") from e

    if fixture:
        fixture_lower = fixture.lower()
        result = [r for r in result if r.input.fixture.lower() == fixture_lower]

    if config:
        config_lower = config.lower()
        result = [r for r in result if r.input.config.lower() == config_lower]

    if patch_set:
        patch_set_lower = patch_set.lower()
        result = [r for r in result if r.input.patch_set.lower() == patch_set_lower]

    if loop:
        loop_lower = loop.lower()
        result = [r for r in result if r.input.loop.lower() == loop_lower]

    # Parse dates and validate range
    start_dt = None
    end_dt = None

    if start_date:
        try:
            # Interpret as UTC midnight
            start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        except ValueError as e:
            raise ValueError(f"Invalid start_date format: {start_date}") from e

    if end_date:
        try:
            # Include entire end date (through 23:59:59 UTC)
            end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC) + timedelta(days=1)
        except ValueError as e:
            raise ValueError(f"Invalid end_date format: {end_date}") from e

    # Validate date range
    if start_dt and end_dt and start_dt >= end_dt:
        raise ValueError("start_date must be before or equal to end_date")

    if start_dt:
        result = [r for r in result if _normalize_datetime(r.started) >= start_dt]

    if end_dt:
        result = [r for r in result if _normalize_datetime(r.started) < end_dt]

    return result


def _normalize_datetime(dt: datetime) -> datetime:
    """Normalize datetime to UTC for comparison.

    Args:
        dt: Datetime to normalize.

    Returns:
        Datetime with UTC timezone.

    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


# =============================================================================
# Sorting
# =============================================================================


def sort_runs(
    runs: list[RunManifest],
    sort_by: str = "started",
    sort_order: str = "desc",
) -> list[RunManifest]:
    """Sort runs by field and direction.

    Args:
        runs: List of runs to sort.
        sort_by: Field to sort by (started, completed, duration, status).
        sort_order: Sort direction (asc, desc).

    Returns:
        Sorted list of runs.

    """
    reverse = sort_order == "desc"

    if sort_by == "started":
        return sorted(runs, key=lambda r: r.started, reverse=reverse)
    elif sort_by == "completed":
        # Nulls last for asc, nulls first for desc (AC4)
        # sorted() is stable; first tuple element determines null position
        if reverse:
            # Desc: We want None first after reverse
            # In natural order, True > False, so after reverse False > True
            # None -> True (comes first after reverse), value -> False
            return sorted(
                runs,
                key=lambda r: (
                    r.completed is None,  # None -> True, value -> False
                    r.completed or datetime.max.replace(tzinfo=UTC),
                ),
                reverse=True,
            )
        else:
            # Asc: We want None last
            # None -> True (sorts after False), value -> False
            return sorted(
                runs,
                key=lambda r: (
                    r.completed is None,  # None -> True (1), value -> False (0)
                    r.completed or datetime.min.replace(tzinfo=UTC),
                ),
            )
    elif sort_by == "duration":
        # Calculate duration from started and completed
        def get_duration(r: RunManifest) -> float | None:
            if r.completed is None:
                return None
            return (r.completed - r.started).total_seconds()

        # Nulls last for asc, nulls first for desc (AC4)
        if reverse:
            # Desc: We want None first after reverse
            # None -> True (comes first after reverse), value -> False
            return sorted(
                runs,
                key=lambda r: (
                    get_duration(r) is None,  # None -> True, value -> False
                    get_duration(r) or float("inf"),
                ),
                reverse=True,
            )
        else:
            # Asc: We want None last
            # None -> True (sorts after False), value -> False
            return sorted(
                runs,
                key=lambda r: (
                    get_duration(r) is None,  # None -> True (1), value -> False (0)
                    get_duration(r) or 0.0,
                ),
            )
    elif sort_by == "status":
        return sorted(runs, key=lambda r: r.status.value, reverse=reverse)
    else:
        # Default to started
        return sorted(runs, key=lambda r: r.started, reverse=reverse)


# =============================================================================
# Helpers
# =============================================================================


def format_duration(seconds: float | None) -> str:
    """Format duration for display.

    Returns human-friendly format: Xh Ym Zs, Ym Zs, or Zs.
    Returns "-" for None (running experiments).

    Args:
        seconds: Duration in seconds or None.

    Returns:
        Formatted duration string.

    """
    if seconds is None:
        return "-"

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def manifest_to_summary(manifest: RunManifest) -> ExperimentRunSummary:
    """Convert RunManifest to ExperimentRunSummary for API response.

    Args:
        manifest: Run manifest to convert.

    Returns:
        Summary suitable for API response.

    """
    # Calculate duration if completed
    duration_seconds = None
    if manifest.completed:
        duration_seconds = (manifest.completed - manifest.started).total_seconds()

    # Build input dict
    input_dict = {
        "fixture": manifest.input.fixture,
        "config": manifest.input.config,
        "patch_set": manifest.input.patch_set,
        "loop": manifest.input.loop,
    }

    # Build results dict if available
    results_dict = None
    if manifest.results:
        results_dict = {
            "stories_attempted": manifest.results.stories_attempted,
            "stories_completed": manifest.results.stories_completed,
            "stories_failed": manifest.results.stories_failed,
        }

    # Build metrics dict if available
    metrics_dict: dict[str, Any] | None = None
    if manifest.metrics:
        metrics_dict = {}
        if manifest.metrics.total_cost is not None:
            metrics_dict["total_cost"] = manifest.metrics.total_cost
        if manifest.metrics.total_tokens is not None:
            metrics_dict["total_tokens"] = manifest.metrics.total_tokens
        if not metrics_dict:
            metrics_dict = None

    return ExperimentRunSummary(
        run_id=manifest.run_id,
        status=manifest.status.value,
        started=manifest.started,
        completed=manifest.completed,
        duration_seconds=duration_seconds,
        input=input_dict,
        results=results_dict,
        metrics=metrics_dict,
    )


def clear_cache() -> None:
    """Clear the runs cache (for testing)."""
    global _runs_cache
    _runs_cache = {}


# =============================================================================
# Run Details Functions (Story 19.2)
# =============================================================================


def validate_run_id(run_id: str) -> bool:
    """Validate run_id format to prevent path traversal.

    Args:
        run_id: The run identifier to validate.

    Returns:
        True if valid, False otherwise.

    """
    return bool(RUN_ID_PATTERN.match(run_id))


def _load_manifest_sync(run_dir: Path) -> RunManifest | None:
    """Load manifest synchronously (for thread pool execution).

    Args:
        run_dir: Path to run directory.

    Returns:
        Loaded manifest or None if not found/invalid.

    Raises:
        Exception: For unexpected errors (not YAML/schema errors).

    """
    manifest_path = run_dir / "manifest.yaml"
    if not manifest_path.exists():
        return None

    try:
        manager = ManifestManager(run_dir)
        return manager.load()
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in manifest %s: %s", manifest_path, e)
        return None
    except (ConfigError, ValueError) as e:
        logger.warning("Invalid manifest schema %s: %s", manifest_path, e)
        return None
    except Exception:
        # Re-raise unexpected errors for 500 handling
        logger.exception("Unexpected error loading manifest %s", manifest_path)
        raise


async def get_run_by_id(
    run_id: str,
    experiments_dir: Path,
) -> RunManifest | None:
    """Get a specific run by ID.

    Args:
        run_id: The run identifier (must be pre-validated).
        experiments_dir: Path to experiments directory.

    Returns:
        Loaded manifest or None if not found.

    Note:
        Caller must validate run_id format before calling this function.

    """
    run_dir = experiments_dir / "runs" / run_id
    if not run_dir.exists():
        return None

    return await to_thread.run_sync(lambda: _load_manifest_sync(run_dir))


def manifest_to_details(manifest: RunManifest) -> ExperimentRunDetails:
    """Convert RunManifest to ExperimentRunDetails for API response.

    Args:
        manifest: Full run manifest.

    Returns:
        Detailed response suitable for API.

    """
    # Calculate duration
    duration_seconds = None
    if manifest.completed:
        duration_seconds = (manifest.completed - manifest.started).total_seconds()

    # Build input dict
    input_dict = {
        "fixture": manifest.input.fixture,
        "config": manifest.input.config,
        "patch_set": manifest.input.patch_set,
        "loop": manifest.input.loop,
    }

    # Build resolved dict (deep conversion with None handling)
    def _resolve_fixture(f: Any) -> dict[str, Any]:
        if f is None:
            return {"name": None, "source": None, "snapshot": None}
        return {"name": f.name, "source": f.source, "snapshot": f.snapshot}

    def _resolve_config(c: Any) -> dict[str, Any]:
        if c is None:
            return {"name": None, "source": None, "providers": {}}
        return {"name": c.name, "source": c.source, "providers": c.providers}

    def _resolve_patch_set(p: Any) -> dict[str, Any]:
        if p is None:
            return {
                "name": None,
                "source": None,
                "workflow_overrides": {},
                "patches": {},
            }
        return {
            "name": p.name,
            "source": p.source,
            "workflow_overrides": dict(p.workflow_overrides) if p.workflow_overrides else {},
            "patches": dict(p.patches) if p.patches else {},
        }

    def _resolve_loop(loop_obj: Any) -> dict[str, Any]:
        if loop_obj is None:
            return {"name": None, "source": None, "sequence": []}
        return {"name": loop_obj.name, "source": loop_obj.source, "sequence": list(loop_obj.sequence)} # noqa: E501

    resolved = ResolvedDetails(
        fixture=_resolve_fixture(manifest.resolved.fixture if manifest.resolved else None),
        config=_resolve_config(manifest.resolved.config if manifest.resolved else None),
        patch_set=_resolve_patch_set(manifest.resolved.patch_set if manifest.resolved else None),
        loop=_resolve_loop(manifest.resolved.loop if manifest.resolved else None),
    )

    # Build results dict if available
    results_dict = None
    if manifest.results:
        results_dict = {
            "stories_attempted": manifest.results.stories_attempted,
            "stories_completed": manifest.results.stories_completed,
            "stories_failed": manifest.results.stories_failed,
        }

    # Build metrics dict if available
    metrics_dict: dict[str, Any] | None = None
    if manifest.metrics:
        metrics_dict = manifest.metrics.model_dump(exclude_none=True)
        if not metrics_dict:
            metrics_dict = None

    # Build phases list
    phases: list[PhaseDetails] = []
    if manifest.results and manifest.results.phases:
        for phase in manifest.results.phases:
            phases.append(
                PhaseDetails(
                    phase=phase.phase,
                    story=phase.story,
                    status=phase.status,
                    duration_seconds=phase.duration_seconds,
                    duration_formatted=format_duration(phase.duration_seconds),
                    tokens=phase.tokens,
                    cost=phase.cost,
                    error=phase.error,
                )
            )

    return ExperimentRunDetails(
        run_id=manifest.run_id,
        status=manifest.status.value,
        started=manifest.started,
        completed=manifest.completed,
        duration_seconds=duration_seconds,
        duration_formatted=format_duration(duration_seconds),
        input=input_dict,
        resolved=resolved,
        results=results_dict,
        metrics=metrics_dict,
        phases=phases,
    )


# =============================================================================
# Fixture Discovery (Story 19.3)
# =============================================================================


class FixtureStats:
    """Statistics for a fixture."""

    __slots__ = ("run_count", "last_run", "recent_runs")

    def __init__(self) -> None:
        self.run_count: int = 0
        self.last_run: datetime | None = None
        self.recent_runs: list[FixtureRunInfo] = []


def _load_fixtures_sync(experiments_dir: Path) -> list[FixtureEntry]:
    """Discover fixtures synchronously (for thread pool).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of discovered fixtures (may be empty).

    """
    fixtures_dir = experiments_dir / "fixtures"
    if not fixtures_dir.exists():
        logger.debug("Fixtures directory not found: %s", fixtures_dir)
        return []

    try:
        manager = FixtureManager(fixtures_dir)
        return manager.discover()
    except ConfigError as e:
        logger.warning("Failed to discover fixtures: %s", e)
        return []


async def discover_fixtures(experiments_dir: Path) -> list[FixtureEntry]:
    """Discover fixtures (thread-safe with caching).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of FixtureEntry objects (may be empty).

    """
    cache_key = str(experiments_dir / "fixtures")
    now = datetime.now(UTC)

    # Check cache with lock
    async with _fixtures_cache_lock:
        if cache_key in _fixtures_cache:
            cached_time, cached_fixtures = _fixtures_cache[cache_key]
            if (now - cached_time).total_seconds() < FIXTURES_CACHE_TTL_SECONDS:
                return cached_fixtures

    # Load outside lock
    fixtures = await to_thread.run_sync(lambda: _load_fixtures_sync(experiments_dir))

    # Update cache with lock
    async with _fixtures_cache_lock:
        _fixtures_cache[cache_key] = (datetime.now(UTC), fixtures)

    return fixtures


def clear_fixtures_cache() -> None:
    """Clear the fixtures cache (for testing)."""
    global _fixtures_cache
    _fixtures_cache = {}


def get_fixture_run_stats(
    fixtures: list[FixtureEntry],
    runs: list[RunManifest],
) -> dict[str, FixtureStats]:
    """Calculate run statistics for each fixture.

    Uses O(N+M) algorithm with lookup map instead of O(N*M) nested loop.

    Args:
        fixtures: List of fixture entries.
        runs: List of run manifests (already sorted by started desc).

    Returns:
        Dict mapping fixture_id to FixtureStats.

    """
    stats: dict[str, FixtureStats] = {}

    # Initialize stats for all fixtures and build case-insensitive lookup map
    # Maps lowercase fixture_id -> original fixture_id (for case-insensitive matching)
    id_lookup: dict[str, str] = {}
    for fixture in fixtures:
        stats[fixture.id] = FixtureStats()
        id_lookup[fixture.id.lower()] = fixture.id

    # Process runs in O(N) - single pass with O(1) lookup
    # Runs are already sorted by started desc from discover_runs
    for run in runs:
        fixture_id_lower = run.input.fixture.lower()

        # O(1) lookup instead of O(M) inner loop
        original_id = id_lookup.get(fixture_id_lower)
        if original_id is None:
            continue

        fixture_stats = stats[original_id]
        fixture_stats.run_count += 1

        # Track last run (first match since runs sorted desc)
        if fixture_stats.last_run is None:
            fixture_stats.last_run = run.started

        # Add to recent runs (up to 5)
        if len(fixture_stats.recent_runs) < 5:
            fixture_stats.recent_runs.append(
                FixtureRunInfo(
                    run_id=run.run_id,
                    status=run.status.value,
                    started=run.started,
                    config=run.input.config,
                )
            )

    return stats


# =============================================================================
# Template Response Models (Story 19.5)
# =============================================================================


class TemplateRunInfo(BaseModel):
    """Brief info about a run using a template."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    started: datetime
    fixture: str | None = None  # For configs/loops
    config: str | None = None  # For patch-sets


class ConfigSummary(BaseModel):
    """Summary of a config template for list view."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    source: str  # Path to YAML file
    providers: dict[str, Any]  # {master: {...}, multi: [...]}
    run_count: int
    last_run: datetime | None


class ConfigDetails(BaseModel):
    """Full details of a config template."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    source: str
    providers: dict[str, Any]
    yaml_content: str | None  # Raw YAML content
    run_count: int
    last_run: datetime | None
    recent_runs: list[TemplateRunInfo]


class LoopSummary(BaseModel):
    """Summary of a loop template for list view."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    source: str
    sequence: list[dict[str, Any]]  # [{workflow: str, required: bool}, ...]
    step_count: int
    run_count: int
    last_run: datetime | None


class LoopDetails(BaseModel):
    """Full details of a loop template."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    source: str
    sequence: list[dict[str, Any]]
    step_count: int
    yaml_content: str | None
    run_count: int
    last_run: datetime | None
    recent_runs: list[TemplateRunInfo]


class PatchSetSummary(BaseModel):
    """Summary of a patch-set manifest for list view."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    source: str
    patches: dict[str, str | None]  # workflow -> patch path
    workflow_overrides: dict[str, str]  # workflow -> override path
    patch_count: int  # Number of non-null patches
    override_count: int  # Number of overrides
    run_count: int
    last_run: datetime | None


class PatchSetDetails(BaseModel):
    """Full details of a patch-set manifest."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    source: str
    patches: dict[str, str | None]
    workflow_overrides: dict[str, str]
    patch_count: int
    override_count: int
    yaml_content: str | None
    run_count: int
    last_run: datetime | None
    recent_runs: list[TemplateRunInfo]


# =============================================================================
# Template Stats Class (Story 19.5)
# =============================================================================


class TemplateStats:
    """Statistics for a template."""

    __slots__ = ("run_count", "last_run", "recent_runs")

    def __init__(self) -> None:
        """Initialize empty template statistics."""
        self.run_count: int = 0
        self.last_run: datetime | None = None
        self.recent_runs: list[TemplateRunInfo] = []


# =============================================================================
# Template Discovery with Caching (Story 19.5)
# =============================================================================


# Thread-safe TTL cache for configs (60 seconds)
_configs_cache: dict[str, tuple[datetime, list[tuple[str, Path]]]] = {}
_configs_cache_lock = asyncio.Lock()
CONFIGS_CACHE_TTL_SECONDS = 60.0

# Thread-safe TTL cache for loops (60 seconds)
_loops_cache: dict[str, tuple[datetime, list[tuple[str, Path]]]] = {}
_loops_cache_lock = asyncio.Lock()
LOOPS_CACHE_TTL_SECONDS = 60.0

# Thread-safe TTL cache for patchsets (60 seconds)
_patchsets_cache: dict[str, tuple[datetime, list[tuple[str, Path]]]] = {}
_patchsets_cache_lock = asyncio.Lock()
PATCHSETS_CACHE_TTL_SECONDS = 60.0


def _discover_configs_sync(experiments_dir: Path) -> list[tuple[str, Path]]:
    """Discover config templates synchronously (for thread pool).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of (name, path) tuples for discovered configs.

    """
    from bmad_assist.experiments import ConfigRegistry

    configs_dir = experiments_dir / "configs"
    if not configs_dir.exists():
        logger.debug("Configs directory not found: %s", configs_dir)
        return []

    try:
        registry = ConfigRegistry(configs_dir)
        templates = registry.discover(configs_dir)
        return list(templates.items())
    except Exception as e:
        logger.warning("Failed to discover configs: %s", e)
        return []


async def discover_configs(experiments_dir: Path) -> list[tuple[str, Path]]:
    """Discover config templates (thread-safe with caching).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of (name, path) tuples for discovered configs.

    """
    cache_key = str(experiments_dir / "configs")
    now = datetime.now(UTC)

    # Check cache with lock
    async with _configs_cache_lock:
        if cache_key in _configs_cache:
            cached_time, cached_configs = _configs_cache[cache_key]
            if (now - cached_time).total_seconds() < CONFIGS_CACHE_TTL_SECONDS:
                return cached_configs

    # Load outside lock
    configs = await to_thread.run_sync(lambda: _discover_configs_sync(experiments_dir))

    # Update cache with lock
    async with _configs_cache_lock:
        _configs_cache[cache_key] = (datetime.now(UTC), configs)

    return configs


def clear_configs_cache() -> None:
    """Clear the configs cache (for testing)."""
    global _configs_cache
    _configs_cache = {}


def _discover_loops_sync(experiments_dir: Path) -> list[tuple[str, Path]]:
    """Discover loop templates synchronously (for thread pool).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of (name, path) tuples for discovered loops.

    """
    from bmad_assist.experiments import LoopRegistry

    loops_dir = experiments_dir / "loops"
    if not loops_dir.exists():
        logger.debug("Loops directory not found: %s", loops_dir)
        return []

    try:
        registry = LoopRegistry(loops_dir)
        templates = registry.discover(loops_dir)
        return list(templates.items())
    except Exception as e:
        logger.warning("Failed to discover loops: %s", e)
        return []


async def discover_loops(experiments_dir: Path) -> list[tuple[str, Path]]:
    """Discover loop templates (thread-safe with caching).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of (name, path) tuples for discovered loops.

    """
    cache_key = str(experiments_dir / "loops")
    now = datetime.now(UTC)

    # Check cache with lock
    async with _loops_cache_lock:
        if cache_key in _loops_cache:
            cached_time, cached_loops = _loops_cache[cache_key]
            if (now - cached_time).total_seconds() < LOOPS_CACHE_TTL_SECONDS:
                return cached_loops

    # Load outside lock
    loops = await to_thread.run_sync(lambda: _discover_loops_sync(experiments_dir))

    # Update cache with lock
    async with _loops_cache_lock:
        _loops_cache[cache_key] = (datetime.now(UTC), loops)

    return loops


def clear_loops_cache() -> None:
    """Clear the loops cache (for testing)."""
    global _loops_cache
    _loops_cache = {}


def _discover_patchsets_sync(experiments_dir: Path) -> list[tuple[str, Path]]:
    """Discover patch-set manifests synchronously (for thread pool).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of (name, path) tuples for discovered patch-sets.

    """
    from bmad_assist.experiments import PatchSetRegistry

    patchsets_dir = experiments_dir / "patch-sets"
    if not patchsets_dir.exists():
        logger.debug("Patch-sets directory not found: %s", patchsets_dir)
        return []

    try:
        registry = PatchSetRegistry(patchsets_dir)
        manifests = registry.discover(patchsets_dir)
        return list(manifests.items())
    except Exception as e:
        logger.warning("Failed to discover patch-sets: %s", e)
        return []


async def discover_patchsets(experiments_dir: Path) -> list[tuple[str, Path]]:
    """Discover patch-set manifests (thread-safe with caching).

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of (name, path) tuples for discovered patch-sets.

    """
    cache_key = str(experiments_dir / "patch-sets")
    now = datetime.now(UTC)

    # Check cache with lock
    async with _patchsets_cache_lock:
        if cache_key in _patchsets_cache:
            cached_time, cached_patchsets = _patchsets_cache[cache_key]
            if (now - cached_time).total_seconds() < PATCHSETS_CACHE_TTL_SECONDS:
                return cached_patchsets

    # Load outside lock
    patchsets = await to_thread.run_sync(lambda: _discover_patchsets_sync(experiments_dir))

    # Update cache with lock
    async with _patchsets_cache_lock:
        _patchsets_cache[cache_key] = (datetime.now(UTC), patchsets)

    return patchsets


def clear_patchsets_cache() -> None:
    """Clear the patchsets cache (for testing)."""
    global _patchsets_cache
    _patchsets_cache = {}


# =============================================================================
# Template Run Statistics (Story 19.5)
# =============================================================================


def _get_template_run_stats(
    template_names: list[str],
    runs: list[RunManifest],
    input_field: str,
    secondary_field: str = "fixture",
) -> dict[str, TemplateStats]:
    """Calculate run statistics for any template type.

    Use O(N+M) algorithm with lookup map.

    Args:
        template_names: List of template names.
        runs: List of run manifests (already sorted by started desc).
        input_field: Field name on run.input to match against (config, loop, patch_set).
        secondary_field: Field to include in recent_runs (fixture or config).

    Returns:
        Dict mapping template name to TemplateStats.

    """
    stats: dict[str, TemplateStats] = {}

    # Initialize stats and build case-insensitive lookup map
    name_lookup: dict[str, str] = {}
    for name in template_names:
        stats[name] = TemplateStats()
        name_lookup[name.lower()] = name

    # Process runs in O(N) - single pass with O(1) lookup
    for run in runs:
        run_value = getattr(run.input, input_field, "").lower()

        original_name = name_lookup.get(run_value)
        if original_name is None:
            continue

        template_stats = stats[original_name]
        template_stats.run_count += 1

        # Track last run (first match since runs sorted desc)
        if template_stats.last_run is None:
            template_stats.last_run = run.started

        # Add to recent runs (up to 5)
        if len(template_stats.recent_runs) < 5:
            # Build TemplateRunInfo with appropriate secondary field
            secondary_value = getattr(run.input, secondary_field, None)
            if secondary_field == "fixture":
                template_stats.recent_runs.append(
                    TemplateRunInfo(
                        run_id=run.run_id,
                        status=run.status.value,
                        started=run.started,
                        fixture=secondary_value,
                    )
                )
            else:
                template_stats.recent_runs.append(
                    TemplateRunInfo(
                        run_id=run.run_id,
                        status=run.status.value,
                        started=run.started,
                        config=secondary_value,
                    )
                )

    return stats


def get_config_run_stats(
    config_names: list[str],
    runs: list[RunManifest],
) -> dict[str, TemplateStats]:
    """Calculate run statistics for each config template.

    Args:
        config_names: List of config template names.
        runs: List of run manifests (already sorted by started desc).

    Returns:
        Dict mapping config name to TemplateStats.

    """
    return _get_template_run_stats(config_names, runs, "config", "fixture")


def get_loop_run_stats(
    loop_names: list[str],
    runs: list[RunManifest],
) -> dict[str, TemplateStats]:
    """Calculate run statistics for each loop template.

    Args:
        loop_names: List of loop template names.
        runs: List of run manifests (already sorted by started desc).

    Returns:
        Dict mapping loop name to TemplateStats.

    """
    return _get_template_run_stats(loop_names, runs, "loop", "fixture")


def get_patchset_run_stats(
    patchset_names: list[str],
    runs: list[RunManifest],
) -> dict[str, TemplateStats]:
    """Calculate run statistics for each patch-set manifest.

    Args:
        patchset_names: List of patch-set manifest names.
        runs: List of run manifests (already sorted by started desc).

    Returns:
        Dict mapping patch-set name to TemplateStats.

    """
    return _get_template_run_stats(patchset_names, runs, "patch_set", "config")


# =============================================================================
# YAML Content Loading (Story 19.5)
# =============================================================================


MAX_YAML_CONTENT_SIZE = 100 * 1024  # 100KB


def _read_yaml_content_sync(source_path: str) -> str | None:
    """Read YAML content synchronously (for thread pool).

    Args:
        source_path: Path to YAML file.

    Returns:
        File content or None if not found/too large.

    """
    path = Path(source_path)
    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8")
        if len(content) > MAX_YAML_CONTENT_SIZE:
            truncated = content[: 95 * 1024]  # First 95KB
            return f"# Content truncated (exceeds {MAX_YAML_CONTENT_SIZE // 1024}KB limit)\n{truncated}" # noqa: E501
        return content
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read YAML content from %s: %s", source_path, e)
        return None


async def get_yaml_content(source_path: str) -> str | None:
    """Get YAML content for template detail view.

    Args:
        source_path: Path to YAML file.

    Returns:
        File content or None if not found.

    """
    return await to_thread.run_sync(lambda: _read_yaml_content_sync(source_path))
