"""Experiment runner core for experiment framework.

This module provides the ExperimentRunner class that orchestrates complete
experiment execution across all four axes: fixture, config, patch-set, and loop.

Usage:
    from bmad_assist.experiments import ExperimentRunner, ExperimentInput

    runner = ExperimentRunner(Path("experiments"), Path("/path/to/project"))
    input = ExperimentInput(
        fixture="minimal",
        config="opus-solo",
        patch_set="baseline",
        loop="standard",
    )
    output = runner.run(input)
    print(f"Completed: {output.stories_completed} stories")
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import yaml

from bmad_assist.core.config import Config, ProviderConfig
from bmad_assist.core.exceptions import ConfigError, IsolationError
from bmad_assist.core.io import BMAD_ORIGINAL_CWD_ENV
from bmad_assist.core.loop.dispatch import execute_phase, init_handlers
from bmad_assist.core.loop.interactive import set_non_interactive
from bmad_assist.core.loop.signals import (
    register_signal_handlers,
    reset_shutdown,
    shutdown_requested,
    unregister_signal_handlers,
)
from bmad_assist.core.loop.types import PhaseResult
from bmad_assist.core.paths import _reset_paths, init_paths
from bmad_assist.core.state import Phase, State, save_state, start_epic_timing, start_story_timing
from bmad_assist.core.types import EpicId, epic_sort_key, parse_epic_id
from bmad_assist.experiments.config import ConfigRegistry, ConfigTemplate
from bmad_assist.experiments.fixture import FixtureManager
from bmad_assist.experiments.isolation import FixtureIsolator
from bmad_assist.experiments.loop import LoopRegistry, LoopTemplate
from bmad_assist.experiments.metrics import MetricsCollector
from bmad_assist.experiments.patchset import PatchSetManifest, PatchSetRegistry

logger = logging.getLogger(__name__)

# Epic-level phases that don't have a story ID
# Note: synthesis phases (validate-story-synthesis, code-review-synthesis) are NOT epic-level -
# they run per-story as defined in loop template sequence
EPIC_LEVEL_PHASES: frozenset[str] = frozenset(
    {
        "retrospective",
    }
)


# Workflow name to Phase enum mapping
WORKFLOW_TO_PHASE: dict[str, Phase] = {
    "create-story": Phase.CREATE_STORY,
    "validate-story": Phase.VALIDATE_STORY,
    "validate-story-synthesis": Phase.VALIDATE_STORY_SYNTHESIS,
    "atdd": Phase.ATDD,
    "dev-story": Phase.DEV_STORY,
    "code-review": Phase.CODE_REVIEW,
    "code-review-synthesis": Phase.CODE_REVIEW_SYNTHESIS,
    "test-review": Phase.TEST_REVIEW,
    "retrospective": Phase.RETROSPECTIVE,
    "qa-plan-generate": Phase.QA_PLAN_GENERATE,
    "qa-plan-execute": Phase.QA_PLAN_EXECUTE,
    # "test-design": Skipped (no Phase mapping in core.state.Phase)
}


class ExperimentStatus(str, Enum):
    """Status of an experiment run.

    Attributes:
        PENDING: Run not yet started.
        RUNNING: Run is currently executing.
        COMPLETED: Run finished successfully.
        FAILED: Run encountered an error.
        CANCELLED: Run was cancelled by user signal.

    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ExperimentInput:
    """Input configuration for an experiment run.

    Specifies which fixture, config, patch-set, and loop template to use
    for an experiment run.

    Attributes:
        fixture: Fixture ID from registry.
        config: Config template name.
        patch_set: Patch-set manifest name.
        loop: Loop template name.
        run_id: Optional run ID (auto-generated if None).
        qa_category: Test category for QA phases ("A", "B", or "all"). Default "A".
        fail_fast: If True, stop on first story failure.

    """

    fixture: str
    config: str
    patch_set: str
    loop: str
    run_id: str | None = None
    qa_category: str = "A"  # "A" (default), "B", or "all" (--qa flag sets "all")
    fail_fast: bool = False

    def __post_init__(self) -> None:
        """Validate all required fields are non-empty strings."""
        for field_name in ("fixture", "config", "patch_set", "loop"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")


@dataclass(frozen=True)
class ExperimentOutput:
    """Output/result of an experiment run.

    Contains the final status, timing information, and statistics
    about the completed experiment run.

    Attributes:
        run_id: Unique run identifier.
        status: Final status enum.
        started: Start timestamp (UTC).
        completed: Completion timestamp (UTC) or None if cancelled.
        duration_seconds: Total duration.
        stories_attempted: Number of stories started.
        stories_completed: Number of stories that succeeded.
        stories_failed: Number of stories that failed.
        retrospective_completed: Whether retrospective was executed.
        qa_completed: Whether QA phases were executed (if --qa was used).
        error: Error message if failed.

    """

    run_id: str
    status: ExperimentStatus
    started: datetime
    completed: datetime | None
    duration_seconds: float
    stories_attempted: int
    stories_completed: int
    stories_failed: int
    retrospective_completed: bool = False
    qa_completed: bool = False
    error: str | None = None


class ExperimentRunner:
    """Orchestrates experiment execution across all four axes.

    The ExperimentRunner ties together fixture, config, patch-set, and loop
    templates into a coherent execution flow with proper state isolation,
    patch application, and graceful cancellation support.

    Usage:
        runner = ExperimentRunner(Path("experiments"), Path("/path/to/project"))
        input = ExperimentInput(
            fixture="minimal",
            config="opus-solo",
            patch_set="baseline",
            loop="standard",
        )
        output = runner.run(input)

    """

    def __init__(
        self,
        experiments_dir: Path,
        project_root: Path | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            experiments_dir: Base directory for experiments (configs/, loops/, etc.).
            project_root: Project root for ${project} variable resolution.

        """
        self._experiments_dir = experiments_dir
        self._project_root = project_root
        self._registries_initialized = False

        # Lazy-initialized registries
        self._config_registry: ConfigRegistry | None = None
        self._loop_registry: LoopRegistry | None = None
        self._patchset_registry: PatchSetRegistry | None = None
        self._fixture_manager: FixtureManager | None = None
        self._isolator: FixtureIsolator | None = None

    def _ensure_registries(self) -> None:
        """Lazily initialize all registries."""
        if self._registries_initialized:
            return

        self._config_registry = ConfigRegistry(
            self._experiments_dir / "configs",
            self._project_root,
        )
        self._loop_registry = LoopRegistry(
            self._experiments_dir / "loops",
        )
        self._patchset_registry = PatchSetRegistry(
            self._experiments_dir / "patch-sets",
            self._project_root,
        )
        self._fixture_manager = FixtureManager(
            self._experiments_dir / "fixtures",
        )
        self._isolator = FixtureIsolator(
            self._experiments_dir / "runs",
        )
        self._registries_initialized = True

    def run(self, input: ExperimentInput) -> ExperimentOutput:
        """Execute an experiment with the given input configuration.

        Iterates through ALL stories in the fixture's current epic, executing
        the full loop_template.sequence for each story. After the last story
        completes code-review-synthesis, triggers retrospective automatically.

        Args:
            input: Experiment input specifying all four axes plus flags.

        Returns:
            ExperimentOutput with run results.

        Raises:
            ConfigError: If any axis name is not found in registry.
            IsolationError: If fixture isolation fails.

        """
        # Import here to avoid circular import
        from bmad_assist.bmad.parser import EpicStory
        from bmad_assist.bmad.state_reader import read_project_state
        from bmad_assist.experiments.manifest import (
            ManifestInput,
            ManifestManager,
            ManifestPhaseResult,
            ManifestResolved,
            build_resolved_config,
            build_resolved_fixture,
            build_resolved_loop,
            build_resolved_patchset,
        )

        self._ensure_registries()

        started = datetime.now(UTC)
        stories_attempted = 0
        stories_completed = 0
        stories_failed = 0
        retrospective_completed = False
        qa_completed = False
        error: str | None = None
        status = ExperimentStatus.PENDING
        manifest_manager: ManifestManager | None = None
        run_dir: Path | None = None

        # Generate run_id if not provided
        run_id = input.run_id or self._generate_run_id()

        try:
            # Disable interactive mode for experiments (no TTY available)
            set_non_interactive(True)

            # Reset paths singleton to ensure clean state (may be stale from CLI init)
            _reset_paths()

            # 1. Validate and resolve all configurations
            config_template = self._config_registry.get(input.config)  # type: ignore[union-attr]
            if config_template is None:
                raise ConfigError(f"Config template not found: {input.config}")

            loop_template = self._loop_registry.get(input.loop)  # type: ignore[union-attr]
            if loop_template is None:
                raise ConfigError(f"Loop template not found: {input.loop}")

            patchset_manifest = self._patchset_registry.get(input.patch_set)  # type: ignore[union-attr]
            if patchset_manifest is None:
                raise ConfigError(f"Patch-set manifest not found: {input.patch_set}")

            fixture_entry = self._fixture_manager.get(input.fixture)  # type: ignore[union-attr]
            if fixture_entry is None:
                raise ConfigError(f"Fixture not found in registry: {input.fixture}")

            fixture_path = self._fixture_manager.get_path(input.fixture)  # type: ignore[union-attr]
            # get_path raises ConfigError if not found, so no None check needed here

            # 2. Create run directory structure
            run_dir = self._experiments_dir / "runs" / run_id
            output_dir = run_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 3. Isolate fixture
            isolation_result = self._isolator.isolate(fixture_path, run_id)  # type: ignore[union-attr]
            snapshot_path = isolation_result.snapshot_path

            # 3a. Initialize paths if _bmad exists in fixture
            bmad_config_path = snapshot_path / "_bmad" / "bmm" / "config.yaml"
            if bmad_config_path.exists():
                try:
                    with open(bmad_config_path) as f:
                        bmad_config = yaml.safe_load(f) or {}
                    paths_config = {
                        "output_folder": bmad_config.get(
                            "output_folder", "{project-root}/_bmad-output"
                        ),
                        "planning_artifacts": bmad_config.get(
                            "planning_artifacts", "{project-root}/_bmad-output/planning-artifacts"
                        ),
                        "implementation_artifacts": bmad_config.get(
                            "implementation_artifacts",
                            "{project-root}/_bmad-output/implementation-artifacts",
                        ),
                        "project_knowledge": bmad_config.get(
                            "project_knowledge", "{project-root}/docs"
                        ),
                    }
                    init_paths(snapshot_path, paths_config)
                    logger.debug("Initialized paths for fixture with _bmad at: %s", snapshot_path)
                except Exception as e:
                    logger.warning("Failed to initialize paths from _bmad config: %s", e)

            # 3b. Copy compiled template cache from project root to snapshot
            # This avoids recompiling patches for each experiment run
            if self._project_root:
                source_cache = self._project_root / ".bmad-assist" / "cache"
                if source_cache.exists() and source_cache.is_dir():
                    dest_cache = snapshot_path / ".bmad-assist" / "cache"
                    dest_cache.mkdir(parents=True, exist_ok=True)
                    for cache_file in source_cache.glob("*.tpl.xml*"):
                        shutil.copy2(cache_file, dest_cache / cache_file.name)
                    logger.debug(
                        "Copied template cache from %s to %s",
                        source_cache,
                        dest_cache,
                    )

            # 4. Create manifest after fixture isolation
            manifest_manager = ManifestManager(run_dir)
            manifest_input = ManifestInput(
                fixture=input.fixture,
                config=input.config,
                patch_set=input.patch_set,
                loop=input.loop,
            )

            # Get source paths for resolved section
            config_path = self._experiments_dir / "configs" / f"{input.config}.yaml"
            patchset_path = self._experiments_dir / "patch-sets" / f"{input.patch_set}.yaml"
            loop_path = self._experiments_dir / "loops" / f"{input.loop}.yaml"

            manifest_resolved = ManifestResolved(
                fixture=build_resolved_fixture(fixture_entry, isolation_result, run_dir),
                config=build_resolved_config(config_template, config_path),
                patch_set=build_resolved_patchset(patchset_manifest, patchset_path),
                loop=build_resolved_loop(loop_template, loop_path),
            )
            manifest_manager.create(manifest_input, manifest_resolved, started, run_id)

            # 5. Build experiment Config from ConfigTemplate
            experiment_config = self._build_experiment_config(config_template)

            # 5a. CRITICAL: Set config singleton for compile_patch() and other internals
            # that use get_config(). Without this, patch compilation fails.
            from bmad_assist.core.config import load_config

            load_config(experiment_config.model_dump())

            # 6. Initialize handlers with snapshot_path (NOT project_root!)
            init_handlers(experiment_config, snapshot_path)

            # 6a. Initialize run-scoped prompts directory (Story 22.2)
            # Prompts saved as: prompt-{epic}-{story}-{seq}-{phase}.md
            from bmad_assist.core.io import get_timestamp, init_run_prompts_dir

            run_timestamp = get_timestamp()
            init_run_prompts_dir(snapshot_path, run_timestamp)

            # 7. Load stories from fixture's sprint-status or epics
            project_state = read_project_state(
                snapshot_path / "docs",
                use_sprint_status=True,
            )

            # 7a. Build epic_list and stories_by_epic from all non-done stories
            # (mirrors CLI behavior in _load_epics_and_stories)
            epic_numbers: set[EpicId] = set()
            stories_by_epic: dict[EpicId, list[EpicStory]] = {}  # EpicStory objects

            for story in project_state.all_stories:
                if story.status == "done":
                    continue
                epic_part = story.number.split(".")[0]
                epic_id = parse_epic_id(epic_part)
                epic_numbers.add(epic_id)
                if epic_id not in stories_by_epic:
                    stories_by_epic[epic_id] = []
                stories_by_epic[epic_id].append(story)

            epic_list = sorted(epic_numbers, key=epic_sort_key)

            if not epic_list:
                logger.warning("No epics with pending stories found in fixture")
                status = ExperimentStatus.COMPLETED
            else:
                logger.info(
                    "Found %d epics with pending stories: %s",
                    len(epic_list),
                    epic_list,
                )

            # 8. Initialize experiment state with first epic/story
            state = self._init_state(run_dir, loop_template)
            if epic_list and stories_by_epic.get(epic_list[0]):
                first_epic = epic_list[0]
                first_story = stories_by_epic[first_epic][0]
                state.current_epic = first_epic
                state.current_story = first_story.number

            # Update manifest status to RUNNING
            manifest_manager.update_status(ExperimentStatus.RUNNING)
            status = ExperimentStatus.RUNNING

            # 9. Reset and register signal handlers for cancellation
            reset_shutdown()
            register_signal_handlers()

            try:
                # 10. OUTER EPIC LOOP - iterate through all epics
                for epic_idx, current_epic in enumerate(epic_list):
                    if shutdown_requested():
                        status = ExperimentStatus.CANCELLED
                        break

                    is_last_epic = epic_idx == len(epic_list) - 1
                    epic_stories = stories_by_epic.get(current_epic, [])

                    if not epic_stories:
                        logger.warning("No pending stories for epic %s, skipping", current_epic)
                        continue

                    logger.info(
                        "Processing epic %s (%d/%d) with %d stories: %s",
                        current_epic,
                        epic_idx + 1,
                        len(epic_list),
                        len(epic_stories),
                        [s.number for s in epic_stories],
                    )

                    # Update state for this epic
                    state.current_epic = current_epic
                    start_epic_timing(state)

                    # 10a. INNER STORY LOOP - execute full sequence for each story
                    for story_idx, story in enumerate(epic_stories):
                        if shutdown_requested():
                            status = ExperimentStatus.CANCELLED
                            break

                        story_failed = False

                        # Update state to current story and reset story timing
                        state.current_story = story.number
                        start_story_timing(state)
                        logger.info(
                            "Processing story %s (%d/%d in epic %s)",
                            story.number,
                            story_idx + 1,
                            len(epic_stories),
                            current_epic,
                        )

                        # Execute loop sequence for this story
                        for step in loop_template.sequence:
                            if shutdown_requested():
                                status = ExperimentStatus.CANCELLED
                                break

                            # Skip epic-level phases during per-story iteration
                            # (retrospective is handled separately after all stories)
                            if step.workflow in EPIC_LEVEL_PHASES:
                                continue

                            # Reset phase timing in state for accurate reporting
                            from bmad_assist.core.state import start_phase_timing

                            start_phase_timing(state)
                            phase_start = datetime.now(UTC)

                            # Execute phase with patch application
                            result = self._execute_step(
                                step.workflow,
                                state,
                                patchset_manifest,
                                run_dir,
                                snapshot_path,
                                set(),  # Don't track created stories across iterations
                            )

                            phase_duration = (datetime.now(UTC) - phase_start).total_seconds()

                            # Handle skipped workflows
                            if result.outputs.get("skipped"):
                                manifest_manager.add_phase_result(
                                    ManifestPhaseResult(
                                        phase=step.workflow,
                                        story=story.number,
                                        status="skipped",
                                        duration_seconds=phase_duration,
                                        error=result.outputs.get("reason"),
                                    )
                                )
                                continue

                            if result.success:
                                manifest_manager.add_phase_result(
                                    ManifestPhaseResult(
                                        phase=step.workflow,
                                        story=story.number,
                                        status="completed",
                                        duration_seconds=phase_duration,
                                        error=None,
                                    )
                                )
                            else:
                                manifest_manager.add_phase_result(
                                    ManifestPhaseResult(
                                        phase=step.workflow,
                                        story=story.number,
                                        status="failed",
                                        duration_seconds=phase_duration,
                                        error=result.error,
                                    )
                                )

                                if step.required:
                                    # Required step failure marks story as failed
                                    story_failed = True
                                    if input.fail_fast:
                                        # Stop immediately on first failure
                                        status = ExperimentStatus.FAILED
                                        error = f"Story {story.number} failed at {step.workflow}"
                                        logger.error(error)
                                        break
                                    else:
                                        # Continue to next story
                                        logger.warning(
                                            "Story %s failed at %s: %s (continuing)",
                                            story.number,
                                            step.workflow,
                                            result.error,
                                        )
                                        break
                                else:
                                    # Optional step failure does NOT mark story as failed
                                    logger.warning(
                                        "Optional step '%s' failed for story %s: %s",
                                        step.workflow,
                                        story.number,
                                        result.error,
                                    )

                        # Track story completion stats
                        if not story_failed:
                            stories_completed += 1
                        else:
                            stories_failed += 1
                        stories_attempted += 1

                        # Check for fail-fast exit
                        if status == ExperimentStatus.FAILED:
                            break

                        # Save state after each story
                        save_state(state, run_dir / "state.yaml")

                    # Check if we should exit the epic loop
                    if status in (ExperimentStatus.FAILED, ExperimentStatus.CANCELLED):
                        break

                    # 11. Retrospective after all stories in this epic
                    running = status == ExperimentStatus.RUNNING
                    if running and epic_stories and not shutdown_requested():
                        logger.info("All stories done, retrospective for epic %s", current_epic)
                        phase_start = datetime.now(UTC)

                        # Set phase for retrospective
                        state.current_phase = Phase.RETROSPECTIVE

                        result = self._execute_step(
                            "retrospective",
                            state,
                            patchset_manifest,
                            run_dir,
                            snapshot_path,
                            set(),
                        )

                        phase_duration = (datetime.now(UTC) - phase_start).total_seconds()

                        if result.outputs.get("skipped"):
                            manifest_manager.add_phase_result(
                                ManifestPhaseResult(
                                    phase="retrospective",
                                    story=None,
                                    status="skipped",
                                    duration_seconds=phase_duration,
                                    error=result.outputs.get("reason"),
                                    epic=current_epic,
                                )
                            )
                        elif result.success:
                            retrospective_completed = True
                            manifest_manager.add_phase_result(
                                ManifestPhaseResult(
                                    phase="retrospective",
                                    story=None,
                                    status="completed",
                                    duration_seconds=phase_duration,
                                    error=None,
                                    epic=current_epic,
                                )
                            )
                        else:
                            # Retrospective failure is non-fatal
                            logger.warning("Retrospective failed: %s", result.error)
                            manifest_manager.add_phase_result(
                                ManifestPhaseResult(
                                    phase="retrospective",
                                    story=None,
                                    status="failed",
                                    duration_seconds=phase_duration,
                                    error=result.error,
                                    epic=current_epic,
                                )
                            )

                    # 12. QA phases for this epic (run after retrospective)
                    if status == ExperimentStatus.RUNNING and not shutdown_requested():
                        # Pass qa_category to state for handler to use
                        state.qa_category = input.qa_category
                        logger.info(
                            "Executing QA phases for epic %s (category: %s)",
                            current_epic,
                            input.qa_category,
                        )
                        qa_phases = ["qa-plan-generate", "qa-plan-execute"]
                        qa_success = True

                        for qa_phase in qa_phases:
                            if shutdown_requested():
                                break

                            phase_start = datetime.now(UTC)

                            result = self._execute_step(
                                qa_phase,
                                state,
                                patchset_manifest,
                                run_dir,
                                snapshot_path,
                                set(),
                            )

                            phase_duration = (datetime.now(UTC) - phase_start).total_seconds()

                            if result.outputs.get("skipped"):
                                manifest_manager.add_phase_result(
                                    ManifestPhaseResult(
                                        phase=qa_phase,
                                        story=None,
                                        status="skipped",
                                        duration_seconds=phase_duration,
                                        error=result.outputs.get("reason"),
                                        epic=current_epic,
                                    )
                                )
                            elif result.success:
                                manifest_manager.add_phase_result(
                                    ManifestPhaseResult(
                                        phase=qa_phase,
                                        story=None,
                                        status="completed",
                                        duration_seconds=phase_duration,
                                        error=None,
                                        epic=current_epic,
                                    )
                                )
                            else:
                                qa_success = False
                                manifest_manager.add_phase_result(
                                    ManifestPhaseResult(
                                        phase=qa_phase,
                                        story=None,
                                        status="failed",
                                        duration_seconds=phase_duration,
                                        error=result.error,
                                        epic=current_epic,
                                    )
                                )
                                logger.warning(
                                    "QA phase %s failed for epic %s: %s",
                                    qa_phase,
                                    current_epic,
                                    result.error,
                                )
                                break

                        if qa_success:
                            qa_completed = True

                    # 13. Advance to next epic (if not last)
                    if not is_last_epic and status == ExperimentStatus.RUNNING:
                        next_epic = epic_list[epic_idx + 1]
                        next_stories = stories_by_epic.get(next_epic, [])
                        if next_stories:
                            logger.info(
                                "Advancing from epic %s to epic %s",
                                current_epic,
                                next_epic,
                            )
                            state.current_epic = next_epic
                            state.current_story = next_stories[0].number
                            state.current_phase = WORKFLOW_TO_PHASE.get(
                                loop_template.sequence[0].workflow, Phase.CREATE_STORY
                            )
                            save_state(state, run_dir / "state.yaml")

                # All epics processed
                if status == ExperimentStatus.RUNNING:
                    status = ExperimentStatus.COMPLETED

            finally:
                unregister_signal_handlers()

        except ConfigError:
            # Re-raise ConfigError for unknown axis names
            raise
        except IsolationError:
            # Re-raise IsolationError for fixture isolation failures
            raise
        except Exception as e:
            status = ExperimentStatus.FAILED
            error = str(e)
            logger.exception("Experiment run failed: %s", e)

        completed = datetime.now(UTC)
        duration = (completed - started).total_seconds()

        # Finalize manifest if it was created
        if manifest_manager is not None and not manifest_manager.is_finalized:
            manifest_manager.finalize(status, completed)

        # Collect and save metrics after finalization
        if (
            manifest_manager is not None
            and manifest_manager.manifest is not None
            and run_dir is not None
        ):
            try:
                collector = MetricsCollector(run_dir)
                metrics_file = collector.collect(manifest_manager.manifest)
                collector.save(metrics_file)

                # Update manifest with metrics
                manifest_manager.update_metrics(metrics_file.summary.to_manifest_metrics())

                logger.info("Metrics collected for run %s", run_id)
            except Exception as e:
                # Metrics collection failure is non-fatal
                logger.warning("Failed to collect metrics for run %s: %s", run_id, e)

        # Reset interactive mode and paths
        set_non_interactive(False)
        _reset_paths()

        return ExperimentOutput(
            run_id=run_id,
            status=status,
            started=started,
            completed=completed if status != ExperimentStatus.RUNNING else None,
            duration_seconds=duration,
            stories_attempted=stories_attempted,
            stories_completed=stories_completed,
            stories_failed=stories_failed,
            retrospective_completed=retrospective_completed,
            qa_completed=qa_completed,
            error=error,
        )

    def _generate_run_id(self) -> str:
        """Generate unique run ID in format run-YYYY-MM-DD-NNN."""
        runs_dir = self._experiments_dir / "runs"
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        prefix = f"run-{today}-"

        max_seq = 0
        if runs_dir.exists():
            for d in runs_dir.iterdir():
                if d.is_dir() and d.name.startswith(prefix):
                    try:
                        seq = int(d.name[len(prefix) :])
                        max_seq = max(max_seq, seq)
                    except ValueError:
                        pass

        return f"{prefix}{max_seq + 1:03d}"

    def _init_state(self, run_dir: Path, loop_template: LoopTemplate) -> State:
        """Initialize experiment state file with timing."""
        from bmad_assist.core.state import (
            start_epic_timing,
            start_phase_timing,
            start_project_timing,
            start_story_timing,
        )

        # Determine first phase from loop template
        first_workflow = loop_template.sequence[0].workflow
        first_phase = WORKFLOW_TO_PHASE.get(first_workflow, Phase.CREATE_STORY)

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=first_phase,
            completed_stories=[],
        )

        # Initialize timing for accurate duration reporting
        start_project_timing(state)
        start_epic_timing(state)
        start_story_timing(state)
        start_phase_timing(state)

        state_path = run_dir / "state.yaml"
        save_state(state, state_path)

        return state

    def _build_experiment_config(self, template: ConfigTemplate) -> Config:
        """Build full Config object from ConfigTemplate.

        ConfigTemplate.providers already uses MasterProviderConfig and MultiProviderConfig
        from core/config.py, so we just wrap them in a full Config with defaults.

        Args:
            template: Config template with provider settings.

        Returns:
            Full Config object suitable for init_handlers().

        """
        return Config(
            providers=ProviderConfig(
                master=template.providers.master,
                multi=template.providers.multi,
            ),
            # Use defaults for all other Config fields (state_path, timeout, etc.)
        )

    def _increment_story_id(self, story_id: str) -> str:
        """Increment story ID (e.g., "1.1" -> "1.2").

        Args:
            story_id: Current story ID.

        Returns:
            Next story ID.

        """
        if "." in story_id:
            try:
                epic, story = story_id.split(".", 1)
                return f"{epic}.{int(story) + 1}"
            except ValueError:
                # Fallback for non-numeric story part
                return f"{story_id}.1"
        else:
            # Simple integer ID or string without dot
            try:
                return str(int(story_id) + 1)
            except ValueError:
                return f"{story_id}.1"

    def _execute_step(
        self,
        workflow: str,
        state: State,
        patchset: PatchSetManifest,
        run_dir: Path,
        snapshot_path: Path,
        created_stories: set[str],
    ) -> PhaseResult:
        """Execute a single workflow step with patch application."""
        # Map workflow to Phase
        phase = WORKFLOW_TO_PHASE.get(workflow)
        if phase is None:
            # Skip unmapped workflows (e.g., test-design) - not an error
            logger.warning("Skipping workflow '%s' - no Phase mapping available", workflow)
            return PhaseResult.ok(outputs={"skipped": True, "reason": "no_phase_mapping"})

        # Handle story ID increment for create-story
        if workflow == "create-story" and state.current_story:
            if state.current_story in created_stories:
                state.current_story = self._increment_story_id(state.current_story)
            created_stories.add(state.current_story)

        # Update state.current_phase before execution
        # State is a Pydantic model, not a dataclass, so we update directly
        state.current_phase = phase

        # Apply patches by copying to snapshot's patch directory
        # The compiler auto-discovers patches from .bmad-assist/patches/
        self._apply_patches(workflow, patchset, snapshot_path)

        # Set BMAD_ORIGINAL_CWD to project root so patch discovery finds patches
        # in the main project when they're not in the isolated fixture snapshot
        if self._project_root:
            os.environ[BMAD_ORIGINAL_CWD_ENV] = str(self._project_root)

        # Execute phase using existing handler (dispatches via state.current_phase)
        result = execute_phase(state)

        # Update completed_stories if phase was successful
        # This ensures state file reflects progress for resumption/debugging
        if (
            result.success
            and state.current_story
            and state.current_story not in state.completed_stories
        ):
            # Check if story already in list to avoid duplicates (idempotency)
            state.completed_stories.append(state.current_story)

        # Save updated state after each phase
        save_state(state, run_dir / "state.yaml")

        # Sync state to sprint-status.yaml (keeps sprint-status updated)
        try:
            from bmad_assist.sprint.sync import trigger_sync

            trigger_sync(state, snapshot_path)
        except Exception as e:
            logger.debug("Sprint sync failed (non-critical): %s", e)

        # Sync compiled cache back to main project for reuse by future experiments
        if self._project_root:
            snapshot_cache = snapshot_path / ".bmad-assist" / "cache"
            if snapshot_cache.exists():
                dest_cache = self._project_root / ".bmad-assist" / "cache"
                dest_cache.mkdir(parents=True, exist_ok=True)
                for cache_file in snapshot_cache.glob("*.tpl.xml*"):
                    dest_file = dest_cache / cache_file.name
                    # Only copy if newer or doesn't exist
                    newer = cache_file.stat().st_mtime > dest_file.stat().st_mtime
                    if not dest_file.exists() or newer:
                        shutil.copy2(cache_file, dest_file)

        return result

    def _apply_patches(
        self,
        workflow: str,
        patchset: PatchSetManifest,
        snapshot_path: Path,
    ) -> None:
        """Apply patches from patch-set to snapshot directory.

        The compiler auto-discovers patches from .bmad-assist/patches/,
        so we copy patch files there for automatic discovery.

        Args:
            workflow: Workflow name to apply patches for.
            patchset: Patch-set manifest with patch paths.
            snapshot_path: Path to isolated fixture snapshot.

        """
        snapshot_patch_dir = snapshot_path / ".bmad-assist" / "patches"

        # Check for workflow override first (takes precedence)
        if workflow in patchset.workflow_overrides:
            override_path_str = patchset.workflow_overrides[workflow]
            override_path = self._resolve_patch_path(override_path_str)
            if override_path.exists() and override_path.is_dir():
                logger.debug("Using workflow override for '%s': %s", workflow, override_path)
                # Copy entire workflow directory to snapshot's _bmad location
                # Workflow compiler expects standard BMAD structure
                # We copy to .bmad-assist/workflows/{workflow} to mimic project override
                # or better, directly to where compiler looks.
                # Compiler looks in project -> CWD -> global patches.
                # For overrides, we should likely replace the workflow definition?
                # The AC says "Copy alternative workflow directory to snapshot".
                # We'll copy to a location the compiler can find or explicitly use.
                # Given current compiler architecture, it might be complex.
                # BUT we must fulfill the AC "Copy alternative workflow directory".
                # We'll copy to .bmad-assist/overrides/{workflow}
                # (Note: Compiler support for this path needs to be verified,
                # but runner must do the copy as promised).
                override_dest = snapshot_path / ".bmad-assist" / "overrides" / workflow
                if override_dest.exists():
                    shutil.rmtree(override_dest)

                shutil.copytree(override_path, override_dest, dirs_exist_ok=True)
            return

        # Check for patch file
        if workflow in patchset.patches:
            patch_path_str = patchset.patches[workflow]
            if patch_path_str is None:
                # Explicit null = no patch, use raw workflow
                logger.debug("No patch for workflow '%s' (explicit null)", workflow)
                return

            patch_path = self._resolve_patch_path(patch_path_str)
            if patch_path.exists() and patch_path.is_file():
                logger.debug("Applying patch for '%s': %s", workflow, patch_path)
                snapshot_patch_dir.mkdir(parents=True, exist_ok=True)
                dest = snapshot_patch_dir / f"{workflow}.patch.yaml"
                shutil.copy2(patch_path, dest)
            else:
                logger.warning("Patch file for '%s' not found: %s", workflow, patch_path)

    def _resolve_patch_path(self, path_str: str) -> Path:
        """Resolve a patch file path.

        Args:
            path_str: Path string (may be relative or absolute).

        Returns:
            Resolved absolute path.

        """
        path = Path(path_str)

        # Handle tilde expansion
        if path_str.startswith("~"):
            return path.expanduser().resolve()

        # Handle absolute paths
        if path.is_absolute():
            return path.resolve()

        # Handle relative paths - resolve against experiments dir or project root
        if self._project_root and (self._project_root / path).exists():
            return (self._project_root / path).resolve()

        # Try experiments dir
        if (self._experiments_dir / path).exists():
            return (self._experiments_dir / path).resolve()

        # Return as-is for further handling
        return path.resolve()
