"""Notification event dispatch for the loop runner.

Story 15.4: Fire-and-forget notification dispatch to external systems.
Extracted from runner.py as part of the runner refactoring.

"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from bmad_assist.core.state import State

logger = logging.getLogger(__name__)

__all__ = ["_dispatch_event"]

# Type alias for state parameter
LoopState = State


def _dispatch_event(
    event_type: str,
    project_path: Path,
    state: LoopState,
    **extra_fields: str | int | None,
) -> None:
    """Fire-and-forget dispatch of notification events.

    Runs dispatch in a new event loop to not block the main loop.
    All errors are caught and logged - never raises.

    Args:
        event_type: Event type name (e.g., "story_started", "phase_completed").
        project_path: Project root path for project name.
        state: Current loop state for epic/story info.
        **extra_fields: Additional fields for payload (phase, duration_ms, etc.).

    """
    try:
        from bmad_assist.notifications.dispatcher import get_dispatcher  # noqa: I001
        from bmad_assist.notifications.events import (  # noqa: I001
            EpicCompletedPayload,
            ErrorOccurredPayload,
            EventPayload,
            EventType,
            PhaseCompletedPayload,
            ProjectCompletedPayload,
            QueueBlockedPayload,
            StoryCompletedPayload,
            StoryStartedPayload,
        )

        dispatcher = get_dispatcher()
        if dispatcher is None:
            return

        # Build payload based on event type
        project = project_path.name
        # Default epic/story to safe values if None
        epic = state.current_epic if state.current_epic is not None else 0
        story = state.current_story if state.current_story is not None else "unknown"

        payload: EventPayload
        event: EventType

        if event_type == "story_started":
            event = EventType.STORY_STARTED
            phase_str = extra_fields.get("phase")
            story_title = extra_fields.get("story_title")
            payload = StoryStartedPayload(
                project=project,
                epic=epic,
                story=story,
                phase=str(phase_str) if phase_str else "",
                story_title=str(story_title) if story_title else None,
            )
        elif event_type == "story_completed":
            event = EventType.STORY_COMPLETED
            duration = extra_fields.get("duration_ms")
            outcome = extra_fields.get("outcome")
            payload = StoryCompletedPayload(
                project=project,
                epic=epic,
                story=story,
                duration_ms=int(duration) if duration else 0,
                outcome=str(outcome) if outcome else "success",
            )
        elif event_type == "phase_completed":
            event = EventType.PHASE_COMPLETED
            phase_val = extra_fields.get("phase")
            next_phase = extra_fields.get("next_phase")
            duration = extra_fields.get("duration_ms")
            payload = PhaseCompletedPayload(
                project=project,
                epic=epic,
                story=story,
                phase=str(phase_val) if phase_val else "",
                next_phase=str(next_phase) if next_phase else None,
                duration_ms=int(duration) if duration else 0,
            )
        elif event_type == "error_occurred":
            event = EventType.ERROR_OCCURRED
            error_type_val = extra_fields.get("error_type")
            message_val = extra_fields.get("message")
            stack_val = extra_fields.get("stack_trace")
            payload = ErrorOccurredPayload(
                project=project,
                epic=epic,
                story=story,
                error_type=str(error_type_val) if error_type_val else "unknown",
                message=str(message_val) if message_val else "",
                stack_trace=str(stack_val) if stack_val else None,
            )
        elif event_type == "queue_blocked":
            event = EventType.QUEUE_BLOCKED
            reason_val = extra_fields.get("reason")
            waiting_val = extra_fields.get("waiting_tasks")
            payload = QueueBlockedPayload(
                project=project,
                epic=epic,
                story=story,
                reason=str(reason_val) if reason_val else "guardian_halt",
                waiting_tasks=int(waiting_val) if waiting_val else 0,
            )
        # Story standalone-03 AC6: Epic completion event
        elif event_type == "epic_completed":
            event = EventType.EPIC_COMPLETED
            duration = extra_fields.get("duration_ms")
            stories_completed = extra_fields.get("stories_completed")
            payload = EpicCompletedPayload(
                project=project,
                epic=epic,
                duration_ms=int(duration) if duration else 0,
                stories_completed=int(stories_completed) if stories_completed else 0,
            )
        # Story standalone-03 AC7: Project completion event
        elif event_type == "project_completed":
            event = EventType.PROJECT_COMPLETED
            duration = extra_fields.get("duration_ms")
            epics_completed = extra_fields.get("epics_completed")
            stories_completed = extra_fields.get("stories_completed")
            payload = ProjectCompletedPayload(
                project=project,
                epic=epic,
                duration_ms=int(duration) if duration else 0,
                epics_completed=int(epics_completed) if epics_completed else 0,
                stories_completed=int(stories_completed) if stories_completed else 0,
            )
        else:
            logger.debug("Unknown event type for dispatch: %s", event_type)
            return

        # Run dispatch with nested event loop safety (AC3 requirement)
        # Check if we're inside an already-running event loop (test environments)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - create task (fire-and-forget)
            loop.create_task(dispatcher.dispatch(event, payload))
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            asyncio.run(dispatcher.dispatch(event, payload))

    except Exception as e:
        logger.debug("Notification dispatch error (ignored): %s", str(e))
