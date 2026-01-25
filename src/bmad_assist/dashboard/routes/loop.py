"""Loop control route handlers.

Provides endpoints for starting, pausing, resuming, and stopping the BMAD loop.
"""

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


async def start_loop(request: Request) -> JSONResponse:
    """POST /api/loop/start - Start the BMAD development loop.

    Runs `bmad-assist run` subprocess which processes stories from sprint-status.yaml.
    """
    server = request.app.state.server

    try:
        result = await server.start_loop()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Failed to start loop")
        return JSONResponse({"error": str(e)}, status_code=500)


async def pause_loop(request: Request) -> JSONResponse:
    """POST /api/loop/pause - Pause the processing loop.

    Pause after current workflow completes. Does NOT interrupt current workflow.
    """
    server = request.app.state.server

    try:
        result = await server.pause_loop()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Failed to pause loop")
        return JSONResponse({"error": str(e)}, status_code=500)


async def resume_loop(request: Request) -> JSONResponse:
    """POST /api/loop/resume - Resume a paused processing loop.

    Story 22.10 - Task 2: Resume functionality backend.
    """
    server = request.app.state.server

    try:
        result = await server.resume_loop()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Failed to resume loop")
        return JSONResponse({"error": str(e)}, status_code=500)


async def stop_loop(request: Request) -> JSONResponse:
    """POST /api/loop/stop - Stop the processing loop immediately.

    Terminates the current workflow.
    """
    server = request.app.state.server

    try:
        result = await server.stop_loop()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Failed to stop loop")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_loop_status(request: Request) -> JSONResponse:
    """GET /api/loop/status - Get loop running status.

    Returns whether the loop is currently running.
    Checks both dashboard internal state AND running.lock file
    to detect runs started externally (CLI, context menu, etc.).

    Story 22.3: Add PID validation to distinguish active runs from stale locks.
    Story 22.10: Return "paused" status when loop_running=True and pause_requested=True.
    """
    server = request.app.state.server

    # Check dashboard internal state
    running = server._loop_running
    paused = server._pause_requested

    # Determine status
    if running and paused:
        status = "paused"
    elif running:
        status = "running"
    else:
        status = "idle"

    # Also check lock file for external runs
    if not running:
        from bmad_assist.core.loop.locking import _is_pid_alive, _read_lock_file

        lock_path = server.project_root / ".bmad-assist" / "running.lock"
        if lock_path.exists():
            pid, timestamp = _read_lock_file(lock_path)
            if pid is not None and _is_pid_alive(pid):
                # Active lock - PID is running
                running = True
                status = "running"
            else:
                # Stale lock - PID not running or invalid file
                status = "crashed"

    return JSONResponse(
        {
            "running": running,
            "paused": paused,
            "status": status,
        }
    )


routes = [
    Route("/api/loop/start", start_loop, methods=["POST"]),
    Route("/api/loop/pause", pause_loop, methods=["POST"]),
    Route("/api/loop/resume", resume_loop, methods=["POST"]),
    Route("/api/loop/stop", stop_loop, methods=["POST"]),
    Route("/api/loop/status", get_loop_status, methods=["GET"]),
]
