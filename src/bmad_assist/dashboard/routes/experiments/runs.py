"""Experiment runs route handlers.

Provides endpoints for listing, viewing, triggering, and canceling experiment runs.
"""

import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


async def get_experiments_runs(request: Request) -> JSONResponse:
    """GET /api/experiments/runs - List experiment runs with filters.

    Query parameters:
        status: Filter by ExperimentStatus value
        fixture: Filter by fixture ID (case-insensitive)
        config: Filter by config template name (case-insensitive)
        patch_set: Filter by patch-set name (case-insensitive)
        loop: Filter by loop template name (case-insensitive)
        start_date: ISO date (YYYY-MM-DD) for runs started on or after
        end_date: ISO date (YYYY-MM-DD) for runs started on or before
        offset: Pagination offset (default: 0)
        limit: Maximum results per page (default: 20, max: 100)
        sort_by: Sort field (started, completed, duration, status)
        sort_order: Sort direction (asc, desc)

    Returns:
        JSON response with runs array and pagination metadata.

    """
    from bmad_assist.dashboard.experiments import (
        discover_runs,
        filter_runs,
        manifest_to_summary,
        sort_runs,
    )

    server = request.app.state.server

    # Parse query params
    status = request.query_params.get("status")
    fixture = request.query_params.get("fixture")
    config = request.query_params.get("config")
    patch_set = request.query_params.get("patch_set")
    loop = request.query_params.get("loop")
    start_date = request.query_params.get("start_date")
    end_date = request.query_params.get("end_date")

    try:
        offset = int(request.query_params.get("offset", "0"))
        if offset < 0:
            return JSONResponse(
                {"error": "invalid_pagination", "message": "offset must be non-negative"},
                status_code=400,
            )
        limit = min(int(request.query_params.get("limit", "20")), 100)
        if limit < 1:
            return JSONResponse(
                {"error": "invalid_pagination", "message": "limit must be at least 1"},
                status_code=400,
            )
    except ValueError as e:
        return JSONResponse(
            {"error": "invalid_pagination", "message": str(e)},
            status_code=400,
        )

    sort_by = request.query_params.get("sort_by", "started")
    sort_order = request.query_params.get("sort_order", "desc")

    if sort_by not in ("started", "completed", "duration", "status"):
        return JSONResponse(
            {"error": "invalid_sort_by", "message": f"Invalid sort_by: {sort_by}"},
            status_code=400,
        )

    if sort_order not in ("asc", "desc"):
        return JSONResponse(
            {"error": "invalid_sort_order", "message": f"Invalid sort_order: {sort_order}"},
            status_code=400,
        )

    experiments_dir = server.project_root / "experiments"

    try:
        # Discover runs (uses thread-safe cache)
        runs = await discover_runs(experiments_dir)

        # Apply filters
        runs = filter_runs(
            runs,
            status=status,
            fixture=fixture,
            config=config,
            patch_set=patch_set,
            loop=loop,
            start_date=start_date,
            end_date=end_date,
        )

        # Sort
        runs = sort_runs(runs, sort_by, sort_order)

        # Paginate
        total = len(runs)
        runs = runs[offset : offset + limit]

        # Build response
        run_summaries = [manifest_to_summary(r) for r in runs]

        return JSONResponse(
            {
                "runs": [r.model_dump(mode="json") for r in run_summaries],
                "pagination": {
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": offset + len(runs) < total,
                },
            }
        )

    except ValueError as e:
        return JSONResponse(
            {"error": "filter_error", "message": str(e)},
            status_code=400,
        )
    except Exception as e:
        logger.exception("Failed to list experiment runs")
        return JSONResponse(
            {"error": "server_error", "message": str(e)},
            status_code=500,
        )


async def get_experiment_run(request: Request) -> JSONResponse:
    """GET /api/experiments/runs/{run_id} - Get experiment run details.

    Path parameters:
        run_id: The experiment run identifier

    Returns:
        JSON response with full run details including phases, metrics, and resolved config.

    """
    from bmad_assist.dashboard.experiments import (
        get_run_by_id,
        manifest_to_details,
        validate_run_id,
    )

    server = request.app.state.server
    run_id = request.path_params["run_id"]

    # Validate run_id format first (security: prevent path traversal)
    if not validate_run_id(run_id):
        return JSONResponse(
            {"error": "bad_request", "message": f"Invalid run_id format: {run_id}"},
            status_code=400,
        )

    experiments_dir = server.project_root / "experiments"

    try:
        manifest = await get_run_by_id(run_id, experiments_dir)
        if manifest is None:
            return JSONResponse(
                {"error": "not_found", "message": f"Run not found: {run_id}"},
                status_code=404,
            )

        details = manifest_to_details(manifest)
        return JSONResponse(details.model_dump(mode="json"))

    except Exception:
        logger.exception("Failed to get experiment run %s", run_id)
        return JSONResponse(
            {"error": "server_error", "message": "Internal server error"},
            status_code=500,
        )


async def get_experiment_manifest(request: Request) -> JSONResponse:
    """GET /api/experiments/runs/{run_id}/manifest - Get raw manifest.

    Path parameters:
        run_id: The experiment run identifier

    Returns:
        JSON response with the raw manifest data.

    """
    from bmad_assist.dashboard.experiments import (
        get_run_by_id,
        validate_run_id,
    )

    server = request.app.state.server
    run_id = request.path_params["run_id"]

    # Validate run_id format first (security: prevent path traversal)
    if not validate_run_id(run_id):
        return JSONResponse(
            {"error": "bad_request", "message": f"Invalid run_id format: {run_id}"},
            status_code=400,
        )

    experiments_dir = server.project_root / "experiments"

    try:
        manifest = await get_run_by_id(run_id, experiments_dir)
        if manifest is None:
            return JSONResponse(
                {"error": "not_found", "message": f"Run not found: {run_id}"},
                status_code=404,
            )

        return JSONResponse(manifest.model_dump(mode="json"))

    except Exception:
        logger.exception("Failed to get manifest for run %s", run_id)
        return JSONResponse(
            {"error": "server_error", "message": "Internal server error"},
            status_code=500,
        )


async def post_experiment_run(request: Request) -> JSONResponse:
    """POST /api/experiments/run - Trigger a new experiment run.

    Request body:
        fixture: Fixture name
        config: Config template name
        patch_set: Patch-set name
        loop: Loop template name

    Returns:
        JSON response with run_id and status, or error.

    """
    from pydantic import ValidationError

    from bmad_assist.dashboard.experiments import (
        ExperimentRunRequest,
        discover_configs,
        discover_fixtures,
        discover_loops,
        discover_patchsets,
    )

    server = request.app.state.server
    experiments_dir = server.project_root / "experiments"

    # Parse request body
    try:
        body = await request.json()
        run_request = ExperimentRunRequest(**body)
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": "bad_request", "message": f"Invalid JSON: {e}"},
            status_code=400,
        )
    except ValidationError as e:
        return JSONResponse(
            {"error": "bad_request", "message": str(e)},
            status_code=400,
        )

    # Check for concurrent run (use lock to prevent race condition)
    async with server._experiment_lock:
        if server._active_experiment_run_id is not None:
            return JSONResponse(
                {
                    "error": "conflict",
                    "message": f"Experiment {server._active_experiment_run_id} is already running",
                },
                status_code=409,
            )

        # Validate templates exist
        try:
            # Check fixtures (returns list[FixtureEntry])
            fixtures_list = await discover_fixtures(experiments_dir)
            if not fixtures_list:
                return JSONResponse(
                    {"error": "bad_request", "message": "No fixtures registry found"},
                    status_code=400,
                )
            fixture_ids = [f.id for f in fixtures_list]
            if run_request.fixture not in fixture_ids:
                return JSONResponse(
                    {
                        "error": "bad_request",
                        "message": f"Fixture not found: {run_request.fixture}",
                    },
                    status_code=400,
                )

            # Check configs
            configs = await discover_configs(experiments_dir)
            config_names = [c[0] for c in configs]
            if run_request.config not in config_names:
                return JSONResponse(
                    {"error": "bad_request", "message": f"Config not found: {run_request.config}"},
                    status_code=400,
                )

            # Check loops
            loops = await discover_loops(experiments_dir)
            loop_names = [loop_entry[0] for loop_entry in loops]
            if run_request.loop not in loop_names:
                return JSONResponse(
                    {"error": "bad_request", "message": f"Loop not found: {run_request.loop}"},
                    status_code=400,
                )

            # Check patch-sets
            patchsets = await discover_patchsets(experiments_dir)
            patchset_names = [p[0] for p in patchsets]
            if run_request.patch_set not in patchset_names:
                return JSONResponse(
                    {
                        "error": "bad_request",
                        "message": f"Patch-set not found: {run_request.patch_set}",
                    },
                    status_code=400,
                )

        except Exception:
            logger.exception("Failed to validate templates")
            return JSONResponse(
                {"error": "server_error", "message": "Failed to validate templates"},
                status_code=500,
            )

        # Generate run_id (check for collisions)
        runs_dir = experiments_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        run_counter = 1

        while True:
            run_id = f"run-{today}-{run_counter:03d}"
            if not (runs_dir / run_id).exists():
                break
            run_counter += 1
            if run_counter > 999:
                return JSONResponse(
                    {
                        "error": "server_error",
                        "message": "Too many runs today, cannot generate unique run_id",
                    },
                    status_code=500,
                )

        # TODO: Implement experiment execution via subprocess (similar to bmad-assist run loop)
        # Experiments should run directly, not via queue
        return JSONResponse(
            {
                "error": "not_implemented",
                "message": "Experiment execution via dashboard is not yet supported. Use CLI: `bmad-assist experiment run ...`", # noqa: E501
            },
            status_code=501,
        )


async def get_experiment_run_status(request: Request) -> Response:
    """GET /api/experiments/run/{run_id}/status - SSE stream for experiment progress.

    Path parameters:
        run_id: The experiment run identifier

    Returns:
        SSE stream with status, progress, output, and complete events.

    """
    from bmad_assist.dashboard.experiments import validate_run_id

    server = request.app.state.server
    run_id = request.path_params["run_id"]

    # Validate run_id format
    if not validate_run_id(run_id):
        return JSONResponse(
            {"error": "bad_request", "message": f"Invalid run_id format: {run_id}"},
            status_code=400,
        )

    # Check if run exists (active or completed)
    is_active = server._active_experiment_run_id == run_id

    if not is_active:
        # Check if already completed (check runs directory)
        experiments_dir = server.project_root / "experiments"
        runs_dir = experiments_dir / "runs"
        if not (runs_dir / run_id).exists():
            return JSONResponse(
                {"error": "not_found", "message": f"Run not found: {run_id}"},
                status_code=404,
            )

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events for this experiment."""
        # Subscribe to SSE broadcaster and filter by run_id
        async for message in server.sse_broadcaster.subscribe():
            # Send message to client
            yield message

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def post_experiment_run_cancel(request: Request) -> JSONResponse:
    """POST /api/experiments/run/{run_id}/cancel - Cancel experiment.

    Path parameters:
        run_id: The experiment run identifier

    Returns:
        JSON response with cancellation status.

    """
    from bmad_assist.dashboard.experiments import (
        ExperimentCancelResponse,
        validate_run_id,
    )

    server = request.app.state.server
    run_id = request.path_params["run_id"]

    # Validate run_id format
    if not validate_run_id(run_id):
        return JSONResponse(
            {"error": "bad_request", "message": f"Invalid run_id format: {run_id}"},
            status_code=400,
        )

    # Check if it's the active run
    if server._active_experiment_run_id == run_id:
        # Cancel running experiment via cancellation event
        if server._active_experiment_cancel_event:
            server._active_experiment_cancel_event.set()

        # Broadcast cancellation event
        await server.sse_broadcaster.broadcast_event(
            "experiment:complete",
            {"status": "cancelled", "run_id": run_id},
        )

        return JSONResponse(
            ExperimentCancelResponse(
                run_id=run_id,
                status="cancelled",
                message="Experiment cancelled",
            ).model_dump()
        )

    # Check if already completed (check runs directory)
    experiments_dir = server.project_root / "experiments"
    runs_dir = experiments_dir / "runs"
    if (runs_dir / run_id).exists():
        return JSONResponse(
            {"error": "conflict", "message": f"Experiment {run_id} already completed"},
            status_code=409,
        )

    return JSONResponse(
        {"error": "not_found", "message": f"Run not found: {run_id}"},
        status_code=404,
    )


routes = [
    Route("/api/experiments/runs", get_experiments_runs, methods=["GET"]),
    Route("/api/experiments/runs/{run_id}", get_experiment_run, methods=["GET"]),
    Route("/api/experiments/runs/{run_id}/manifest", get_experiment_manifest, methods=["GET"]),
    Route("/api/experiments/run", post_experiment_run, methods=["POST"]),
    Route("/api/experiments/run/{run_id}/status", get_experiment_run_status, methods=["GET"]),
    Route("/api/experiments/run/{run_id}/cancel", post_experiment_run_cancel, methods=["POST"]),
]
