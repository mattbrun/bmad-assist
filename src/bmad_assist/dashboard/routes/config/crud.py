"""Config CRUD route handlers.

Provides endpoints for reading and updating configuration.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from anyio import to_thread
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from bmad_assist.core.config import PROJECT_CONFIG_NAME, reload_config
from bmad_assist.core.exceptions import ConfigError, ConfigValidationError

from . import utils

logger = logging.getLogger(__name__)


async def get_config(request: Request) -> JSONResponse:
    """GET /api/config - Return merged config with provenance.

    Returns merged configuration (defaults → global → project) with source
    information for each value. DANGEROUS fields are excluded.
    """
    async with utils._config_editor_lock:
        try:
            editor = await to_thread.run_sync(lambda: utils._create_config_editor(request))
            merged = await to_thread.run_sync(editor.get_merged_with_provenance)

            # Filter dangerous fields
            full_schema = utils._get_full_schema()
            filtered = utils._filter_dangerous_fields(merged, full_schema)

            return JSONResponse(filtered)
        except ConfigError as e:
            logger.exception("Failed to get config")
            return JSONResponse(
                {"error": "config_error", "message": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception("Failed to get config")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


async def get_config_global(request: Request) -> JSONResponse:
    """GET /api/config/global - Return global config only with provenance.

    Returns raw global configuration with all sources marked as "global".
    DANGEROUS fields are excluded.
    """
    async with utils._config_editor_lock:
        try:
            editor = await to_thread.run_sync(lambda: utils._create_config_editor(request))
            global_data = await to_thread.run_sync(editor.get_global_raw)

            # Add provenance (all global)
            with_provenance = utils._add_provenance_to_raw(global_data, "global")

            # Filter dangerous fields
            full_schema = utils._get_full_schema()
            filtered = utils._filter_dangerous_fields(with_provenance, full_schema)

            return JSONResponse(filtered)
        except ConfigError as e:
            logger.exception("Failed to get global config")
            return JSONResponse(
                {"error": "config_error", "message": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception("Failed to get global config")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


async def get_config_project(request: Request) -> JSONResponse:
    """GET /api/config/project - Return project config only with provenance.

    Returns raw project configuration with all sources marked as "project".
    Returns empty object if no project config exists.
    DANGEROUS fields are excluded.
    """
    async with utils._config_editor_lock:
        try:
            editor = await to_thread.run_sync(lambda: utils._create_config_editor(request))
            project_data = await to_thread.run_sync(editor.get_project_raw)

            if not project_data:
                return JSONResponse({})

            # Add provenance (all project)
            with_provenance = utils._add_provenance_to_raw(project_data, "project")

            # Filter dangerous fields
            full_schema = utils._get_full_schema()
            filtered = utils._filter_dangerous_fields(with_provenance, full_schema)

            return JSONResponse(filtered)
        except ConfigError as e:
            logger.exception("Failed to get project config")
            return JSONResponse(
                {"error": "config_error", "message": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception("Failed to get project config")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


async def put_config_global(request: Request) -> JSONResponse:
    """PUT /api/config/global - Update global configuration.

    Request body:
    {
        "updates": [
            {"path": "benchmarking.enabled", "value": false},
            {"path": "testarch.playwright.browsers", "value": ["chromium", "firefox"]}
        ],
        "confirmed": false
    }

    Returns 428 Precondition Required if RISKY fields without confirmed=true.
    Returns 403 if trying to update DANGEROUS fields.
    Returns 400 for invalid paths or validation errors.
    """
    return await _put_config(request, "global")


async def put_config_project(request: Request) -> JSONResponse:
    """PUT /api/config/project - Update project configuration.

    Same request/response format as PUT /api/config/global.
    Returns 400 if no project path is configured.
    """
    server = request.app.state.server
    project_config_path = server.project_root / PROJECT_CONFIG_NAME

    # Check if project config exists or can be created
    if not project_config_path.exists() and not server.project_root.exists():
        return JSONResponse(
            {"error": "no_project", "message": "No project path configured"},
            status_code=400,
        )

    return await _put_config(request, "project")


async def _put_config(request: Request, scope: str) -> JSONResponse:
    """Handle PUT /api/config/{scope} request.

    Args:
        request: Starlette request.
        scope: "global" or "project".

    Returns:
        JSONResponse with updated config or error.

    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": "invalid_json", "message": f"Invalid JSON body: {e}"},
            status_code=400,
        )

    updates = body.get("updates", [])
    confirmed = body.get("confirmed", False)

    # Empty updates is a no-op - return current config
    if not updates:
        return await get_config(request)

    # Validate updates structure
    if not isinstance(updates, list):
        return JSONResponse(
            {"error": "invalid_updates", "message": "updates must be an array"},
            status_code=400,
        )

    full_schema = utils._get_full_schema()

    # Validate all paths and collect info
    paths = []
    dangerous_paths = []
    risky_paths = []

    for update in updates:
        if not isinstance(update, dict):
            return JSONResponse(
                {"error": "invalid_update", "message": "Each update must be an object"},
                status_code=400,
            )

        path = update.get("path")
        if not path or not isinstance(path, str):
            return JSONResponse(
                {"error": "invalid_path", "path": str(path), "message": "path is required"},
                status_code=400,
            )

        # Story 17.8: Validate value key exists (can be null for delete operations)
        if "value" not in update:
            return JSONResponse(
                {"error": "invalid_update", "path": path, "message": "value is required"},
                status_code=400,
            )

        # Check path exists in schema
        exists, error_msg = utils._validate_path_exists(path, full_schema)
        if not exists:
            return JSONResponse(
                {"error": "invalid_path", "path": path, "message": error_msg},
                status_code=400,
            )

        paths.append(path)

        # Check security level
        security = utils._get_field_security(path, full_schema)
        if security == "dangerous":
            dangerous_paths.append(path)
        elif security == "risky":
            risky_paths.append(path)

    # Reject dangerous field updates
    if dangerous_paths:
        return JSONResponse(
            {
                "error": "forbidden",
                "message": "Cannot modify dangerous fields",
                "dangerous_fields": dangerous_paths,
            },
            status_code=403,
        )

    # Require confirmation for risky fields
    if risky_paths and not confirmed:
        return JSONResponse(
            {
                "requires_confirmation": True,
                "risky_fields": risky_paths,
            },
            status_code=428,
        )

    # Apply updates
    async with utils._config_editor_lock:
        try:
            server = request.app.state.server

            def apply_updates_and_get_merged() -> dict[str, Any]:
                editor = utils._create_config_editor(request)

                # For project scope, ensure project path is set
                if scope == "project" and editor.project_path is None:
                    # Create project config path
                    project_path = server.project_root / PROJECT_CONFIG_NAME
                    editor.project_path = project_path

                for update in updates:
                    # Story 17.8: Handle null values as delete operations
                    # This enables "Reset to default" / "Reset to global" functionality
                    if update["value"] is None:
                        editor.remove(scope, update["path"])
                    else:
                        editor.update(scope, update["path"], update["value"])

                # Validate before saving
                editor.validate()
                editor.save(scope)

                # Return merged config (within same lock)
                return editor.get_merged_with_provenance()

            merged = await to_thread.run_sync(apply_updates_and_get_merged)

            # Filter dangerous fields
            filtered = utils._filter_dangerous_fields(merged, full_schema)
            return JSONResponse(filtered)

        except ConfigValidationError as e:
            # Return structured validation errors with 422 status
            logger.warning("Config validation failed: %s", e)
            # Convert loc tuples to lists for JSON serialization
            details = []
            for err in e.errors:
                loc = err["loc"]
                details.append(
                    {
                        "loc": list(loc) if isinstance(loc, tuple) else loc,
                        "msg": err["msg"],
                        "type": err["type"],
                    }
                )
            return JSONResponse(
                {"error": "validation_failed", "details": details},
                status_code=422,
            )
        except ConfigError as e:
            logger.exception("Config error occurred")
            return JSONResponse(
                {"error": "config_error", "message": str(e)},
                status_code=400,
            )
        except ValueError as e:
            logger.exception("Config update failed")
            return JSONResponse(
                {"error": "update_error", "message": str(e)},
                status_code=400,
            )
        except Exception as e:
            logger.exception("Failed to update config")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


async def post_config_reload(request: Request) -> JSONResponse:
    """POST /api/config/reload - Reload configuration singleton.

    Performs atomic reload of the config singleton without restart.
    Broadcasts SSE event to notify clients.
    """
    server = request.app.state.server

    async with utils._config_editor_lock:
        try:
            await to_thread.run_sync(lambda: reload_config(server.project_root))

            # Broadcast config_reloaded event
            if server.sse_broadcaster is not None:
                timestamp = datetime.now(UTC).isoformat()
                await server.sse_broadcaster.broadcast_event(
                    "config_reloaded",
                    {"timestamp": timestamp, "source": "api"},
                )

            return JSONResponse(
                {
                    "reloaded": True,
                    "message": "Configuration reloaded",
                }
            )
        except ConfigError as e:
            logger.exception("Failed to reload config")
            return JSONResponse(
                {"error": "reload_error", "message": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception("Failed to reload config")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


async def get_loop_config(request: Request) -> JSONResponse:
    """GET /api/loop-config - Return current loop configuration.

    Returns the resolved loop configuration with fallback chain applied.
    Hot-reloads from config files on each request.

    Query Parameters:
        None

    Returns:
        JSON object with:
        - epic_setup: list of phase IDs to run at epic start
        - story: list of phase IDs to run for each story
        - epic_teardown: list of phase IDs to run at epic end

    Example Response:
        {
            "epic_setup": [],
            "story": [
                "create_story",
                "validate_story",
                "validate_story_synthesis",
                "dev_story",
                "code_review",
                "code_review_synthesis"
            ],
            "epic_teardown": ["retrospective"]
        }

    """
    from bmad_assist.core.config import load_loop_config

    # Get project path from app state (set by DashboardServer)
    server = request.app.state.server
    loop_config = load_loop_config(server.project_root)

    return JSONResponse(
        {
            "epic_setup": loop_config.epic_setup,
            "story": loop_config.story,
            "epic_teardown": loop_config.epic_teardown,
        }
    )


routes = [
    Route("/api/config", get_config, methods=["GET"]),
    Route("/api/config/global", get_config_global, methods=["GET"]),
    Route("/api/config/project", get_config_project, methods=["GET"]),
    Route("/api/config/global", put_config_global, methods=["PUT"]),
    Route("/api/config/project", put_config_project, methods=["PUT"]),
    Route("/api/config/reload", post_config_reload, methods=["POST"]),
    Route("/api/loop-config", get_loop_config, methods=["GET"]),
]
