"""Config backup route handlers.

Provides endpoints for listing, viewing, and restoring config backups.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from anyio import to_thread
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from bmad_assist.core.config import reload_config
from bmad_assist.core.config_editor import ConfigEditor
from bmad_assist.core.exceptions import ConfigError

from . import utils

logger = logging.getLogger(__name__)


async def get_config_backups(request: Request) -> JSONResponse:
    """GET /api/config/backups?scope=global|project - List available backups.

    Query params:
        scope: Required. "global" or "project".

    Returns list of backup metadata.
    """
    scope = request.query_params.get("scope")

    if not scope:
        return JSONResponse(
            {"error": "missing_scope", "message": "scope query parameter is required"},
            status_code=400,
        )

    if scope not in ("global", "project"):
        return JSONResponse(
            {"error": "invalid_scope", "message": "scope must be 'global' or 'project'"},
            status_code=400,
        )

    async with utils._config_editor_lock:
        try:
            editor = await to_thread.run_sync(lambda: utils._create_config_editor(request))
            backups = await to_thread.run_sync(lambda: editor.list_backups(scope))
            return JSONResponse({"backups": backups})
        except Exception as e:
            logger.exception("Failed to list backups")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


async def get_config_backup_content(request: Request) -> Response:
    """GET /api/config/backup/content?scope=<scope>&version=<version> - Get backup content.

    Story 17.10 AC6: View backup content in modal.
    This dedicated endpoint allows viewing both global and project backups safely.

    Query params:
        scope: Required. "global" or "project".
        version: Required. Backup version number (1 = newest, 5 = oldest).

    Returns backup content as text/plain for YAML display.
    Returns 404 if backup version doesn't exist.
    """
    scope = request.query_params.get("scope")
    version_str = request.query_params.get("version")

    if not scope:
        return JSONResponse(
            {"error": "missing_scope", "message": "scope query parameter is required"},
            status_code=400,
        )

    if scope not in ("global", "project"):
        return JSONResponse(
            {"error": "invalid_scope", "message": "scope must be 'global' or 'project'"},
            status_code=400,
        )

    if not version_str:
        return JSONResponse(
            {"error": "missing_version", "message": "version query parameter is required"},
            status_code=400,
        )

    try:
        version = int(version_str)
        if version < 1 or version > ConfigEditor.MAX_BACKUPS:
            raise ValueError("out of range")
    except ValueError:
        return JSONResponse(
            {
                "error": "invalid_version",
                "message": f"version must be integer 1-{ConfigEditor.MAX_BACKUPS}",
            },
            status_code=400,
        )

    async with utils._config_editor_lock:
        try:
            editor = await to_thread.run_sync(lambda: utils._create_config_editor(request))

            # Get the backup file path
            base_path: Path | None
            base_path = editor.global_path if scope == "global" else editor.project_path

            if base_path is None:
                return JSONResponse(
                    {"error": "not_found", "message": f"No {scope} config available"},
                    status_code=404,
                )

            backup_path = Path(f"{base_path}.{version}")
            if not backup_path.exists():
                return JSONResponse(
                    {"error": "not_found", "message": f"Backup version {version} not found"},
                    status_code=404,
                )

            # Read and return content
            content = await to_thread.run_sync(lambda: backup_path.read_text(encoding="utf-8"))
            return Response(content, media_type="text/plain; charset=utf-8")

        except Exception as e:
            logger.exception("Failed to read backup content")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


async def post_config_restore(request: Request) -> JSONResponse:
    """POST /api/config/restore - Restore config from backup.

    Request body:
    {
        "scope": "global" | "project",
        "version": 1
    }

    Validates backup content before applying.
    Returns 404 if backup version doesn't exist.
    Returns 400 if backup content is invalid.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": "invalid_json", "message": f"Invalid JSON body: {e}"},
            status_code=400,
        )

    scope = body.get("scope")
    version = body.get("version")

    if not scope:
        return JSONResponse(
            {"error": "missing_scope", "message": "scope is required"},
            status_code=400,
        )

    if scope not in ("global", "project"):
        return JSONResponse(
            {"error": "invalid_scope", "message": "scope must be 'global' or 'project'"},
            status_code=400,
        )

    if version is None:
        return JSONResponse(
            {"error": "missing_version", "message": "version is required"},
            status_code=400,
        )

    if not isinstance(version, int) or version < 1:
        return JSONResponse(
            {"error": "invalid_version", "message": "version must be a positive integer"},
            status_code=400,
        )

    server = request.app.state.server

    async with utils._config_editor_lock:
        try:
            editor = await to_thread.run_sync(lambda: utils._create_config_editor(request))

            def do_restore() -> None:
                editor.restore_backup(scope, version)

            await to_thread.run_sync(do_restore)

            # Reload config singleton to apply restored config
            await to_thread.run_sync(lambda: reload_config(server.project_root))

            # Story 17.9: Broadcast config_reloaded event to notify other clients
            if server.sse_broadcaster is not None:
                timestamp = datetime.now(UTC).isoformat()
                await server.sse_broadcaster.broadcast_event(
                    "config_reloaded",
                    {"timestamp": timestamp, "source": "api"},
                )

            return JSONResponse(
                {
                    "restored": True,
                    "version": version,
                    "scope": scope,
                }
            )
        except ConfigError as e:
            error_msg = str(e)
            if "does not exist" in error_msg:
                return JSONResponse(
                    {"error": "not_found", "message": error_msg},
                    status_code=404,
                )
            # Validation error
            logger.exception("Backup validation failed")
            return JSONResponse(
                {"error": "validation_error", "message": error_msg},
                status_code=400,
            )
        except Exception as e:
            logger.exception("Failed to restore backup")
            return JSONResponse(
                {"error": "server_error", "message": str(e)},
                status_code=500,
            )


routes = [
    Route("/api/config/backups", get_config_backups, methods=["GET"]),
    Route("/api/config/backup/content", get_config_backup_content, methods=["GET"]),
    Route("/api/config/restore", post_config_restore, methods=["POST"]),
]
