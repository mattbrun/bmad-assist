"""Content route handlers.

Provides endpoints for prompts, validation reports, report content, and
reviewer identity mapping (Story 23.8).
"""

import logging
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from bmad_assist.dashboard.utils.validator_mapping import get_mapping_for_story

logger = logging.getLogger(__name__)


async def get_prompt(request: Request) -> Response:
    """GET /api/prompt/{epic}/{story}/{phase} - Get compiled prompt.

    Returns the cached prompt file for a specific workflow phase.
    Prompts are phase-agnostic; epic/story params retained for API consistency.
    """
    server = request.app.state.server

    epic = request.path_params["epic"]
    story = request.path_params["story"]
    phase = request.path_params["phase"]

    try:
        prompt_path = server.get_prompt_path(epic, story, phase)
        if prompt_path and prompt_path.exists():
            content = prompt_path.read_text(encoding="utf-8")
            return Response(
                content,
                media_type="text/plain; charset=utf-8",  # AC 1.4
                headers={"X-Prompt-Path": str(prompt_path)},
            )
        else:
            # Story 24.2 AC3: Use "Prompt" terminology in error message
            return JSONResponse(
                {"error": f"Prompt not found for phase: {phase}"},
                status_code=404,
            )
    except Exception as e:
        logger.exception("Failed to get prompt")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_validation(request: Request) -> Response:
    """GET /api/validation/{epic}/{story} - Get validation reports.

    Returns validation report files for a story.
    """
    server = request.app.state.server

    epic = request.path_params["epic"]
    story = request.path_params["story"]

    try:
        reports = server.get_validation_reports(epic, story)
        return JSONResponse(reports)
    except Exception as e:
        logger.exception("Failed to get validation reports")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_reviews(request: Request) -> Response:
    """GET /api/reviews/{epic}/{story} - Get all review reports (validation + code-review).

    Returns both validation and code-review reports for a story,
    each with their own synthesis and individual reports.
    """
    server = request.app.state.server

    epic = request.path_params["epic"]
    story = request.path_params["story"]

    try:
        reviews = server.get_all_reviews(epic, story)
        return JSONResponse(reviews)
    except Exception as e:
        logger.exception("Failed to get reviews")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_report_content(request: Request) -> Response:
    """GET /api/report/content?path=<path> - Get report file content.

    Returns the content of a report file with security validation.
    Path must be within project root and not a symlink.
    """
    path_param = request.query_params.get("path")

    # AC 2b.1: path is required
    if not path_param:
        return JSONResponse({"error": "Missing path parameter"}, status_code=400)

    server = request.app.state.server
    report_path = Path(path_param)

    # Security validation
    try:
        # AC 2b.3: Reject symlinks before resolving (defense against symlink attacks)
        if report_path.is_symlink():
            return JSONResponse({"error": "Symlinks not allowed"}, status_code=403)

        # AC 2b.2: Resolve and validate path containment
        resolved_path = report_path.resolve()
        project_root = server.project_root.resolve()

        # Use is_relative_to for proper path containment check (Python 3.9+)
        if not resolved_path.is_relative_to(project_root):
            return JSONResponse({"error": "Path outside project"}, status_code=403)
    except (ValueError, OSError):
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    # AC 2b.4: Return 404 for missing file
    if not resolved_path.exists():
        return JSONResponse({"error": "Report not found"}, status_code=404)

    # AC 2b.5: Return content as text/plain
    try:
        content = resolved_path.read_text(encoding="utf-8")
        return Response(content, media_type="text/plain; charset=utf-8")
    except Exception as e:
        logger.exception("Failed to read report file")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_story_content(request: Request) -> Response:
    """GET /api/story/{epic}/{story}/content - Get story file content.

    Story 24.5: Returns story file content from implementation-artifacts.
    Searches for files matching pattern {epic}-{story}-*.md and returns
    the most recent one if multiple matches exist.

    Returns:
        JSON with content, file_path, and title on success.
        404 JSON error if story file not found.

    """
    server = request.app.state.server

    epic = request.path_params["epic"]
    story = request.path_params["story"]

    try:
        result = server.get_story_file_content(epic, story)
        if result is None:
            return JSONResponse(
                {"error": f"Story file not found for {epic}.{story}"},
                status_code=404,
            )
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Failed to get story content")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_reviewer_mapping(request: Request) -> Response:
    """GET /api/mapping/{type}/{epic}/{story} - Get reviewer identity mapping.

    Story 23.8: Returns mapping of anonymous validator IDs (Validator A, B, C...)
    to actual model names. Used by frontend to replace anonymous IDs with
    real model names in reports and prompts.

    Path params:
        type: Mapping type - "validation" or "code-review"
        epic: Epic identifier
        story: Story number

    Query params:
        session_id: Optional session_id for direct lookup

    Returns:
        JSON with validators mapping and metadata:
        {
            "session_id": "...",
            "timestamp": "...",
            "validators": {"Validator A": "glm-4.7", "Validator B": "opus", ...}
        }

        Returns 404 if no mapping found.

    """
    server = request.app.state.server

    mapping_type = request.path_params["type"]
    epic = request.path_params["epic"]
    story = request.path_params["story"]
    session_id = request.query_params.get("session_id")

    # Validate mapping type
    if mapping_type not in ("validation", "code-review"):
        return JSONResponse(
            {"error": f"Invalid mapping type: {mapping_type}. Use 'validation' or 'code-review'"},
            status_code=400,
        )

    try:
        mapping = get_mapping_for_story(
            server.project_root,
            mapping_type,
            epic,
            story,
            session_id,
        )

        if mapping is None:
            return JSONResponse(
                {"error": f"No {mapping_type} mapping found for {epic}-{story}"},
                status_code=404,
            )

        return JSONResponse(mapping)

    except Exception as e:
        logger.exception("Failed to get reviewer mapping")
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/api/prompt/{epic}/{story}/{phase}", get_prompt, methods=["GET"]),
    Route("/api/validation/{epic}/{story}", get_validation, methods=["GET"]),
    Route("/api/reviews/{epic}/{story}", get_reviews, methods=["GET"]),
    Route("/api/report/content", get_report_content, methods=["GET"]),
    Route("/api/mapping/{type}/{epic}/{story}", get_reviewer_mapping, methods=["GET"]),
    Route("/api/story/{epic}/{story}/content", get_story_content, methods=["GET"]),
]
