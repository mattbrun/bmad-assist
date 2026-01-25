"""Experiment fixtures route handlers.

Provides endpoints for listing and viewing experiment fixtures.
"""

import logging
from datetime import datetime
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


async def get_experiments_fixtures(request: Request) -> JSONResponse:
    """GET /api/experiments/fixtures - List fixtures with filters and sorting.

    Query parameters:
        tags: Comma-separated tags (AND logic)
        difficulty: Filter by difficulty level (easy, medium, hard)
        sort_by: Sort field (name, difficulty, estimated_cost, run_count)
        sort_order: Sort direction (asc, desc)

    Returns:
        JSON response with fixtures array and total count.

    """
    from bmad_assist.dashboard.experiments import (
        FixtureSummary,
        discover_fixtures,
        discover_runs,
        get_fixture_run_stats,
    )
    from bmad_assist.experiments import parse_cost

    server = request.app.state.server
    experiments_dir = server.project_root / "experiments"

    # Parse query params
    tags_param = request.query_params.get("tags", "")
    difficulty = request.query_params.get("difficulty")
    sort_by = request.query_params.get("sort_by", "name")
    sort_order = request.query_params.get("sort_order", "asc")

    # Parse tags (comma-separated)
    tags: list[str] = []
    if tags_param:
        tags = [t.strip() for t in tags_param.split(",") if t.strip()]

    # Validate difficulty
    # Support both legacy (easy/medium/hard) and current (trivial/simple/medium/complex/expert) values # noqa: E501
    valid_difficulties = {"trivial", "simple", "easy", "medium", "complex", "hard", "expert"}
    if difficulty and difficulty.lower() not in valid_difficulties:
        return JSONResponse(
            {
                "error": "invalid_difficulty",
                "message": f"Invalid difficulty: {difficulty}. Valid values: trivial, simple, medium, complex, expert", # noqa: E501
            },
            status_code=400,
        )

    # Validate sort_by
    valid_sort_fields = {"name", "difficulty", "estimated_cost", "run_count", "last_run"}
    if sort_by not in valid_sort_fields:
        return JSONResponse(
            {
                "error": "invalid_sort_by",
                "message": f"Invalid sort_by: {sort_by}. Valid values: name, difficulty, estimated_cost, run_count, last_run", # noqa: E501
            },
            status_code=400,
        )

    # Validate sort_order
    if sort_order not in {"asc", "desc"}:
        return JSONResponse(
            {
                "error": "invalid_sort_order",
                "message": f"Invalid sort_order: {sort_order}. Valid values: asc, desc",
            },
            status_code=400,
        )

    try:
        # Discover fixtures (returns list[FixtureEntry])
        fixtures_list = await discover_fixtures(experiments_dir)
        if not fixtures_list:
            return JSONResponse({"fixtures": [], "total": 0})

        # Get all fixtures, then filter
        fixtures = list(fixtures_list)

        # Apply tag filter (AND logic)
        if tags:
            tag_set = set(tags)
            fixtures = [f for f in fixtures if tag_set.issubset(set(f.tags))]

        # Apply difficulty filter
        if difficulty:
            difficulty_lower = difficulty.lower()
            fixtures = [f for f in fixtures if f.difficulty == difficulty_lower]

        # Get run statistics
        runs = await discover_runs(experiments_dir)
        stats = get_fixture_run_stats(fixtures, runs)

        # Build response
        summaries: list[FixtureSummary] = []
        for f in fixtures:
            fixture_stats = stats.get(f.id)
            run_count = fixture_stats.run_count if fixture_stats else 0
            last_run = fixture_stats.last_run if fixture_stats else None
            summaries.append(
                FixtureSummary(
                    id=f.id,
                    name=f.name,
                    description=f.description,
                    path=str(f.path) if f.path else "",
                    tags=list(f.tags),
                    difficulty=f.difficulty or "medium",
                    estimated_cost=f.estimated_cost or "",
                    estimated_cost_value=parse_cost(f.estimated_cost) if f.estimated_cost else 0.0,
                    run_count=run_count,
                    last_run=last_run,
                )
            )

        # Apply sorting
        # Difficulty ordering: trivial < simple < easy < medium < complex < hard < expert
        difficulty_order = {
            "trivial": 0,
            "simple": 1,
            "easy": 2,
            "medium": 3,
            "complex": 4,
            "hard": 5,
            "expert": 6,
        }
        sort_key_funcs: dict[str, Any] = {
            "name": lambda s: s.name.lower(),
            "difficulty": lambda s: difficulty_order.get(s.difficulty, 99),
            "estimated_cost": lambda s: s.estimated_cost_value,
            "run_count": lambda s: s.run_count,
            # For last_run: None values go last when ascending, first when descending
            "last_run": lambda s: (s.last_run is None, s.last_run or datetime.min),
        }
        summaries.sort(key=sort_key_funcs[sort_by], reverse=(sort_order == "desc"))

        return JSONResponse(
            {
                "fixtures": [s.model_dump(mode="json") for s in summaries],
                "total": len(summaries),
            }
        )

    except Exception:
        logger.exception("Failed to list fixtures")
        return JSONResponse(
            {"error": "server_error", "message": "Internal server error"},
            status_code=500,
        )


async def get_experiment_fixture(request: Request) -> JSONResponse:
    """GET /api/experiments/fixtures/{fixture_id} - Get fixture details.

    Path parameters:
        fixture_id: The fixture identifier

    Returns:
        JSON response with full fixture details or 404.

    """
    from bmad_assist.dashboard.experiments import (
        FixtureDetails,
        discover_fixtures,
        discover_runs,
        get_fixture_run_stats,
        validate_run_id,
    )
    from bmad_assist.experiments import parse_cost

    server = request.app.state.server
    fixture_id = request.path_params["fixture_id"]

    # Validate fixture_id format (reuse pattern from runs - same format)
    if not validate_run_id(fixture_id):
        return JSONResponse(
            {"error": "bad_request", "message": f"Invalid fixture_id format: {fixture_id}"},
            status_code=400,
        )

    experiments_dir = server.project_root / "experiments"

    try:
        # Discover fixtures (returns list[FixtureEntry])
        fixtures_list = await discover_fixtures(experiments_dir)
        if not fixtures_list:
            return JSONResponse(
                {"error": "not_found", "message": f"Fixture not found: {fixture_id}"},
                status_code=404,
            )

        # Find fixture by ID (case-sensitive for exact match)
        fixture = None
        for f in fixtures_list:
            if f.id == fixture_id:
                fixture = f
                break

        if fixture is None:
            return JSONResponse(
                {"error": "not_found", "message": f"Fixture not found: {fixture_id}"},
                status_code=404,
            )

        # FixtureEntry.path is already absolute Path in new auto-discovery API
        resolved_path = fixture.path

        # Get run statistics
        runs = await discover_runs(experiments_dir)
        stats = get_fixture_run_stats([fixture], runs)
        fixture_stats = stats.get(fixture.id)
        run_count = fixture_stats.run_count if fixture_stats else 0
        last_run = fixture_stats.last_run if fixture_stats else None
        recent_runs = fixture_stats.recent_runs if fixture_stats else []

        # Build response
        details = FixtureDetails(
            id=fixture.id,
            name=fixture.name,
            description=fixture.description,
            path=str(fixture.path) if fixture.path else "",
            resolved_path=str(resolved_path),
            tags=list(fixture.tags),
            difficulty=fixture.difficulty or "medium",
            estimated_cost=fixture.estimated_cost or "",
            estimated_cost_value=parse_cost(fixture.estimated_cost) if fixture.estimated_cost else 0.0, # noqa: E501
            run_count=run_count,
            last_run=last_run,
            recent_runs=recent_runs,
        )

        return JSONResponse(details.model_dump(mode="json"))

    except Exception:
        logger.exception("Failed to get fixture %s", fixture_id)
        return JSONResponse(
            {"error": "server_error", "message": "Internal server error"},
            status_code=500,
        )


routes = [
    Route("/api/experiments/fixtures", get_experiments_fixtures, methods=["GET"]),
    Route("/api/experiments/fixtures/{fixture_id}", get_experiment_fixture, methods=["GET"]),
]
