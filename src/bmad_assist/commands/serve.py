"""Serve command for bmad-assist CLI.

Starts the dashboard web server.
"""

import typer

from bmad_assist.cli_utils import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    _error,
    _validate_project_path,
    console,
)
from bmad_assist.core.config import load_config_with_project
from bmad_assist.core.exceptions import ConfigError, DashboardError
from bmad_assist.core.paths import init_paths


def serve_command(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        # NOTE: No -h short form - conflicts with --help
        help="Address to bind server to",
    ),
    port: int = typer.Option(
        9600,
        "--port",
        "-p",
        help="Port to bind server to",
    ),
    project: str = typer.Option(
        ".",
        "--project",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (shows HTTP request logs)",
    ),
    no_auto_port: bool = typer.Option(
        False,
        "--no-auto-port",
        help="Fail if port is busy instead of auto-discovering",
    ),
) -> None:
    """Start the dashboard web server.

    Provides real-time visibility into bmad-assist execution via web browser.
    """
    import asyncio

    from bmad_assist.dashboard import DashboardServer
    from bmad_assist.dashboard.server import find_available_port

    # Validate project path
    project_path = _validate_project_path(project)

    # Config loading is optional - wrap in try/except per AC4, AC7
    try:
        loaded_config = load_config_with_project(project_path=project_path)
        # Initialize paths singleton if config loaded
        paths_config = {
            "output_folder": loaded_config.paths.output_folder,
            "planning_artifacts": loaded_config.paths.planning_artifacts,
            "implementation_artifacts": loaded_config.paths.implementation_artifacts,
            "project_knowledge": loaded_config.paths.project_knowledge,
        }
        # Add bmad_paths.epics if configured (supports custom epic locations)
        if loaded_config.bmad_paths and loaded_config.bmad_paths.epics:
            paths_config["epics"] = loaded_config.bmad_paths.epics
        init_paths(project_path, paths_config)
    except ConfigError:
        # No config - that's OK, dashboard reads BMAD files directly
        # DashboardServer.get_sprint_status() handles missing BMAD project gracefully
        pass

    # Build dashboard static files from partials (REQUIRED before server start)
    console.print("[dim]Building dashboard static files...[/dim]")
    from bmad_assist.dashboard.build_static import build

    try:
        build()
    except Exception as e:
        _error(f"Failed to build dashboard static files: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    # Port auto-discovery (unless --no-auto-port)
    actual_port = port
    if not no_auto_port:
        try:
            actual_port = find_available_port(port, host)
            if actual_port != port:
                console.print(f"[yellow]Port {port} unavailable, using port {actual_port}[/yellow]")
        except DashboardError as e:
            _error(str(e))
            raise typer.Exit(code=EXIT_ERROR) from None

    # Display startup message (AC1)
    console.print("[green]Dashboard server starting...[/green]")
    console.print(f"  URL: http://{host}:{actual_port}/")
    console.print(f"  Project: {project_path}")
    console.print("  Press Ctrl+C to stop")

    # Create server (uses DashboardServer.run() which has proper signal handlers)
    server = DashboardServer(project_root=project_path, host=host, port=actual_port)
    log_level = "debug" if verbose else "info"

    # Run server with error handling
    try:
        asyncio.run(server.run(log_level=log_level))
    except OSError as e:
        import errno

        if e.errno == errno.EADDRINUSE:
            _error(f"Port {actual_port} is already in use")
            raise typer.Exit(code=EXIT_ERROR) from None
        raise

    # Shutdown message and clean exit (AC5)
    console.print("\n[yellow]Server stopped.[/yellow]")
    raise typer.Exit(code=EXIT_SUCCESS)
