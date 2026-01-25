"""Playwright executor for Category B tests.

Executes Playwright tests directly without LLM.
Generates spec files from test plan and runs via npx playwright test.
Manages server lifecycle for E2E tests.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bmad_assist.core.types import EpicId
from bmad_assist.qa.parser import TestCase

if TYPE_CHECKING:
    from bmad_assist.core.config import PlaywrightConfig

logger = logging.getLogger(__name__)

# Track running processes for cleanup on exit
_running_processes: set[subprocess.Popen[str]] = set()
_cleanup_registered = False


def _cleanup_processes() -> None:
    """Kill all tracked processes on exit."""
    for proc in list(_running_processes):
        try:
            if proc.poll() is None:  # Still running
                logger.debug("Cleanup: killing process %d", proc.pid)
                # Kill process group if possible (Unix)
                if sys.platform != "win32":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        proc.terminate()
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception as e:
            logger.debug("Cleanup error for process %d: %s", proc.pid, e)
        _running_processes.discard(proc)


def _register_cleanup() -> None:
    """Register cleanup handler (once)."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_processes)
        _cleanup_registered = True


def _track_process(proc: subprocess.Popen[str]) -> None:
    """Track a process for cleanup."""
    _register_cleanup()
    _running_processes.add(proc)


def _untrack_process(proc: subprocess.Popen[str]) -> None:
    """Stop tracking a process."""
    _running_processes.discard(proc)


@dataclass
class PlaywrightTestResult:
    """Result from a single Playwright test.

    Attributes:
        test_id: Test identifier (E19-B01).
        name: Test name.
        status: PASS, FAIL, or SKIP.
        duration_ms: Test duration in milliseconds.
        error: Error message if failed.
        output: Test output/logs.

    """

    test_id: str
    name: str
    status: str
    duration_ms: int = 0
    error: str = ""
    output: str = ""


@dataclass
class PlaywrightExecutionResult:
    """Result of Playwright execution.

    Attributes:
        success: Whether execution completed.
        results: List of test results.
        spec_file: Path to generated spec file.
        total_duration_ms: Total execution time.
        error: Error message if execution failed.

    """

    success: bool
    results: list[PlaywrightTestResult]
    spec_file: Path | None = None
    total_duration_ms: int = 0
    error: str = ""


# =============================================================================
# Server Management
# =============================================================================


def is_server_running(url: str, timeout: float = 2.0) -> bool:
    """Check if server is responding at URL.

    Args:
        url: URL to check (e.g., "http://localhost:8765").
        timeout: Request timeout in seconds.

    Returns:
        True if server responds with any HTTP status, False otherwise.

    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        # Server is running but returned error (e.g., 404, 500)
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def wait_for_server(url: str, timeout: int = 30, poll_interval: float = 0.5) -> bool:
    """Wait for server to become ready.

    Args:
        url: URL to poll (e.g., "http://localhost:8765").
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between checks.

    Returns:
        True if server became ready, False if timeout.

    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if is_server_running(url):
            elapsed = time.monotonic() - start
            logger.info("Server ready at %s (%.1fs)", url, elapsed)
            return True
        time.sleep(poll_interval)
    return False


def start_server_process(
    command: str,
    cwd: Path,
    startup_timeout: int = 30,
    base_url: str = "http://localhost:3000",
) -> subprocess.Popen[str]:
    """Start server subprocess and wait for it to be ready.

    Args:
        command: Shell command to start server (e.g., "npm run dev").
        cwd: Working directory for the command.
        startup_timeout: Seconds to wait for server to be ready.
        base_url: URL to poll for readiness.

    Returns:
        Running subprocess.Popen object.

    Raises:
        RuntimeError: If server fails to start or become ready.

    """
    logger.info("Starting server: %s", command)

    try:
        # Use start_new_session to create process group (for clean kill)
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # New process group for clean cleanup
        )
        _track_process(process)  # Track for atexit cleanup
    except Exception as e:
        raise RuntimeError(f"Failed to start server: {e}") from e

    # Wait for server to be ready
    if not wait_for_server(base_url, timeout=startup_timeout):
        # Check if process died
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            _untrack_process(process)
            raise RuntimeError(
                f"Server process exited with code {process.returncode}: {stderr[:500]}"
            )
        # Process running but not responding - kill it
        _kill_process_group(process)
        _untrack_process(process)
        raise RuntimeError(f"Server did not become ready at {base_url} within {startup_timeout}s")

    logger.info("Server started successfully (PID %d)", process.pid)
    return process


def _kill_process_group(process: subprocess.Popen[str], timeout: int = 5) -> None:
    """Kill a process and its entire process group."""
    if process.poll() is not None:
        return  # Already dead

    # Try to kill process group (Unix)
    if sys.platform != "win32":
        try:
            pgid = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
                process.wait(timeout=timeout)
                return
            except (ProcessLookupError, PermissionError):
                pass  # Fall through to regular terminate
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    process.wait(timeout=2)
                    return
                except Exception:
                    pass

    # Fallback: just terminate the process
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def stop_server_process(process: subprocess.Popen[str], timeout: int = 10) -> None:
    """Stop server subprocess gracefully.

    Kills entire process group to ensure child processes are also terminated.

    Args:
        process: Server subprocess to stop.
        timeout: Seconds to wait for graceful shutdown before killing.

    """
    if process.poll() is not None:
        logger.debug("Server already stopped (exit code %d)", process.returncode)
        _untrack_process(process)
        return

    logger.info("Stopping server (PID %d)...", process.pid)
    _kill_process_group(process, timeout=timeout)
    _untrack_process(process)
    logger.info("Server stopped")


@contextmanager
def managed_server(
    config: PlaywrightConfig | None,
    project_path: Path,
) -> Generator[bool, None, None]:
    """Context manager for server lifecycle.

    Starts server if configured and not already running.
    Stops server on exit if we started it.

    Args:
        config: Playwright configuration (None = no server management).
        project_path: Project root directory.

    Yields:
        True if server is available (either started or already running).

    Example:
        with managed_server(config, project_path) as server_ok:
            if server_ok:
                run_tests()

    """
    if config is None:
        # No config - assume server management is manual
        yield True
        return

    base_url = config.base_url
    server_config = config.server
    process: subprocess.Popen[str] | None = None

    # Check if server is already running
    if is_server_running(base_url):
        if server_config.reuse_existing:
            logger.info("Server already running at %s", base_url)
            yield True
            return
        else:
            logger.error("Server already running at %s but reuse_existing=False", base_url)
            yield False
            return

    # Server not running - check if we should start it
    if not server_config.command:
        logger.warning("Server not running at %s and no start command configured", base_url)
        yield False
        return

    # Start server
    try:
        process = start_server_process(
            command=server_config.command,
            cwd=project_path,
            startup_timeout=server_config.startup_timeout,
            base_url=base_url,
        )
        yield True
    except RuntimeError as e:
        logger.error("Failed to start server: %s", e)
        yield False
        return
    finally:
        # Stop server if we started it
        if process is not None:
            stop_server_process(process)


def check_playwright_available(project_path: Path | None = None) -> tuple[bool, str]:
    """Check if Playwright is installed and available.

    Args:
        project_path: Optional project root to check for local installation.

    Returns:
        Tuple of (available, version_or_error).

    """
    try:
        result = subprocess.run(
            ["npx", "playwright", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_path,
        )
        if result.returncode == 0:
            version = result.stdout.strip()

            # Check if @playwright/test is installed locally
            if project_path:
                pkg_json = project_path / "package.json"
                if pkg_json.exists():
                    import json

                    try:
                        pkg = json.loads(pkg_json.read_text())
                        deps = pkg.get("dependencies", {})
                        dev_deps = pkg.get("devDependencies", {})
                        if "@playwright/test" not in deps and "@playwright/test" not in dev_deps:
                            return False, (
                                f"Playwright CLI found ({version}) but @playwright/test "
                                "not in package.json. Run: npm install -D @playwright/test"
                            )
                    except json.JSONDecodeError:
                        pass  # Ignore invalid package.json

            return True, version
        return False, result.stderr or "Unknown error"
    except FileNotFoundError:
        return False, "npx not found - Node.js may not be installed"
    except subprocess.TimeoutExpired:
        return False, "Timeout checking Playwright version"
    except Exception as e:
        return False, str(e)


def generate_playwright_config(
    output_dir: Path,
    config: PlaywrightConfig | None = None,
) -> Path:
    """Generate playwright.config.ts with reporters and artifact settings.

    Creates a config that:
    - Uses 'list' reporter for real-time progress to terminal
    - Uses 'json' reporter writing to results.json for parsing
    - Configures screenshot/video/trace capture
    - Sets artifact output directory

    Args:
        output_dir: Directory for config and artifacts.
        config: Playwright configuration (None = defaults).

    Returns:
        Path to generated config file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "playwright.config.ts"
    artifacts_dir = output_dir / "artifacts"
    results_file = output_dir / "results.json"

    # Get settings from config or use defaults
    screenshot = config.screenshot if config else "only-on-failure"
    video = config.video if config else "retain-on-failure"
    trace = config.trace if config else "retain-on-failure"
    headless = config.headless if config else True
    timeout_ms = (config.timeout if config else 300) * 1000

    # Action timeout (for element lookups) - 10 seconds is a good balance
    action_timeout_ms = 10000

    config_content = f"""// Auto-generated Playwright config
import {{ defineConfig }} from '@playwright/test';

export default defineConfig({{
  // Reporters: line for compact progress, json for parsing
  // 'line' is quieter than 'list' - no call logs on failure
  reporter: [
    ['line'],
    ['json', {{ outputFile: '{results_file}' }}]
  ],

  // Artifact capture settings
  use: {{
    screenshot: '{screenshot}',
    video: '{video}',
    trace: '{trace}',
    headless: {str(headless).lower()},
    // Large viewport to ensure footer is visible
    viewport: {{ width: 1920, height: 1080 }},
    // Balanced timeout (10s instead of 30s default)
    actionTimeout: {action_timeout_ms},
  }},

  // Expect timeout (for assertions)
  expect: {{
    timeout: {action_timeout_ms},
  }},

  // Output directory for artifacts
  outputDir: '{artifacts_dir}',

  // Timeout settings (per test)
  timeout: {timeout_ms},

  // Run sequentially for clearer output
  fullyParallel: false,
  workers: 1,

  // Quiet mode - less verbose output
  quiet: true,
}});
"""
    config_path.write_text(config_content, encoding="utf-8")
    logger.debug("Generated Playwright config: %s", config_path)
    return config_path


def generate_spec_file(
    tests: list[TestCase],
    epic_id: EpicId,
    output_dir: Path,
    base_url: str = "http://localhost:8765",
) -> Path:
    """Generate Playwright spec file from test cases.

    Combine all TypeScript test blocks into a single spec file
    that Playwright can execute.

    Args:
        tests: List of Category B test cases with TypeScript scripts.
        epic_id: Epic identifier.
        output_dir: Directory to write spec file.
        base_url: Base URL for the test server.

    Returns:
        Path to generated spec file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / f"epic-{epic_id}.spec.ts"

    # Build spec file content
    lines = [
        "// Auto-generated Playwright spec file",
        f"// Epic: {epic_id}",
        f"// Generated: {datetime.now(UTC).isoformat()}",
        f"// Tests: {len(tests)}",
        "",
        "import { test, expect } from '@playwright/test';",
        "",
        "// Helper: click element via pure JS - bypasses all Playwright visibility checks",
        "async function forceClick(page: any, selector: string) {",
        "  await page.evaluate((sel: string) => {",
        "    const el = document.querySelector(sel) as HTMLElement;",
        "    if (el) { el.scrollIntoView({ block: 'center' }); el.click(); }",
        "  }, selector);",
        "}",
        "",
    ]

    for tc in tests:
        if not tc.script:
            # Generate skip placeholder for tests without scripts
            lines.append(f"test.skip('{tc.id}: {tc.name}', async () => {{")
            lines.append(f"  // No script provided for {tc.id}")
            lines.append("});")
            lines.append("")
            continue

        # Check if script already contains test() wrapper or test.describe()
        # Script may have imports/helpers before the actual test, so check entire script
        script = tc.script.strip()
        script_lines = script.split("\n")
        has_test_wrapper = any(
            line.strip().startswith("test(")
            or line.strip().startswith("test.")
            or line.strip().startswith("test.describe(")
            for line in script_lines
        )

        # Also check if script has imports - if so, it's self-contained
        has_imports = any(line.strip().startswith("import ") for line in script_lines)
        if has_imports:
            has_test_wrapper = True  # Treat as self-contained
            # Strip imports and forceClick helper - they're already in the header
            cleaned_lines = []
            skip_forceclick = False
            for line in script_lines:
                stripped = line.strip()
                # Skip import lines
                if stripped.startswith("import "):
                    continue
                # Skip forceClick helper (multi-line)
                if "async function forceClick" in stripped:
                    skip_forceclick = True
                    continue
                if skip_forceclick:
                    if stripped == "}":
                        skip_forceclick = False
                    continue
                cleaned_lines.append(line)
            script = "\n".join(cleaned_lines).strip()
            script_lines = script.split("\n")

        if has_test_wrapper:
            # Script already has test wrapper - inject our ID into the test name
            # Replace test('name' or test("name" with test('ID: name'
            test_id = tc.id
            id_prefix = f"{test_id}: "

            def inject_test_id(
                match: re.Match[str],
                *,
                _test_id: str = test_id,
                _id_prefix: str = id_prefix,
            ) -> str:
                """Inject test ID into test name, avoiding duplicates."""
                prefix = match.group(1)  # test( or test.skip( etc.
                quote = match.group(2)  # ' or "
                name = match.group(3)  # original test name
                # Don't double-prefix if ID already present
                if name.startswith(_test_id):
                    return match.group(0)
                return f"{prefix}{quote}{_id_prefix}{name}{quote}"

            # Match: test('name') or test("name") or test.skip('name') etc.
            modified_script = re.sub(
                r"(test(?:\.[a-z]+)?\s*\()(['\"])([^'\"]+)\2",
                inject_test_id,
                script,
                count=1,  # Only first occurrence
            )
            lines.append(modified_script)
        else:
            # Wrap script in test() function
            lines.append(f"test('{tc.id}: {tc.name}', async ({{ page }}) => {{")
            # Indent the script content
            for line in script_lines:
                lines.append(f"  {line}")
            lines.append("});")
        lines.append("")

    spec_content = "\n".join(lines)

    # Post-process: normalize all localhost URLs to configured base_url
    # QA plans may have hardcoded different ports
    spec_content = re.sub(
        r"http://localhost:\d+",
        base_url.rstrip("/"),
        spec_content,
    )

    # Post-process: replace 'networkidle' with 'domcontentloaded'
    # Dashboard uses SSE (Server-Sent Events) which keeps connection open forever
    # so 'networkidle' will never complete - use 'domcontentloaded' instead
    spec_content = spec_content.replace("'networkidle'", "'domcontentloaded'")
    spec_content = spec_content.replace('"networkidle"', '"domcontentloaded"')

    # Post-process: replace page.click for footer buttons with forceClick
    # These elements may be off-screen in some viewport configurations
    footer_buttons = [
        "experiments-button",
        "settings-button",
    ]
    for btn in footer_buttons:
        # Match both single and double quotes
        pattern = (
            rf"await page\.click\(\s*['\"]"
            rf"(\[data-testid=\"{btn}\"\])"
            rf"['\"](?:\s*,\s*\{{[^}}]*\}})?\s*\);"
        )
        spec_content = re.sub(pattern, r"await forceClick(page, '\1');", spec_content)

    spec_path.write_text(spec_content, encoding="utf-8")

    logger.info("Generated spec file: %s (%d tests)", spec_path, len(tests))
    return spec_path


def run_playwright_tests(
    spec_file: Path,
    project_path: Path,
    config_file: Path,
    results_file: Path,
    process_timeout: int = 600,
    headless: bool = True,
) -> tuple[int, str, str]:
    """Run Playwright tests via npx with real-time progress output.

    Uses a generated playwright.config.ts that:
    - Shows progress via 'list' reporter (to terminal)
    - Writes JSON results to file for parsing

    Note: process_timeout is for the ENTIRE test suite execution.
    Per-test timeouts are configured in playwright.config.ts.

    Args:
        spec_file: Path to spec file to execute.
        project_path: Project root directory.
        config_file: Path to playwright.config.ts.
        results_file: Path where JSON results will be written.
        process_timeout: Timeout for entire process in seconds (NOT per-test).
        headless: Run in headless mode.

    Returns:
        Tuple of (exit_code, json_output, stderr).

    """
    cmd = [
        "npx",
        "playwright",
        "test",
        str(spec_file),
        f"--config={config_file}",
    ]

    # Playwright is headless by default, --headed shows browser
    if not headless:
        cmd.append("--headed")

    logger.info("Running: %s", " ".join(cmd))
    print()  # Blank line before Playwright output

    process: subprocess.Popen[str] | None = None
    try:
        # Use Popen so we can track and kill on Ctrl+C
        process = subprocess.Popen(
            cmd,
            stdout=None,  # Inherit - shows to terminal
            stderr=subprocess.PIPE,  # Capture errors
            text=True,
            cwd=project_path,
            start_new_session=True,  # New process group for clean cleanup
        )
        _track_process(process)

        # Wait for completion with timeout
        try:
            _, stderr = process.communicate(timeout=process_timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Playwright process timed out after %ds, killing...", process_timeout)
            _kill_process_group(process)
            _untrack_process(process)
            print()
            return -1, "", f"Playwright process timed out after {process_timeout}s"

        _untrack_process(process)
        print()  # Blank line after Playwright output

        # Read JSON results from file (written by json reporter)
        json_output = ""
        if results_file.exists():
            json_output = results_file.read_text(encoding="utf-8")
            logger.debug("Read %d bytes from %s", len(json_output), results_file.name)
        else:
            logger.warning("Results file not created: %s", results_file)

        return process.returncode, json_output, stderr or ""

    except KeyboardInterrupt:
        # User pressed Ctrl+C - kill Playwright cleanly
        logger.info("Interrupted by user, stopping Playwright...")
        if process is not None:
            _kill_process_group(process)
            _untrack_process(process)
        print()
        raise  # Re-raise so caller knows about interrupt
    except Exception as e:
        if process is not None:
            _kill_process_group(process)
            _untrack_process(process)
        return -1, "", str(e)


def parse_playwright_json_results(
    json_output: str,
    stderr: str,
    tests: list[TestCase],
) -> list[PlaywrightTestResult]:
    """Parse Playwright JSON reporter output.

    Maps Playwright results back to our test IDs.

    Args:
        json_output: JSON output from Playwright (stdout).
        stderr: Error output from Playwright.
        tests: Original test cases for ID mapping.

    Returns:
        List of PlaywrightTestResult.

    """
    results: list[PlaywrightTestResult] = []

    # Build test name to ID mapping (multiple formats for flexibility)
    name_to_test: dict[str, TestCase] = {}
    for tc in tests:
        # Match by test ID
        name_to_test[tc.id] = tc
        # Match by "ID: Name" format (our generated tests)
        name_to_test[f"{tc.id}: {tc.name}"] = tc
        # Match by just name (tests with existing wrapper)
        name_to_test[tc.name] = tc
        # Match by lowercase name (case-insensitive fallback)
        name_to_test[tc.name.lower()] = tc

    # If stdout is empty, check stderr for errors
    if not json_output.strip():
        error_msg = stderr[:500] if stderr else "No output from Playwright"
        logger.error("Playwright produced no JSON output. Stderr: %s", error_msg[:200])
        return [
            PlaywrightTestResult(
                test_id=tc.id,
                name=tc.name,
                status="ERROR",
                error=error_msg[:200],
            )
            for tc in tests
        ]

    try:
        data = json.loads(json_output)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Playwright JSON: %s", e)
        logger.debug("Raw output (first 500 chars): %s", json_output[:500])
        # Return all tests as ERROR
        return [
            PlaywrightTestResult(
                test_id=tc.id,
                name=tc.name,
                status="ERROR",
                error=f"Failed to parse results: {e}",
            )
            for tc in tests
        ]

    # Check for root-level errors (e.g., missing module, config errors)
    root_errors = data.get("errors", [])
    if root_errors and not data.get("suites"):
        # Execution failed before tests could run
        error_msg = root_errors[0].get("message", "Unknown error")[:300]
        logger.error("Playwright setup error: %s", error_msg[:100])
        return [
            PlaywrightTestResult(
                test_id=tc.id,
                name=tc.name,
                status="ERROR",
                error=error_msg,
            )
            for tc in tests
        ]

    # Playwright JSON format has suites > specs > tests
    suites = data.get("suites", [])
    parsed_ids: set[str] = set()

    for suite in suites:
        for spec in suite.get("specs", []):
            title = spec.get("title", "")
            # Extract test ID from title (format: "E19-B01: Test name")
            test_id = title.split(":")[0].strip() if ":" in title else title

            # Try multiple lookup strategies
            found_tc = (
                name_to_test.get(test_id)
                or name_to_test.get(title)
                or name_to_test.get(title.lower())  # Case-insensitive fallback
            )
            if found_tc is None:
                logger.warning("Unknown test in results: %s", title)
                continue

            # Get test results (may have multiple if retried)
            test_results = spec.get("tests", [])
            if not test_results:
                continue

            # Use last result (after retries)
            last_result = test_results[-1]
            results_list = last_result.get("results", [])
            if not results_list:
                continue

            last_run = results_list[-1]
            status_map = {
                "passed": "PASS",
                "failed": "FAIL",
                "skipped": "SKIP",
                "timedOut": "ERROR",
            }

            pw_status = last_run.get("status", "failed")
            status = status_map.get(pw_status, "ERROR")
            duration = last_run.get("duration", 0)

            error = ""
            if status == "FAIL":
                # Extract error message
                error_obj = last_run.get("error", {})
                error = error_obj.get("message", "")[:500]

            results.append(
                PlaywrightTestResult(
                    test_id=found_tc.id,
                    name=found_tc.name,
                    status=status,
                    duration_ms=duration,
                    error=error,
                )
            )
            parsed_ids.add(found_tc.id)

    # Add results for tests not found in output (likely skipped)
    for tc in tests:
        if tc.id not in parsed_ids:
            results.append(
                PlaywrightTestResult(
                    test_id=tc.id,
                    name=tc.name,
                    status="SKIP" if not tc.script else "ERROR",
                    error="" if not tc.script else "Test not found in Playwright output",
                )
            )

    return results


def execute_playwright_tests(
    tests: list[TestCase],
    epic_id: EpicId,
    project_path: Path,
    timeout: int = 300,
    headless: bool = True,
    config: PlaywrightConfig | None = None,
) -> PlaywrightExecutionResult:
    """Execute Playwright tests for Category B.

    Main entry point for Playwright execution:
    1. Manage server lifecycle (start if configured)
    2. Check Playwright is available
    3. Generate spec file from tests
    4. Run Playwright with JSON reporter
    5. Parse results
    6. Stop server if we started it

    Args:
        tests: Category B test cases.
        epic_id: Epic identifier.
        project_path: Project root directory.
        timeout: Timeout for test execution.
        headless: Run in headless mode.
        config: Playwright configuration (for server management).

    Returns:
        PlaywrightExecutionResult with test outcomes.

    """
    start_time = datetime.now(UTC)

    # Use config values if provided
    if config is not None:
        headless = config.headless
        timeout = config.timeout

    # Manage server lifecycle
    with managed_server(config, project_path) as server_ok:
        if not server_ok:
            base_url = config.base_url if config else "unknown"
            return PlaywrightExecutionResult(
                success=False,
                results=[
                    PlaywrightTestResult(
                        test_id=tc.id,
                        name=tc.name,
                        status="SKIP",
                        error=f"Server not available at {base_url}",
                    )
                    for tc in tests
                ],
                error=f"Server not available at {base_url}",
            )

        # Check Playwright availability
        available, version_or_error = check_playwright_available(project_path)
        if not available:
            logger.error("Playwright not available: %s", version_or_error)
            return PlaywrightExecutionResult(
                success=False,
                results=[
                    PlaywrightTestResult(
                        test_id=tc.id,
                        name=tc.name,
                        status="SKIP",
                        error=f"Playwright not available: {version_or_error}",
                    )
                    for tc in tests
                ],
                error=f"Playwright not available: {version_or_error}",
            )

        logger.info("Playwright version: %s", version_or_error)

        # Filter tests that have scripts
        tests_with_scripts = [tc for tc in tests if tc.script]

        if not tests_with_scripts:
            logger.warning("No Category B tests have scripts - all will be skipped")
            return PlaywrightExecutionResult(
                success=True,
                results=[
                    PlaywrightTestResult(
                        test_id=tc.id,
                        name=tc.name,
                        status="SKIP",
                        error="No TypeScript script provided",
                    )
                    for tc in tests
                ],
            )

        # Generate spec file and config
        spec_dir = project_path / "_bmad-output" / "qa-artifacts" / "playwright"
        base_url = config.base_url if config else "http://localhost:8765"
        try:
            spec_file = generate_spec_file(tests, epic_id, spec_dir, base_url=base_url)
            config_file = generate_playwright_config(spec_dir, config)
            results_file = spec_dir / "results.json"
        except Exception as e:
            logger.error("Failed to generate spec/config files: %s", e)
            return PlaywrightExecutionResult(
                success=False,
                results=[],
                error=f"Failed to generate spec/config files: {e}",
            )

        # Calculate process timeout based on number of tests
        # Per-test timeout is in config (written to playwright.config.ts)
        # Process timeout needs to be generous enough for all tests
        num_tests = len(tests_with_scripts)
        # Formula: per_test_timeout * num_tests + 60s buffer, minimum 120s
        process_timeout = max(timeout * num_tests + 60, 120)
        logger.debug(
            "Process timeout: %ds (per-test: %ds × %d tests + 60s buffer)",
            process_timeout,
            timeout,
            num_tests,
        )

        # Run Playwright with progress output
        logger.info("Executing %d Playwright tests...", num_tests)
        exit_code, json_output, stderr = run_playwright_tests(
            spec_file=spec_file,
            project_path=project_path,
            config_file=config_file,
            results_file=results_file,
            process_timeout=process_timeout,
            headless=headless,
        )

        # Parse results
        if exit_code == -1:
            # Execution failed
            logger.error("Playwright execution failed: %s", stderr)
            results = [
                PlaywrightTestResult(
                    test_id=tc.id,
                    name=tc.name,
                    status="ERROR",
                    error=stderr[:200],
                )
                for tc in tests
            ]
        else:
            # Parse JSON output from results file
            results = parse_playwright_json_results(json_output, stderr, tests)

    end_time = datetime.now(UTC)
    total_duration = int((end_time - start_time).total_seconds() * 1000)

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errored = sum(1 for r in results if r.status == "ERROR")
    skipped = sum(1 for r in results if r.status == "SKIP")

    # Print summary
    print()
    print(f"{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {errored} error, {skipped} skipped")
    print(f"Duration: {total_duration / 1000:.1f}s")
    print(f"{'=' * 60}")

    # Print failed/errored tests with short error messages
    failed_tests = [r for r in results if r.status in ("FAIL", "ERROR")]
    if failed_tests:
        print("\nFailed tests:")
        for r in failed_tests:
            # Extract first line of error (usually the most useful)
            error_line = r.error.split("\n")[0][:100] if r.error else "Unknown error"
            print(f"  {r.test_id}: {r.name}")
            print(f"    -> {error_line}")
        print()

    logger.info(
        "Playwright execution complete: %d passed, %d failed, %d total",
        passed,
        failed,
        len(results),
    )

    return PlaywrightExecutionResult(
        success=exit_code != -1,
        results=results,
        spec_file=spec_file,
        total_duration_ms=total_duration,
        error=stderr if exit_code != 0 else "",
    )
