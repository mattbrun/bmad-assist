"""Environment variable and credential handling for bmad-assist."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Known credential environment variable names (AC7)
# Note: API keys (ANTHROPIC, OPENAI, GEMINI) are NOT used by bmad-assist.
# bmad-assist orchestrates CLI tools which handle their own authentication.
# These are notification credentials used by the notification system.
ENV_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    {
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "DISCORD_WEBHOOK_URL",
    }
)

# .env file name constant
ENV_FILE_NAME: str = ".env"


def _mask_credential(value: str | None) -> str:
    """Mask credential value for safe logging.

    Args:
        value: Credential value to mask. None values are handled gracefully.

    Returns:
        Masked value showing only first 7 characters + "***",
        or "***" if value is None, empty, or 7 characters or shorter.

    """
    if not value:
        return "***"
    if len(value) <= 7:
        return "***"
    return value[:7] + "***"


def _check_env_file_permissions(path: Path) -> None:
    """Check if .env file has secure permissions (600 or 400 on Unix).

    Only checks on Unix-like systems (Linux, macOS).
    Logs warning if permissions are too permissive.
    Accepts both 0600 (owner read-write) and 0400 (owner read-only).

    Args:
        path: Path to .env file.

    """
    if sys.platform == "win32":
        return  # Windows has different permission model

    try:
        mode = path.stat().st_mode & 0o777
        # Accept 0600 (rw owner) or 0400 (r owner) - both secure
        if mode not in (0o600, 0o400):
            logger.warning(
                ".env file %s has insecure permissions %03o, "
                "expected 600 or 400. Run: chmod 600 %s",
                path,
                mode,
                path,
            )
    except OSError:
        pass  # File may have been deleted between check and stat


def load_env_file(
    project_path: str | Path | None = None,
    *,
    check_permissions: bool = True,
) -> bool:
    """Load environment variables from .env file.

    Loads environment variables from {project_path}/.env or {cwd}/.env.
    Does NOT override existing environment variables (override=False).

    Args:
        project_path: Path to project directory. Defaults to current working directory.
        check_permissions: Whether to check file permissions (default True).

    Returns:
        True if .env file was found and loaded, False otherwise.

    Note:
        - Missing .env file is not an error (returns False)
        - On Unix, warns if permissions are not 600
        - On Windows, permission check is skipped

    """
    # Resolve project path
    resolved_path = Path.cwd() if project_path is None else Path(project_path).expanduser()

    # Build .env file path
    env_file = resolved_path / ENV_FILE_NAME

    # Check if .env file exists
    if not env_file.exists():
        logger.debug(".env file not found at %s, skipping", env_file)
        return False

    if not env_file.is_file():
        logger.debug(".env path %s is not a file, skipping", env_file)
        return False

    # Check permissions before loading
    if check_permissions:
        _check_env_file_permissions(env_file)

    # Load environment variables - CRITICAL: override=False preserves existing env vars
    load_dotenv(env_file, encoding="utf-8", override=False)
    logger.debug("Loaded environment variables from %s", env_file)

    return True
