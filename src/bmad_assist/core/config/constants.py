"""Shared constants for configuration modules.

This module provides constants used across config submodules to avoid
duplication and circular import issues.
"""

from pathlib import Path

# Maximum parent directory levels to search for loop config
MAX_LOOP_CONFIG_PARENT_DEPTH: int = 10

# Constants for global configuration
GLOBAL_CONFIG_PATH: Path = Path.home() / ".bmad-assist" / "config.yaml"
PROJECT_CONFIG_NAME: str = "bmad-assist.yaml"
MAX_CONFIG_SIZE: int = 1_048_576  # 1MB - protection against YAML bombs
