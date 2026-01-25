"""Fixture discovery system for experiment framework.

This module provides fixture management for experiment runs,
enabling discovery and access to test fixtures (projects/scenarios)
used as controlled inputs for experiments.

Fixtures are discovered by scanning directories in experiments/fixtures/.
Tar files are ignored - only unpacked directories count as fixtures.
Optional metadata can be stored in .bmad-assist.yaml within each fixture.

Usage:
    from bmad_assist.experiments import FixtureEntry, FixtureManager

    # Discover fixtures from directory
    manager = FixtureManager(Path("experiments/fixtures"))
    available = manager.list()
    fixture = manager.get("auth-service")
    fixture_path = manager.get_path("auth-service")

    # Filter fixtures (if metadata available)
    quick_fixtures = manager.filter_by_tags(["quick"])

"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from bmad_assist.core.config import MAX_CONFIG_SIZE
from bmad_assist.core.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Valid fixture ID pattern (directory name)
NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")

# Valid cost format: $X.XX (e.g., "$0.10", "$1.00", "$99.99")
COST_PATTERN = re.compile(r"^\$\d+\.\d{2}$")


def parse_cost(cost_str: str) -> float:
    """Parse cost string like '$0.10' into float 0.10.

    Args:
        cost_str: Cost string in format '$X.XX'.

    Returns:
        Float value representing the cost.

    Raises:
        ValueError: If cost_str doesn't match COST_PATTERN format.

    """
    if not COST_PATTERN.match(cost_str):
        raise ValueError(f"Invalid cost format '{cost_str}': must match '$X.XX' pattern")
    return float(cost_str[1:])  # Strip $ and convert


class FixtureEntry(BaseModel):
    """Single fixture entry discovered from directory.

    Attributes:
        id: Unique identifier (directory name).
        name: Human-readable name (from metadata or same as id).
        description: Optional description from metadata.
        path: Absolute path to fixture directory.
        tags: Optional categorization tags from metadata.
        difficulty: Optional complexity indicator from metadata.
        estimated_cost: Optional estimated API cost from metadata.

    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        ...,
        description="Unique identifier (directory name)",
    )
    name: str = Field(
        ...,
        description="Human-readable name",
    )
    description: str | None = Field(
        default=None,
        description="Optional description",
    )
    path: Path = Field(
        ...,
        description="Absolute path to fixture directory",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional categorization tags",
    )
    difficulty: Literal["easy", "medium", "hard"] | None = Field(
        default=None,
        description="Optional complexity indicator",
    )
    estimated_cost: str | None = Field(
        default=None,
        description="Optional estimated API cost (e.g., '$0.10')",
    )

    @field_validator("id", mode="after")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate fixture ID format."""
        v = v.strip()
        if not v:
            raise ValueError("id cannot be empty")
        if not NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid id '{v}': must start with letter/underscore, "
                "contain only alphanumeric, hyphens, underscores"
            )
        return v

    @field_validator("estimated_cost", mode="after")
    @classmethod
    def validate_cost(cls, v: str | None) -> str | None:
        """Validate cost format if provided."""
        if v is not None and not COST_PATTERN.match(v):
            raise ValueError(f"Invalid estimated_cost '{v}': must match format '$X.XX'")
        return v


class FixtureMetadata(BaseModel):
    """Optional metadata from .bmad-assist.yaml fixture section."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    estimated_cost: str | None = None


def _load_fixture_metadata(fixture_dir: Path) -> FixtureMetadata | None:
    """Load optional metadata from .bmad-assist.yaml or bmad-assist.yaml.

    Looks for 'fixture' section in the config file.

    Args:
        fixture_dir: Path to fixture directory.

    Returns:
        FixtureMetadata if found and valid, None otherwise.

    """
    # Try .bmad-assist.yaml first, then bmad-assist.yaml
    for config_name in [".bmad-assist.yaml", "bmad-assist.yaml"]:
        config_path = fixture_dir / config_name
        if not config_path.exists():
            continue

        try:
            with config_path.open("r", encoding="utf-8") as f:
                content = f.read(MAX_CONFIG_SIZE + 1)

            if len(content) > MAX_CONFIG_SIZE:
                logger.warning("Config %s exceeds size limit, skipping", config_path)
                return None

            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                return None

            # Look for 'fixture' section
            fixture_data = data.get("fixture")
            if fixture_data and isinstance(fixture_data, dict):
                return FixtureMetadata.model_validate(fixture_data)

        except (yaml.YAMLError, OSError) as e:
            logger.debug("Could not load metadata from %s: %s", config_path, e)

    return None


def discover_fixtures(fixtures_dir: Path) -> list[FixtureEntry]:
    """Discover all fixtures in a directory.

    Scans for subdirectories, ignoring files (like .tar archives).
    Optionally loads metadata from .bmad-assist.yaml in each fixture.

    Args:
        fixtures_dir: Path to fixtures directory (e.g., experiments/fixtures/).

    Returns:
        List of discovered FixtureEntry objects, sorted by id.

    Raises:
        ConfigError: If fixtures_dir doesn't exist or isn't a directory.

    """
    if not fixtures_dir.exists():
        raise ConfigError(f"Fixtures directory not found: {fixtures_dir}")

    if not fixtures_dir.is_dir():
        raise ConfigError(f"Fixtures path is not a directory: {fixtures_dir}")

    fixtures: list[FixtureEntry] = []

    for item in fixtures_dir.iterdir():
        # Skip files (including .tar archives)
        if not item.is_dir():
            continue

        # Skip hidden directories
        if item.name.startswith("."):
            continue

        # Validate directory name as fixture ID
        if not NAME_PATTERN.match(item.name):
            logger.warning(
                "Skipping directory '%s': invalid fixture ID format",
                item.name,
            )
            continue

        # Load optional metadata
        metadata = _load_fixture_metadata(item)

        # Create fixture entry
        entry = FixtureEntry(
            id=item.name,
            name=metadata.name if metadata and metadata.name else item.name,
            description=metadata.description if metadata else None,
            path=item.resolve(),
            tags=metadata.tags if metadata else [],
            difficulty=metadata.difficulty if metadata else None,
            estimated_cost=metadata.estimated_cost if metadata else None,
        )
        fixtures.append(entry)

    # Sort by ID for consistent ordering
    return sorted(fixtures, key=lambda f: f.id)


class FixtureManager:
    """Manager for fixture discovery and access.

    Discovers fixtures by scanning directories (not from registry.yaml).
    Tar files are ignored - only unpacked directories are fixtures.

    Usage:
        manager = FixtureManager(Path("experiments/fixtures"))
        fixtures = manager.filter_by_tags(["quick"])
        fixture_path = manager.get_path("auth-service")

    """

    def __init__(self, fixtures_dir: Path) -> None:
        """Initialize the manager.

        Args:
            fixtures_dir: Path to fixtures directory.

        """
        self._fixtures_dir = fixtures_dir
        self._fixtures: list[FixtureEntry] | None = None
        self._index: dict[str, FixtureEntry] | None = None

    def discover(self) -> list[FixtureEntry]:
        """Discover and cache fixtures from directory.

        Returns:
            List of discovered FixtureEntry objects.

        """
        if self._fixtures is None:
            self._fixtures = discover_fixtures(self._fixtures_dir)
            self._index = {f.id: f for f in self._fixtures}
        return self._fixtures

    def refresh(self) -> list[FixtureEntry]:
        """Clear cache and rediscover fixtures.

        Returns:
            Freshly discovered fixtures.

        """
        self._fixtures = None
        self._index = None
        return self.discover()

    def get(self, fixture_id: str) -> FixtureEntry:
        """Get a fixture entry by ID.

        Args:
            fixture_id: Unique fixture identifier (directory name).

        Returns:
            The FixtureEntry matching the ID.

        Raises:
            ConfigError: If fixture ID not found.

        """
        self.discover()
        if self._index and fixture_id in self._index:
            return self._index[fixture_id]
        available = ", ".join(sorted(self._index.keys())) if self._index else "none"
        raise ConfigError(f"Fixture '{fixture_id}' not found. Available: {available}")

    def filter_by_tags(self, tags: list[str]) -> list[FixtureEntry]:
        """Filter fixtures by tags (AND logic).

        Args:
            tags: List of tags that must all be present.
                  Empty list returns all fixtures.

        Returns:
            List of fixtures matching all tags.

        """
        fixtures = self.discover()
        if not tags:
            return fixtures
        tag_set: set[str] = set(tags)
        return [f for f in fixtures if tag_set.issubset(set(f.tags))]

    def filter_by_difficulty(
        self, difficulty: Literal["easy", "medium", "hard"]
    ) -> list[FixtureEntry]:
        """Filter fixtures by difficulty level.

        Args:
            difficulty: Difficulty level to filter by.

        Returns:
            List of fixtures matching the difficulty.

        """
        fixtures = self.discover()
        return [f for f in fixtures if f.difficulty == difficulty]

    def get_path(self, fixture_id: str) -> Path:
        """Get absolute path for a fixture.

        Args:
            fixture_id: Unique fixture identifier.

        Returns:
            Absolute path to the fixture directory.

        """
        entry = self.get(fixture_id)
        return entry.path

    def list(self) -> list[str]:
        """List all fixture IDs.

        Returns:
            Sorted list of fixture IDs.

        """
        fixtures = self.discover()
        return [f.id for f in fixtures]

    def has_fixtures(self) -> bool:
        """Check if any fixtures are available.

        Returns:
            True if at least one fixture directory exists.

        """
        fixtures = self.discover()
        return len(fixtures) > 0


# =============================================================================
# Backwards compatibility aliases (deprecated)
# =============================================================================

# Keep old names for any existing code that imports them
FixtureRegistry = None  # No longer used
FixtureRegistryManager = FixtureManager  # Alias for compatibility


def load_fixture_registry(path: Path, validate_paths: bool = True) -> list[FixtureEntry]:
    """Load fixtures from registry (deprecated, use FixtureManager instead)."""
    logger.warning("load_fixture_registry() is deprecated. Use FixtureManager instead.")
    fixtures_dir = path.parent
    return discover_fixtures(fixtures_dir)
