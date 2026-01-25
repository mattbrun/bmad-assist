"""Tests for strategic context service and load_antipatterns helper."""

from unittest.mock import MagicMock, patch

import pytest

from bmad_assist.compiler.strategic_context import load_antipatterns
from bmad_assist.compiler.types import CompilerContext


class TestLoadAntipatterns:
    """Tests for load_antipatterns() helper function."""

    @pytest.fixture
    def mock_context(self, tmp_path):
        """Create a mock compiler context."""
        context = MagicMock(spec=CompilerContext)
        context.project_root = tmp_path
        context.resolved_variables = {"epic_num": 24}
        return context

    @pytest.fixture
    def impl_artifacts(self, tmp_path):
        """Create implementation artifacts directory."""
        artifacts = tmp_path / "_bmad-output" / "implementation-artifacts"
        artifacts.mkdir(parents=True)
        return artifacts

    def test_load_existing_file_new_path(self, mock_context, impl_artifacts):
        """Test loading antipatterns file from new path (antipatterns/)."""
        # Create file in new location
        antipatterns_dir = impl_artifacts / "antipatterns"
        antipatterns_dir.mkdir()
        antipatterns_file = antipatterns_dir / "epic-24-code-antipatterns.md"
        antipatterns_file.write_text("# Code Antipatterns\nSome content here")

        with (
            patch("bmad_assist.core.config.get_config") as mock_config,
            patch("bmad_assist.core.paths.get_paths") as mock_paths,
        ):
            mock_config.return_value.antipatterns.enabled = True
            mock_paths.return_value.implementation_artifacts = impl_artifacts

            result = load_antipatterns(mock_context, "code")

        assert "[ANTIPATTERNS - DO NOT REPEAT]" in result
        assert "Code Antipatterns" in result["[ANTIPATTERNS - DO NOT REPEAT]"]

    def test_load_legacy_path_fallback(self, mock_context, impl_artifacts):
        """Test fallback to legacy path when new path doesn't exist."""
        # Create file in legacy location (no antipatterns/ subdir)
        legacy_file = impl_artifacts / "epic-24-code-antipatterns.md"
        legacy_file.write_text("# Legacy Antipatterns\nOld format content")

        with (
            patch("bmad_assist.core.config.get_config") as mock_config,
            patch("bmad_assist.core.paths.get_paths") as mock_paths,
        ):
            mock_config.return_value.antipatterns.enabled = True
            mock_paths.return_value.implementation_artifacts = impl_artifacts

            result = load_antipatterns(mock_context, "code")

        assert "[ANTIPATTERNS - DO NOT REPEAT]" in result
        assert "Legacy Antipatterns" in result["[ANTIPATTERNS - DO NOT REPEAT]"]

    def test_load_prefers_new_path(self, mock_context, impl_artifacts):
        """Test that new path is preferred when both exist."""
        # Create files in both locations
        antipatterns_dir = impl_artifacts / "antipatterns"
        antipatterns_dir.mkdir()

        new_file = antipatterns_dir / "epic-24-code-antipatterns.md"
        new_file.write_text("# New Path Content")

        legacy_file = impl_artifacts / "epic-24-code-antipatterns.md"
        legacy_file.write_text("# Legacy Path Content")

        with (
            patch("bmad_assist.core.config.get_config") as mock_config,
            patch("bmad_assist.core.paths.get_paths") as mock_paths,
        ):
            mock_config.return_value.antipatterns.enabled = True
            mock_paths.return_value.implementation_artifacts = impl_artifacts

            result = load_antipatterns(mock_context, "code")

        assert "New Path Content" in result["[ANTIPATTERNS - DO NOT REPEAT]"]
        assert "Legacy Path Content" not in result["[ANTIPATTERNS - DO NOT REPEAT]"]

    def test_load_missing_file_returns_empty(self, mock_context, impl_artifacts):
        """Test that missing file returns empty dict."""
        with (
            patch("bmad_assist.core.config.get_config") as mock_config,
            patch("bmad_assist.core.paths.get_paths") as mock_paths,
        ):
            mock_config.return_value.antipatterns.enabled = True
            mock_paths.return_value.implementation_artifacts = impl_artifacts

            result = load_antipatterns(mock_context, "code")

        assert result == {}

    def test_load_disabled_returns_empty(self, mock_context, impl_artifacts):
        """Test that disabled config returns empty dict."""
        # Create file
        antipatterns_dir = impl_artifacts / "antipatterns"
        antipatterns_dir.mkdir()
        antipatterns_file = antipatterns_dir / "epic-24-code-antipatterns.md"
        antipatterns_file.write_text("# Should not load")

        with patch("bmad_assist.core.config.get_config") as mock_config:
            mock_config.return_value.antipatterns.enabled = False

            result = load_antipatterns(mock_context, "code")

        assert result == {}

    def test_load_missing_epic_id_returns_empty(self, tmp_path):
        """Test that missing epic_num returns empty dict."""
        context = MagicMock(spec=CompilerContext)
        context.project_root = tmp_path
        context.resolved_variables = {}  # No epic_num

        with patch("bmad_assist.core.config.get_config") as mock_config:
            mock_config.return_value.antipatterns.enabled = True

            result = load_antipatterns(context, "code")

        assert result == {}

    def test_load_story_antipatterns(self, mock_context, impl_artifacts):
        """Test loading story-type antipatterns."""
        mock_context.resolved_variables = {"epic_num": 24}

        antipatterns_dir = impl_artifacts / "antipatterns"
        antipatterns_dir.mkdir()
        antipatterns_file = antipatterns_dir / "epic-24-story-antipatterns.md"
        antipatterns_file.write_text("# Story Antipatterns\nValidation issues")

        with (
            patch("bmad_assist.core.config.get_config") as mock_config,
            patch("bmad_assist.core.paths.get_paths") as mock_paths,
        ):
            mock_config.return_value.antipatterns.enabled = True
            mock_paths.return_value.implementation_artifacts = impl_artifacts

            result = load_antipatterns(mock_context, "story")

        assert "Story Antipatterns" in result["[ANTIPATTERNS - DO NOT REPEAT]"]

    def test_load_string_epic_id(self, tmp_path, impl_artifacts):
        """Test loading with string epic ID (e.g., 'testarch')."""
        context = MagicMock(spec=CompilerContext)
        context.project_root = tmp_path
        context.resolved_variables = {"epic_num": "testarch"}

        antipatterns_dir = impl_artifacts / "antipatterns"
        antipatterns_dir.mkdir()
        antipatterns_file = antipatterns_dir / "epic-testarch-code-antipatterns.md"
        antipatterns_file.write_text("# Testarch Antipatterns")

        with (
            patch("bmad_assist.core.config.get_config") as mock_config,
            patch("bmad_assist.core.paths.get_paths") as mock_paths,
        ):
            mock_config.return_value.antipatterns.enabled = True
            mock_paths.return_value.implementation_artifacts = impl_artifacts

            result = load_antipatterns(context, "code")

        assert "Testarch Antipatterns" in result["[ANTIPATTERNS - DO NOT REPEAT]"]
