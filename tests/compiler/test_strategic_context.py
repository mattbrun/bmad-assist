"""Tests for strategic context service and load_antipatterns helper."""

from unittest.mock import MagicMock, patch

import pytest

from bmad_assist.compiler.strategic_context import (
    TRUNCATION_NOTICE,
    _truncate_content,
    load_antipatterns,
)
from bmad_assist.compiler.types import CompilerContext


class TestTruncateContent:
    """Tests for _truncate_content() helper function."""

    def test_short_content_not_truncated(self):
        """Content shorter than budget is returned unchanged."""
        content = "Short content"
        result, tokens = _truncate_content(content, 1000)
        assert result == content
        assert TRUNCATION_NOTICE not in result

    def test_truncation_adds_notice(self):
        """Truncated content includes truncation notice."""
        # Create content that exceeds budget (1000 tokens = ~4000 chars)
        content = "## Section 1\n\nParagraph 1.\n\n" * 200  # ~6000 chars
        result, tokens = _truncate_content(content, 500)
        assert TRUNCATION_NOTICE in result
        assert len(result) < len(content)

    def test_truncation_at_markdown_header(self):
        """Truncation prefers markdown header boundaries."""
        # Create longer content (~2000 chars = 500 tokens)
        section = "## Section {n}\n\n" + "Content for section with more text. " * 10 + "\n\n"
        content = "".join(section.format(n=i) for i in range(1, 6))

        # Budget for ~40% of content (200 tokens)
        result, _ = _truncate_content(content, 200)
        assert TRUNCATION_NOTICE in result
        # Should cut before a ## header, not mid-sentence
        assert result.count("## Section") < content.count("## Section")

    def test_truncation_at_blank_line(self):
        """Truncation falls back to blank line boundaries."""
        # Content without markdown headers, just paragraphs (~1600 chars = 400 tokens)
        content = "\n\n".join(f"Paragraph {i} with some content here. " * 10 for i in range(1, 5))

        # Budget for ~half (200 tokens)
        result, _ = _truncate_content(content, 200)
        assert TRUNCATION_NOTICE in result
        # Should preserve complete paragraphs (end with content, not mid-word)
        text_before_notice = result.replace(TRUNCATION_NOTICE, "").rstrip()
        assert text_before_notice[-1] in ".!? \n"  # Ends cleanly

    def test_budget_overrun_allowed(self):
        """Slightly exceeding budget is allowed for better cut points."""
        # Content with header at ~4400 chars (1100 tokens)
        content = "A" * 4000 + "\n\n## Header\n\n" + "B" * 1000
        # Budget 1000 tokens = 4000 chars, but header is at 4400
        result, tokens = _truncate_content(content, 1000)
        # Should include content slightly over budget to reach header
        assert tokens <= 1100 * 1.2  # Allow 20% margin for truncation notice


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
