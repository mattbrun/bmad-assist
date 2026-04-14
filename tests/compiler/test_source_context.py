"""Tests for the source_context module.

Tests the SourceContextService class and utility functions for
configurable source file collection in workflow compilers.
"""

from pathlib import Path

import pytest

from bmad_assist.compiler.source_context import (
    GitDiffFile,
    ScoredFile,
    SourceContextService,
    _extract_file_list_section,
    extract_file_paths_from_section,
    extract_file_paths_from_story,
    get_git_diff_files,
    is_binary_file,
    safe_read_file,
)
from bmad_assist.compiler.types import CompilerContext
from bmad_assist.core.config import (
    SourceContextBudgetsConfig,
    SourceContextConfig,
    SourceContextExtractionConfig,
    SourceContextScoringConfig,
)


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project structure for testing."""
    # Create directory structure
    docs = tmp_path / "docs"
    docs.mkdir()

    sprint_artifacts = docs / "sprint-artifacts"
    sprint_artifacts.mkdir()

    src = tmp_path / "src"
    src.mkdir()

    # Create project_context.md
    (docs / "project-context.md").write_text("# Project Context\n")

    return tmp_path


def create_test_context(project: Path) -> CompilerContext:
    """Create a test CompilerContext."""
    return CompilerContext(
        project_root=project,
        output_folder=project / "docs",
        resolved_variables={},
    )


class TestSourceContextBudgetsConfig:
    """Tests for SourceContextBudgetsConfig."""

    def test_default_budgets(self) -> None:
        """Default budgets match tech-spec."""
        config = SourceContextBudgetsConfig()

        assert config.code_review == 15000
        assert config.code_review_synthesis == 5000
        assert config.create_story == 20000
        assert config.dev_story == 20000
        assert config.validate_story == 10000
        assert config.validate_story_synthesis == 10000
        assert config.default == 20000

    def test_get_budget_by_name(self) -> None:
        """get_budget returns correct budget by name."""
        config = SourceContextBudgetsConfig()

        assert config.get_budget("code_review") == 15000
        assert config.get_budget("dev_story") == 20000
        assert config.get_budget("unknown_workflow") == 20000  # Falls back to default

    def test_get_budget_normalizes_hyphens(self) -> None:
        """get_budget handles hyphenated names."""
        config = SourceContextBudgetsConfig()

        assert config.get_budget("code-review") == 15000
        assert config.get_budget("dev-story") == 20000


class TestSourceContextScoringConfig:
    """Tests for SourceContextScoringConfig."""

    def test_default_scoring_weights(self) -> None:
        """Default scoring weights match tech-spec."""
        config = SourceContextScoringConfig()

        assert config.in_file_list == 50
        assert config.in_git_diff == 50
        assert config.is_test_file == -10
        assert config.is_config_file == -5
        assert config.change_lines_factor == 1
        assert config.change_lines_cap == 50


class TestSourceContextExtractionConfig:
    """Tests for SourceContextExtractionConfig."""

    def test_default_extraction_settings(self) -> None:
        """Default extraction settings match tech-spec."""
        config = SourceContextExtractionConfig()

        assert config.adaptive_threshold == 0.25
        assert config.hunk_context_lines == 20
        assert config.hunk_context_scale == 0.3
        assert config.max_files == 15


class TestSourceContextConfig:
    """Tests for SourceContextConfig nested configuration."""

    def test_nested_defaults(self) -> None:
        """Nested configs have correct defaults."""
        config = SourceContextConfig()

        assert config.budgets.code_review == 15000
        assert config.scoring.in_file_list == 50
        assert config.extraction.max_files == 15


class TestExtractFilePathsFromStory:
    """Tests for extract_file_paths_from_story function."""

    def test_extracts_basic_paths(self) -> None:
        """Extracts file paths from standard File List section."""
        story = """# Story 1.1

## File List

- `src/module.py` - Main module
- `tests/test_module.py` - Tests

## Other Section
"""
        paths = extract_file_paths_from_story(story)

        assert len(paths) == 2
        assert "src/module.py" in paths
        assert "tests/test_module.py" in paths

    def test_handles_h3_header(self) -> None:
        """Handles ### File List header."""
        story = """### File List

- `src/file.py`
"""
        paths = extract_file_paths_from_story(story)

        assert len(paths) == 1
        assert "src/file.py" in paths

    def test_handles_paths_without_backticks(self) -> None:
        """Extracts paths without backticks."""
        story = """## File List

- src/plain/path.ts
* tests/another.py
"""
        paths = extract_file_paths_from_story(story)

        assert "src/plain/path.ts" in paths
        assert "tests/another.py" in paths

    def test_numbered_list_format(self) -> None:
        """Numbered lists (1. `file`) are extracted."""
        story = """## File List

1. `src/module.py` - Main module
2. `src/utils.py` - Utility functions

## Other
"""
        paths = extract_file_paths_from_story(story)

        assert len(paths) == 2
        assert "src/module.py" in paths
        assert "src/utils.py" in paths

    def test_table_format(self) -> None:
        """Markdown table entries (| `file` |) are extracted."""
        story = """## File List

| File | Description |
|------|-------------|
| `src/main.py` | Entry point |
| `src/config.py` | Configuration |

## Other
"""
        paths = extract_file_paths_from_story(story)

        assert len(paths) == 2
        assert "src/main.py" in paths
        assert "src/config.py" in paths

    def test_subheaders_within_file_list(self) -> None:
        """Sub-headers (### Modified Files under ## File List) don't terminate section."""
        story = """## File List

### Modified Files
- `src/module.py`

### New Files
- `src/new_module.py`

## Other Section
"""
        paths = extract_file_paths_from_story(story)

        assert len(paths) == 2
        assert "src/module.py" in paths
        assert "src/new_module.py" in paths

    def test_h4_header(self) -> None:
        """#### File List is matched."""
        story = """#### File List

- `src/deep.py`

#### Other
"""
        paths = extract_file_paths_from_story(story)

        assert len(paths) == 1
        assert "src/deep.py" in paths

    def test_mixed_formats(self) -> None:
        """Mix of bullets, numbered, and table entries in same section."""
        story = """## File List

- `src/bullet.py` - From bullet
1. `src/numbered.py` - From numbered
| `src/table.py` | From table |

## Other
"""
        paths = extract_file_paths_from_story(story)

        assert len(paths) == 3
        assert "src/bullet.py" in paths
        assert "src/numbered.py" in paths
        assert "src/table.py" in paths

    def test_returns_empty_for_no_section(self) -> None:
        """Returns empty list when no File List section."""
        story = """# Story

## Implementation
Some code here.
"""
        paths = extract_file_paths_from_story(story)
        assert paths == []


class TestIsBinaryFile:
    """Tests for is_binary_file function."""

    def test_detects_binary_by_extension(self, tmp_path: Path) -> None:
        """Detects binary files by extension."""
        png = tmp_path / "image.png"
        png.write_bytes(b"fake png content")

        assert is_binary_file(png) is True

    def test_detects_binary_by_null_bytes(self, tmp_path: Path) -> None:
        """Detects binary files by null bytes."""
        binary = tmp_path / "data.bin"
        binary.write_bytes(b"some\x00binary\x00data")

        assert is_binary_file(binary) is True

    def test_allows_text_files(self, tmp_path: Path) -> None:
        """Returns False for text files."""
        text = tmp_path / "file.txt"
        text.write_text("hello world")

        assert is_binary_file(text) is False


class TestSourceContextService:
    """Tests for SourceContextService class."""

    def test_initialization_with_defaults(self, tmp_project: Path) -> None:
        """Service initializes with default config when not loaded."""
        context = create_test_context(tmp_project)
        service = SourceContextService(context, "code_review")

        assert service.budget == 15000
        assert service.is_enabled()

    def test_is_enabled_with_low_budget(
        self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_enabled returns False for budget < 100."""
        # Create custom budgets config with a disabled workflow
        custom_budgets = SourceContextBudgetsConfig(
            code_review=15000,
            code_review_synthesis=15000,
            create_story=20000,
            dev_story=20000,
            validate_story=0,  # Explicitly disabled for this test
            validate_story_synthesis=0,
            default=20000,
        )

        # Patch the service to use our custom budget
        context = create_test_context(tmp_project)
        service = SourceContextService(context, "validate_story")
        monkeypatch.setattr(service, "budget", custom_budgets.validate_story)

        assert service.budget == 0
        assert service.is_enabled() is False

    def test_collect_files_basic(self, tmp_project: Path) -> None:
        """Collects files from File List."""
        src = tmp_project / "src"
        (src / "main.py").write_text("def main():\n    pass")

        context = create_test_context(tmp_project)
        service = SourceContextService(context, "dev_story")

        result = service.collect_files(["src/main.py"], None)

        assert len(result) == 1
        assert "def main():" in list(result.values())[0]

    def test_collect_files_skips_missing(self, tmp_project: Path) -> None:
        """Skips non-existent files."""
        context = create_test_context(tmp_project)
        service = SourceContextService(context, "dev_story")

        result = service.collect_files(["nonexistent.py"], None)

        assert len(result) == 0

    def test_collect_files_skips_binary(self, tmp_project: Path) -> None:
        """Skips binary files."""
        src = tmp_project / "src"
        (src / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        context = create_test_context(tmp_project)
        service = SourceContextService(context, "dev_story")

        result = service.collect_files(["src/image.png"], None)

        assert len(result) == 0

    def test_collect_files_uses_intersection(self, tmp_project: Path) -> None:
        """Uses intersection when both File List and git diff provided."""
        src = tmp_project / "src"
        (src / "in_both.py").write_text("# in both")
        (src / "file_list_only.py").write_text("# file list only")
        (src / "git_only.py").write_text("# git only")

        context = create_test_context(tmp_project)
        service = SourceContextService(context, "code_review")

        file_list = ["src/in_both.py", "src/file_list_only.py"]
        git_diff = [
            GitDiffFile(path="src/in_both.py", change_lines=10, hunk_ranges=[(1, 5)]),
            GitDiffFile(path="src/git_only.py", change_lines=5, hunk_ranges=[(1, 3)]),
        ]

        result = service.collect_files(file_list, git_diff)

        # Only "in_both.py" should be included (intersection)
        assert len(result) == 1
        paths = list(result.keys())
        assert "in_both.py" in paths[0]

    def test_disabled_returns_empty(
        self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns empty when budget is disabled."""
        src = tmp_project / "src"
        (src / "main.py").write_text("content")

        context = create_test_context(tmp_project)
        service = SourceContextService(context, "dev_story")  # Use any workflow
        # Disable the service by setting budget to 0
        monkeypatch.setattr(service, "budget", 0)

        result = service.collect_files(["src/main.py"], None)

        assert len(result) == 0


class TestScoring:
    """Tests for file scoring logic."""

    def test_file_list_bonus(self, tmp_project: Path) -> None:
        """Files in File List get bonus score."""
        src = tmp_project / "src"
        (src / "a.py").write_text("# file a")
        (src / "b.py").write_text("# file b")

        context = create_test_context(tmp_project)
        service = SourceContextService(context, "dev_story")

        # b.py is in file_list only
        result = service.collect_files(["src/b.py"], None)

        assert len(result) == 1
        assert "b.py" in list(result.keys())[0]

    def test_test_files_penalty(self, tmp_project: Path) -> None:
        """Test files get negative score adjustment."""
        src = tmp_project / "tests"
        src.mkdir()
        (src / "test_main.py").write_text("# test file")

        context = create_test_context(tmp_project)
        service = SourceContextService(context, "dev_story")

        result = service.collect_files(["tests/test_main.py"], None)

        # Still included if only file available
        assert len(result) == 1


class TestTruncation:
    """Tests for content truncation."""

    def test_truncates_large_files(self, tmp_project: Path) -> None:
        """Large files are truncated to fit budget."""
        src = tmp_project / "src"
        large_content = "x" * 200000  # Much larger than default budget
        (src / "large.py").write_text(large_content)

        context = create_test_context(tmp_project)
        service = SourceContextService(context, "code_review")  # 15000 budget

        result = service.collect_files(["src/large.py"], None)

        assert len(result) == 1
        content = list(result.values())[0]
        assert len(content) < len(large_content)
        assert "truncated" in content.lower()


# =============================================================================
# Synthesis source file capping with compression
# =============================================================================
#
# Tests for cap_synthesis_source_files — replaces the old hard-cap-and-drop
# behavior in validate_story_synthesis / code_review_synthesis with
# compression-aware trimming for markdown and drop-only for source code.


from unittest.mock import patch

from bmad_assist.compiler.source_context import (
    SynthesisCapResult,
    cap_synthesis_source_files,
)


class TestCapSynthesisSourceFiles:
    """cap_synthesis_source_files preserves top-N and compresses markdown tail."""

    def _patch_compress(self, return_content: str):
        """Mock _compress_or_truncate to return a fixed compressed result."""
        return patch(
            "bmad_assist.compiler.strategic_context._compress_or_truncate",
            return_value=(return_content, max(len(return_content) // 4, 1)),
        )

    def test_under_cap_returns_all_unchanged(self, tmp_path: Path) -> None:
        """If input size <= max_files, nothing is capped."""
        files = {
            "src/a.py": "a" * 100,
            "src/b.py": "b" * 100,
        }
        result = cap_synthesis_source_files(
            files, max_files=3, project_root=tmp_path
        )
        assert isinstance(result, SynthesisCapResult)
        assert result.files == files
        assert result.kept_paths == ["src/a.py", "src/b.py"]
        assert result.compressed_paths == []
        assert result.dropped_paths == []

    def test_preserves_top_n_verbatim(self, tmp_path: Path) -> None:
        """First max_files entries appear in result unchanged."""
        files = {
            "top-1.py": "ONE",
            "top-2.py": "TWO",
            "top-3.py": "THREE",
            "overflow.py": "FOUR",
        }
        result = cap_synthesis_source_files(
            files, max_files=3, project_root=tmp_path
        )
        assert result.files["top-1.py"] == "ONE"
        assert result.files["top-2.py"] == "TWO"
        assert result.files["top-3.py"] == "THREE"
        # Source code overflow is dropped (never compressed)
        assert "overflow.py" not in result.files
        assert result.dropped_paths == ["overflow.py"]

    def test_markdown_overflow_compressed(self, tmp_path: Path) -> None:
        """Markdown files in the overflow are compressed in place."""
        # ~1000 tokens of markdown, over the 500-token threshold
        big_md = "x" * 4000
        files = {
            "keep-1.py": "a" * 100,
            "keep-2.py": "b" * 100,
            "keep-3.py": "c" * 100,
            "docs/big.md": big_md,
        }
        with self._patch_compress("compressed doc"):
            result = cap_synthesis_source_files(
                files, max_files=3, project_root=tmp_path
            )

        # big.md kept (compressed), not dropped.
        assert "docs/big.md" in result.files
        assert result.compressed_paths == ["docs/big.md"]
        assert result.dropped_paths == []
        # Content was replaced by the compressed version (with annotation).
        assert "compressed doc" in result.files["docs/big.md"]
        assert "compressed from" in result.files["docs/big.md"]
        # Original content is gone.
        assert big_md not in result.files["docs/big.md"]

    def test_source_code_overflow_always_dropped(self, tmp_path: Path) -> None:
        """Source code in overflow is dropped, not compressed."""
        big_py = "x" * 4000
        files = {
            "keep-1.md": "a" * 100,
            "keep-2.md": "b" * 100,
            "keep-3.md": "c" * 100,
            "src/big.py": big_py,
        }
        with patch(
            "bmad_assist.compiler.strategic_context._compress_or_truncate"
        ) as mock_compress:
            result = cap_synthesis_source_files(
                files, max_files=3, project_root=tmp_path
            )
        # Compression must NEVER fire for .py files
        mock_compress.assert_not_called()
        assert "src/big.py" not in result.files
        assert result.dropped_paths == ["src/big.py"]
        assert result.compressed_paths == []

    def test_tiny_markdown_overflow_dropped(self, tmp_path: Path) -> None:
        """Markdown below the compressible threshold is dropped, not compressed."""
        tiny_md = "tiny"  # << 500 tokens
        files = {
            "keep-1.py": "a" * 100,
            "keep-2.py": "b" * 100,
            "keep-3.py": "c" * 100,
            "docs/tiny.md": tiny_md,
        }
        with patch(
            "bmad_assist.compiler.strategic_context._compress_or_truncate"
        ) as mock_compress:
            result = cap_synthesis_source_files(
                files, max_files=3, project_root=tmp_path
            )
        mock_compress.assert_not_called()
        assert result.dropped_paths == ["docs/tiny.md"]
        assert result.compressed_paths == []

    def test_compression_no_op_falls_back_to_drop(self, tmp_path: Path) -> None:
        """If compression doesn't actually shrink, the file is dropped."""
        big_md = "y" * 4000
        files = {
            "k1.py": "a" * 100,
            "k2.py": "b" * 100,
            "k3.py": "c" * 100,
            "docs/noop.md": big_md,
        }
        # Mock returns same content (no shrink)
        with self._patch_compress(big_md):
            result = cap_synthesis_source_files(
                files, max_files=3, project_root=tmp_path
            )
        assert result.compressed_paths == []
        assert result.dropped_paths == ["docs/noop.md"]

    def test_project_root_none_skips_compression_entirely(self) -> None:
        """When project_root is None, all overflow is dropped."""
        big_md = "z" * 4000
        files = {
            "k1.py": "a" * 100,
            "k2.py": "b" * 100,
            "k3.py": "c" * 100,
            "docs/big.md": big_md,
        }
        with patch(
            "bmad_assist.compiler.strategic_context._compress_or_truncate"
        ) as mock_compress:
            result = cap_synthesis_source_files(
                files, max_files=3, project_root=None
            )
        mock_compress.assert_not_called()
        assert result.dropped_paths == ["docs/big.md"]
        assert result.compressed_paths == []

    def test_mixed_overflow_markdown_and_code(self, tmp_path: Path) -> None:
        """Mixed overflow: markdown compressed, source code dropped."""
        big_md = "m" * 4000
        big_py = "p" * 4000
        files = {
            "k1.py": "a" * 100,
            "k2.py": "b" * 100,
            "k3.py": "c" * 100,
            "docs/info.md": big_md,
            "src/extra.py": big_py,
        }
        with self._patch_compress("summary"):
            result = cap_synthesis_source_files(
                files, max_files=3, project_root=tmp_path
            )
        assert result.compressed_paths == ["docs/info.md"]
        assert result.dropped_paths == ["src/extra.py"]
        assert "docs/info.md" in result.files
        assert "src/extra.py" not in result.files

    def test_mdx_and_markdown_extensions(self, tmp_path: Path) -> None:
        """All three extensions (.md, .markdown, .mdx) are compressed."""
        big = "x" * 4000
        files = {
            "k1.py": "keep",
            "a.md": big,
            "b.markdown": big,
            "c.mdx": big,
        }
        with self._patch_compress("tiny"):
            result = cap_synthesis_source_files(
                files, max_files=1, project_root=tmp_path
            )
        assert set(result.compressed_paths) == {"a.md", "b.markdown", "c.mdx"}
        assert result.dropped_paths == []

    def test_kept_order_matches_input(self, tmp_path: Path) -> None:
        """kept_paths reflects the input dict order (score ranking)."""
        files = {
            "top.py": "a",
            "mid.py": "b",
            "low.py": "c",
            "drop.py": "d",
        }
        result = cap_synthesis_source_files(
            files, max_files=3, project_root=tmp_path
        )
        assert result.kept_paths == ["top.py", "mid.py", "low.py"]

    def test_compression_failure_drops_file(self, tmp_path: Path) -> None:
        """If _compress_or_truncate raises, the file is dropped silently."""
        big_md = "x" * 4000
        files = {
            "k1.py": "a",
            "k2.py": "b",
            "k3.py": "c",
            "docs/broken.md": big_md,
        }
        with patch(
            "bmad_assist.compiler.strategic_context._compress_or_truncate",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = cap_synthesis_source_files(
                files, max_files=3, project_root=tmp_path
            )
        assert result.dropped_paths == ["docs/broken.md"]
        assert result.compressed_paths == []

    def test_compression_logs_info_line(
        self, tmp_path: Path, caplog
    ) -> None:
        """Successful compression emits an info log with before/after tokens."""
        import logging

        big_md = "x" * 4000
        files = {
            "k1.py": "a",
            "k2.py": "b",
            "k3.py": "c",
            "docs/big.md": big_md,
        }
        with self._patch_compress("short"), caplog.at_level(logging.INFO):
            cap_synthesis_source_files(
                files, max_files=3, project_root=tmp_path
            )
        assert "Synthesis compress" in caplog.text
        assert "docs/big.md" in caplog.text

    def test_negative_max_files_raises(self, tmp_path: Path) -> None:
        """max_files must be non-negative."""
        with pytest.raises(ValueError, match="max_files"):
            cap_synthesis_source_files(
                {"a.py": "x"}, max_files=-1, project_root=tmp_path
            )

    def test_zero_max_files_drops_everything(self, tmp_path: Path) -> None:
        """max_files=0 drops all non-markdown and compresses all markdown."""
        files = {
            "a.py": "x" * 100,
            "b.md": "y" * 4000,
        }
        with self._patch_compress("tiny"):
            result = cap_synthesis_source_files(
                files, max_files=0, project_root=tmp_path
            )
        assert result.kept_paths == []
        assert "a.py" in result.dropped_paths
        assert "b.md" in result.compressed_paths
