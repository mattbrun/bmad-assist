"""Tests for post-compilation prompt budget trimming (Fix #3 and #6)."""

import logging

from bmad_assist.core.loop.handlers.base import _trim_source_context


def _make_prompt(file_count: int, content_size: int = 1000) -> str:
    """Build a fake compiled prompt XML with file blocks.

    Files are named file_1.py through file_N.py (highest priority first).
    """
    files = []
    for i in range(1, file_count + 1):
        content = "x" * content_size
        files.append(
            f'<file id="f{i}" path="src/file_{i}.py" label="file_{i}">'
            f"<![CDATA[{content}]]></file>"
        )
    context = "<context>\n" + "\n".join(files) + "\n</context>"
    return (
        f"<compiled-workflow>\n<mission>Do something</mission>\n"
        f"{context}\n</compiled-workflow>"
    )


class TestTrimSourceContext:
    """Test _trim_source_context budget enforcement."""

    def test_no_trimming_when_within_budget(self) -> None:
        """Prompt within budget returned unchanged."""
        prompt = _make_prompt(3, content_size=100)
        current_tokens = len(prompt) // 4
        budget = current_tokens + 1000  # Well within budget
        result = _trim_source_context(prompt, current_tokens, budget)
        assert result == prompt

    def test_trims_last_file_when_over_budget(self) -> None:
        """Slightly over budget removes last (lowest priority) file."""
        prompt = _make_prompt(3, content_size=2000)
        current_tokens = len(prompt) // 4
        # Set budget to be less than current but achievable by removing 1 file
        budget = current_tokens - 400  # ~2000/4 = 500 tokens per file
        result = _trim_source_context(prompt, current_tokens, budget)
        assert "file_3.py" not in result
        assert "file_1.py" in result
        assert "file_2.py" in result

    def test_trims_multiple_files_when_significantly_over(self) -> None:
        """Significantly over budget removes multiple lowest-priority files."""
        prompt = _make_prompt(5, content_size=2000)
        current_tokens = len(prompt) // 4
        # Budget achievable only by removing 3+ files
        budget = current_tokens - 1200
        result = _trim_source_context(prompt, current_tokens, budget)
        # file_1 should always survive (highest priority)
        assert "file_1.py" in result
        # At least one file was removed
        assert len(result) < len(prompt)

    def test_preserves_non_source_content(self) -> None:
        """Strategic context and mission are never trimmed."""
        prompt = _make_prompt(3, content_size=2000)
        current_tokens = len(prompt) // 4
        budget = current_tokens - 800
        result = _trim_source_context(prompt, current_tokens, budget)
        assert "<mission>Do something</mission>" in result
        assert "<compiled-workflow>" in result

    def test_no_crash_on_missing_context(self) -> None:
        """Prompt without any <file> blocks returns unchanged."""
        prompt = "<compiled-workflow><mission>test</mission></compiled-workflow>"
        result = _trim_source_context(prompt, 100, 50)
        assert result == prompt

    def test_logs_removed_files(self, caplog) -> None:
        """Each removed file is logged."""
        prompt = _make_prompt(3, content_size=2000)
        current_tokens = len(prompt) // 4
        budget = current_tokens - 800
        with caplog.at_level(logging.INFO):
            _trim_source_context(prompt, current_tokens, budget)
        assert "Budget trim: removed source file" in caplog.text
        assert "Trimmed" in caplog.text

    def test_trim_result_estimates_lower_tokens(self) -> None:
        """After trimming, len(result)//4 should be closer to budget."""
        prompt = _make_prompt(4, content_size=4000)
        current_tokens = len(prompt) // 4
        budget = current_tokens // 2
        result = _trim_source_context(prompt, current_tokens, budget)
        new_tokens = len(result) // 4
        assert new_tokens < current_tokens
