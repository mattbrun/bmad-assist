"""Tests for BMAD markdown frontmatter parser.

Tests cover all acceptance criteria (AC1-AC8) from Story 2.1.
"""

import datetime
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from bmad_assist.bmad import BmadDocument, parse_bmad_file
from bmad_assist.bmad.parser import parse_epic_file
from bmad_assist.core.exceptions import ParserError


class TestParseValidFrontmatter:
    """AC1: Parse valid frontmatter."""

    def test_parse_standard_frontmatter(self, sample_bmad_file: Path) -> None:
        """Test parsing file with standard YAML frontmatter."""
        doc = parse_bmad_file(sample_bmad_file)

        assert doc.frontmatter["title"] == "PRD Document"
        assert doc.frontmatter["status"] == "complete"
        # YAML parses dates as datetime.date objects
        assert doc.frontmatter["date"] == datetime.date(2025, 12, 8)
        assert "# Content here" in doc.content

    def test_parse_with_string_path(self, sample_bmad_file: Path) -> None:
        """Test parsing with string path input."""
        doc = parse_bmad_file(str(sample_bmad_file))

        assert doc.frontmatter["title"] == "PRD Document"
        assert doc.path == str(sample_bmad_file)

    def test_parse_with_path_object(self, sample_bmad_file: Path) -> None:
        """Test parsing with Path object input."""
        doc = parse_bmad_file(sample_bmad_file)

        assert doc.frontmatter["title"] == "PRD Document"
        assert doc.path == str(sample_bmad_file)

    def test_content_separation(self, sample_bmad_file: Path) -> None:
        """Test that content is properly separated from frontmatter."""
        doc = parse_bmad_file(sample_bmad_file)

        # Content should not contain frontmatter delimiters
        assert doc.content.strip() == "# Content here"
        # Frontmatter should not be in content
        assert "title:" not in doc.content
        assert "status:" not in doc.content


class TestParseNoFrontmatter:
    """AC2: Parse file without frontmatter."""

    def test_file_without_frontmatter(self, file_without_frontmatter: Path) -> None:
        """Test parsing file that has no frontmatter."""
        doc = parse_bmad_file(file_without_frontmatter)

        assert doc.frontmatter == {}
        assert "# Just Content" in doc.content
        assert "Some markdown text." in doc.content

    def test_plain_markdown_content(self, tmp_path: Path) -> None:
        """Test parsing plain markdown without any frontmatter markers."""
        content = "# Simple Heading\n\nJust text.\n"
        path = tmp_path / "plain.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter == {}
        assert "# Simple Heading" in doc.content
        assert "Just text." in doc.content


class TestParseMalformedFrontmatter:
    """AC3: Handle malformed frontmatter."""

    def test_invalid_yaml_raises_parser_error(self, malformed_yaml_file: Path) -> None:
        """Test that invalid YAML frontmatter raises ParserError."""
        with pytest.raises(ParserError) as exc_info:
            parse_bmad_file(malformed_yaml_file)

        # Error message should contain file path
        assert str(malformed_yaml_file) in str(exc_info.value)

    def test_parser_error_indicates_yaml_failure(self, malformed_yaml_file: Path) -> None:
        """Test that ParserError message indicates YAML parsing failed."""
        with pytest.raises(ParserError) as exc_info:
            parse_bmad_file(malformed_yaml_file)

        error_msg = str(exc_info.value).lower()
        # Should indicate parsing failure
        assert "failed to parse" in error_msg or "parse" in error_msg

    def test_unclosed_bracket_error(self, tmp_path: Path) -> None:
        """Test specific unclosed bracket YAML error."""
        content = """---
list: [1, 2, 3
---
"""
        path = tmp_path / "unclosed.md"
        path.write_text(content)

        with pytest.raises(ParserError) as exc_info:
            parse_bmad_file(path)

        assert str(path) in str(exc_info.value)

    def test_invalid_yaml_colon_in_value(self, tmp_path: Path) -> None:
        """Test YAML with invalid colon usage."""
        content = """---
key: value: more
---
"""
        path = tmp_path / "bad_colon.md"
        path.write_text(content)

        with pytest.raises(ParserError):
            parse_bmad_file(path)


class TestParseMissingFile:
    """AC4: Handle missing file."""

    def test_missing_file_raises_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_bmad_file("/path/to/missing.md")

    def test_file_not_found_error_contains_path(self) -> None:
        """Test that FileNotFoundError contains the file path."""
        missing_path = "/path/to/nonexistent/file.md"

        with pytest.raises(FileNotFoundError) as exc_info:
            parse_bmad_file(missing_path)

        # The error should reference the path
        assert "file.md" in str(exc_info.value) or missing_path in str(exc_info.value)

    def test_missing_directory_raises_file_not_found(self) -> None:
        """Test missing directory also raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_bmad_file("/nonexistent/directory/doc.md")


class TestComplexFrontmatterTypes:
    """AC5: Handle complex frontmatter types."""

    def test_list_in_frontmatter(self, complex_frontmatter_file: Path) -> None:
        """Test parsing list values in frontmatter."""
        doc = parse_bmad_file(complex_frontmatter_file)

        assert doc.frontmatter["stepsCompleted"] == [1, 2, 3, 4]

    def test_nested_list_in_frontmatter(self, complex_frontmatter_file: Path) -> None:
        """Test parsing nested list of strings in frontmatter."""
        doc = parse_bmad_file(complex_frontmatter_file)

        assert doc.frontmatter["inputDocuments"] == [
            "docs/prd.md",
            "docs/architecture.md",
        ]

    def test_nested_dict_in_frontmatter(self, complex_frontmatter_file: Path) -> None:
        """Test parsing nested dictionary in frontmatter."""
        doc = parse_bmad_file(complex_frontmatter_file)

        assert doc.frontmatter["metadata"]["author"] == "Pawel"
        assert doc.frontmatter["metadata"]["validated"] is True

    def test_boolean_values(self, tmp_path: Path) -> None:
        """Test parsing boolean values in frontmatter."""
        content = """---
enabled: true
disabled: false
---

# Content
"""
        path = tmp_path / "booleans.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter["enabled"] is True
        assert doc.frontmatter["disabled"] is False

    def test_integer_values(self, tmp_path: Path) -> None:
        """Test parsing integer values in frontmatter."""
        content = """---
count: 42
negative: -10
zero: 0
---

# Content
"""
        path = tmp_path / "integers.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter["count"] == 42
        assert doc.frontmatter["negative"] == -10
        assert doc.frontmatter["zero"] == 0

    def test_float_values(self, tmp_path: Path) -> None:
        """Test parsing float values in frontmatter."""
        content = """---
version: 1.5
percentage: 0.95
---

# Content
"""
        path = tmp_path / "floats.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter["version"] == 1.5
        assert doc.frontmatter["percentage"] == 0.95

    def test_null_values(self, tmp_path: Path) -> None:
        """Test parsing null values in frontmatter."""
        content = """---
empty: null
also_empty: ~
---

# Content
"""
        path = tmp_path / "nulls.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter["empty"] is None
        assert doc.frontmatter["also_empty"] is None

    def test_unicode_values(self, tmp_path: Path) -> None:
        """Test parsing Unicode values in frontmatter."""
        content = """---
author: Paweł
emoji: 🚀
japanese: 日本語
---

# Content
"""
        path = tmp_path / "unicode.md"
        path.write_text(content, encoding="utf-8")

        doc = parse_bmad_file(path)

        assert doc.frontmatter["author"] == "Paweł"
        assert doc.frontmatter["emoji"] == "🚀"
        assert doc.frontmatter["japanese"] == "日本語"


class TestEmptyFrontmatter:
    """AC6: Handle empty frontmatter."""

    def test_empty_frontmatter_returns_empty_dict(self, empty_frontmatter_file: Path) -> None:
        """Test that empty frontmatter returns empty dict."""
        doc = parse_bmad_file(empty_frontmatter_file)

        assert doc.frontmatter == {}

    def test_empty_frontmatter_preserves_content(self, empty_frontmatter_file: Path) -> None:
        """Test that content is preserved with empty frontmatter."""
        doc = parse_bmad_file(empty_frontmatter_file)

        assert "# Content" in doc.content

    def test_whitespace_only_frontmatter(self, tmp_path: Path) -> None:
        """Test frontmatter with only whitespace."""
        content = """---

---

# Content
"""
        path = tmp_path / "whitespace_fm.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter == {}
        assert "# Content" in doc.content


class TestBmadDocumentDataclass:
    """AC7: Return type consistency."""

    def test_result_is_bmad_document(self, sample_bmad_file: Path) -> None:
        """Test that result is a BmadDocument instance."""
        doc = parse_bmad_file(sample_bmad_file)

        assert isinstance(doc, BmadDocument)

    def test_bmad_document_has_frontmatter_field(self, sample_bmad_file: Path) -> None:
        """Test that BmadDocument has frontmatter field as dict."""
        doc = parse_bmad_file(sample_bmad_file)

        assert hasattr(doc, "frontmatter")
        assert isinstance(doc.frontmatter, dict)

    def test_bmad_document_has_content_field(self, sample_bmad_file: Path) -> None:
        """Test that BmadDocument has content field as str."""
        doc = parse_bmad_file(sample_bmad_file)

        assert hasattr(doc, "content")
        assert isinstance(doc.content, str)

    def test_bmad_document_has_path_field(self, sample_bmad_file: Path) -> None:
        """Test that BmadDocument has path field as str."""
        doc = parse_bmad_file(sample_bmad_file)

        assert hasattr(doc, "path")
        assert isinstance(doc.path, str)

    def test_bmad_document_is_dataclass(self) -> None:
        """Test that BmadDocument is a dataclass with correct fields."""
        field_names = {f.name for f in fields(BmadDocument)}

        assert field_names == {"frontmatter", "content", "path"}

    def test_bmad_document_field_types(self) -> None:
        """Test that BmadDocument fields have correct type annotations.

        Note: With 'from __future__ import annotations' (PEP 563), annotations
        are stored as strings. We compare string representations.
        """
        field_types = {f.name: f.type for f in fields(BmadDocument)}

        # With PEP 563, types are stored as strings
        assert field_types["frontmatter"] in (dict[str, Any], "dict[str, Any]")
        assert field_types["content"] in (str, "str")
        assert field_types["path"] in (str, "str")

    def test_path_is_original_file_path(self, sample_bmad_file: Path) -> None:
        """Test that path field contains the original file path."""
        doc = parse_bmad_file(sample_bmad_file)

        assert doc.path == str(sample_bmad_file)


class TestContentWithDelimiters:
    """AC8: Handle `---` delimiters in content."""

    def test_code_block_with_yaml_frontmatter(self, content_with_delimiters_file: Path) -> None:
        """Test that --- in code blocks is preserved in content."""
        doc = parse_bmad_file(content_with_delimiters_file)

        # Frontmatter should only contain title
        assert doc.frontmatter == {"title": "Architecture Doc"}

        # Content should preserve the YAML code block
        assert "```yaml" in doc.content
        assert "config: value" in doc.content

    def test_horizontal_rule_preserved(self, content_with_delimiters_file: Path) -> None:
        """Test that --- horizontal rules are preserved in content."""
        doc = parse_bmad_file(content_with_delimiters_file)

        # The horizontal rule (standalone ---) should be in content
        assert "More content after horizontal rule." in doc.content

    def test_frontmatter_not_confused_by_content_delimiters(
        self, content_with_delimiters_file: Path
    ) -> None:
        """Test frontmatter extraction ignores delimiters in content."""
        doc = parse_bmad_file(content_with_delimiters_file)

        # Should NOT have picked up config from the code block
        assert "config" not in doc.frontmatter

    def test_multiple_code_blocks_with_yaml(self, tmp_path: Path) -> None:
        """Test file with multiple YAML code blocks."""
        content = """---
title: Multi Code Block Test
---

First code block:

```yaml
---
first: block
---
```

Second code block:

```yaml
---
second: block
---
```

End of document.
"""
        path = tmp_path / "multi_code.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter == {"title": "Multi Code Block Test"}
        assert "first: block" in doc.content
        assert "second: block" in doc.content
        assert "End of document." in doc.content

    def test_horizontal_rules_not_confused_with_frontmatter(self, tmp_path: Path) -> None:
        """Test that multiple horizontal rules don't affect parsing."""
        content = """---
title: HR Test
---

# Section 1

---

# Section 2

---

# Section 3
"""
        path = tmp_path / "hr_test.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter == {"title": "HR Test"}
        assert "# Section 1" in doc.content
        assert "# Section 2" in doc.content
        assert "# Section 3" in doc.content


class TestRealWorldBmadFiles:
    """Tests using real-world BMAD file patterns."""

    def test_prd_style_frontmatter(self, real_prd_style_file: Path) -> None:
        """Test parsing PRD-style frontmatter from actual project."""
        doc = parse_bmad_file(real_prd_style_file)

        assert doc.frontmatter["stepsCompleted"] == [1, 2, 3, 4, 7, 8, 9, 10, 11]
        assert doc.frontmatter["inputDocuments"] == []
        assert doc.frontmatter["documentCounts"]["briefs"] == 0
        assert doc.frontmatter["workflowType"] == "prd"
        assert doc.frontmatter["lastStep"] == 11
        assert doc.frontmatter["project_name"] == "bmad-assist"
        assert doc.frontmatter["user_name"] == "Pawel"

    def test_architecture_style_frontmatter(self, real_architecture_style_file: Path) -> None:
        """Test parsing architecture-style frontmatter from actual project."""
        doc = parse_bmad_file(real_architecture_style_file)

        assert doc.frontmatter["stepsCompleted"] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert doc.frontmatter["inputDocuments"] == ["docs/prd.md"]
        assert doc.frontmatter["workflowType"] == "architecture"
        assert doc.frontmatter["status"] == "complete"
        assert doc.frontmatter["completedAt"] == "2025-12-08"

    def test_real_file_content_structure(self, real_prd_style_file: Path) -> None:
        """Test that real file content is properly extracted."""
        doc = parse_bmad_file(real_prd_style_file)

        assert "# Product Requirements Document" in doc.content
        assert "## Introduction" in doc.content


class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test parsing completely empty file."""
        path = tmp_path / "empty.md"
        path.write_text("")

        doc = parse_bmad_file(path)

        assert doc.frontmatter == {}
        assert doc.content == ""

    def test_only_frontmatter_no_content(self, tmp_path: Path) -> None:
        """Test file with frontmatter but no content after."""
        content = """---
title: No Content
---
"""
        path = tmp_path / "no_content.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter == {"title": "No Content"}
        assert doc.content.strip() == ""

    def test_frontmatter_with_multiline_string(self, tmp_path: Path) -> None:
        """Test frontmatter with YAML multiline string."""
        content = """---
description: |
  This is a multiline
  description that spans
  multiple lines.
---

# Content
"""
        path = tmp_path / "multiline.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert "multiline" in doc.frontmatter["description"]
        assert "multiple lines" in doc.frontmatter["description"]

    def test_frontmatter_with_folded_string(self, tmp_path: Path) -> None:
        """Test frontmatter with YAML folded string."""
        content = """---
summary: >
  This is a folded string
  that becomes one line.
---

# Content
"""
        path = tmp_path / "folded.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert "folded string" in doc.frontmatter["summary"]

    def test_large_content(self, tmp_path: Path) -> None:
        """Test parsing file with large content."""
        large_content = "# Heading\n\n" + "Paragraph.\n\n" * 1000
        content = f"""---
title: Large Document
---

{large_content}
"""
        path = tmp_path / "large.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter == {"title": "Large Document"}
        assert len(doc.content) > 10000  # Should be substantial

    def test_special_characters_in_frontmatter_values(self, tmp_path: Path) -> None:
        """Test frontmatter with special characters in values."""
        content = """---
path: "/home/user/project/file.md"
regex: "^[a-z]+$"
url: "https://example.com/path?query=value&other=123"
---

# Content
"""
        path = tmp_path / "special.md"
        path.write_text(content)

        doc = parse_bmad_file(path)

        assert doc.frontmatter["path"] == "/home/user/project/file.md"
        assert doc.frontmatter["regex"] == "^[a-z]+$"
        assert "query=value" in doc.frontmatter["url"]

    def test_parser_error_is_bmad_assist_error(self) -> None:
        """Test that ParserError inherits from BmadAssistError."""
        from bmad_assist.core.exceptions import BmadAssistError

        assert issubclass(ParserError, BmadAssistError)


class TestSampleBmadProjectParsing:
    """Integration tests using real BMAD sample project.

    These tests validate parsing against authentic BMAD documentation
    from tests/fixtures/bmad-sample-project/.
    """

    def test_parse_real_prd(self, sample_bmad_project: Path) -> None:
        """Parse real PRD document from sample project."""
        prd_path = sample_bmad_project / "prd.md"
        doc = parse_bmad_file(prd_path)

        # Verify frontmatter structure
        assert "stepsCompleted" in doc.frontmatter
        assert "project_name" in doc.frontmatter
        assert doc.frontmatter["project_name"] == "bmad-assist"

        # Verify content structure
        assert "# Product Requirements Document" in doc.content
        assert "## Functional Requirements" in doc.content

    def test_parse_real_architecture(self, sample_bmad_project: Path) -> None:
        """Parse real architecture document from sample project."""
        arch_path = sample_bmad_project / "architecture.md"
        doc = parse_bmad_file(arch_path)

        # Verify frontmatter
        assert "stepsCompleted" in doc.frontmatter
        assert "inputDocuments" in doc.frontmatter

        # Verify content contains architecture sections
        assert "# Architecture" in doc.content or "# System Architecture" in doc.content

    def test_parse_real_epics(self, sample_bmad_project: Path) -> None:
        """Parse real epics document from sample project."""
        epics_path = sample_bmad_project / "epics.md"
        doc = parse_bmad_file(epics_path)

        # Verify frontmatter with epic metadata
        assert doc.frontmatter["total_epics"] == 9
        assert doc.frontmatter["total_stories"] == 60
        assert doc.frontmatter["total_story_points"] == 132
        assert doc.frontmatter["status"] == "complete"
        assert doc.frontmatter["validated"] is True

        # Verify content has epic sections
        assert "# Epic" in doc.content or "## Epic" in doc.content
        assert "Story" in doc.content

    def test_parse_real_sprint_status(self, sample_sprint_artifacts: Path) -> None:
        """Parse real sprint-status.yaml file."""
        import yaml

        status_path = sample_sprint_artifacts / "sprint-status.yaml"
        with open(status_path) as f:
            data = yaml.safe_load(f)

        # Verify structure
        assert "development_status" in data
        assert "project" in data
        assert data["project"] == "bmad-assist"

        # Verify story statuses
        dev_status = data["development_status"]
        assert dev_status["1-1-project-initialization-with-pyproject-toml"] == "done"
        assert dev_status["epic-1"] == "in-progress"
        assert dev_status["epic-2"] == "in-progress"

    def test_parse_real_story_file(self, sample_sprint_artifacts: Path) -> None:
        """Parse real story file from sprint-artifacts."""
        story_path = sample_sprint_artifacts / "2-3-project-state-reader.md"
        doc = parse_bmad_file(story_path)

        # Story files may or may not have frontmatter
        # but should have story content
        assert "Project State Reader" in doc.content or "2.3" in doc.content

    def test_parse_real_retrospective(self, sample_sprint_artifacts: Path) -> None:
        """Parse real retrospective file."""
        retro_path = sample_sprint_artifacts / "epic-1-retrospective.md"
        if retro_path.exists():
            doc = parse_bmad_file(retro_path)

            # Retrospective should have content
            assert len(doc.content) > 100
            assert "Epic" in doc.content or "Retrospective" in doc.content

    def test_all_story_files_parseable(self, sample_sprint_artifacts: Path) -> None:
        """Verify all story files in sprint-artifacts are parseable."""
        story_files = list(sample_sprint_artifacts.glob("[0-9]-[0-9]-*.md"))

        assert len(story_files) >= 10, "Expected at least 10 story files"

        for story_file in story_files:
            # Should not raise any exceptions
            doc = parse_bmad_file(story_file)
            assert doc.content, f"Story file {story_file.name} has no content"

    def test_code_review_files_parseable(self, sample_sprint_artifacts: Path) -> None:
        """Verify code review files are parseable."""
        code_reviews_dir = sample_sprint_artifacts / "code-reviews"
        if code_reviews_dir.exists():
            review_files = list(code_reviews_dir.glob("*.md"))

            assert len(review_files) >= 5, "Expected at least 5 code review files"

            for review_file in review_files:
                doc = parse_bmad_file(review_file)
                assert doc.content, f"Review file {review_file.name} has no content"


class TestFallbackStoryParsing:
    """Tests for non-standard story format parsing (AC2-AC13 from tech-spec)."""

    def test_fallback_parses_prsp_format(self, tmp_path: Path) -> None:
        """Test parsing PRSP-5-1 style stories."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test Epic

## Stories

### PRSP-5-1: Add availability_mode Column
**Status:** ready-for-dev
**Estimate:** 2 SP

Some description.

### PRSP-5-2: Implement Room Drawer
**Status:** backlog
**Estimate:** 3 SP

Another description.
"""
        )

        epic = parse_epic_file(epic_file)

        assert len(epic.stories) == 2
        assert epic.stories[0].number == "1.1"
        assert epic.stories[0].title == "Add availability_mode Column"
        assert epic.stories[0].code == "PRSP-5-1"
        assert epic.stories[0].status == "ready-for-dev"
        assert epic.stories[0].estimate == 2

        assert epic.stories[1].number == "1.2"
        assert epic.stories[1].title == "Implement Room Drawer"
        assert epic.stories[1].code == "PRSP-5-2"
        assert epic.stories[1].status == "backlog"

    def test_fallback_parses_refactor_format(self, tmp_path: Path) -> None:
        """Test parsing REFACTOR-2-1 style stories."""
        epic_file = tmp_path / "epic-2.md"
        epic_file.write_text(
            """# Epic 2: Refactoring

## Stories by Component

### REFACTOR-2-1: Replace RadzenTextBox
**Status:** done
**Estimate:** 8 SP

Description here.
"""
        )

        epic = parse_epic_file(epic_file)

        assert len(epic.stories) == 1
        assert epic.stories[0].number == "2.1"
        assert epic.stories[0].title == "Replace RadzenTextBox"
        assert epic.stories[0].code == "REFACTOR-2-1"
        assert epic.stories[0].status == "done"

    def test_fallback_sequential_numbering(self, tmp_path: Path) -> None:
        """Test stories are numbered 1, 2, 3... regardless of original IDs."""
        epic_file = tmp_path / "epic-5.md"
        epic_file.write_text(
            """# Epic 5: Test

## Stories

### ABC-99-1: First
**Status:** backlog

### XYZ-1-999: Second
**Status:** backlog

### DEF-42-7: Third
**Status:** backlog
"""
        )

        epic = parse_epic_file(epic_file)

        assert len(epic.stories) == 3
        assert epic.stories[0].number == "5.1"
        assert epic.stories[1].number == "5.2"
        assert epic.stories[2].number == "5.3"

    def test_fallback_preserves_code(self, tmp_path: Path) -> None:
        """Test original code (PRSP-5-1) is preserved in code field."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

### MY-CUSTOM-CODE-123: Some Title
**Status:** backlog
"""
        )

        epic = parse_epic_file(epic_file)

        assert epic.stories[0].code == "MY-CUSTOM-CODE-123"
        assert epic.stories[0].title == "Some Title"

    def test_fallback_mixed_heading_levels(self, tmp_path: Path) -> None:
        """Test stories at ###, ####, ##### levels all detected."""
        epic_file = tmp_path / "epic-3.md"
        epic_file.write_text(
            """# Epic 3: Mixed

## Stories

### CODE-1: Level 3
**Status:** backlog

#### CODE-2: Level 4
**Status:** done

##### CODE-3: Level 5
**Status:** in-progress
"""
        )

        epic = parse_epic_file(epic_file)

        assert len(epic.stories) == 3
        assert epic.stories[0].title == "Level 3"
        assert epic.stories[1].title == "Level 4"
        assert epic.stories[2].title == "Level 5"

    def test_fallback_logs_info_not_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test INFO (not WARNING) is logged when fallback is used."""
        import logging

        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

### PRSP-1: Test Story
**Status:** backlog
"""
        )

        with caplog.at_level(logging.INFO):
            parse_epic_file(epic_file)

        # Should have INFO log about fallback parser
        assert any(
            "Non-standard story format" in record.message
            and record.levelname == "INFO"
            for record in caplog.records
        )

    def test_fallback_error_status_before_header(self, tmp_path: Path) -> None:
        """Test ParserError raised when Status appears with no valid story headers."""
        epic_file = tmp_path / "epic-1.md"
        # Status in Stories section but no story headers at all
        epic_file.write_text(
            """# Epic 1: Test

## Stories

Some text with **Status:** orphan-status but no story headers.
"""
        )

        with pytest.raises(ParserError, match="Found.*Status.*but no valid story headers"):
            parse_epic_file(epic_file)

    def test_fallback_skips_empty_headers(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test headers with no text are skipped with warning."""
        import logging

        epic_file = tmp_path / "epic-1.md"
        # Create file with an empty header (### followed by just whitespace)
        epic_file.write_text(
            """# Epic 1: Test

## Stories

###
**Status:** backlog

### PRSP-1: Valid Story
**Status:** done
"""
        )

        with caplog.at_level(logging.WARNING):
            epic = parse_epic_file(epic_file)

        # Only valid story should be parsed
        assert len(epic.stories) == 1
        assert epic.stories[0].title == "Valid Story"

    def test_fallback_cleans_status_trailing_asterisks(self, tmp_path: Path) -> None:
        """Test '**Status:** done**' becomes 'done'."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

### PRSP-1: Test
**Status:** done**

Description.
"""
        )

        epic = parse_epic_file(epic_file)

        assert epic.stories[0].status == "done"

    def test_fallback_ignores_status_outside_stories_section(
        self, tmp_path: Path
    ) -> None:
        """Test **Status:** in Epic Overview is not detected as story."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

**Status:** active

This epic is active.

## Stories

### PRSP-1: Real Story
**Status:** backlog

## Another Section

**Status:** should-be-ignored
"""
        )

        epic = parse_epic_file(epic_file)

        # Only the story in Stories section should be found
        assert len(epic.stories) == 1
        assert epic.stories[0].title == "Real Story"

    def test_fallback_handles_duplicate_codes(self, tmp_path: Path) -> None:
        """Test two stories with same code get different numbers."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

### CODE-1: First Instance
**Status:** done

### CODE-1: Second Instance (duplicate code)
**Status:** backlog
"""
        )

        epic = parse_epic_file(epic_file)

        assert len(epic.stories) == 2
        assert epic.stories[0].number == "1.1"
        assert epic.stories[1].number == "1.2"
        # Both have same code
        assert epic.stories[0].code == "CODE-1"
        assert epic.stories[1].code == "CODE-1"

    def test_mixed_format_logs_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test INFO logged when standard and non-standard stories mixed.

        Standard stories use ### Story X.Y (level 3) inside ## Stories (level 2).
        Non-standard stories also use ### but with different format.
        """
        import logging

        epic_file = tmp_path / "epic-1.md"
        # File with mixed formats - standard ### Story X.Y AND non-standard ### CODE:
        # Both inside the ## Stories section
        epic_file.write_text(
            """---
epic_num: 1
title: "Test"
---

## Stories

### Story 1.1: Standard Format
**Status:** done

### Story 1.2: Another Standard
**Status:** backlog

### PRSP-3: Non-standard (won't be parsed)
**Status:** in-progress

### PRSP-4: Another non-standard
**Status:** pending

## Next Section
This ends the Stories section.
"""
        )

        with caplog.at_level(logging.INFO):
            epic = parse_epic_file(epic_file)

        # Should only parse the 2 standard stories (fallback not triggered)
        assert len(epic.stories) == 2
        assert epic.stories[0].number == "1.1"
        assert epic.stories[1].number == "1.2"
        assert epic.stories[0].code is None  # Standard format has no code

        # Should log about mixed format (2 standard, 4 total Status fields)
        assert any(
            "Mixed story format" in record.message for record in caplog.records
        )

    def test_standard_format_unchanged(self, tmp_path: Path) -> None:
        """Test standard Story X.Y format still works (backward compat)."""
        epic_file = tmp_path / "epic-3.md"
        epic_file.write_text(
            """---
epic_num: 3
title: "Test Epic"
---

## Story 3.1: First Story
**Status:** done
**Estimate:** 2 SP

## Story 3.2: Second Story
**Status:** backlog
"""
        )

        epic = parse_epic_file(epic_file)

        assert len(epic.stories) == 2
        assert epic.stories[0].number == "3.1"
        assert epic.stories[0].title == "First Story"
        assert epic.stories[1].number == "3.2"
        assert epic.stories[1].title == "Second Story"

    def test_standard_format_no_code_field(self, tmp_path: Path) -> None:
        """Test code field is None for standard format."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """---
epic_num: 1
title: "Test"
---

## Story 1.1: Standard Story
**Status:** done
"""
        )

        epic = parse_epic_file(epic_file)

        assert epic.stories[0].code is None

    def test_fallback_extracts_dependencies_non_standard(self, tmp_path: Path) -> None:
        """Test dependencies with non-standard codes are extracted."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

### PRSP-5-1: First
**Status:** done

### PRSP-5-2: Second
**Status:** backlog
**Dependencies:** PRSP-5-1

### PRSP-5-3: Third
**Status:** backlog
**Dependencies:** PRSP-5-1, PRSP-5-2
"""
        )

        epic = parse_epic_file(epic_file)

        assert epic.stories[0].dependencies == []
        assert epic.stories[1].dependencies == ["PRSP-5-1"]
        assert epic.stories[2].dependencies == ["PRSP-5-1", "PRSP-5-2"]

    def test_fallback_title_only_no_code(self, tmp_path: Path) -> None:
        """Test header without colon results in title only, no code."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

### Simple Title Without Code
**Status:** backlog
"""
        )

        epic = parse_epic_file(epic_file)

        assert epic.stories[0].title == "Simple Title Without Code"
        assert epic.stories[0].code is None

    def test_parse_real_epic1_file(self) -> None:
        """Integration test with epic-1.md from project root."""
        epic_path = Path(__file__).parents[2] / "epic-1.md"
        if not epic_path.exists():
            pytest.skip("epic-1.md not found in project root")

        epic = parse_epic_file(epic_path)

        # Should have parsed stories using fallback
        assert len(epic.stories) > 0
        # First story should have code like PRSP-5-1
        assert epic.stories[0].code is not None
        assert "PRSP" in epic.stories[0].code
        # Should have sequential numbering
        assert epic.stories[0].number == "1.1"

    def test_parse_real_epic2_file(self) -> None:
        """Integration test with epic-2.md from project root."""
        epic_path = Path(__file__).parents[2] / "epic-2.md"
        if not epic_path.exists():
            pytest.skip("epic-2.md not found in project root")

        epic = parse_epic_file(epic_path)

        # Should have parsed stories using fallback
        assert len(epic.stories) > 0
        # First story should have code like REFACTOR-2-1
        assert epic.stories[0].code is not None
        assert "REFACTOR" in epic.stories[0].code
        # Should have sequential numbering
        assert epic.stories[0].number == "2.1"

    def test_fallback_phase_headers_skipped(self, tmp_path: Path) -> None:
        """Test phase headers without Status are skipped."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories by Phase

### Phase 5: Room Rota System

#### PRSP-5-1: First Story
**Status:** done

### Phase 6: Staff Planner

#### PRSP-6-1: Second Story
**Status:** backlog
"""
        )

        epic = parse_epic_file(epic_file)

        # Only actual stories (with Status) should be parsed
        assert len(epic.stories) == 2
        assert epic.stories[0].title == "First Story"
        assert epic.stories[1].title == "Second Story"
        # Phase headers should not be in results
        assert not any("Phase" in s.title for s in epic.stories)

    def test_fallback_acceptance_criteria_counted(self, tmp_path: Path) -> None:
        """Test acceptance criteria checkboxes are counted."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

### PRSP-1: Story With Criteria
**Status:** in-progress

**Acceptance Criteria:**
- [x] First criterion done
- [x] Second criterion done
- [ ] Third criterion pending
- [ ] Fourth criterion pending
"""
        )

        epic = parse_epic_file(epic_file)

        assert epic.stories[0].completed_criteria == 2
        assert epic.stories[0].total_criteria == 4

    def test_fallback_empty_stories_section(self, tmp_path: Path) -> None:
        """Test empty Stories section returns empty list without error."""
        epic_file = tmp_path / "epic-1.md"
        epic_file.write_text(
            """# Epic 1: Test

## Stories

## Next Section
"""
        )

        epic = parse_epic_file(epic_file)

        # Empty stories section should return empty list, not error
        assert len(epic.stories) == 0
