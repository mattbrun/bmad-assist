"""Tests for cache management."""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from bmad_assist import __version__
from bmad_assist.compiler.patching.cache import (
    CacheMeta,
    TemplateCache,
    compute_file_hash,
)
from bmad_assist.core.exceptions import PatchError


class TestComputeFileHash:
    """Tests for file hash computation."""

    def test_compute_hash_simple_file(self, tmp_path: Path) -> None:
        """Test computing hash of a simple file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = compute_file_hash(test_file)

        # Verify it's a valid SHA-256 hex digest (64 characters)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_compute_hash_consistent(self, tmp_path: Path) -> None:
        """Test that same content produces same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        content = "Same content"
        file1.write_text(content)
        file2.write_text(content)

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 == hash2

    def test_compute_hash_different_content(self, tmp_path: Path) -> None:
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 != hash2

    def test_compute_hash_nonexistent_file(self, tmp_path: Path) -> None:
        """Test computing hash of nonexistent file raises error."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            compute_file_hash(nonexistent)


class TestCacheMeta:
    """Tests for CacheMeta dataclass."""

    def test_create_cache_meta(self) -> None:
        """Test creating a CacheMeta instance."""
        meta = CacheMeta(
            compiled_at="2025-01-01T12:00:00Z",
            bmad_version="0.1.0",
            source_hashes={"workflow.yaml": "abc123", "instructions.xml": "def456"},
            patch_hash="ghi789",
        )

        assert meta.compiled_at == "2025-01-01T12:00:00Z"
        assert meta.bmad_version == "0.1.0"
        assert meta.source_hashes["workflow.yaml"] == "abc123"
        assert meta.patch_hash == "ghi789"


class TestTemplateCache:
    """Tests for TemplateCache class."""

    def test_get_cache_path_project(self, tmp_path: Path) -> None:
        """Test getting project cache path."""
        cache = TemplateCache()
        path = cache.get_cache_path("create-story", tmp_path)

        assert path == tmp_path / ".bmad-assist" / "cache" / "create-story.tpl.xml"

    def test_get_cache_path_global(self, tmp_path: Path) -> None:
        """Test getting global cache path."""
        cache = TemplateCache()

        with patch("pathlib.Path.home", return_value=tmp_path):
            path = cache.get_cache_path("create-story", project_root=None)

        expected = tmp_path / ".bmad-assist" / "cache" / __version__ / "create-story.tpl.xml"
        assert path == expected

    def test_is_valid_no_cache(self, tmp_path: Path) -> None:
        """Test cache validity when no cache exists."""
        cache = TemplateCache()
        result = cache.is_valid(
            workflow="create-story",
            project_root=tmp_path,
            source_files={"workflow.yaml": tmp_path / "w.yaml"},
            patch_path=tmp_path / "patch.yaml",
        )

        assert result is False

    def test_is_valid_stale_version(self, tmp_path: Path) -> None:
        """Test cache invalidation when bmad_version differs."""
        cache = TemplateCache()

        # Create cache directory and files
        cache_dir = tmp_path / ".bmad-assist" / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "create-story.tpl.xml").write_text("<compiled/>")

        # Create meta with different version
        meta = {
            "compiled_at": "2025-01-01T12:00:00Z",
            "bmad_version": "0.0.1",  # Different from current
            "source_hashes": {},
            "patch_hash": "abc",
        }
        # Meta file is per-workflow: create-story.tpl.xml.meta.yaml
        (cache_dir / "create-story.tpl.xml.meta.yaml").write_text(yaml.dump(meta))

        # Create source file
        source_file = tmp_path / "workflow.yaml"
        source_file.write_text("content")

        # Create patch file
        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text("patch")

        result = cache.is_valid(
            workflow="create-story",
            project_root=tmp_path,
            source_files={"workflow.yaml": source_file},
            patch_path=patch_file,
        )

        assert result is False

    def test_save_and_load_cached(self, tmp_path: Path) -> None:
        """Test saving and loading cached template."""
        cache = TemplateCache()
        content = "<compiled>Template content</compiled>"

        # Create source and patch files
        source_file = tmp_path / "workflow.yaml"
        source_file.write_text("source content")
        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text("patch content")

        meta = CacheMeta(
            compiled_at=datetime.now(timezone.utc).isoformat(),
            bmad_version="0.1.0",
            source_hashes={"workflow.yaml": compute_file_hash(source_file)},
            patch_hash=compute_file_hash(patch_file),
        )

        cache.save(
            workflow="create-story",
            content=content,
            metadata=meta,
            project_root=tmp_path,
        )

        # Load it back
        loaded = cache.load_cached("create-story", tmp_path)

        assert loaded is not None
        assert content in loaded

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Test that save creates cache directory if it doesn't exist."""
        cache = TemplateCache()
        content = "<compiled/>"
        meta = CacheMeta(
            compiled_at="2025-01-01T12:00:00Z",
            bmad_version="0.1.0",
            source_hashes={},
            patch_hash="abc",
        )

        # Directory shouldn't exist yet
        cache_dir = tmp_path / ".bmad-assist" / "cache"
        assert not cache_dir.exists()

        cache.save("test-workflow", content, meta, tmp_path)

        # Now it should exist
        assert cache_dir.exists()
        assert (cache_dir / "test-workflow.tpl.xml").exists()

    def test_save_atomic_write(self, tmp_path: Path) -> None:
        """Test that save uses atomic write pattern."""
        cache = TemplateCache()
        content = "<compiled/>"
        meta = CacheMeta(
            compiled_at="2025-01-01T12:00:00Z",
            bmad_version="0.1.0",
            source_hashes={},
            patch_hash="abc",
        )

        cache.save("test-workflow", content, meta, tmp_path)

        # Verify no temp files left behind
        cache_dir = tmp_path / ".bmad-assist" / "cache"
        temp_files = list(cache_dir.glob("*.tmp"))
        assert len(temp_files) == 0

    @pytest.mark.skipif(os.geteuid() == 0, reason="Root ignores permissions")
    def test_save_permission_error(self, tmp_path: Path) -> None:
        """Test that save raises PatchError on permission error."""
        cache = TemplateCache()
        content = "<compiled/>"
        meta = CacheMeta(
            compiled_at="2025-01-01T12:00:00Z",
            bmad_version="0.1.0",
            source_hashes={},
            patch_hash="abc",
        )

        # Create read-only directory
        cache_dir = tmp_path / ".bmad-assist" / "cache"
        cache_dir.mkdir(parents=True)
        os.chmod(cache_dir, 0o444)

        try:
            with pytest.raises(PatchError) as exc_info:
                cache.save("test-workflow", content, meta, tmp_path)

            assert (
                "permission" in str(exc_info.value).lower()
                or "write" in str(exc_info.value).lower()
            )
        finally:
            # Restore permissions for cleanup
            os.chmod(cache_dir, 0o755)

    def test_is_valid_source_hash_mismatch(self, tmp_path: Path) -> None:
        """Test cache invalidation when source file hash changes."""
        cache = TemplateCache()

        # Create source file with initial content
        source_file = tmp_path / "workflow.yaml"
        source_file.write_text("original content")
        original_hash = compute_file_hash(source_file)

        # Create patch file
        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text("patch content")

        # Create cache with original hash
        cache_dir = tmp_path / ".bmad-assist" / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "create-story.tpl.xml").write_text("<compiled/>")

        meta = {
            "compiled_at": "2025-01-01T12:00:00Z",
            "bmad_version": "0.1.0",
            "source_hashes": {"workflow.yaml": original_hash},
            "patch_hash": compute_file_hash(patch_file),
        }
        # Meta file is per-workflow: create-story.tpl.xml.meta.yaml
        (cache_dir / "create-story.tpl.xml.meta.yaml").write_text(yaml.dump(meta))

        # Modify source file
        source_file.write_text("modified content")

        result = cache.is_valid(
            workflow="create-story",
            project_root=tmp_path,
            source_files={"workflow.yaml": source_file},
            patch_path=patch_file,
        )

        assert result is False

    def test_load_cached_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test that load_cached returns None when cache doesn't exist."""
        cache = TemplateCache()
        result = cache.load_cached("nonexistent", tmp_path)
        assert result is None

    def test_is_valid_with_matching_hashes(self, tmp_path: Path) -> None:
        """Test cache is valid when all hashes match."""
        cache = TemplateCache()

        # Create source file
        source_file = tmp_path / "workflow.yaml"
        source_file.write_text("source content")

        # Create patch file
        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text("patch content")

        # Create cache with matching hashes
        cache_dir = tmp_path / ".bmad-assist" / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "create-story.tpl.xml").write_text("<compiled/>")

        meta = {
            "compiled_at": "2025-01-01T12:00:00Z",
            "bmad_version": "0.1.0",  # Must match bmad_assist.__version__
            "source_hashes": {"workflow.yaml": compute_file_hash(source_file)},
            "patch_hash": compute_file_hash(patch_file),
        }
        # Meta file is per-workflow: create-story.tpl.xml.meta.yaml
        (cache_dir / "create-story.tpl.xml.meta.yaml").write_text(yaml.dump(meta))

        result = cache.is_valid(
            workflow="create-story",
            project_root=tmp_path,
            source_files={"workflow.yaml": source_file},
            patch_path=patch_file,
        )

        assert result is True
