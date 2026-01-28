"""Tests for credentials security with .env.

Story 1.5 tests (AC1-AC11):
- AC1-AC11: Credentials Security with .env

Note: Edge cases are in test_config_env_edge_cases.py.
Extracted from test_config.py as part of Story 1.8 (Test Suite Refactoring).
"""

import os
from pathlib import Path

import pytest

from bmad_assist.core.config import (
    ENV_CREDENTIAL_KEYS,
    ENV_FILE_NAME,
    _check_env_file_permissions,
    _mask_credential,
    load_config_with_project,
    load_env_file,
)


class TestEnvFileConstants:
    """Tests for Story 1.5: .env file constants."""

    def test_env_credential_keys_contains_telegram_bot_token(self) -> None:
        """ENV_CREDENTIAL_KEYS contains TELEGRAM_BOT_TOKEN."""
        assert "TELEGRAM_BOT_TOKEN" in ENV_CREDENTIAL_KEYS

    def test_env_credential_keys_contains_telegram_chat_id(self) -> None:
        """ENV_CREDENTIAL_KEYS contains TELEGRAM_CHAT_ID."""
        assert "TELEGRAM_CHAT_ID" in ENV_CREDENTIAL_KEYS

    def test_env_credential_keys_contains_discord_webhook_url(self) -> None:
        """ENV_CREDENTIAL_KEYS contains DISCORD_WEBHOOK_URL."""
        assert "DISCORD_WEBHOOK_URL" in ENV_CREDENTIAL_KEYS

    def test_env_credential_keys_is_frozenset(self) -> None:
        """ENV_CREDENTIAL_KEYS is a frozenset (immutable)."""
        assert isinstance(ENV_CREDENTIAL_KEYS, frozenset)

    def test_env_file_name_is_dotenv(self) -> None:
        """ENV_FILE_NAME is .env."""
        assert ENV_FILE_NAME == ".env"


# === AC7: Credential Masking Tests ===


class TestCredentialMasking:
    """Tests for AC7: Credential values masked in logs."""

    def test_mask_credential_shows_first_7_chars(self) -> None:
        """_mask_credential shows first 7 characters for long values."""
        result = _mask_credential("sk-ant-api03-1234567890abcdef")
        assert result == "sk-ant-***"

    def test_mask_credential_short_value_is_fully_masked(self) -> None:
        """_mask_credential fully masks values <= 7 characters."""
        assert _mask_credential("abc") == "***"
        assert _mask_credential("1234567") == "***"

    def test_mask_credential_exactly_8_chars(self) -> None:
        """_mask_credential shows first 7 for exactly 8 character values."""
        result = _mask_credential("12345678")
        assert result == "1234567***"

    def test_mask_credential_empty_string(self) -> None:
        """_mask_credential handles empty string."""
        result = _mask_credential("")
        assert result == "***"

    def test_mask_credential_none_value(self) -> None:
        """_mask_credential handles None gracefully."""
        result = _mask_credential(None)
        assert result == "***"


# === AC1: Environment Variables Loaded from .env ===


class TestLoadEnvFile:
    """Tests for AC1: Environment variables loaded from .env."""

    def test_loads_env_file_successfully(self, tmp_path: Path) -> None:
        """AC1: Environment variables loaded from .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_LOAD_ENV_VAR=test_value_123\n")

        # Clean up any previous value
        os.environ.pop("TEST_LOAD_ENV_VAR", None)

        result = load_env_file(project_path=tmp_path)

        assert result is True
        assert os.environ.get("TEST_LOAD_ENV_VAR") == "test_value_123"

        # Cleanup
        os.environ.pop("TEST_LOAD_ENV_VAR", None)

    def test_existing_env_var_not_overridden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC1a: Existing environment variables are NOT overridden."""
        # Set existing env var BEFORE loading .env
        monkeypatch.setenv("EXISTING_TEST_VAR", "system_value")

        # Create .env with different value for same key
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_TEST_VAR=dotenv_value\n")

        result = load_env_file(project_path=tmp_path)

        assert result is True
        # CRITICAL: System value must be preserved, NOT overwritten
        assert os.environ.get("EXISTING_TEST_VAR") == "system_value"

    def test_new_env_var_added(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """New env vars from .env are added when not already set."""
        # Ensure var doesn't exist
        monkeypatch.delenv("NEW_TEST_VAR", raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text("NEW_TEST_VAR=new_value\n")

        result = load_env_file(project_path=tmp_path)

        assert result is True
        assert os.environ.get("NEW_TEST_VAR") == "new_value"

        # Cleanup
        os.environ.pop("NEW_TEST_VAR", None)


# === AC4: Missing .env File Returns False ===


class TestMissingEnvFile:
    """Tests for AC4: Missing .env file returns False."""

    def test_missing_env_file_returns_false(self, tmp_path: Path) -> None:
        """AC4: Missing .env file is not an error."""
        # No .env file created
        result = load_env_file(project_path=tmp_path)
        assert result is False

    def test_missing_env_file_no_exception(self, tmp_path: Path) -> None:
        """Missing .env file does not raise exception."""
        # Should not raise
        result = load_env_file(project_path=tmp_path)
        assert result is False


# === AC2, AC3: Permission Checks ===


class TestEnvFilePermissions:
    """Tests for AC2, AC3: .env file permissions."""

    def test_insecure_permissions_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC2: Warning logged for insecure permissions."""
        import sys

        if sys.platform == "win32":
            pytest.skip("Permission tests not applicable on Windows")

        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=value\n")
        env_file.chmod(0o644)  # Insecure: world-readable

        import logging

        with caplog.at_level(logging.WARNING):
            load_env_file(project_path=tmp_path)

        assert "insecure permissions" in caplog.text.lower()
        assert "644" in caplog.text

        # Cleanup
        os.environ.pop("TEST_VAR", None)

    def test_secure_permissions_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC3: No warning for secure permissions (600)."""
        import sys

        if sys.platform == "win32":
            pytest.skip("Permission tests not applicable on Windows")

        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR_SECURE=value\n")
        env_file.chmod(0o600)  # Secure

        import logging

        with caplog.at_level(logging.WARNING):
            load_env_file(project_path=tmp_path)

        assert "insecure permissions" not in caplog.text.lower()

        # Cleanup
        os.environ.pop("TEST_VAR_SECURE", None)

    def test_readonly_permissions_400_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC3: No warning for read-only permissions (400) - even more secure."""
        import sys

        if sys.platform == "win32":
            pytest.skip("Permission tests not applicable on Windows")

        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR_READONLY=value\n")
        env_file.chmod(0o400)  # Secure: read-only for owner

        import logging

        with caplog.at_level(logging.WARNING):
            load_env_file(project_path=tmp_path)

        assert "insecure permissions" not in caplog.text.lower()

        # Cleanup (need to restore write permission to delete)
        env_file.chmod(0o600)
        os.environ.pop("TEST_VAR_READONLY", None)

    def test_check_permissions_false_skips_check(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """check_permissions=False skips permission check."""
        import sys

        if sys.platform == "win32":
            pytest.skip("Permission tests not applicable on Windows")

        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR_SKIP=value\n")
        env_file.chmod(0o644)  # Would normally warn

        import logging

        with caplog.at_level(logging.WARNING):
            load_env_file(project_path=tmp_path, check_permissions=False)

        assert "insecure permissions" not in caplog.text.lower()

        # Cleanup
        os.environ.pop("TEST_VAR_SKIP", None)


# === AC10: Windows Skips Permission Check ===


class TestWindowsPermissions:
    """Tests for AC10: Windows skips permission check."""

    def test_check_env_file_permissions_windows_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC10: Permission check is no-op on Windows."""
        # Patch sys.platform to simulate Windows
        monkeypatch.setattr("sys.platform", "win32")

        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=value\n")
        env_file.chmod(0o644)  # Would warn on Unix

        # Should not raise or warn on "Windows"
        _check_env_file_permissions(env_file)


# === AC11: UTF-8 Encoding Support ===


class TestUtf8Encoding:
    """Tests for AC11: UTF-8 encoding support."""

    def test_utf8_values_loaded_correctly(self, tmp_path: Path) -> None:
        """AC11: UTF-8 values in .env are loaded correctly."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TEST_UTF8_VAR=Paweł_żółć_ąęś\n",
            encoding="utf-8",
        )

        # Clean up any previous value
        os.environ.pop("TEST_UTF8_VAR", None)

        result = load_env_file(project_path=tmp_path)

        assert result is True
        assert os.environ.get("TEST_UTF8_VAR") == "Paweł_żółć_ąęś"

        # Cleanup
        os.environ.pop("TEST_UTF8_VAR", None)


# === AC8: .env Available via os.getenv ===


class TestEnvAvailableOsGetenv:
    """Tests for AC8: Loaded env vars available via os.getenv."""

    def test_loaded_vars_available_via_os_getenv(self, tmp_path: Path) -> None:
        """AC8: Loaded values are accessible via os.getenv()."""
        env_file = tmp_path / ".env"
        env_file.write_text("GETENV_TEST_VAR=getenv_value\n")

        # Clean up any previous value
        os.environ.pop("GETENV_TEST_VAR", None)

        load_env_file(project_path=tmp_path)

        # Access via os.getenv (standard Python API)
        assert os.getenv("GETENV_TEST_VAR") == "getenv_value"

        # Cleanup
        os.environ.pop("GETENV_TEST_VAR", None)


# === AC9: Integration with load_config_with_project ===


class TestEnvIntegrationWithConfig:
    """Tests for AC9: .env integration with config loading."""

    def test_env_loaded_before_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC9: .env is loaded before config validation."""
        monkeypatch.chdir(tmp_path)
        # Create valid global config
        global_config = tmp_path / "global.yaml"
        global_config.write_text(
            """
providers:
  master:
    provider: claude
    model: opus_4
"""
        )

        # Create project directory with .env
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        env_file = project_dir / ".env"
        env_file.write_text("CONFIG_INTEGRATION_TEST=integrated_value\n")

        # Clean up any previous value
        os.environ.pop("CONFIG_INTEGRATION_TEST", None)

        # Load config - should also load .env
        load_config_with_project(
            project_path=project_dir,
            global_config_path=global_config,
        )

        # .env should have been loaded
        assert os.getenv("CONFIG_INTEGRATION_TEST") == "integrated_value"

        # Cleanup
        os.environ.pop("CONFIG_INTEGRATION_TEST", None)

    def test_env_loaded_even_if_project_config_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC9: .env is loaded even when bmad-assist.yaml doesn't exist."""
        monkeypatch.chdir(tmp_path)
        # Create valid global config
        global_config = tmp_path / "global.yaml"
        global_config.write_text(
            """
providers:
  master:
    provider: claude
    model: opus_4
"""
        )

        # Create project directory with .env but NO bmad-assist.yaml
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        env_file = project_dir / ".env"
        env_file.write_text("ENV_NO_PROJECT_CONFIG=value_from_env\n")

        # Clean up any previous value
        os.environ.pop("ENV_NO_PROJECT_CONFIG", None)

        # Load config - should still load .env
        load_config_with_project(
            project_path=project_dir,
            global_config_path=global_config,
        )

        # .env should have been loaded
        assert os.getenv("ENV_NO_PROJECT_CONFIG") == "value_from_env"

        # Cleanup
        os.environ.pop("ENV_NO_PROJECT_CONFIG", None)
