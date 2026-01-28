"""Tests for Story 1.7: Interactive Config Generation.

Comprehensive tests covering:
- Provider and model constants (AC2, AC3)
- Provider selection (AC2)
- Model selection (AC3)
- Config generation and validation (AC4, AC5)
- Default values (AC9)
- Confirmation flow (AC10)
- Cancellation handling (AC8)
- Atomic write (AC11)
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from rich.console import Console

from bmad_assist.core.config_generator import (
    AVAILABLE_PROVIDERS,
    CONFIG_FILENAME,
    ConfigGenerator,
    run_config_wizard,
)

# =============================================================================
# Test: Constants and Provider Definitions
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_config_filename_is_bmad_assist_yaml(self) -> None:
        """Config filename matches expected value."""
        assert CONFIG_FILENAME == "bmad-assist.yaml"

    def test_available_providers_is_final_dict(self) -> None:
        """AVAILABLE_PROVIDERS is a dictionary."""
        assert isinstance(AVAILABLE_PROVIDERS, dict)
        assert len(AVAILABLE_PROVIDERS) > 0


class TestProviderDefinitions:
    """Tests for provider and model definitions (AC2, AC3)."""

    def test_claude_provider_exists(self) -> None:
        """AC2: Claude provider is available."""
        assert "claude" in AVAILABLE_PROVIDERS

    def test_claude_has_required_fields(self) -> None:
        """Claude provider has display_name, models, default_model."""
        claude = AVAILABLE_PROVIDERS["claude"]
        assert "display_name" in claude
        assert "models" in claude
        assert "default_model" in claude

    def test_claude_models_include_opus_4(self) -> None:
        """AC3: Claude models include opus_4."""
        claude = AVAILABLE_PROVIDERS["claude"]
        assert "opus_4" in claude["models"]

    def test_claude_models_include_sonnet_4(self) -> None:
        """AC3: Claude models include sonnet_4."""
        claude = AVAILABLE_PROVIDERS["claude"]
        assert "sonnet_4" in claude["models"]

    def test_claude_models_include_sonnet_3_5(self) -> None:
        """AC3: Claude models include sonnet_3_5."""
        claude = AVAILABLE_PROVIDERS["claude"]
        assert "sonnet_3_5" in claude["models"]

    def test_claude_models_include_haiku_3_5(self) -> None:
        """AC3: Claude models include haiku_3_5."""
        claude = AVAILABLE_PROVIDERS["claude"]
        assert "haiku_3_5" in claude["models"]

    def test_claude_default_model_is_opus_4(self) -> None:
        """AC3: Claude default model is opus_4."""
        claude = AVAILABLE_PROVIDERS["claude"]
        assert claude["default_model"] == "opus_4"

    def test_codex_provider_exists(self) -> None:
        """AC2: Codex provider is available."""
        assert "codex" in AVAILABLE_PROVIDERS

    def test_codex_has_required_fields(self) -> None:
        """Codex provider has display_name, models, default_model."""
        codex = AVAILABLE_PROVIDERS["codex"]
        assert "display_name" in codex
        assert "models" in codex
        assert "default_model" in codex

    def test_gemini_provider_exists(self) -> None:
        """AC2: Gemini provider is available."""
        assert "gemini" in AVAILABLE_PROVIDERS

    def test_gemini_has_required_fields(self) -> None:
        """Gemini provider has display_name, models, default_model."""
        gemini = AVAILABLE_PROVIDERS["gemini"]
        assert "display_name" in gemini
        assert "models" in gemini
        assert "default_model" in gemini


# =============================================================================
# Test: ConfigGenerator Class
# =============================================================================


class TestConfigGeneratorInit:
    """Tests for ConfigGenerator initialization."""

    def test_init_with_no_console_creates_one(self) -> None:
        """ConfigGenerator creates console if none provided."""
        generator = ConfigGenerator()
        assert generator.console is not None
        assert isinstance(generator.console, Console)

    def test_init_with_console_uses_provided(self) -> None:
        """ConfigGenerator uses provided console."""
        custom_console = Console()
        generator = ConfigGenerator(console=custom_console)
        assert generator.console is custom_console


class TestProviderSelection:
    """Tests for provider selection prompt (AC2)."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_provider_returns_selection(self, mock_ask: MagicMock) -> None:
        """AC2: Provider selection returns user choice."""
        mock_ask.return_value = "claude"
        generator = ConfigGenerator(Console())
        result = generator._prompt_provider()
        assert result == "claude"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_provider_default_is_claude(self, mock_ask: MagicMock) -> None:
        """AC2: Default provider is claude."""
        mock_ask.return_value = "claude"
        generator = ConfigGenerator(Console())
        generator._prompt_provider()
        mock_ask.assert_called_once()
        call_kwargs = mock_ask.call_args[1]
        assert call_kwargs.get("default") == "claude"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_provider_choices_include_all_providers(self, mock_ask: MagicMock) -> None:
        """AC2: All providers available as choices."""
        mock_ask.return_value = "codex"
        generator = ConfigGenerator(Console())
        generator._prompt_provider()
        call_kwargs = mock_ask.call_args[1]
        choices = call_kwargs.get("choices")
        assert "claude" in choices
        assert "codex" in choices
        assert "gemini" in choices

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_provider_returns_codex(self, mock_ask: MagicMock) -> None:
        """AC2: Can select codex provider."""
        mock_ask.return_value = "codex"
        generator = ConfigGenerator(Console())
        result = generator._prompt_provider()
        assert result == "codex"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_provider_returns_gemini(self, mock_ask: MagicMock) -> None:
        """AC2: Can select gemini provider."""
        mock_ask.return_value = "gemini"
        generator = ConfigGenerator(Console())
        result = generator._prompt_provider()
        assert result == "gemini"


class TestModelSelection:
    """Tests for model selection prompt (AC3)."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_model_for_claude_returns_selection(self, mock_ask: MagicMock) -> None:
        """AC3: Model selection for claude returns user choice."""
        mock_ask.return_value = "opus_4"
        generator = ConfigGenerator(Console())
        result = generator._prompt_model("claude")
        assert result == "opus_4"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_model_for_claude_default_is_opus_4(self, mock_ask: MagicMock) -> None:
        """AC3: Default model for claude is opus_4."""
        mock_ask.return_value = "opus_4"
        generator = ConfigGenerator(Console())
        generator._prompt_model("claude")
        call_kwargs = mock_ask.call_args[1]
        assert call_kwargs.get("default") == "opus_4"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_model_for_codex_default_is_gpt4o(self, mock_ask: MagicMock) -> None:
        """AC3: Default model for codex is gpt-4o."""
        mock_ask.return_value = "gpt-4o"
        generator = ConfigGenerator(Console())
        generator._prompt_model("codex")
        call_kwargs = mock_ask.call_args[1]
        assert call_kwargs.get("default") == "gpt-4o"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_model_for_gemini_default_is_pro(self, mock_ask: MagicMock) -> None:
        """AC3: Default model for gemini is gemini_2_5_pro."""
        mock_ask.return_value = "gemini_2_5_pro"
        generator = ConfigGenerator(Console())
        generator._prompt_model("gemini")
        call_kwargs = mock_ask.call_args[1]
        assert call_kwargs.get("default") == "gemini_2_5_pro"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_prompt_model_choices_match_provider(self, mock_ask: MagicMock) -> None:
        """AC3: Model choices match selected provider."""
        mock_ask.return_value = "sonnet_4"
        generator = ConfigGenerator(Console())
        generator._prompt_model("claude")
        call_kwargs = mock_ask.call_args[1]
        choices = call_kwargs.get("choices")
        # Should contain claude models
        assert "opus_4" in choices
        assert "sonnet_4" in choices
        # Should NOT contain codex/gemini models
        assert "gpt-4o" not in choices
        assert "gemini_2_5_pro" not in choices


class TestBuildConfig:
    """Tests for config dictionary building (AC4, AC9)."""

    def test_build_config_contains_provider(self) -> None:
        """AC4: Built config contains provider."""
        generator = ConfigGenerator(Console())
        config = generator._build_config("claude", "opus_4")
        assert config["providers"]["master"]["provider"] == "claude"

    def test_build_config_contains_model(self) -> None:
        """AC4: Built config contains model."""
        generator = ConfigGenerator(Console())
        config = generator._build_config("claude", "opus_4")
        assert config["providers"]["master"]["model"] == "opus_4"

    def test_build_config_no_state_path(self) -> None:
        """AC9: Built config does NOT include state_path (uses project-based default)."""
        generator = ConfigGenerator(Console())
        config = generator._build_config("claude", "opus_4")
        # state_path is not set - get_state_path() uses project_root instead
        assert "state_path" not in config

    def test_build_config_has_default_timeout(self) -> None:
        """AC9: Built config has default timeout of 300."""
        generator = ConfigGenerator(Console())
        config = generator._build_config("claude", "opus_4")
        assert config["timeout"] == 300


# =============================================================================
# Test: Config Generation and Validation
# =============================================================================


class TestConfigGeneration:
    """Tests for config file generation (AC4, AC5)."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_generates_yaml_file(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC4: Generated config is a YAML file."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        config_path = run_config_wizard(tmp_path, Console())

        assert config_path.exists()
        assert config_path.suffix == ".yaml"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_generates_valid_yaml(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC4: Generated config is valid YAML."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        config_path = run_config_wizard(tmp_path, Console())

        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert isinstance(config, dict)

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_yaml_has_providers_section(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC4: Generated YAML has providers section."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        config_path = run_config_wizard(tmp_path, Console())

        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "providers" in config
        assert "master" in config["providers"]

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_yaml_has_header_comments(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC4: Generated YAML has header comments."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        config_path = run_config_wizard(tmp_path, Console())

        content = config_path.read_text()
        assert "# bmad-assist configuration" in content
        assert "# Generated by interactive setup wizard" in content

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_config_validates_with_pydantic(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC5: Generated config validates with load_config_with_project."""
        monkeypatch.chdir(tmp_path)
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        run_config_wizard(tmp_path, Console())

        # Import here to avoid circular import issues in tests
        from bmad_assist.core.config import load_config_with_project

        # Should not raise
        config = load_config_with_project(tmp_path)
        assert config is not None
        assert config.providers.master.provider == "claude"
        assert config.providers.master.model == "opus_4"
        # Verify timeout field is accessible (not silently discarded)
        assert config.timeout == 300

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_config_with_codex_validates(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC5: Config with codex provider validates."""
        monkeypatch.chdir(tmp_path)
        mock_prompt.side_effect = ["codex", "gpt-4o"]
        mock_confirm.return_value = True

        run_config_wizard(tmp_path, Console())

        from bmad_assist.core.config import load_config_with_project

        config = load_config_with_project(tmp_path)
        assert config.providers.master.provider == "codex"
        assert config.providers.master.model == "gpt-4o"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_config_with_gemini_validates(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC5: Config with gemini provider validates."""
        monkeypatch.chdir(tmp_path)
        mock_prompt.side_effect = ["gemini", "gemini_2_5_pro"]
        mock_confirm.return_value = True

        run_config_wizard(tmp_path, Console())

        from bmad_assist.core.config import load_config_with_project

        config = load_config_with_project(tmp_path)
        assert config.providers.master.provider == "gemini"
        assert config.providers.master.model == "gemini_2_5_pro"


# =============================================================================
# Test: Confirmation Flow
# =============================================================================


class TestConfirmation:
    """Tests for save confirmation flow (AC10)."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_confirm_save_is_called(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC10: Confirm.ask is called before saving."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        run_config_wizard(tmp_path, Console())

        mock_confirm.assert_called_once()

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_no_save_on_rejection(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC10: Config not saved if user rejects."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            run_config_wizard(tmp_path, Console())

        assert exc_info.value.code == 1
        config_path = tmp_path / CONFIG_FILENAME
        assert not config_path.exists()

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_rejection_exit_code_is_one(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC10: User rejection exits with code 1."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            run_config_wizard(tmp_path, Console())

        assert exc_info.value.code == 1


# =============================================================================
# Test: Cancellation Handling
# =============================================================================


class TestCancellation:
    """Tests for cancellation handling (AC8)."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_keyboard_interrupt_propagates(self, mock_ask: MagicMock, tmp_path: Path) -> None:
        """AC8: KeyboardInterrupt propagates to caller."""
        mock_ask.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            run_config_wizard(tmp_path, Console())

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_eof_error_propagates(self, mock_ask: MagicMock, tmp_path: Path) -> None:
        """AC8: EOFError propagates to caller (piped input scenario)."""
        mock_ask.side_effect = EOFError()

        with pytest.raises(EOFError):
            run_config_wizard(tmp_path, Console())

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_no_partial_file_on_keyboard_interrupt(
        self, mock_ask: MagicMock, tmp_path: Path
    ) -> None:
        """AC8: No partial config file on KeyboardInterrupt."""
        mock_ask.side_effect = KeyboardInterrupt()
        config_path = tmp_path / CONFIG_FILENAME

        with pytest.raises(KeyboardInterrupt):
            run_config_wizard(tmp_path, Console())

        assert not config_path.exists()

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    def test_no_partial_file_on_eof_error(self, mock_ask: MagicMock, tmp_path: Path) -> None:
        """AC8: No partial config file on EOFError."""
        mock_ask.side_effect = EOFError()
        config_path = tmp_path / CONFIG_FILENAME

        with pytest.raises(EOFError):
            run_config_wizard(tmp_path, Console())

        assert not config_path.exists()

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_keyboard_interrupt_during_confirm_propagates(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC8: KeyboardInterrupt during confirmation propagates."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            run_config_wizard(tmp_path, Console())


# =============================================================================
# Test: Atomic Write
# =============================================================================


class TestAtomicWrite:
    """Tests for atomic write pattern (AC11)."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_file_created_atomically(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC11: Config file is created atomically."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        config_path = run_config_wizard(tmp_path, Console())

        # File should exist and be valid
        assert config_path.exists()
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_no_temp_files_left_on_success(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC11: No temp files left behind on success."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        run_config_wizard(tmp_path, Console())

        temp_files = list(tmp_path.glob(".bmad-assist-*.yaml.tmp"))
        assert len(temp_files) == 0

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    @patch("bmad_assist.core.config_generator.os.rename")
    def test_temp_file_cleanup_on_rename_failure(
        self,
        mock_rename: MagicMock,
        mock_confirm: MagicMock,
        mock_prompt: MagicMock,
        tmp_path: Path,
    ) -> None:
        """AC11: Temp file cleaned up if rename fails."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True
        mock_rename.side_effect = OSError("Rename failed")

        with pytest.raises(OSError):
            run_config_wizard(tmp_path, Console())

        # Verify no temp files left behind
        temp_files = list(tmp_path.glob(".bmad-assist-*.yaml.tmp"))
        assert len(temp_files) == 0

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_os_error_propagates(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """AC11: OSError on save propagates to caller."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        # Make directory read-only to cause write failure
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        os.chmod(read_only_dir, 0o444)

        try:
            with pytest.raises(OSError):
                run_config_wizard(read_only_dir, Console())
        finally:
            # Restore permissions for cleanup
            os.chmod(read_only_dir, 0o755)


# =============================================================================
# Test: run_config_wizard Function
# =============================================================================


class TestRunConfigWizard:
    """Tests for run_config_wizard function."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_returns_path_to_config(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """run_config_wizard returns path to generated config."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        result = run_config_wizard(tmp_path, Console())

        assert isinstance(result, Path)
        assert result.name == CONFIG_FILENAME
        assert result.parent == tmp_path

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_accepts_none_console(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """run_config_wizard works with console=None."""
        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        # Should not raise
        result = run_config_wizard(tmp_path)

        assert result.exists()


# =============================================================================
# Test: Display Methods (Smoke Tests)
# =============================================================================


class TestDisplayMethods:
    """Smoke tests for display methods."""

    def test_display_welcome_does_not_raise(self) -> None:
        """_display_welcome executes without error."""
        generator = ConfigGenerator(Console())
        # Should not raise
        generator._display_welcome()

    def test_display_summary_does_not_raise(self) -> None:
        """_display_summary executes without error."""
        generator = ConfigGenerator(Console())
        config = generator._build_config("claude", "opus_4")
        # Should not raise
        generator._display_summary(config)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_different_model_selections(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Can select non-default model."""
        monkeypatch.chdir(tmp_path)
        mock_prompt.side_effect = ["claude", "haiku_3_5"]
        mock_confirm.return_value = True

        run_config_wizard(tmp_path, Console())

        from bmad_assist.core.config import load_config_with_project

        config = load_config_with_project(tmp_path)
        assert config.providers.master.model == "haiku_3_5"

    @patch("bmad_assist.core.config_generator.Prompt.ask")
    @patch("bmad_assist.core.config_generator.Confirm.ask")
    def test_config_in_subdirectory(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """Config can be generated in subdirectory."""
        subdir = tmp_path / "my-project"
        subdir.mkdir()

        mock_prompt.side_effect = ["claude", "opus_4"]
        mock_confirm.return_value = True

        config_path = run_config_wizard(subdir, Console())

        assert config_path.parent == subdir
        assert config_path.exists()
