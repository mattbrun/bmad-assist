"""Pydantic configuration models for bmad-assist.

This package contains all configuration model classes organized by domain.
All models are re-exported here for convenient imports.
"""

from bmad_assist.core.config.models.features import (
    AntipatternConfig,
    BenchmarkingConfig,
    CompilerConfig,
    PlaywrightConfig,
    PlaywrightServerConfig,
    QAConfig,
    TimeoutsConfig,
)
from bmad_assist.core.config.models.loop import (
    DEFAULT_LOOP_CONFIG,
    LoopConfig,
    SprintConfig,
    WarningsConfig,
)
from bmad_assist.core.config.models.main import Config
from bmad_assist.core.config.models.paths import (
    BmadPathsConfig,
    PowerPromptConfig,
    ProjectPathsConfig,
)
from bmad_assist.core.config.models.providers import (
    HelperProviderConfig,
    MasterProviderConfig,
    MultiProviderConfig,
    ProviderConfig,
)
from bmad_assist.core.config.models.source_context import (
    SourceContextBudgetsConfig,
    SourceContextConfig,
    SourceContextExtractionConfig,
    SourceContextScoringConfig,
)
from bmad_assist.core.config.models.strategic_context import (
    StrategicContextConfig,
    StrategicContextDefaultsConfig,
    StrategicContextWorkflowConfig,
    StrategicDocType,
    _create_story_defaults,
    _validate_story_defaults,
    _validate_story_synthesis_defaults,
)

__all__ = [
    # providers.py
    "MasterProviderConfig",
    "MultiProviderConfig",
    "HelperProviderConfig",
    "ProviderConfig",
    # paths.py
    "PowerPromptConfig",
    "BmadPathsConfig",
    "ProjectPathsConfig",
    # source_context.py
    "SourceContextBudgetsConfig",
    "SourceContextScoringConfig",
    "SourceContextExtractionConfig",
    "SourceContextConfig",
    # strategic_context.py
    "StrategicDocType",
    "StrategicContextDefaultsConfig",
    "StrategicContextWorkflowConfig",
    "StrategicContextConfig",
    "_create_story_defaults",
    "_validate_story_defaults",
    "_validate_story_synthesis_defaults",
    # features.py
    "AntipatternConfig",
    "CompilerConfig",
    "TimeoutsConfig",
    "BenchmarkingConfig",
    "PlaywrightServerConfig",
    "PlaywrightConfig",
    "QAConfig",
    # loop.py
    "LoopConfig",
    "SprintConfig",
    "WarningsConfig",
    "DEFAULT_LOOP_CONFIG",
    # main.py
    "Config",
]
