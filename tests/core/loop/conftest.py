"""Pytest fixtures for bmad_assist.core.loop tests.

Shared fixtures extracted from test_loop.py as part of loop.py refactor.
"""

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from bmad_assist.core.loop.signals import reset_shutdown


@pytest.fixture(autouse=True)
def reset_shutdown_state() -> None:
    """Reset shutdown state before and after each test.

    This fixture ensures test isolation by clearing the shutdown state.
    The autouse=True makes it run automatically for all tests
    in the loop directory.
    """
    reset_shutdown()
    yield
    reset_shutdown()


@pytest.fixture(autouse=True)
def auto_continue_prompts() -> Iterator[None]:
    """Auto-continue all interactive prompts in tests.

    This fixture patches checkpoint_and_prompt to always return True,
    simulating user pressing Enter to continue. Tests that need to
    test the prompt behavior should patch it explicitly.
    """
    with patch(
        "bmad_assist.core.loop.runner.checkpoint_and_prompt",
        return_value=True,
    ):
        yield
