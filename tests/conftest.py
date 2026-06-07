"""Shared test fixtures for SupportBrainAI.

Analogous to Spring's @TestConfiguration — centralized test setup
that all test modules can reuse.
"""

import pytest

from config.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Provide a Settings instance with test defaults.

    Uses _env_file=None to prevent loading from .env file,
    ensuring test isolation regardless of local environment.
    """
    return Settings(
        _env_file=None,
        model_name="test-model",
        model_provider="openai",
        api_key="test-api-key-for-testing",
        base_url=None,
        default_temperature=0.0,
        default_max_tokens=4096,
    )  # type: ignore[call-arg]
