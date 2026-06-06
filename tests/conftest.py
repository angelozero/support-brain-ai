"""Shared test fixtures for SupportBrainAI.

Analogous to Spring's @TestConfiguration — centralized test setup
that all test modules can reuse.
"""

import pytest

from support_brain.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Provide a Settings instance with test defaults.

    Uses a fake API key so tests don't require real credentials.
    """
    return Settings(anthropic_api_key="sk-ant-test-key-for-testing")  # type: ignore[call-arg]
