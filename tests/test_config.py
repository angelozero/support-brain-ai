"""Tests for centralized configuration."""

import pytest

from support_brain.config import Settings


class TestSettings:
    """Validate Settings loading and validation."""

    def test_loads_with_defaults(self, settings: Settings) -> None:
        """Settings should load with sensible defaults."""
        assert settings.default_model == "claude-sonnet-4-20250514"
        assert settings.default_temperature == 0.7
        assert settings.default_max_tokens == 4096
        assert settings.app_name == "SupportBrainAI"
        assert settings.debug is False

    def test_requires_api_key(self) -> None:
        """Settings must reject missing API key."""
        with pytest.raises(Exception):  # noqa: B017
            Settings()  # type: ignore[call-arg]

    def test_validates_temperature_upper_bound(self) -> None:
        """Temperature above 2.0 must be rejected."""
        with pytest.raises(Exception):  # noqa: B017
            Settings(anthropic_api_key="test", default_temperature=3.0)  # type: ignore[call-arg]

    def test_validates_temperature_lower_bound(self) -> None:
        """Temperature below 0.0 must be rejected."""
        with pytest.raises(Exception):  # noqa: B017
            Settings(anthropic_api_key="test", default_temperature=-0.1)  # type: ignore[call-arg]

    def test_validates_max_tokens_positive(self) -> None:
        """max_tokens must be positive."""
        with pytest.raises(Exception):  # noqa: B017
            Settings(anthropic_api_key="test", default_max_tokens=0)  # type: ignore[call-arg]

    def test_custom_values(self) -> None:
        """Settings should accept custom values within valid ranges."""
        s = Settings(
            anthropic_api_key="sk-ant-custom",
            default_model="claude-haiku-4-20250414",
            default_temperature=0.3,
            default_max_tokens=2048,
            debug=True,
        )  # type: ignore[call-arg]
        assert s.default_model == "claude-haiku-4-20250414"
        assert s.default_temperature == 0.3
        assert s.default_max_tokens == 2048
        assert s.debug is True
