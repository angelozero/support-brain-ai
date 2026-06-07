"""Tests for centralized configuration."""

import pytest

from config import Settings


class TestSettings:
    """Validate Settings loading and validation.

    Note: Tests use _env_file=None to prevent loading from .env file,
    ensuring test isolation (similar to @TestPropertySource in Spring).
    """

    def _make_settings(self, **kwargs) -> Settings:
        """Create Settings with .env loading disabled for test isolation."""
        defaults = {"model_name": "test-model", "api_key": "test-key"}
        defaults.update(kwargs)
        return Settings(_env_file=None, **defaults)  # type: ignore[call-arg]

    def test_loads_with_defaults(self, settings: Settings) -> None:
        """Settings should load with sensible defaults."""
        assert settings.model_name == "test-model"
        assert settings.model_provider == "openai"
        assert settings.default_temperature == 0.0
        assert settings.default_max_tokens == 4096
        assert settings.app_name == "SupportBrainAI"
        assert settings.debug is False

    def test_requires_model_name(self) -> None:
        """Settings must reject missing model_name."""
        with pytest.raises(Exception):  # noqa: B017
            Settings(_env_file=None, api_key="test")  # type: ignore[call-arg]

    def test_requires_api_key(self) -> None:
        """Settings must reject missing API key."""
        with pytest.raises(Exception):  # noqa: B017
            Settings(_env_file=None, model_name="test")  # type: ignore[call-arg]

    def test_validates_temperature_upper_bound(self) -> None:
        """Temperature above 2.0 must be rejected."""
        with pytest.raises(Exception):  # noqa: B017
            self._make_settings(default_temperature=3.0)

    def test_validates_temperature_lower_bound(self) -> None:
        """Temperature below 0.0 must be rejected."""
        with pytest.raises(Exception):  # noqa: B017
            self._make_settings(default_temperature=-0.1)

    def test_validates_max_tokens_positive(self) -> None:
        """max_tokens must be positive."""
        with pytest.raises(Exception):  # noqa: B017
            self._make_settings(default_max_tokens=0)

    def test_custom_values(self) -> None:
        """Settings should accept custom values within valid ranges."""
        s = self._make_settings(
            model_name="anthropic:/claude-haiku-4-20250414",
            model_provider="anthropic",
            api_key="sk-ant-custom",
            base_url="https://proxy.example.com",
            default_temperature=0.3,
            default_max_tokens=2048,
            embedding_model_name="text-embedding-3-small",
            embedding_provider="openai",
            debug=True,
        )
        assert s.model_name == "anthropic:/claude-haiku-4-20250414"
        assert s.model_provider == "anthropic"
        assert s.base_url == "https://proxy.example.com"
        assert s.default_temperature == 0.3
        assert s.default_max_tokens == 2048
        assert s.embedding_model_name == "text-embedding-3-small"
        assert s.debug is True

    def test_base_url_optional(self) -> None:
        """base_url should default to None when not provided."""
        s = self._make_settings()
        assert s.base_url is None

    def test_embedding_model_optional(self) -> None:
        """embedding_model_name should default to None when not provided."""
        s = self._make_settings()
        assert s.embedding_model_name is None
