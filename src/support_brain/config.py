"""Centralized configuration using Pydantic Settings.

Analogous to Spring's @ConfigurationProperties + @Validated.
All settings are loaded from environment variables or .env file,
with automatic type validation on startup.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuração centralizada do SupportBrainAI.

    Todas as variáveis são carregadas de:
    1. Variáveis de ambiente do sistema (prioridade máxima)
    2. Arquivo .env (fallback)
    3. Valores default definidos aqui (fallback final)
    """

    # --- LLM ---
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default LLM model identifier",
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses",
    )
    default_max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in LLM response",
    )

    # --- Application ---
    app_name: str = Field(default="SupportBrainAI", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# ---------------------------------------------------------------------------
# Singleton accessor — lazy initialization
# ---------------------------------------------------------------------------
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance, creating it on first call.

    Raises:
        ValidationError: If required env vars are missing or invalid.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
