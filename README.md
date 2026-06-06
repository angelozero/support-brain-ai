# 🧠 SupportBrainAI

AI-powered customer support platform with RAG, conversational agents, and full observability.

## Architecture

```
src/support_brain/
├── config.py           ← Centralized configuration (Pydantic Settings)
├── models.py           ← Shared domain models
├── agent/              ← LangGraph agents, tools, skills, hooks
├── rag/                ← RAG pipeline, embeddings, HyDE, vector stores
├── memory/             ← Long-term episodic & semantic memory
├── gateway/            ← AI Gateway (LiteLLM), rate limiting, fallback
├── pii/                ← PII detection, anonymization, prompt injection shield
├── eval/               ← LLM-as-a-Judge, MLflow, evaluation harness
└── observability/      ← OpenTelemetry, LangSmith, event streaming
```

## Setup

```bash
# Install dependencies
uv sync

# Copy environment template and fill in your API keys
cp .env.example .env

# Run tests
uv run pytest

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Tech Stack

- **Python 3.14** with modern type hints
- **UV** for package management
- **Pydantic** for validation and settings
- **Anthropic Claude** as primary LLM provider
- **Pytest** for testing
- **Ruff** for linting and formatting
