"""LLM service — provider-agnostic chat and embedding models."""

from .llm_service import LLMService, get_llm_service

__all__ = ["LLMService", "get_llm_service"]
