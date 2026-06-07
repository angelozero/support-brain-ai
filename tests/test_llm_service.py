"""Tests for the LLM service."""

import pytest

from config.config import Settings
from llm.llm_service import LLMService
from model.models import Conversation, Role


class TestLLMServiceInit:
    """Validate LLMService initialization and configuration."""

    def test_creates_with_settings(self, settings: Settings) -> None:
        """LLMService should accept explicit settings."""
        service = LLMService(settings)
        assert service._settings is settings

    def test_creates_with_default_settings(self, monkeypatch) -> None:
        """LLMService should use get_settings() when no settings provided."""
        monkeypatch.setenv("MODEL_NAME", "test-model")
        monkeypatch.setenv("API_KEY", "test-key")

        # Reset singleton
        import config.config as config_mod
        config_mod._settings = None

        service = LLMService()
        assert service._settings.model_name == "test-model"

        # Cleanup
        config_mod._settings = None

    def test_chat_model_lazy_init(self, settings: Settings) -> None:
        """Chat model should not be initialized until first access."""
        service = LLMService(settings)
        assert service._chat_model is None

    def test_embeddings_lazy_init(self, settings: Settings) -> None:
        """Embeddings model should not be initialized until first access."""
        service = LLMService(settings)
        assert service._embeddings is None

    def test_embeddings_raises_without_config(self, settings: Settings) -> None:
        """Accessing embeddings without config should raise ValueError."""
        service = LLMService(settings)
        with pytest.raises(ValueError, match="No embedding model configured"):
            _ = service.embeddings


class TestLLMServiceMessageBuilding:
    """Validate message conversion logic (no API calls needed)."""

    def test_build_messages_simple(self) -> None:
        """Should build a single HumanMessage."""
        messages = LLMService._build_messages("Hello")
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    def test_build_messages_with_system(self) -> None:
        """Should prepend SystemMessage when system_prompt provided."""
        messages = LLMService._build_messages(
            "Hello",
            system_prompt="You are helpful.",
        )
        assert len(messages) == 2
        assert messages[0].content == "You are helpful."
        assert messages[1].content == "Hello"

    def test_conversation_to_langchain_empty(self) -> None:
        """Empty conversation should produce empty list."""
        conv = Conversation()
        messages = LLMService._conversation_to_langchain(conv)
        assert messages == []

    def test_conversation_to_langchain_with_system(self) -> None:
        """Conversation with system prompt should include SystemMessage."""
        conv = Conversation(system_prompt="Be concise.")
        conv.add_message(Role.USER, "Hi")
        conv.add_message(Role.ASSISTANT, "Hello!")

        messages = LLMService._conversation_to_langchain(conv)
        assert len(messages) == 3
        assert messages[0].content == "Be concise."
        assert messages[1].content == "Hi"
        assert messages[2].content == "Hello!"

    def test_conversation_to_langchain_role_mapping(self) -> None:
        """Each Role should map to the correct LangChain message type."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        conv = Conversation()
        conv.add_message(Role.SYSTEM, "System msg")
        conv.add_message(Role.USER, "User msg")
        conv.add_message(Role.ASSISTANT, "AI msg")

        messages = LLMService._conversation_to_langchain(conv)
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)


class TestLLMServiceResponseParsing:
    """Validate response parsing logic (no API calls needed)."""

    def test_parse_response_basic(self) -> None:
        """Should parse a basic AIMessage into LLMResponse."""
        from langchain_core.messages import AIMessage

        result = AIMessage(
            content="Hello!",
            response_metadata={"model_name": "gpt-4o-mini"},
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )

        response = LLMService._parse_response(result)
        assert response.content == "Hello!"
        assert response.model == "gpt-4o-mini"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15

    def test_parse_response_stop_reason_mapping(self) -> None:
        """Should map various stop reasons correctly."""
        from langchain_core.messages import AIMessage

        # OpenAI-style "stop"
        result = AIMessage(
            content="Done",
            response_metadata={"finish_reason": "stop", "model": "gpt-4o"},
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )
        response = LLMService._parse_response(result)
        assert response.stop_reason == "end_turn"

        # Anthropic-style "end_turn"
        result2 = AIMessage(
            content="Done",
            response_metadata={"stop_reason": "end_turn", "model_name": "claude-3"},
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )
        response2 = LLMService._parse_response(result2)
        assert response2.stop_reason == "end_turn"

    def test_parse_response_no_metadata(self) -> None:
        """Should handle missing metadata gracefully."""
        from langchain_core.messages import AIMessage

        result = AIMessage(content="Hello")
        response = LLMService._parse_response(result)
        assert response.content == "Hello"
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0
        assert response.stop_reason is None
