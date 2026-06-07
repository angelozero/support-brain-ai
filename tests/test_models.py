"""Tests for shared domain models."""

from model.models import (
    Conversation,
    LLMResponse,
    Message,
    Role,
    StopReason,
    TokenUsage,
)


class TestMessage:
    """Validate Message model."""

    def test_create_user_message(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_assistant_message(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="Hi there!")
        assert msg.role == "assistant"


class TestTokenUsage:
    """Validate TokenUsage model."""

    def test_total_tokens(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_rejects_negative_tokens(self) -> None:
        import pytest

        with pytest.raises(Exception):  # noqa: B017
            TokenUsage(input_tokens=-1, output_tokens=10)


class TestLLMResponse:
    """Validate LLMResponse model."""

    def test_create_response(self) -> None:
        response = LLMResponse(
            content="Hello!",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            stop_reason=StopReason.END_TURN,
        )
        assert response.content == "Hello!"
        assert response.usage.total_tokens == 15
        assert response.stop_reason == "end_turn"

    def test_response_without_stop_reason(self) -> None:
        response = LLMResponse(
            content="Hi",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=5, output_tokens=3),
        )
        assert response.stop_reason is None


class TestConversation:
    """Validate Conversation model."""

    def test_empty_conversation(self) -> None:
        conv = Conversation()
        assert conv.messages == []
        assert conv.system_prompt is None

    def test_add_messages(self) -> None:
        conv = Conversation(system_prompt="You are a helpful assistant.")
        conv.add_message(Role.USER, "Hello")
        conv.add_message(Role.ASSISTANT, "Hi! How can I help?")

        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER
        assert conv.messages[1].role == Role.ASSISTANT

    def test_to_api_messages(self) -> None:
        conv = Conversation()
        conv.add_message(Role.USER, "Test")

        api_msgs = conv.to_api_messages()
        assert api_msgs == [{"role": "user", "content": "Test"}]
