"""Shared domain models for SupportBrainAI.

These models represent the core data structures used across services.
Built with Pydantic for automatic validation and serialization.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Role(StrEnum):
    """Message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class StopReason(StrEnum):
    """Reasons why the LLM stopped generating."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


# ---------------------------------------------------------------------------
# Message Models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Token usage statistics from an LLM response."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider.

    This abstraction allows us to swap providers (Anthropic, OpenAI, etc.)
    without changing downstream code — similar to a port/adapter pattern.
    """

    content: str
    model: str
    usage: TokenUsage
    stop_reason: StopReason | None = None


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
class Conversation(BaseModel):
    """A conversation with message history."""

    messages: list[Message] = Field(default_factory=list)
    system_prompt: str | None = None

    def add_message(self, role: Role, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))

    def to_api_messages(self) -> list[dict[str, str]]:
        """Convert to the format expected by LLM APIs."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
