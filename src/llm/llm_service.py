"""Provider-agnostic LLM service for chat and embeddings.

This module provides a unified interface to any LLM provider supported by
LangChain (OpenAI, Anthropic, Google, corporate proxies like LiteLLM, etc.).

Architecture pattern: Port/Adapter (Hexagonal Architecture)
- Port: LLMService defines the interface the application uses
- Adapter: LangChain's init_chat_model/init_embeddings handle provider specifics
"""

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import Settings, get_settings
from model import (
    Conversation,
    LLMResponse,
    Role,
    StopReason,
    TokenUsage,
)


class LLMService:
    """Provider-agnostic LLM service for chat completions and embeddings.

    Wraps LangChain's generic model initialization to provide a clean,
    typed interface that integrates with our domain models.

    Usage:
        service = LLMService(settings)
        response = await service.complete("Hello, who are you?")
        vectors = await service.embed_texts(["Hello", "World"])
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the LLM service.

        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._chat_model: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None

    # ------------------------------------------------------------------
    # Lazy-initialized models (created on first use)
    # ------------------------------------------------------------------

    @property
    def chat_model(self) -> BaseChatModel:
        """Return the chat model, initializing on first access."""
        if self._chat_model is None:
            self._chat_model = self._init_chat_model()
        return self._chat_model

    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings model, initializing on first access.

        Raises:
            ValueError: If no embedding model is configured.
        """
        if self._embeddings is None:
            self._embeddings = self._init_embeddings()
        return self._embeddings

    # ------------------------------------------------------------------
    # Chat Completion Methods
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a single prompt and return a structured response.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system instruction.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Structured LLMResponse with content, usage, and metadata.
        """
        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return await self._invoke(messages, temperature=temperature, max_tokens=max_tokens)

    async def chat(
        self,
        conversation: Conversation,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a full conversation and return a structured response.

        Args:
            conversation: Conversation with message history.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Structured LLMResponse with content, usage, and metadata.
        """
        messages = self._conversation_to_langchain(conversation)
        return await self._invoke(messages, temperature=temperature, max_tokens=max_tokens)

    def complete_sync(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Synchronous version of complete() for non-async contexts.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system instruction.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Structured LLMResponse with content, usage, and metadata.
        """
        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return self._invoke_sync(messages, temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Embedding Methods
    # ------------------------------------------------------------------

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Optimized for search queries (some providers use different
        models/strategies for queries vs documents).

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return await self.embeddings.aembed_query(text)

    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous version of embed_texts().

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)

    def embed_query_sync(self, text: str) -> list[float]:
        """Synchronous version of embed_query().

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embeddings.embed_query(text)

    # ------------------------------------------------------------------
    # Private: Model Initialization
    # ------------------------------------------------------------------

    def _init_chat_model(self) -> BaseChatModel:
        """Initialize the chat model using LangChain's generic factory.

        Uses init_chat_model which supports any provider:
        OpenAI, Anthropic, Google, Ollama, Azure, corporate proxies, etc.
        """
        kwargs: dict[str, object] = {
            "model": self._settings.model_name,
            "model_provider": self._settings.model_provider,
            "api_key": self._settings.api_key,
            "temperature": self._settings.default_temperature,
            "max_tokens": self._settings.default_max_tokens,
        }

        if self._settings.base_url:
            kwargs["base_url"] = self._settings.base_url

        return init_chat_model(**kwargs)  # type: ignore[arg-type]

    def _init_embeddings(self) -> Embeddings:
        """Initialize the embeddings model using LangChain's generic factory.

        Uses init_embeddings which supports any provider.

        Raises:
            ValueError: If no embedding model is configured in settings.
        """
        if not self._settings.embedding_model_name:
            msg = (
                "No embedding model configured. "
                "Set EMBEDDING_MODEL_NAME in your .env file "
                "(e.g., 'openai:text-embedding-3-small')."
            )
            raise ValueError(msg)

        return init_embeddings(
            model=f"{self._settings.embedding_provider}:{self._settings.embedding_model_name}",
        )

    # ------------------------------------------------------------------
    # Private: Message Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[SystemMessage | HumanMessage]:
        """Build LangChain message list from a simple prompt."""
        messages: list[SystemMessage | HumanMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    @staticmethod
    def _conversation_to_langchain(
        conversation: Conversation,
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """Convert our Conversation model to LangChain message format."""
        messages: list[SystemMessage | HumanMessage | AIMessage] = []

        if conversation.system_prompt:
            messages.append(SystemMessage(content=conversation.system_prompt))

        for msg in conversation.messages:
            match msg.role:
                case Role.SYSTEM:
                    messages.append(SystemMessage(content=msg.content))
                case Role.USER:
                    messages.append(HumanMessage(content=msg.content))
                case Role.ASSISTANT:
                    messages.append(AIMessage(content=msg.content))

        return messages

    # ------------------------------------------------------------------
    # Private: Invocation
    # ------------------------------------------------------------------

    async def _invoke(
        self,
        messages: list[SystemMessage | HumanMessage | AIMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Invoke the chat model asynchronously and return structured response."""
        kwargs = self._build_invoke_kwargs(temperature, max_tokens)
        result = await self.chat_model.ainvoke(messages, **kwargs)
        return self._parse_response(result)

    def _invoke_sync(
        self,
        messages: list[SystemMessage | HumanMessage | AIMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Invoke the chat model synchronously and return structured response."""
        kwargs = self._build_invoke_kwargs(temperature, max_tokens)
        result = self.chat_model.invoke(messages, **kwargs)
        return self._parse_response(result)

    def _build_invoke_kwargs(
        self,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, object]:
        """Build kwargs for model invocation with optional overrides."""
        kwargs: dict[str, object] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    @staticmethod
    def _parse_response(result: AIMessage) -> LLMResponse:
        """Parse LangChain AIMessage into our domain LLMResponse."""
        # Extract usage metadata (provider-agnostic)
        usage_meta = getattr(result, "usage_metadata", None) or {}
        input_tokens = usage_meta.get("input_tokens", 0)
        output_tokens = usage_meta.get("output_tokens", 0)

        # Extract stop reason
        response_meta = getattr(result, "response_metadata", None) or {}
        raw_stop = response_meta.get("stop_reason") or response_meta.get(
            "finish_reason"
        )

        stop_reason = None
        if raw_stop:
            stop_map = {
                "end_turn": StopReason.END_TURN,
                "stop": StopReason.END_TURN,
                "max_tokens": StopReason.MAX_TOKENS,
                "length": StopReason.MAX_TOKENS,
                "stop_sequence": StopReason.STOP_SEQUENCE,
                "tool_use": StopReason.TOOL_USE,
                "tool_calls": StopReason.TOOL_USE,
            }
            stop_reason = stop_map.get(raw_stop)

        # Extract model name
        model_name = response_meta.get("model_name", response_meta.get("model", "unknown"))

        return LLMResponse(
            content=str(result.content),
            model=str(model_name),
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# Module-level convenience accessor
# ---------------------------------------------------------------------------
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Return the singleton LLMService instance.

    Uses the global Settings from get_settings().
    """
    global _llm_service  # noqa: PLW0603
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
