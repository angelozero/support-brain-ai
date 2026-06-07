"""Interactive chat demo using the NovaMind support agent prompt.

Demonstrates:
- Conversation model for maintaining chat history (LLMs are stateless)
- System prompt injection via Conversation.system_prompt
- Async chat loop with proper await
- Graceful exit with 'sair' or Ctrl+C

Usage:
    uv run python scripts/demo_chat.py
"""

import asyncio

from llm.llm_service import LLMService
from model.domain import Conversation, Role
from prompts.prompt_virtual_assistance import PROMPT_VIRTUAL_ASSISTANCE


async def chat_loop() -> None:
    """Run an interactive chat session with the NovaMind support agent."""
    llm = LLMService()

    conversation = Conversation(system_prompt=PROMPT_VIRTUAL_ASSISTANCE)

    print("=" * 60)
    print("🧠 NovaMind — Assistente Virtual de Suporte")
    print("   Digite 'sair' para encerrar.")
    print("=" * 60)

    while True:
        user_input = input("\n👤 Você: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "sair":
            print("\n👋 Até logo! Obrigado por usar o suporte NovaMind.")
            break

        conversation.add_message(Role.USER, user_input)

        response = await llm.chat(conversation)

        conversation.add_message(Role.ASSISTANT, response.content)

        print(f"\n🤖 NovaMind: {response.content}")
        print(f"   [tokens: {response.usage.input_tokens}↓ {response.usage.output_tokens}↑ | modelo: {response.model}]")


def main() -> None:
    """Entry point — runs the async chat loop."""
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n\n👋 Sessão encerrada. Até logo!")


if __name__ == "__main__":
    main()
