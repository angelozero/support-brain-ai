"""Quick connection test for the LLM proxy.

Run with:
    uv run python scripts/test_connection.py

This script validates that:
1. Settings load correctly from .env
2. The LLM proxy is reachable
3. A basic completion works end-to-end
"""

import asyncio
import sys

from config.config import get_settings
from llm import LLMService


async def test_connection() -> None:
    """Test the LLM connection with a simple prompt."""
    print("=" * 60)
    print("🧪 SupportBrainAI — Connection Test")
    print("=" * 60)

    # 1. Load settings
    print("\n📋 Loading settings...")
    try:
        settings = get_settings()
        print(f"   ✅ Model:       {settings.model_name}")
        print(f"   ✅ Provider:    {settings.model_provider}")
        print(f"   ✅ Base URL:    {settings.base_url or '(default)'}")
        print(f"   ✅ Temperature: {settings.default_temperature}")
        print(f"   ✅ Max Tokens:  {settings.default_max_tokens}")
        print(f"   ✅ API Key:     {settings.api_key[:20]}...")
    except Exception as e:
        print(f"   ❌ Failed to load settings: {e}")
        sys.exit(1)

    # 2. Initialize LLM Service
    print("\n🔌 Initializing LLM Service...")
    service = LLMService(settings)

    # 3. Make a simple completion
    print("\n💬 Sending test prompt: 'Say hello in one sentence.'")
    try:
        response = await service.complete(
            "Say hello in one sentence.",
            system_prompt="You are a test assistant. Be brief.",
        )
        print(f"\n   ✅ Response:    {response.content}")
        print(f"   📊 Model:      {response.model}")
        print(f"   📊 Tokens:     {response.usage.input_tokens} in / "
              f"{response.usage.output_tokens} out / "
              f"{response.usage.total_tokens} total")
        print(f"   📊 Stop:       {response.stop_reason}")
    except Exception as e:
        print(f"\n   ❌ Failed: {type(e).__name__}: {e}")
        print("\n   💡 Troubleshooting tips:")
        print("      - Check MODEL_NAME in .env (try without prefix, e.g., 'gpt-4o-mini')")
        print("      - Check BASE_URL is correct and accessible")
        print("      - Check API_KEY is valid and not expired")
        print("      - Try: curl -X POST <BASE_URL>/chat/completions ...")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Connection test passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_connection())
