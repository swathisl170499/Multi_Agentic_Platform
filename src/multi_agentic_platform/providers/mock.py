from multi_agentic_platform.providers.base import LLMProvider


class MockProvider(LLMProvider):
    async def generate(self, system: str, prompt: str) -> str:
        return (
            "# Mock output\n"
            "def example():\n"
            "    return 'Replace with real model output'\n"
        )
