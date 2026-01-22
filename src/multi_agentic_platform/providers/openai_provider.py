from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agentic_platform.config import settings
from multi_agentic_platform.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("MAP_OPENAI_API_KEY is required for the OpenAI provider.")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def generate(self, system: str, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        return response.choices[0].message.content or ""
