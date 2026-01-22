from dataclasses import dataclass

from multi_agentic_platform.providers.base import LLMProvider


@dataclass
class Agent:
    name: str
    system_prompt: str
    provider: LLMProvider

    async def act(self, prompt: str) -> str:
        return await self.provider.generate(self.system_prompt, prompt)
