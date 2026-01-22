from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, system: str, prompt: str) -> str:
        raise NotImplementedError
