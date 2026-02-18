from __future__ import annotations

from multi_agentic_platform.config import settings
from multi_agentic_platform.providers.base import LLMProvider


class HuggingFaceProvider(LLMProvider):
    """Local open-source model provider via transformers pipeline."""

    def __init__(self) -> None:
        model_name = settings.hf_model
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("transformers is required. Install: pip install transformers") from exc

        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map="auto",
        )

    async def generate(self, system: str, prompt: str) -> str:
        full_prompt = f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
        out = self._pipe(full_prompt, max_new_tokens=settings.max_tokens, temperature=settings.temperature)
        text = out[0].get("generated_text", "")
        return text[len(full_prompt) :].strip() if text.startswith(full_prompt) else text
