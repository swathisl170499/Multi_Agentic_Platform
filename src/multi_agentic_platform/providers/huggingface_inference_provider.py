from __future__ import annotations

from multi_agentic_platform.config import settings
from multi_agentic_platform.providers.base import LLMProvider


class HuggingFaceInferenceProvider(LLMProvider):
    """Hosted open-source inference provider via Hugging Face Inference API."""

    def __init__(self) -> None:
        if not settings.hf_api_token:
            raise ValueError("MAP_HF_API_TOKEN is required for provider=hf_inference")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError("huggingface_hub is required. Install: pip install huggingface_hub") from exc

        self._client = InferenceClient(token=settings.hf_api_token)
        self._model = settings.hf_inference_model

    async def generate(self, system: str, prompt: str) -> str:
        full_prompt = (
            f"<|system|>\n{system}\n</s>\n<|user|>\n{prompt}\n</s>\n<|assistant|>\n"
        )
        text = self._client.text_generation(
            prompt=full_prompt,
            model=self._model,
            max_new_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        return str(text).strip()
