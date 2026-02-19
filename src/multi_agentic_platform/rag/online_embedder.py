from __future__ import annotations

from multi_agentic_platform.config import settings


class HuggingFaceAPIEmbedder:
    """Embedding client via HF Inference API (no local model downloads)."""

    def __init__(self, model_name: str) -> None:
        if not settings.hf_api_token:
            raise ValueError("MAP_HF_API_TOKEN is required when MAP_RAG_USE_ONLINE_INFERENCE=true")
        self._model = model_name
        self._token = settings.hf_api_token

    def encode(self, texts: list[str]):
        try:
            import numpy as np
            import httpx
        except ImportError as exc:
            raise ImportError("numpy and httpx are required for online embeddings") from exc

        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self._model}"
        headers = {"Authorization": f"Bearer {self._token}"}
        vectors = []
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                response = client.post(url, headers=headers, json={"inputs": text})
                response.raise_for_status()
                payload = response.json()
                if payload and isinstance(payload[0], list):
                    token_vectors = np.asarray(payload, dtype="float32")
                    emb = token_vectors.mean(axis=0)
                else:
                    emb = np.asarray(payload, dtype="float32")
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
                vectors.append(emb)

        return np.asarray(vectors, dtype="float32")
