from __future__ import annotations


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding. Install with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for embeddings. Install with: pip install numpy") from exc

        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")
