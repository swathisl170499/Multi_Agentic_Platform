from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredChunk:
    chunk_id: int
    score: float


class FaissStore:
    def __init__(self, dimension: int) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

        self._index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings) -> None:
        import numpy as np

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self._index.add(embeddings)

    def search(self, query_embedding, top_k: int) -> list[ScoredChunk]:
        import numpy as np

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")

        scores, indices = self._index.search(query_embedding, top_k)
        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(ScoredChunk(chunk_id=int(idx), score=float(score)))
        return results

    @property
    def size(self) -> int:
        return int(self._index.ntotal)
