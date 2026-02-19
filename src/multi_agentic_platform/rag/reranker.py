from __future__ import annotations

from dataclasses import dataclass

from multi_agentic_platform.providers.base import LLMProvider


@dataclass
class Candidate:
    chunk_id: int
    text: str
    retrieval_score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. Install with: pip install sentence-transformers"
            ) from exc

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Candidate], top_k: int = 5) -> list[Candidate]:
        if not candidates:
            return []

        pairs = [[query, candidate.text] for candidate in candidates]
        scores = self._model.predict(pairs)
        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        return [item[0] for item in scored[:top_k]]


class LLMRerankerAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def rerank(self, query: str, candidates: list[Candidate], top_k: int = 5) -> list[Candidate]:
        if not candidates:
            return []

        indexed_candidates = "\n\n".join(
            f"[{idx}] retrieval_score={candidate.retrieval_score:.4f}\n{candidate.text}"
            for idx, candidate in enumerate(candidates)
        )
        prompt = (
            "Rank the candidate chunks for relevance to the query. Return only comma-separated "
            f"indexes of the best {top_k} chunks in descending order.\n\n"
            f"Query:\n{query}\n\nCandidates:\n{indexed_candidates}"
        )
        output = await self._provider.generate(
            system=(
                "You are a retrieval reranking agent. Respond only with comma-separated indexes "
                "such as: 2,0,3"
            ),
            prompt=prompt,
        )

        chosen: list[Candidate] = []
        for raw in output.replace(" ", "").split(","):
            if not raw.isdigit():
                continue
            idx = int(raw)
            if 0 <= idx < len(candidates):
                candidate = candidates[idx]
                if candidate not in chosen:
                    chosen.append(candidate)
            if len(chosen) >= top_k:
                break

        return chosen or candidates[:top_k]
