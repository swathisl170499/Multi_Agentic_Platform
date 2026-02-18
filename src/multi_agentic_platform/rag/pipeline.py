from __future__ import annotations

from dataclasses import dataclass

from multi_agentic_platform.config import settings
from multi_agentic_platform.providers.base import LLMProvider
from multi_agentic_platform.rag.chunking import chunk_text
from multi_agentic_platform.rag.embedder import SentenceTransformerEmbedder
from multi_agentic_platform.rag.loaders import load_document
from multi_agentic_platform.rag.reranker import Candidate, CrossEncoderReranker, LLMRerankerAgent
from multi_agentic_platform.rag.vector_store import FaissStore


@dataclass
class ChunkRecord:
    chunk_id: int
    source: str
    text: str


class RAGPipeline:
    def __init__(self, rerank_with_agent_provider: LLMProvider | None = None) -> None:
        self._embedder = SentenceTransformerEmbedder(settings.rag_embedding_model)
        self._reranker = CrossEncoderReranker(settings.rag_reranker_model)
        self._agent_reranker = (
            LLMRerankerAgent(rerank_with_agent_provider) if rerank_with_agent_provider else None
        )

        self._store: FaissStore | None = None
        self._chunks: list[ChunkRecord] = []

    def ingest_paths(self, paths: list[str]) -> dict[str, int]:
        documents: list[tuple[str, str]] = [load_document(path) for path in paths]
        return self.ingest_documents(documents)

    def ingest_documents(self, documents: list[tuple[str, str]]) -> dict[str, int]:
        added_chunks = 0

        for source, content in documents:
            chunks = chunk_text(
                content,
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )
            if not chunks:
                continue

            embeddings = self._embedder.encode(chunks)
            if self._store is None:
                self._store = FaissStore(dimension=int(embeddings.shape[1]))
            self._store.add(embeddings)

            for chunk in chunks:
                self._chunks.append(
                    ChunkRecord(chunk_id=len(self._chunks), source=source, text=chunk)
                )
            added_chunks += len(chunks)

        return {
            "documents": len(documents),
            "chunks": added_chunks,
            "index_size": len(self._chunks),
        }

    async def query(
        self,
        text: str,
        top_k: int = 5,
        use_agent_reranker: bool = False,
    ) -> list[ChunkRecord]:
        if self._store is None or self._store.size == 0:
            return []

        query_embedding = self._embedder.encode([text])[0]
        retrieved = self._store.search(query_embedding, top_k=max(top_k * 3, top_k))

        candidates = [
            Candidate(
                chunk_id=item.chunk_id,
                text=self._chunks[item.chunk_id].text,
                retrieval_score=item.score,
            )
            for item in retrieved
        ]

        if use_agent_reranker and self._agent_reranker is not None:
            reranked = await self._agent_reranker.rerank(text, candidates, top_k=top_k)
        else:
            reranked = self._reranker.rerank(text, candidates, top_k=top_k)

        return [self._chunks[item.chunk_id] for item in reranked]

    @property
    def indexed_chunks(self) -> int:
        return len(self._chunks)
