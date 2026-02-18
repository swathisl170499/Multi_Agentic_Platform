from __future__ import annotations

from dataclasses import dataclass

from multi_agentic_platform.config import settings
from multi_agentic_platform.rag.loaders import load_document


@dataclass
class RetrievedContext:
    source: str
    text: str
    score: float


class LangChainRAGService:
    """Open-source RAG stack backed by LangChain components."""

    def __init__(self) -> None:
        self._vs = None
        self._documents: list[tuple[str, str]] = []

    def _ensure_imports(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(
                "LangChain deps missing. Install: pip install langchain langchain-community "
                "langchain-text-splitters"
            ) from exc

        return HuggingFaceEmbeddings, FAISS, RecursiveCharacterTextSplitter

    def ingest_paths(self, paths: list[str]) -> dict[str, int]:
        docs = [load_document(path) for path in paths]
        return self.ingest_documents(docs)

    def ingest_documents(self, docs: list[tuple[str, str]]) -> dict[str, int]:
        HuggingFaceEmbeddings, FAISS, RecursiveCharacterTextSplitter = self._ensure_imports()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        embeddings = HuggingFaceEmbeddings(model_name=settings.rag_embedding_model)

        texts: list[str] = []
        metadatas: list[dict[str, str]] = []
        for source, content in docs:
            chunks = splitter.split_text(content)
            texts.extend(chunks)
            metadatas.extend({"source": source} for _ in chunks)
            self._documents.extend((source, chunk) for chunk in chunks)

        if not texts:
            return {"documents": len(docs), "chunks": 0, "index_size": len(self._documents)}

        new_vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        if self._vs is None:
            self._vs = new_vs
        else:
            self._vs.merge_from(new_vs)

        return {"documents": len(docs), "chunks": len(texts), "index_size": len(self._documents)}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedContext]:
        if self._vs is None:
            return []

        docs_and_scores = self._vs.similarity_search_with_score(query, k=max(top_k * 3, top_k))
        contexts = [
            RetrievedContext(
                source=item[0].metadata.get("source", "unknown"),
                text=item[0].page_content,
                score=float(item[1]),
            )
            for item in docs_and_scores
        ]
        contexts.sort(key=lambda row: row.score)
        return contexts[:top_k]
