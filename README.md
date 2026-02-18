# Multi_Agentic_Platform

A production-ready multi-agent orchestration platform with a lightweight sandbox runner.

## Features
- Pluggable LLM providers (mock + OpenAI starter implementation).
- Multi-agent workflow (planner → coder → reviewer).
- Optional sandbox execution for Python code.
- FastAPI service for testing and integration.
- RAG pipeline with document ingestion (PDF/JSON/CSV/TXT), chunking, embeddings, FAISS retrieval, and reranking.
- Simple built-in UI for sample ingestion, adding custom text/JSON/CSV content, and querying.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn multi_agentic_platform.main:app --reload
```

Open:
- API docs: `http://localhost:8000/docs`
- RAG UI: `http://localhost:8000/`

## Running with Docker

```bash
docker compose up --build
```

## API usage

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that adds two numbers.", "language": "python"}'
```

## RAG usage

### 1) Ingest documents by path

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["/data/handbook.pdf", "/data/records.csv", "/data/config.json"]}'
```

### 2) Ingest sample docs bundled in repo

```bash
curl -X POST http://localhost:8000/rag/ingest/samples
```

### 3) Ingest custom text document

```bash
curl -X POST http://localhost:8000/rag/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"source": "notes.txt", "content": "MFA is required for production access."}]}'
```

### 4) Query (top 5)

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are onboarding requirements?", "top_k": 5}'
```

### 5) Query with reranker agent

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "security controls", "top_k": 5, "use_agent_reranker": true}'
```

## Provider configuration

Set environment variables (or copy `.env.example` to `.env`) to configure providers:

- `MAP_PROVIDER=mock` (default) or `openai`
- `MAP_OPENAI_API_KEY=...`
- `MAP_OPENAI_MODEL=gpt-4o-mini`

## RAG configuration

- `MAP_RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `MAP_RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `MAP_RAG_CHUNK_SIZE=600`
- `MAP_RAG_CHUNK_OVERLAP=120`

## Sandbox note

The included sandbox is a lightweight subprocess executor for quick local testing.
For production workloads, use container or VM isolation with network and filesystem
restrictions.
