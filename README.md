# Multi_Agentic_Platform

Open-source multi-agent platform for **RAG + LangChain + LangGraph + MCP** company workflows.

## What is included
- Multi-agent orchestration (`planner -> coder -> reviewer`).
- Native RAG pipeline (chunking + embeddings + FAISS + reranker).
- LangChain RAG service for company knowledge retrieval.
- LangGraph workflow for enterprise-style request handling (retrieve, draft, compliance, finalize).
- MCP gateway endpoints to inspect configured MCP servers/tools.
- Browser UI (`/`) to run the full flow quickly.

## Fastest way (online, no local model download)
If you don’t want to download local models, use hosted open-source inference:

```bash
export MAP_PROVIDER=hf_inference
export MAP_HF_API_TOKEN=<your_huggingface_token>
export MAP_HF_INFERENCE_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
export MAP_RAG_USE_ONLINE_INFERENCE=true
```

This keeps models on Hugging Face Inference API while your app orchestrates the workflow locally.

## Local model setup (fully local)
```bash
export MAP_PROVIDER=hf
export MAP_HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn multi_agentic_platform.main:app --reload
```

Open:
- UI: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`
- Online setup helper: `http://localhost:8000/setup/online`

## Core company workflow style (end-to-end)

### 1) Ingest sample company docs (native RAG)
```bash
curl -X POST http://localhost:8000/rag/ingest/samples
```

### 2) Ingest sample docs for LangChain/LangGraph workflow
```bash
curl -X POST http://localhost:8000/workflow/ingest/samples
```

### 3) Ask a retrieval question (top 5)
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are onboarding requirements?", "top_k": 5}'
```

### 4) Run full company workflow (LangGraph)
```bash
curl -X POST http://localhost:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{"query": "Create onboarding workflow for support engineers with security steps"}'
```

## MCP server integration
Configure one or more MCP servers via env vars.

Example (filesystem server):

```bash
export MAP_MCP_SERVER_NAMES=filesystem
export MAP_MCP_FILESYSTEM_COMMAND=npx
export MAP_MCP_FILESYSTEM_ARGS="-y @modelcontextprotocol/server-filesystem /workspace"
```

Then check configured servers/tools:

```bash
curl http://localhost:8000/mcp/servers
curl http://localhost:8000/mcp/servers/filesystem/tools
```

## Key API routes
- `POST /run` - existing multi-agent code workflow.
- `POST /rag/ingest`
- `POST /rag/ingest/text`
- `POST /rag/ingest/samples`
- `POST /rag/query`
- `POST /workflow/ingest`
- `POST /workflow/ingest/samples`
- `POST /workflow/run`
- `GET /mcp/servers`
- `GET /mcp/servers/{server_name}/tools`
- `GET /setup/online`

## Notes
- Heavy dependencies (LangChain/LangGraph/MCP/FAISS) initialize lazily and only when endpoints are used.
- Hosted mode (`hf_inference`) helps avoid local model downloads.
