# Multi_Agentic_Platform

A production-ready multi-agent orchestration platform with a lightweight sandbox runner.

## Features
- Pluggable LLM providers (mock + OpenAI starter implementation).
- Multi-agent workflow (planner → coder → reviewer).
- Optional sandbox execution for Python code.
- FastAPI service for testing and integration.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn multi_agentic_platform.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API.

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

## Provider configuration

Set environment variables (or copy `.env.example` to `.env`) to configure providers:

- `MAP_PROVIDER=mock` (default) or `openai`
- `MAP_OPENAI_API_KEY=...`
- `MAP_OPENAI_MODEL=gpt-4o-mini`

## Sandbox note

The included sandbox is a lightweight subprocess executor for quick local testing.
For production workloads, use container or VM isolation with network and filesystem
restrictions.
