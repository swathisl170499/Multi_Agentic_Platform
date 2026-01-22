from fastapi import FastAPI

from multi_agentic_platform.orchestrator import Orchestrator
from multi_agentic_platform.schemas import RunRequest, RunResponse

app = FastAPI(title="Multi-Agentic Platform", version="0.1.0")
orchestrator = Orchestrator()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    return await orchestrator.run(request)
