from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from multi_agentic_platform.config import settings
from multi_agentic_platform.mcp import MCPServerConfig, MCPService
from multi_agentic_platform.orchestrator import Orchestrator
from multi_agentic_platform.rag.pipeline import RAGPipeline
from multi_agentic_platform.schemas import (
    MCPServerInfo,
    MCPToolsResponse,
    RAGIngestRequest,
    RAGIngestResponse,
    RAGIngestTextRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGResult,
    RunRequest,
    RunResponse,
    WorkflowIngestRequest,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from multi_agentic_platform.workflow import CompanyWorkflow, LangChainRAGService

app = FastAPI(title="Multi-Agentic Platform", version="0.3.0")
orchestrator = Orchestrator()


class RAGService:
    def __init__(self) -> None:
        self._pipeline: RAGPipeline | None = None

    def _get_pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            try:
                self._pipeline = RAGPipeline(rerank_with_agent_provider=orchestrator.provider)
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"RAG dependencies missing: {exc}. Run: pip install -e .",
                ) from exc
        return self._pipeline

    def ingest_paths(self, paths: list[str]) -> dict[str, int]:
        return self._get_pipeline().ingest_paths(paths)

    def ingest_text_documents(self, documents: list[tuple[str, str]]) -> dict[str, int]:
        return self._get_pipeline().ingest_documents(documents)

    async def query(self, text: str, top_k: int, use_agent_reranker: bool) -> list[RAGResult]:
        rows = await self._get_pipeline().query(
            text=text,
            top_k=top_k,
            use_agent_reranker=use_agent_reranker,
        )
        return [RAGResult(chunk_id=row.chunk_id, source=row.source, text=row.text) for row in rows]


class WorkflowService:
    def __init__(self) -> None:
        self._rag: LangChainRAGService | None = None
        self._workflow: CompanyWorkflow | None = None

    def _get_rag(self) -> LangChainRAGService:
        if self._rag is None:
            try:
                self._rag = LangChainRAGService()
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"LangChain dependencies missing: {exc}",
                ) from exc
        return self._rag

    def _get_workflow(self) -> CompanyWorkflow:
        if self._workflow is None:
            self._workflow = CompanyWorkflow(provider=orchestrator.provider, rag_service=self._get_rag())
        return self._workflow

    def ingest_paths(self, paths: list[str]) -> dict[str, int]:
        try:
            return self._get_rag().ingest_paths(paths)
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def run(self, query: str) -> WorkflowRunResponse:
        try:
            result = await self._get_workflow().run(query)
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return WorkflowRunResponse(**result)


class MCPGatewayService:
    def __init__(self) -> None:
        servers: list[MCPServerConfig] = []
        names = [name.strip() for name in settings.mcp_server_names.split(",") if name.strip()]
        for name in names:
            key = name.upper().replace("-", "_")
            command = os.getenv(f"MAP_MCP_{key}_COMMAND", "")
            args_raw = os.getenv(f"MAP_MCP_{key}_ARGS", "")
            if command:
                servers.append(MCPServerConfig(name=name, command=command, args=args_raw.split()))
        self._service = MCPService(servers)

    def servers(self) -> list[MCPServerInfo]:
        return [MCPServerInfo(**item) for item in self._service.configured_servers()]

    async def list_tools(self, server_name: str) -> MCPToolsResponse:
        try:
            tools = await self._service.list_tools(server_name)
        except (ImportError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return MCPToolsResponse(server=server_name, tools=tools)


rag_service = RAGService()
workflow_service = WorkflowService()
mcp_service = MCPGatewayService()


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\"/>
    <title>Open-Source Company Workflow Platform</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 980px; }
      textarea, input, button { font-size: 1rem; }
      textarea { width: 100%; min-height: 110px; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
      .result { border-left: 4px solid #4f46e5; padding-left: .75rem; margin: .75rem 0; white-space: pre-wrap; }
      .muted { color: #666; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
      h1 { margin-bottom: 0.3rem; }
      .pill { background: #eef2ff; border-radius: 20px; display: inline-block; padding: 0.2rem 0.7rem; margin-right: .35rem; }
    </style>
  </head>
  <body>
    <h1>Open-Source Company Workflow Platform</h1>
    <div class=\"muted\">RAG + LangChain + LangGraph + MCP integration endpoints</div>
    <p><span class=\"pill\">RAG</span><span class=\"pill\">LangChain</span><span class=\"pill\">LangGraph</span><span class=\"pill\">MCP</span></p>

    <div class=\"card\">
      <h3>1) Ingest bundled sample docs to native RAG</h3>
      <button onclick=\"ingestNativeSamples()\">Ingest Native RAG Samples</button>
      <p id=\"nativeStatus\" class=\"muted\"></p>
    </div>

    <div class=\"card\">
      <h3>2) Ingest bundled sample docs to LangChain workflow</h3>
      <button onclick=\"ingestWorkflowSamples()\">Ingest Workflow Samples</button>
      <p id=\"workflowStatus\" class=\"muted\"></p>
    </div>

    <div class=\"card\">
      <h3>3) Query Native RAG (Top 5)</h3>
      <textarea id=\"ragQuery\" placeholder=\"Ask a retrieval question...\"></textarea>
      <label><input type=\"checkbox\" id=\"useAgent\"/> Use agent reranker</label><br/><br/>
      <button onclick=\"runRagQuery()\">Run RAG Query</button>
      <div id=\"ragResults\"></div>
    </div>

    <div class=\"card\">
      <h3>4) Run LangGraph company workflow</h3>
      <textarea id=\"workflowQuery\" placeholder=\"Example: Create onboarding workflow for new support engineers\"></textarea>
      <button onclick=\"runCompanyWorkflow()\">Run Workflow</button>
      <div id=\"workflowResults\"></div>
    </div>

    <script>
      async function showOnlineSetup() {
        const res = await fetch('/setup/online');
        document.getElementById('onlineSetup').innerText = JSON.stringify(await res.json());
      }

      async function ingestNativeSamples() {
        const res = await fetch('/rag/ingest/samples', { method: 'POST' });
        document.getElementById('nativeStatus').innerText = JSON.stringify(await res.json());
      }

      async function ingestWorkflowSamples() {
        const res = await fetch('/workflow/ingest/samples', { method: 'POST' });
        document.getElementById('workflowStatus').innerText = JSON.stringify(await res.json());
      }

      async function runRagQuery() {
        const query = document.getElementById('ragQuery').value;
        const use_agent_reranker = document.getElementById('useAgent').checked;
        const res = await fetch('/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: 5, use_agent_reranker })
        });
        const data = await res.json();
        const html = (data.results || []).map((r) => (`<div class=\"result\"><b>${r.source}</b><div>${r.text}</div></div>`)).join('');
        document.getElementById('ragResults').innerHTML = html || '<p class="muted">No results yet.</p>';
      }

      async function runCompanyWorkflow() {
        const query = document.getElementById('workflowQuery').value;
        const res = await fetch('/workflow/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        const html = `
          <div class=\"result\"><b>Draft</b><div>${data.draft || ''}</div></div>
          <div class=\"result\"><b>Compliance Notes</b><div>${data.compliance_notes || ''}</div></div>
          <div class=\"result\"><b>Final Answer</b><div>${data.final_answer || ''}</div></div>
        `;
        document.getElementById('workflowResults').innerHTML = html;
      }
    </script>
  </body>
</html>
"""




@app.get("/setup/online")
async def setup_online() -> dict[str, str]:
    return {
        "mode": "hosted_open_source",
        "provider": "hf_inference",
        "env": "MAP_PROVIDER=hf_inference MAP_HF_API_TOKEN=<your_token> MAP_HF_INFERENCE_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct MAP_RAG_USE_ONLINE_INFERENCE=true",
        "note": "This mode uses hosted open-source inference so local model downloads are not required.",
    }

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    return await orchestrator.run(request)


@app.post("/rag/ingest", response_model=RAGIngestResponse)
async def rag_ingest(request: RAGIngestRequest) -> RAGIngestResponse:
    return RAGIngestResponse(**rag_service.ingest_paths(request.paths))


@app.post("/rag/ingest/text", response_model=RAGIngestResponse)
async def rag_ingest_text(request: RAGIngestTextRequest) -> RAGIngestResponse:
    docs = [(doc.source, doc.content) for doc in request.documents]
    return RAGIngestResponse(**rag_service.ingest_text_documents(docs))


@app.post("/rag/ingest/samples", response_model=RAGIngestResponse)
async def rag_ingest_samples() -> RAGIngestResponse:
    sample_dir = Path(__file__).resolve().parent.parent.parent / "sample_data"
    paths = [str(path) for path in sample_dir.glob("*") if path.is_file()]
    if not paths:
        raise HTTPException(status_code=404, detail="No sample documents found.")
    return RAGIngestResponse(**rag_service.ingest_paths(paths))


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    results = await rag_service.query(request.query, request.top_k, request.use_agent_reranker)
    return RAGQueryResponse(query=request.query, results=results)


@app.post("/workflow/ingest", response_model=RAGIngestResponse)
async def workflow_ingest(request: WorkflowIngestRequest) -> RAGIngestResponse:
    return RAGIngestResponse(**workflow_service.ingest_paths(request.paths))


@app.post("/workflow/ingest/samples", response_model=RAGIngestResponse)
async def workflow_ingest_samples() -> RAGIngestResponse:
    sample_dir = Path(__file__).resolve().parent.parent.parent / "sample_data"
    paths = [str(path) for path in sample_dir.glob("*") if path.is_file()]
    if not paths:
        raise HTTPException(status_code=404, detail="No sample documents found.")
    return RAGIngestResponse(**workflow_service.ingest_paths(paths))


@app.post("/workflow/run", response_model=WorkflowRunResponse)
async def workflow_run(request: WorkflowRunRequest) -> WorkflowRunResponse:
    return await workflow_service.run(request.query)


@app.get("/mcp/servers", response_model=list[MCPServerInfo])
async def mcp_servers() -> list[MCPServerInfo]:
    return mcp_service.servers()


@app.get("/mcp/servers/{server_name}/tools", response_model=MCPToolsResponse)
async def mcp_server_tools(server_name: str) -> MCPToolsResponse:
    return await mcp_service.list_tools(server_name)
