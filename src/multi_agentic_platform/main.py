from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from multi_agentic_platform.orchestrator import Orchestrator
from multi_agentic_platform.rag.pipeline import RAGPipeline
from multi_agentic_platform.schemas import (
    RAGIngestRequest,
    RAGIngestResponse,
    RAGIngestTextRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGResult,
    RunRequest,
    RunResponse,
)

app = FastAPI(title="Multi-Agentic Platform", version="0.1.0")
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

    async def query(
        self,
        text: str,
        top_k: int,
        use_agent_reranker: bool,
    ) -> list[RAGResult]:
        rows = await self._get_pipeline().query(
            text=text,
            top_k=top_k,
            use_agent_reranker=use_agent_reranker,
        )
        return [RAGResult(chunk_id=row.chunk_id, source=row.source, text=row.text) for row in rows]


rag_service = RAGService()


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\"/>
    <title>RAG Demo UI</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 900px; }
      textarea, input, button { font-size: 1rem; }
      textarea { width: 100%; min-height: 110px; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
      .result { border-left: 4px solid #4f46e5; padding-left: .75rem; margin: .75rem 0; white-space: pre-wrap; }
      .muted { color: #666; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    </style>
  </head>
  <body>
    <h1>RAG Demo UI</h1>
    <p class=\"muted\">Ingest sample docs (bundled), add your own text/CSV/JSON content, then query top-5 results.</p>

    <div class=\"card\">
      <h3>1) Ingest sample documents</h3>
      <button onclick=\"ingestSamples()\">Ingest Sample Data</button>
      <p id=\"sampleStatus\" class=\"muted\"></p>
    </div>

    <div class=\"card\">
      <h3>2) Add custom content</h3>
      <div class=\"grid\">
        <div>
          <label>Source name</label>
          <input id=\"source\" placeholder=\"my_notes.txt\" style=\"width:100%\"/>
        </div>
        <div>
          <label>Format hint</label>
          <input id=\"format\" placeholder=\"txt / csv / json\" style=\"width:100%\"/>
        </div>
      </div>
      <br/>
      <textarea id=\"content\" placeholder=\"Paste content here...\"></textarea>
      <br/><br/>
      <button onclick=\"ingestText()\">Ingest Text Content</button>
      <p id=\"textStatus\" class=\"muted\"></p>
    </div>

    <div class=\"card\">
      <h3>3) Query</h3>
      <textarea id=\"query\" placeholder=\"Ask a question...\"></textarea>
      <label><input type=\"checkbox\" id=\"useAgent\"/> Use agent reranker</label><br/><br/>
      <button onclick=\"runQuery()\">Search Top 5</button>
      <div id=\"results\"></div>
    </div>

    <script>
      async function ingestSamples() {
        const res = await fetch('/rag/ingest/samples', { method: 'POST' });
        const data = await res.json();
        document.getElementById('sampleStatus').innerText = JSON.stringify(data);
      }

      async function ingestText() {
        const source = document.getElementById('source').value || 'pasted.txt';
        const format = document.getElementById('format').value || 'txt';
        const content = document.getElementById('content').value;
        const wrapped = format.toLowerCase() === 'json'
          ? JSON.stringify(JSON.parse(content), null, 2)
          : content;

        const res = await fetch('/rag/ingest/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ documents: [{ source, content: wrapped }] })
        });
        const data = await res.json();
        document.getElementById('textStatus').innerText = JSON.stringify(data);
      }

      async function runQuery() {
        const query = document.getElementById('query').value;
        const use_agent_reranker = document.getElementById('useAgent').checked;
        const res = await fetch('/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: 5, use_agent_reranker })
        });
        const data = await res.json();
        const html = (data.results || []).map((r) => (
          `<div class=\"result\"><div><b>#${r.chunk_id}</b> <span class=\"muted\">${r.source}</span></div><div>${r.text}</div></div>`
        )).join('') || '<p class="muted">No results yet. Ingest docs first.</p>';
        document.getElementById('results').innerHTML = html;
      }
    </script>
  </body>
</html>
"""


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    return await orchestrator.run(request)


@app.post("/rag/ingest", response_model=RAGIngestResponse)
async def rag_ingest(request: RAGIngestRequest) -> RAGIngestResponse:
    stats = rag_service.ingest_paths(request.paths)
    return RAGIngestResponse(**stats)


@app.post("/rag/ingest/text", response_model=RAGIngestResponse)
async def rag_ingest_text(request: RAGIngestTextRequest) -> RAGIngestResponse:
    docs = [(doc.source, doc.content) for doc in request.documents]
    stats = rag_service.ingest_text_documents(docs)
    return RAGIngestResponse(**stats)


@app.post("/rag/ingest/samples", response_model=RAGIngestResponse)
async def rag_ingest_samples() -> RAGIngestResponse:
    sample_dir = Path(__file__).resolve().parent.parent.parent / "sample_data"
    paths = [str(path) for path in sample_dir.glob("*") if path.is_file()]
    if not paths:
        raise HTTPException(status_code=404, detail="No sample documents found.")
    stats = rag_service.ingest_paths(paths)
    return RAGIngestResponse(**stats)


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    results = await rag_service.query(
        text=request.query,
        top_k=request.top_k,
        use_agent_reranker=request.use_agent_reranker,
    )
    return RAGQueryResponse(query=request.query, results=results)
