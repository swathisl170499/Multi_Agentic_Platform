from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=8000)
    language: str = Field("python", min_length=2, max_length=32)
    require_review: bool = True


class AgentTrace(BaseModel):
    agent: str
    output: str


class RunResponse(BaseModel):
    request_id: str
    code: str
    review: str | None
    traces: list[AgentTrace]


class RAGIngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, min_length=1)


class RAGTextDocument(BaseModel):
    source: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)


class RAGIngestTextRequest(BaseModel):
    documents: list[RAGTextDocument] = Field(default_factory=list, min_length=1)


class RAGIngestResponse(BaseModel):
    documents: int
    chunks: int
    index_size: int


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=8000)
    top_k: int = Field(5, ge=1, le=20)
    use_agent_reranker: bool = False


class RAGResult(BaseModel):
    chunk_id: int
    source: str
    text: str


class RAGQueryResponse(BaseModel):
    query: str
    results: list[RAGResult]
