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
