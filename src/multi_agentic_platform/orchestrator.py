import uuid

from multi_agentic_platform.agents.presets import (
    create_coder,
    create_planner,
    create_reviewer,
)
from multi_agentic_platform.config import settings
from multi_agentic_platform.providers.base import LLMProvider
from multi_agentic_platform.providers.huggingface_provider import HuggingFaceProvider
from multi_agentic_platform.providers.mock import MockProvider
from multi_agentic_platform.providers.openai_provider import OpenAIProvider
from multi_agentic_platform.schemas import AgentTrace, RunRequest, RunResponse
from multi_agentic_platform.sandbox.executor import SandboxExecutor


def _load_provider() -> LLMProvider:
    if settings.provider == "openai":
        return OpenAIProvider()
    if settings.provider in {"hf", "huggingface"}:
        return HuggingFaceProvider()
    return MockProvider()


class Orchestrator:
    def __init__(self) -> None:
        self.provider = _load_provider()
        self.planner = create_planner(self.provider)
        self.coder = create_coder(self.provider)
        self.reviewer = create_reviewer(self.provider)
        self.sandbox = SandboxExecutor()

    async def run(self, request: RunRequest) -> RunResponse:
        request_id = str(uuid.uuid4())
        plan = await self.planner.act(request.prompt)
        code_prompt = (
            f"User request:\n{request.prompt}\n\n"
            f"Target language: {request.language}\n\n"
            f"Plan:\n{plan}\n"
        )
        code = await self.coder.act(code_prompt)
        review = None
        traces = [
            AgentTrace(agent=self.planner.name, output=plan),
            AgentTrace(agent=self.coder.name, output=code),
        ]

        if request.require_review:
            review_prompt = (
                f"Review the following code for correctness, security, and tests.\n\n{code}"
            )
            review = await self.reviewer.act(review_prompt)
            traces.append(AgentTrace(agent=self.reviewer.name, output=review))

        if request.language.lower() == "python":
            sandbox_result = await self.sandbox.run_python(code)
            traces.append(
                AgentTrace(
                    agent="sandbox",
                    output=(
                        "Sandbox executed.\n"
                        f"stdout:\n{sandbox_result.stdout}\n"
                        f"stderr:\n{sandbox_result.stderr}\n"
                        f"return_code: {sandbox_result.return_code}\n"
                        f"note: {self.sandbox.safety_notice()}"
                    ),
                )
            )

        return RunResponse(
            request_id=request_id,
            code=code,
            review=review,
            traces=traces,
        )
