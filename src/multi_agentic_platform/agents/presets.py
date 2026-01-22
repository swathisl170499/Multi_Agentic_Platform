from multi_agentic_platform.agents.base import Agent
from multi_agentic_platform.providers.base import LLMProvider


def create_planner(provider: LLMProvider) -> Agent:
    return Agent(
        name="planner",
        system_prompt=(
            "You are a planning agent. Break down the user request into steps, "
            "list requirements, and highlight risks or missing info."
        ),
        provider=provider,
    )


def create_coder(provider: LLMProvider) -> Agent:
    return Agent(
        name="coder",
        system_prompt=(
            "You are a coding agent. Produce production-ready code for the user's request. "
            "Include clear structure and comments where appropriate."
        ),
        provider=provider,
    )


def create_reviewer(provider: LLMProvider) -> Agent:
    return Agent(
        name="reviewer",
        system_prompt=(
            "You are a reviewer. Evaluate the code for correctness, security, tests, "
            "and suggest improvements."
        ),
        provider=provider,
    )
