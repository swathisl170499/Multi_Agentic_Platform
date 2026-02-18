from __future__ import annotations

from typing import TypedDict

from multi_agentic_platform.providers.base import LLMProvider
from multi_agentic_platform.workflow.langchain_rag import LangChainRAGService


class CompanyWorkflowState(TypedDict):
    query: str
    contexts: list[str]
    draft: str
    compliance_notes: str
    final_answer: str


class CompanyWorkflow:
    """LangGraph workflow representing a core company request lifecycle."""

    def __init__(self, provider: LLMProvider, rag_service: LangChainRAGService) -> None:
        self._provider = provider
        self._rag = rag_service

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise ImportError("langgraph is required. Install: pip install langgraph") from exc

        graph = StateGraph(CompanyWorkflowState)

        async def retrieve_node(state: CompanyWorkflowState) -> CompanyWorkflowState:
            results = self._rag.retrieve(state["query"], top_k=5)
            return {**state, "contexts": [f"[{r.source}] {r.text}" for r in results]}

        async def draft_node(state: CompanyWorkflowState) -> CompanyWorkflowState:
            context_block = "\n\n".join(state["contexts"]) if state["contexts"] else "No context"
            draft = await self._provider.generate(
                system=(
                    "You are an open-source enterprise operations assistant. "
                    "Write a practical response based on retrieved company documents."
                ),
                prompt=f"User request:\n{state['query']}\n\nContext:\n{context_block}",
            )
            return {**state, "draft": draft}

        async def compliance_node(state: CompanyWorkflowState) -> CompanyWorkflowState:
            notes = await self._provider.generate(
                system=(
                    "You are a compliance reviewer. Return 2-4 concise bullets focused on "
                    "privacy, security, and policy risks."
                ),
                prompt=f"Evaluate this draft response:\n{state['draft']}",
            )
            return {**state, "compliance_notes": notes}

        async def finalize_node(state: CompanyWorkflowState) -> CompanyWorkflowState:
            final_answer = await self._provider.generate(
                system=(
                    "You are a final response editor. Produce a clean final answer with action "
                    "items and include a short 'Risk Notes' section."
                ),
                prompt=(
                    f"Original request:\n{state['query']}\n\n"
                    f"Draft:\n{state['draft']}\n\n"
                    f"Compliance notes:\n{state['compliance_notes']}"
                ),
            )
            return {**state, "final_answer": final_answer}

        graph.add_node("retrieve", retrieve_node)
        graph.add_node("draft", draft_node)
        graph.add_node("compliance", compliance_node)
        graph.add_node("finalize", finalize_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "draft")
        graph.add_edge("draft", "compliance")
        graph.add_edge("compliance", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    async def run(self, query: str) -> dict[str, str | list[str]]:
        app = self._build_graph()
        initial_state: CompanyWorkflowState = {
            "query": query,
            "contexts": [],
            "draft": "",
            "compliance_notes": "",
            "final_answer": "",
        }
        result = await app.ainvoke(initial_state)
        return {
            "query": query,
            "contexts": result.get("contexts", []),
            "draft": result.get("draft", ""),
            "compliance_notes": result.get("compliance_notes", ""),
            "final_answer": result.get("final_answer", ""),
        }
