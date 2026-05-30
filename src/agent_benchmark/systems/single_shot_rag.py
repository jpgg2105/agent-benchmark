from __future__ import annotations

from ..models import SystemAnswer

# A deliberately plain prompt: retrieve once, answer once, no planning or critique.
_SYSTEM = (
    "Answer the user's question using only the provided sources, concisely and "
    "accurately. If the sources are insufficient, say so plainly."
)


class SingleShotRAGSystem:
    """Baseline: a single retrieve-then-answer pass with no agent loop.

    This is the control the multi-agent system is measured against -- it shows
    whether planning + critique + iteration actually add value. It reuses the
    research-agent project's LLM and retrieval building blocks (lazy import), so
    that project must be importable on the path to use this system.
    """

    def __init__(
        self,
        name: str,
        llm_provider: str = "fake",
        retrieval_mode: str = "stub",
        top_k: int = 5,
    ) -> None:
        self.name = name
        self.llm_provider = llm_provider
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k

    async def run(self, question: str) -> SystemAnswer:
        from research_agent.config import Settings
        from research_agent.llm.factory import build_llm_client
        from research_agent.tools.factory import build_retrieval_tool

        settings = Settings(
            llm_provider=self.llm_provider,
            retrieval_mode=self.retrieval_mode,
            top_k=self.top_k,
        )
        llm = build_llm_client(settings)
        tool = build_retrieval_tool(settings)

        sources = await tool.search(question, top_k=self.top_k)
        rendered = "\n\n".join(
            f"[Source {i}] {s.title or 'untitled'}\n{s.snippet or ''}"
            for i, s in enumerate(sources, start=1)
        )
        text = await llm.complete(system=_SYSTEM, user=f"Question:\n{question}\n\nSources:\n{rendered}")
        citations = [s.title or s.locator or "source" for s in sources]
        return SystemAnswer(answer=text, citations=[c for c in citations if c])
