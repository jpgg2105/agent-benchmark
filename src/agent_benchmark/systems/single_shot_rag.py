from __future__ import annotations

from ..models import SystemAnswer
from ._settings import agent_settings, usage_of

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
    both systems share the same model and the same retriever.
    """

    def __init__(
        self,
        name: str,
        llm_provider: str = "fake",
        retrieval_mode: str = "stub",
        top_k: int = 5,
        local_corpus_path: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.name = name
        self.top_k = top_k
        self._params = dict(
            llm_provider=llm_provider, retrieval_mode=retrieval_mode, top_k=top_k,
            local_corpus_path=local_corpus_path, model=model, base_url=base_url,
        )
        self._tool = None

    async def run(self, question: str) -> SystemAnswer:
        from research_agent.llm.factory import build_llm_client
        from research_agent.tools.factory import build_retrieval_tool

        settings = agent_settings(**self._params)
        llm = build_llm_client(settings)          # one client per question = per-question token count
        if self._tool is None:                    # the index is built once and reused
            self._tool = build_retrieval_tool(settings)

        sources = await self._tool.search(question, top_k=self.top_k)
        rendered = "\n\n".join(
            f"[Source {i}] {s.title or 'untitled'}\n{s.snippet or ''}"
            for i, s in enumerate(sources, start=1)
        )
        text = await llm.complete(system=_SYSTEM, user=f"Question:\n{question}\n\nSources:\n{rendered}")
        titles = [s.title or s.locator or "source" for s in sources]
        return SystemAnswer(answer=text, citations=titles, retrieved=titles, usage=usage_of(llm))
