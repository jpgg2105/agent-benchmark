from __future__ import annotations

from ..models import SystemAnswer
from ._settings import agent_settings, usage_of


class ResearchAgentSystem:
    """Adapter that runs the project #2 research-agent as a system-under-test.

    Requires the research-agent package to be importable. Parameters are passed
    through to its Settings, so different agent configurations (iterations,
    sub-question budget, model) become separate leaderboard rows.
    """

    def __init__(
        self,
        name: str,
        llm_provider: str = "fake",
        retrieval_mode: str = "stub",
        max_iterations: int = 2,
        max_sub_questions: int = 5,
        top_k: int = 5,
        local_corpus_path: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        rpm: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.name = name
        self._params = dict(
            llm_provider=llm_provider, retrieval_mode=retrieval_mode, max_iterations=max_iterations,
            max_sub_questions=max_sub_questions, top_k=top_k, local_corpus_path=local_corpus_path,
            model=model, base_url=base_url, rpm=rpm, reasoning_effort=reasoning_effort,
        )
        self._tool = None

    async def run(self, question: str) -> SystemAnswer:
        from research_agent.agents import Critic, Planner, Researcher, Synthesizer
        from research_agent.llm.factory import build_llm_client
        from research_agent.orchestrator import ResearchOrchestrator
        from research_agent.tools.factory import build_retrieval_tool

        settings = agent_settings(**self._params)
        llm = build_llm_client(settings)          # shared by the four agents: one token count per question
        if self._tool is None:
            self._tool = build_retrieval_tool(settings)
        orchestrator = ResearchOrchestrator(
            planner=Planner(llm, max_sub_questions=settings.max_sub_questions),
            researcher=Researcher(llm, self._tool, top_k=settings.top_k),
            critic=Critic(llm),
            synthesizer=Synthesizer(llm),
            max_iterations=settings.max_iterations,
        )
        state = await orchestrator.run(question)

        retrieved: list[str] = []
        for f in state.findings:
            for s in f.sources:
                t = s.title or s.locator
                if t and t not in retrieved:
                    retrieved.append(t)
        trace = {
            "sub_questions": [sq.question for sq in (state.plan.sub_questions if state.plan else [])],
            "rounds": state.iterations,
            "critiques": [
                {"sufficient": c.is_sufficient, "missing": c.missing, "follow_ups": c.follow_up_questions}
                for c in state.critiques
            ],
            "findings": [
                {"q": f.sub_question, "a": f.answer, "confidence": f.confidence, "sources": [s.title for s in f.sources]}
                for f in state.findings
            ],
        }
        usage = usage_of(llm) | {"rounds": state.iterations, "findings": len(state.findings)}
        report = state.report
        if report is None:
            return SystemAnswer(answer="", retrieved=retrieved, usage=usage, trace=trace)
        return SystemAnswer(
            answer=f"{report.summary}\n\n{report.body}".strip(),
            citations=[c.title or c.locator or "source" for c in report.citations],
            retrieved=retrieved,
            usage=usage,
            trace=trace,
        )
