from __future__ import annotations

from ..models import SystemAnswer


class ResearchAgentSystem:
    """Adapter that runs the project #2 research-agent as a system-under-test.

    Requires the research-agent package to be importable. Parameters are passed
    straight through to its Settings, so you can benchmark different agent
    configurations (iterations, sub-question budget, provider) as separate
    leaderboard rows.
    """

    def __init__(
        self,
        name: str,
        llm_provider: str = "fake",
        retrieval_mode: str = "stub",
        max_iterations: int = 2,
        max_sub_questions: int = 5,
        top_k: int = 5,
    ) -> None:
        self.name = name
        self._overrides = dict(
            llm_provider=llm_provider,
            retrieval_mode=retrieval_mode,
            max_iterations=max_iterations,
            max_sub_questions=max_sub_questions,
            top_k=top_k,
        )

    async def run(self, question: str) -> SystemAnswer:
        from research_agent.config import Settings
        from research_agent.factory import build_orchestrator

        orchestrator = build_orchestrator(Settings(**self._overrides))
        state = await orchestrator.run(question)

        report = state.report
        if report is None:
            return SystemAnswer(answer="")
        citations = [c.title or c.locator or "source" for c in report.citations]
        return SystemAnswer(
            answer=f"{report.summary}\n\n{report.body}".strip(),
            citations=[c for c in citations if c],
            usage={"iterations": state.iterations, "findings": len(state.findings)},
        )
