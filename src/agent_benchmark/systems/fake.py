from __future__ import annotations

from ..models import SystemAnswer


class FakeSystem:
    """Deterministic, offline system-under-test.

    Exists so the harness and tests run with no API key and no sibling
    projects. The ``quality`` knob lets the default config show two contrasting
    leaderboard rows; a "high" system echoes the question (so it covers
    key-points that appear in it), a "low" one returns a terse answer.
    These numbers illustrate the leaderboard format -- they are not a real
    evaluation, which requires real systems (see single_shot_rag / research_agent).
    """

    def __init__(self, name: str, quality: str = "high", with_citations: bool = True) -> None:
        self.name = name
        self.quality = quality
        self.with_citations = with_citations

    async def run(self, question: str) -> SystemAnswer:
        if self.quality == "high":
            answer = (
                f"{question} In summary, the relevant distinctions and trade-offs are "
                "addressed directly and completely."
            )
        else:
            answer = "Short, incomplete answer."
        citations = ["stub://doc/1"] if self.with_citations else []
        return SystemAnswer(answer=answer, citations=citations)
