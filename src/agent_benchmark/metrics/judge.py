from __future__ import annotations

import re

from ..models import SystemAnswer, Task

_SYSTEM = (
    "You are a strict evaluator. Rate from 0.0 to 1.0 how well the ANSWER addresses "
    "the QUESTION and is consistent with the listed KEY POINTS. Reward correctness "
    "and completeness; penalise omissions and contradictions. Respond with only a "
    "single number between 0 and 1."
)


class CorrectnessJudge:
    """LLM-as-judge correctness score against the task's key-points.

    Optional: needs a real LLM client (and therefore an API key). Wired in via
    the 'correctness' metric name plus ``judge_provider`` in the run config.
    """

    name = "correctness"

    def __init__(self, llm) -> None:
        self.llm = llm

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        key_points = "\n".join(f"- {k}" for k in task.key_points)
        user = (
            f"QUESTION:\n{task.question}\n\n"
            f"KEY POINTS:\n{key_points}\n\n"
            f"ANSWER:\n{answer.answer}"
        )
        raw = await self.llm.complete(system=_SYSTEM, user=user, temperature=0.0)
        match = re.search(r"[01](?:\.\d+)?", raw)
        if not match:
            return 0.0
        return max(0.0, min(1.0, float(match.group())))
