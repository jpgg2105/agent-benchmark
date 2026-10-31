from __future__ import annotations

import re

from ..models import SystemAnswer, Task

_RUBRIC_SYSTEM = (
    "You are a strict evaluator. Rate from 0.0 to 1.0 how well the ANSWER addresses "
    "the QUESTION and is consistent with the listed KEY POINTS. Reward correctness "
    "and completeness; penalise omissions and contradictions. Respond with only a "
    "single number between 0 and 1."
)

_GOLD_SYSTEM = (
    "You are a strict grader. Decide whether the RESPONSE correctly answers the QUESTION, "
    "using the GOLD ANSWER as ground truth. Accept answers that mean the same thing "
    "(aliases, extra correct detail, different formatting). Reject answers that are wrong, "
    "hedge between options, or say the information is unavailable. Output only 1 (correct) or 0 (incorrect)."
)


class CorrectnessJudge:
    """LLM-as-judge correctness.

    With a gold short answer (HotpotQA) it is a binary pass/fail check, which is
    easier to validate against human labels than a 0-1 scale. Without one
    (techdocs set) it falls back to a 0-1 rubric rating against key-points.
    """

    name = "correctness"

    def __init__(self, llm) -> None:
        self.llm = llm

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        if not answer.answer.strip():
            return 0.0
        if task.answer:
            raw = await self.llm.complete(
                system=_GOLD_SYSTEM,
                user=f"QUESTION:\n{task.question}\n\nGOLD ANSWER:\n{task.answer}\n\nRESPONSE:\n{answer.answer}",
                temperature=0.0,
            )
            m = re.search(r"[01]", raw)
            return float(m.group()) if m else 0.0
        key_points = "\n".join(f"- {k}" for k in task.key_points)
        raw = await self.llm.complete(
            system=_RUBRIC_SYSTEM,
            user=f"QUESTION:\n{task.question}\n\nKEY POINTS:\n{key_points}\n\nANSWER:\n{answer.answer}",
            temperature=0.0,
        )
        match = re.search(r"[01](?:\.\d+)?", raw)
        return max(0.0, min(1.0, float(match.group()))) if match else 0.0
