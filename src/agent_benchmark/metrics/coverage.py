from __future__ import annotations

from ..models import SystemAnswer, Task
from ._text import tokens


class CoverageMetric:
    """Fraction of a task's key-points that the answer covers.

    A lexical proxy: a key-point counts as covered when at least ``threshold``
    of its content tokens appear in the answer. Cheap, deterministic, and
    offline. For higher fidelity, swap in an embedding- or LLM-based check --
    the Metric interface is identical.
    """

    name = "coverage"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        if not task.key_points:
            return 1.0
        answer_tokens = set(tokens(answer.answer))

        def covered(key_point: str) -> bool:
            kp = tokens(key_point)
            if not kp:
                return True
            hits = sum(1 for t in kp if t in answer_tokens)
            return (hits / len(kp)) >= self.threshold

        n_covered = sum(1 for kp in task.key_points if covered(kp))
        return round(n_covered / len(task.key_points), 4)
