from __future__ import annotations

from ..models import SystemAnswer, Task


class GroundingMetric:
    """Whether the answer is backed by at least one citation (1.0) or not (0.0).

    When a task specifies ``expected_sources``, this also requires that the
    returned citations overlap them, rewarding correct attribution rather than
    merely citing something.
    """

    name = "grounding"

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        if not answer.citations:
            return 0.0
        if not task.expected_sources:
            return 1.0
        cited = {c.lower() for c in answer.citations}
        expected = {e.lower() for e in task.expected_sources}
        return 1.0 if (cited & expected) else 0.0
