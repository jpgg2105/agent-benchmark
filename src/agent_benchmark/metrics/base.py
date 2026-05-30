from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import SystemAnswer, Task


@runtime_checkable
class Metric(Protocol):
    name: str

    async def score(self, task: Task, answer: SystemAnswer) -> float: ...
