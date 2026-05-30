from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import SystemAnswer


@runtime_checkable
class System(Protocol):
    name: str

    async def run(self, question: str) -> SystemAnswer: ...
