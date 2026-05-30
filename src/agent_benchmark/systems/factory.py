from __future__ import annotations

from ..models import SystemConfig
from .base import System


def build_system(config: SystemConfig) -> System:
    kind = config.system.lower()

    if kind == "fake":
        from .fake import FakeSystem

        return FakeSystem(name=config.name, **config.params)

    if kind == "single_shot_rag":
        from .single_shot_rag import SingleShotRAGSystem

        return SingleShotRAGSystem(name=config.name, **config.params)

    if kind == "research_agent":
        from .research_agent import ResearchAgentSystem

        return ResearchAgentSystem(name=config.name, **config.params)

    raise ValueError(
        f"Unknown system: {config.system!r} (expected 'fake', 'single_shot_rag', or 'research_agent')"
    )
