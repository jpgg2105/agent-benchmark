from __future__ import annotations

from .base import Metric


def build_metrics(names: list[str], judge_provider: str | None = None) -> list[Metric]:
    metrics: list[Metric] = []
    for name in names:
        key = name.lower()
        if key == "coverage":
            from .coverage import CoverageMetric

            metrics.append(CoverageMetric())
        elif key == "grounding":
            from .grounding import GroundingMetric

            metrics.append(GroundingMetric())
        elif key == "correctness":
            from .judge import CorrectnessJudge

            metrics.append(CorrectnessJudge(_build_judge_llm(judge_provider)))
        else:
            raise ValueError(f"Unknown metric: {name!r} (expected coverage | grounding | correctness)")
    return metrics


def _build_judge_llm(provider: str | None):
    # Reuses the research-agent project's provider-agnostic client.
    from research_agent.config import Settings
    from research_agent.llm.factory import build_llm_client

    return build_llm_client(Settings(llm_provider=provider or "fake"))
