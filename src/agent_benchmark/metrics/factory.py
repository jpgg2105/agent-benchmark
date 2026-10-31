from __future__ import annotations

from .base import Metric

_KNOWN = "coverage | grounding | correctness | answer_em | answer_f1 | evidence_recall"


def build_metrics(names: list[str], judge_provider: str | None = None, judge=None) -> list[Metric]:
    """Build metrics by name. LLM-backed metrics share one judge client."""
    metrics: list[Metric] = []
    llm = None
    extractor = None
    combined = bool(judge is not None and getattr(judge, "combined", False))

    def get_extractor():
        nonlocal extractor
        if extractor is None:
            from .short_answer import CombinedGoldJudge, ShortAnswerExtractor

            extractor = (CombinedGoldJudge if combined else ShortAnswerExtractor)(judge_llm())
        return extractor

    def judge_llm():
        nonlocal llm
        if llm is None:
            llm = _build_judge_llm(judge, judge_provider)
        return llm

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

            if combined:
                from .short_answer import CombinedCorrectness

                metrics.append(CombinedCorrectness(get_extractor(), CorrectnessJudge(judge_llm())))
            else:
                metrics.append(CorrectnessJudge(judge_llm()))
        elif key in {"answer_em", "answer_f1"}:
            from .short_answer import AnswerEM, AnswerF1

            metrics.append(AnswerEM(get_extractor()) if key == "answer_em" else AnswerF1(get_extractor()))
        elif key == "evidence_recall":
            from .short_answer import EvidenceRecall

            metrics.append(EvidenceRecall())
        else:
            raise ValueError(f"Unknown metric: {name!r} (expected {_KNOWN})")
    return metrics


def _build_judge_llm(judge=None, provider: str | None = None):
    # Reuses the research-agent project's provider-agnostic client.
    from research_agent.llm.factory import build_llm_client

    from ..systems._settings import agent_settings

    if judge is not None:
        return build_llm_client(agent_settings(
            llm_provider=judge.provider, model=judge.model, base_url=judge.base_url,
            rpm=judge.rpm, reasoning_effort=judge.reasoning_effort,
        ))
    return build_llm_client(agent_settings(llm_provider=provider or "fake"))
