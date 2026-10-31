from __future__ import annotations

from .base import Metric

_KNOWN = "coverage | grounding | correctness | answer_em | answer_f1 | evidence_recall"


def build_metrics(names: list[str], judge_provider: str | None = None, judge=None) -> list[Metric]:
    """Build metrics by name. LLM-backed metrics share one judge client."""
    metrics: list[Metric] = []
    llm = None
    extractor = None

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

            metrics.append(CorrectnessJudge(judge_llm()))
        elif key in {"answer_em", "answer_f1"}:
            from .short_answer import AnswerEM, AnswerF1, ShortAnswerExtractor

            extractor = extractor or ShortAnswerExtractor(judge_llm())
            metrics.append(AnswerEM(extractor) if key == "answer_em" else AnswerF1(extractor))
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
        return build_llm_client(agent_settings(llm_provider=judge.provider, model=judge.model, base_url=judge.base_url))
    return build_llm_client(agent_settings(llm_provider=provider or "fake"))
