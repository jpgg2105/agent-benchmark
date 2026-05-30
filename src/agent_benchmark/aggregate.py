from __future__ import annotations

from collections import defaultdict

from .models import ScoreCard, TaskResult


def build_scorecards(results: list[TaskResult], metric_names: list[str]) -> list[ScoreCard]:
    """Collapse per-task results into one mean scorecard per system config."""
    by_config: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_config[r.config_name].append(r)

    cards: list[ScoreCard] = []
    for name, rs in by_config.items():
        ok = [r for r in rs if r.error is None]
        metrics: dict[str, float] = {}
        for m in metric_names:
            vals = [r.scores[m] for r in ok if m in r.scores]
            metrics[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
        mean_latency = round(sum(r.answer.latency_s for r in ok) / len(ok), 4) if ok else 0.0
        cards.append(
            ScoreCard(
                config_name=name,
                n_tasks=len(rs),
                n_errors=len(rs) - len(ok),
                metrics=metrics,
                mean_latency_s=mean_latency,
            )
        )
    return cards
