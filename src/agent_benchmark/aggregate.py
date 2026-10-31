from __future__ import annotations

import random
from collections import defaultdict

from .models import ScoreCard, TaskResult


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_ci(xs: list[float], samples: int = 2000, seed: int = 0) -> list[float]:
    """Percentile bootstrap 95% CI of the mean. Deterministic for a given seed."""
    if len(xs) < 2:
        m = _mean(xs)
        return [m, m]
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(_mean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    return [round(_pct(means, 0.025), 4), round(_pct(means, 0.975), 4)]


def build_scorecards(
    results: list[TaskResult],
    metric_names: list[str],
    *,
    baseline: str | None = None,
    price_per_mtok: dict[str, float] | None = None,
    samples: int = 2000,
    seed: int = 0,
) -> list[ScoreCard]:
    """Per-config means with bootstrap CIs, cost and latency, plus paired
    differences against the baseline config (same tasks, so the comparison
    removes per-question difficulty from the noise)."""
    by_config: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_config[r.config_name].append(r)

    cards: list[ScoreCard] = []
    for name, rs in by_config.items():
        ok = [r for r in rs if r.error is None]
        metrics, ci = {}, {}
        for m in metric_names:
            vals = [r.scores[m] for r in ok if m in r.scores]
            metrics[m] = round(_mean(vals), 4)
            ci[m] = bootstrap_ci(vals, samples, seed)
        lat = [r.answer.latency_s for r in ok]
        inp = [r.answer.usage.get("input_tokens", 0) for r in ok]
        out = [r.answer.usage.get("output_tokens", 0) for r in ok]
        calls = [r.answer.usage.get("calls", 0) for r in ok]
        cost = None
        if price_per_mtok and ok:
            cost = round(
                (_mean(inp) * price_per_mtok.get("input", 0) + _mean(out) * price_per_mtok.get("output", 0)) / 1e6, 6
            )
        cards.append(
            ScoreCard(
                config_name=name,
                n_tasks=len(rs),
                n_errors=len(rs) - len(ok),
                metrics=metrics,
                ci95=ci,
                mean_latency_s=round(_mean(lat), 3),
                p50_latency_s=round(_pct(lat, 0.5), 3),
                p95_latency_s=round(_pct(lat, 0.95), 3),
                mean_calls=round(_mean(calls), 2),
                mean_input_tokens=round(_mean(inp), 1),
                mean_output_tokens=round(_mean(out), 1),
                mean_cost_usd=cost,
            )
        )

    if baseline and baseline in by_config:
        base = {r.task_id: r for r in by_config[baseline] if r.error is None}
        for card in cards:
            if card.config_name == baseline:
                continue
            mine = {r.task_id: r for r in by_config[card.config_name] if r.error is None}
            common = sorted(set(base) & set(mine))
            for m in metric_names:
                d = [mine[t].scores.get(m, 0.0) - base[t].scores.get(m, 0.0) for t in common]
                if not d:
                    continue
                lo, hi = bootstrap_ci(d, samples, seed)
                card.vs_baseline[m] = {
                    "diff": round(_mean(d), 4), "low": lo, "high": hi,
                    "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d),
                    "n": len(d),
                }
    return cards
