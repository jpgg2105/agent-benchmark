from __future__ import annotations

from .models import ScoreCard


def composite(card: ScoreCard) -> float:
    """Unweighted mean of a card's 0..1 metrics; the leaderboard sort key."""
    vals = list(card.metrics.values())
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _cell(card: ScoreCard, m: str) -> str:
    v = card.metrics.get(m, 0.0)
    lo, hi = (card.ci95.get(m) or [v, v])
    return f"{v:.3f} [{lo:.2f}, {hi:.2f}]" if card.n_tasks > 1 else f"{v:.3f}"


def render_markdown(
    cards: list[ScoreCard], metric_names: list[str], dataset_version: str, baseline: str | None = None
) -> str:
    ranked = sorted(cards, key=composite, reverse=True)
    has_cost = any(c.mean_cost_usd is not None for c in cards)
    cols = ["Rank", "Config"] + [m for m in metric_names] + ["LLM calls", "Tokens in/out"]
    cols += (["Cost per question (USD)"] if has_cost else []) + ["Latency p50 / p95 (s)", "Errors", "N"]
    lines = [
        "# Leaderboard",
        "",
        f"_Dataset version: `{dataset_version}`. Each cell: mean [95% bootstrap CI]._",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rank, c in enumerate(ranked, start=1):
        row = [str(rank), c.config_name] + [_cell(c, m) for m in metric_names]
        row += [f"{c.mean_calls:.1f}", f"{c.mean_input_tokens:,.0f} / {c.mean_output_tokens:,.0f}"]
        row += ([f"{c.mean_cost_usd:.5f}" if c.mean_cost_usd is not None else "-"] if has_cost else [])
        row += [f"{c.p50_latency_s:.2f} / {c.p95_latency_s:.2f}", str(c.n_errors), str(c.n_tasks)]
        lines.append("| " + " | ".join(row) + " |")

    if baseline and any(c.vs_baseline for c in cards):
        lines += [
            "",
            f"## Paired difference vs `{baseline}`",
            "",
            "_Same questions for both systems; difference = config minus baseline, with a 95% bootstrap CI. "
            "If the interval contains 0, the data can't tell them apart._",
            "",
            "| Config | Metric | Difference [95% CI] | Wins / ties / losses |",
            "| --- | --- | --- | --- |",
        ]
        for c in ranked:
            for m, d in c.vs_baseline.items():
                lines.append(
                    f"| {c.config_name} | {m} | {d['diff']:+.3f} [{d['low']:+.2f}, {d['high']:+.2f}] | "
                    f"{d['wins']} / {d['ties']} / {d['losses']} |"
                )
    return "\n".join(lines) + "\n"


def render_console(cards: list[ScoreCard], metric_names: list[str]) -> str:
    ranked = sorted(cards, key=composite, reverse=True)
    out = []
    for rank, c in enumerate(ranked, start=1):
        metric_str = "  ".join(f"{m}={_cell(c, m)}" for m in metric_names)
        cost = f"  ${c.mean_cost_usd:.5f}/q" if c.mean_cost_usd is not None else ""
        out.append(
            f"{rank}. {c.config_name:<22} {metric_str}  calls={c.mean_calls:.1f}{cost}  "
            f"p50={c.p50_latency_s:.2f}s  errors={c.n_errors}  n={c.n_tasks}"
        )
    return "\n".join(out)
