from __future__ import annotations

from .models import ScoreCard


def composite(card: ScoreCard) -> float:
    """Unweighted mean of a card's 0..1 metrics; the leaderboard sort key."""
    vals = list(card.metrics.values())
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def render_markdown(cards: list[ScoreCard], metric_names: list[str], dataset_version: str) -> str:
    ranked = sorted(cards, key=composite, reverse=True)
    cols = ["Rank", "Config"] + [m.capitalize() for m in metric_names] + [
        "Score",
        "Latency (s)",
        "Errors",
        "N",
    ]
    lines = [
        "# Leaderboard",
        "",
        f"_Dataset version: `{dataset_version}`_",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rank, c in enumerate(ranked, start=1):
        row = (
            [str(rank), c.config_name]
            + [f"{c.metrics.get(m, 0.0):.3f}" for m in metric_names]
            + [f"{composite(c):.3f}", f"{c.mean_latency_s:.3f}", str(c.n_errors), str(c.n_tasks)]
        )
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "_Score is the unweighted mean of the metric columns. Higher is better; lower latency is better._"]
    return "\n".join(lines)


def render_console(cards: list[ScoreCard], metric_names: list[str]) -> str:
    ranked = sorted(cards, key=composite, reverse=True)
    out = []
    for rank, c in enumerate(ranked, start=1):
        metric_str = "  ".join(f"{m}={c.metrics.get(m, 0.0):.3f}" for m in metric_names)
        out.append(
            f"{rank}. {c.config_name:<20} {metric_str}  score={composite(c):.3f}  "
            f"lat={c.mean_latency_s:.3f}s  errors={c.n_errors}  n={c.n_tasks}"
        )
    return "\n".join(out)
