"""HuggingFace Spaces leaderboard app.

Reads the most recent results JSON and renders the leaderboard as a table.
Run locally with: pip install ".[app]" && python app.py
"""
from __future__ import annotations

import glob
import json

import gradio as gr  # optional dependency: pip install gradio


def _latest_results() -> dict | None:
    files = sorted(glob.glob("results/results-*.json"))
    if not files:
        return None
    return json.loads(open(files[-1]).read())


def _composite(card: dict) -> float:
    vals = list(card["metrics"].values())
    return sum(vals) / len(vals) if vals else 0.0


def _rows(data: dict | None, metric_names: list[str]) -> list[list]:
    if not data:
        return [["Run `python -m agent_benchmark.cli run` first"] + [""] * (len(metric_names) + 2)]
    rows = []
    for c in sorted(data["scorecards"], key=_composite, reverse=True):
        rows.append(
            [c["config_name"]]
            + [round(c["metrics"].get(m, 0.0), 3) for m in metric_names]
            + [round(_composite(c), 3), round(c["mean_latency_s"], 3)]
        )
    return rows


def build_demo() -> "gr.Blocks":
    data = _latest_results()
    metric_names = data["metrics"] if data else ["coverage", "grounding"]
    headers = ["Config"] + [m.capitalize() for m in metric_names] + ["Score", "Latency (s)"]
    with gr.Blocks(title="Agent Benchmark Leaderboard") as demo:
        gr.Markdown("# Agent Benchmark Leaderboard")
        gr.Dataframe(value=_rows(data, metric_names), headers=headers, interactive=False)
        if data:
            gr.Markdown(f"_Dataset version: `{data['dataset_version']}`_")
    return demo


if __name__ == "__main__":
    build_demo().launch()
