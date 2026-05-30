from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from .aggregate import build_scorecards
from .config import load_run_config
from .dataset import load_dataset
from .leaderboard import render_console, render_markdown
from .metrics.factory import build_metrics
from .runner import BenchmarkRunner


async def _run(config_path: str, out_dir: str) -> None:
    cfg = load_run_config(config_path)
    tasks, version = load_dataset(cfg.dataset)
    metrics = build_metrics(cfg.metrics, judge_provider=cfg.judge_provider)

    runner = BenchmarkRunner(metrics, concurrency=cfg.concurrency)
    results = await runner.run(cfg.systems, tasks)
    cards = build_scorecards(results, cfg.metrics)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = out / f"results-{stamp}.json"
    results_path.write_text(
        json.dumps(
            {
                "dataset_version": version,
                "metrics": cfg.metrics,
                "scorecards": [c.model_dump() for c in cards],
                "results": [r.model_dump() for r in results],
            },
            indent=2,
        )
    )

    Path("LEADERBOARD.md").write_text(render_markdown(cards, cfg.metrics, version))
    print(render_console(cards, cfg.metrics))
    print(f"\nWrote {results_path} and LEADERBOARD.md")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run the benchmark from a config file")
    run_p.add_argument("--config", default="configs/default.yaml")
    run_p.add_argument("--out", default="results")

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(_run(args.config, args.out))


if __name__ == "__main__":
    main()
