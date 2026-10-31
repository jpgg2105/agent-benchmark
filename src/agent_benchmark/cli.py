from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .aggregate import build_scorecards
from .config import load_run_config
from .dataset import load_dataset
from .leaderboard import render_console, render_markdown
from .metrics.factory import build_metrics
from .runner import BenchmarkRunner


def _load_checkpoint(path: Path) -> dict:
    """Successful results from earlier runs of this config, last one wins."""
    from .models import TaskResult

    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = TaskResult.model_validate_json(line)
                if r.error is None:
                    done[(r.config_name, r.task_id)] = r
    return done


async def _run(config_path: str, out_dir: str, limit: int | None, resume: bool, only: list[str] | None) -> None:
    cfg = load_run_config(config_path)
    tasks, version = load_dataset(cfg.dataset)
    if limit:
        tasks = tasks[:limit]
        version = f"{version}-first{limit}"
    metrics = build_metrics(cfg.metrics, judge_provider=cfg.judge_provider, judge=cfg.judge)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"checkpoint-{Path(config_path).stem}.jsonl"
    if not resume and ckpt.exists():
        ckpt.unlink()                      # a fresh run starts a fresh checkpoint
    done = _load_checkpoint(ckpt) if resume else {}

    selected = [s for s in cfg.systems if not only or s.name in only]
    print(f"Dataset {cfg.dataset}: {len(tasks)} tasks (version {version})")
    print(f"Running: {', '.join(s.name for s in selected)}" + (f"  (resuming, {len(done)} results kept)" if resume else ""))
    t0 = time.perf_counter()
    runner = BenchmarkRunner(metrics, concurrency=cfg.concurrency, checkpoint=ckpt, done=done)
    results = await runner.run(selected, tasks)
    # systems not run today still count if an earlier run finished them
    ids = {t.id for t in tasks}
    for s in cfg.systems:
        if s not in selected:
            results += [r for (c, t), r in done.items() if c == s.name and t in ids]
    cards = build_scorecards(
        results, cfg.metrics, baseline=cfg.baseline, price_per_mtok=cfg.price_per_mtok,
        samples=cfg.bootstrap_samples, seed=cfg.seed,
    )

    judge_usage = {}
    for m in metrics:
        llm = getattr(m, "llm", None) or getattr(getattr(m, "extractor", None), "llm", None)
        if llm is not None and getattr(llm, "usage", None) is not None:
            judge_usage = llm.usage.snapshot()
            break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = out / f"results-{stamp}.json"
    results_path.write_text(
        json.dumps(
            {
                "dataset": cfg.dataset,
                "dataset_version": version,
                "config": cfg.model_dump(),
                "metrics": cfg.metrics,
                "wall_clock_s": round(time.perf_counter() - t0, 1),
                "judge_usage": judge_usage,
                "scorecards": [c.model_dump() for c in cards],
                "results": [r.model_dump() for r in results],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    board = out / f"LEADERBOARD-{Path(config_path).stem}.md"
    board.write_text(render_markdown(cards, cfg.metrics, version, cfg.baseline), encoding="utf-8")
    print(render_console(cards, cfg.metrics))
    errors = sum(c.n_errors for c in cards)
    if errors:
        first = next(r for r in results if r.error)
        print(f"\n{errors} task(s) errored. First error ({first.config_name}, {first.task_id}): {first.error}")
        print("If you hit a daily quota, run the same command again with --resume after it resets.")
    print(f"\nWrote {results_path} and {board}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run the benchmark from a config file")
    run_p.add_argument("--config", default="configs/default.yaml")
    run_p.add_argument("--out", default="results")
    run_p.add_argument("--limit", type=int, default=None, help="Only the first N tasks (smoke test)")
    run_p.add_argument("--resume", action="store_true", help="Keep finished tasks from the last run of this config")
    run_p.add_argument("--systems", default=None, help="Comma-separated subset of system names to run now")

    args = parser.parse_args()
    if args.command == "run":
        only = [s.strip() for s in args.systems.split(",")] if args.systems else None
        asyncio.run(_run(args.config, args.out, args.limit, args.resume, only))


if __name__ == "__main__":
    main()
