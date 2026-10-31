from __future__ import annotations

import asyncio
import time

from .metrics.base import Metric
from .models import SystemAnswer, SystemConfig, Task, TaskResult
from .systems.factory import build_system


class BenchmarkRunner:
    """Runs each system config across every task and scores the answers.

    Tasks within a config run concurrently (bounded by a semaphore so we stay
    polite to rate-limited APIs). A system that raises on one task records an
    error for that task instead of aborting the whole run.
    """

    def __init__(self, metrics: list[Metric], concurrency: int = 4) -> None:
        self.metrics = metrics
        self.concurrency = max(1, concurrency)

    async def _run_one(self, system, config_name: str, task: Task, sem: asyncio.Semaphore) -> TaskResult:
        async with sem:
            t0 = time.perf_counter()
            try:
                answer = await system.run(task.question)
            except Exception as exc:  # one bad task shouldn't kill the run
                return TaskResult(
                    task_id=task.id,
                    config_name=config_name,
                    answer=SystemAnswer(answer=""),
                    error=f"{type(exc).__name__}: {exc}",
                )
            if not answer.latency_s:
                answer.latency_s = time.perf_counter() - t0
            try:
                scores = {m.name: await m.score(task, answer) for m in self.metrics}
            except Exception as exc:  # a judge outage is recorded, not fatal
                return TaskResult(
                    task_id=task.id, config_name=config_name, answer=answer,
                    error=f"scoring failed: {type(exc).__name__}: {exc}",
                )
            return TaskResult(task_id=task.id, config_name=config_name, answer=answer, scores=scores)

    async def run_config(self, config: SystemConfig, tasks: list[Task]) -> list[TaskResult]:
        system = build_system(config)
        sem = asyncio.Semaphore(self.concurrency)
        return list(await asyncio.gather(*(self._run_one(system, config.name, t, sem) for t in tasks)))

    async def run(self, configs: list[SystemConfig], tasks: list[Task]) -> list[TaskResult]:
        results: list[TaskResult] = []
        for config in configs:
            print(f"  running {config.name} on {len(tasks)} tasks...", flush=True)
            results.extend(await self.run_config(config, tasks))
        return results
