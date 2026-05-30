import asyncio

from agent_benchmark.aggregate import build_scorecards
from agent_benchmark.leaderboard import composite, render_markdown
from agent_benchmark.metrics.coverage import CoverageMetric
from agent_benchmark.metrics.grounding import GroundingMetric
from agent_benchmark.models import SystemConfig, Task
from agent_benchmark.runner import BenchmarkRunner


def _tasks():
    # Key-points are words that appear in the question, so the "high" stub
    # (which echoes the question) covers them while the "low" stub does not.
    return [
        Task(id="t1", domain="d", question="explain database index trade-offs clearly",
             key_points=["database index", "trade-offs"]),
        Task(id="t2", domain="d", question="describe http status codes precisely",
             key_points=["http status", "codes"]),
    ]


def test_end_to_end_offline_and_ranking():
    configs = [
        SystemConfig(name="strong-stub", system="fake", params={"quality": "high", "with_citations": True}),
        SystemConfig(name="weak-stub", system="fake", params={"quality": "low", "with_citations": False}),
    ]
    runner = BenchmarkRunner([CoverageMetric(), GroundingMetric()], concurrency=2)
    results = asyncio.run(runner.run(configs, _tasks()))

    assert len(results) == len(configs) * len(_tasks())   # configs x tasks
    assert all(r.error is None for r in results)

    cards = build_scorecards(results, ["coverage", "grounding"])
    by_name = {c.config_name: c for c in cards}
    assert composite(by_name["strong-stub"]) > composite(by_name["weak-stub"])
    assert by_name["strong-stub"].metrics["grounding"] == 1.0
    assert by_name["weak-stub"].metrics["grounding"] == 0.0

    md = render_markdown(cards, ["coverage", "grounding"], "abc123abc123")
    assert "strong-stub" in md and "weak-stub" in md
    assert md.index("strong-stub") < md.index("weak-stub")   # strong ranked first


def test_failing_system_records_error_not_crash():
    # 'single_shot_rag' will fail at runtime here because the research-agent
    # package isn't installed in this test environment -- the runner must
    # capture that as a per-task error rather than raising.
    configs = [SystemConfig(name="broken", system="single_shot_rag", params={})]
    runner = BenchmarkRunner([CoverageMetric()], concurrency=1)
    results = asyncio.run(runner.run(configs, _tasks()))
    assert len(results) == len(_tasks())
    assert all(r.error is not None for r in results)
