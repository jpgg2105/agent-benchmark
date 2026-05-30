import asyncio

from agent_benchmark.metrics.coverage import CoverageMetric
from agent_benchmark.metrics.grounding import GroundingMetric
from agent_benchmark.models import SystemAnswer, Task


def _task(key_points, expected_sources=None):
    return Task(
        id="t1",
        domain="d",
        question="q",
        key_points=key_points,
        expected_sources=expected_sources or [],
    )


def test_coverage_full_and_zero():
    cov = CoverageMetric(threshold=0.6)
    task = _task(["index speeds up reads", "slows down writes"])
    full = SystemAnswer(answer="An index speeds up reads but slows down writes considerably.")
    none = SystemAnswer(answer="Totally unrelated text about cooking pasta.")
    assert asyncio.run(cov.score(task, full)) == 1.0
    assert asyncio.run(cov.score(task, none)) == 0.0


def test_coverage_partial():
    cov = CoverageMetric(threshold=0.6)
    task = _task(["index speeds up reads", "slows down writes"])
    half = SystemAnswer(answer="An index speeds up reads quickly.")
    assert asyncio.run(cov.score(task, half)) == 0.5


def test_grounding_presence():
    g = GroundingMetric()
    task = _task(["x"])
    assert asyncio.run(g.score(task, SystemAnswer(answer="a", citations=["doc/1"]))) == 1.0
    assert asyncio.run(g.score(task, SystemAnswer(answer="a", citations=[]))) == 0.0


def test_grounding_requires_expected_source_overlap_when_specified():
    g = GroundingMetric()
    task = _task(["x"], expected_sources=["doc/correct"])
    hit = SystemAnswer(answer="a", citations=["doc/correct"])
    miss = SystemAnswer(answer="a", citations=["doc/wrong"])
    assert asyncio.run(g.score(task, hit)) == 1.0
    assert asyncio.run(g.score(task, miss)) == 0.0
