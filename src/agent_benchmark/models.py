from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Task(BaseModel):
    """One benchmark item: a question plus a rubric for scoring answers."""

    id: str
    domain: str
    question: str
    key_points: list[str] = Field(default_factory=list)  # facts a good answer must cover
    reference_answer: str = ""
    difficulty: str = "medium"                            # easy | medium | hard
    expected_sources: list[str] = Field(default_factory=list)


class SystemAnswer(BaseModel):
    """What a system-under-test returns for a question."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    """One leaderboard entry: which system to run, with which parameters."""

    name: str
    system: str                                           # fake | single_shot_rag | research_agent
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """A full benchmark run, loaded from YAML."""

    dataset: str
    systems: list[SystemConfig]
    metrics: list[str] = Field(default_factory=lambda: ["coverage", "grounding"])
    concurrency: int = 4
    judge_provider: str | None = None                     # required if 'correctness' in metrics


class TaskResult(BaseModel):
    task_id: str
    config_name: str
    answer: SystemAnswer
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class ScoreCard(BaseModel):
    """Aggregated results for one system config across all tasks."""

    config_name: str
    n_tasks: int
    n_errors: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)  # metric name -> mean
    mean_latency_s: float = 0.0
