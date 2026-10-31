from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Task(BaseModel):
    """One benchmark item: a question plus what a correct answer looks like."""

    id: str
    domain: str
    question: str
    key_points: list[str] = Field(default_factory=list)  # rubric facts (techdocs set)
    reference_answer: str = ""
    answer: str = ""                                      # gold short answer (HotpotQA set)
    kind: str = ""                                        # e.g. bridge | comparison
    difficulty: str = "medium"                            # easy | medium | hard
    expected_sources: list[str] = Field(default_factory=list)  # gold supporting titles


class SystemAnswer(BaseModel):
    """What a system-under-test returns for a question."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)   # titles of every source the system looked at
    latency_s: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)  # calls, input_tokens, output_tokens, ...
    trace: dict[str, Any] | None = None                  # agent plan / critiques, for inspection


class SystemConfig(BaseModel):
    """One leaderboard entry: which system to run, with which parameters."""

    name: str
    system: str                                           # fake | single_shot_rag | research_agent
    params: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    provider: str = "fake"                                # openai | anthropic | fake
    model: str | None = None
    base_url: str | None = None


class RunConfig(BaseModel):
    """A full benchmark run, loaded from YAML."""

    dataset: str
    systems: list[SystemConfig]
    metrics: list[str] = Field(default_factory=lambda: ["coverage", "grounding"])
    concurrency: int = 4
    judge_provider: str | None = None                     # legacy shorthand for judge.provider
    judge: JudgeConfig | None = None
    baseline: str | None = None                           # config name the others are compared against
    price_per_mtok: dict[str, float] | None = None        # {"input": usd, "output": usd} per million tokens
    bootstrap_samples: int = 2000
    seed: int = 0


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
    metrics: dict[str, float] = Field(default_factory=dict)       # metric -> mean
    ci95: dict[str, list[float]] = Field(default_factory=dict)    # metric -> [low, high] (bootstrap)
    mean_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    mean_calls: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_cost_usd: float | None = None
    vs_baseline: dict[str, dict[str, float]] = Field(default_factory=dict)  # metric -> {diff, low, high, wins, ties, losses, n}
