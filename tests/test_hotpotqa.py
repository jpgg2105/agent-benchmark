import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from agent_benchmark.aggregate import bootstrap_ci, build_scorecards
from agent_benchmark.dataset import load_dataset
from agent_benchmark.leaderboard import render_markdown
from agent_benchmark.metrics.short_answer import EvidenceRecall, f1, normalize
from agent_benchmark.models import SystemAnswer, SystemConfig, Task, TaskResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_hotpotqa  # noqa: E402

EXAMPLES = [
    {"id": "x1", "question": "Which city is the tower designed by Gustave Eiffel in?", "answer": "Paris",
     "type": "bridge", "level": "hard", "supporting_titles": ["Eiffel Tower", "Gustave Eiffel"],
     "paragraphs": [("Eiffel Tower", "The Eiffel Tower is a lattice tower in Paris, France, designed by Gustave Eiffel's company."),
                    ("Gustave Eiffel", "Gustave Eiffel was a French civil engineer."),
                    ("Colosseum", "The Colosseum is an amphitheatre in Rome.")]},
    {"id": "x2", "question": "Are the Colosseum and the Eiffel Tower in the same country?", "answer": "no",
     "type": "comparison", "level": "hard", "supporting_titles": ["Colosseum", "Eiffel Tower"],
     "paragraphs": [("Colosseum", "The Colosseum is an amphitheatre in Rome, Italy."),
                    ("Eiffel Tower", "The Eiffel Tower is a lattice tower in Paris, France.")]},
]


def test_normalize_and_f1_follow_hotpotqa_rules():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"
    assert f1("Paris, France", "Paris") > 0.6
    assert f1("yes", "no") == 0.0
    assert f1("no", "no") == 1.0


def test_evidence_recall_counts_gold_titles_retrieved():
    task = Task(id="t", domain="d", question="q", expected_sources=["A", "B"])
    got = asyncio.run(EvidenceRecall().score(task, SystemAnswer(answer="x", retrieved=["a", "C"])))
    assert got == 0.5


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    xs = [1.0] * 60 + [0.0] * 40
    lo, hi = bootstrap_ci(xs, 1000, seed=1)
    assert lo < 0.6 < hi and hi - lo < 0.25
    assert bootstrap_ci(xs, 1000, seed=1) == [lo, hi]


def test_paired_difference_against_baseline():
    def res(cfg, tid, v):
        return TaskResult(task_id=tid, config_name=cfg, answer=SystemAnswer(answer="a", usage={"calls": 1, "input_tokens": 100, "output_tokens": 10}), scores={"answer_em": v})
    rs = [res("base", f"t{i}", 0.0) for i in range(20)] + [res("agent", f"t{i}", 1.0 if i < 10 else 0.0) for i in range(20)]
    cards = build_scorecards(rs, ["answer_em"], baseline="base", price_per_mtok={"input": 1.0, "output": 2.0}, samples=500)
    agent = next(c for c in cards if c.config_name == "agent")
    d = agent.vs_baseline["answer_em"]
    assert d["diff"] == 0.5 and d["wins"] == 10 and d["losses"] == 0 and d["low"] > 0
    assert abs(agent.mean_cost_usd - 0.00012) < 1e-9
    md = render_markdown(cards, ["answer_em"], "v1", baseline="base")
    assert "Paired difference" in md and "+0.500" in md


def test_prepare_writes_tasks_and_deduplicated_corpus(tmp_path):
    prepare_hotpotqa.write(EXAMPLES, tmp_path, 2)
    tasks, version = load_dataset(tmp_path / "hotpotqa_dev2.jsonl")
    assert [t.answer for t in tasks] == ["Paris", "no"]
    assert tasks[0].expected_sources == ["Eiffel Tower", "Gustave Eiffel"]
    corpus = [json.loads(l) for l in (tmp_path / "hotpotqa_corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(d["title"] for d in corpus) == ["Colosseum", "Eiffel Tower", "Gustave Eiffel"]


def test_end_to_end_offline_with_research_agent(tmp_path):
    if importlib.util.find_spec("research_agent") is None:
        return  # needs the sibling project installed; covered on a full install
    from agent_benchmark.metrics.factory import build_metrics
    from agent_benchmark.models import JudgeConfig
    from agent_benchmark.runner import BenchmarkRunner

    prepare_hotpotqa.write(EXAMPLES, tmp_path, 2)
    tasks, _ = load_dataset(tmp_path / "hotpotqa_dev2.jsonl")
    corpus = str(tmp_path / "hotpotqa_corpus.jsonl")
    configs = [
        SystemConfig(name="base", system="single_shot_rag",
                     params={"llm_provider": "fake", "retrieval_mode": "local", "local_corpus_path": corpus, "top_k": 2}),
        SystemConfig(name="agent", system="research_agent",
                     params={"llm_provider": "fake", "retrieval_mode": "local", "local_corpus_path": corpus,
                             "top_k": 2, "max_iterations": 2, "max_sub_questions": 2}),
    ]
    names = ["answer_em", "answer_f1", "correctness", "evidence_recall"]
    metrics = build_metrics(names, judge=JudgeConfig(provider="fake"))
    results = asyncio.run(BenchmarkRunner(metrics, concurrency=2).run(configs, tasks))
    assert all(r.error is None for r in results), [r.error for r in results]
    agent = [r for r in results if r.config_name == "agent"]
    assert all(r.answer.usage.get("calls", 0) >= 3 for r in agent)
    assert all(r.answer.trace and r.answer.trace["sub_questions"] for r in agent)
    assert all(0.0 <= r.scores["evidence_recall"] <= 1.0 for r in results)
