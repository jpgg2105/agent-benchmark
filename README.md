# Agent Benchmark

> A reproducible benchmark and leaderboard for question-answering and research-agent systems. It measures the [research-agent](../research-agent) (a multi-agent system) against a single-shot RAG baseline on a fixed dataset, so you can prove the agent loop actually earns its complexity.

## Where this sits

This is the third piece of a three-project loop:

```
ai-document-analyst   ──▶   research-agent   ──▶   agent-benchmark
(RAG retrieval)             (multi-agent loop        (measures the agent
                             that calls the RAG        against a baseline,
                             system as a tool)          on a fixed dataset)
```

The first project retrieves. The second composes retrieval into a planning/critique/synthesis loop. This third project is the **evidence layer**: it runs several system configurations against the same questions, scores them, and publishes a leaderboard. Anyone can reproduce the numbers.

A benchmark you publish is different from "I evaluated my own app." It curates a dataset, defines metrics, compares configurations head-to-head, and ships a reproducible leaderboard others can run.

## What it measures

| Metric | What it captures | Needs an API key? |
| --- | --- | --- |
| **coverage** | Fraction of a question's rubric key-points present in the answer (lexical proxy) | No |
| **grounding** | Whether the answer cites a source — and, when a task specifies expected sources, whether the citation matches | No |
| **correctness** | LLM-as-judge: pass/fail against a gold answer, or a 0–1 rubric rating when there is none | Yes (optional) |
| **answer_em / answer_f1** | Exact match and token F1 of the short answer the response commits to (extracted by the judge model, scored with SQuAD/HotpotQA normalization) | Yes |
| **evidence_recall** | Share of the gold supporting documents the system retrieved | No |

`coverage` and `grounding` are deterministic and offline, so the whole harness — and its test suite — runs with no API key. `correctness` is an optional LLM judge you switch on in the config.

Every metric implements the same one-method interface (`async score(task, answer) -> float`), so adding a new one (faithfulness, citation precision, embedding-based coverage) is a single file.

## The dataset

`data/techdocs_qa.jsonl` is a hand-authored set of technical Q&A items spanning web, databases, systems, distributed systems, security, and AI engineering, at mixed difficulty. Each item carries a rubric of key-points used for scoring. The dataset is content-hashed on load, so every result is stamped with the exact dataset version it came from. See `data/DATASET.md` for the datasheet (provenance, schema, scoring notes, and limitations).

To benchmark retrieval grounding specifically, point the systems at your own document corpus and add `expected_sources` to each task.

## The systems under test

| System | What it is |
| --- | --- |
| `fake` | Deterministic offline stub — lets the harness and tests run with no keys or sibling projects |
| `single_shot_rag` | **Baseline**: one retrieve-then-answer pass, no planning or critique |
| `research_agent` | The project #2 multi-agent system, parameterised (iterations, sub-question budget, provider) |

`single_shot_rag` and `research_agent` reuse the research-agent project's building blocks, so that project must be importable to run them (see below). The point of the baseline is the comparison: if the agent loop doesn't beat single-shot on coverage and correctness, that's a finding worth knowing.

## Quick start

### Offline demo (no API key, no sibling projects)

```bash
pip install -e ".[dev]"
python -m agent_benchmark.cli run --config configs/default.yaml
```

This runs two offline stub systems and writes `results/LEADERBOARD-default.md` plus a timestamped `results-*.json`. The stub numbers only illustrate the leaderboard *format* — real numbers require real systems.

### Real comparison (research-agent vs single-shot)

```bash
# 1. Make the research-agent project importable on the same environment:
pip install -e ../research-agent          # adjust path to where it lives

# 2. Provide a key and edit configs/default.yaml to uncomment the real systems:
export OPENAI_API_KEY=sk-...

# 3. Run:
python -m agent_benchmark.cli run --config configs/default.yaml
```

Uncomment the `single-shot-rag`, `agent-1-iteration`, and `agent-2-iterations` blocks (and the `correctness` metric + `judge_provider`) in the config to produce a head-to-head leaderboard.

### The real run: 100 HotpotQA multi-hop questions

The techdocs set above is small and tests general knowledge. The headline comparison uses
[HotpotQA](https://hotpotqa.github.io/) (Yang et al., 2018), whose questions need two
documents to answer, which is exactly where planning and follow-up searches should help.

```bash
pip install -e "../research-agent[openai]" -e ".[dev]"
python scripts/prepare_hotpotqa.py                 # 100 questions (balanced bridge/comparison) + paragraph pool
python -m agent_benchmark.cli run --config configs/hotpotqa-offline.yaml   # free dry run, fake LLM
python -m agent_benchmark.cli run --config configs/hotpotqa.yaml --limit 5 # 5 real questions first
python -m agent_benchmark.cli run --config configs/hotpotqa.yaml           # the full run
```

What makes the comparison fair:

- **Same retriever for everyone.** A BM25 index over the pooled context paragraphs of all sampled
  questions (two gold paragraphs and eight distractors each, about 1,000 paragraphs). The agent and
  the baseline call the same tool; only what they do with it differs.
- **Same model for every system**, set once in the config.
- **Two baselines.** `single-shot-k5` reads as many paragraphs per search as the agent;
  `single-shot-k10` gets a larger context in one call, so the agent has to beat more reading,
  not just less.
- **Standard metrics.** Exact match and token F1 on the short answer (SQuAD/HotpotQA normalization),
  a pass/fail LLM judge against the gold answer, and evidence recall (share of the gold supporting
  paragraphs the system retrieved).
- **Uncertainty reported.** Every number comes with a 95% bootstrap interval, and each system is
  compared with the baseline question by question (paired differences, wins/ties/losses).
- **Cost and latency.** Tokens and LLM calls per question, cost from `price_per_mtok`, p50 and p95
  latency.

**On Gemini's free tier** (`configs/hotpotqa-gemini.yaml`): free quotas are small (measured in
September 2026 at about 15 requests a minute and 500 a day for Flash-Lite, and 20 a day for Flash),
so that config compares two systems, caps itself at 12 requests a minute and grades each answer
with one judge call. A full run needs about 1,100 calls, so it spans two or three days: when it
stops on the daily limit, run the same command with `--resume` after the quota resets (midnight
Pacific). `--systems name1,name2` runs a subset now and merges it with earlier runs.

On Windows, `scripts\run_gemini.ps1` does all of it in one command (setup, tests, data, dry run, the run itself, and `--resume` on later days):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1 -Limit 5   # first, 5 questions
powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1            # then once a day until done
```

Results land in `results/` as a `LEADERBOARD-<config>.md` and a full `results-*.json` with every
answer, score, token count and agent trace.

### Publish the leaderboard (HuggingFace Spaces)

```bash
pip install -e ".[app]"
python app.py        # serves a Gradio leaderboard from the latest results JSON
```

Push the repo to a HuggingFace Space (Gradio SDK) and it renders the leaderboard publicly — the artifact that puts the work in front of other practitioners.

## How a run works

```
config.yaml ─▶ load dataset (+ version hash)
            ─▶ build metrics      (coverage, grounding, [correctness])
            ─▶ for each system config:
                 build system ─▶ run all tasks concurrently ─▶ score each answer
            ─▶ aggregate into per-config scorecards
            ─▶ render LEADERBOARD-<config>.md + results-<timestamp>.json + console table
```

A system that errors on a task records the error for that task and keeps going; the scorecard reports how many tasks errored.

## Design decisions

- **Offline-first core.** The default metrics and stub systems need no network, so CI and a quick demo always work; the expensive, key-gated parts (LLM judge, real systems) are strictly opt-in.
- **Reproducibility by construction.** The dataset is content-hashed and every results file records that hash, the metric list, and per-task outputs — a result is never ambiguous about what produced it.
- **A baseline is mandatory, not optional.** The single-shot RAG system is the control; without it, agent numbers are unanchored.
- **One interface per extension point.** Systems and metrics are each a single `async` method behind a `Protocol`, so the runner never changes when you add a system or a metric.
- **Config over code.** Which systems run, with which parameters, and which metrics, all live in YAML.

## Lessons learned

- Publishing a *comparison* (agent vs baseline) is far more informative than a single system's scores — the delta is the story, and sometimes the baseline is closer than expected.
- A cheap lexical coverage metric correlates surprisingly well with the LLM judge on clear-cut questions, and costs nothing to run in CI; the judge earns its keep on nuanced answers.
- Content-hashing the dataset caught a silent case where an edited question would have made two result files quietly incomparable.
- Capturing per-task errors instead of failing the run matters the moment a provider rate-limits you mid-benchmark.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Runs fully offline. Covers dataset integrity, the coverage/grounding metrics (full, zero, partial, and expected-source matching), the end-to-end runner with leaderboard ranking, and graceful per-task error handling.

## Project layout

```
data/
├── techdocs_qa.jsonl    # the benchmark dataset
└── DATASET.md           # datasheet: provenance, schema, limitations
configs/
└── default.yaml         # which systems + metrics to run
src/agent_benchmark/
├── models.py            # Task, SystemAnswer, SystemConfig, TaskResult, ScoreCard
├── dataset.py           # load + content-hash the dataset
├── config.py            # YAML run config
├── runner.py            # run systems x tasks, score, capture errors
├── aggregate.py         # per-config scorecards
├── leaderboard.py       # results -> markdown + console
├── cli.py               # `agent-benchmark run --config ...`
├── systems/             # systems under test
│   ├── base.py          # System protocol
│   ├── fake.py          # offline stub
│   ├── single_shot_rag.py   # baseline (reuses research-agent)
│   └── research_agent.py    # adapter to project #2
└── metrics/             # scoring metrics
    ├── base.py          # Metric protocol
    ├── coverage.py      # rubric key-point coverage (offline)
    ├── grounding.py     # citation presence / match (offline)
    └── judge.py         # LLM-as-judge correctness (optional)
app.py                   # Gradio leaderboard for HuggingFace Spaces
tests/                   # offline pytest suite
```

## License

MIT
