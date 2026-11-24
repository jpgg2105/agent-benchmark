# Results: does a four-agent loop beat a single LLM call?

**Yes on bridge questions, by 18 points. Overall by 8 points, with an interval that reaches zero.**
On 100 HotpotQA questions the research agent (planner, researchers, critic, synthesizer) answered
86% correctly. The first run (31 October) compared it with a single retrieve-and-answer call that
reads 10 paragraphs and got 74%, a +12 point gap. But only the baseline's prompt invited it to say
the sources were insufficient, and 13 of the agent's 15 wins were questions where it did. Rerun with
the same rule for both (24 November), the baseline got 78%: the difference is **+8 points, 95% CI
[0, +17]**, with 13 wins and 5 losses for the agent. All of it is on "bridge" questions, where the
answer needs a second search for something the first one revealed: 90% against 72%, **+18 [+6, +30]**.
On comparison questions the two are level (82% against 84%).

Runs: 31 October 2026 (both systems) and 24 November 2026 (baseline with matched prompts) · model `gemini-3.5-flash-lite` (free tier) for every system and for the
judge · dataset version `eda7c50eb4e1` · full per-question output in
[`results/published/hotpotqa-gemini-2026-10-31.json`](results/published/hotpotqa-gemini-2026-10-31.json)
and [`results/published/hotpotqa-gemini-matched-2026-11-24.json`](results/published/hotpotqa-gemini-matched-2026-11-24.json).

## Setup

- **Questions:** 100 from the HotpotQA dev set (distractor setting), 50 bridge and 50 comparison,
  fixed seed. See [`data/DATASET.md`](data/DATASET.md).
- **Retrieval:** the same in-process BM25 index for both systems, over the 1,000-odd paragraphs
  that come with the sampled questions (two gold paragraphs and eight distractors each).
- **Systems:**
  - `single-shot-k10`: one search for the question, top 10 paragraphs, one LLM call.
  - `agent-2-rounds`: the planner writes up to 3 sub-questions; each is searched (top 5) and
    answered; the critic either accepts or asks follow-ups for one more round; the synthesizer
    writes the answer.
- **Scoring:** exact match and token F1 on the extracted short answer (SQuAD/HotpotQA
  normalization), a pass/fail LLM judge against the gold answer, and evidence recall (share of the
  gold supporting paragraphs the system retrieved). 95% bootstrap intervals; paired differences on
  the same questions.

## Results of the first run (31 October)

This is the original comparison, with the asymmetric prompt. The next section reruns the baseline
without it.


| | single-shot-k10 | agent-2-rounds | Paired difference [95% CI] |
| --- | --- | --- | --- |
| Correct (judge) | 74% [65, 83] | **86%** [79, 92] | **+12** [+4, +20] |
| Exact match | 51% [41, 60] | **61%** [51, 70] | +10 [+2, +18] |
| Token F1 | 0.62 [0.54, 0.70] | **0.74** [0.67, 0.81] | +0.12 [+0.05, +0.20] |
| Evidence recall | 0.92 [0.88, 0.96] | 0.95 [0.92, 0.97] | +0.03 [−0.01, +0.08] |
| LLM calls per question | 1 | 5.7 | |
| Tokens per question, in / out | 1,496 / 39 | 3,710 / 449 | |

Question by question on correctness: the agent wins 15, loses 3, ties 82.

**By question type**

| | single-shot-k10 | agent-2-rounds | Difference [95% CI] |
| --- | --- | --- | --- |
| Bridge (50) | 70% | 90% | **+20** [+10, +32] |
| Comparison (50) | 78% | 82% | +4 [−6, +16] |

**Latency** is not reported as a result: both systems ran under a 12-requests-per-minute cap to stay
inside the free quota, so most of the measured time is waiting. Calls and tokens are the fair cost
comparison.

## Rerun with matched prompts (24 November)

The baseline's prompt said "If the sources are insufficient, say so plainly"; the agent's synthesizer
had no such line, only "Do not introduce facts that are not present in the findings." The rerun gives
the baseline that same rule instead (`abstain: false` in
[`configs/hotpotqa-gemini-matched.yaml`](configs/hotpotqa-gemini-matched.yaml)). Everything else is
unchanged: model, judge, retriever, 10 paragraphs, the same 100 questions. The agent's answers are the
published ones; [`scripts/compare_matched.py`](scripts/compare_matched.py) does the comparison and
writes [`results/published/matched-prompts.json`](results/published/matched-prompts.json).

| Correct (judge) | Baseline | Agent | Paired difference [95% CI] | Agent wins / losses |
| --- | --- | --- | --- | --- |
| Original baseline prompt | 74% | 86% | +12 [+4, +20] | 15 / 3 |
| **Matched prompts** | **78%** | **86%** | **+8 [0, +17]** | 13 / 5 |
| Matched, bridge (50) | 72% | 90% | **+18 [+6, +30]** | |
| Matched, comparison (50) | 84% | 82% | −2 [−12, +8] | |

- **About a third of the original gap came from the instruction.** Without the invitation, the
  baseline answered 4 more questions correctly (8 verdicts changed). An exact McNemar test on the 18
  questions where the systems disagree gives p = 0.10, against 0.008 in the first run.
- **The baseline still gave up 10 times** without being told it could (keyword count, answers judged
  wrong), and 7 of the agent's 13 wins are among them. On those questions the information really was
  missing from one search, which is what the agent's extra searches fix.
- **The agent's advantage is bridge questions, and it holds:** +18 points with an interval well clear
  of zero. On comparison questions a single call is as good and costs a sixth of the calls.
- One rerun of one system: some of the 4-point change is run-to-run variation of the model, which
  wasn't measured separately.

## Where the gain comes from (first run)

- **The baseline gives up; the agent asks the second question.** 13 of the agent's 15 wins are
  questions where the baseline answered that the sources were insufficient (it did that 17 times;
  the agent 7, counted by keyword). In 11 of those 15, the single search had retrieved only one of the two gold
  paragraphs; the agent's searches found both in 14.
- **Example.** "What year did the prison where William Penn wrote *No Cross, No Crown* stop being
  used as a prison?" The baseline found the Tower of London but no closing date. The agent's
  sub-questions were "Where was William Penn imprisoned…?" and "In what year did this prison stop
  being used as a prison?"; the second search returned the Tower of London paragraph, and it
  answered 1952.
- **The critic's second round lands on the hardest questions.** It accepted the first round for 74
  questions (agent 89% vs baseline 81% on those) and asked for a second round on 26 (agent 77% vs
  baseline 54%). 6 of the 15 wins came from second-round questions. This is an observation, not a
  measured effect of the critic: the critic chose those questions itself, and there is no
  agent-without-critic run to compare against.
- **Overall evidence recall barely moves** (+3 points, not significant): on most questions one
  search of 10 paragraphs already finds both hops. The agent's advantage is concentrated in the
  minority where it doesn't.

## Where the agent loses (first run)

- **Over-answering comparisons.** "Which is a flowering plant, Pueraria or Pleiospilos?" The
  baseline said Pleiospilos; the agent said both. Writing a long answer invites claims the sources
  don't support.
- **Vague follow-ups.** The planner writes all sub-questions before any search, so a later one can't
  name what an earlier one found ("this prison"). It worked in the example above because BM25
  matched the word "prison"; in other cases the second search drifts.

## How far to trust the judge

The same model answers and judges, so a second LLM grader, from a different model family than the
judge, graded 40 answers (20 per system) against the gold answers, without seeing which system wrote each answer or
what the judge said. The sample over-represents the questions where the two systems disagreed:
13 of the 40 answers come from those 18 questions, which is where a judge error would change the
result. It agreed with the judge on **39 of 40** (Cohen's κ = 0.92). The one
disagreement: the gold answer is "Plymouth Gin" and the agent said "gin"; the judge marked it
wrong, the second evaluator right. The judge was, if anything, slightly strict. The grades are in
[`results/published/second-grader-40.csv`](results/published/second-grader-40.csv).

HotpotQA has some debatable labels (for example, it expects "no" to "Are Karl Shapiro and Elie
Wiesel both American writers?", though Wiesel became a US citizen). Both systems lose those
equally, so they don't change the difference.

## Limits

- 100 questions: intervals are about ±9 points per system. The overall difference is clear; the
  comparison-question difference is not.
- One model, a small one. A stronger model may close the gap for the baseline, or widen it.
- Wikipedia-style questions with short answers. Real document Q&A has longer answers and messier
  retrieval.
- **The first run's prompts weren't symmetric on abstaining**, which inflated its gap from about 8 to
  12 points. The rerun above fixes it for the baseline; the agent wasn't rerun.
- This benchmark predates the projects where hypotheses were written down before running. The
  headline comparison and metrics were fixed in advance; the breakdowns in "Where the gain comes
  from" were done after seeing the results.

## What I'd change next

1. **Sequential hops:** let each sub-question see earlier findings, so "this prison" becomes "the
   Tower of London".
2. **Answer length discipline** in the synthesizer for comparison questions.
3. **Cost-aware routing:** a single call first, and the agent only for bridge-style questions or when
   that call says the sources are insufficient. On the first run this simulated to 87% correct at
   about 2.1 calls per question (after the fact, so a hypothesis to test); with matched prompts the
   baseline abstains less often, so the trigger should be the question type rather than abstention.

## Reproduce

```bash
python scripts/prepare_hotpotqa.py
python -m agent_benchmark.cli run --config configs/hotpotqa-gemini.yaml   # GEMINI_API_KEY in .env
python -m agent_benchmark.cli run --config configs/hotpotqa-gemini-matched.yaml
python scripts/compare_matched.py results/published/hotpotqa-gemini-2026-10-31.json results/results-<timestamp>.json
```

On Windows: `powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1`, then
`powershell -ExecutionPolicy Bypass -File scripts\run_matched.ps1`.
