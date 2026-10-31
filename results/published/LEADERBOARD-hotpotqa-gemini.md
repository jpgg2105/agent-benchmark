# Leaderboard

_Dataset version: `eda7c50eb4e1`. Each cell: mean [95% bootstrap CI]._

| Rank | Config | answer_em | answer_f1 | correctness | evidence_recall | LLM calls | Tokens in/out | Cost per question (USD) | Latency p50 / p95 (s) | Errors | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | agent-2-rounds | 0.610 [0.51, 0.70] | 0.744 [0.67, 0.81] | 0.860 [0.79, 0.92] | 0.950 [0.92, 0.97] | 5.7 | 3,710 / 449 | 0.00000 | 50.29 / 85.89 | 0 | 100 |
| 2 | single-shot-k10 | 0.510 [0.41, 0.60] | 0.623 [0.54, 0.70] | 0.740 [0.65, 0.83] | 0.920 [0.88, 0.96] | 1.0 | 1,496 / 39 | 0.00000 | 10.29 / 12.96 | 0 | 100 |

## Paired difference vs `single-shot-k10`

_Same questions for both systems; difference = config minus baseline, with a 95% bootstrap CI. If the interval contains 0, the data can't tell them apart._

| Config | Metric | Difference [95% CI] | Wins / ties / losses |
| --- | --- | --- | --- |
| agent-2-rounds | answer_em | +0.100 [+0.02, +0.18] | 13 / 84 / 3 |
| agent-2-rounds | answer_f1 | +0.121 [+0.05, +0.20] | 17 / 78 / 5 |
| agent-2-rounds | correctness | +0.120 [+0.04, +0.20] | 15 / 82 / 3 |
| agent-2-rounds | evidence_recall | +0.030 [-0.01, +0.08] | 12 / 81 / 7 |
