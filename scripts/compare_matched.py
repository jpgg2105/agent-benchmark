"""Compare the agent's published answers with the baseline rerun on matched prompts.

    python scripts/compare_matched.py results/published/hotpotqa-gemini-2026-10-31.json results/results-<timestamp>.json

Prints accuracy (LLM-judge correctness) for the agent, the original baseline and the matched
baseline, the paired difference agent minus each baseline with a 95% bootstrap interval, wins and
losses, and how often each baseline abstained. Writes the same numbers to
results/published/matched-prompts.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_benchmark.aggregate import bootstrap_ci  # noqa: E402

AGENT, ORIGINAL = "agent-2-rounds", "single-shot-k10"
# A keyword heuristic for "the sources don't say", applied the same way to both baselines. On the
# published run it finds 16 abstentions (14 of them among the agent's wins); RESULTS.md counts 17 and 13.
ABSTAIN = re.compile(
    r"insufficient|(do|does) not (contain|provide|mention|include|specify|state)|"
    r"not (enough|mentioned|provided|specified)|cannot (be )?(determined|determine|answer)|"
    r"no information|unable to (determine|answer)",
    re.I,
)


def _text(answer) -> str:
    a = answer.get("answer", answer) if isinstance(answer, dict) else answer
    return a if isinstance(a, str) else json.dumps(a)


def _by_task(results: list[dict], config: str) -> dict[str, dict]:
    return {r["task_id"]: r for r in results if r["config_name"] == config and r.get("error") is None}


def _compare(agent: dict, base: dict, ids: list[str]) -> dict:
    diffs = [agent[t]["scores"]["correctness"] - base[t]["scores"]["correctness"] for t in ids]
    wins = [t for t, x in zip(ids, diffs) if x > 0]
    # counted only when the answer was also judged wrong, so a correct answer with a hedge isn't an abstention
    abstained = {t for t in ids if base[t]["scores"]["correctness"] == 0 and ABSTAIN.search(_text(base[t]["answer"]))}
    return {
        "n": len(ids),
        "baseline_accuracy": round(sum(base[t]["scores"]["correctness"] for t in ids) / len(ids), 3),
        "difference": round(sum(diffs) / len(diffs), 3),
        "ci95": bootstrap_ci(diffs, 2000, 0),
        "agent_wins": len(wins),
        "agent_losses": sum(1 for x in diffs if x < 0),
        "baseline_abstained": len(abstained),
        "wins_where_baseline_abstained": len(abstained & set(wins)),
    }


def main(published_path: str, matched_path: str) -> None:
    published = json.loads(Path(published_path).read_text(encoding="utf-8"))["results"]
    matched_run = json.loads(Path(matched_path).read_text(encoding="utf-8"))["results"]
    agent, original = _by_task(published, AGENT), _by_task(published, ORIGINAL)
    matched_name = next(r["config_name"] for r in matched_run)
    matched = _by_task(matched_run, matched_name)
    ids = sorted(set(agent) & set(original) & set(matched))
    if not ids:
        sys.exit("No questions in common; check the two files.")

    agent_acc = round(sum(agent[t]["scores"]["correctness"] for t in ids) / len(ids), 3)
    out = {
        "questions": len(ids),
        "agent_accuracy": agent_acc,
        "original_baseline": _compare(agent, original, ids),
        "matched_baseline": _compare(agent, matched, ids),
        "matched_run_file": Path(matched_path).name,
    }
    for label, key in (("original prompt (may abstain)", "original_baseline"), ("matched prompt", "matched_baseline")):
        c = out[key]
        lo, hi = c["ci95"]
        print(f"\nBaseline, {label}")
        print(f"  accuracy {c['baseline_accuracy']:.0%}   agent {agent_acc:.0%}")
        print(f"  agent minus baseline {c['difference']:+.1%}  (95% CI {lo:+.1%} to {hi:+.1%})")
        print(f"  agent wins {c['agent_wins']}, losses {c['agent_losses']}")
        print(f"  baseline abstained {c['baseline_abstained']} times; {c['wins_where_baseline_abstained']} of the agent's wins")
    dest = Path(published_path).parent / "matched-prompts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
