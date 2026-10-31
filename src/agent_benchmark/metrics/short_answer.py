from __future__ import annotations

import re
import string
from collections import Counter

from ..models import SystemAnswer, Task

_EXTRACT_SYSTEM = (
    "You extract the final answer from a response. Given a QUESTION and a RESPONSE, "
    "output only the short answer the RESPONSE commits to: an entity, a name, a number, "
    "a date, or yes/no. Copy it from the response; do not answer the question yourself "
    "and do not correct the response. If the response gives no answer, output NONE."
)


def normalize(s: str) -> str:
    """SQuAD/HotpotQA answer normalization: lowercase, drop punctuation, articles and extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    # HotpotQA rule: yes/no/noanswer only count as exact matches
    if (normalize(pred) in {"yes", "no", "noanswer"} or normalize(gold) in {"yes", "no", "noanswer"}) and normalize(pred) != normalize(gold):
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


class ShortAnswerExtractor:
    """Turns a free-form response into the short answer it commits to, with one
    LLM call per (task, response), cached so EM and F1 share it. Both systems go
    through the same extractor, so neither is favored by its output format."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self._cache: dict[tuple[str, int], str] = {}

    async def extract(self, task: Task, answer: SystemAnswer) -> str:
        key = (task.id, hash(answer.answer))
        if key not in self._cache:
            if not answer.answer.strip():
                self._cache[key] = ""
            else:
                raw = await self.llm.complete(
                    system=_EXTRACT_SYSTEM,
                    user=f"QUESTION:\n{task.question}\n\nRESPONSE:\n{answer.answer}",
                    temperature=0.0,
                )
                out = raw.strip().strip('"').splitlines()[0].strip() if raw.strip() else ""
                self._cache[key] = "" if out.upper() == "NONE" else out
        return self._cache[key]


_COMBINED_SYSTEM = (
    "You grade a RESPONSE to a QUESTION. Do two things.\n"
    "1. short_answer: copy the short answer the RESPONSE commits to (an entity, name, number, "
    "date, or yes/no), exactly as the response gives it. Base this ONLY on the response; never "
    "copy it from the gold answer. If the response gives no answer, use \"NONE\".\n"
    "2. correct: true if the response's answer means the same as the GOLD ANSWER (aliases, extra "
    "correct detail and formatting are fine), false if it is wrong, hedges between options or says "
    "the information is unavailable.\n"
    'Reply with only JSON: {"short_answer": "...", "correct": true}'
)


class CombinedGoldJudge(ShortAnswerExtractor):
    """One judge call per response that returns both the extracted short answer
    and the pass/fail verdict: half the calls, for tight free-tier quotas.
    The extraction instruction forbids using the gold answer."""

    def __init__(self, llm) -> None:
        super().__init__(llm)
        self._verdicts: dict[tuple[str, int], float] = {}

    async def _assess(self, task: Task, answer: SystemAnswer) -> None:
        import json

        key = (task.id, hash(answer.answer))
        if key in self._cache:
            return
        if not answer.answer.strip():
            self._cache[key], self._verdicts[key] = "", 0.0
            return
        raw = await self.llm.complete(
            system=_COMBINED_SYSTEM,
            user=f"QUESTION:\n{task.question}\n\nGOLD ANSWER:\n{task.answer}\n\nRESPONSE:\n{answer.answer}",
            temperature=0.0,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        try:
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
            short = str(data.get("short_answer", "")).strip()
            verdict = 1.0 if data.get("correct") in (True, "true", 1, "1") else 0.0
        except (ValueError, AttributeError):
            short, verdict = "", 0.0
        self._cache[key] = "" if short.upper() == "NONE" else short
        self._verdicts[key] = verdict

    async def extract(self, task: Task, answer: SystemAnswer) -> str:
        await self._assess(task, answer)
        return self._cache[(task.id, hash(answer.answer))]

    async def verdict(self, task: Task, answer: SystemAnswer) -> float:
        await self._assess(task, answer)
        return self._verdicts[(task.id, hash(answer.answer))]


class CombinedCorrectness:
    name = "correctness"

    def __init__(self, judge: CombinedGoldJudge, fallback) -> None:
        self.judge, self.fallback = judge, fallback

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        if task.answer:
            return await self.judge.verdict(task, answer)
        return await self.fallback.score(task, answer)


class AnswerEM:
    name = "answer_em"

    def __init__(self, extractor: ShortAnswerExtractor) -> None:
        self.extractor = extractor

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        if not task.answer:
            return 0.0
        pred = await self.extractor.extract(task, answer)
        return 1.0 if normalize(pred) == normalize(task.answer) else 0.0


class AnswerF1:
    name = "answer_f1"

    def __init__(self, extractor: ShortAnswerExtractor) -> None:
        self.extractor = extractor

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        if not task.answer:
            return 0.0
        pred = await self.extractor.extract(task, answer)
        return round(f1(pred, task.answer), 4)


class EvidenceRecall:
    """Share of the gold supporting documents the system retrieved at any point.

    For multi-hop questions this is where an agent can earn its keep: a single
    search often finds one hop, decomposition can find both."""

    name = "evidence_recall"

    async def score(self, task: Task, answer: SystemAnswer) -> float:
        if not task.expected_sources:
            return 0.0
        got = {normalize(t) for t in (answer.retrieved or answer.citations)}
        gold = {normalize(t) for t in task.expected_sources}
        return round(len(got & gold) / len(gold), 4)
