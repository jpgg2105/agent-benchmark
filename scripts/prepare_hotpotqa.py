"""Build the HotpotQA benchmark files: a fixed random sample of questions plus
the pool of paragraphs the systems retrieve from.

    python scripts/prepare_hotpotqa.py              # 100 questions, seed 13
    python scripts/prepare_hotpotqa.py --n 200

Source: HotpotQA (Yang et al., 2018), "distractor" setting, dev split, via the
Hugging Face dataset viewer API (no extra packages needed). Licence CC BY-SA 4.0.
Offline alternative: download hotpot_dev_distractor_v1.json from the HotpotQA
site yourself and pass --from-file path/to/it.

The API is rate-limited, so rows are fetched in pages of 100 at random offsets
(chosen by --seed), cached in data/.hotpotqa_cache.json, and the balanced sample is
drawn from that pool. Re-running after an interruption reuses the cache.

Writes:
  data/hotpotqa_dev<N>.jsonl   one task per line (question, gold answer, gold supporting titles)
  data/hotpotqa_corpus.jsonl   every context paragraph of the sampled questions (2 gold + 8
                               distractors each, deduplicated by title): the retrieval pool
The sample is balanced between "bridge" and "comparison" questions and fixed by --seed,
so anyone can rebuild exactly the same files.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"}
TOTAL_DEV = 7405


PAGE = 100


def _get(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode({**DATASET, **params})}"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # 429 rate limit or a transient 5xx: back off
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait = int(retry_after) if retry_after and retry_after.isdigit() else min(15 * 2 ** attempt, 300)
            print(f"  waiting {wait}s ({exc.code} {exc.reason})", flush=True)
            time.sleep(wait)
        except Exception as exc:
            wait = min(10 * 2 ** attempt, 300)
            print(f"  waiting {wait}s ({exc})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {url}. Run the script again later: pages already fetched are cached.")


def _from_api(n: int, seed: int, cache_path: Path) -> list[dict]:
    rng = random.Random(seed)
    pages = list(range(TOTAL_DEV // PAGE))
    rng.shuffle(pages)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    pool: list[dict] = []
    want_pages = max(4, (n * 3) // PAGE + 2)   # about 3 candidates per question, balanced later
    for k, page in enumerate(pages[:want_pages], start=1):
        key = str(page)
        if key not in cache:
            print(f"  page {k}/{want_pages} (rows {page * PAGE}-{page * PAGE + PAGE - 1})", flush=True)
            cache[key] = [r["row"] for r in _get({"offset": page * PAGE, "length": PAGE})["rows"]]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            time.sleep(2)
        pool.extend(cache[key])
    rng.shuffle(pool)
    want = {"bridge": n // 2, "comparison": n - n // 2}
    picked = []
    for row in pool:
        if want.get(row["type"], 0) > 0:
            want[row["type"]] -= 1
            picked.append(_normalize_hf(row))
    if len(picked) < n:
        raise RuntimeError(f"Only {len(picked)} balanced questions in the fetched pages; use a smaller --n.")
    return picked


def _normalize_hf(row: dict) -> dict:
    ctx = row["context"]
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_titles": sorted(set(row["supporting_facts"]["title"])),
        "paragraphs": [(t, " ".join(s)) for t, s in zip(ctx["title"], ctx["sentences"])],
    }


def _from_file(path: str, n: int, seed: int) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(data)
    want = {"bridge": n // 2, "comparison": n - n // 2}
    picked = []
    for ex in data:
        if want.get(ex["type"], 0) > 0:
            want[ex["type"]] -= 1
            picked.append({
                "id": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex["type"],
                "level": ex["level"], "supporting_titles": sorted({t for t, _ in ex["supporting_facts"]}),
                "paragraphs": [(t, " ".join(s)) for t, s in ex["context"]],
            })
        if not any(want.values()):
            break
    return picked


def write(examples: list[dict], out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / f"hotpotqa_dev{n}.jsonl"
    corpus_path = out_dir / "hotpotqa_corpus.jsonl"
    with tasks_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples, start=1):
            f.write(json.dumps({
                "id": f"h{i:03d}",
                "domain": f"hotpotqa-{ex['type']}",
                "question": ex["question"],
                "answer": ex["answer"],
                "reference_answer": ex["answer"],
                "key_points": [ex["answer"]],
                "kind": ex["type"],
                "difficulty": ex["level"],
                "expected_sources": ex["supporting_titles"],
                "source_id": ex["id"],
            }, ensure_ascii=False) + "\n")
    seen: dict[str, str] = {}
    for ex in examples:
        for title, text in ex["paragraphs"]:
            seen.setdefault(title, text)
    with corpus_path.open("w", encoding="utf-8") as f:
        for title, text in seen.items():
            f.write(json.dumps({"id": title, "title": title, "text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {tasks_path} ({len(examples)} questions) and {corpus_path} ({len(seen)} paragraphs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data")
    ap.add_argument("--from-file", default=None, help="Local hotpot_dev_distractor_v1.json instead of the API")
    a = ap.parse_args()
    out = Path(a.out)
    examples = _from_file(a.from_file, a.n, a.seed) if a.from_file else _from_api(a.n, a.seed, out / ".hotpotqa_cache.json")
    write(examples, out, a.n)


if __name__ == "__main__":
    main()
