# Dataset: techdocs_qa

A small, hand-authored benchmark of technical question-answering items, used to score QA and research-agent systems by rubric coverage, grounding, and (optionally) LLM-judged correctness.

## Format

JSON Lines (`.jsonl`), one task per line. Schema:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique task id (e.g. `q01`) |
| `domain` | string | Topic area (web, databases, systems, distributed-systems, security, ai-engineering) |
| `question` | string | The question posed to the system |
| `key_points` | string[] | Facts a correct answer must cover — the scoring rubric |
| `reference_answer` | string | A concise gold answer (for human inspection; not used by the lexical metrics) |
| `difficulty` | string | `easy` \| `medium` \| `hard` |
| `expected_sources` | string[] | Optional source locators a correct answer should cite (empty for general-knowledge items) |

## Provenance

All items were hand-authored for this benchmark. They deliberately cover stable, widely-agreed software- and AI-engineering concepts so the rubric key-points are uncontroversial and the benchmark stays valid over time. No items were scraped or machine-generated.

Current size: 12 tasks (3 easy, 6 medium, 3 hard). This is intentionally small — large enough to exercise the harness and surface differences between systems, small enough to read in full and trust.

## How scoring uses it

- **coverage** tokenises each key-point and the system's answer and checks how many key-points are lexically covered. It is a cheap proxy, not semantic understanding.
- **grounding** checks for the presence of a citation; if `expected_sources` is set, it requires overlap.
- **correctness** (optional) sends the question, key-points, and answer to an LLM judge for a 0–1 rating.

## Limitations

- **Lexical coverage is a proxy.** An answer that conveys a key-point in different words may be under-credited. For higher fidelity, enable the LLM judge or swap in an embedding-based coverage metric (the interface is unchanged).
- **Small N.** Twelve items show direction, not statistical significance. Expand the file before drawing strong conclusions.
- **General knowledge by default.** These questions test answer quality, not retrieval over a private corpus. To benchmark retrieval grounding, repoint the systems at your own documents and populate `expected_sources`.
- **English only**, single-turn questions only.

## Extending it

Append JSONL lines following the schema. Keep `id` unique. The dataset is content-hashed on load, so any edit produces a new dataset version — results from before and after an edit will be correctly distinguished rather than silently compared.
