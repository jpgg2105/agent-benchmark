from pathlib import Path

from agent_benchmark.dataset import load_dataset

DATA = Path(__file__).resolve().parents[1] / "data" / "techdocs_qa.jsonl"


def test_dataset_loads_and_is_well_formed():
    tasks, version = load_dataset(DATA)
    assert len(tasks) >= 10
    assert len(version) == 12                                  # short content hash
    assert all(t.key_points for t in tasks)                    # every task has a rubric
    assert all(t.difficulty in {"easy", "medium", "hard"} for t in tasks)
    assert len({t.id for t in tasks}) == len(tasks)            # ids are unique


def test_dataset_version_is_deterministic():
    _, v1 = load_dataset(DATA)
    _, v2 = load_dataset(DATA)
    assert v1 == v2
