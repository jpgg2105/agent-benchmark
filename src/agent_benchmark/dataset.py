from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import Task


def load_dataset(path: str | Path) -> tuple[list[Task], str]:
    """Load a JSONL dataset and return (tasks, version).

    The version is a short content hash so every result is tied to the exact
    dataset it was produced against -- the backbone of reproducibility.
    """
    raw = Path(path).read_bytes()
    tasks = [
        Task.model_validate(json.loads(line))
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    version = hashlib.sha256(raw).hexdigest()[:12]
    return tasks, version
