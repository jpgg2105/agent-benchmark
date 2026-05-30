from __future__ import annotations

from pathlib import Path

from .models import RunConfig


def load_run_config(path: str | Path) -> RunConfig:
    import yaml  # lazy import

    data = yaml.safe_load(Path(path).read_text())
    return RunConfig.model_validate(data)
