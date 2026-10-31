from __future__ import annotations

from typing import Any


def agent_settings(**params: Any):
    """Build research-agent Settings from benchmark params.

    ``model`` and ``base_url`` are provider-neutral shorthands; API keys still
    come from the environment / .env, never from the config file.
    """
    from research_agent.config import Settings

    p = {k: v for k, v in params.items() if v is not None}
    model = p.pop("model", None)
    base_url = p.pop("base_url", None)
    provider = p.get("llm_provider", "fake")
    if model:
        p["anthropic_model" if provider == "anthropic" else "openai_model"] = model
    if base_url:
        p["openai_base_url"] = base_url
    return Settings(**p)


def usage_of(llm) -> dict[str, int]:
    u = getattr(llm, "usage", None)
    return u.snapshot() if u is not None else {}
