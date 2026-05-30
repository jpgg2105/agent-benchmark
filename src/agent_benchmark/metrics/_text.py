from __future__ import annotations

import re

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for", "on",
    "with", "that", "this", "it", "as", "by", "be", "at", "from", "was", "were",
    "which", "how", "what", "why", "does", "do", "but", "than", "into", "such",
}


def tokens(text: str) -> list[str]:
    """Lowercase content tokens, stopwords and single characters removed."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]
