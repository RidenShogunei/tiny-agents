"""Text similarity utilities for paper matching."""

from __future__ import annotations


def normalize(s: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> set:
    """Return word tokens as a set."""
    return set(normalize(s).split())


def title_similarity(title_a: str, title_b: str) -> float:
    """Compute similarity between two paper titles (0.0–1.0).

    Uses Jaccard similarity on normalized word tokens.
    """
    if not title_a or not title_b:
        return 0.0

    tokens_a = tokenize(title_a)
    tokens_b = tokenize(title_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union if union > 0 else 0.0


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_ratio(a: list, b: list) -> float:
    """Overlap ratio between two lists (size of intersection / size of smaller)."""
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / min(len(set_a), len(set_b))
