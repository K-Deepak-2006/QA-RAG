"""Lightweight helpers (no heavy imports) so they can be unit-tested quickly."""
import re

RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion over several ranked lists of chunk ids."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
