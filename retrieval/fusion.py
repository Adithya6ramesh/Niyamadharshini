"""
Reciprocal Rank Fusion.

    RRF(d) = sum_i 1 / (k + rank_i(d))

Ranks are 1-based positions in each retriever's list. Only ranks are used, so
the incomparable raw scores of BM25 and vector distance never get mixed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass
class FusedItem:
    chunk_id: str
    score: float
    ranks: Dict[str, int] = field(default_factory=dict)  # retriever name -> 1-based rank


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[str]],
    k: int = 60,
    weights: Optional[Mapping[str, float]] = None,
) -> List[FusedItem]:
    """
    Fuse ranked lists of chunk IDs.

    Duplicate IDs within one list keep their best (first) rank. Lists may be
    empty; an ID missing from a list simply gets no contribution from it.
    Ties are broken by best individual rank, then chunk ID, for determinism.
    """
    fused: Dict[str, FusedItem] = {}
    for name, ids in ranked_lists.items():
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        seen = set()
        rank = 0
        for chunk_id in ids or ():
            if chunk_id is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            item = fused.setdefault(chunk_id, FusedItem(chunk_id, 0.0))
            item.score += weight / (k + rank)
            item.ranks[name] = rank
    return sorted(fused.values(), key=lambda it: (-it.score, min(it.ranks.values()), it.chunk_id))
