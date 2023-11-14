"""Utility functions for search documents."""

from dataclasses import dataclass
from typing import Iterable, Any
from hashlib import md5
import json
from collections import defaultdict


@dataclass
class Page:
    """Store a page of text."""
    page_num: int
    text_blocks: list[str]


Document = Iterable[Page]


@dataclass
class SearchResult:
    """Structure for consistent search results."""
    similarity_score: float
    file: str
    tree_index: tuple
    paragraph_index: int
    chunk_id: int
    text: str
    reranked_score: float = -1000


def clean_text(text: str) -> str:
    """Strip white space and newlines from a string."""
    return text.replace('\n','').strip()


def reciprocal_rank_fusion(
      ranked_lists: list[list[Any]], k: float = 60.0
    ) -> list[Any]:
    """Apply reciprocal rank fusion to a list of lists."""
    scores = defaultdict(float)
    for lyst in ranked_lists:
        for rank,item in enumerate(lyst):
            scores[item] += 1/(k + rank)

    scored_items = sorted(
        [(score,item) for item,score in scores.items()],
        key=lambda x: x[0],
        reverse=True
    )

    return [item for _,item in scored_items]


def get_dict_hash(dictionary):
    """
    Calculates a repeatable hash for a dictionary.

    Args:
      dictionary: The dictionary to hash.

    Returns:
      A hash value for the dictionary.
    """
    d_encode = json.dumps(dictionary).encode()
    hasher = md5()
    hasher.update(d_encode)

    return hasher.hexdigest()
