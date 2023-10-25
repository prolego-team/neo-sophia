from dataclasses import dataclass
from typing import Iterable, Any
from hashlib import md5
import json
from collections import defaultdict


@dataclass
class Page:
    page_num: int
    text_blocks: list[str]


Document = Iterable[Page]


@dataclass
class SearchResult:
   similarity_score: float
   file: str
   tree_index: tuple
   paragraph_index: int
   chunk_id: int
   text: str
   reranked_score: float = -1000


def clean_text(text: str) -> str:
    return text.replace('\n','').strip()


def reciprocal_rank_fusion(
      ranked_lists: list[list[Any]], k: float = 60
    ) -> list[Any]:
    """"""
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
  h = md5()
  h.update(d_encode)

  return h.hexdigest()