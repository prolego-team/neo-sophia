from dataclasses import dataclass
from typing import Iterable
from hashlib import md5
import json


@dataclass
class Page:
    page_num: int
    text_blocks: list[str]

Document = Iterable[Page]


def clean_text(text: str) -> str:
    return text.replace('\n','').strip()


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