""" Text utilities """
from typing import List


def split_text_into_sections(text: str, n: int) -> List[str]:
    """ Splits a string into `n` sections """
    page_length = len(text)
    section_length = page_length // n
    sections = [
        text[i:i+section_length] for i in range(
            0, page_length, section_length)
    ]
    return sections

