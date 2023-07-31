""" Text utilities """
from neosophia.llmtools import openaiapi as oaiapi


def split_text_into_sections(text, n):
    page_length = len(text)
    section_length = page_length // n
    sections = [
        text[i:i+section_length] for i in range(
            0, page_length, section_length)
    ]
    return sections

