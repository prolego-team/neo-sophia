""" Tools for wrangling with PDF documents """
# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Cameron Fabbri
import io
import re

from typing import Any, List
from dataclasses import dataclass

import fitz

from PIL import Image


class Colors:
    # Define ANSI escape sequences for different colors
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    BLACK = "\033[30m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    WHITE = "\033[37m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


def colorize(text, color):
    return f"{color}{text}{Colors.RESET}"


@dataclass
class Rule:
    uid: str
    description: str
    sections: List[Any]
    interpretations: List[Any]
    amendments: List[Any]

    def __str__(self):
        a = colorize('uid: ', Colors.GREEN) + self.uid + '\n'
        b = colorize('description: ', Colors.GREEN) + self.description + '\n'
        c = colorize('sections: ', Colors.GREEN)
        for s in self.sections:
            c += str(s) + '\n'
        d = colorize('interpretations: ', Colors.GREEN)
        for s in self.interpretations:
            d += s + '\n'
        e = colorize('amendments: ', Colors.GREEN)
        for s in self.amendments:
            e += s + '\n'

        return a + b + c + d + e


def extract_text_by_paragraphs(file_in, delimiter, start_page, end_page):
    """ """

    doc = fitz.open(file_in)
    sections = []
    num_pages = len(doc)
    for idx, page in enumerate(doc):
        if idx + 1 < start_page:
            continue
        if idx + 1 == end_page:
            break
        text = page.get_text()

        text_nl = text.split('\n')

        current_section = []
        for line in text_nl:
            matches = re.findall(delimiter, line)
            if matches:
                sections.append(' '.join(current_section))
                current_section = []
            current_section.append(line)
        sections.append(' '.join(current_section))
    doc.close()

    return sections, num_pages


def pdf_to_image(path: str, zoom_x: int, zoom_y: int):
    """ Renders the page to a PDF then loads and converts to an image """
    doc = fitz.open(path)
    mat = fitz.Matrix(zoom_x, zoom_y)
    images = []
    for page in doc:
        image = page.get_pixmap(matrix=mat).tobytes()
        images.append(Image.open(io.BytesIO(image)))
    return images

