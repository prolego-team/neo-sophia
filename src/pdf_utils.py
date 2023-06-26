""" Tools for wrangling with PDF documents """
# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Cameron Fabbri
import io

import fitz

from PIL import Image


def extract_text_by_paragraphs(file_in, delimiter, start_page, end_page):
    """ """
    doc = fitz.open(file_in)
    paragraphs = []
    for idx, page in enumerate(doc):
        if idx + 1 < start_page:
            continue
        if idx + 1 == end_page:
            break
        text = page.get_text()
        paragraphs.extend(text.split(delimiter))
    doc.close()

    return paragraphs


def pdf_to_image(path: str, zoom_x: int, zoom_y: int):
    """ Renders the page to a PDF then loads and converts to an image """
    doc = fitz.open(path)
    mat = fitz.Matrix(zoom_x, zoom_y)
    images = []
    for page in doc:
        image = page.get_pixmap(matrix=mat).tobytes()
        images.append(Image.open(io.BytesIO(image)))
    return images

