"""
Tools for wrangling with PDF documents.
"""

from typing import List
import io

import fitz
from PIL import Image


def pdf_to_image(path: str, zoom_x: int, zoom_y: int) -> List[Image.Image]:
    """Render PDF pages to images, then load as Image objects."""
    doc = fitz.open(path)
    mat = fitz.Matrix(zoom_x, zoom_y)
    images = []
    for page in doc:
        image = page.get_pixmap(matrix=mat).tobytes()
        images.append(Image.open(io.BytesIO(image)))
    return images
