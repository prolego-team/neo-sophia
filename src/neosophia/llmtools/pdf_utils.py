"""
Tools for wrangling with PDF documents.
"""
import io

from typing import List

import fitz
import PyPDF2

from PIL import Image


def extract_text_from_pdf(file_path: str) -> dict:
    """ """
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    num_pages = len(pdf_reader.pages)
    data = {}
    for page_num in range(0, num_pages):
        if page_num not in data:
            data[page_num] = []
        page_obj = pdf_reader.pages[page_num]
        data[page_num] = page_obj.extract_text()
        data[page_num]
    pdf_file_obj.close()
    return data


def pdf_to_image(path: str, zoom_x: int, zoom_y: int) -> List[Image.Image]:
    """Render PDF pages to images, then load as Image objects."""
    doc = fitz.open(path)
    mat = fitz.Matrix(zoom_x, zoom_y)
    images = []
    for page in doc:
        image = page.get_pixmap(matrix=mat).tobytes()
        images.append(Image.open(io.BytesIO(image)))
    return images
