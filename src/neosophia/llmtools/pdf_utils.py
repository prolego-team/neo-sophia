"""
Tools for wrangling with PDF documents.
"""
import io
import os
import fnmatch

from typing import List

import fitz
import PyPDF2

from PIL import Image


def find_pdfs_in_directory(directory: str) -> List[str]:
    pdf_paths = []
    for root, _, files in os.walk(directory):
        for filename in fnmatch.filter(files, "*.pdf"):
            pdf_path = os.path.join(root, filename)
            pdf_paths.append(pdf_path)
    return pdf_paths


def extract_text_from_pdf(filepath: str) -> List[str]:
    """ """
    with open(filepath, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        data = []
        for page_obj in pdf_reader.pages:
            data.append(page_obj.extract_text())
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
