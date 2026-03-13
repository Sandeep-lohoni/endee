from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List

from pypdf import PdfReader


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    text: str


def load_pdf_pages(file_bytes: bytes) -> List[PDFPage]:
    reader = PdfReader(BytesIO(file_bytes))
    pages: List[PDFPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        normalized = extracted.strip()
        if normalized:
            pages.append(PDFPage(page_number=page_number, text=normalized))

    return pages
