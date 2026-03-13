from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.pdf_loader import PDFPage


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str


def split_pages_into_chunks(
    pages: List[PDFPage],
    document_id: str,
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[TextChunk] = []
    chunk_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page.text)
        for raw_chunk in page_chunks:
            text = raw_chunk.strip()
            if not text:
                continue

            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}-{chunk_index}",
                    document_id=document_id,
                    source_file=source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            chunk_index += 1

    return chunks
