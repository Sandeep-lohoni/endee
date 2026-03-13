from __future__ import annotations

import uuid
from typing import Dict

from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.utils.pdf_loader import load_pdf_pages
from app.utils.text_chunker import split_pages_into_chunks


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_pdf(self, file_bytes: bytes, file_name: str) -> Dict[str, object]:
        document_id = str(uuid.uuid4())

        pages = load_pdf_pages(file_bytes)
        if not pages:
            raise ValueError(
                "No readable text was extracted from this PDF. "
                "Try a text-based PDF instead of a scanned image."
            )

        chunks = split_pages_into_chunks(
            pages=pages,
            document_id=document_id,
            source_file=file_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Chunking produced no usable text.")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(texts)
        indexed_count = self.vector_store_service.store_chunks(chunks, embeddings)

        return {
            "document_id": document_id,
            "file_name": file_name,
            "pages": len(pages),
            "chunks_indexed": indexed_count,
        }
