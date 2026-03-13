from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_id: str = ""
    source_file: str = ""
    page_number: int = 0
    chunk_index: int = 0


class RetrievedChunk(BaseModel):
    id: str
    score: float
    text: str
    metadata: ChunkMetadata


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    pages: int
    chunks_indexed: int


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: Optional[str] = None


class SearchResponse(BaseModel):
    matches: List[RetrievedChunk]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[RetrievedChunk]
