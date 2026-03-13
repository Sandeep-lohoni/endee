from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database.endee_client import EndeeClient, EndeeClientError
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkMetadata,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_endee_client() -> EndeeClient:
    return EndeeClient(
        base_url=settings.endee_base_url,
        auth_token=settings.endee_auth_token,
        timeout_seconds=settings.http_timeout_seconds,
    )


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=settings.embedding_model)


@lru_cache
def get_vector_store_service() -> VectorStoreService:
    return VectorStoreService(
        client=get_endee_client(),
        index_name=settings.endee_index_name,
        space_type=settings.endee_space_type,
        precision=settings.endee_precision,
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        default_top_k=settings.retrieval_top_k_default,
    )


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(
        google_api_key=settings.google_api_key,
        google_model=settings.google_model,
    )


def _to_retrieved_chunk(payload: dict) -> RetrievedChunk:
    metadata = payload.get("metadata", {})
    return RetrievedChunk(
        id=str(payload.get("id", "")),
        score=float(payload.get("score", 0.0)),
        text=str(payload.get("text", "")),
        metadata=ChunkMetadata(
            document_id=str(metadata.get("document_id", "")),
            source_file=str(metadata.get("source_file", "")),
            page_number=int(metadata.get("page_number", 0)),
            chunk_index=int(metadata.get("chunk_index", 0)),
        ),
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "endee_reachable": get_endee_client().health_check(),
    }


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = get_ingestion_service().ingest_pdf(file_bytes=file_bytes, file_name=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EndeeClientError as exc:
        raise HTTPException(status_code=502, detail=f"Endee error: {exc}") from exc

    return UploadResponse(**result)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    top_k = min(max(1, request.top_k), settings.retrieval_top_k_max)
    try:
        matches = get_retrieval_service().retrieve(
            query=request.question,
            top_k=top_k,
            document_id=request.document_id,
        )
    except EndeeClientError as exc:
        raise HTTPException(status_code=502, detail=f"Endee error: {exc}") from exc

    return SearchResponse(matches=[_to_retrieved_chunk(item) for item in matches])


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    top_k = min(max(1, request.top_k), settings.retrieval_top_k_max)
    try:
        matches = get_retrieval_service().retrieve(
            query=request.question,
            top_k=top_k,
            document_id=request.document_id,
        )
    except EndeeClientError as exc:
        raise HTTPException(status_code=502, detail=f"Endee error: {exc}") from exc

    answer = get_rag_service().generate_answer(question=request.question, chunks=matches)
    sources = [_to_retrieved_chunk(item) for item in matches]
    return ChatResponse(answer=answer, sources=sources)
