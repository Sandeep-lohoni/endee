from __future__ import annotations

import json
from typing import Any, Dict, List

from app.database.endee_client import EndeeClient
from app.utils.text_chunker import TextChunk


def _safe_json_loads(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


class VectorStoreService:
    def __init__(
        self,
        client: EndeeClient,
        index_name: str,
        space_type: str = "cosine",
        precision: str = "float32",
    ):
        self.client = client
        self.index_name = index_name
        self.space_type = space_type
        self.precision = precision

    def store_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> int:
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count must match.")

        dim = len(embeddings[0])
        self.client.ensure_index(
            index_name=self.index_name,
            dim=dim,
            space_type=self.space_type,
            precision=self.precision,
        )

        vectors: List[Dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            meta = json.dumps(
                {
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                },
                ensure_ascii=True,
            )
            filter_value = json.dumps(
                {
                    "document_id": chunk.document_id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                },
                ensure_ascii=True,
            )

            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "meta": meta,
                    "filter": filter_value,
                }
            )

        self.client.insert_vectors(self.index_name, vectors)
        return len(vectors)

    def search_chunks(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        raw_results = self.client.search(
            index_name=self.index_name,
            query_vector=query_embedding,
            k=top_k,
        )

        parsed_results: List[Dict[str, Any]] = []
        for result in raw_results:
            meta = _safe_json_loads(result.get("meta", ""))
            filter_data = _safe_json_loads(result.get("filter", ""))
            text = str(meta.get("text", "")).strip()

            if not text:
                continue

            metadata = {
                "document_id": str(
                    meta.get("document_id")
                    or filter_data.get("document_id")
                    or ""
                ),
                "source_file": str(
                    meta.get("source_file")
                    or filter_data.get("source_file")
                    or ""
                ),
                "page_number": int(meta.get("page_number") or 0),
                "chunk_index": int(meta.get("chunk_index") or 0),
            }

            parsed_results.append(
                {
                    "id": str(result.get("id", "")),
                    "score": float(result.get("similarity", 0.0)),
                    "text": text,
                    "metadata": metadata,
                }
            )

        return parsed_results
