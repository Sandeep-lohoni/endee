from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        default_top_k: int = 5,
    ):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.default_top_k
        k = max(1, k)

        query_embedding = self.embedding_service.embed_query(query)

        overfetch = max(k * 4, k)
        matches = self.vector_store_service.search_chunks(query_embedding, top_k=overfetch)

        if document_id:
            filtered = [
                match
                for match in matches
                if match.get("metadata", {}).get("document_id") == document_id
            ]

            # If we filtered too aggressively, retry with a larger search window once.
            if len(filtered) < k and overfetch < 50:
                expanded_matches = self.vector_store_service.search_chunks(
                    query_embedding, top_k=50
                )
                filtered = [
                    match
                    for match in expanded_matches
                    if match.get("metadata", {}).get("document_id") == document_id
                ]

            matches = filtered

        return matches[:k]
