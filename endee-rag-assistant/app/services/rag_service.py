from __future__ import annotations

from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = (
    "You are an AI research assistant. Answer only from the provided context. "
    "If the context is insufficient, explicitly say you do not know. "
    "When possible, cite sources using [S1], [S2], etc."
)


class RAGService:
    def __init__(self, google_api_key: str, google_model: str):
        self.enabled = bool(google_api_key)
        self.llm = None
        if self.enabled:
            self.llm = ChatGoogleGenerativeAI(
                model=google_model,
                google_api_key=google_api_key,
                temperature=0.1,
            )

    @staticmethod
    def _context_block(chunks: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            page = metadata.get("page_number", "?")
            source_file = metadata.get("source_file", "unknown")
            blocks.append(
                f"[S{index}] file={source_file}, page={page}\n{chunk.get('text', '').strip()}"
            )

        return "\n\n".join(blocks)

    def generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "I could not find relevant context in the indexed papers."

        context = self._context_block(chunks)

        if not self.enabled or self.llm is None:
            preview = "\n\n".join(
                [f"[S{i}] {chunk.get('text', '')[:280]}..." for i, chunk in enumerate(chunks, start=1)]
            )
            return (
                "LLM generation is unavailable because GOOGLE_API_KEY is not configured.\n\n"
                "Most relevant retrieved context:\n"
                f"{preview}"
            )

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            return (
                "The retrieval step succeeded, but answer generation failed. "
                f"Error: {exc}"
            )

        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(str(item) for item in content)

        answer = str(content).strip()
        return answer or "The model returned an empty answer."
