from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_document_id" not in st.session_state:
        st.session_state.active_document_id = ""
    if "active_file_name" not in st.session_state:
        st.session_state.active_file_name = ""
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE_URL


def _api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error")
            if detail:
                return str(detail)
    except ValueError:
        pass
    return response.text.strip() or f"HTTP {response.status_code}"


def upload_pdf(api_base_url: str, uploaded_file) -> Dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }
    response = requests.post(
        f"{api_base_url}/documents/upload",
        files=files,
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(_api_error_message(response))
    return response.json()


def ask_question(
    api_base_url: str,
    question: str,
    top_k: int,
    document_id: str | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "question": question,
        "top_k": top_k,
    }
    if document_id:
        payload["document_id"] = document_id

    response = requests.post(
        f"{api_base_url}/chat",
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(_api_error_message(response))
    return response.json()


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        st.caption("No sources returned.")
        return

    st.markdown("**Sources**")
    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {})
        st.markdown(
            f"{index}. Page {metadata.get('page_number', '?')} | "
            f"{metadata.get('source_file', 'unknown')} | "
            f"score={source.get('score', 0.0):.4f}"
        )


def main() -> None:
    st.set_page_config(
        page_title="AI Research Paper Assistant",
        layout="wide",
    )
    _init_state()

    st.title("AI Research Paper Assistant")
    st.caption("Semantic Search + RAG powered by Endee Vector DB")

    with st.sidebar:
        st.subheader("Configuration")
        st.session_state.api_base_url = st.text_input(
            "FastAPI URL",
            value=st.session_state.api_base_url,
        ).rstrip("/")
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=5)
        restrict_to_current_doc = st.checkbox(
            "Restrict chat to latest uploaded PDF",
            value=True,
        )

    st.subheader("1) Upload and Index PDF")
    uploaded_file = st.file_uploader("Choose a research paper PDF", type=["pdf"])

    if st.button("Index PDF", disabled=uploaded_file is None):
        with st.spinner("Extracting, chunking, embedding, and storing vectors..."):
            try:
                result = upload_pdf(st.session_state.api_base_url, uploaded_file)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Indexing failed: {exc}")
            else:
                st.session_state.active_document_id = result["document_id"]
                st.session_state.active_file_name = result["file_name"]
                st.success(
                    "Indexed successfully "
                    f"({result['chunks_indexed']} chunks from {result['pages']} pages)."
                )

    if st.session_state.active_document_id:
        st.info(
            f"Current document: {st.session_state.active_file_name} "
            f"(id: {st.session_state.active_document_id})"
        )

    st.subheader("2) Ask Questions")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your indexed papers...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        document_id = (
            st.session_state.active_document_id
            if restrict_to_current_doc and st.session_state.active_document_id
            else None
        )

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                try:
                    response = ask_question(
                        api_base_url=st.session_state.api_base_url,
                        question=question,
                        top_k=top_k,
                        document_id=document_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    answer = f"Request failed: {exc}"
                    sources: List[Dict[str, Any]] = []
                else:
                    answer = response.get("answer", "")
                    sources = response.get("sources", [])

            st.markdown(answer)
            render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
