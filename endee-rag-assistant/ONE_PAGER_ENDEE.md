# One-Pager: AI Research Paper Assistant (RAG + Endee)

## Candidate Project Summary
This project is an end-to-end Retrieval-Augmented Generation (RAG) assistant for research papers. Users upload a PDF, the system indexes semantically meaningful chunks into Endee, and then answers natural-language questions with source-grounded context.

The goal is to reduce time spent manually searching long technical documents while keeping answers traceable to the original paper.

## Problem
Traditional keyword search is weak for research workflows because:
- key ideas may be described with different terminology
- relevant context is spread across sections/pages
- users need quick, source-backed answers instead of raw matches

## Solution
I built a production-style pipeline with:
- FastAPI backend for ingestion, retrieval, and chat APIs
- Streamlit frontend for upload + interactive Q&A
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- Endee as the vector retrieval layer
- Gemini (`gemini-2.5-flash-lite`) for final answer synthesis

## System Flow
1. Upload PDF -> extract text page-wise (`pypdf`)
2. Chunk text (`1000` size, `200` overlap) using recursive splitting
3. Generate normalized embeddings for each chunk
4. Ensure/create Endee index with correct embedding dimension
5. Store vectors with rich metadata (`document_id`, `source_file`, `page_number`, `chunk_index`)
6. On question, embed query and retrieve top-k semantically similar chunks
7. Build grounded prompt and generate answer with source references

## Key Engineering Decisions
- Modular service architecture:
  - `IngestionService`, `RetrievalService`, `VectorStoreService`, `RAGService`, `EmbeddingService`
  - clean separation makes the system testable and replaceable
- Robust retrieval quality:
  - overfetch-first retrieval (`k * 4`) improves filtered recall
  - second-pass expansion when document filtering is too strict
- Endee integration reliability:
  - index dimension compatibility checks prevent silent mismatch errors
  - msgpack search response parsing handles multiple payload formats
- Graceful degradation:
  - if `GOOGLE_API_KEY` is missing, app still returns best retrieved context

## API Surface
- `POST /documents/upload` -> ingest and index a PDF
- `POST /search` -> semantic retrieval only
- `POST /chat` -> retrieval + answer generation
- `GET /health` -> app + Endee connectivity status

## Why This Is Relevant to Endee
This implementation demonstrates practical Endee usage in a real RAG workflow:
- index lifecycle management from application code
- dense vector search over document chunks
- metadata-aware filtering for document-scoped retrieval
- low-friction integration through Endee HTTP APIs

It shows how Endee can serve as the retrieval core for knowledge assistants that need both speed and source-grounded responses.

## Current Scope and Next Steps
Current scope:
- single-file PDF ingestion per upload
- document-level filtering and source attribution
- configurable retrieval controls (`top_k`, chunking, index settings)

Potential next improvements:
- OCR fallback for scanned PDFs
- reranking layer for higher answer precision
- evaluation harness (recall/groundedness metrics)
- multi-tenant auth and index isolation

## Tech Stack
Python 3.12, FastAPI, Streamlit, SentenceTransformers, LangChain text splitters, Google Gemini via `langchain-google-genai`, Endee vector DB, `requests`, `msgpack`, `pypdf`.
