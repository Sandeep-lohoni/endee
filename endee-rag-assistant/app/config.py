from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if not value:
        return default
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or default


@dataclass(frozen=True)
class Settings:
    app_name: str
    endee_base_url: str
    endee_auth_token: str
    endee_index_name: str
    endee_space_type: str
    endee_precision: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k_default: int
    retrieval_top_k_max: int
    google_api_key: str
    google_model: str
    http_timeout_seconds: int
    cors_allow_origins: List[str]


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "AI Research Paper Assistant"),
        endee_base_url=os.getenv("ENDEE_BASE_URL", "http://localhost:8080").rstrip("/"),
        endee_auth_token=os.getenv("ENDEE_AUTH_TOKEN", ""),
        endee_index_name=os.getenv("ENDEE_INDEX_NAME", "research_papers"),
        endee_space_type=os.getenv("ENDEE_SPACE_TYPE", "cosine"),
        endee_precision=os.getenv("ENDEE_PRECISION", "float32"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        retrieval_top_k_default=_env_int("RETRIEVAL_TOP_K_DEFAULT", 5),
        retrieval_top_k_max=_env_int("RETRIEVAL_TOP_K_MAX", 20),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        google_model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
        http_timeout_seconds=_env_int("HTTP_TIMEOUT_SECONDS", 30),
        cors_allow_origins=_env_list("CORS_ALLOW_ORIGINS", ["*"]),
    )
