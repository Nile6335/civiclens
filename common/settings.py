"""Central configuration for CivicLens.

Every knob is overridable via environment variables or .env. Code defaults are the
full-scale profile (llama3.1:8b, bge-m3, bge-reranker-base); .env.example ships a lean
profile for modest hardware. See docs/DECISIONS.md for the rationale.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://civiclens:civiclens@localhost:5442/civiclens"
    # Read-only DSN used by the tabular agent (SELECT-only role).
    database_url_ro: str = "postgresql://civiclens_ro:civiclens_ro@localhost:5442/civiclens"

    # LLM backend
    llm_backend: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"

    # Topic tagging
    topic_tagger: Literal["keyword", "zeroshot"] = "keyword"
    zero_shot_model: str = "facebook/bart-large-mnli"

    # Voice mode (Phase 7)
    piper_voice: str = "en_US-lessac-medium"
    piper_data_dir: str = ".models/piper"
    whisper_model: str = "small"

    # Safety (Phase 8)
    pii_redaction: bool = True  # redact transcript chunks at ingest; originals quarantined
    # person-name NER is measured on the seeded eval but NOT applied to the live record
    # by default: named public officials are the record (see ingestion._maybe_redact)
    pii_redact_persons: bool = False
    ner_model: str = "dslim/bert-base-NER"
    harden_prompts: bool = True  # retrieved-content demarcation + instruction hierarchy

    # Langfuse (optional; tracing silently no-ops when unreachable)
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
