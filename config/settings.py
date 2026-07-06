"""
Central configuration. Loads from environment variables with sane defaults.
Keeping every tunable here means no magic numbers/strings buried in modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class LLMConfig:
    """Which provider + model to use for each pipeline stage.

    Different stages have different cost/quality tradeoffs -- e.g. vision
    extraction needs a strong multimodal model, but entity canonicalization
    disambiguation can often run on a cheaper/faster model.
    """
    provider: str = _env("RAG_LLM_PROVIDER", "anthropic")  # "anthropic" | "azure_openai"

    # Anthropic settings
    anthropic_api_key: str = _env("ANTHROPIC_API_KEY")
    anthropic_vision_model: str = _env("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
    anthropic_text_model: str = _env("ANTHROPIC_TEXT_MODEL", "claude-sonnet-4-6")

    # Azure OpenAI settings
    azure_api_key: str = _env("AZURE_OPENAI_API_KEY")
    azure_endpoint: str = _env("AZURE_OPENAI_ENDPOINT")
    azure_api_version: str = _env("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    azure_vision_deployment: str = _env("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4o")
    azure_text_deployment: str = _env("AZURE_OPENAI_TEXT_DEPLOYMENT", "gpt-4o")

    max_retries: int = 3
    request_timeout_s: int = 90


@dataclass
class ChunkingConfig:
    max_tokens_per_chunk: int = 800
    chunk_overlap_tokens: int = 100
    min_chunk_tokens: int = 50


@dataclass
class EntityResolutionConfig:
    # Cosine similarity above which two entity mentions are candidates
    # for the same canonical entity (before LLM disambiguation).
    similarity_threshold: float = 0.82
    # Above this, auto-merge without LLM call (saves cost on obvious matches).
    auto_merge_threshold: float = 0.95
    embedding_model: str = _env("RAG_EMBEDDING_MODEL", "text-embedding-3-large")


@dataclass
class RetrievalConfig:
    vector_top_k: int = 12
    bm25_top_k: int = 12
    graph_hops: int = 2
    graph_max_neighbors_per_hop: int = 8
    final_context_chunks: int = 10
    # Weights for hybrid re-ranking (must sum roughly to 1.0, not enforced)
    weight_vector: float = 0.45
    weight_bm25: float = 0.25
    weight_graph: float = 0.30


@dataclass
class StorageConfig:
    postgres_dsn: str = _env("RAG_POSTGRES_DSN", "sqlite:///./rag_kb.db")
    faiss_index_dir: str = _env("RAG_FAISS_DIR", "./data/faiss_index")
    raw_units_dir: str = _env("RAG_RAW_UNITS_DIR", "./data/raw_units")


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    entity_resolution: EntityResolutionConfig = field(default_factory=EntityResolutionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    min_confidence_for_citation: float = 0.55


def load_config() -> AppConfig:
    return AppConfig()
