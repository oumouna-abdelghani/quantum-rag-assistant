"""
config.py - Centralized settings for the Quantum RAG Assistant (Stage 1).

Every tunable value lives here and can be overridden from the environment
(.env file or real environment variables). Other modules import the shared
`settings` object and read it at call time, so the CLI can change values such
as the relevance threshold while it is running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Chroma collection names (two collections, searched together at query time).
BASE_COLLECTION = "quantum_kb"          # curated knowledge base, built by ingest.py
UPLOAD_COLLECTION = "user_uploads"      # files added with /upload in the CLI

# File types the loaders can read.
SUPPORTED_EXTENSIONS = (".pdf", ".md", ".mdx", ".txt", ".ipynb", ".csv")


def _read(name: str, default, cast=str):
    """Read an environment variable, falling back to `default` when unset/blank."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    try:
        if cast is bool:
            return raw.lower() in {"1", "true", "yes", "on"}
        return cast(raw)
    except ValueError as error:
        raise ValueError(f"Invalid value for {name}: {raw!r}") from error


@dataclass
class Settings:
    # --- Gemini ---------------------------------------------------------
    gemini_api_key: str
    generation_model: str
    embedding_model: str
    embedding_dim: int
    temperature: float
    max_output_tokens: int
    thinking_level: str           # "minimal" | "low" | "medium" | "high"

    # --- Embedding requests / retry ---------------------------------------
    embed_batch_size: int
    embed_min_interval: float     # minimum seconds between embedding requests
    max_retries: int
    retry_base_delay: float       # seconds, doubled on every retry

    # --- Chunking ----------------------------------------------------------
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int

    # --- Storage -----------------------------------------------------------
    data_dir: Path
    raw_dir: Path
    chroma_dir: Path
    sources_file: Path
    max_upload_mb: int

    # --- Retrieval -----------------------------------------------------------
    top_k: int                    # chunks passed to the LLM
    fetch_k: int                  # candidates fetched before MMR
    mmr_lambda: float             # 1.0 = pure relevance, 0.0 = pure diversity
    relevance_threshold: float    # best cosine score needed to answer from documents
    threshold_margin: float       # chunks this far below the threshold are still kept
    hybrid_enabled: bool          # dense + BM25 fusion
    rrf_k: int                    # Reciprocal Rank Fusion constant

    # --- Chat -----------------------------------------------------------------
    max_history_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(_read("DATA_DIR", str(PROJECT_ROOT / "data")))
        return cls(
            gemini_api_key=_read("GEMINI_API_KEY", ""),
            generation_model=_read("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite"),
            embedding_model=_read("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dim=_read("EMBEDDING_DIM", 768, int),
            temperature=_read("GENERATION_TEMPERATURE", 0.2, float),
            max_output_tokens=_read("MAX_OUTPUT_TOKENS", 2048, int),
            thinking_level=_read("THINKING_LEVEL", "low"),
            embed_batch_size=_read("EMBED_BATCH_SIZE", 32, int),
            embed_min_interval=_read("EMBED_MIN_INTERVAL", 0.6, float),
            max_retries=_read("MAX_RETRIES", 6, int),
            retry_base_delay=_read("RETRY_BASE_DELAY", 2.0, float),
            chunk_size=_read("CHUNK_SIZE", 1000, int),
            chunk_overlap=_read("CHUNK_OVERLAP", 150, int),
            min_chunk_chars=_read("MIN_CHUNK_CHARS", 60, int),
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            chroma_dir=Path(_read("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))),
            sources_file=Path(_read("SOURCES_FILE", str(PROJECT_ROOT / "sources.yaml"))),
            max_upload_mb=_read("MAX_UPLOAD_MB", 50, int),
            top_k=_read("TOP_K", 6, int),
            fetch_k=_read("FETCH_K", 20, int),
            mmr_lambda=_read("MMR_LAMBDA", 0.7, float),
            relevance_threshold=_read("RELEVANCE_THRESHOLD", 0.55, float),
            threshold_margin=_read("THRESHOLD_MARGIN", 0.10, float),
            hybrid_enabled=_read("HYBRID_ENABLED", False, bool),
            rrf_k=_read("RRF_K", 60, int),
            max_history_turns=_read("MAX_HISTORY_TURNS", 3, int),
        )

    def require_api_key(self) -> str:
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key "
                "from Google AI Studio (https://aistudio.google.com/apikey)."
            )
        return self.gemini_api_key


settings = Settings.from_env()


def configure_console() -> None:
    """Make stdout/stderr UTF-8 so Unicode math (|ψ⟩, ⊗) prints on Windows consoles."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass