"""
embeddings.py - Explicit Gemini embeddings, batching, retry and SHA-256 hashing.

Why explicit (same philosophy as the movie recommender):
  * task types - chunks are embedded with RETRIEVAL_DOCUMENT and questions with
    RETRIEVAL_QUERY (asymmetric retrieval, better accuracy than one shared mode)
  * cost control - `content_hash` fingerprints every chunk, so unchanged chunks
    are never sent to the API again (see store.VectorStore.sync)
  * resilience - rate limits (HTTP 429) and transient server errors are retried
    with exponential backoff and jitter

Vectors are L2-normalized, so a dot product equals cosine similarity.
"""

from __future__ import annotations

import hashlib
import random
import time
from functools import lru_cache
from typing import Callable, TypeVar

import httpx
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config import settings

__all__ = [
    "GeminiError",
    "TASK_DOCUMENT",
    "TASK_QUERY",
    "call_with_retry",
    "content_hash",
    "get_client",
    "GeminiEmbedder",
]

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

T = TypeVar("T")


class GeminiError(RuntimeError):
    """A Gemini API call failed permanently (after retries, or a non-retryable error)."""


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Shared Gemini client (created on first use so offline commands need no API key)."""
    return genai.Client(api_key=settings.require_api_key())


def is_retryable(error: Exception) -> bool:
    if isinstance(error, genai_errors.APIError):
        return getattr(error, "code", None) in _RETRYABLE_CODES
    return isinstance(error, httpx.TransportError)


def _describe(error: Exception) -> str:
    code = getattr(error, "code", None)
    if code in (401, 403):
        return f"{error} - check GEMINI_API_KEY and that the model is enabled for your key."
    if code == 404:
        return f"{error} - check the model name in your .env."
    if code == 429:
        return f"{error} - rate limit or quota exhausted; wait a while and re-run (progress is saved)."
    return str(error)


def call_with_retry(fn: Callable[[], T], *, label: str) -> T:
    """Run `fn`, retrying transient failures with exponential backoff + jitter."""
    last_error: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - classified just below
            if not is_retryable(error):
                raise GeminiError(f"{label} failed: {_describe(error)}") from error
            last_error = error
            if attempt == settings.max_retries:
                break
            delay = min(60.0, settings.retry_base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"  [retry] {label}: {type(error).__name__} - waiting {delay:.1f}s "
                  f"({attempt + 1}/{settings.max_retries})")
            time.sleep(delay)
    raise GeminiError(f"{label} failed after {settings.max_retries} retries: {_describe(last_error)}") from last_error


def content_hash(text: str, source: str) -> str:
    """SHA-256 fingerprint of everything that determines a chunk's vector.

    Same text + same source + same model/dimension/task => same hash => skip embedding.
    Changing any of them changes the hash and triggers a re-embed.
    """
    payload = "\x1f".join([settings.embedding_model, str(settings.embedding_dim), TASK_DOCUMENT, source, text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class GeminiEmbedder:
    """Embeds text with gemini-embedding-001 using explicit task types."""

    def __init__(self, client: genai.Client | None = None) -> None:
        self._client = client
        self._last_request = 0.0

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = get_client()
        return self._client

    def _throttle(self) -> None:
        wait = settings.embed_min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _request(self, texts: list[str], task_type: str) -> np.ndarray:
        def call() -> np.ndarray:
            self._throttle()
            try:
                response = self.client.models.embed_content(
                    model=settings.embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=settings.embedding_dim,
                    ),
                )
            finally:
                self._last_request = time.monotonic()
            vectors = [item.values for item in (response.embeddings or [])]
            if len(vectors) != len(texts):
                raise GeminiError(f"Expected {len(texts)} embeddings, received {len(vectors)}.")
            return np.asarray(vectors, dtype=np.float32)

        return _normalize(call_with_retry(call, label="embedding request"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts (RETRIEVAL_DOCUMENT). Returns an (n, dim) normalized array."""
        if not texts:
            return np.zeros((0, settings.embedding_dim), dtype=np.float32)
        batches = [
            self._request(texts[start:start + settings.embed_batch_size], TASK_DOCUMENT)
            for start in range(0, len(texts), settings.embed_batch_size)
        ]
        return np.vstack(batches)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a user question (RETRIEVAL_QUERY). Returns a (dim,) normalized vector."""
        return self._request([text], TASK_QUERY)[0]