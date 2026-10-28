"""
store.py - Persistent ChromaDB storage, hash synchronization, MMR search.

Two local collections live in one Chroma directory:
    quantum_kb      curated knowledge base (built by ingest.py)
    user_uploads    files added at runtime with /upload

SYNC     Every chunk id is its SHA-256 content hash. `sync()` compares the hashes
         we want with the hashes already stored: unchanged chunks are skipped
         (0 tokens), new ones are embedded and upserted batch by batch (so an
         interrupted run resumes where it stopped), and stale chunks of changed
         files are deleted.

SEARCH   1. embed the question (RETRIEVAL_QUERY)
         2. fetch `fetch_k` nearest chunks from every collection
         3. optional hybrid: add BM25 keyword candidates, fuse ranks with RRF
         4. MMR re-ranking: keep `top_k` chunks that are relevant AND diverse
         5. relevance gate: if even the best chunk is below the threshold the
            result is flagged ungrounded and the caller falls back to general
            knowledge (clearly labeled).

Scores shown to the user are always cosine similarities (vectors are normalized).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from config import BASE_COLLECTION, UPLOAD_COLLECTION, settings
from embeddings import GeminiEmbedder, content_hash

__all__ = ["RetrievedChunk", "SearchResult", "SyncReport", "VectorStore"]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float                      # cosine similarity with the question
    collection: str
    rrf_score: float = 0.0            # only meaningful in hybrid mode
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SearchResult:
    chunks: list[RetrievedChunk]      # chunks that passed the relevance gate (empty if ungrounded)
    grounded: bool
    best_score: float
    threshold: float
    reason: str                       # "ok" | "below_threshold" | "empty_index"
    hybrid: bool


@dataclass
class SyncReport:
    collection: str
    total: int = 0                    # unique chunks wanted
    embedded: int = 0                 # new chunks sent to the embedding API
    skipped: int = 0                  # unchanged chunks (hash match) - zero tokens
    deleted: int = 0                  # stale / removed chunks deleted


def _clean_metadata(metadata: dict) -> dict:
    """Chroma accepts only str / int / float / bool values."""
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _mmr(embeddings: np.ndarray, relevance: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: greedily pick items that are relevant yet not redundant."""
    count = len(relevance)
    k = min(k, count)
    if k <= 0:
        return []
    similarity = embeddings @ embeddings.T
    selected = [int(np.argmax(relevance))]
    remaining = [i for i in range(count) if i not in selected]
    while remaining and len(selected) < k:
        scores = [lam * relevance[i] - (1.0 - lam) * float(np.max(similarity[i, selected])) for i in remaining]
        pick = remaining[int(np.argmax(scores))]
        selected.append(pick)
        remaining.remove(pick)
    return selected


class VectorStore:
    def __init__(self, embedder: Optional[GeminiEmbedder] = None, persist_dir=None) -> None:
        directory = persist_dir or settings.chroma_dir
        directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = embedder
        self._bm25_cache: dict[tuple, tuple] = {}

    @property
    def embedder(self) -> GeminiEmbedder:
        if self._embedder is None:
            self._embedder = GeminiEmbedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def collection(self, name: str):
        return self.client.get_or_create_collection(name=name, configuration={"hnsw": {"space": "cosine"}})

    def count(self, name: str) -> int:
        return self.collection(name).count()

    def reset(self, name: str) -> None:
        """Drop and recreate a collection (needed if the embedding model or dimension changes)."""
        try:
            self.client.delete_collection(name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self.collection(name)
        self._bm25_cache.clear()

    # ------------------------------------------------------------------
    # Hash synchronization
    # ------------------------------------------------------------------

    def sync(
        self,
        collection_name: str,
        chunks: list[Document],
        *,
        prune_missing_sources: bool = False,
        progress: bool = True,
    ) -> SyncReport:
        """Make the collection match `chunks`, embedding only what changed.

        Stale chunks of any source present in `chunks` are always removed. Chunks of
        sources that are absent from `chunks` are removed only with prune_missing_sources.
        """
        collection = self.collection(collection_name)
        scope = "upload" if collection_name == UPLOAD_COLLECTION else "base"
        report = SyncReport(collection=collection_name)

        desired: dict[str, tuple[str, dict]] = {}
        for chunk in chunks:
            metadata = _clean_metadata(chunk.metadata)
            source = str(metadata.get("source", ""))
            chunk_id = content_hash(chunk.page_content, source)
            if chunk_id in desired:
                continue
            metadata["content_hash"] = chunk_id
            metadata["scope"] = scope
            desired[chunk_id] = (chunk.page_content, metadata)
        report.total = len(desired)

        existing_raw = collection.get(include=["metadatas"])
        existing = {cid: (meta or {}).get("source", "") for cid, meta in zip(existing_raw["ids"], existing_raw["metadatas"])}
        desired_sources = {meta["source"] for _, meta in desired.values()}

        new_ids = [cid for cid in desired if cid not in existing]
        report.skipped = report.total - len(new_ids)

        for start in range(0, len(new_ids), settings.embed_batch_size):
            batch_ids = new_ids[start:start + settings.embed_batch_size]
            texts = [desired[cid][0] for cid in batch_ids]
            vectors = self.embedder.embed_documents(texts)
            collection.upsert(
                ids=batch_ids,
                embeddings=vectors.tolist(),
                documents=texts,
                metadatas=[desired[cid][1] for cid in batch_ids],
            )
            report.embedded += len(batch_ids)
            if progress:
                print(f"\r  embedding {report.embedded}/{len(new_ids)} new chunks", end="", file=sys.stdout, flush=True)
        if progress and new_ids:
            print()

        stale = [
            cid for cid, source in existing.items()
            if cid not in desired and (source in desired_sources or prune_missing_sources)
        ]
        for start in range(0, len(stale), 500):
            collection.delete(ids=stale[start:start + 500])
        report.deleted = len(stale)

        self._bm25_cache.clear()
        return report

    def delete_source(self, collection_name: str, source: str) -> int:
        collection = self.collection(collection_name)
        ids = collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        self._bm25_cache.clear()
        return len(ids)

    def list_sources(self, collection_name: str) -> list[dict]:
        """One row per indexed file: source, title, doc_type, chunks."""
        raw = self.collection(collection_name).get(include=["metadatas"])
        rows: dict[str, dict] = {}
        for meta in raw["metadatas"]:
            meta = meta or {}
            source = meta.get("source", "?")
            row = rows.setdefault(source, {
                "source": source,
                "title": meta.get("title", source),
                "doc_type": meta.get("doc_type", ""),
                "origin": meta.get("origin", ""),
                "chunks": 0,
            })
            row["chunks"] += 1
        return sorted(rows.values(), key=lambda row: row["source"])

    # ------------------------------------------------------------------
    # BM25 (optional sparse retrieval)
    # ------------------------------------------------------------------

    def _bm25(self, names: tuple[str, ...]):
        if names in self._bm25_cache:
            return self._bm25_cache[names]
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise RuntimeError("Hybrid search needs rank-bm25: pip install rank-bm25") from error

        ids: list[str] = []
        owners: list[str] = []
        corpus: list[list[str]] = []
        for name in names:
            raw = self.collection(name).get(include=["documents"])
            for chunk_id, document in zip(raw["ids"], raw["documents"]):
                ids.append(chunk_id)
                owners.append(name)
                corpus.append(_tokenize(document or ""))
        index = BM25Okapi(corpus) if corpus else None
        self._bm25_cache[names] = (index, ids, owners)
        return self._bm25_cache[names]

    def _bm25_top(self, query: str, names: tuple[str, ...], limit: int) -> list[tuple[str, str]]:
        index, ids, owners = self._bm25(names)
        tokens = _tokenize(query)
        if index is None or not tokens:
            return []
        scores = index.get_scores(tokens)
        order = np.argsort(scores)[::-1][:limit]
        return [(ids[i], owners[i]) for i in order if scores[i] > 0]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        threshold: Optional[float] = None,
        hybrid: Optional[bool] = None,
        collections: Optional[tuple[str, ...]] = None,
    ) -> SearchResult:
        top_k = top_k or settings.top_k
        fetch_k = fetch_k or settings.fetch_k
        threshold = settings.relevance_threshold if threshold is None else threshold
        hybrid = settings.hybrid_enabled if hybrid is None else hybrid
        names = tuple(collections or (BASE_COLLECTION, UPLOAD_COLLECTION))

        searchable = [name for name in names if self.count(name) > 0]
        if not searchable:
            return SearchResult([], False, 0.0, threshold, "empty_index", hybrid)

        q = self.embedder.embed_query(query)

        # 1) dense candidates from every collection
        pool: dict[str, RetrievedChunk] = {}
        for name in searchable:
            collection = self.collection(name)
            raw = collection.query(
                query_embeddings=[q.tolist()],
                n_results=min(fetch_k, collection.count()),
                include=["documents", "metadatas", "embeddings"],
            )
            for chunk_id, text, meta, emb in zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["embeddings"][0]):
                vector = np.asarray(emb, dtype=np.float32)
                vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
                pool[chunk_id] = RetrievedChunk(chunk_id, text, meta or {}, float(vector @ q), name, embedding=vector)

        dense_ranked = sorted(pool.values(), key=lambda c: c.score, reverse=True)[:fetch_k]
        pool = {c.id: c for c in dense_ranked}

        # 2) optional BM25 candidates, fused with dense ranks (Reciprocal Rank Fusion)
        if hybrid:
            bm25_hits = self._bm25_top(query, tuple(searchable), fetch_k)
            missing: dict[str, list[str]] = {}
            for chunk_id, owner in bm25_hits:
                if chunk_id not in pool:
                    missing.setdefault(owner, []).append(chunk_id)
            for owner, wanted in missing.items():
                raw = self.collection(owner).get(ids=wanted, include=["documents", "metadatas", "embeddings"])
                for chunk_id, text, meta, emb in zip(raw["ids"], raw["documents"], raw["metadatas"], raw["embeddings"]):
                    vector = np.asarray(emb, dtype=np.float32)
                    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
                    pool[chunk_id] = RetrievedChunk(chunk_id, text, meta or {}, float(vector @ q), owner, embedding=vector)

            dense_order = [c.id for c in sorted(pool.values(), key=lambda c: c.score, reverse=True)]
            for rank, chunk_id in enumerate(dense_order, start=1):
                pool[chunk_id].rrf_score += 1.0 / (settings.rrf_k + rank)
            for rank, (chunk_id, _) in enumerate(bm25_hits, start=1):
                pool[chunk_id].rrf_score += 1.0 / (settings.rrf_k + rank)

        candidates = list(pool.values())
        embeddings = np.vstack([c.embedding for c in candidates])
        if hybrid:
            fused = np.asarray([c.rrf_score for c in candidates], dtype=np.float32)
            span = float(fused.max() - fused.min())
            relevance = (fused - fused.min()) / span if span > 0 else np.ones_like(fused)
        else:
            relevance = np.asarray([c.score for c in candidates], dtype=np.float32)

        # 3) MMR keeps the most relevant, least redundant chunks
        picked = [candidates[i] for i in _mmr(embeddings, relevance, top_k, settings.mmr_lambda)]

        # 4) relevance gate (query level) + margin (chunk level)
        best_score = max(c.score for c in candidates)
        if best_score < threshold:
            return SearchResult([], False, best_score, threshold, "below_threshold", hybrid)
        floor = threshold - settings.threshold_margin
        kept = [c for c in picked if c.score >= floor] or [max(candidates, key=lambda c: c.score)]
        for chunk in candidates:
            chunk.embedding = None
        return SearchResult(kept, True, best_score, threshold, "ok", hybrid)