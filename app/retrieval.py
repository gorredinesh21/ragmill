"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) fused with Reciprocal
Rank Fusion, then a rerank step (Gemini when credentials exist; a transparent
lexical-overlap heuristic locally).

RRF (Cormack et al. 2009):   score(d) = Σ_lists 1 / (k + rank_i(d)),  k=60.
Pure function -> unit-testable with known math (tests/test_rrf.py).
"""
import logging
import time

from . import config, corpus as corpus_mod, embeddings, store
from .embeddings import content_tokens

log = logging.getLogger("ragmill.retrieval")


# ---------------------------------------------------------------------------
# RRF — pure
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists: list[list[str]], k: int = None,
             weights: list[float] = None) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into [(id, rrf_score)] sorted desc.

    ranked_lists: each list is IDs best-first (rank 1 = first element).
    k: RRF constant (60 standard). weights: per-list multiplier (default 1).
    Ties break deterministically by id for stable evals/tests.
    """
    k = config.RRF_K if k is None else k
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# BM25 sparse index (in-process; fine to ~100K docs — see README for the
# 1M plan, where sparse moves into Qdrant server-side).
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        self._bm25 = None
        self._ids: list[str] = []
        self._docs: dict[str, dict] = {}

    def build(self, records: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._ids = [r["point_id"] for r in records]
        self._docs = {r["point_id"]: r for r in records}
        tokenized = [content_tokens(r["text"]) for r in records]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int) -> list[dict]:
        if not self._bm25:
            return []
        t0 = time.time()
        scores = self._bm25.get_scores(content_tokens(query))
        took_ms = (time.time() - t0) * 1000
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{
            "point_id": self._ids[i],
            "score": float(scores[i]),
            "arxiv_id": self._docs[self._ids[i]]["arxiv_id"],
            "title": self._docs[self._ids[i]]["title"],
            "abstract": self._docs[self._ids[i]]["abstract"],
            "categories": self._docs[self._ids[i]].get("categories") or [],
            "took_ms": took_ms,
        } for i in order if scores[i] > 0]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Owns the Qdrant collection, the BM25 index and the corpus map.
    In local profile everything is in-process (Qdrant embedded + rank-bm25).
    """

    def __init__(self, client=None):
        self.client = client or store.get_client()
        self.embedder = embeddings.get_embedder()
        self.bm25 = BM25Index()
        self.records: dict[str, dict] = {}
        store.ensure_collection(self.client, self.embedder.dim)
        self._bm25_ready = False

    # -- ingest -------------------------------------------------------------
    def ingest(self, docs: list[dict]) -> dict:
        """Chunk -> embed -> idempotent Qdrant upsert + BM25 build.
        Returns {n_in, skipped, took_s, docs_per_s, count}."""
        t0 = time.time()
        records = corpus_mod.to_records(corpus_mod.chunk_docs(docs))
        vectors = self.embedder.embed([r["text"] for r in records])
        store.upsert_records(self.client, records, vectors)
        # corpus map + BM25 are rebuilt over everything currently in memory
        for r in records:
            self.records[r["point_id"]] = r
        self.bm25.build(list(self.records.values()))
        self._bm25_ready = True
        took = time.time() - t0
        n_in = len(records)
        return {
            "n_in": n_in,
            "skipped": len(docs) - n_in,
            "took_s": round(took, 3),
            "docs_per_s": round(n_in / took, 1) if took else None,
            "count": store.count(self.client),
        }

    @property
    def ready(self) -> bool:
        return self._bm25_ready and len(self.records) > 0

    # -- reload persisted index (no re-embedding) ---------------------------
    def load_from_store(self) -> int:
        """Rebuild the in-process BM25 index + corpus map from what is
        already persisted in Qdrant (payloads carry everything needed).
        This is what makes the service restart / demo boot instant."""
        from qdrant_client import models
        offset = None
        n = 0
        while True:
            points, offset = self.client.scroll(
                collection_name=config.COLLECTION,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                pl = p.payload or {}
                text = f"{pl.get('title', '')}\n\n{pl.get('abstract', '')}"
                rec = {
                    "point_id": str(p.id),
                    "arxiv_id": pl.get("arxiv_id"),
                    "title": pl.get("title", ""),
                    "abstract": pl.get("abstract", ""),
                    "categories": pl.get("categories") or [],
                    "chunk_idx": pl.get("chunk_idx", 0),
                    "text": text,
                }
                self.records[rec["point_id"]] = rec
                n += 1
            if offset is None:
                break
        if n:
            self.bm25.build(list(self.records.values()))
            self._bm25_ready = True
        return n

    # -- search -------------------------------------------------------------
    def dense_search(self, query: str, k: int):
        vec = self.embedder.embed([query])[0]
        return store.dense_search(self.client, vec, k)

    def sparse_search(self, query: str, k: int):
        return self.bm25.search(query, k)

    def hybrid_search(self, query: str, k: int,
                      fetch: int = None) -> dict:
        """Dense + sparse -> RRF. `fetch` per-leg candidates (default 4xk)."""
        fetch = fetch or max(k * 4, 20)
        dense = self.dense_search(query, fetch)
        sparse = self.sparse_search(query, fetch)
        fused = rrf_fuse(
            [[h["point_id"] for h in dense],
             [h["point_id"] for h in sparse]],
            k=config.RRF_K,
        )[:k]
        by_id = {h["point_id"]: h for h in dense}
        by_id.update({h["point_id"]: h for h in sparse})
        results = []
        for pid, score in fused:
            h = dict(by_id[pid])
            h["rrf_score"] = round(score, 6)
            results.append(h)
        return {"dense": dense[:k], "sparse": sparse[:k], "fused": results}


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------

def lexical_rerank(query: str, candidates: list[dict],
                   k: int = None) -> list[dict]:
    """Keyless rerank: RRF score + title-overlap bonus. Honest about being
    a heuristic — the README and the SSE `rerank.stage` event both say
    'lexical' vs 'gemini'."""
    k = k or config.TOP_K_FINAL
    q = set(content_tokens(query))
    scored = []
    for c in candidates:
        title_toks = set(content_tokens(c.get("title", "")))
        overlap = len(q & title_toks)
        bonus = 0.15 * overlap / (overlap + 2)  # saturating bonus
        scored.append((c["rrf_score"] + bonus, c))
    scored.sort(key=lambda t: (-t[0], t[1]["point_id"]))
    out = []
    for s, c in scored[:k]:
        c = dict(c)
        c["rerank_score"] = round(s, 6)
        out.append(c)
    return out
