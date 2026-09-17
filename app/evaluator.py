"""Retrieval evaluation: golden set -> hit@3 / hit@10 / MRR per mode
(dense / sparse / hybrid). Deterministic — no LLM involved — so it runs in
CI (tests/test_retrieval.py asserts hybrid >= dense, making the hybrid claim
a test, not a sentence).
"""
import json
import time
from pathlib import Path


def load_golden(path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _rank_of(expected_id: str, ranked_ids: list[str]) -> int | None:
    for i, pid in enumerate(ranked_ids, start=1):
        if pid == expected_id:
            return i
    return None


def metrics_for(ranks: list[int | None]) -> dict:
    """ranks: 1-based rank of the gold doc per query (None = miss)."""
    n = len(ranks) or 1
    return {
        "hit@3": round(sum(1 for r in ranks if r is not None and r <= 3) / n, 4),
        "hit@10": round(sum(1 for r in ranks if r is not None and r <= 10) / n, 4),
        "mrr": round(sum(0.0 if r is None else 1.0 / r for r in ranks) / n, 4),
    }


def evaluate(retriever, golden: list[dict]) -> dict:
    """golden: [{query, arxiv_id}]. Runs all three retrieval modes once per
    query (dense search, sparse search, RRF fusion of both)."""
    dense_ranks, sparse_ranks, hybrid_ranks = [], [], []
    t0 = time.time()
    for row in golden:
        want = row["arxiv_id"]
        res = retriever.hybrid_search(row["query"], k=10, fetch=40)
        dense_ranks.append(_rank_of(want, [c["arxiv_id"] for c in res["dense"]]))
        sparse_ranks.append(_rank_of(want, [c["arxiv_id"] for c in res["sparse"]]))
        hybrid_ranks.append(_rank_of(want, [c["arxiv_id"] for c in res["fused"]]))
    per_query_s = (time.time() - t0) / max(len(golden), 1)
    return {
        "dense": metrics_for(dense_ranks),
        "sparse": metrics_for(sparse_ranks),
        "hybrid": metrics_for(hybrid_ranks),
        "delta_hit10_hybrid_vs_dense": round(
            metrics_for(hybrid_ranks)["hit@10"]
            - metrics_for(dense_ranks)["hit@10"], 4),
        "latency_ms_per_query": round(per_query_s * 1000, 1),
    }
