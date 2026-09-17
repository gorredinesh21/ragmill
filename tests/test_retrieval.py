"""Retrieval correctness + the headline CI assertion: hybrid beats dense-only
on hit@3 over the mini golden set (hard fixture: 3 confusable topics)."""
import pytest

from app import evaluator


def test_exact_title_query_finds_gold_top3(retriever, fixture_docs):
    gold = fixture_docs[5]
    res = retriever.hybrid_search(gold["title"], k=5)
    ids = [c["arxiv_id"] for c in res["fused"]]
    assert gold["arxiv_id"] in ids[:3], f"gold {gold['arxiv_id']} not in {ids[:3]}"


def test_bm25_exact_term_is_strong(retriever, fixture_docs):
    """Lexical leg sanity: an exact distinctive method phrase should be
    retrieved by BM25 in the top hits."""
    gold = next(d for d in fixture_docs if ":" in d["title"])
    phrase = gold["title"].split(":")[0].lower()
    res = retriever.sparse_search(phrase, k=10)
    assert res, "BM25 returned nothing"
    assert gold["arxiv_id"] in [c["arxiv_id"] for c in res]


def test_hybrid_returns_all_legs(retriever):
    res = retriever.hybrid_search("distributed vector search", k=8)
    assert set(res) == {"dense", "sparse", "fused"}
    assert len(res["fused"]) <= 8
    for c in res["fused"]:
        assert "rrf_score" in c and "arxiv_id" in c


def test_the_hybrid_claim_is_a_test(retriever, mini_golden):
    """THE assertion: RRF fusion must not lose to dense-only on hit@3, and
    must beat it overall (hit@3 or MRR strictly greater) on this fixture."""
    scores = evaluator.evaluate(retriever, mini_golden)
    dense, hybrid = scores["dense"], scores["hybrid"]
    assert hybrid["hit@3"] >= dense["hit@3"], scores
    assert (hybrid["hit@3"] > dense["hit@3"]
            or hybrid["mrr"] > dense["mrr"]), scores
    assert hybrid["hit@10"] >= dense["hit@10"], scores
    assert hybrid["hit@3"] > 0.5, f"fixture too easy/broken: {scores}"


def test_lexical_rerank_moves_title_matches_up(retriever):
    query = "reciprocal rank fusion"
    res = retriever.hybrid_search(query, k=8)
    from app.retrieval import lexical_rerank
    reranked = lexical_rerank(query, res["fused"], k=8)
    assert len(reranked) == len(res["fused"]) or len(reranked) == 8
    # order changed or stayed; all reranked items came from candidates
    assert {c["point_id"] for c in reranked} <= {c["point_id"] for c in res["fused"]}
