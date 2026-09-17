"""RRF math — pure function, hand-computed expectations."""
from app.retrieval import rrf_fuse


def _close(a, b, eps=1e-9):
    return abs(a - b) < eps


def test_rrf_known_scores():
    # k=60: A rank1 in list1 + rank2 in list2 = 1/61 + 1/62
    fused = rrf_fuse([["A", "B"], ["B", "A"]], k=60)
    d = dict(fused)
    assert _close(d["A"], 1 / 61 + 1 / 62)
    assert _close(d["B"], 1 / 62 + 1 / 61)
    # tie -> deterministic id order
    assert [i for i, _ in fused] == ["A", "B"]


def test_rrf_disjoint_and_overlap_order():
    # C appears in both lists (1/61 + 1/63); A only list1 rank1 (1/61);
    # B only list1 rank2 (1/62)
    fused = rrf_fuse([["A", "B", "C"], ["C"]], k=60)
    assert [i for i, _ in fused] == ["C", "A", "B"]
    assert _close(dict(fused)["C"], 1 / 61 + 1 / 63)


def test_rrf_standard_k_constant():
    # default k comes from config (60); explicit override works
    fused = rrf_fuse([["x", "y"]], k=1)
    assert _close(dict(fused)["x"], 1 / 2)


def test_rrf_weights():
    fused = rrf_fuse([["A"], ["B"]], k=60, weights=[2.0, 1.0])
    d = dict(fused)
    assert _close(d["A"], 2.0 / 61)
    assert _close(d["B"], 1.0 / 61)
    assert fused[0][0] == "A"


def test_rrf_empty_and_single():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([["only"]]) == [("only", 1 / 61)]


def test_rrf_truncation_by_caller():
    fused = rrf_fuse([["a", "b", "c"], ["c", "b", "a"]])
    assert len(fused) == 3
    assert set(dict(fused)) == {"a", "b", "c"}
