"""Ingestion mechanics: shard math, uuid5 idempotency, chunker rules."""
import pytest

from app import corpus as corpus_mod, store
from app.corpus import chunk_docs, point_id
from jobs.ingest_job import shard_range


# -- shard math --------------------------------------------------------------

def test_shard_range_even():
    assert shard_range(100, 7, 10) == (70, 80)
    assert shard_range(100, 0, 10) == (0, 10)
    assert shard_range(100, 9, 10) == (90, 100)


def test_shard_range_remainder_goes_to_first_shards():
    # 105 docs, 10 shards: first 5 shards get 11, last 5 get 10
    assert shard_range(105, 0, 10) == (0, 11)
    assert shard_range(105, 4, 10) == (44, 55)
    assert shard_range(105, 5, 10) == (55, 65)
    assert shard_range(105, 9, 10) == (95, 105)
    # full coverage, no overlap
    ranges = [shard_range(105, i, 10) for i in range(10)]
    flat = [x for r in ranges for x in range(*r)]
    assert flat == list(range(105))


def test_shard_range_invalid():
    with pytest.raises(ValueError):
        shard_range(100, 10, 10)
    with pytest.raises(ValueError):
        shard_range(100, -1, 10)
    with pytest.raises(ValueError):
        shard_range(100, 0, 0)


# -- uuid5 point ids ----------------------------------------------------------

def test_point_id_deterministic_and_chunk_scoped():
    a1, a2 = point_id("2401.12345", 0), point_id("2401.12345", 0)
    b = point_id("2401.12345", 1)
    c = point_id("2401.12346", 0)
    assert a1 == a2          # stable across runs -> idempotent upserts
    assert a1 != b           # chunk-scoped
    assert a1 != c           # doc-scoped
    import uuid
    uuid.UUID(a1)            # parses as a UUID (qdrant accepts)


# -- chunker -------------------------------------------------------------------

def test_chunk_docs_dedupe_and_filters():
    good = {"arxiv_id": "1", "title": "T", "abstract": "x" * 80}
    dup = {"arxiv_id": "1", "title": "T2", "abstract": "y" * 80}
    empty_abs = {"arxiv_id": "2", "title": "T", "abstract": "   "}
    short_abs = {"arxiv_id": "3", "title": "T", "abstract": "tiny"}
    no_title = {"arxiv_id": "4", "title": "", "abstract": "z" * 100}
    out = chunk_docs([good, dup, empty_abs, short_abs, no_title])
    assert [d["arxiv_id"] for d in out] == ["1"]
    assert out[0]["text"].startswith("T\n\n")
    assert out[0]["chunk_idx"] == 0


def test_to_records_attaches_point_ids():
    docs = chunk_docs([{"arxiv_id": "9", "title": "A", "abstract": "b" * 100}])
    recs = corpus_mod.to_records(docs)
    assert recs[0]["point_id"] == point_id("9", 0)


# -- idempotent upserts (against embedded qdrant, cheap vectors) ---------------

def _tmp_client(tmp_path):
    from qdrant_client import QdrantClient
    return QdrantClient(path=str(tmp_path / "q"))


def test_double_upsert_count_unchanged(tmp_path):
    client = _tmp_client(tmp_path)
    dim = 16
    store.ensure_collection(client, dim, collection="t-idem")
    recs = corpus_mod.to_records(chunk_docs([
        {"arxiv_id": f"2401.{i:05d}", "title": f"Doc {i}",
         "abstract": "content " * 20} for i in range(30)
    ]))
    vecs = [[0.1] * dim for _ in recs]
    store.upsert_records(client, recs, vecs, collection="t-idem")
    assert store.count(client, "t-idem") == 30
    # re-run the same shard: uuid5 ids overwrite, count must not grow
    store.upsert_records(client, recs, vecs, collection="t-idem")
    assert store.count(client, "t-idem") == 30
    # a changed vector for the same id still keeps the count
    vecs2 = [[0.9] * dim for _ in recs]
    store.upsert_records(client, recs, vecs2, collection="t-idem")
    assert store.count(client, "t-idem") == 30
    client.close()


def test_dense_search_returns_payload(tmp_path):
    client = _tmp_client(tmp_path)
    dim = 8
    store.ensure_collection(client, dim, collection="t-search")
    recs = corpus_mod.to_records(chunk_docs([
        {"arxiv_id": "2401.00001", "title": "Alpha paper",
         "abstract": "words " * 30},
        {"arxiv_id": "2401.00002", "title": "Beta paper",
         "abstract": "other " * 30},
    ]))
    store.upsert_records(client, recs,
                         [[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6],
                         collection="t-search")
    hits = store.dense_search(client, [1.0] + [0.0] * 7, 2, collection="t-search")
    assert hits[0]["arxiv_id"] == "2401.00001"
    assert hits[0]["score"] > 0.9
    client.close()
