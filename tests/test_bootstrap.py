"""Boot-time snapshot restore — the cloud cold-start mechanism.

A fresh Cloud Run instance has an empty live store; the image bundles a
pre-ingested Qdrant snapshot (data/index_snapshot) which store.restore_
snapshot() materializes at startup. These tests cover the restore rules and
the full boot flow (snapshot -> open -> BM25 rebuild -> searchable), without
touching the real bundled 23MB artifact.
"""
from pathlib import Path

from app import store


def _fake_snapshot(path: Path) -> Path:
    col = path / "collection" / "ragmill"
    col.mkdir(parents=True)
    (col / "storage.sqlite").write_bytes(b"snapshot-bytes")
    (path / "meta.json").write_text("{}")
    return path


def test_restore_into_missing_live_dir(tmp_path):
    snap = _fake_snapshot(tmp_path / "snap")
    live = tmp_path / "live"
    assert store.restore_snapshot(live, snap) is True
    assert (live / "collection" / "ragmill" / "storage.sqlite").read_bytes() \
        == b"snapshot-bytes"
    assert (live / "meta.json").exists()


def test_restore_replaces_stale_empty_live_dir(tmp_path):
    snap = _fake_snapshot(tmp_path / "snap")
    live = tmp_path / "live"
    (live / "collection").mkdir(parents=True)  # exists but EMPTY (no collections)
    live.joinpath("meta.json").write_text("{}")
    assert store.restore_snapshot(live, snap) is True
    assert (live / "collection" / "ragmill" / "storage.sqlite").exists()


def test_restore_never_clobbers_existing_store(tmp_path):
    snap = _fake_snapshot(tmp_path / "snap")
    live = _fake_snapshot(tmp_path / "live")
    (live / "collection" / "ragmill" / "storage.sqlite").write_bytes(b"live")
    assert store.restore_snapshot(live, snap) is False
    assert (live / "collection" / "ragmill" / "storage.sqlite").read_bytes() \
        == b"live"


def test_restore_without_snapshot_is_a_noop(tmp_path):
    live = tmp_path / "live"
    assert store.restore_snapshot(live, tmp_path / "no-such-snapshot") is False
    assert not live.exists()


def test_boot_flow_snapshot_to_ready_retriever(tmp_path, monkeypatch):
    """Mechanism end-to-end: ingest into a build store -> publish as the
    snapshot -> a FRESH instance (empty live dir) restores it and rebuilds
    BM25 from payloads — zero re-embedding, ready to search."""
    import app.config as config
    from app.retrieval import HybridRetriever
    from scripts.build_index_snapshot import write_snapshot
    from scripts.gen_corpus import gen_docs

    # 1. build a small real store
    monkeypatch.setattr(config, "QDRANT_LOCAL_PATH", str(tmp_path / "build"))
    r1 = HybridRetriever()
    stats = r1.ingest(gen_docs(60))
    n = stats["n_in"]
    assert n > 0
    r1.client.close()

    # 2. publish it as the bundled snapshot
    snap = write_snapshot(tmp_path / "build", tmp_path / "snap")

    # 3. fresh instance: live dir empty -> restore kicks in via config defaults
    monkeypatch.setattr(config, "QDRANT_LOCAL_PATH", str(tmp_path / "live"))
    monkeypatch.setattr(config, "INDEX_SNAPSHOT_DIR", snap)
    assert store.restore_snapshot() is True

    r2 = HybridRetriever()
    assert not r2.ready                       # BM25 not rebuilt yet
    assert r2.load_from_store() == n          # payloads -> BM25 + corpus map
    assert r2.ready and len(r2.records) == n
    hits = r2.hybrid_search("sparse attention for language modeling", k=5)
    assert hits["fused"], "restored index must be searchable"
    r2.client.close()
