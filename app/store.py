"""Qdrant store — one client constructor for local (embedded) and cloud modes.

Local:  QdrantClient(path=...)   embedded qdrant, file-persisted, zero infra.
Cloud:  QdrantClient(url=..., api_key=...)  Qdrant Cloud free cluster, used by
        the sharded ingest job and (later) the deployed service.

Boot bootstrap: a pre-ingested store snapshot (data/index_snapshot, built by
scripts/build_index_snapshot.py) is shipped in the Docker image; a fresh
instance with an empty live store restores it at startup (see
restore_snapshot) and rebuilds BM25 from the payloads — no re-embedding.
"""
import logging
import shutil
import time
from pathlib import Path

from . import config

log = logging.getLogger("ragmill.store")


def get_client():
    from qdrant_client import QdrantClient
    if config.QDRANT_URL:
        return QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=30,
        )
    return QdrantClient(path=config.QDRANT_LOCAL_PATH)


def _has_collection(path: Path) -> bool:
    """A local qdrant dir holds data iff its collection/ subdir is non-empty."""
    col = Path(path) / "collection"
    try:
        return col.is_dir() and any(col.iterdir())
    except OSError:
        return False


def restore_snapshot(live_path=None, snapshot_dir=None) -> bool:
    """Materialize the bundled read-only index snapshot into the live
    (writable) qdrant path when the live store is empty. Never clobbers an
    existing collection. Returns True if a restore happened."""
    live = Path(live_path or config.QDRANT_LOCAL_PATH)
    snap = Path(snapshot_dir or config.INDEX_SNAPSHOT_DIR)
    if _has_collection(live):
        return False  # live store already has data — nothing to do
    if not _has_collection(snap):
        return False  # no bundled snapshot (e.g. repo checkout without one)
    t0 = time.time()
    live.mkdir(parents=True, exist_ok=True)
    col = live / "collection"
    if col.exists():
        shutil.rmtree(col)
    shutil.copytree(snap / "collection", col)
    shutil.copy2(snap / "meta.json", live / "meta.json")
    log.info("restored index snapshot %s -> %s (%.2fs)",
             snap, live, time.time() - t0)
    return True


def ensure_collection(client, dim: int, collection: str = None) -> bool:
    """Create the collection if missing. Returns True if it already existed."""
    from qdrant_client import models
    collection = collection or config.COLLECTION
    if client.collection_exists(collection):
        return True
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE),
    )
    log.info("created collection %s (dim=%d)", collection, dim)
    return False


def upsert_records(client, records: list[dict], vectors: list[list[float]],
                   collection: str = None) -> None:
    """Idempotent upsert: same point_id overwrites, count never inflates."""
    from qdrant_client import models
    collection = collection or config.COLLECTION
    for i in range(0, len(records), config.UPSERT_BATCH):
        rb = records[i:i + config.UPSERT_BATCH]
        vb = vectors[i:i + config.UPSERT_BATCH]
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=r["point_id"],
                    vector=v,
                    payload={
                        "arxiv_id": r["arxiv_id"],
                        "title": r["title"],
                        "abstract": r["abstract"],
                        "categories": r.get("categories") or [],
                        "chunk_idx": r["chunk_idx"],
                    },
                )
                for r, v in zip(rb, vb)
            ],
            wait=True,
        )


def dense_search(client, vector: list[float], k: int,
                 collection: str = None) -> list[dict]:
    from qdrant_client import models
    collection = collection or config.COLLECTION
    t0 = time.time()
    hits = client.query_points(
        collection_name=collection,
        query=vector,
        limit=k,
        with_payload=True,
    ).points
    took_ms = (time.time() - t0) * 1000
    return [{
        "point_id": str(h.id),
        "score": float(h.score),
        "arxiv_id": h.payload.get("arxiv_id"),
        "title": h.payload.get("title"),
        "abstract": h.payload.get("abstract"),
        "categories": h.payload.get("categories") or [],
        "took_ms": took_ms,
    } for h in hits]


def count(client, collection: str = None) -> int:
    collection = collection or config.COLLECTION
    try:
        return client.count(collection, exact=True).count
    except Exception:  # collection missing
        return 0
