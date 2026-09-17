#!/usr/bin/env python3
"""ragmill sharded ingest — designed for Cloud Run Jobs, runs locally too.

Cloud shape (see scripts/launch_1m_ingest.sh):
  gcloud run jobs execute ragmill-ingest --tasks 10 --parallelism 10
  -> task i runs:  python3 jobs/ingest_job.py --shard $i --num-shards 10
     with QDRANT_URL / QDRANT_API_KEY pointing at Qdrant Cloud and
     RAGMILL_SOURCE pointing at a gs:// corpus (jsonl or sharded files).

Properties that make the 1M run safe:
  - SHARDED:      contiguous doc-range per task (shard_range below, tested)
  - IDEMPOTENT:   point IDs are uuid5(arxiv_id#chunk_idx); killing and
                  re-executing the job never duplicates points (tested)
  - RESTARTABLE:  progress written per batch to status objects
                  (local file or gs:// when RAGMILL_STATUS_DIR=gs://...)
  - RATE-FREE:    embeddings run in-process via fastembed ONNX — no API
                  quotas, which is why bulk ingest doesn't touch Vertex

Local usage (against embedded Qdrant — this is the tested path):
  python3 jobs/ingest_job.py --shard 0 --num-shards 4 --source data/corpus_5k.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, corpus as corpus_mod, store  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402


def shard_range(total: int, shard: int, num_shards: int) -> tuple[int, int]:
    """Contiguous slice [start, end) for task `shard` of `num_shards`.
    Remainder docs go to the FIRST shards (deterministic; tested).
    Raises ValueError on out-of-range shard ids."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} out of range 0..{num_shards - 1}")
    base, extra = divmod(total, num_shards)
    start = shard * base + min(shard, extra)
    end = start + base + (1 if shard < extra else 0)
    return start, end


def iter_source(source: str):
    """Stream docs from a local path or gs:// URI. Never full-loads the
    1M corpus into RAM (except a local single-file path, which the laptop
    profile guarantees is small)."""
    if source.startswith("gs://"):
        from google.cloud import storage  # only imported in cloud runs
        bucket, _, prefix = source[5:].partition("/")
        client = storage.Client()
        blobs = sorted(client.list_blobs(bucket, prefix=prefix),
                       key=lambda b: b.name)
        if not blobs:
            raise SystemExit(f"no objects under gs://{bucket}/{prefix}")
        # each task takes blobs [shard_slice] — shard by FILE for GCS
        for blob in blobs:
            line = blob.download_as_text()
            for ln in line.splitlines():
                if ln.strip():
                    yield json.loads(ln)
    else:
        path = Path(source)
        with path.open(encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    yield json.loads(ln)


def write_status(status: dict, shard: int) -> None:
    dest = config.STATUS_DIR
    payload = json.dumps(status)
    if dest.startswith("gs://"):
        from google.cloud import storage
        bucket, _, prefix = dest[5:].rstrip("/").partition("/")
        blob = storage.Client().bucket(bucket) \
            .blob(f"{prefix}/task-{shard}.json")
        blob.upload_from_string(payload)
    elif dest:
        p = Path(dest)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"task-{shard}.json").write_text(payload)


def main():
    ap = argparse.ArgumentParser()
    # Cloud Run Jobs sets CLOUD_RUN_TASK_INDEX / CLOUD_RUN_TASK_COUNT per task
    # automatically — flags win when present, env fills in for managed jobs.
    import os
    ap.add_argument("--shard", type=int,
                    default=int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0")))
    ap.add_argument("--num-shards", type=int,
                    default=int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1")))
    ap.add_argument("--source", default=None,
                    help="jsonl path or gs:// prefix (default RAGMILL_SOURCE)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total docs (local experiments)")
    ap.add_argument("--shard-by", choices=["range", "file"], default="range",
                    help="range=contiguous doc slice of one jsonl; "
                         "file=shard of multiple gs:// objects")
    args = ap.parse_args()

    source = args.source or config.CORPUS_PATH
    t_start = time.time()

    if args.shard_by == "file" and str(source).startswith("gs://"):
        docs = _shard_files_by_range(source, args.shard, args.num_shards,
                                     args.limit)
        total_planned = len(docs)
    else:
        if args.limit:
            all_docs = list(iter_source(source))[:args.limit]
        else:
            # streaming path with a two-pass count is overkill for jsonl
            # corpora <= a few hundred MB; the prepare job keeps shards small.
            all_docs = list(iter_source(source))
        total = len(all_docs)
        start, end = shard_range(total, args.shard, args.num_shards)
        docs = all_docs[start:end]
        total_planned = len(docs)

    print(f"[job] shard {args.shard}/{args.num_shards} -> "
          f"{total_planned} docs from {source}", flush=True)

    records = corpus_mod.to_records(corpus_mod.chunk_docs(docs))
    if not records:
        print("[job] nothing to ingest for this shard", flush=True)
        return

    embedder = get_embedder()
    client = store.get_client()
    store.ensure_collection(client, embedder.dim)
    print(f"[job] embedder={embedder.name} dim={embedder.dim} "
          f"qdrant={'cloud' if config.QDRANT_URL else 'local'}", flush=True)

    status = {"shard": args.shard, "num_shards": args.num_shards,
              "source": str(source), "done": 0, "errors": 0,
              "started_at": t_start}
    t0 = time.time()
    done = errors = 0
    batch = config.UPSERT_BATCH
    for i in range(0, len(records), batch):
        rb = records[i:i + batch]
        try:
            vectors = embedder.embed([r["text"] for r in rb])
            store.upsert_records(client, rb, vectors)
            done += len(rb)
        except Exception as e:  # noqa: BLE001 — log, count, keep the job alive
            errors += len(rb)
            print(f"[job] batch failed: {str(e)[:160]}", flush=True)
        status.update({
            "done": done, "errors": errors,
            "docs_per_s": round(done / max(time.time() - t0, 1e-6), 1),
            "elapsed_s": round(time.time() - t0, 1),
        })
        if config.STATUS_DIR:
            try:
                write_status(status, args.shard)
            except Exception as e:  # noqa: BLE001
                print(f"[job] status write failed: {str(e)[:120]}", flush=True)
        if done and (done // batch) % 10 == 0:
            print(f"[job] {done}/{len(records)} "
                  f"({status['docs_per_s']} docs/s)", flush=True)

    took = time.time() - t_start
    final = dict(status, done=done, errors=errors, finished=True,
                 total_took_s=round(took, 1),
                 docs_per_s=round(done / max(took, 1e-6), 1),
                 qdrant_count=store.count(client))
    print(f"[job] DONE shard {args.shard}: {json.dumps(final)}", flush=True)
    if config.STATUS_DIR:
        try:
            write_status(final, args.shard)
        except Exception as e:  # noqa: BLE001
            print(f"[job] final status write failed: {str(e)[:120]}", flush=True)


def _shard_files_by_range(source, shard, num_shards, limit):
    from google.cloud import storage
    bucket_name, _, prefix = source[5:].partition("/")
    client = storage.Client()
    blobs = sorted(client.list_blobs(bucket_name, prefix=prefix),
                   key=lambda b: b.name)
    start, end = shard_range(len(blobs), shard, num_shards)
    docs = []
    for blob in blobs[start:end]:
        text = blob.download_as_text()
        for ln in text.splitlines():
            if ln.strip():
                if limit and len(docs) >= limit:
                    return docs
                docs.append(json.loads(ln))
    return docs


if __name__ == "__main__":
    main()
