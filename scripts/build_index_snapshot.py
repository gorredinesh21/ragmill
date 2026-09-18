#!/usr/bin/env python3
"""Build the boot-time index snapshot bundled into the Docker image.

WHY: the deployed Cloud Run service starts with an empty filesystem — the
laptop's `data/qdrant_local/` never ships. To make the live demo usable
within seconds of a cold start, the image carries a pre-ingested Qdrant
store (`data/index_snapshot/`, ~23MB for the 5K corpus). At startup
`app.store.restore_snapshot()` copies it to the live (writable) path and
`HybridRetriever.load_from_store()` rebuilds BM25 from the payloads —
measured ~8-10s total, no re-embedding, no external services.

Regenerating the snapshot (deterministic — same corpus seed, same uuid5
point ids, so the result is byte-equivalent in content):

    python3 scripts/build_index_snapshot.py            # 5K corpus, ~18 min CPU
    python3 scripts/build_index_snapshot.py --n 1000   # smaller subset

The existing live store can also be re-published as a snapshot without
re-embedding (what was done for the initial artifact):

    python3 scripts/build_index_snapshot.py --from data/qdrant_local
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def snapshot_has_collection(path: Path) -> bool:
    return (path / "collection").is_dir() and any(
        (path / "collection").iterdir())


def write_snapshot(src: Path, dst: Path) -> Path:
    """Copy the storage tree only — never runtime .lock files."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "collection", dst / "collection")
    shutil.copy2(src / "meta.json", dst / "meta.json")
    for lock in dst.rglob(".lock"):
        lock.unlink()
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000,
                    help="corpus size to (re)generate and ingest")
    ap.add_argument("--from", dest="from_dir", type=Path, default=None,
                    help="publish an existing live qdrant dir as the snapshot "
                         "(no re-embedding)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "index_snapshot")
    args = ap.parse_args()

    if args.from_dir:
        src = args.from_dir
        if not snapshot_has_collection(src):
            sys.exit(f"no qdrant collection at {src}")
    else:
        from app.retrieval import HybridRetriever
        from scripts.gen_corpus import gen_docs
        with tempfile.TemporaryDirectory(prefix="ragmill-snap-") as td:
            os_env = {}
            # point the live store at the temp dir for this build
            import os
            os_env["RAGMILL_QDRANT_PATH"] = str(Path(td) / "qdrant")
            for k, v in os_env.items():
                os.environ[k] = v
            r = HybridRetriever()
            docs = gen_docs(args.n)
            stats = r.ingest(docs)
            print(f"ingested: {json.dumps(stats)}")
            src = Path(os_env["RAGMILL_QDRANT_PATH"])

    write_snapshot(src, args.out)

    # verify the snapshot opens standalone
    from qdrant_client import QdrantClient
    c = QdrantClient(path=str(args.out))
    n = c.count("ragmill", exact=True).count
    c.close()
    for lock in args.out.rglob(".lock"):
        lock.unlink()
    size_mb = sum(f.stat().st_size for f in args.out.rglob("*")
                  if f.is_file()) / 1e6
    print(f"snapshot: {n} points, {size_mb:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
