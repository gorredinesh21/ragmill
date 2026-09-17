#!/usr/bin/env python3
"""Run the golden-set eval from the CLI (same code path as POST /api/eval).

Usage: python3 eval/run_eval.py [--limit 60] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, corpus as corpus_mod, evaluator, store  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", default=str(config.CORPUS_PATH))
    ap.add_argument("--golden", default=str(config.GOLDEN_PATH))
    ap.add_argument("--json", default=None, help="also write results to file")
    args = ap.parse_args()

    r = HybridRetriever()
    restored = r.load_from_store() if store.count(r.client) else 0
    if restored:
        print(f"restored {restored} docs from persisted qdrant (no re-embed)")
    else:
        stats = r.ingest(corpus_mod.load_corpus(args.corpus))
        print(f"ingested: {json.dumps(stats)}")

    golden = evaluator.load_golden(args.golden)
    if args.limit:
        golden = golden[:args.limit]
    scores = evaluator.evaluate(r, golden)
    out = {"n_queries": len(golden), "embedder": r.embedder.name,
           "metrics": scores}

    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
