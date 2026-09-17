#!/usr/bin/env python3
"""Real arXiv corpus loader — NOT exercised in the default local mode.

The local profile ships a synthetic 5K corpus (scripts/gen_corpus.py) so the
whole pipeline runs offline. This script loads the real Kaggle arXiv snapshot
(Cornell-University/arxiv, ~2.2M papers, ~4GB) via kagglehub, or the first N
records via the HuggingFace datasets mirror, and writes the same JSONL shape
the rest of ragmill consumes: {arxiv_id, title, abstract, categories}.

Status: written tonight, NOT run to completion on the laptop by design —
the 1M-doc path is meant to run as the Cloud Run prepare job (see
scripts/launch_1m_ingest.sh), never on the 7.5GB laptop. Kept honest.

Usage:
  python3 scripts/fetch_arxiv.py --source hf    --n 5000 --out data/corpus_hf.jsonl
  python3 scripts/fetch_arxiv.py --source kaggle --n 5000 --out data/corpus_kaggle.jsonl
"""
import argparse
import json
from pathlib import Path

HF_DATASET = "boly28/arxiv-metadata"  # community mirror of the Kaggle dump


def load_hf(n: int):
    from datasets import load_dataset  # pip install datasets (heavy: ~1GB)
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= n:
            break
        aid = str(row.get("id") or f"hf-{i}")
        abstract = (row.get("abstract") or "").strip()
        title = (row.get("title") or "").strip().replace("\n", " ")
        cats = [c.strip() for c in str(row.get("categories") or "").split() if c.strip()]
        yield {"arxiv_id": aid, "title": title, "abstract": abstract,
               "categories": cats}


def load_kaggle(n: int):
    import kagglehub  # pip install kagglehub
    import json as _json
    path = Path(kagglehub.dataset_download("Cornell-University/arxiv"))
    files = sorted(path.rglob("arxiv-metadata-oai*.json")) or sorted(path.rglob("*.json"))
    count = 0
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if count >= n:
                    return
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                count += 1
                yield {
                    "arxiv_id": str(row.get("id", "")),
                    "title": (row.get("title") or "").strip().replace("\n", " "),
                    "abstract": (row.get("abstract") or "").strip(),
                    "categories": [c for c in str(row.get("categories") or "").split()],
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "kaggle"], default="hf")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="data/corpus_real.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    loader = load_hf if args.source == "hf" else load_kaggle
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for doc in loader(args.n):
            f.write(json.dumps(doc) + "\n")
            n += 1
    print(f"wrote {n} docs -> {out}")


if __name__ == "__main__":
    main()
