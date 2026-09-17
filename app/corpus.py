"""Corpus loading + chunking.

A "doc" is {arxiv_id, title, abstract, categories}. The canonical chunk for
an arXiv-style record is `title + "\n\n" + abstract` — almost always a single
chunk (~150–300 tokens), so chunk_idx is 0 for the local corpus but the code
is chunk-ready (point IDs include the chunk index) for the multi-chunk
roadmap (full-text PDFs).
"""
import json
import uuid
from pathlib import Path

from .embeddings import doc_text

NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # RFC 4122 URL ns
MIN_ABSTRACT_CHARS = 40  # drop stub records with no real abstract


def point_id(arxiv_id: str, chunk_idx: int = 0) -> str:
    """Deterministic UUIDv5 point ID -> upserts are idempotent across shards
    and re-runs (the property that makes the 1M ingest restartable)."""
    return str(uuid.uuid5(NAMESPACE, f"ragmill#{arxiv_id}#{chunk_idx}"))


def load_corpus(path) -> list[dict]:
    path = Path(path)
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return chunk_docs(docs)


def chunk_docs(docs: list[dict]) -> list[dict]:
    """Dedupe by arxiv_id, drop empty/short abstracts, attach text+chunk_idx.
    Returns a stable new list (input order preserved otherwise)."""
    seen: set[str] = set()
    out = []
    for d in docs:
        aid = str(d.get("arxiv_id", "")).strip()
        abstract = (d.get("abstract") or "").strip()
        title = (d.get("title") or "").strip()
        if not aid or aid in seen:
            continue
        if len(abstract) < MIN_ABSTRACT_CHARS or not title:
            continue
        seen.add(aid)
        text = doc_text({"title": title, "abstract": abstract})
        out.append({
            "arxiv_id": aid,
            "title": title,
            "abstract": abstract,
            "categories": d.get("categories") or [],
            "chunk_idx": 0,
            "text": text,
        })
    return out


def to_records(docs: list[dict]) -> list[dict]:
    """Attach the deterministic point_id to each chunk."""
    for d in docs:
        d["point_id"] = point_id(d["arxiv_id"], d["chunk_idx"])
    return docs
