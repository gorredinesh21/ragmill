"""Embedding layer: fastembed ONNX bge-small-en-v1.5 (384d) with an honest
hashing fallback so ragmill runs even on a machine with no model cache and
no network.

Two backends behind one interface:
  FastEmbedDense  — BAAI/bge-small-en-v1.5 via fastembed (ONNX int8, no torch,
                    ~80MB weights cached once under ~/.cache/ragmill-fastembed)
  HashingDense    — deterministic feature hashing: unigram+bigram tokens are
                    hashed into 384 dims with sublinear-TF + L2 norm. Not
                    semantic — it behaves like a compressed lexical index —
                    but it keeps the whole pipeline (Qdrant, RRF, evals,
                    SSE API) runnable keyless and offline. Which backend is
                    active is always reported by /api/healthz and the README.
"""
import hashlib
import logging
import math
import re
import threading

from . import config

log = logging.getLogger("ragmill.embed")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A tiny stopword list shared by the hashing embedder and BM25 tokenization.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "be", "been", "by", "as", "at", "from",
    "this", "that", "these", "those", "it", "its", "we", "our", "their",
    "than", "then", "so", "such", "can", "could", "may", "might", "will",
    "would", "should", "has", "have", "had", "not", "no", "but", "into",
    "over", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "again", "further", "once",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer with stopwords kept OUT of dense hashing
    input but handled by BM25 separately. Returns raw tokens (no stopword
    removal) — callers filter when needed."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens minus stopwords; used by hashing embedder + rerank heuristics."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def doc_text(doc: dict) -> str:
    """The canonical 'chunk' of an arXiv-style doc: title + abstract.
    (arXiv abstracts fit one chunk; multi-chunk is a documented roadmap item.)"""
    return f"{doc.get('title', '').strip()}\n\n{doc.get('abstract', '').strip()}"


# ---------------------------------------------------------------------------
# Backend 1: fastembed (real dense embeddings)
# ---------------------------------------------------------------------------

class FastEmbedDense:
    name = "fastembed:BAAI/bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding  # deferred: heavy import
        self._model = TextEmbedding(
            config.EMBED_MODEL, cache_dir=config.EMBED_CACHE)
        self.dim = int(list(self._model.embed(["warmup"]))[0].shape[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = []
        for i in range(0, len(texts), config.EMBED_BATCH):
            batch = texts[i:i + config.EMBED_BATCH]
            out.extend([v.tolist() for v in self._model.embed(batch)])
        return out


# ---------------------------------------------------------------------------
# Backend 2: deterministic hashing embedder (honest fallback — lexical, not
# semantic). Unigrams + bigrams of content tokens hashed to 384 dims with
# sublinear TF and L2 normalization, so cosine similarity is a lexical
# overlap proxy (close to a compressed TF-IDF without the IDF).
# ---------------------------------------------------------------------------

class HashingDense:
    name = "hashing:unigram+bigram-384d (lexical fallback)"

    def __init__(self, dim: int = config.EMBED_DIM_FALLBACK):
        self.dim = dim

    @staticmethod
    def _bucket(term: str, dim: int) -> int:
        h = hashlib.md5(term.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = content_tokens(text)
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for i in range(len(toks) - 1):
                bg = f"{toks[i]}_{toks[i+1]}"
                counts[bg] = counts.get(bg, 0) + 1
            for term, c in counts.items():
                vec[self._bucket(term, self.dim)] += 1.0 + math.log(c)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# ---------------------------------------------------------------------------
# Singleton accessor with auto-detection
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_embedder = None
_init_attempted = False


def get_embedder():
    """Process-wide embedder. `auto` tries fastembed once and permanently
    falls back to hashing on failure (no retry storms)."""
    global _embedder, _init_attempted
    with _lock:
        if _embedder is not None:
            return _embedder
        if _init_attempted:
            return _embedder
        _init_attempted = True
        choice = config.EMBEDDER
        if choice in ("auto", "fastembed"):
            try:
                _embedder = FastEmbedDense()
                log.info("embedder ready: %s (dim=%d)", _embedder.name, _embedder.dim)
                return _embedder
            except Exception as e:  # noqa: BLE001 — any failure => fallback
                if choice == "fastembed":
                    raise
                log.warning("fastembed unavailable (%s); "
                            "falling back to hashing embedder", str(e)[:120])
        _embedder = HashingDense()
        log.info("embedder ready: %s (dim=%d)", _embedder.name, _embedder.dim)
        return _embedder


def reset_embedder():
    """Test hook."""
    global _embedder, _init_attempted
    with _lock:
        _embedder = None
        _init_attempted = False
