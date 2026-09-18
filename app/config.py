"""ragmill configuration — everything env-driven, local mode is the default.

Local mode (default): zero API keys.
  - dense  : fastembed ONNX bge-small-en-v1.5 (384d), cached under ~/.cache
  - sparse : BM25 (rank-bm25) in-process
  - store  : Qdrant local (embedded, file-persisted) via qdrant-client
  - llm    : mock (RAGMILL_LLM=mock) unless GOOGLE credentials exist

Cloud mode (later): RAGMILL_PROFILE=cloud + QDRANT_URL/QDRANT_API_KEY,
Vertex AI gemini-2.5-flash for rerank/answer via the metadata-server pattern.
"""
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAGMILL_DATA_DIR", APP_ROOT / "data"))

# -- profile ----------------------------------------------------------------
PROFILE = os.environ.get("RAGMILL_PROFILE", "local")  # local | cloud

# -- embeddings -------------------------------------------------------------
# auto: try fastembed (real 384d bge-small), fall back to hashing embedder.
EMBEDDER = os.environ.get("RAGMILL_EMBEDDER", "auto")  # auto | fastembed | hash
EMBED_MODEL = os.environ.get("RAGMILL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Model weights are ~80MB. Downloaded once; cache is shared across runs.
EMBED_CACHE = os.environ.get(
    "RAGMILL_EMBED_CACHE",
    str(Path.home() / ".cache" / "ragmill-fastembed"),
)
EMBED_BATCH = int(os.environ.get("RAGMILL_EMBED_BATCH", "64"))
EMBED_DIM_FALLBACK = 384  # hashing embedder also emits 384d

# -- vector store -----------------------------------------------------------
COLLECTION = os.environ.get("RAGMILL_COLLECTION", "ragmill")
QDRANT_URL = os.environ.get("QDRANT_URL", "")            # cloud mode only
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_LOCAL_PATH = os.environ.get(
    "RAGMILL_QDRANT_PATH", str(DATA_DIR / "qdrant_local"))
# Read-only pre-ingested Qdrant store shipped in the Docker image
# (scripts/build_index_snapshot.py). At startup, if the live store above is
# empty, it is materialized from this snapshot so a fresh Cloud Run instance
# answers queries within seconds — no external services, no re-embedding.
INDEX_SNAPSHOT_DIR = Path(os.environ.get(
    "RAGMILL_SNAPSHOT_DIR", str(DATA_DIR / "index_snapshot")))

# -- corpus / retrieval -----------------------------------------------------
CORPUS_PATH = Path(os.environ.get(
    "RAGMILL_CORPUS", str(DATA_DIR / "corpus_5k.jsonl")))
GOLDEN_PATH = Path(os.environ.get(
    "RAGMILL_GOLDEN", str(DATA_DIR / "golden_60.jsonl")))
TOP_K_RETRIEVE = int(os.environ.get("RAGMILL_TOP_K", "20"))
TOP_K_FINAL = int(os.environ.get("RAGMILL_TOP_K_FINAL", "8"))
RRF_K = int(os.environ.get("RAGMILL_RRF_K", "60"))  # standard RRF constant

# -- LLM --------------------------------------------------------------------
# mock | vertex  (mock is default in local profile; tests force mock)
LLM_MODE = os.environ.get("RAGMILL_LLM", "mock" if PROFILE == "local" else "vertex")
LLM_MODEL = os.environ.get("RAGMILL_LLM_MODEL", "gemini-2.5-flash")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "personal-project-dg21")
GCP_REGION = os.environ.get("RAGMILL_REGION", "us-central1")
LLM_TIMEOUT_S = float(os.environ.get("RAGMILL_LLM_TIMEOUT", "60"))
LLM_MAX_ATTEMPTS = int(os.environ.get("RAGMILL_LLM_ATTEMPTS", "4"))

# -- ingest -----------------------------------------------------------------
UPSERT_BATCH = int(os.environ.get("RAGMILL_UPSERT_BATCH", "100"))
STATUS_DIR = os.environ.get("RAGMILL_STATUS_DIR", "")  # cloud: gs://bucket/path
MAX_INGEST_DOCS = int(os.environ.get("RAGMILL_MAX_INGEST", "20000"))
