import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Env must be set BEFORE app.config import — keyless, deterministic tests.
os.environ.setdefault("RAGMILL_LLM", "mock")
os.environ.setdefault("RAGMILL_PROFILE", "local")
_TMP = tempfile.mkdtemp(prefix="ragmill-test-")
os.environ["RAGMILL_QDRANT_PATH"] = os.path.join(_TMP, "qdrant")

from app import corpus as corpus_mod  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402
from scripts.gen_corpus import gen_docs, gen_golden  # noqa: E402


@pytest.fixture(scope="session")
def fixture_docs():
    """150 docs drawn from ONLY 3 topics so within-topic confusion is real —
    a harder retrieval fixture than the full 5K spread (deliberate: the
    hybrid-vs-dense CI assertion needs dense to genuinely miss sometimes)."""
    docs = gen_docs(900, seed=99)
    keep = [d for d in docs if any(
        t in d["title"].lower() + d["abstract"].lower()
        for t in ("retrieval-augmented", "vector databases", "federated learning"))]
    keep = keep[:150]
    assert len(keep) >= 60, "fixture corpus too small — widen topic filter"
    return keep


@pytest.fixture(scope="session")
def retriever(fixture_docs):
    """One HybridRetriever for the whole suite (model load is the slow part).
    Injected into the FastAPI app so API tests share it (qdrant local mode
    locks its storage dir — one client per path)."""
    r = HybridRetriever()
    stats = r.ingest(fixture_docs)
    assert stats["n_in"] == len(fixture_docs)

    from app import main as app_main
    app_main._state["retriever"] = r
    app_main._state["ingest_stats"] = stats
    return r


@pytest.fixture(scope="session")
def mini_golden(fixture_docs):
    return gen_golden(fixture_docs, n=40, seed=3)


@pytest.fixture(scope="session")
def client(retriever):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
