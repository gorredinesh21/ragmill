"""ragmill — RAG service over an arXiv-style corpus.

Local mode (default): zero API keys. fastembed ONNX dense + BM25 sparse +
RRF fusion in Qdrant-local; extractive MockLLM answer with [n] citations.
Cloud mode: same retrieval path; Gemini rerank/answer via Vertex
(RAGMILL_LLM=vertex). Cold-start bootstrap: a pre-ingested index snapshot
shipped in the image (data/index_snapshot) is restored at startup when the
live store is empty, so a fresh instance serves queries in seconds.

Endpoints:
  GET  /api/healthz      liveness + profile/embedder/llm/corpus count
  POST /api/ingest       {source: corpus|folder, path?, arxiv_ids?, limit?}
  POST /api/search       {query, k}  hybrid retrieval, no LLM (keyless demo)
  POST /api/query        {query}  SSE: retrieve -> rerank -> answer (citations)
                         (alias: POST /api/ask — same stream). Always
                         terminates: `done` on success, `error` otherwise.
  POST /api/eval         {limit?}  golden-set hit@3 / hit@10 / MRR, 3 modes
  GET   /api/documents   paginated corpus browse (?q= substring search)
  GET   /  /ask  /sources  /eval  /about   product pages   GET /docs  OpenAPI
"""
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from . import config, corpus as corpus_mod, evaluator, llm as llm_mod, store
from .retrieval import HybridRetriever, lexical_rerank

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("ragmill")

_state: dict = {"retriever": None, "ingest_stats": None, "query_count": 0,
                "bootstrap": None}


def get_retriever() -> HybridRetriever:
    if _state["retriever"] is None:
        t0 = time.time()
        # cloud bootstrap: a fresh instance (empty live store) materializes
        # the bundled pre-ingested snapshot before opening the client
        restored = store.restore_snapshot()
        r = HybridRetriever()
        if not r.ready:
            # service restart with a persisted collection: rebuild BM25 +
            # corpus map from Qdrant payloads (no re-embedding)
            n = r.load_from_store()
            if n:
                log.info("loaded %d docs from persisted qdrant in %.2fs",
                         n, time.time() - t0)
        _state["retriever"] = r
        if len(r.records):
            _state["ingest_stats"] = {"n_in": len(r.records), "restored": True}
        if restored or len(r.records):
            _state["bootstrap"] = {
                "snapshot_restored": restored,
                "boot_s": round(time.time() - t0, 2),
                "corpus_docs": len(r.records),
            }
        log.info("retriever init %.2fs (embedder=%s, docs=%d)",
                 time.time() - t0, r.embedder.name, len(r.records))
    return _state["retriever"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Eager boot: restore the bundled index snapshot + rebuild BM25 at
    startup (not lazily on the first request) so a cold Cloud Run instance
    is query-ready within seconds of serving."""
    t0 = time.time()
    r = get_retriever()
    log.info("startup complete in %.1fs — corpus_docs=%d ready=%s",
             time.time() - t0, len(r.records), r.ready)
    yield


app = FastAPI(title="ragmill", version="1.1.0",
              description="RAG at 1M documents — hybrid retrieval, RRF fusion, "
                          "cited answers. Local mode runs keyless.",
              lifespan=lifespan)


# ---------------------------------------------------------------- models

class IngestReq(BaseModel):
    source: str = Field(default="corpus", description="corpus | folder")
    path: str | None = None
    arxiv_ids: list[str] | None = None
    limit: int | None = None


class SearchReq(BaseModel):
    query: str
    k: int = 8


class QueryReq(BaseModel):
    query: str
    k_retrieve: int = Field(default=config.TOP_K_RETRIEVE)
    k_final: int = Field(default=config.TOP_K_FINAL)


class EvalReq(BaseModel):
    limit: int | None = None  # subsample golden set for a quick run


# ---------------------------------------------------------------- health

@app.get("/api/healthz")
def healthz():
    r = get_retriever()
    return {
        "status": "ok",
        "profile": config.PROFILE,
        "embedder": r.embedder.name,
        "dim": r.embedder.dim,
        "llm": llm_mod.get_llm().name,
        "corpus_docs": len(r.records),
        "qdrant_count": (r.ready and _qdrant_count(r)) or 0,
        "ingest": _state["ingest_stats"],
        "bootstrap": _state["bootstrap"],
        "queries_served": _state["query_count"],
    }


def _qdrant_count(r) -> int:
    from . import store
    try:
        return store.count(r.client)
    except Exception:
        return 0


# ---------------------------------------------------------------- ingest

@app.post("/api/ingest")
def ingest(req: IngestReq):
    """Local-mode ingestion: chunk -> embed -> idempotent Qdrant upsert +
    BM25 build. The same pipeline (app.corpus + app.store) is what the
    sharded Cloud Run job (jobs/ingest_job.py) runs at 1M scale."""
    if req.source == "existing":
        """Restore the in-process BM25/corpus map from the persisted Qdrant
        collection without re-embedding (fast boot after a restart)."""
        n = get_retriever().load_from_store()
        _state["ingest_stats"] = {"n_in": n, "restored": True}
        return _state["ingest_stats"]
    if req.source == "folder":
        if not req.path:
            raise HTTPException(422, "source=folder requires path")
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(404, f"no corpus at {path}")
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        if not files:
            raise HTTPException(404, f"no .jsonl corpus files at {path}")
        docs = []
        for f in files:
            docs.extend(_read_jsonl(f))
    else:  # bundled corpus
        if not config.CORPUS_PATH.exists():
            raise HTTPException(404, f"corpus not found at "
                                     f"{config.CORPUS_PATH} — generate it: "
                                     f"python3 scripts/gen_corpus.py")
        docs = _read_jsonl(config.CORPUS_PATH)

    if req.arxiv_ids:
        want = set(req.arxiv_ids)
        docs = [d for d in docs if d.get("arxiv_id") in want]
        if not docs:
            raise HTTPException(404, "none of the arxiv_ids matched")
    if req.limit:
        docs = docs[:req.limit]
    if len(docs) > config.MAX_INGEST_DOCS:
        raise HTTPException(413, f"limit {len(docs)} > "
                                 f"RAGMILL_MAX_INGEST ({config.MAX_INGEST_DOCS}); "
                                 f"use the sharded cloud job for scale")
    stats = get_retriever().ingest(docs)
    _state["ingest_stats"] = stats
    log.info("ingest %s", stats)
    return stats


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- search

@app.post("/api/search")
def search(req: SearchReq):
    r = get_retriever()
    if not r.ready:
        raise HTTPException(409, "corpus empty — POST /api/ingest first")
    t0 = time.time()
    res = r.hybrid_search(req.query, req.k)
    return {
        "query": req.query,
        "fused": [_brief(c) for c in res["fused"]],
        "dense_top3": [c["arxiv_id"] for c in res["dense"][:3]],
        "sparse_top3": [c["arxiv_id"] for c in res["sparse"][:3]],
        "took_ms": round((time.time() - t0) * 1000, 1),
    }


def _brief(c: dict) -> dict:
    return {"arxiv_id": c["arxiv_id"], "title": c["title"],
            "categories": c.get("categories") or [],
            "rrf_score": c.get("rrf_score"), "score": c.get("score")}


# ---------------------------------------------------------------- query (SSE)

def _sse_error(code: str, message: str) -> EventSourceResponse:
    """A terminal `error` SSE event. The ask stream must ALWAYS terminate
    with a terminal event (answer+done, or error) — the frontend renders
    error events as a red box instead of spinning forever."""
    async def one():
        yield {"event": "error",
               "data": json.dumps({"code": code, "message": message})}
    return EventSourceResponse(one())


@app.post("/api/query")
@app.post("/api/ask")
async def query(req: QueryReq):
    r = get_retriever()
    if not r.ready:
        return _sse_error(
            "corpus_empty",
            "corpus empty — ingest first (POST /api/ingest, or wait for "
            "the boot snapshot to finish restoring)")
    _state["query_count"] += 1
    llm = llm_mod.get_llm()

    async def gen():
        timings = {}
        try:
            # -- stage 1: retrieve --------------------------------------------
            t0 = time.time()
            res = r.hybrid_search(req.query, k=req.k_final,
                                  fetch=req.k_retrieve)
            timings["retrieve_ms"] = round((time.time() - t0) * 1000, 1)
            if not res["fused"]:
                yield {"event": "error", "data": json.dumps({
                    "code": "no_results",
                    "message": "retrieval returned 0 documents for this "
                               "query — try different wording"})}
                return
            yield {"event": "stage", "data": json.dumps({
                "stage": "retrieve",
                "dense_top3": [c["arxiv_id"] for c in res["dense"][:3]],
                "sparse_top3": [c["arxiv_id"] for c in res["sparse"][:3]],
                "fused_top": [_brief(c) for c in res["fused"][:req.k_final]],
                "took_ms": timings["retrieve_ms"],
            })}
            # -- stage 2: rerank ------------------------------------------------
            t1 = time.time()
            try:
                reranked = llm.rerank(req.query, res["fused"], k=req.k_final)
                reranker = llm.name
            except Exception as e:  # noqa: BLE001 — degrade, never kill the stream
                log.warning("rerank failed (%s); lexical fallback", str(e)[:100])
                reranked = lexical_rerank(req.query, res["fused"], req.k_final)
                reranker = "lexical-fallback"
            timings["rerank_ms"] = round((time.time() - t1) * 1000, 1)
            yield {"event": "stage", "data": json.dumps({
                "stage": "rerank",
                "reranker": reranker,
                "top": [_brief(c) | {"rerank_score": c.get("rerank_score")}
                        for c in reranked],
                "took_ms": timings["rerank_ms"],
            })}
            # -- stage 3: answer ------------------------------------------------
            t2 = time.time()
            try:
                ans = llm.answer(req.query, reranked)
            except Exception as e:  # noqa: BLE001
                log.warning("answer failed: %s", str(e)[:200])
                ans = {"answer": f"LLM error ({str(e)[:120]}). Retrieved papers "
                                 "are listed in the rerank stage.",
                       "cited": False, "llm": "error"}
            timings["answer_ms"] = round((time.time() - t2) * 1000, 1)
            citations = [{"n": i, "arxiv_id": c["arxiv_id"], "title": c["title"],
                          "url": f"https://arxiv.org/abs/{c['arxiv_id']}"}
                         for i, c in enumerate(reranked[:req.k_final], 1)]
            yield {"event": "stage", "data": json.dumps({
                "stage": "answer",
                "answer": ans["answer"],
                "cited": ans.get("cited", False),
                "llm": ans.get("llm", "?"),
                "citations": citations,
                "took_ms": timings["answer_ms"],
            })}
            yield {"event": "done", "data": json.dumps({"timings": timings})}
        except Exception as e:  # noqa: BLE001 — the stream must always terminate
            log.exception("query pipeline failed")
            yield {"event": "error", "data": json.dumps({
                "code": "internal",
                "message": f"query pipeline failed: {str(e)[:180]}"})}

    return EventSourceResponse(gen())


# ---------------------------------------------------------------- eval

EVAL_RESULTS_PATH = config.APP_ROOT / "eval" / "results_local_5k.json"


@app.post("/api/eval")
def run_eval(req: EvalReq):
    r = get_retriever()
    if not r.ready:
        raise HTTPException(409, "corpus empty — POST /api/ingest first")
    if not config.GOLDEN_PATH.exists():
        raise HTTPException(404, f"golden set not found at "
                                 f"{config.GOLDEN_PATH} — python3 "
                                 f"scripts/gen_corpus.py regenerates it")
    golden = evaluator.load_golden(config.GOLDEN_PATH)
    if req.limit:
        golden = golden[:req.limit]
    t0 = time.time()
    scores = evaluator.evaluate(r, golden)
    return {"n_queries": len(golden),
            "took_s": round(time.time() - t0, 2),
            "metrics": scores}


@app.get("/api/eval/results")
def eval_results():
    """The measured eval JSON referenced by the landing page."""
    if not EVAL_RESULTS_PATH.exists():
        raise HTTPException(404, "eval results not generated yet")
    return FileResponse(EVAL_RESULTS_PATH, media_type="application/json")


# ---------------------------------------------------------------- documents

DOC_SNIPPET_CHARS = 200   # browse-list snippet length (first ~200 chars)
DOC_BROWSE_PER_PAGE = 20  # default page size when browsing
DOC_SEARCH_PER_PAGE = 25  # default page size when searching (?q=)


def _corpus_documents(r) -> list[dict]:
    """The loaded corpus as a stable document list: deduped by arxiv_id
    (first chunk wins) and sorted by arxiv_id so pagination is stable."""
    seen: dict[str, dict] = {}
    for rec in r.records.values():
        aid = str(rec.get("arxiv_id") or "")
        if aid and aid not in seen:
            seen[aid] = rec
    return [seen[aid] for aid in sorted(seen)]


def _snippet(text: str, n: int = DOC_SNIPPET_CHARS) -> str:
    text = (text or "").strip()
    return text[:n] + ("…" if len(text) > n else "")


@app.get("/api/documents")
def documents(
    q: str | None = Query(
        default=None, min_length=1, max_length=200,
        description="case-insensitive substring over title + abstract"),
    page: int = Query(default=1, ge=1, le=100_000),
    per_page: int | None = Query(default=None, ge=1, le=100),
):
    """Browse / search the stored corpus payloads — a plain data endpoint
    for the Sources page (no retrieval, no LLM)."""
    needle = (q or "").strip().lower() or None
    size = per_page or (DOC_SEARCH_PER_PAGE if needle else DOC_BROWSE_PER_PAGE)
    docs = _corpus_documents(get_retriever())
    if needle:
        docs = [d for d in docs if needle in
                f"{d.get('title', '')} {d.get('abstract', '')}".lower()]
    total = len(docs)
    start = (page - 1) * size
    return {
        "q": needle,
        "total": total,
        "page": page,
        "per_page": size,
        "pages": max(1, math.ceil(total / size)),
        "documents": [{
            "arxiv_id": d["arxiv_id"],
            "title": d.get("title", ""),
            "link": f"https://arxiv.org/abs/{d['arxiv_id']}",
            "categories": d.get("categories") or [],
            "snippet": _snippet(d.get("abstract", "")),
            "abstract": (d.get("abstract") or "").strip(),
        } for d in docs[start:start + size]],
    }


# ---------------------------------------------------------------- ui

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _page(name: str) -> FileResponse:
    return FileResponse(STATIC_DIR / name)


@app.get("/", include_in_schema=False)
def index():
    return _page("index.html")


@app.get("/ask", include_in_schema=False)
def ask_page():
    return _page("ask.html")


@app.get("/sources", include_in_schema=False)
def sources_page():
    return _page("sources.html")


@app.get("/eval", include_in_schema=False)
def eval_page():
    return _page("eval.html")


@app.get("/about", include_in_schema=False)
def about_page():
    return _page("about.html")


@app.get("/api/ingest/status")
def ingest_status():
    """Local mode: report the last in-process ingest stats. Cloud mode
    (documented in scripts/launch_1m_ingest.sh) reads per-task status JSON
    objects from GCS instead."""
    return {"profile": config.PROFILE, "last_ingest": _state["ingest_stats"]}
