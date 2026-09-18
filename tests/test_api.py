"""API tests with the mock LLM — keyless end-to-end including the SSE
query stream with citations."""
import json


def test_healthz(client):
    r = client.get("/api/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm"].startswith("mock")
    assert body["embedder"]
    assert body["corpus_docs"] > 0


def test_search_endpoint(client):
    r = client.post("/api/search", json={"query": "reciprocal rank fusion", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert 0 < len(body["fused"]) <= 5
    assert body["fused"][0]["arxiv_id"]
    assert "took_ms" in body and "dense_top3" in body and "sparse_top3" in body


def test_ingest_endpoint_folder_and_limit(client, tmp_path):
    # tiny extra slice ingested through the same API the demo uses
    docs = [
        {"arxiv_id": f"9999.{i:05d}", "title": f"Regression test paper {i}",
         "abstract": ("regression plumbing test abstract " * 8)[:400],
         "categories": ["cs.LG"]}
        for i in range(5)
    ]
    p = tmp_path / "extra.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in docs))
    r = client.post("/api/ingest",
                    json={"source": "folder", "path": str(p), "limit": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["n_in"] == 3
    assert body["count"] >= 3


def test_ingest_endpoint_bad_path(client):
    r = client.post("/api/ingest",
                    json={"source": "folder", "path": "/nope/missing.jsonl"})
    assert r.status_code == 404


def test_query_sse_full_pipeline_with_citations(client):
    events = []  # (sse_event_name, parsed_data)
    with client.stream("POST", "/api/query",
                       json={"query": "hybrid BM25 dense fusion",
                             "k_retrieve": 20, "k_final": 5}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        stage, data = None, None
        for line in r.iter_lines():
            if line.startswith("event:"):
                stage = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and stage:
                events.append((stage, json.loads(line.split(":", 1)[1].strip())))
    # stage events carry the pipeline stage name inside the payload
    stages = [d.get("stage") for ev, d in events if ev == "stage"]
    assert stages == ["retrieve", "rerank", "answer"]
    assert any(ev == "done" for ev, _ in events)
    by_stage = {d["stage"]: d for ev, d in events if ev == "stage"}
    # retrieve
    assert len(by_stage["retrieve"]["fused_top"]) <= 5
    # rerank
    assert by_stage["rerank"]["reranker"].startswith("mock") or \
        by_stage["rerank"]["reranker"] == "lexical-fallback"
    assert by_stage["rerank"]["top"]
    # answer — mock LLM always cites [1]/[2] and we attach citation metadata
    ans = by_stage["answer"]
    assert "[1]" in ans["answer"], ans["answer"]
    assert ans["cited"] is True
    cites = ans["citations"]
    assert cites and cites[0]["url"].startswith("https://arxiv.org/abs/")
    assert cites[0]["arxiv_id"]
    # done (its own SSE event name, payload has stage timings)
    done = next(d for ev, d in events if ev == "done")
    assert done["timings"]["retrieve_ms"] >= 0


def test_eval_endpoint(client):
    import app.config as config
    from pathlib import Path
    # point the endpoint at a mini golden derived from the fixture corpus
    golden = _mini_golden_for_api()
    p = Path(config.GOLDEN_PATH)
    backup = p.read_text() if p.exists() else None
    tmp_golden = p.with_suffix(".api-test.jsonl")
    tmp_golden.write_text("\n".join(json.dumps(g) for g in golden))
    old = config.GOLDEN_PATH
    config.GOLDEN_PATH = tmp_golden
    try:
        r = client.post("/api/eval", json={"limit": 20})
        assert r.status_code == 200
        body = r.json()
        assert body["n_queries"] == 20
        for mode in ("dense", "sparse", "hybrid"):
            for metric in ("hit@3", "hit@10", "mrr"):
                assert metric in body["metrics"][mode]
    finally:
        config.GOLDEN_PATH = old
        tmp_golden.unlink(missing_ok=True)
        if backup is not None:
            p.write_text(backup)


def _mini_golden_for_api():
    from scripts.gen_corpus import gen_docs, gen_golden
    docs = [d for d in gen_docs(900, seed=99) if any(
        t in d["title"].lower() + d["abstract"].lower()
        for t in ("retrieval-augmented", "vector databases", "federated learning"))][:150]
    return gen_golden(docs, n=25, seed=3)


def test_openapi_served(client):
    r = client.get("/docs")
    assert r.status_code == 200
    r = client.get("/openapi.json")
    assert "/api/query" in r.json()["paths"]
    assert "/api/eval" in r.json()["paths"]


def test_eval_results_route(client):
    r = client.get("/api/eval/results")
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["hybrid"]["hit@10"] > body["metrics"]["dense"]["hit@10"]


# -- empty-corpus ask stream MUST terminate with an error event ---------------

def _stream_events(cl, path, payload):
    events = []
    with cl.stream("POST", path, json=payload) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        stage = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                stage = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and stage:
                events.append((stage, json.loads(line.split(":", 1)[1].strip())))
    return events


class _EmptyCorpusStub:
    """Not ready, no records — what a fresh instance looks like with no
    snapshot and no ingest."""
    ready = False
    records: dict = {}


def test_query_sse_empty_corpus_emits_terminal_error(client, monkeypatch):
    from app import main as app_main
    monkeypatch.setattr(app_main, "get_retriever", lambda: _EmptyCorpusStub())
    events = _stream_events(client, "/api/query", {"query": "anything"})
    # a terminal `error` event — never a silent, endless stream
    errs = [d for ev, d in events if ev == "error"]
    assert len(errs) == 1
    assert errs[0]["code"] == "corpus_empty"
    assert "corpus empty" in errs[0]["message"]
    # and no pipeline stages ran
    assert not [1 for ev, _ in events if ev == "stage"]
    assert not [1 for ev, _ in events if ev == "done"]


def test_ask_alias_empty_corpus_same_terminal_error(client, monkeypatch):
    from app import main as app_main
    monkeypatch.setattr(app_main, "get_retriever", lambda: _EmptyCorpusStub())
    events = _stream_events(client, "/api/ask", {"query": "anything"})
    errs = [d for ev, d in events if ev == "error"]
    assert len(errs) == 1
    assert errs[0]["code"] == "corpus_empty"
