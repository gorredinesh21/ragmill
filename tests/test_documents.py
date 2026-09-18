"""GET /api/documents (browse + substring search) and the multi-page UI routes."""
import math

PAGES = [("/", "RAGMill — answers with numbered sources"),
         ("/ask", "Ask — RAGMill"),
         ("/sources", "Sources — RAGMill"),
         ("/eval", "Eval — RAGMill"),
         ("/about", "About — RAGMill")]


# ------------------------------------------------------------- browse

def test_documents_browse(client):
    r = client.get("/api/documents")
    assert r.status_code == 200
    b = r.json()
    assert b["q"] is None
    assert b["total"] > 0
    assert b["page"] == 1 and b["per_page"] == 20 and b["pages"] >= 1
    assert 0 < len(b["documents"]) <= 20
    d = b["documents"][0]
    assert d["arxiv_id"] and d["title"]
    assert d["link"] == f"https://arxiv.org/abs/{d['arxiv_id']}"
    assert isinstance(d["categories"], list)
    assert d["snippet"] and d["abstract"]
    assert len(d["snippet"]) <= 201  # 200 chars + optional ellipsis


def test_documents_sorted_and_paginated(client):
    p1 = client.get("/api/documents",
                    params={"per_page": 5, "page": 1}).json()
    p2 = client.get("/api/documents",
                    params={"per_page": 5, "page": 2}).json()
    ids1 = [d["arxiv_id"] for d in p1["documents"]]
    ids2 = [d["arxiv_id"] for d in p2["documents"]]
    assert len(ids1) == 5 and len(ids2) == 5
    assert not set(ids1) & set(ids2)          # pages do not overlap
    assert ids1 + ids2 == sorted(ids1 + ids2)  # stable arxiv_id ordering
    assert p1["pages"] == math.ceil(p1["total"] / 5)


def test_documents_out_of_range_page_is_empty(client):
    b = client.get("/api/documents",
                   params={"page": 100_000}).json()
    assert b["documents"] == []


# ------------------------------------------------------------- search

def test_documents_search(client):
    b = client.get("/api/documents", params={"q": "retrieval"}).json()
    assert b["q"] == "retrieval"
    assert b["total"] > 0
    assert len(b["documents"]) <= 25          # search default page size
    assert b["per_page"] == 25
    for d in b["documents"]:                  # every hit really matches
        assert "retrieval" in (d["title"] + " " + d["abstract"]).lower()


def test_documents_search_case_insensitive(client):
    up = client.get("/api/documents", params={"q": "RETRIEVAL"}).json()
    lo = client.get("/api/documents", params={"q": "retrieval"}).json()
    assert up["total"] == lo["total"] > 0


def test_documents_search_no_match(client):
    b = client.get("/api/documents",
                   params={"q": "zzzz-no-such-term-qqq"}).json()
    assert b["total"] == 0 and b["documents"] == []


# ------------------------------------------------------------- validation

def test_documents_validation_rejects_bad_params(client):
    assert client.get("/api/documents", params={"per_page": 0}).status_code == 422
    assert client.get("/api/documents", params={"per_page": 101}).status_code == 422
    assert client.get("/api/documents", params={"page": 0}).status_code == 422
    assert client.get("/api/documents", params={"q": ""}).status_code == 422


# ------------------------------------------------------------- ui pages

def test_page_routes_serve_distinct_html(client):
    bodies = []
    for path, title in PAGES:
        r = client.get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers["content-type"]
        body = r.text
        assert f"<title>{title}</title>" in body, path  # right page per route
        # shared layout: sidebar nav + shared stylesheet on every page
        assert 'class="sidebar"' in body
        assert "/static/assets/style.css" in body
        assert 'href="/ask"' in body and 'href="/sources"' in body
        bodies.append(body)
    assert len(set(bodies)) == len(PAGES)     # all pages distinct


def test_static_assets_served(client):
    css = client.get("/static/assets/style.css")
    assert css.status_code == 200 and "sidebar" in css.text
    for js in ("app.js", "ask.js", "sources.js", "eval.js"):
        r = client.get(f"/static/assets/{js}")
        assert r.status_code == 200, js
