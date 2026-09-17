"""LLM layer: Gemini (Vertex) rerank + cited answers, with a deterministic
MockLLM so the entire API works keyless offline (RAGMILL_LLM=mock).

VertexLLM reuses the proven support-copilot / quackquery auth pattern:
metadata-server token on GCP, `gcloud auth print-access-token` locally,
429/5xx backoff honoring Retry-After. Plain REST via requests, no SDK weight.
"""
import logging
import re
import subprocess
import time

import requests

from . import config
from .retrieval import lexical_rerank

log = logging.getLogger("ragmill.llm")


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Vertex auth (metadata server -> gcloud fallback), cached token
# ---------------------------------------------------------------------------

def _fetch_token() -> str:
    try:
        r = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/"
            "service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"}, timeout=3)
        if r.status_code == 200:
            return r.json()["access_token"]
    except requests.RequestException:
        pass
    out = subprocess.run(["gcloud", "auth", "print-access-token"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise LLMError(f"gcloud auth failed: {out.stderr[:200]}")
    return out.stdout.strip()


_token_cache = {"tok": None, "exp": 0.0}


def _auth_header() -> dict:
    now = time.time()
    if _token_cache["tok"] is None or now > _token_cache["exp"]:
        _token_cache["tok"] = _fetch_token()
        _token_cache["exp"] = now + 45 * 60
    return {"Authorization": f"Bearer {_token_cache['tok']}"}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RERANK_PROMPT = """You are reranking arXiv papers by relevance to a query.

Query: {query}

Candidates (id | title | abstract-snippet):
{candidates}

Reply with ONLY a JSON array of the candidate numbers, most relevant first,
top {k}. Example: [3, 1, 4]. No prose."""

ANSWER_PROMPT = """You are a research assistant over an arXiv corpus.
Answer the question using ONLY the numbered excerpts below.
Cite every claim with [n] markers referencing excerpt numbers.
If the excerpts do not answer the question, say exactly:
"I could not find this in the corpus." and nothing else.
Be concise (max ~130 words).

Excerpts:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# VertexLLM
# ---------------------------------------------------------------------------

class VertexLLM:
    name = "vertex:gemini-2.5-flash"

    def __init__(self):
        self.model = config.LLM_MODEL
        self.project = config.GCP_PROJECT
        self.region = config.GCP_REGION

    def _call(self, prompt: str, temperature: float = 0.1,
              max_tokens: int = 1024) -> str:
        url = (f"https://{self.region}-aiplatform.googleapis.com/v1/projects/"
               f"{self.project}/locations/{self.region}/publishers/google/"
               f"models/{self.model}:generateContent")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }
        r = None
        for attempt in range(1, config.LLM_MAX_ATTEMPTS + 1):
            r = requests.post(url, headers=_auth_header(), json=payload,
                              timeout=config.LLM_TIMEOUT_S)
            if r.status_code in (429, 500, 503):
                wait = int(r.headers.get("Retry-After", "20")) + (attempt - 1) * 5
                log.warning("LLM %s; backing off %ss (attempt %s)",
                            r.status_code, wait, attempt)
                time.sleep(wait)
                continue
            break
        if r is None or r.status_code != 200:
            raise LLMError(f"LLM API {getattr(r, 'status_code', '?')}: "
                           f"{getattr(r, 'text', '')[:200]}")
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response shape: {e}")

    # -- pipeline API -------------------------------------------------------

    def rerank(self, query: str, candidates: list[dict], k: int) -> list[dict]:
        """Listwise permutation, ONE call, capped inputs (no unbounded cost)."""
        if not candidates:
            return []
        lines = []
        for i, c in enumerate(candidates[:20], 1):
            snippet = c.get("abstract", "")[:220].replace("\n", " ")
            lines.append(f"{i}. {c['arxiv_id']} | {c['title']} | {snippet}")
        raw = self._call(RERANK_PROMPT.format(
            query=query, candidates="\n".join(lines), k=min(k, len(candidates))))
        m = re.search(r"\[.*?\]", raw, re.S)
        order: list[int] = []
        if m:
            try:
                order = [int(x) for x in re.findall(r"\d+", m.group(0))]
            except ValueError:
                order = []
        out = []
        for n in order:
            if 1 <= n <= len(candidates):
                out.append(candidates[n - 1])
        # append any candidates the model skipped (stable, lossless)
        seen = {c["point_id"] for c in out}
        out.extend(c for c in candidates if c["point_id"] not in seen)
        return out[:k]

    def answer(self, query: str, contexts: list[dict]) -> dict:
        ctx = "\n\n".join(
            f"[{i}] {c['arxiv_id']} — {c['title']}\n{c['abstract'][:600]}"
            for i, c in enumerate(contexts, 1))
        base = ANSWER_PROMPT.format(context=ctx, question=query)
        text, wait_ms = self._call(base), 0
        cited = bool(re.search(r"\[\d+\]", text))
        if contexts and not cited and "could not find" not in text.lower():
            text2 = self._call(base + "\n\nREMINDER: cite every claim with [n].")
            if re.search(r"\[\d+\]", text2):
                text, cited = text2, True
        return {"answer": text, "cited": cited, "llm": self.name}


# ---------------------------------------------------------------------------
# MockLLM — deterministic, extractive, keyless
# ---------------------------------------------------------------------------

from .embeddings import content_tokens  # noqa: E402  (kept near use)


class MockLLM:
    """Two deterministic behaviors:
    rerank -> lexical_rerank (the same keyless heuristic used by default)
    answer -> template answer that ALWAYS cites the top excerpts [1][2],
              quoting each cited paper's title. Proves the citation plumbing
              end-to-end without any API key."""
    name = "mock:extractive"

    def rerank(self, query: str, candidates: list[dict], k: int) -> list[dict]:
        return lexical_rerank(query, candidates, k)

    def answer(self, query: str, contexts: list[dict]) -> dict:
        if not contexts:
            return {"answer": "I could not find this in the corpus.",
                    "cited": False, "llm": self.name}
        tops = contexts[:2]
        lines = []
        for i, c in enumerate(tops, 1):
            lines.append(f"[{i}] {c['title']} ({c['arxiv_id']})")
        answer = (
            f"Mock-grounded answer (no LLM key configured): the most relevant "
            f"papers for '{query}' are " + " and ".join(lines) + ". "
            f"Set RAGMILL_LLM=vertex with GCP credentials for a Gemini answer.")
        return {"answer": answer, "cited": True, "llm": self.name}


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        if config.LLM_MODE == "vertex":
            _llm = VertexLLM()
        else:
            _llm = MockLLM()
    return _llm


def set_llm(llm):
    """Test hook."""
    global _llm
    _llm = llm
