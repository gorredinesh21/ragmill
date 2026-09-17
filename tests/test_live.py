"""Live smoke test against deployed infra. Skipped unless RAGMILL_LIVE_URL
is set (CI runs it in the smoke job; local default skips)."""
import os

import pytest
import requests

LIVE = os.environ.get("RAGMILL_LIVE_URL", "").rstrip("/")

pytestmark = pytest.mark.live
requires_live = pytest.mark.skipif(
    not LIVE, reason="RAGMILL_LIVE_URL not set")


@requires_live
def test_live_healthz():
    r = requests.get(f"{LIVE}/api/healthz", timeout=30)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@requires_live
def test_live_search():
    r = requests.post(f"{LIVE}/api/search",
                      json={"query": "reciprocal rank fusion", "k": 5},
                      timeout=60)
    assert r.status_code == 200
    assert r.json()["fused"]
