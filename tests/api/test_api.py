"""API tests. Endpoints that need data or models use real cached postings and skip otherwise."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main
from api.settings import settings
from jobnlp.config import ROOT

client = TestClient(main.app)
CORPUS = ROOT / "data" / "processed" / "postings.parquet"


@pytest.fixture(scope="module")
def real_postings() -> list[str]:
    if not CORPUS.exists():
        pytest.skip("processed corpus absent")
    df = pd.read_parquet(CORPUS)
    df = df[df.in_germany & ~df.is_dup & (df.n_chars > 800)].sample(3, random_state=0)
    return df.text.tolist()


def test_health() -> None:
    r = client.get("/v1/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_validation_error_schema_has_no_trace() -> None:
    r = client.post("/v1/extract", json={"text": ""})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "invalid_input"
    assert "Traceback" not in r.text


def test_oversized_text_rejected() -> None:
    r = client.post("/v1/extract", json={"text": "x" * 30_001})
    assert r.status_code == 422


def test_api_key_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "secret")
    r = client.post("/v1/extract", json={"text": "Python"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorised"


def test_upload_rejects_unsupported_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_llm", False)
    r = client.post(
        "/v1/gap/upload", data={"job_text": "Python"}, files={"cv_file": ("cv.exe", b"MZ binary")}
    )
    assert r.status_code == 415


def test_upload_rejects_large_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    r = client.post(
        "/v1/gap/upload", data={"job_text": "Python"}, files={"cv_file": ("cv.txt", b"x" * 100)}
    )
    assert r.status_code == 413


def test_metrics_endpoint() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200 and "jobnlp_requests_total" in r.text


def test_extract_on_real_posting(real_postings: list[str]) -> None:
    r = client.post("/v1/extract", json={"text": real_postings[0]})
    assert r.status_code == 200
    skills = r.json()["skills"]
    assert all(real_postings[0][s["start"] : s["end"]] == s["text"] for s in skills)


def test_gap_on_real_postings_without_llm(
    real_postings: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "enable_llm", False)
    r = client.post(
        "/v1/gap",
        json={"job_text": real_postings[0], "cv_text": real_postings[0], "suggestions": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["score"] is None or body["score"] >= 0.99  # a posting covers its own skills
    assert "heuristic" in body["disclaimer"]


def test_classify_real_posting(real_postings: list[str]) -> None:
    if not (ROOT / "models" / "classifier_tfidf").exists():
        pytest.skip("no trained classifier")
    r = client.post("/v1/classify", json={"text": real_postings[1]})
    assert r.status_code == 200
    probs = r.json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_market_endpoints() -> None:
    if not (ROOT / "results" / "market" / "market_summary.json").exists():
        pytest.skip("market results absent")
    r = client.get("/v1/market/skills", params={"group_by": "region"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert all(row["ci_low"] <= row["share"] <= row["ci_high"] for row in rows)
    assert client.get("/v1/market/skills", params={"group_by": "nope"}).status_code == 400
    assert client.get("/v1/market/profile").status_code == 200
