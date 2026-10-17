"""FastAPI service. Logs contain only endpoint, status, sizes and timings, never text."""

from __future__ import annotations

import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import ClassifyOut, ExtractOut, GapIn, MarketOut, SkillOut, TextIn
from api.settings import settings
from jobnlp.config import ROOT
from jobnlp.manifest import ManifestError, verify_manifest

log = logging.getLogger("api")
REQUESTS = Counter("jobnlp_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("jobnlp_request_seconds", "Latency", ["endpoint"])
MARKET = ROOT / "results" / "market"
GROUPS = {
    "overall",
    "state",
    "region",
    "category",
    "language",
    "experience",
    "remote",
    "data_role",
    "week",
}
_pool = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="jobnlp", version="0.1.0", docs_url="/docs", redoc_url=None)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status, self.code, self.message = status, code, message


@app.exception_handler(ApiError)
async def api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, exc.status)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = sorted({".".join(str(x) for x in e.get("loc", [])) for e in exc.errors()})
    return JSONResponse(
        {"error": {"code": "invalid_input", "message": f"invalid fields: {', '.join(fields)}"}},
        422,
    )


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": "http_error", "message": str(exc.detail)}}, exc.status_code
    )


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error type=%s", type(exc).__name__)  # no message: it may hold text
    return JSONResponse({"error": {"code": "internal", "message": "internal error"}}, 500)


@app.middleware("http")
async def metrics_mw(request: Request, call_next: Any) -> Response:
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    path = request.url.path
    REQUESTS.labels(path, str(response.status_code)).inc()
    LATENCY.labels(path).observe(time.perf_counter() - t0)
    log.info(
        "path=%s status=%s ms=%.0f", path, response.status_code, (time.perf_counter() - t0) * 1000
    )
    return response


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise ApiError(401, "unauthorised", "missing or wrong X-API-Key")


@lru_cache(maxsize=1)
def extractor() -> Any:
    from jobnlp.skills.runtime import get_extractor

    return get_extractor()


@lru_cache(maxsize=1)
def classifier() -> tuple[Any, str, list[str]]:
    """Fine-tuned BERT when present with a valid manifest, otherwise TF-IDF; refuses models
    whose manifest is missing or inconsistent."""
    bert_dir = ROOT / "models" / "classifier_bert" / "model"
    tfidf_dir = ROOT / "models" / "classifier_tfidf"
    for d, kind in ((bert_dir, "bert"), (tfidf_dir, "tfidf_lr")):
        if not d.exists():
            continue
        m = verify_manifest(d, "classifier")
        if m.get("model") != kind:
            raise ManifestError(f"{d.name}: manifest model {m.get('model')} != {kind}")
        if kind == "bert":
            return d, kind, list(m["classes"])
        with (d / "model.pkl").open("rb") as f:
            return pickle.load(f), kind, list(m["classes"])
    raise ManifestError("no trained classifier found")


def _lang(text: str, lang: str | None) -> str:
    if lang:
        return lang
    from jobnlp.text.lang import detect

    return "en" if detect(text) == "en" else "de"


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/ready")
def ready() -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        extractor()
        checks["extractor"] = "ok"
    except Exception as exc:
        checks["extractor"] = f"unavailable: {type(exc).__name__}"
    try:
        checks["classifier"] = f"ok ({classifier()[1]})"
    except Exception as exc:
        checks["classifier"] = (
            f"unavailable: {exc}"
            if isinstance(exc, ManifestError)
            else f"unavailable: {type(exc).__name__}"
        )
    checks["market"] = "ok" if (MARKET / "market_summary.json").exists() else "missing"
    ok = all(v.startswith("ok") for v in checks.values())
    return JSONResponse({"ready": ok, "checks": checks}, 200 if ok else 503)


@app.post("/v1/extract", response_model=ExtractOut, dependencies=[Depends(require_key)])
def extract(body: TextIn) -> ExtractOut:
    lang = _lang(body.text, body.lang)
    mentions = extractor().extract(body.text, lang, use_ner=False)
    return ExtractOut(
        lang=lang,
        skills=[SkillOut(**{k: getattr(m, k) for k in SkillOut.model_fields}) for m in mentions],
    )


@app.post("/v1/classify", response_model=ClassifyOut, dependencies=[Depends(require_key)])
def classify(body: TextIn) -> ClassifyOut:
    try:
        model, kind, classes = classifier()
    except ManifestError as exc:
        raise ApiError(503, "model_unavailable", str(exc)) from exc
    if kind == "bert":
        from jobnlp.classify.bert import predict_proba

        proba, order, _ = predict_proba(model, [body.text])
    else:
        proba, order = model.predict_proba([body.text]), model.classes
    probs = {c: round(float(p), 4) for c, p in zip(order, proba[0], strict=True)}
    return ClassifyOut(label=max(probs, key=lambda c: probs[c]), probabilities=probs, model=kind)


def _run_gap(job_text: str, cv_text: str, suggestions: bool) -> dict[str, Any]:
    from jobnlp.gap.llm import LocalLLM
    from jobnlp.gap.report import run_gap

    thr_path = ROOT / "results" / "related_threshold.json"
    thr = json.loads(thr_path.read_text("utf-8"))["threshold"] if thr_path.exists() else 0.9
    llm = (
        LocalLLM(settings.llm_model, settings.ollama_host)
        if (suggestions and settings.enable_llm)
        else None
    )
    return run_gap(job_text, cv_text, extractor(), related_threshold=float(thr), llm=llm)


@app.post("/v1/gap", dependencies=[Depends(require_key)])
def gap_json(body: GapIn) -> dict[str, Any]:
    return _run_gap(body.job_text, body.cv_text, body.suggestions)


@app.post("/v1/gap/upload", dependencies=[Depends(require_key)])
async def gap_upload(
    job_text: str = Form(..., max_length=30_000),
    cv_file: UploadFile = File(...),
    suggestions: bool = Form(True),
) -> dict[str, Any]:
    from jobnlp.gap.parse_cv import CVError, read_cv

    data = await cv_file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise ApiError(413, "too_large", "upload larger than the limit")
    try:
        cv_text = _pool.submit(read_cv, data, cv_file.filename or "cv.txt").result(
            timeout=settings.parse_timeout_s
        )
    except CVError as exc:
        raise ApiError(415, "unsupported_file", str(exc)) from exc
    except FutureTimeout as exc:
        raise ApiError(408, "parse_timeout", "the file took too long to parse") from exc
    return _run_gap(job_text, cv_text, suggestions)


@app.get("/v1/market/skills", response_model=MarketOut, dependencies=[Depends(require_key)])
def market_skills(
    group_by: str = "overall", group: str | None = None, min_count: int = 5, top: int = 30
) -> MarketOut:
    if group_by not in GROUPS:
        raise ApiError(400, "bad_group", f"group_by must be one of {sorted(GROUPS)}")
    path = MARKET / f"skill_shares_by_{group_by}.csv"
    if not path.exists():
        raise ApiError(503, "not_ready", "run `jobnlp market` first")
    df = pd.read_csv(path)
    if group is not None:
        df = df[df.group.astype(str) == group]
    df = df[df.postings_with_skill >= min_count].groupby("group").head(max(1, min(top, 100)))
    summary = json.loads((MARKET / "market_summary.json").read_text("utf-8"))
    return MarketOut(
        group_by=group_by,
        rows=[{str(k): v for k, v in r.items()} for r in df.to_dict("records")],
        suppressed_groups=summary["suppressed_groups"].get(group_by, []),
        min_group_size=summary["min_group_size"],
        note=(
            "Shares are postings mentioning the skill, with 95% Wilson intervals; "
            f"groups under {summary['min_group_size']} postings are suppressed. Postings "
            "are a proxy for demand, from the sources and window in the summary."
        ),
    )


@app.get("/v1/market/profile", dependencies=[Depends(require_key)])
def market_profile() -> dict[str, Any]:
    path = MARKET / "market_summary.json"
    if not path.exists():
        raise ApiError(503, "not_ready", "run `jobnlp market` first")
    s = json.loads(path.read_text("utf-8"))
    return {
        k: s.get(k)
        for k in (
            "postings",
            "published_min",
            "published_max",
            "role_profiles",
            "data_roles",
            "regions",
            "employer_concentration",
        )
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
