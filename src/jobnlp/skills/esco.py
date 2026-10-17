"""ESCO skills via the public ESCO web service API (no key). Raw pages are cached by the polite
client; the result is a flat table of skills with German and English labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jobnlp.harvest.client import PoliteClient
from jobnlp.logging_utils import get_logger

API = "https://ec.europa.eu/esco/api/search"
PAGE = 100
log = get_logger("skills.esco", "esco.log")


def _labels(field: Any, lang: str) -> list[str]:
    if not isinstance(field, dict):
        return []
    v = field.get(lang)
    if v is None:
        return []
    return [v] if isinstance(v, str) else [str(x) for x in v]


def _link_uris(rec: dict[str, Any], key: str) -> list[str]:
    return [str(x.get("uri")) for x in (rec.get("_links") or {}).get(key) or []]


def _link_titles(rec: dict[str, Any], key: str) -> list[str]:
    return [str(x.get("title")) for x in (rec.get("_links") or {}).get(key) or []]


def flatten(rec: dict[str, Any]) -> dict[str, Any]:
    links = rec.get("_links") or {}
    return {
        "uri": rec["uri"],
        "pref_de": next(iter(_labels(rec.get("preferredLabel"), "de")), None),
        "pref_en": next(iter(_labels(rec.get("preferredLabel"), "en")), None),
        "alt_de": _labels(rec.get("alternativeLabel"), "de"),
        "alt_en": _labels(rec.get("alternativeLabel"), "en"),
        "skill_type": (_link_uris(rec, "hasSkillType") or [""])[0].rsplit("/", 1)[-1],
        "reuse_level": (_link_uris(rec, "hasReuseLevel") or [""])[0].rsplit("/", 1)[-1],
        "broader_codes": [
            str(x.get("code") or "") for x in links.get("broaderHierarchyConcept") or []
        ],
        "broader_titles": _link_titles(rec, "broaderHierarchyConcept"),
        "schemes": _link_uris(rec, "isInScheme"),
        "scheme_titles": _link_titles(rec, "isInScheme"),
    }


def fetch_skills(client: PoliteClient, raw_dir: Path) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    # The API's `offset` parameter is a page index, not a row offset, and some pages return
    # HTTP 500 for full records (observed 2026-10-17). Failing ranges are re-read in smaller
    # pages; single records that still fail are skipped and counted.
    failed: list[int] = []

    def read(start: int, size: int) -> tuple[int, list[dict[str, Any]]]:
        url = f"{API}?type=skill&language=de&limit={size}&offset={start // size}&full=true"
        try:
            r = client.get(url, namespace="esco")
            status = r.status
        except RuntimeError:
            status = 500
        if status == 200:
            data = json.loads(r.text)
            return int(data["total"]), list((data.get("_embedded") or {}).get("results") or [])
        if size == 1:
            failed.append(start)
            return -1, []
        smaller = 10 if size > 10 else 1
        out: list[dict[str, Any]] = []
        total = -1
        for s in range(start, start + size, smaller):
            t, rows = read(s, smaller)
            total = max(total, t)
            out.extend(rows)
        return total, out

    start, total = 0, None
    recs: list[dict[str, Any]] = []
    while total is None or start < total:
        t, results = read(start, PAGE)
        total = t if t > 0 else total
        if not results and t > 0:
            break
        recs.extend(flatten(x) for x in results)
        start += PAGE
        log.info("esco %d of %s skills, failed records %d", len(recs), total, len(failed))
    log.info("esco skipped records at positions %s", failed)
    df = pd.DataFrame(recs).drop_duplicates("uri")
    df["is_digital"] = df["scheme_titles"].apply(
        lambda ts: any("digital" in t.lower() or "ikt" in t.lower() for t in ts)
    )
    return df
