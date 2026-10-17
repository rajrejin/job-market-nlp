"""Build the processed corpus: unify sources, clean, redact, detect language, split sections,
locate, and mark duplicates. Writes data/processed/postings.parquet (never committed)."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from jobnlp.text.clean import html_to_text
from jobnlp.text.dedupe import cluster_near_duplicates
from jobnlp.text.geo import GermanPlaces
from jobnlp.text.lang import detect
from jobnlp.text.redact import PersonFinder, redact
from jobnlp.text.sections import split_sections

# feed footer appended to every Arbeitnow posting
BOILERPLATE = re.compile(
    r"^Find (?:more )?(?:English Speaking )?Jobs in Germany on Arbeitnow\s*$", re.M
)


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v]
    return [str(x)]


def unify(source: str, pid: str, p: dict[str, Any], first: str, last: str) -> dict[str, Any]:
    if source == "arbeitnow":
        published = dt.datetime.fromtimestamp(int(p["created_at"]), dt.UTC).date().isoformat()
        return {
            "source": source,
            "posting_id": pid,
            "title": p.get("title") or "",
            "employer": p.get("company_name"),
            "location_raw": p.get("location") or "",
            "remote": bool(p.get("remote")),
            "categories": _as_list(p.get("tags")),
            "job_types": _as_list(p.get("job_types")),
            "published": published,
            "html": p.get("description") or "",
            "first_seen": first,
            "last_seen": last,
        }
    published_raw = str(p.get("date_posted") or "")[:10]
    return {
        "source": source,
        "posting_id": pid,
        "title": p.get("title") or "",
        "employer": p.get("employer"),
        "location_raw": p.get("city") or "",
        "remote": False,
        "categories": _as_list(p.get("occupational_category")) + _as_list(p.get("industry")),
        "job_types": _as_list(p.get("employment_type")),
        "published": published_raw or None,
        "html": _text_field(p.get("description")),
        "first_seen": first,
        "last_seen": last,
    }


def _text_field(x: Any) -> str:
    """JSON-LD descriptions are usually strings but some sites emit lists or objects."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_text_field(v) for v in x)
    return ""


def region_of(city: str | None, regions: dict[str, list[str]]) -> str | None:
    if not city:
        return None
    for name, cities in regions.items():
        if city in cities:
            return name
    return None


def build(
    rows: Iterable[tuple[str, str, dict[str, Any], str, str]],
    places: GermanPlaces,
    regions: dict[str, list[str]],
    person_finder: PersonFinder | None,
) -> pd.DataFrame:
    recs = []
    for source, pid, payload, first, last in rows:
        r = unify(source, pid, payload, first, last)
        clean = BOILERPLATE.sub("", html_to_text(r.pop("html"))).strip()
        lang = detect(clean)
        text = redact(clean, person_finder, lang)
        place = places.match(r["location_raw"])
        in_de = source == "crawl" or places.is_germany(r["location_raw"])
        secs = split_sections(text)
        r.update(
            text=text,
            n_chars=len(text),
            lang=lang,
            city=place.name if place else None,
            state=place.state if place else None,
            in_germany=in_de,
            region=region_of(place.name if place else None, regions),
            sections=json.dumps([s.__dict__ for s in secs], ensure_ascii=False),
            section_labels=sorted({s.label for s in secs}),
        )
        recs.append(r)
    df = pd.DataFrame(recs)
    df["dup_cluster"] = cluster_near_duplicates((df["title"] + "\n" + df["text"]).tolist())
    df["is_dup"] = df.groupby("dup_cluster").cumcount() > 0
    return df


def corpus_stats(df: pd.DataFrame) -> dict[str, Any]:
    de = df[df.in_germany & ~df.is_dup]
    emp = de["employer"].fillna("unknown").value_counts()
    top10_share = float(emp.head(10).sum() / max(1, len(de)))
    shares = emp / max(1, len(de))
    return {
        "n_total": int(len(df)),
        "n_duplicates": int(df.is_dup.sum()),
        "n_germany_dedup": int(len(de)),
        "by_source": df.groupby("source").size().to_dict(),
        "germany_by_source": de.groupby("source").size().to_dict(),
        "germany_by_lang": de["lang"].value_counts().to_dict(),
        "germany_by_state": de["state"].fillna("unknown").value_counts().to_dict(),
        "germany_by_region": de["region"].fillna("other").value_counts().to_dict(),
        "germany_top_cities": de["city"].fillna("unknown").value_counts().head(20).to_dict(),
        "employers": int(emp.size),
        "employer_top10_share": round(top10_share, 4),
        "employer_hhi": round(float((shares**2).sum()), 5),
        "max_postings_one_employer": int(emp.iloc[0]) if emp.size else 0,
        "published_min": str(de["published"].dropna().min()) if len(de) else None,
        "published_max": str(de["published"].dropna().max()) if len(de) else None,
    }
