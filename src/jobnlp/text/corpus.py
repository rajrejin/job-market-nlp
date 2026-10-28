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


MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [
            ("jan", "januar", "january", "jänner"),
            ("feb", "februar", "february"),
            ("mar", "mär", "märz", "march"),
            ("apr", "april"),
            ("may", "mai"),
            ("jun", "juni", "june"),
            ("jul", "juli", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "okt", "oktober", "october"),
            ("nov", "november"),
            ("dec", "dez", "dezember", "december"),
        ]
    )
    for m in names
}
ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
EN_DATE = re.compile(r"^([A-Za-zä]+)\.?\s+(\d{1,2}),?\s+(\d{4})")  # "Mar 01, 2025"
DE_DATE = re.compile(r"^(\d{1,2})\.\s*([A-Za-zä]+)\.?\s+(\d{4})")  # "24. August 2025"
DOT_DATE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})")  # "24.08.2025"


def parse_published(raw: Any) -> str | None:
    """JSON-LD datePosted as an ISO date. Employers write ISO, English and German forms;
    anything unrecognised becomes None rather than a wrong date."""
    s = str(raw or "").strip()
    try:
        if m := ISO_DATE.match(s):
            y, mo, d = int(m[1]), int(m[2]), int(m[3])
        elif m := EN_DATE.match(s):
            y, mo, d = int(m[3]), MONTHS[m[1].lower()], int(m[2])
        elif m := DE_DATE.match(s):
            y, mo, d = int(m[3]), MONTHS[m[2].lower()], int(m[1])
        elif m := DOT_DATE.match(s):
            y, mo, d = int(m[3]), int(m[2]), int(m[1])
        else:
            return None
        return dt.date(y, mo, d).isoformat()
    except (KeyError, ValueError):
        return None


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
    return {
        "source": source,
        "posting_id": pid,
        "title": p.get("title") or "",
        "employer": p.get("employer"),
        "location_raw": p.get("city") or "",
        "remote": False,
        "categories": _as_list(p.get("occupational_category")) + _as_list(p.get("industry")),
        "job_types": _as_list(p.get("employment_type")),
        "published": parse_published(p.get("date_posted")),
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
    rows = list(rows)
    aggregators = aggregator_hosts(rows)
    excluded = 0
    recs = []
    for source, pid, payload, first, last in rows:
        if source == "crawl" and payload.get("host") in aggregators:
            excluded += 1
            continue
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
    df.attrs["excluded_aggregator_hosts"] = len(aggregators)
    df.attrs["excluded_aggregator_postings"] = excluded
    return df


def aggregator_hosts(
    rows: list[tuple[str, str, dict[str, Any], str, str]], max_employers: int = 3
) -> set[str]:
    """Crawled hosts whose postings name more than `max_employers` hiring organisations are
    job boards or agencies, not employer career sites; their postings are excluded."""
    seen: dict[str, set[str]] = {}
    for source, _, payload, _, _ in rows:
        if source == "crawl":
            seen.setdefault(str(payload.get("host")), set()).add(
                str(payload.get("employer") or "").lower()
            )
    return {h for h, emps in seen.items() if len(emps - {""}) > max_employers}


def corpus_stats(df: pd.DataFrame) -> dict[str, Any]:
    de = df[df.in_germany & ~df.is_dup]
    emp = de["employer"].fillna("unknown").value_counts()
    top10_share = float(emp.head(10).sum() / max(1, len(de)))
    shares = emp / max(1, len(de))
    return {
        "n_total": int(len(df)),
        "excluded_aggregator_hosts": int(df.attrs.get("excluded_aggregator_hosts", 0)),
        "excluded_aggregator_postings": int(df.attrs.get("excluded_aggregator_postings", 0)),
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
