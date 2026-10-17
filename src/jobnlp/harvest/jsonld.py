"""Extract schema.org JobPosting objects from JSON-LD blocks in HTML."""

from __future__ import annotations

import json
from typing import Any

from bs4 import BeautifulSoup


def _walk(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(node, list):
        for item in node:
            out.extend(_walk(item))
    elif isinstance(node, dict):
        t = node.get("@type")
        types = t if isinstance(t, list) else [t]
        if "JobPosting" in types:
            out.append(node)
        for key in ("@graph", "mainEntity", "itemListElement", "item"):
            if key in node:
                out.extend(_walk(node[key]))
    return out


def extract_jobpostings(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    found: list[dict[str, Any]] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        try:
            data = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        found.extend(_walk(data))
    return found


def _first(x: Any) -> Any:
    return x[0] if isinstance(x, list) and x else x


def country_of(posting: dict[str, Any]) -> str | None:
    """addressCountry of the first job location, as an upper-case string where possible."""
    loc = _first(posting.get("jobLocation"))
    if not isinstance(loc, dict):
        return None
    addr = loc.get("address")
    if not isinstance(addr, dict):
        return None
    c = addr.get("addressCountry")
    if isinstance(c, dict):
        c = c.get("name") or c.get("identifier")
    return str(c).strip().upper() if c else None


def city_of(posting: dict[str, Any]) -> str | None:
    loc = _first(posting.get("jobLocation"))
    if isinstance(loc, dict) and isinstance(loc.get("address"), dict):
        city = loc["address"].get("addressLocality")
        return str(city).strip() if city else None
    return None


def employer_of(posting: dict[str, Any]) -> str | None:
    org = _first(posting.get("hiringOrganization"))
    if isinstance(org, dict):
        name = org.get("name")
        return str(name).strip() if name else None
    return str(org).strip() if isinstance(org, str) else None


GERMANY = {"DE", "DEU", "GERMANY", "DEUTSCHLAND"}


def is_germany(posting: dict[str, Any]) -> bool:
    return country_of(posting) in GERMANY
