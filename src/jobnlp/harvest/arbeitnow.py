"""Snapshot harvest of the Arbeitnow public job board API (no key; terms ask for a link back)."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from jobnlp.harvest.client import PoliteClient
from jobnlp.harvest.store import PostingStore
from jobnlp.logging_utils import get_logger

API = "https://www.arbeitnow.com/api/job-board-api"
SOURCE = "arbeitnow"
log = get_logger("harvest.arbeitnow", "harvest.log")


def parse_page(text: str) -> tuple[list[dict[str, Any]], str | None]:
    data = json.loads(text)
    return list(data.get("data", [])), (data.get("links") or {}).get("next")


def harvest_snapshot(
    client: PoliteClient, store: PostingStore, snapshot: str, max_pages: int = 200
) -> dict[str, int]:
    """Page through the API until it runs out. Pages are cached per snapshot, so a rerun on the
    same snapshot date resumes without refetching."""
    store.start_snapshot(SOURCE, snapshot, dt.datetime.now(dt.UTC).isoformat())
    seen = new = pages = 0
    url: str | None = f"{API}?page=1"
    while url and pages < max_pages:
        r = client.get(url, namespace=f"{SOURCE}:{snapshot}")
        if r.status != 200:
            log.warning("status %s on %s", r.status, url)
            break
        rows, url = parse_page(r.text)
        pages += 1
        if not rows:
            break
        for row in rows:
            seen += 1
            new += store.upsert(SOURCE, str(row["slug"]), snapshot, row)
        store.commit()
        log.info(
            "page %d rows %d seen %d new %d cached=%s", pages, len(rows), seen, new, r.from_cache
        )
    store.finish_snapshot(SOURCE, snapshot, dt.datetime.now(dt.UTC).isoformat(), seen)
    return {"pages": pages, "seen": seen, "new": new}
