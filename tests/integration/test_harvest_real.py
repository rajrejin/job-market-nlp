"""Harvest and crawl code replayed against real cached responses (no network)."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import SNAPSHOT

from jobnlp.harvest.arbeitnow import harvest_snapshot, parse_page
from jobnlp.harvest.crawl import (
    HostReport,
    candidate_hosts,
    run_crawl,
    sitemap_urls,
    summarise,
)
from jobnlp.harvest.jsonld import city_of, employer_of, extract_jobpostings, is_germany
from jobnlp.harvest.store import PostingStore


def test_arbeitnow_snapshot_replays_from_cache(cached_client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store = PostingStore(tmp_path / "p.sqlite")
    stats = harvest_snapshot(cached_client, store, SNAPSHOT, max_pages=3)
    assert stats["pages"] == 3 and stats["seen"] > 0
    assert store.counts()["arbeitnow"] == stats["new"]
    # second run on the same snapshot: nothing new, presence already recorded
    again = harvest_snapshot(cached_client, store, SNAPSHOT, max_pages=3)
    assert again["new"] == 0 and again["seen"] == stats["seen"]
    rows = list(store.iter_rows())
    assert rows and rows[0][3] == SNAPSHOT


def test_parse_page_structure(cached_client) -> None:  # type: ignore[no-untyped-def]
    r = cached_client.get(
        "https://www.arbeitnow.com/api/job-board-api?page=1", namespace=f"arbeitnow:{SNAPSHOT}"
    )
    rows, nxt = parse_page(r.text)
    assert r.from_cache and rows and nxt and "page=2" in nxt
    assert {"slug", "title", "description", "location", "created_at"} <= set(rows[0])


def test_crawl_replays_one_career_site(cached_client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store = PostingStore(tmp_path / "p.sqlite")
    reports = run_crawl(
        cached_client, store, ["zag.de"], "jobnlp-research", SNAPSHOT, max_pages=5, workers=1
    )
    rep = reports[0]
    assert rep.status in {"crawled", "aggregator"} and rep.pages > 0 and rep.postings_de > 0
    summary = summarise(reports)
    assert summary["hosts_with_postings"] == 1
    payload = next(store.iter_payloads("crawl"))[2]
    assert payload["host"] == "zag.de" and payload["title"]


def test_jsonld_on_a_cached_job_page(cached_client) -> None:  # type: ignore[no-untyped-def]
    sm = cached_client.get("https://www.zag.de/sitemap.xml", namespace=f"crawl:{SNAPSHOT}")
    locs, _ = sitemap_urls(sm.text)
    job = next(u for u in locs if "stellenangebote" in u)
    page = cached_client.get(job, namespace=f"crawl:{SNAPSHOT}")
    postings = extract_jobpostings(page.text)
    assert postings and is_germany(postings[0])
    assert employer_of(postings[0]) and city_of(postings[0])


def test_candidate_hosts_exclude_boards() -> None:
    from jobnlp.config import ROOT

    csv = ROOT / "data" / "interim" / "wdc_jobposting_hosts.csv"
    if not csv.exists():
        import pytest

        pytest.skip("WDC host list absent")
    hosts = candidate_hosts(csv)
    assert len(hosts) > 500 and all(h.endswith(".de") for h in hosts)
    assert "stellenanzeigen.de" not in hosts and "jobs.de" not in hosts


def test_host_report_defaults() -> None:
    r = HostReport("example.de")
    assert r.status == "pending" and r.postings_de == 0
