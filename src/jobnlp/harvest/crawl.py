"""Polite crawler for employer career sites that publish schema.org JobPosting markup.

Each host is processed as a generator that yields URLs and receives responses; the scheduler
interleaves hosts round robin so every host sees at most one request per `min_interval_s`.
Resume: responses are cached per snapshot, so a rerun on the same snapshot replays the cache
and continues where it stopped.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
from collections import Counter
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from jobnlp.harvest.client import PoliteClient, Response
from jobnlp.harvest.jsonld import city_of, employer_of, extract_jobpostings, is_germany
from jobnlp.harvest.policy import (
    is_excluded_host,
    page_reserved,
    parse_robots,
    robots_allows,
    robots_sitemaps,
    tdmrep_reserved,
)
from jobnlp.harvest.store import PostingStore
from jobnlp.logging_utils import get_logger

SOURCE = "crawl"
# a host whose postings name more hiring organisations than this is treated as a job board
MAX_EMPLOYERS_PER_HOST = 3
JOB_PATH = re.compile(
    r"/(jobs?|stellen|stellenangebote?|offene-stellen|karriere|career|careers|vacanc\w*|"
    r"position\w*|ausschreibung\w*|jobangebote?)(/|-|$)",
    re.IGNORECASE,
)
LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
log = get_logger("harvest.crawl", "crawl.log")

HostTask = Generator[str, Response, None]


@dataclass
class HostReport:
    host: str
    status: str = "pending"
    sitemaps: int = 0
    job_urls: int = 0
    pages: int = 0
    postings: int = 0
    postings_de: int = 0
    notes: list[str] = field(default_factory=list)


def candidate_hosts(stats_csv: Path, min_rows: int = 5) -> list[str]:
    """German hosts from the Web Data Commons JobPosting table statistics, boards excluded."""
    with stats_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hosts = [
        r["host"]
        for r in rows
        if r["host"].endswith(".de")
        and int(r["number_of_rows"]) >= min_rows
        and not is_excluded_host(r["host"])
    ]
    return sorted(set(hosts))


def sitemap_urls(xml: str) -> tuple[list[str], bool]:
    """Return (<loc> URLs, is_index)."""
    return LOC.findall(xml), "<sitemapindex" in xml[:2000].lower()


def host_task(
    host: str,
    user_agent: str,
    store: PostingStore,
    snapshot: str,
    report: HostReport,
    max_pages: int,
    max_sitemaps: int = 20,
) -> HostTask:
    base = f"https://{host}"
    r = yield f"{base}/robots.txt"
    robots_text = r.text if r.status == 200 else ""
    rp = parse_robots(robots_text, f"{base}/robots.txt")
    if r.status in (401, 403):
        report.status = "blocked_robots_status"
        return
    if not robots_allows(rp, user_agent, f"{base}/"):
        report.status = "blocked_robots"
        return
    r = yield f"{base}/.well-known/tdmrep.json"
    if r.status == 200 and tdmrep_reserved(r.text):
        report.status = "tdm_reserved"
        return

    queue = robots_sitemaps(robots_text) or [f"{base}/sitemap.xml"]
    seen_maps: set[str] = set()
    job_urls: list[str] = []
    while queue and len(seen_maps) < max_sitemaps and len(job_urls) < max_pages:
        sm = queue.pop(0)
        if sm in seen_maps or not robots_allows(rp, user_agent, sm):
            continue
        seen_maps.add(sm)
        if sm.endswith(".gz"):
            report.notes.append("gz sitemap skipped")
            continue
        r = yield sm
        if r.status != 200:
            continue
        locs, is_index = sitemap_urls(r.text)
        if is_index:
            # job-looking child sitemaps first
            queue = sorted(locs, key=lambda u: not JOB_PATH.search(u) and "job" not in u) + queue
        else:
            job_urls.extend(u for u in locs if JOB_PATH.search(urlparse(u).path))
    report.sitemaps = len(seen_maps)
    job_urls = list(dict.fromkeys(job_urls))[:max_pages]
    report.job_urls = len(job_urls)
    if not job_urls:
        report.status = "no_job_urls"
        return
    report.status = "crawled"
    empty_streak = 0
    employers: set[str] = set()
    for url in job_urls:
        if urlparse(url).netloc.removeprefix("www.") != host.removeprefix("www."):
            continue
        if not robots_allows(rp, user_agent, url):
            continue
        r = yield url
        report.pages += 1
        if r.status != 200:
            continue
        if page_reserved(r.text):
            report.status = "tdm_reserved_page"
            return
        postings = extract_jobpostings(r.text)
        empty_streak = 0 if postings else empty_streak + 1
        if empty_streak >= 15 and report.postings == 0:
            report.status = "no_markup"
            return
        for p in postings:
            report.postings += 1
            employers.add((employer_of(p) or "").lower())
            if not is_germany(p):
                continue
            report.postings_de += 1
            pid = hashlib.sha1(f"{url}|{p.get('title', '')}".encode()).hexdigest()
            store.upsert(
                SOURCE,
                pid,
                snapshot,
                {
                    "url": url,
                    "host": host,
                    "title": p.get("title"),
                    "description": p.get("description"),
                    "employer": employer_of(p),
                    "city": city_of(p),
                    "date_posted": p.get("datePosted"),
                    "employment_type": p.get("employmentType"),
                    "occupational_category": p.get("occupationalCategory"),
                    "industry": p.get("industry"),
                },
            )
        store.commit()
        if len(employers - {""}) > MAX_EMPLOYERS_PER_HOST:
            report.status = "aggregator"  # a job board or agency, not an employer career site
            return


def run_crawl(
    client: PoliteClient,
    store: PostingStore,
    hosts: list[str],
    user_agent: str,
    snapshot: str,
    max_pages: int = 200,
    workers: int = 16,
) -> list[HostReport]:
    """Crawl hosts with `workers` threads, one host per worker at a time. The client's per-host
    rate limiter still spaces requests to each host; concurrency is only across hosts."""
    store.start_snapshot(SOURCE, snapshot, dt.datetime.now(dt.UTC).isoformat())
    reports = {h: HostReport(h) for h in hosts}

    def run_host(h: str) -> None:
        gen = host_task(h, user_agent, store, snapshot, reports[h], max_pages)
        try:
            url = next(gen)
            while True:
                resp = client.get(url, namespace=f"{SOURCE}:{snapshot}")
                url = gen.send(resp)
        except StopIteration:
            pass
        except Exception as exc:  # one broken host must not stop the crawl
            reports[h].status = "error"
            reports[h].notes.append(type(exc).__name__)
        log.info(
            "host done %s %s pages=%d de=%d",
            h,
            reports[h].status,
            reports[h].pages,
            reports[h].postings_de,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_host, hosts))
    total = sum(r.postings_de for r in reports.values())
    store.finish_snapshot(SOURCE, snapshot, dt.datetime.now(dt.UTC).isoformat(), total)
    return list(reports.values())


def summarise(reports: list[HostReport]) -> dict[str, object]:
    return {
        "hosts": len(reports),
        "status": dict(Counter(r.status for r in reports)),
        "pages_fetched": sum(r.pages for r in reports),
        "postings_seen": sum(r.postings for r in reports),
        "postings_germany": sum(r.postings_de for r in reports),
        "hosts_with_postings": sum(r.postings_de > 0 for r in reports),
    }
