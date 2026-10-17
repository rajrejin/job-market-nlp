"""jobnlp command line interface."""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING

import typer

from jobnlp.config import ROOT, load_config

if TYPE_CHECKING:
    from jobnlp.harvest.client import PoliteClient
    from jobnlp.harvest.store import PostingStore

app = typer.Typer(no_args_is_help=True, help="German job market NLP analyser and CV gap tool.")


@app.command()
def version() -> None:
    """Print the package version."""
    from jobnlp import __version__

    typer.echo(__version__)


def _client(min_interval: float | None = None, max_retries: int = 5) -> PoliteClient:
    from jobnlp.harvest.client import PoliteClient

    cfg = load_config()["harvest"]
    return PoliteClient(
        user_agent=cfg["user_agent"],
        cache_path=ROOT / "stores" / "http_cache.sqlite",
        min_interval_s=min_interval or cfg["min_interval_s"],
        max_retries=max_retries,
    )


def _store() -> PostingStore:
    from jobnlp.harvest.store import PostingStore

    return PostingStore(ROOT / "stores" / "postings.sqlite")


@app.command()
def harvest(snapshot: str = typer.Option("", help="Snapshot date, default today (UTC).")) -> None:
    """Take a snapshot of the Arbeitnow public API."""
    from jobnlp.harvest.arbeitnow import harvest_snapshot

    snap = snapshot or dt.datetime.now(dt.UTC).date().isoformat()
    store = _store()
    stats = harvest_snapshot(_client(), store, snap)
    out = ROOT / "results" / "harvest_progress.json"
    out.write_text(
        json.dumps({"snapshot": snap, **stats, "totals": store.counts()}, indent=1),
        encoding="utf-8",
    )
    typer.echo(json.dumps(stats))


@app.command()
def crawl(
    n_hosts: int = typer.Option(0, help="Random sample of hosts (seed 0); 0 means all."),
    max_pages: int = typer.Option(200, help="Job pages per host per snapshot."),
    snapshot: str = typer.Option("", help="Snapshot date, default today (UTC)."),
    out: str = typer.Option("results/crawl_summary.json", help="Summary output path."),
) -> None:
    """Crawl employer career sites with JobPosting markup (robots.txt and TDM opt-outs honoured)."""
    import io
    import random
    import zipfile

    from jobnlp.harvest.client import PoliteClient
    from jobnlp.harvest.crawl import candidate_hosts, run_crawl, summarise

    cfg = load_config()["harvest"]
    stats_zip = ROOT / "data" / "raw" / "wdc" / "stats.zip"
    csv_path = ROOT / "data" / "interim" / "wdc_jobposting_hosts.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(stats_zip) as z:
            raw = z.read("table_statistics/JobPosting_September2020_statistics.csv")
        csv_path.write_text(io.BytesIO(raw).read().decode("utf-8"), encoding="utf-8")
    hosts = candidate_hosts(csv_path)
    if n_hosts:
        hosts = random.Random(0).sample(hosts, min(n_hosts, len(hosts)))
    snap = snapshot or dt.datetime.now(dt.UTC).date().isoformat()
    client = PoliteClient(
        user_agent=cfg["user_agent"],
        cache_path=ROOT / "stores" / "http_cache.sqlite",
        min_interval_s=cfg["crawl_min_interval_s"],
        max_retries=2,
        timeout_s=20.0,
    )
    store = _store()
    reports = run_crawl(client, store, hosts, cfg["user_agent"], snap, max_pages)
    summary = {"snapshot": snap, "candidate_hosts": len(hosts), **summarise(reports)}
    (ROOT / out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    hosts_out = ROOT / "data" / "interim" / f"crawl_hosts_{snap}.json"
    hosts_out.write_text(json.dumps([r.__dict__ for r in reports], indent=1), encoding="utf-8")
    typer.echo(json.dumps(summary))


@app.command()
def esco() -> None:
    """Fetch ESCO skills (German and English labels) from the public ESCO API."""
    from jobnlp.skills.esco import fetch_skills

    df = fetch_skills(_client(max_retries=1), ROOT / "data" / "raw" / "esco")
    out = ROOT / "data" / "interim" / "esco_skills.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    summary = {
        "skills": int(len(df)),
        "api_total": None,
        "with_german_label": int(df.pref_de.notna().sum()),
        "digital": int(df.is_digital.sum()),
        "by_type": df.skill_type.value_counts().to_dict(),
        "accessed": dt.datetime.now(dt.UTC).date().isoformat(),
        "source": "ESCO web service API https://ec.europa.eu/esco/api",
    }
    (ROOT / "results" / "esco_summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    typer.echo(json.dumps(summary))


@app.command("build-corpus")
def build_corpus(no_ner: bool = typer.Option(False, help="Skip spaCy name redaction.")) -> None:
    """Clean, redact, locate and deduplicate all stored postings into Parquet."""
    from jobnlp.text.corpus import build, corpus_stats
    from jobnlp.text.geo import GermanPlaces
    from jobnlp.text.redact import spacy_person_finder

    cfg = load_config()
    places = GermanPlaces(ROOT / "data" / "raw" / "geonames" / "DE.zip")
    finder = None if no_ner else spacy_person_finder()
    df = build(_store().iter_rows(), places, cfg["regions"], finder)
    out = ROOT / "data" / "processed" / "postings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    stats = corpus_stats(df)
    (ROOT / "results" / "corpus_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps({k: stats[k] for k in ("n_total", "n_duplicates", "n_germany_dedup")}))


if __name__ == "__main__":
    app()
