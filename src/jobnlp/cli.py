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


def _client(min_interval: float | None = None) -> PoliteClient:
    from jobnlp.harvest.client import PoliteClient

    cfg = load_config()["harvest"]
    return PoliteClient(
        user_agent=cfg["user_agent"],
        cache_path=ROOT / "stores" / "http_cache.sqlite",
        min_interval_s=min_interval or cfg["min_interval_s"],
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


if __name__ == "__main__":
    app()
