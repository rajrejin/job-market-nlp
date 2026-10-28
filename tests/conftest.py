"""Shared fixtures. Anything that needs real data uses the local caches and skips without them;
nothing here creates synthetic postings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jobnlp.config import ROOT

HTTP_CACHE = ROOT / "stores" / "http_cache.sqlite"
STORE = ROOT / "stores" / "postings.sqlite"
CORPUS = ROOT / "data" / "processed" / "postings.parquet"
ESCO = ROOT / "data" / "interim" / "esco_skills.parquet"
GOLD = ROOT / "annotation" / "gold_annotations.jsonl"
GEONAMES = ROOT / "data" / "raw" / "geonames" / "DE.zip"
SNAPSHOT = "2026-10-17"  # first harvest; its responses are in the HTTP cache


def need(path: Path, hint: str) -> None:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} absent: {hint}")


@pytest.fixture(scope="session")
def cached_client():  # type: ignore[no-untyped-def]
    """A polite client that answers from the real HTTP cache (4 GB, read in place). Only
    uncached URLs would be written; the tests ask for cached ones only."""
    from jobnlp.harvest.client import PoliteClient

    need(HTTP_CACHE, "run `jobnlp harvest` and `jobnlp crawl`")
    client = PoliteClient(
        "jobnlp-tests", HTTP_CACHE, min_interval_s=0.0, max_retries=0, timeout_s=10
    )
    yield client
    client.close()


@pytest.fixture(scope="session")
def raw_rows() -> list[tuple[str, str, dict, str, str]]:  # type: ignore[type-arg]
    from jobnlp.harvest.store import PostingStore

    need(STORE, "run `jobnlp harvest`")
    return list(PostingStore(STORE).iter_rows())


@pytest.fixture(scope="session")
def corpus_df() -> pd.DataFrame:
    need(CORPUS, "run `jobnlp build-corpus`")
    return pd.read_parquet(CORPUS)


@pytest.fixture(scope="session")
def esco_df() -> pd.DataFrame:
    need(ESCO, "run `jobnlp esco`")
    return pd.read_parquet(ESCO)


@pytest.fixture(scope="session")
def gold_items():  # type: ignore[no-untyped-def]
    from jobnlp.skills.evaluate import load_gold

    need(GOLD, "the gold set lives in annotation/ (not committed)")
    return load_gold(ROOT / "annotation")


@pytest.fixture(scope="session")
def extractor():  # type: ignore[no-untyped-def]
    from jobnlp.skills.runtime import get_extractor

    need(ESCO, "run `jobnlp esco`")
    return get_extractor()
