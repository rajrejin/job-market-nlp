"""Integration tests run on real cached data and skip when it is absent."""

from __future__ import annotations

import pandas as pd
import pytest

from jobnlp.config import ROOT

CORPUS = ROOT / "data" / "processed" / "postings.parquet"
STORE = ROOT / "stores" / "postings.sqlite"


@pytest.fixture(scope="session")
def corpus() -> pd.DataFrame:
    if not CORPUS.exists():
        pytest.skip("processed corpus absent: run `jobnlp harvest` and `jobnlp build-corpus`")
    return pd.read_parquet(CORPUS)


@pytest.fixture(scope="session")
def raw_payloads() -> list[dict]:  # type: ignore[type-arg]
    if not STORE.exists():
        pytest.skip("posting store absent: run `jobnlp harvest`")
    from jobnlp.harvest.store import PostingStore

    rows = [p for _, _, p in PostingStore(STORE).iter_payloads("arbeitnow")]
    return rows[:300]
