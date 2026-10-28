"""Corpus build, location, language and sections on real stored postings."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.conftest import GEONAMES, need

from jobnlp.config import load_config
from jobnlp.text.corpus import aggregator_hosts, build, corpus_stats
from jobnlp.text.dedupe import cluster_near_duplicates
from jobnlp.text.geo import GermanPlaces
from jobnlp.text.lang import detect, detect_langdetect
from jobnlp.text.sections import section_text, split_sections


@pytest.fixture(scope="module")
def places() -> GermanPlaces:
    need(GEONAMES, "download GeoNames DE.zip")
    return GermanPlaces(GEONAMES)


@pytest.fixture(scope="module")
def small_corpus(raw_rows, places) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    arbeitnow = [r for r in raw_rows if r[0] == "arbeitnow"][:60]
    crawl = [r for r in raw_rows if r[0] == "crawl"][:40]
    return build(arbeitnow + crawl, places, load_config()["regions"], person_finder=None)


def test_build_produces_clean_located_rows(small_corpus: pd.DataFrame) -> None:
    df = small_corpus
    assert len(df) > 0 and set(df.source) <= {"arbeitnow", "crawl"}
    assert not df.text.str.contains("<div", regex=False).any()
    assert set(df.lang) <= {"de", "en", "other"}
    assert df[df.source == "crawl"].in_germany.all()
    stats = corpus_stats(df)
    assert stats["n_total"] == len(df) and stats["n_germany_dedup"] <= len(df)


def test_known_cities_resolve(raw_rows, places) -> None:  # type: ignore[no-untyped-def]
    locations = [r[2]["location"] for r in raw_rows if r[0] == "arbeitnow"][:500]
    berlin = [loc for loc in locations if loc.strip() == "Berlin"]
    assert berlin and places.match(berlin[0]).state == "Berlin"  # type: ignore[union-attr]
    london = [loc for loc in locations if "London" in loc]
    assert all(not places.is_germany(loc) for loc in london)


def test_two_language_detectors_mostly_agree(corpus_df: pd.DataFrame) -> None:
    sample = corpus_df[corpus_df.n_chars > 500].sample(80, random_state=0)
    agree = sum(detect(t) == detect_langdetect(t) for t in sample.text)
    assert agree / len(sample) > 0.9


def test_sections_found_in_real_postings(corpus_df: pd.DataFrame) -> None:
    sample = corpus_df[corpus_df.in_germany & ~corpus_df.is_dup].sample(50, random_state=0)
    with_req = 0
    for text in sample.text:
        secs = split_sections(text)
        assert secs and all(
            s.label in {"company", "tasks", "requirements", "offer", "other"} for s in secs
        )
        with_req += bool(section_text(secs, "requirements"))
    assert with_req / len(sample) > 0.5


def test_duplicates_cluster_on_real_text(corpus_df: pd.DataFrame) -> None:
    dup = corpus_df[corpus_df.is_dup].head(5)
    if dup.empty:
        pytest.skip("no duplicates in corpus")
    texts = []
    for cluster in dup.dup_cluster:
        members = corpus_df[corpus_df.dup_cluster == cluster].head(2)
        texts += (members.title + "\n" + members.text).tolist()
    ids = cluster_near_duplicates(texts)
    assert ids[0] == ids[1]


def test_aggregator_rule_on_real_store(raw_rows) -> None:  # type: ignore[no-untyped-def]
    hosts = aggregator_hosts(raw_rows)
    for h in hosts:
        emps = {
            str(p.get("employer") or "").lower()
            for s, _, p, _, _ in raw_rows
            if s == "crawl" and p.get("host") == h
        }
        assert len(emps - {""}) > 3
