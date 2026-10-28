"""Recompute only the `published` column of the existing corpus with `parse_published`.

The first corpus build cut crawl dates to 10 characters, so "Mar 01, 2025" became
"Mar 01, 20" (77 of 3,924 crawl postings). A full `jobnlp build-corpus` would also rerun
redaction and could shift texts that the gold set and the classifier splits refer to, so
this script changes the one affected column and the corpus statistics, nothing else.
Run `jobnlp market` afterwards: the trend test reads the dates.
"""

from __future__ import annotations

import json

import pandas as pd

from jobnlp.config import ROOT
from jobnlp.harvest.store import PostingStore
from jobnlp.text.corpus import corpus_stats, parse_published


def main() -> None:
    path = ROOT / "data" / "processed" / "postings.parquet"
    df = pd.read_parquet(path)
    store = PostingStore(ROOT / "stores" / "postings.sqlite")
    fixed = {
        pid: parse_published(p.get("date_posted"))
        for src, pid, p in store.iter_payloads("crawl")
        if src == "crawl"
    }
    crawl = df.source == "crawl"
    before = df.loc[crawl, "published"].copy()
    df.loc[crawl, "published"] = df.loc[crawl, "posting_id"].map(fixed)
    changed = int((before.fillna("") != df.loc[crawl, "published"].fillna("")).sum())
    df.to_parquet(path, index=False)
    (ROOT / "results" / "corpus_stats.json").write_text(
        json.dumps(corpus_stats(df), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"crawl_postings": int(crawl.sum()), "published_changed": changed}))


if __name__ == "__main__":
    main()
