"""Export 100 real postings (not in the gold set) for labelling section headings.

Usage (in the container): python scripts/splitter_sample.py
Writes annotation/splitter_sample.jsonl (gitignored) and prints candidate lines to label."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jobnlp.config import ROOT

corpus = pd.read_parquet(ROOT / "data" / "processed" / "postings.parquet")
gold = {
    json.loads(x)["posting_id"]
    for x in (ROOT / "annotation" / "gold_sample.jsonl").read_text("utf-8").splitlines()
}
pool = corpus[
    corpus.in_germany
    & ~corpus.is_dup
    & (corpus.n_chars >= 600)
    & ~corpus.posting_id.isin(gold)
    & corpus.lang.isin(["de", "en"])
]
sample = pd.concat(
    [
        pool[pool.source == "arbeitnow"].sample(60, random_state=7),
        pool[pool.source == "crawl"].sample(40, random_state=7),
    ]
).sample(frac=1, random_state=7)
out = Path(ROOT / "annotation" / "splitter_sample.jsonl")
with out.open("w", encoding="utf-8") as f:
    for i, r in enumerate(sample.itertuples()):
        sid = f"s{i:03d}"
        f.write(
            json.dumps({"sid": sid, "posting_id": r.posting_id, "text": r.text}, ensure_ascii=False)
            + "\n"
        )
print(f"wrote {len(sample)} postings to {out}")
