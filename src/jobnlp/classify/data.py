"""Classification dataset: the platform's own department tags as labels, grouped splits.

Labels are Arbeitnow tags (free-form, set by employers' applicant tracking systems). Classes are
the most frequent tags with enough postings carrying exactly one class tag; postings with two or
more class tags are excluded and counted. Splits keep every employer and every near-duplicate
cluster on one side (70/10/20), so the test set has no employer seen in training.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from jobnlp.text.dedupe import UnionFind

IGNORE_TAGS = {"Remote", "All Departments", "Directors", "Chief Executives", "Team Leader"}


def label_table(
    corpus: pd.DataFrame, min_support: int = 60, max_classes: int = 12
) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = corpus[(corpus.source == "arbeitnow") & ~corpus.is_dup & corpus.lang.isin(["de", "en"])]
    tag_counts = Counter(t for ts in d.categories for t in ts if t not in IGNORE_TAGS)
    candidates = [t for t, _ in tag_counts.most_common(60)]
    # single-label support per candidate, then keep the top classes above min_support
    single: Counter[str] = Counter()
    for ts in d.categories:
        hit = [t for t in ts if t in candidates]
        if len(hit) == 1:
            single[hit[0]] += 1
    classes = [t for t, n in single.most_common(max_classes) if n >= min_support]
    rows: list[dict[str, Any]] = []
    multi = none = 0
    for r in d.to_dict("records"):
        hit = sorted({t for t in r["categories"] if t in classes})
        if len(hit) == 1:
            rows.append(
                {
                    "source": r["source"],
                    "posting_id": r["posting_id"],
                    "employer": r["employer"] or "",
                    "dup_cluster": int(r["dup_cluster"]),
                    "lang": r["lang"],
                    "title": r["title"],
                    "description": r["text"],
                    "label": hit[0],
                }
            )
        elif len(hit) > 1:
            multi += 1
        else:
            none += 1
    df = pd.DataFrame(rows)
    info = {
        "classes": classes,
        "n_examples": len(df),
        "excluded_multi_class": multi,
        "excluded_no_class_tag": none,
        "label_counts": df.label.value_counts().to_dict(),
    }
    return df, info


def group_split(
    df: pd.DataFrame, fractions: tuple[float, float, float] = (0.7, 0.1, 0.2), seed: int = 0
) -> pd.Series:
    """'train' / 'val' / 'test' per row; groups = connected components of employer and
    duplicate cluster. Groups are assigned greedily to the split furthest below its target."""
    uf = UnionFind(len(df))
    first: dict[tuple[str, Any], int] = {}
    for i, (emp, cl) in enumerate(zip(df.employer, df.dup_cluster, strict=True)):
        for key in (("e", emp), ("c", cl)):
            if key[1] in ("", None):
                continue
            if key in first:
                uf.union(first[key], i)
            else:
                first[key] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(df)):
        groups[uf.find(i)].append(i)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    order.sort(key=len, reverse=True)  # place large groups first, ties in shuffled order
    names = ("train", "val", "test")
    target = [f * len(df) for f in fractions]
    filled = [0, 0, 0]
    assign = [""] * len(df)
    for g in order:
        k = max(range(3), key=lambda j: target[j] - filled[j])
        for i in g:
            assign[i] = names[k]
        filled[k] += len(g)
    return pd.Series(assign, index=df.index)


def text_view(df: pd.DataFrame, view: str) -> list[str]:
    if view == "title":
        return df.title.fillna("").tolist()
    if view == "description":
        return df.description.fillna("").tolist()
    return (df.title.fillna("") + "\n" + df.description.fillna("")).tolist()
