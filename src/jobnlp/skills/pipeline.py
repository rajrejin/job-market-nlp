"""Extraction experiment: tune on the gold development half, evaluate every method.

Methods: gazetteer with ESCO labels only; gazetteer with ESCO and mined terms; the learned NER
(trained on weak labels from the gazetteer, never on gold data); and the union of the last two.
Thresholds (mining, label blocking) are chosen on the development half only.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

import pandas as pd
from spacy.language import Language

from jobnlp.skills.evaluate import GoldItem, evaluate, necessity_accuracy, partial_tp
from jobnlp.skills.extract import (
    SkillExtractor,
    build_gazetteer,
    load_nlp,
    merge_matches,
    mined_phrases,
)
from jobnlp.skills.gazetteer import Gazetteer, Match
from jobnlp.skills.mine import MinedTerm, mine, select
from jobnlp.text.sections import Section

MINING_GRID = [(d, r) for d in (2, 3, 5) for r in (0.0, 0.1, 0.2, 0.3, 0.5)]


def requirement_text(sections_json: str, max_chars: int = 3000) -> str:
    secs = [Section(**x) for x in json.loads(sections_json)]
    return "\n".join(s.text for s in secs if s.label in ("requirements", "tasks"))[:max_chars]


def unseen_postings(corpus: pd.DataFrame, gold_ids: set[tuple[str, str]]) -> pd.DataFrame:
    """German, deduplicated postings that are neither gold items nor in a gold item's
    duplicate cluster (so no gold text leaks into mining or NER training)."""
    keys = list(zip(corpus.source, corpus.posting_id, strict=True))
    is_gold = pd.Series([k in gold_ids for k in keys], index=corpus.index)
    gold_clusters = set(corpus.loc[is_gold, "dup_cluster"])
    keep = corpus.in_germany & ~corpus.is_dup & ~corpus.dup_cluster.isin(gold_clusters)
    return corpus[keep]


def mining_docs(corpus: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (str(e or "unknown"), requirement_text(s))
        for e, s in zip(corpus.employer, corpus.sections, strict=True)
    ]


def with_mined(base: Gazetteer, terms: list[MinedTerm]) -> Gazetteer:
    gaz = base.copy()
    names = [t.term for t in terms]
    for lang in ("de", "en"):
        gaz.add(load_nlp(lang), mined_phrases(names), override=True)
    return gaz


def canon_for(gaz: Gazetteer) -> Callable[[str, str], str]:
    cache: dict[tuple[str, str], str] = {}

    def canon(surface: str, lang: str) -> str:
        key = (surface, lang)
        if key not in cache:
            ms = gaz.match(load_nlp(lang)(surface))
            whole = [m for m in ms if m.start == 0 and m.end == len(surface)]
            cache[key] = whole[0].skill_id if whole else "surface:" + surface.lower().strip()
        return cache[key]

    return canon


def predictor(
    gaz: Gazetteer | None, ner: Language | None, canon: Callable[[str, str], str]
) -> Callable[[GoldItem], list[tuple[int, int, str]]]:
    def predict(it: GoldItem) -> list[tuple[int, int, str]]:
        matches: list[Match] = []
        if gaz is not None:
            matches = gaz.match(load_nlp(it.lang)(it.text))
        if ner is not None:
            extra = SkillExtractor(Gazetteer(), ner).ner_matches(it.text)
            matches = merge_matches(matches, extra)
        return [(m.start, m.end, canon(m.text, it.lang)) for m in matches]

    return predict


def tune_blocking(gaz: Gazetteer, dev: list[GoldItem], max_precision: float = 0.2) -> set[str]:
    """Surface keys whose matches on the dev half are mostly false positives (partial match)."""
    pred_n: Counter[str] = Counter()
    pred_tp: Counter[str] = Counter()
    for it in dev:
        doc = load_nlp(it.lang)(it.text)
        gold = [(a, b) for a, b, _, _ in it.spans]
        for m in gaz.match(doc):
            key = " ".join(t.lower_ for t in doc.char_span(m.start, m.end) or [])
            pred_n[key] += 1
            pred_tp[key] += partial_tp([(m.start, m.end)], gold)
    return {k for k, n in pred_n.items() if n >= 2 and pred_tp[k] / n <= max_precision}


def tune(
    esco: pd.DataFrame,
    corpus: pd.DataFrame,
    dev: list[GoldItem],
    gold_ids: set[tuple[str, str]],
) -> tuple[Gazetteer, Gazetteer, dict[str, Any], list[MinedTerm]]:
    """Return (ESCO gazetteer, ESCO+mined gazetteer, tuning record, selected mined terms)."""
    base = build_gazetteer(esco)
    terms = mine(mining_docs(unseen_postings(corpus, gold_ids)))
    known = set(base.surface) | {k.lower() for k in base.exact_short}
    grid = []
    for min_df, ratio in MINING_GRID:
        sel = select(terms, known, min_df=min_df, min_cue_ratio=ratio)
        gaz = with_mined(base, sel)
        canon = canon_for(gaz)
        f1 = evaluate(dev, predictor(gaz, None, canon), canon)["all"]["partial"]
        grid.append({"min_df": min_df, "min_cue_ratio": ratio, "n_terms": len(sel), **f1})
    best = max(grid, key=lambda g: (g["f1"], -g["n_terms"]))
    selected = select(terms, known, min_df=best["min_df"], min_cue_ratio=best["min_cue_ratio"])
    mined_gaz = with_mined(base, selected)
    blocked = tune_blocking(mined_gaz, dev)
    base.blocked |= blocked
    mined_gaz.blocked |= blocked
    info = {
        "mining_grid": grid,
        "chosen": best,
        "blocked_keys": sorted(blocked),
        "candidates": len(terms),
        "mined_selected": len(selected),
    }
    return base, mined_gaz, info, selected


def evaluate_all(
    base: Gazetteer, mined_gaz: Gazetteer, ner: Language | None, gold: list[GoldItem]
) -> dict[str, Any]:
    canon = canon_for(mined_gaz)
    methods: dict[str, Callable[[GoldItem], list[tuple[int, int, str]]]] = {
        "gazetteer_esco": predictor(base, None, canon),
        "gazetteer_esco_mined": predictor(mined_gaz, None, canon),
    }
    if ner is not None:
        methods["ner_weak"] = predictor(None, ner, canon)
        methods["union"] = predictor(mined_gaz, ner, canon)
    results: dict[str, Any] = {}
    for split in ("dev", "test"):
        items = [g for g in gold if g.split == split]
        results[split] = {name: evaluate(items, fn, canon) for name, fn in methods.items()}
        results[split]["necessity"] = necessity_accuracy(items)
    return results


def mined_table(selected: list[MinedTerm]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": t.term,
                "df_cue": t.df_cue,
                "df_all": t.df_all,
                "employers": t.employers,
                "cue_ratio": round(t.cue_ratio, 3),
            }
            for t in selected
        ]
    )
