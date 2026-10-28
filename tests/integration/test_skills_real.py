"""Skill extraction on real ESCO labels, real postings and the gold set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jobnlp.skills.esco import fetch_skills
from jobnlp.skills.evaluate import evaluate, necessity_accuracy, place_spans
from jobnlp.skills.extract import build_gazetteer, load_nlp, merge_matches
from jobnlp.skills.mine import mine, select
from jobnlp.skills.pipeline import canon_for, mining_docs, requirement_text, unseen_postings

LABELS = {
    "Python (Computerprogrammierung)",
    "SQL",
    "Deutsch",
    "Englisch",
    "Projektmanagement",
    "R",
    "Datenbank",
}


@pytest.fixture(scope="module")
def small_gaz(esco_df: pd.DataFrame):  # type: ignore[no-untyped-def]
    sub = esco_df[esco_df.pref_de.isin(LABELS)]
    return build_gazetteer(sub, mined=["Docker", "CI/CD", "Kubernetes"])


def test_esco_replays_from_cache(cached_client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    df = fetch_skills(cached_client, tmp_path)
    assert len(df) > 13_000 and df.uri.is_unique
    assert df.pref_de.notna().all()
    assert {"skill", "knowledge"} <= set(df.skill_type)


def test_gazetteer_surface_lemma_short_and_compounds(small_gaz) -> None:  # type: ignore[no-untyped-def]
    nlp = load_nlp("de")
    text = (
        "Sehr gute Kenntnisse in Python und SQL, Erfahrung mit Docker und CI/CD. "
        "Fließende Deutschkenntnisse und Englisch. Erfahrung mit Datenbanken."
    )
    found = {m.text for m in small_gaz.match(nlp(text))}
    assert {"Python", "SQL", "Docker", "Deutschkenntnisse", "Englisch", "Datenbanken"} <= found


def test_one_letter_skill_needs_context(small_gaz) -> None:  # type: ignore[no-untyped-def]
    nlp = load_nlp("de")
    assert "R" in {m.text for m in small_gaz.match(nlp("Kenntnisse in Python, R und SQL"))}
    assert "R" not in {m.text for m in small_gaz.match(nlp("Abteilung R wurde gegründet."))}


def test_gazetteer_on_real_postings(small_gaz, corpus_df: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    texts = corpus_df[corpus_df.text.str.contains("Python", regex=False)].text.head(10)
    assert any("Python" in {m.text for m in small_gaz.match(load_nlp("de")(t))} for t in texts)


def test_mining_on_real_requirements(corpus_df: pd.DataFrame) -> None:
    pool = corpus_df[corpus_df.in_germany & ~corpus_df.is_dup].head(400)
    terms = mine(mining_docs(pool))
    picked = {t.term for t in select(terms, set(), min_df=3, min_cue_ratio=0.1)}
    assert picked & {"Excel", "Docker", "SAP", "Python", "Teamfähigkeit", "AWS"}
    assert not picked & {"Wort", "Schrift", "C1", "B2"}


def test_gold_spans_score_perfectly_against_themselves(gold_items, small_gaz) -> None:  # type: ignore[no-untyped-def]
    canon = canon_for(small_gaz)
    items = gold_items[:20]
    res = evaluate(
        items, lambda it: [(a, b, canon(it.text[a:b], it.lang)) for a, b, _, _ in it.spans], canon
    )
    assert res["all"]["exact"]["f1"] == 1.0 and res["all"]["skill_set"]["f1"] == 1.0
    nec = necessity_accuracy(items)
    assert nec["n_spans"] > 0 and 0 <= nec["accuracy"] <= 1


def test_place_spans_longest_first_word_bounded() -> None:
    text = "Erfahrung mit AI-Agents und AI, nicht MAIL"
    spans = place_spans(
        text,
        [
            {"text": "AI", "type": "tool", "nec": "must"},
            {"text": "AI-Agents", "type": "tool", "nec": "must"},
        ],
    )
    assert [text[a:b] for a, b, _, _ in spans] == ["AI-Agents", "AI"]


def test_unseen_postings_excludes_gold(corpus_df: pd.DataFrame, gold_items) -> None:  # type: ignore[no-untyped-def]
    import json

    from jobnlp.config import ROOT

    rows = (ROOT / "annotation" / "gold_sample.jsonl").read_text("utf-8").splitlines()
    ids = {(r["source"], r["posting_id"]) for r in map(json.loads, rows)}
    pool = unseen_postings(corpus_df, ids)
    assert not ids & set(zip(pool.source, pool.posting_id, strict=True))
    assert requirement_text(pool.sections.iloc[0]) is not None


def test_weak_ner_trains_and_predicts(small_gaz, corpus_df: pd.DataFrame, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import spacy

    from jobnlp.skills.ner_train import train

    texts = [requirement_text(s) for s in corpus_df[corpus_df.in_germany].sections.head(40)]
    stats = train(small_gaz, texts, tmp_path / "ner", epochs=1)
    assert stats["n_train"] > 0 and len(stats["history"]) == 1
    nlp = spacy.load(tmp_path / "ner")
    assert nlp("Erfahrung mit Python").ents is not None


def test_runtime_extractor_on_real_posting(extractor, corpus_df: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    text = corpus_df[corpus_df.text.str.contains("Excel", regex=False)].text.iloc[0]
    mentions = extractor.extract(text, "de", use_ner=False)
    assert mentions and all(text[m.start : m.end] == m.text for m in mentions)
    assert {m.necessity for m in mentions} <= {"must", "nice"}
    assert merge_matches([], []) == []
