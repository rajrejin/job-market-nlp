"""Market analysis and classifier code on real corpus slices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jobnlp.classify.baselines import TfidfLR
from jobnlp.classify.data import group_split, label_table, text_view
from jobnlp.classify.evaluate import bootstrap_ci, metrics, paired_bootstrap
from jobnlp.classify.experiment import breakdown, merge_results
from jobnlp.market.aggregate import data_role, employer_concentration, experience_level, shares
from jobnlp.market.robustness import is_staffing, robustness, suppression_sensitivity, trend_test
from jobnlp.market.run import extract_skills, market_postings
from jobnlp.market.run import run as run_market
from jobnlp.market.structure import binary_matrix, pmi_edges, role_profiles


@pytest.fixture(scope="module")
def market_slice(corpus_df: pd.DataFrame, extractor):  # type: ignore[no-untyped-def]
    p = market_postings(corpus_df, include_duplicates=True).sample(300, random_state=0)
    return p, extract_skills(p, extractor.gazetteer)


def test_shares_have_valid_intervals_and_suppression(market_slice) -> None:  # type: ignore[no-untyped-def]
    p, skills = market_slice
    table, suppressed = shares(p, skills, "state", min_group=30)
    assert (table.ci_low <= table.share).all() and (table.share <= table.ci_high).all()
    assert (table.postings_in_group >= 30).all()
    assert all(g["n_postings"] < 30 for g in suppressed)


def test_structure_and_robustness(market_slice) -> None:  # type: ignore[no-untyped-def]
    p, skills = market_slice
    edges = pmi_edges(skills, len(p), min_pair=3)
    assert not edges.empty and edges.pmi.is_monotonic_decreasing
    m, keys, top = binary_matrix(skills, top_n=40)
    prof = role_profiles(m, top, k=3, seeds=(0, 1))
    assert len(prof["profiles"]) == 3 and -1 <= prof["stability_mean_ari"] <= 1
    rob = robustness(p, skills, top=10)
    assert set(rob["variants"]) >= {"with_duplicates", "one_posting_per_employer"}
    sens = suppression_sensitivity(p[~p.is_dup], ["state"])
    assert sens["state"]["min_10"]["groups_kept"] >= sens["state"]["min_50"]["groups_kept"]
    trend = trend_test(p[~p.is_dup], skills, min_mentions=10, n_boot=50)
    assert trend["tested"] >= 0 and trend["flagged_bonferroni"] <= trend["flagged_uncorrected"]


def test_rules_on_real_titles(corpus_df: pd.DataFrame) -> None:
    titles = corpus_df.title.head(400).tolist()
    levels = {experience_level(t, "") for t in titles}
    assert levels <= {"entry", "senior", "experienced", "unspecified"} and "senior" in levels
    assert any(data_role(t) for t in corpus_df.title)
    assert employer_concentration(corpus_df.head(200))["postings"] == 200
    assert is_staffing("ARWA Personaldienstleistungen GmbH") and not is_staffing("Bosch")


def test_market_run_end_to_end(corpus_df: pd.DataFrame, extractor, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    small = corpus_df[corpus_df.in_germany].sample(250, random_state=1)
    out = tmp_path / "results" / "market"
    summary = run_market(small, extractor.gazetteer, out, min_group=20)
    assert summary["postings"] > 0
    assert summary["postings_with_skills"] <= summary["postings"]  # duplicates not counted
    assert (out / "skill_shares_by_overall.csv").exists() and (out / "robustness.json").exists()
    assert (out / "top_skills.png").exists()


@pytest.fixture(scope="module")
def labelled(corpus_df: pd.DataFrame) -> pd.DataFrame:
    df, info = label_table(corpus_df)
    df["split"] = group_split(df)
    df["n_words"] = [len(str(t).split()) for t in df.description]
    return df


def test_grouped_split_keeps_employers_apart(labelled: pd.DataFrame) -> None:
    tr, te = labelled[labelled.split == "train"], labelled[labelled.split == "test"]
    assert not (set(tr.employer) & set(te.employer)) - {""}
    assert not set(tr.dup_cluster) & set(te.dup_cluster)
    frac = labelled.split.value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8


def test_tfidf_baseline_beats_chance(labelled: pd.DataFrame) -> None:
    tr, va, te = (labelled[labelled.split == s] for s in ("train", "val", "test"))
    model = TfidfLR()
    model.fit(
        text_view(tr, "title+description"),
        tr.label.tolist(),
        text_view(va, "title+description"),
        va.label.tolist(),
    )
    classes = sorted(set(labelled.label))
    proba = model.predict_proba(text_view(te, "title+description"))
    proba = proba[:, [model.classes.index(c) for c in classes]]
    res = metrics(te.label.tolist(), proba, classes)
    assert res["macro_f1"] > 1 / len(classes)
    ci = bootstrap_ci(te.label.tolist(), proba, classes, n=100)
    assert ci["macro_f1_95ci"][0] <= res["macro_f1"] <= ci["macro_f1_95ci"][1]
    same = paired_bootstrap(te.label.tolist(), proba, proba, classes, n=50)
    assert same["mean_diff"] == 0.0
    b = breakdown(te, proba, classes)
    assert set(b) == {"by_language", "by_length_tercile"}


def test_merge_results_keeps_old_models(tmp_path: Path) -> None:
    import json

    path = tmp_path / "r.json"
    path.write_text(json.dumps({"models": {"a": 1}, "n": 1}), encoding="utf-8")
    merged = merge_results(path, {"models": {"b": 2}, "n": 2})
    assert merged["models"] == {"a": 1, "b": 2} and merged["n"] == 2


def test_breakdown_terciles_cover_test_set(labelled: pd.DataFrame) -> None:
    te = labelled[labelled.split == "test"]
    classes = sorted(set(labelled.label))
    proba = np.eye(len(classes))[[classes.index(y) for y in te.label]]
    b = breakdown(te, proba, classes)
    assert sum(v["n"] for v in b["by_length_tercile"].values()) == len(te)
    assert all(v["accuracy"] == 1.0 for v in b["by_language"].values() if v["n"])
