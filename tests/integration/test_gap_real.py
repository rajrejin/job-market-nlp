"""CV gap analyser: parsing, matching, scoring, guarded suggestions, evaluation helpers.

CVs are personal data and none is used: real posting text stands in for CV text, and the LLM is
replaced by scripted replies so the guard logic is tested deterministically."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from jobnlp.gap.evaluate import guard_eval, sanity_auc
from jobnlp.gap.llm import LocalLLM, Suggestion, build_prompt, guarded_suggestion
from jobnlp.gap.match import analyse, importance
from jobnlp.gap.parse_cv import CVError, bullets, read_cv, split_cv
from jobnlp.gap.related import tune_related_threshold
from jobnlp.gap.report import run_gap


@pytest.fixture(scope="module")
def two_postings(corpus_df: pd.DataFrame) -> tuple[str, str]:
    p = corpus_df[
        corpus_df.in_germany
        & ~corpus_df.is_dup
        & (corpus_df.lang == "de")
        & (corpus_df.n_chars > 1500)
    ].sample(2, random_state=0)
    return p.text.iloc[0], p.text.iloc[1]


def test_read_cv_txt_and_docx(two_postings) -> None:  # type: ignore[no-untyped-def]
    from docx import Document

    text = two_postings[0]
    assert read_cv(text.encode("utf-8"), "cv.txt") == text.strip()
    doc = Document()
    for line in text.split("\n")[:30]:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    parsed = read_cv(buf.getvalue(), "cv.docx")
    assert text.split("\n")[0] in parsed


def test_read_cv_rejects_bad_input() -> None:
    with pytest.raises(CVError):
        read_cv(b"x", "cv.exe")
    with pytest.raises(CVError):
        read_cv(b"%PDF-1.4 broken", "cv.pdf")
    with pytest.raises(CVError):
        read_cv(b"x" * 3_000_000, "cv.txt")


def test_cv_section_rules() -> None:
    secs = split_cv("Berufserfahrung\nData Analyst 2020\nAusbildung\nMSc\nSkills:\nPython")
    assert [s.label for s in secs] == ["experience", "education", "skills"]
    assert bullets("- built dashboards in Tableau for sales\n- x") == [
        "built dashboards in Tableau for sales"
    ]


def test_posting_against_itself_and_another(extractor, two_postings) -> None:  # type: ignore[no-untyped-def]
    a, b = two_postings
    same = analyse(a, a, extractor)
    assert same.score is None or same.score == 1.0
    other = analyse(a, b, extractor)
    assert other.score is None or other.score <= 1.0
    if other.missing:
        ranked = [s.importance for s in other.missing]
        assert ranked == sorted(ranked, reverse=True)
    assert all(importance(s) == s.importance for s in other.skills)


class PairEmbedder:
    """Similarity 1.0 for equal lower-cased strings, 0.95 when one contains the other, else 0.
    A stand-in for e5 to test the 'related' branch deterministically."""

    def similarity(self, a: list[str], b: list[str]) -> list[list[float]]:
        def s(x: str, y: str) -> float:
            x, y = x.lower(), y.lower()
            return 1.0 if x == y else 0.95 if (x in y or y in x) else 0.0

        return [[s(x, y) for y in b] for x in a]


def test_related_status_from_embedder(extractor) -> None:  # type: ignore[no-untyped-def]
    job = "Sehr gute Kenntnisse in Projektmanagement und SQL"
    cv = "Erfahrung im agilen Projektmanagement mit Scrum"
    res = analyse(job, cv, extractor, PairEmbedder(), related_threshold=0.9)
    assert {s.status for s in res.skills} <= {"covered", "related", "missing"}


def scripted(replies: list[dict | str]):  # type: ignore[no-untyped-def,type-arg]
    it = iter(replies)
    return lambda prompt: (lambda r: r if isinstance(r, str) else json.dumps(r))(next(it))


def test_guard_accepts_truthful_rewrite_and_rejects_invented_skill(extractor) -> None:  # type: ignore[no-untyped-def]
    keys = lambda t: {m.skill_id for m in extractor.extract(t, "de", use_ner=False)}  # noqa: E731
    bullets_ = ["Erstellung von Berichten in Excel für den Vertrieb"]
    good = {
        "evidence_found": True,
        "bullet_index": 0,
        "rewrite": "Datenanalyse und Berichte in Excel für den Vertrieb",
        "suggestion": None,
    }
    bad = {
        "evidence_found": True,
        "bullet_index": 0,
        "rewrite": "Berichte mit Python, Docker und Kubernetes für 5000 Kunden",
        "suggestion": None,
    }
    target = next(iter(keys("Datenanalyse")), "x")
    s = guarded_suggestion(
        scripted([good]),
        target,
        "Datenanalyse",
        "Datenanalyse",
        "Datenanalyse",
        bullets_,
        keys(bullets_[0]),
        keys,
    )
    assert s.guard_passed and s.attempts == 1
    s2 = guarded_suggestion(
        scripted([bad, "not json", bad]),
        target,
        "Datenanalyse",
        "Datenanalyse",
        "Datenanalyse",
        bullets_,
        keys(bullets_[0]),
        keys,
    )
    assert not s2.guard_passed and s2.fallback and s2.rewrite is None
    assert any("unsupported skill" in v for v in s2.history[0])
    assert any("number" in v for v in s2.history[0])
    assert "JSON" in s2.history[1][0]


def test_prompt_and_schema() -> None:
    p = build_prompt("SQL", ["SQL ist ein Muss"], ["Berichte in Excel"], ["unsupported skill: x"])
    assert "0: Berichte in Excel" in p and "unsupported skill" in p
    assert Suggestion.model_validate({"evidence_found": False, "suggestion": "Kurs"}).suggestion


def test_llm_unavailable_falls_back(extractor, two_postings) -> None:  # type: ignore[no-untyped-def]
    llm = LocalLLM(host="http://127.0.0.1:9")
    assert not llm.available()
    res = run_gap(two_postings[0], two_postings[1], extractor, with_embeddings=False, llm=llm)
    assert res["suggestions"] == [] and any("LLM unavailable" in n for n in res["notes"])
    assert "heuristic" in res["disclaimer"]


class ScriptedLLM:
    model = "scripted"

    def available(self) -> bool:
        return True

    def ask(self, prompt: str) -> str:
        return json.dumps({"evidence_found": False, "suggestion": "Ein kleines Projekt umsetzen"})


def test_run_gap_with_scripted_llm(extractor, two_postings) -> None:  # type: ignore[no-untyped-def]
    res = run_gap(
        two_postings[0],
        two_postings[1],
        extractor,
        with_embeddings=False,
        llm=ScriptedLLM(),
        max_suggestions=2,
    )  # type: ignore[arg-type]
    assert all(s["guard_passed"] for s in res["suggestions"])


def test_evaluation_helpers_on_real_postings(extractor, corpus_df: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    from jobnlp.classify.data import label_table
    from jobnlp.skills.pipeline import requirement_text

    labelled, _ = label_table(corpus_df)
    res = sanity_auc(labelled.sample(80, random_state=0), extractor, None, n_pairs=12, n_self=4)
    assert 0 <= res["auc_same_vs_different"] <= 1 and res["self_match_mean"] == 1.0
    pool = corpus_df[corpus_df.in_germany & ~corpus_df.is_dup].head(300)
    pool = pool.assign(req=[requirement_text(s) for s in pool.sections])
    g = guard_eval(pool, extractor, ScriptedLLM(), n=3)  # type: ignore[arg-type]
    assert g["n_cases"] <= 3 and g["final_pass_rate"] == 1.0


def test_related_threshold_on_dev_pairs(gold_items, extractor, esco_df) -> None:  # type: ignore[no-untyped-def]
    from jobnlp.skills.pipeline import canon_for

    dev = [g for g in gold_items if g.split == "dev"][:30]
    res = tune_related_threshold(dev, canon_for(extractor.gazetteer), PairEmbedder(), esco_df)
    assert 0.7 <= res["threshold"] <= 0.99
