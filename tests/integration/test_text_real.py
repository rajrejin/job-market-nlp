"""Text pipeline checks on real harvested postings."""

from __future__ import annotations

import re

import pandas as pd

from jobnlp.text.clean import html_to_text
from jobnlp.text.redact import has_contact_data, redact


def test_clean_removes_all_markup(raw_payloads: list[dict]) -> None:  # type: ignore[type-arg]
    for p in raw_payloads:
        text = html_to_text(p["description"])
        assert not re.search(r"</?(div|p|li|ul|strong|br|span)\b", text)
        assert "&lt;" not in text and "&amp;" not in text


def test_redaction_leaves_no_contact_patterns(raw_payloads: list[dict]) -> None:  # type: ignore[type-arg]
    for p in raw_payloads:
        residual = has_contact_data(redact(html_to_text(p["description"])))
        assert residual == {"email": 0, "url": 0, "phone": 0, "salutation_name": 0}


def test_processed_corpus_has_no_contact_patterns(corpus: pd.DataFrame) -> None:
    for t in corpus["text"]:
        assert sum(has_contact_data(t).values()) == 0


def test_raw_text_really_contained_contact_data(raw_payloads: list[dict]) -> None:  # type: ignore[type-arg]
    """Guards against a vacuous redaction test: the raw sample must contain some contact data."""
    found = sum(
        sum(has_contact_data(html_to_text(p["description"])).values()) for p in raw_payloads
    )
    assert found > 0


def test_most_postings_have_requirements_or_tasks(corpus: pd.DataFrame) -> None:
    de = corpus[corpus.in_germany]
    has = de["section_labels"].apply(lambda s: "requirements" in s or "tasks" in s)
    assert has.mean() > 0.6


def test_no_bare_contact_name_next_to_role_line(corpus: pd.DataFrame) -> None:
    """Real staffing-agency postings put the contact name on its own line next to the role
    ("Personaldisponentin"); after redaction no such line may remain."""
    from jobnlp.text.redact import ROLE_LINE, plausible_name

    sample = corpus[corpus.source == "crawl"].sample(400, random_state=0)
    for text in sample.text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            bare = line.strip().removeprefix("## ").strip()
            if plausible_name(bare) and len(bare.split()) <= 3:
                neighbours = [lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
                assert not any(ROLE_LINE.search(n) for n in neighbours), "contact name left"
