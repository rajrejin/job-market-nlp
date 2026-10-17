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
