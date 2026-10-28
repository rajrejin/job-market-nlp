"""Publication dates as employers write them in JSON-LD, with the answers worked out by hand."""

from __future__ import annotations

import pytest

from jobnlp.text.corpus import parse_published


@pytest.mark.parametrize(
    ("raw", "iso"),
    [
        ("2025-09-14", "2025-09-14"),
        ("2025-09-14T08:00:00+02:00", "2025-09-14"),
        ("Mar 01, 2025", "2025-03-01"),
        ("Sep 13, 2025", "2025-09-13"),
        ("14. September 2025", "2025-09-14"),
        ("24. August 2025", "2025-08-24"),
        ("3. März 2025", "2025-03-03"),
        ("24.08.2025", "2025-08-24"),
    ],
)
def test_known_forms_become_iso(raw: str, iso: str) -> None:
    assert parse_published(raw) == iso


@pytest.mark.parametrize("raw", [None, "", "soon", "31.02.2025", "Foo 01, 2025"])
def test_unknown_or_impossible_dates_become_none(raw: object) -> None:
    assert parse_published(raw) is None
