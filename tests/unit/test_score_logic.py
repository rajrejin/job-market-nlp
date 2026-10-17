import pytest

from jobnlp.gap.score import match_score


def test_score_formula_hand_computed() -> None:
    # must covered (1*1) + must missing (1*0) + nice related (0.4*0.5) = 1.2 over 2.4 = 0.5
    score, rows = match_score(
        [
            ("a", "A", "must", "covered"),
            ("b", "B", "must", "missing"),
            ("c", "C", "nice", "related"),
        ]
    )
    assert score == pytest.approx(0.5)
    assert [r.weight for r in rows] == [1.0, 1.0, 0.4]


def test_score_none_without_requirements() -> None:
    assert match_score([])[0] is None


def test_all_covered_is_one() -> None:
    assert match_score([("a", "A", "nice", "covered")])[0] == 1.0
