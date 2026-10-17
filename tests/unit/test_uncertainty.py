import pytest

from jobnlp.market.uncertainty import wilson_interval


def test_wilson_half_of_ten() -> None:
    # hand computation: centre 0.5, half = 1.96*sqrt(0.025+0.0096)/1.3842 = 0.2634
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_rejects_empty() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
