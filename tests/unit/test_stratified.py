"""Tests for Mantel-Haenszel pooling.

The estimator is checked against a constructed confounded dataset where the
stratum-specific odds ratio is known by design, so the correct adjusted answer
is known too -- and differs from the crude answer.
"""

from __future__ import annotations

import math

import pytest

from prodrome.stats.contingency import Contingency
from prodrome.stats.stratified import mantel_haenszel_ror


def test_recovers_common_odds_ratio_under_confounding() -> None:
    """Two strata each with OR ~= 2, but exposure prevalence differing sharply.

    Collapsing them produces a crude OR well below 2. Mantel-Haenszel must
    recover the common stratum-specific value, because that is the estimand.
    """
    strata = [
        Contingency(a=20, b=80, c=11, d=89),
        Contingency(a=80, b=20, c=67, d=33),
    ]
    result = mantel_haenszel_ror(strata)
    assert result.ror_mh == pytest.approx(2.0, rel=0.05)
    assert result.ror_crude < 1.7, "fixture must actually be confounded"
    assert result.ror_mh_ci.lower < result.ror_mh < result.ror_mh_ci.upper
    assert result.n_strata == result.n_strata_informative == 2
    # Negative confounding: the crude estimate understates the true effect.
    assert result.confounding_ratio < 0.9


def test_no_confounding_leaves_the_estimate_alone() -> None:
    """When exposure prevalence is constant across strata there is nothing to
    adjust for, so crude and adjusted must agree."""
    strata = [
        Contingency(a=20, b=80, c=11, d=89),
        Contingency(a=20, b=80, c=11, d=89),
    ]
    result = mantel_haenszel_ror(strata)
    assert result.confounding_ratio == pytest.approx(1.0, rel=0.02)


def test_uninformative_strata_are_skipped_not_fatal() -> None:
    """A stratum with an empty margin carries no odds-ratio information.

    It must be excluded from the estimate while still being counted, so a
    reviewer can see how much of the data was unusable.
    """
    strata = [
        Contingency(a=20, b=80, c=11, d=89),
        Contingency(a=0, b=0, c=5, d=5),
        Contingency(a=80, b=20, c=67, d=33),
    ]
    result = mantel_haenszel_ror(strata)
    assert result.n_strata == 3
    assert result.n_strata_informative == 2
    assert result.ror_mh == pytest.approx(2.0, rel=0.05)


def test_all_strata_uninformative_returns_nan_not_zero() -> None:
    """NaN means "no information"; 0.0 would read as "no association"."""
    result = mantel_haenszel_ror([Contingency(a=0, b=0, c=1, d=1)])
    assert math.isnan(result.ror_mh)
    assert result.n_strata_informative == 0


def test_requires_at_least_one_stratum() -> None:
    with pytest.raises(ValueError, match="at least one stratum"):
        mantel_haenszel_ror([])
