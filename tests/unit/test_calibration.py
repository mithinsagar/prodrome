"""Tests for empirical calibration against negative controls."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from prodrome.stats.calibration import (
    InsufficientControlsError,
    fit_null,
)


@pytest.fixture
def biased_controls() -> tuple[np.ndarray, np.ndarray]:
    """400 negative controls whose null is biased upward.

    Standard errors are kept small relative to the between-control spread so
    that `sd` is identifiable; when the standard errors dominate, `sd` is only
    weakly identified and the fit will shrink it. That is a property of the
    model, not a defect, and it is documented in METHODS.md.
    """
    rng = np.random.default_rng(424242)
    n = 400
    ses = rng.uniform(0.05, 0.15, n)
    est = rng.normal(0.35, np.sqrt(0.25**2 + ses**2))
    return est, ses


def test_recovers_systematic_bias(biased_controls: tuple[np.ndarray, np.ndarray]) -> None:
    est, ses = biased_controls
    null = fit_null(est, ses)
    assert null.converged
    assert null.mu == pytest.approx(0.35, abs=0.06)
    assert null.sd == pytest.approx(0.25, rel=0.20)
    assert null.systematic_bias_ratio == pytest.approx(math.exp(null.mu))


def test_calibration_is_more_conservative_than_the_theoretical_null(
    biased_controls: tuple[np.ndarray, np.ndarray],
) -> None:
    """The whole point: an apparently significant estimate should survive
    calibration only if it stands out against where the controls actually sit."""
    est, ses = biased_controls
    null = fit_null(est, ses)
    log_estimate, log_se = 0.80, 0.30
    raw_p = float(2 * stats.norm.sf(abs(log_estimate) / log_se))
    calibrated_p = null.calibrated_p(log_estimate, log_se)
    assert raw_p < 0.01, "fixture should look significant before calibration"
    assert calibrated_p > raw_p
    assert calibrated_p > 0.05, "a modest estimate should not survive a biased null"


def test_a_genuinely_large_estimate_still_survives(
    biased_controls: tuple[np.ndarray, np.ndarray],
) -> None:
    """Calibration must not be a blanket suppressor."""
    est, ses = biased_controls
    null = fit_null(est, ses)
    assert null.calibrated_p(3.0, 0.20) < 0.001


def test_calibrated_interval_is_recentred_and_wider(
    biased_controls: tuple[np.ndarray, np.ndarray],
) -> None:
    est, ses = biased_controls
    null = fit_null(est, ses)
    lower, upper = null.calibrated_interval(0.80, 0.30)
    raw_half = 1.959963984540054 * 0.30
    assert math.log(upper) - math.log(lower) > 2 * raw_half
    assert lower < math.exp(0.80 - null.mu) < upper


def test_too_few_controls_raises_rather_than_degrading_silently() -> None:
    with pytest.raises(InsufficientControlsError, match="need at least"):
        fit_null([0.1, 0.2, 0.3], [0.1, 0.1, 0.1])


def test_non_finite_inputs_are_excluded_from_the_fit() -> None:
    rng = np.random.default_rng(1)
    est = [*rng.normal(0.2, 0.3, 40), math.nan, math.inf]
    ses = [*rng.uniform(0.1, 0.2, 40), 0.1, 0.1]
    assert fit_null(est, ses).n_controls == 40


def test_calibrated_p_is_nan_for_unusable_target() -> None:
    rng = np.random.default_rng(2)
    null = fit_null(rng.normal(0.2, 0.3, 40), rng.uniform(0.1, 0.2, 40))
    assert math.isnan(null.calibrated_p(math.nan, 0.2))
