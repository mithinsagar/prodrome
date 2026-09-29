"""Tests for the gamma-Poisson shrinker.

The estimator has no closed-form answer to assert against, so it is tested three
ways: parameter recovery from a known generating process, the ordering
invariants the posterior must satisfy, and the shrinkage behaviour that is the
entire reason to prefer it over PRR.
"""

from __future__ import annotations

import numpy as np
import pytest

from prodrome.stats.ebgm import fit_prior, score_cells

# Generating process, written in the component order fit_prior sorts to:
# component 1 is the lower-mean ("noise") component.
TRUE_A1, TRUE_B1 = 2.0, 4.0  # prior mean 0.50
TRUE_A2, TRUE_B2 = 0.2, 0.1  # prior mean 2.00
TRUE_P = 0.80


@pytest.fixture(scope="module")
def simulated() -> tuple[np.ndarray, np.ndarray]:
    """8,000 cells drawn from the GPS model with expected counts spanning
    four orders of magnitude, as a real FAERS quarter does."""
    rng = np.random.default_rng(20240101)
    n_cells = 8000
    expected = np.exp(rng.normal(0.0, 1.8, n_cells)).clip(1e-3, None)
    from_low = rng.random(n_cells) < TRUE_P
    lam = np.where(
        from_low,
        rng.gamma(TRUE_A1, 1 / TRUE_B1, n_cells),
        rng.gamma(TRUE_A2, 1 / TRUE_B2, n_cells),
    )
    return rng.poisson(lam * expected), expected


def test_recovers_prior_component_means(simulated: tuple[np.ndarray, np.ndarray]) -> None:
    """The component *means* are the identifiable part of a gamma mixture.

    Shape and rate are only weakly identified individually -- they trade off
    along a ridge -- so the means and the mixing weight are what a correct
    implementation must recover, and what the posterior actually depends on.
    """
    counts, expected = simulated
    prior = fit_prior(counts, expected)
    assert prior.converged
    assert prior.a1 / prior.b1 == pytest.approx(TRUE_A1 / TRUE_B1, rel=0.20)
    assert prior.a2 / prior.b2 == pytest.approx(TRUE_A2 / TRUE_B2, rel=0.20)
    assert prior.p == pytest.approx(TRUE_P, abs=0.10)


def test_components_are_ordered_by_prior_mean(simulated: tuple[np.ndarray, np.ndarray]) -> None:
    """Ordering is what makes the fit reproducible and `noise_weight` meaningful."""
    counts, expected = simulated
    prior = fit_prior(counts, expected)
    assert prior.a1 / prior.b1 <= prior.a2 / prior.b2
    assert prior.noise_weight == prior.p


def test_posterior_intervals_are_ordered(simulated: tuple[np.ndarray, np.ndarray]) -> None:
    counts, expected = simulated
    result = score_cells(counts, expected)
    assert np.all(result.eb05 > 0.0)
    assert np.all(result.eb05 <= result.ebgm + 1e-8)
    assert np.all(result.ebgm <= result.eb95 + 1e-8)
    assert np.all((result.posterior_weight >= 0.0) & (result.posterior_weight <= 1.0))


def test_large_counts_are_barely_shrunk(simulated: tuple[np.ndarray, np.ndarray]) -> None:
    """With plenty of data the prior should stop mattering."""
    counts, expected = simulated
    result = score_cells(counts, expected)
    rrr = counts / expected
    plentiful = counts >= 50
    assert plentiful.sum() > 20, "fixture no longer exercises the large-count regime"
    assert np.allclose(result.ebgm[plentiful], rrr[plentiful], rtol=0.10)


def test_rare_cells_shrink_toward_the_prior(simulated: tuple[np.ndarray, np.ndarray]) -> None:
    """The behaviour PRR cannot provide.

    A single report against a tiny expected count yields an enormous raw ratio.
    Shrinkage must pull it down, and the 5th percentile must stay below the
    conventional EB05 >= 2 signalling threshold -- otherwise the estimator would
    manufacture exactly the false alarms it exists to prevent.
    """
    counts, expected = simulated
    result = score_cells(counts, expected)
    rrr = counts / expected
    singleton = (counts == 1) & (expected < 0.05)
    assert singleton.sum() > 10, "fixture no longer exercises the rare-cell regime"
    assert np.all(result.ebgm[singleton] < rrr[singleton])
    assert np.all(result.eb05[singleton] < 2.0)


def test_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError, match="must align"):
        fit_prior(np.array([1, 2, 3]), np.array([1.0, 2.0]))


def test_rejects_input_with_no_usable_cell() -> None:
    with pytest.raises(ValueError, match="no cell has a usable expected count"):
        fit_prior(np.array([0, 0]), np.array([0.0, 0.0]))
