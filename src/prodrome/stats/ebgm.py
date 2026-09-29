"""Gamma-Poisson shrinkage (GPS/MGPS) -- DuMouchel's empirical Bayes estimator.

This is the method behind Empirica Signal, the tool FDA's own safety reviewers
use, and it is the most defensible measure in this package. Unlike PRR and ROR
it does not blow up on rare pairs, because the amount of shrinkage applied to a
cell is *learned from the whole database* rather than fixed a priori.

The model
---------
For each drug-reaction cell with observed count `n` and expected count `E`::

    n | E, lambda  ~  Poisson(lambda * E)
    lambda         ~  P * Gamma(a1, b1)  +  (1 - P) * Gamma(a2, b2)

`lambda` is the reporting rate ratio -- the quantity PRR and ROR estimate
directly and noisily. Integrating the Poisson against the gamma mixture gives a
mixture of two negative binomials for `n | E`, which has a closed form, so the
five prior parameters are fitted by maximum likelihood over every cell at once::

    n | E  ~  P * NB(a1, b1/(b1+E))  +  (1 - P) * NB(a2, b2/(b2+E))

The two components are what make this work on spontaneous-report data: one
absorbs the enormous mass of cells that are pure noise, the other describes the
genuinely elevated ones. A single-gamma prior cannot do both, and over-shrinks
real signals toward 1.

Reported quantities
-------------------
``EBGM``
    ``exp(E[log lambda])`` under the posterior. The shrunk point estimate. Read
    it as "the observed-to-expected ratio, after discounting for how much of it
    the database says is probably noise".
``EB05``
    5th percentile of the posterior. The conventional signalling threshold is
    ``EB05 >= 2``; it is the most conservative rule in common use and almost
    never produces a false alarm on a small count.

Why prodrome fits this per quarter
----------------------------------
The prior is estimated from the database as it stood at each quarter cutoff, not
once on the final snapshot. Fitting it once on all data and applying it
retrospectively would leak later reporting behaviour into earlier estimates,
which is exactly the contamination this project exists to measure.

Reference
---------
DuMouchel (1999) *The American Statistician* 53:177-190.
DuMouchel & Pregibon (2001) KDD '01 -- the multi-item extension.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

logger = logging.getLogger(__name__)

#: Cells below this expected count carry essentially no information and
#: destabilise the likelihood surface. They are still *scored* -- they are just
#: excluded from the sample the prior is fitted on.
MIN_EXPECTED_FOR_FIT = 1e-6

#: Starting point for the optimiser: DuMouchel's worked-example values, stated
#: in the component order this module sorts to (component 1 = lower prior mean,
#: so the start is self-consistent with :func:`fit_prior`'s output convention).
#: The likelihood surface is multimodal, so a literature-standard start matters
#: more here than it would for a convex problem.
DEFAULT_START = (2.0, 4.0, 0.2, 0.1, 2.0 / 3.0)


@dataclass(frozen=True, slots=True)
class GpsPrior:
    """A fitted five-parameter gamma mixture prior.

    Components are ordered so that component 1 has the smaller prior mean; this
    makes the fit reproducible and lets ``noise_weight`` mean what it says.
    """

    a1: float
    b1: float
    a2: float
    b2: float
    p: float
    log_likelihood: float
    n_cells: int
    converged: bool

    @property
    def noise_weight(self) -> float:
        """Prior mass on the lower component -- the share of cells the database
        itself says are indistinguishable from background reporting."""
        return self.p

    def as_row(self) -> dict[str, float | int | bool]:
        return {
            "a1": self.a1,
            "b1": self.b1,
            "a2": self.a2,
            "b2": self.b2,
            "p": self.p,
            "log_likelihood": self.log_likelihood,
            "n_cells": self.n_cells,
            "converged": self.converged,
        }


@dataclass(frozen=True, slots=True)
class EbgmResult:
    """Posterior summaries for a batch of cells, aligned to the input order."""

    ebgm: np.ndarray
    eb05: np.ndarray
    eb95: np.ndarray
    posterior_weight: np.ndarray
    prior: GpsPrior


def _unpack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
    """Map the unconstrained optimiser vector onto the constrained parameters."""
    a1, b1, a2, b2 = np.exp(theta[:4])
    p = 1.0 / (1.0 + math.exp(-theta[4]))
    return float(a1), float(b1), float(a2), float(b2), float(p)


def _pack(a1: float, b1: float, a2: float, b2: float, p: float) -> np.ndarray:
    return np.array(
        [math.log(a1), math.log(b1), math.log(a2), math.log(b2), math.log(p / (1.0 - p))]
    )


def _component_log_pmf(
    n: np.ndarray, expected: np.ndarray, shape: float, rate: float
) -> np.ndarray:
    """Log pmf of one negative-binomial mixture component.

    ``NB(r = shape, prob = rate / (rate + E))`` is the Poisson-gamma marginal.
    Computed through :func:`scipy.stats.nbinom.logpmf` rather than by hand so the
    gamma functions stay stable for large `n`.
    """
    prob = rate / (rate + expected)
    return np.asarray(stats.nbinom.logpmf(n, shape, prob), dtype=float)


def _neg_log_likelihood(theta: np.ndarray, n: np.ndarray, expected: np.ndarray) -> float:
    a1, b1, a2, b2, p = _unpack(theta)
    log_p1 = _component_log_pmf(n, expected, a1, b1) + math.log(p)
    log_p2 = _component_log_pmf(n, expected, a2, b2) + math.log1p(-p)
    total = np.logaddexp(log_p1, log_p2)
    if not np.all(np.isfinite(total)):
        return float(np.inf)
    return float(-total.sum())


def fit_prior(
    counts: np.ndarray,
    expected: np.ndarray,
    *,
    start: tuple[float, float, float, float, float] = DEFAULT_START,
    max_iter: int = 500,
) -> GpsPrior:
    """Fit the gamma mixture prior by maximum likelihood over all cells.

    Args:
        counts: observed co-occurrence counts, one per cell.
        expected: independence-model expected counts, same length.

    Returns:
        The fitted prior. ``converged`` is False when the optimiser gave up, in
        which case the parameters are still returned -- callers should record the
        flag rather than silently trusting the numbers. The pipeline writes it to
        the warehouse and the dashboard surfaces it.

    Raises:
        ValueError: if the inputs disagree in length or no cell is usable.
    """
    counts = np.asarray(counts, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if counts.shape != expected.shape:
        raise ValueError(f"counts {counts.shape} and expected {expected.shape} must align")
    usable = np.isfinite(counts) & np.isfinite(expected) & (expected > MIN_EXPECTED_FOR_FIT)
    if not usable.any():
        raise ValueError("no cell has a usable expected count; cannot fit a prior")
    n_fit, e_fit = counts[usable], expected[usable]

    result = optimize.minimize(
        _neg_log_likelihood,
        _pack(*start),
        args=(n_fit, e_fit),
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    a1, b1, a2, b2, p = _unpack(result.x)
    # Order components by prior mean so the fit is identifiable and
    # `noise_weight` refers to the lower component in every run.
    if (a1 / b1) > (a2 / b2):
        a1, b1, a2, b2, p = a2, b2, a1, b1, 1.0 - p
    if not result.success:
        logger.warning("GPS prior did not converge: %s", result.message)
    return GpsPrior(
        a1=a1,
        b1=b1,
        a2=a2,
        b2=b2,
        p=p,
        log_likelihood=float(-result.fun),
        n_cells=int(usable.sum()),
        converged=bool(result.success),
    )


def _posterior_weight(n: np.ndarray, expected: np.ndarray, prior: GpsPrior) -> np.ndarray:
    """P(cell came from the lower component | n, E)."""
    log_p1 = _component_log_pmf(n, expected, prior.a1, prior.b1) + math.log(prior.p)
    log_p2 = _component_log_pmf(n, expected, prior.a2, prior.b2) + math.log1p(-prior.p)
    return np.exp(log_p1 - np.logaddexp(log_p1, log_p2))


def _mixture_quantile(
    q: float,
    weight: float,
    shape1: float,
    rate1: float,
    shape2: float,
    rate2: float,
) -> float:
    """Quantile of a two-component gamma mixture posterior, by root finding.

    A mixture has no closed-form quantile. The CDF is monotone, so Brent's
    method on ``CDF(x) - q`` is exact to tolerance; the bracket is widened from
    the component quantiles, which always contain the mixture quantile.
    """

    def cdf(x: float) -> float:
        mixture = weight * stats.gamma.cdf(x, shape1, scale=1.0 / rate1) + (
            1.0 - weight
        ) * stats.gamma.cdf(x, shape2, scale=1.0 / rate2)
        return float(mixture) - q

    lo = min(
        stats.gamma.ppf(q, shape1, scale=1.0 / rate1), stats.gamma.ppf(q, shape2, scale=1.0 / rate2)
    )
    hi = max(
        stats.gamma.ppf(q, shape1, scale=1.0 / rate1), stats.gamma.ppf(q, shape2, scale=1.0 / rate2)
    )
    if not math.isfinite(lo) or lo <= 0:
        lo = 1e-12
    if not math.isfinite(hi) or hi <= lo:
        return float(lo)
    # Brent needs a sign change; nudge the bracket outward if the mixture
    # quantile sits marginally outside the component quantiles.
    lo, hi = lo * 0.5, hi * 2.0
    if cdf(lo) > 0 or cdf(hi) < 0:
        return float(np.clip((lo + hi) / 2, 0.0, None))
    return float(optimize.brentq(cdf, lo, hi, xtol=1e-10, rtol=1e-10))


def score_cells(
    counts: np.ndarray,
    expected: np.ndarray,
    *,
    prior: GpsPrior | None = None,
) -> EbgmResult:
    """Compute EBGM, EB05 and EB95 for every cell.

    Args:
        counts: observed counts.
        expected: expected counts.
        prior: a prior fitted elsewhere -- pass the *point-in-time* prior for
            that quarter. When omitted the prior is fitted on `counts` and
            `expected` themselves, which is correct only if that array is
            already the point-in-time snapshot.
    """
    counts = np.asarray(counts, dtype=float)
    expected = np.asarray(expected, dtype=float)
    fitted = prior if prior is not None else fit_prior(counts, expected)

    weight = _posterior_weight(counts, expected, fitted)
    shape1, rate1 = fitted.a1 + counts, fitted.b1 + expected
    shape2, rate2 = fitted.a2 + counts, fitted.b2 + expected

    # E[log lambda] under a gamma is digamma(shape) - log(rate); the mixture
    # expectation is the posterior-weighted average of the two.
    e_log = weight * (special.digamma(shape1) - np.log(rate1)) + (1.0 - weight) * (
        special.digamma(shape2) - np.log(rate2)
    )
    ebgm = np.exp(e_log)

    eb05 = np.empty_like(ebgm)
    eb95 = np.empty_like(ebgm)
    for i in range(ebgm.size):
        eb05[i] = _mixture_quantile(0.05, weight[i], shape1[i], rate1[i], shape2[i], rate2[i])
        eb95[i] = _mixture_quantile(0.95, weight[i], shape1[i], rate1[i], shape2[i], rate2[i])

    return EbgmResult(ebgm=ebgm, eb05=eb05, eb95=eb95, posterior_weight=weight, prior=fitted)
