"""Empirical calibration of signal scores against negative controls.

The problem
-----------
Every disproportionality estimator assumes its own null distribution: that under
no association, log(ROR) is centred on 0 with variance given by the Wald formula.
In spontaneous-report data this is false. Confounding by indication, reporting
channel effects, event co-reporting and duplicate records all push the null off
centre and widen it. The practical consequence is that a nominal p < 0.05 does
not mean a 5% false-positive rate -- in observational pharmacovigilance it is
routinely far worse.

The fix
-------
Estimate the null *empirically*. Take a set of drug-reaction pairs believed not
to be causally associated, run them through the identical pipeline, and observe
where their estimates actually land. Fit a systematic-error distribution to that
spread, then re-express every real estimate as a p-value against the *observed*
null rather than the theoretical one.

Model: for a negative control with true effect 1, the observed log estimate is::

    log(RR_obs)  ~  Normal(mu, sd^2 + se_obs^2)

`mu` is the systematic bias and `sd` the between-pair heterogeneity of that
bias; both are fitted by maximum likelihood across the controls, with each
control's own standard error entering as known measurement error. A calibrated
two-sided p-value for a target estimate follows directly.

prodrome reports raw and calibrated p-values side by side. The calibration is a
property of a *quarter*, refitted each time, so drift in FAERS reporting
behaviour shows up as drift in `mu` -- which is itself worth plotting.

Reference
---------
Schuemie, Ryan, DuMouchel, Suchard & Madigan (2014) *Stat Med* 33:209-218,
"Interpreting observational studies: why empirical calibration is needed".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True, slots=True)
class NullDistribution:
    """A fitted empirical null.

    Attributes:
        mu: systematic bias on the log scale. 0 would mean the theoretical null
            was right; in practice it is positive, because disproportionality
            methods are biased toward finding associations.
        sd: between-control heterogeneity of that bias, on the log scale.
        n_controls: how many controls the fit used. Below ~20 the fit is too
            unstable to trust, and :func:`fit_null` says so.
        converged: optimiser status.
    """

    mu: float
    sd: float
    n_controls: int
    converged: bool

    @property
    def systematic_bias_ratio(self) -> float:
        """`mu` expressed as a ratio, which is how to report it to a human:
        "the median negative control comes out at ROR = 1.4, not 1.0"."""
        return math.exp(self.mu)

    def calibrated_p(self, log_estimate: float, log_se: float) -> float:
        """Two-sided p-value against the empirical null.

        Args:
            log_estimate: the target's point estimate on the log scale.
            log_se: its standard error on the log scale.
        """
        if not all(math.isfinite(v) for v in (log_estimate, log_se)):
            return math.nan
        scale = math.sqrt(self.sd**2 + log_se**2)
        if scale <= 0:
            return math.nan
        z = (log_estimate - self.mu) / scale
        return float(2.0 * stats.norm.sf(abs(z)))

    def calibrated_interval(self, log_estimate: float, log_se: float) -> tuple[float, float]:
        """95% interval recentred and rewidened by the empirical null.

        Returned on the *ratio* scale so it is directly comparable to the
        uncalibrated ROR interval.
        """
        if not all(math.isfinite(v) for v in (log_estimate, log_se)):
            return math.nan, math.nan
        scale = math.sqrt(self.sd**2 + log_se**2)
        centre = log_estimate - self.mu
        half = 1.959963984540054 * scale
        return math.exp(centre - half), math.exp(centre + half)


def _neg_log_likelihood(theta: np.ndarray, estimates: np.ndarray, ses: np.ndarray) -> float:
    mu, log_sd = float(theta[0]), float(theta[1])
    sd = math.exp(log_sd)
    var = sd**2 + ses**2
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return float(np.inf)
    ll = -0.5 * np.log(2 * np.pi * var) - (estimates - mu) ** 2 / (2 * var)
    return float(-ll.sum())


class InsufficientControlsError(ValueError):
    """Too few usable negative controls to fit a null.

    Raised rather than returning a degenerate null, because silently falling back
    to the theoretical null would defeat the entire purpose of calibrating.
    """


#: Below this, the two-parameter fit is dominated by sampling noise. Schuemie et
#: al. use on the order of 50; prodrome accepts 20 and records the count so a
#: reader can discount accordingly.
MIN_CONTROLS = 20


def fit_null(
    log_estimates: Sequence[float],
    log_ses: Sequence[float],
    *,
    min_controls: int = MIN_CONTROLS,
) -> NullDistribution:
    """Fit the empirical null from negative-control estimates.

    Args:
        log_estimates: one log-scale point estimate per negative control.
        log_ses: the matching log-scale standard errors.

    Raises:
        InsufficientControlsError: when fewer than `min_controls` controls have
            finite estimate and standard error.
    """
    est = np.asarray(log_estimates, dtype=float)
    ses = np.asarray(log_ses, dtype=float)
    if est.shape != ses.shape:
        raise ValueError("estimates and standard errors must align")
    usable = np.isfinite(est) & np.isfinite(ses) & (ses > 0)
    if int(usable.sum()) < min_controls:
        raise InsufficientControlsError(
            f"{int(usable.sum())} usable negative controls, need at least {min_controls}"
        )
    est, ses = est[usable], ses[usable]

    result = optimize.minimize(
        _neg_log_likelihood,
        np.array([float(np.median(est)), math.log(max(float(np.std(est)), 1e-3))]),
        args=(est, ses),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8},
    )
    return NullDistribution(
        mu=float(result.x[0]),
        sd=float(math.exp(result.x[1])),
        n_controls=int(est.size),
        converged=bool(result.success),
    )
