"""Frequentist and shrinkage disproportionality estimators.

Four families are computed for every table, because the central finding of this
project is that *they disagree about when a signal starts*:

``PRR``
    Proportional reporting ratio. The MHRA/EMA workhorse. Compared against the
    reporting rate in the rest of the database.
``ROR``
    Reporting odds ratio. The EudraVigilance workhorse; the measure that admits
    a Mantel-Haenszel stratified form, so it is the one prodrome adjusts for
    confounding (see :mod:`prodrome.stats.stratified`).
``chi-squared``
    Yates-corrected, used only as the significance gate in the MHRA triple.
``shrinkage log O/E``
    The observed-to-expected ratio with an additive shrinkage prior, after
    Noren et al. (2006). This is the measure that behaves sanely at `a = 1`,
    where PRR and ROR are numerically enormous and epistemically worthless.

Confidence intervals are Wald intervals on the log scale, which is standard in
this literature and is what the published signal-detection criteria were
calibrated against. They are *not* exact, and for very small `a` the shrinkage
measure should be preferred -- prodrome enforces this by making minimum-count
thresholds part of each criterion rather than a global filter.

References
----------
Evans, Waller & Davis (2001) *Pharmacoepidemiol Drug Saf* 10:483-486 -- PRR
  and the a>=3 / PRR>=2 / chi2>=4 triple.
Rothman, Lanes & Sacks (2004) *Pharmacoepidemiol Drug Saf* 13:519-523 -- ROR.
Noren, Bate, Orre & Edwards (2006) *Stat Med* 25:3740-3757 -- shrinkage O/E.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy import stats as sps

from prodrome.stats.contingency import Contingency

#: Additive shrinkage applied to both observed and expected counts in the
#: log O/E measure. 0.5 is the value used by the Uppsala Monitoring Centre and
#: reduces a single spontaneous report from "infinitely disproportionate" to
#: "barely distinguishable from chance", which is the correct reading of it.
SHRINKAGE_ALPHA = 0.5

#: Two-sided 95% normal quantile. Named so the 1.96 is never a bare literal.
Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence or credibility interval."""

    lower: float
    upper: float

    def excludes(self, value: float) -> bool:
        """True when `value` lies outside the interval."""
        return value < self.lower or value > self.upper


@dataclass(frozen=True, slots=True)
class Disproportionality:
    """Every estimator prodrome computes for one 2x2 table.

    Attributes are plain floats so the whole record round-trips through
    DuckDB and Arrow without adapters.
    """

    a: int
    b: int
    c: int
    d: int
    n: int
    expected: float

    prr: float
    prr_ci: Interval
    ror: float
    ror_ci: Interval
    chi2_yates: float
    rrr: float
    log2_oe_shrunk: float
    log2_oe_ci: Interval

    #: True when a margin was empty, so the log-scale estimators are NaN rather
    #: than merely imprecise. Downstream code must not treat NaN as "no signal".
    degenerate: bool

    def as_row(self) -> dict[str, object]:
        """Flatten to a warehouse row, unnesting the intervals."""
        row: dict[str, object] = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict) and set(value) == {"lower", "upper"}:
                row[f"{key}_lower"] = value["lower"]
                row[f"{key}_upper"] = value["upper"]
            else:
                row[key] = value
        return row


def _log_wald(point: float, log_variance: float) -> Interval:
    """Wald interval built on the log scale and exponentiated back."""
    if point <= 0 or not math.isfinite(point) or log_variance < 0:
        return Interval(math.nan, math.nan)
    half = Z_95 * math.sqrt(log_variance)
    return Interval(math.exp(math.log(point) - half), math.exp(math.log(point) + half))


def proportional_reporting_ratio(t: Contingency) -> tuple[float, Interval]:
    """PRR with its 95% Wald interval.

    ``PRR = [a / (a+b)] / [c / (c+d)]``; the log variance is
    ``1/a - 1/(a+b) + 1/c - 1/(c+d)``.
    """
    if t.a == 0 or t.c == 0 or t.drug_total == 0 or (t.c + t.d) == 0:
        return math.nan, Interval(math.nan, math.nan)
    prr = (t.a / t.drug_total) / (t.c / (t.c + t.d))
    var = 1 / t.a - 1 / t.drug_total + 1 / t.c - 1 / (t.c + t.d)
    return prr, _log_wald(prr, var)


def reporting_odds_ratio(t: Contingency) -> tuple[float, Interval]:
    """ROR with its 95% Wald interval.

    ``ROR = ad / bc``; the log variance is ``1/a + 1/b + 1/c + 1/d``.
    """
    if 0 in (t.a, t.b, t.c, t.d):
        return math.nan, Interval(math.nan, math.nan)
    ror = (t.a * t.d) / (t.b * t.c)
    var = 1 / t.a + 1 / t.b + 1 / t.c + 1 / t.d
    return ror, _log_wald(ror, var)


def chi2_yates(t: Contingency) -> float:
    """Yates continuity-corrected chi-squared on one degree of freedom.

    Returns 0.0 rather than NaN on a degenerate table: a table with an empty
    margin carries no evidence of association, and 0 is the value that makes
    the MHRA gate fail closed.
    """
    row1, row2 = t.drug_total, t.c + t.d
    col1, col2 = t.reaction_total, t.b + t.d
    if 0 in (row1, row2, col1, col2):
        return 0.0
    numerator = t.n * max(abs(t.a * t.d - t.b * t.c) - t.n / 2, 0.0) ** 2
    return numerator / (row1 * row2 * col1 * col2)


def relative_reporting_ratio(t: Contingency) -> float:
    """``RRR = a / E``, the unshrunk observed-to-expected ratio."""
    expected = t.expected
    return t.a / expected if expected > 0 else math.nan


def log2_oe_shrunk(t: Contingency, alpha: float = SHRINKAGE_ALPHA) -> tuple[float, Interval]:
    """Shrinkage log2 observed-to-expected ratio with a credibility interval.

    The point estimate is ``log2((a + alpha) / (E + alpha))``. The interval comes
    from the Gamma(a + alpha, rate = E + alpha) posterior on the reporting rate
    ratio, taken to base 2 -- so unlike the Wald intervals above it stays finite
    and honest when `a` is 0 or 1. This is the measure to quote for rare pairs.
    """
    expected = t.expected
    if expected <= 0:
        return math.nan, Interval(math.nan, math.nan)
    point = math.log2((t.a + alpha) / (expected + alpha))
    posterior = sps.gamma(a=t.a + alpha, scale=1.0 / (expected + alpha))
    lo, hi = posterior.ppf(0.025), posterior.ppf(0.975)
    interval = Interval(
        math.log2(lo) if lo > 0 else -math.inf,
        math.log2(hi) if hi > 0 else math.nan,
    )
    return point, interval


def score(t: Contingency) -> Disproportionality:
    """Compute every estimator for one table.

    This is the single entry point the pipeline uses; adding an estimator means
    adding it here and to :data:`prodrome.stats.criteria.CRITERIA`, so a new
    measure can never silently fail to be evaluated.
    """
    prr, prr_ci = proportional_reporting_ratio(t)
    ror, ror_ci = reporting_odds_ratio(t)
    oe, oe_ci = log2_oe_shrunk(t)
    return Disproportionality(
        a=t.a,
        b=t.b,
        c=t.c,
        d=t.d,
        n=t.n,
        expected=t.expected,
        prr=prr,
        prr_ci=prr_ci,
        ror=ror,
        ror_ci=ror_ci,
        chi2_yates=chi2_yates(t),
        rrr=relative_reporting_ratio(t),
        log2_oe_shrunk=oe,
        log2_oe_ci=oe_ci,
        degenerate=t.has_empty_margin,
    )
