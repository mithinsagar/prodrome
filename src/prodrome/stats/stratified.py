"""Mantel-Haenszel stratified reporting odds ratio.

Why this is here
----------------
A crude ROR treats the whole database as one exchangeable population, which it
emphatically is not. Three structural confounders dominate FAERS:

*age and sex*
    Reporting rates for whole reaction classes differ sharply by both, and drug
    populations are not age- or sex-balanced. An oncology drug's apparent
    association with, say, fractures partly reflects who takes it.
*report year*
    Total report volume has grown by more than an order of magnitude, and the
    reaction mix has shifted with it. Pooling years compares a drug's early
    reports against a different era's background.
*reporter type and country*
    Consumer-sourced and litigation-sourced reports have a completely different
    reaction profile from clinician reports.

Stratifying on these and pooling with Mantel-Haenszel weights removes the part
of the association that is explained by stratum membership. The residual is a
better -- never a perfect -- estimate of the drug-specific effect.

prodrome reports crude and adjusted side by side, plus the ratio between them,
because *the gap is itself a diagnostic*: a signal that vanishes on adjustment
was a confounding artefact, and that is exactly the kind of false alarm a
signal-management team wants filtered before a human reads it.

The estimator uses the Robins-Breslow-Greenland variance, which is consistent
both when strata are few and large and when they are many and sparse -- the
sparse regime is the normal one here, since stratifying by year x sex x age band
produces many thin strata.

References
----------
Mantel & Haenszel (1959) *J Natl Cancer Inst* 22:719-748.
Robins, Breslow & Greenland (1986) *Biometrics* 42:311-323.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from prodrome.stats.contingency import Contingency
from prodrome.stats.disproportionality import Z_95, Interval


@dataclass(frozen=True, slots=True)
class StratifiedResult:
    """A pooled adjusted estimate and the diagnostics that qualify it."""

    ror_mh: float
    ror_mh_ci: Interval
    n_strata: int
    n_strata_informative: int
    total_a: int

    #: Crude ROR on the collapsed table, for the confounding comparison.
    ror_crude: float

    @property
    def confounding_ratio(self) -> float:
        """``crude / adjusted``.

        Above 1 means the crude estimate was inflated by stratum composition.
        Conventionally a departure of more than ~10% from 1.0 is treated as
        material confounding; prodrome records the number and lets the dashboard
        threshold it rather than hard-coding a rule.
        """
        if not math.isfinite(self.ror_mh) or self.ror_mh == 0:
            return math.nan
        return self.ror_crude / self.ror_mh


def mantel_haenszel_ror(strata: Sequence[Contingency]) -> StratifiedResult:
    """Pool per-stratum 2x2 tables into an adjusted ROR.

    Args:
        strata: one :class:`Contingency` per stratum. Strata with an empty
            margin contribute nothing to either the estimate or the variance and
            are counted but skipped -- dropping them is correct, not a
            convenience, because they carry no information about the odds ratio.

    Returns:
        The pooled estimate. ``ror_mh`` is NaN when no stratum is informative,
        which callers must distinguish from "no association".

    Raises:
        ValueError: if `strata` is empty.
    """
    if not strata:
        raise ValueError("at least one stratum is required")

    numerator = denominator = 0.0
    # Robins-Breslow-Greenland accumulators.
    pr_sum = pr_ps_sum = ps_qs_sum = 0.0
    informative = 0
    total_a = 0

    for t in strata:
        total_a += t.a
        if t.n == 0 or (0 in (t.b, t.c) and 0 in (t.a, t.d)):
            continue
        n = t.n
        r = t.a * t.d / n
        s = t.b * t.c / n
        if r == 0 and s == 0:
            continue
        informative += 1
        numerator += r
        denominator += s
        p = (t.a + t.d) / n
        q = (t.b + t.c) / n
        pr_sum += p * r
        pr_ps_sum += p * s + q * r
        ps_qs_sum += q * s

    collapsed = Contingency(
        a=sum(t.a for t in strata),
        b=sum(t.b for t in strata),
        c=sum(t.c for t in strata),
        d=sum(t.d for t in strata),
    )
    crude = (
        (collapsed.a * collapsed.d) / (collapsed.b * collapsed.c)
        if 0 not in (collapsed.b, collapsed.c)
        else math.nan
    )

    if informative == 0 or denominator == 0 or numerator == 0:
        return StratifiedResult(
            ror_mh=math.nan,
            ror_mh_ci=Interval(math.nan, math.nan),
            n_strata=len(strata),
            n_strata_informative=informative,
            total_a=total_a,
            ror_crude=crude,
        )

    ror = numerator / denominator
    log_var = (
        pr_sum / (2 * numerator**2)
        + pr_ps_sum / (2 * numerator * denominator)
        + ps_qs_sum / (2 * denominator**2)
    )
    half = Z_95 * math.sqrt(log_var) if log_var > 0 else math.nan
    ci = (
        Interval(math.exp(math.log(ror) - half), math.exp(math.log(ror) + half))
        if math.isfinite(half)
        else Interval(math.nan, math.nan)
    )
    return StratifiedResult(
        ror_mh=ror,
        ror_mh_ci=ci,
        n_strata=len(strata),
        n_strata_informative=informative,
        total_a=total_a,
        ror_crude=crude,
    )
