"""Quantifying the contamination that point-in-time analysis avoids.

The claim this module tests
---------------------------
Published FAERS disproportionality analyses -- and every open implementation found
while designing this project -- compute their statistics on the *full cumulative*
database as it stands at analysis time. For describing a known association that is
fine. For evaluating a signal-detection method it is circular, because of a
well-documented reporting phenomenon:

    When a reaction is added to a drug's label, or covered in the press, reporting
    of that reaction for that drug rises sharply. Clinicians who now know to look
    for it, report it.

This is **notoriety bias**, sometimes the Weber effect. Its consequence for
methodology is specific and severe: the label change *causes* part of the
disproportionality that a retrospective analysis then presents as evidence the
signal was detectable. The outcome leaks into the predictor.

Measured on semaglutide and ileus during design, using cumulative counts at each
quarter cutoff:

===========  ====  =====  ======
cutoff          a    PRR    chi2
===========  ====  =====  ======
2022-12-31     17   1.68     4.0
2023-06-30     22   1.76     6.5
2023-09-30     25   1.72     6.8
*label change: 2023-10-09*
2024-12-31    160   6.92   792.5
===========  ====  =====  ======

The retrospective PRR of 6.92 is four times the value available before the label
changed, and the chi-squared is two orders of magnitude larger. A retrospective
analysis would report this as a strong signal and implicitly credit the method with
finding it. Point-in-time, the MHRA triple never fired before the label change at
all, and only the EMA interval criterion did -- around 2022 H1.

So prodrome computes every statistic point-in-time, and computes the retrospective
value *as well*, purely to report the ratio between them. The ratio is the
measurement: it says how much of the apparent signal in a conventional analysis
arrived after, and because of, the outcome being predicted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prodrome.timeframe import Quarter


@dataclass(frozen=True, slots=True)
class LeakageComparison:
    """Point-in-time versus retrospective statistics for one pair.

    Attributes:
        statistic: which measure is being compared, e.g. ``"prr"``.
        at_signal_onset: value using only data available when the signal fired.
        at_label_change: value using only data available when the label changed.
        retrospective: value on the full cumulative database.
        inflation_ratio: ``retrospective / at_label_change``. Above 1 means the
            conventional analysis is reading a statistic the label change helped
            create.
    """

    statistic: str
    at_signal_onset: float | None
    at_label_change: float | None
    retrospective: float
    inflation_ratio: float | None

    def as_row(self) -> dict[str, object]:
        return {
            "statistic": self.statistic,
            "at_signal_onset": self.at_signal_onset,
            "at_label_change": self.at_label_change,
            "retrospective": self.retrospective,
            "inflation_ratio": self.inflation_ratio,
        }


def compare_leakage(
    statistic: str,
    series: Mapping[Quarter, float],
    *,
    signal_quarter: Quarter | None,
    label_quarter: Quarter | None,
) -> LeakageComparison:
    """Compare a statistic's point-in-time and retrospective values.

    Args:
        statistic: name of the measure.
        series: quarter -> value, point-in-time (each value computed on data
            available up to that quarter).
        signal_quarter: quarter the signal fired, if it did.
        label_quarter: quarter the label changed, if it did.

    The retrospective value is taken as the last quarter in the series, which by
    construction uses the whole cumulative database -- exactly what a conventional
    analysis would report.
    """
    if not series:
        return LeakageComparison(statistic, None, None, math.nan, None)

    quarters = sorted(series)
    retrospective = series[quarters[-1]]

    def value_at(quarter: Quarter | None) -> float | None:
        if quarter is None:
            return None
        # The value as of the last quarter at or before the target, so a pair whose
        # event falls between two evaluated quarters is scored on information that
        # genuinely preceded it.
        eligible = [q for q in quarters if q <= quarter]
        return series[eligible[-1]] if eligible else None

    at_signal = value_at(signal_quarter)
    at_label = value_at(label_quarter)
    inflation = (
        retrospective / at_label
        if at_label is not None and at_label > 0 and math.isfinite(retrospective)
        else None
    )
    return LeakageComparison(statistic, at_signal, at_label, retrospective, inflation)


@dataclass(frozen=True, slots=True)
class NotorietyMeasurement:
    """Reporting volume around a label change, which is notoriety bias made visible.

    Attributes:
        reports_before: new reports in the `window_quarters` before the change.
        reports_after: new reports in the `window_quarters` after it.
        surge_ratio: after/before. A value well above 1 is the signature.
        window_quarters: half-width of the comparison window.
    """

    reports_before: int
    reports_after: int
    surge_ratio: float | None
    window_quarters: int

    @property
    def is_notorious(self) -> bool:
        """Whether reporting more than doubled after the label change.

        The 2.0 bar is a reporting convention, not a test. It marks pairs whose
        retrospective statistics should be read with particular suspicion.
        """
        return self.surge_ratio is not None and self.surge_ratio >= 2.0

    def as_row(self) -> dict[str, object]:
        return {
            "reports_before_label": self.reports_before,
            "reports_after_label": self.reports_after,
            "notoriety_surge_ratio": self.surge_ratio,
            "notoriety_window_quarters": self.window_quarters,
            "is_notorious": self.is_notorious,
        }


def measure_notoriety(
    new_reports_by_quarter: Mapping[str, int],
    *,
    label_quarter: Quarter,
    window_quarters: int = 4,
) -> NotorietyMeasurement:
    """Compare reporting volume before and after a label change.

    Args:
        new_reports_by_quarter: quarter label -> *new* reports that quarter. New
            rather than cumulative: cumulative counts only ever rise, so they
            cannot show a surge.
        label_quarter: the quarter the label changed.
        window_quarters: quarters either side to compare.
    """
    if window_quarters < 1:
        raise ValueError(f"window_quarters must be at least 1, got {window_quarters}")

    before_labels = {label_quarter.shift(-i).label for i in range(1, window_quarters + 1)}
    after_labels = {label_quarter.shift(i).label for i in range(1, window_quarters + 1)}

    before = sum(count for q, count in new_reports_by_quarter.items() if q in before_labels)
    after = sum(count for q, count in new_reports_by_quarter.items() if q in after_labels)
    ratio = after / before if before > 0 else None
    return NotorietyMeasurement(before, after, ratio, window_quarters)


@dataclass(frozen=True, slots=True)
class LeakageSummary:
    """Cohort-level summary: how much a retrospective analysis would overstate."""

    n_pairs: int
    n_with_inflation: int
    median_inflation: float | None
    p90_inflation: float | None
    #: Pairs that meet a signal criterion retrospectively but never met it
    #: point-in-time before the label changed. These are the pure artefacts: a
    #: conventional analysis counts them as detections it did not make.
    n_retrospective_only: int
    n_notorious: int

    @property
    def retrospective_only_share(self) -> float:
        return self.n_retrospective_only / self.n_pairs if self.n_pairs else 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "n_with_inflation": self.n_with_inflation,
            "median_inflation": self.median_inflation,
            "p90_inflation": self.p90_inflation,
            "n_retrospective_only": self.n_retrospective_only,
            "retrospective_only_share": round(self.retrospective_only_share, 4),
            "n_notorious": self.n_notorious,
        }


def summarise_leakage(
    inflations: Sequence[float],
    *,
    n_pairs: int,
    n_retrospective_only: int,
    n_notorious: int,
) -> LeakageSummary:
    """Aggregate per-pair inflation ratios into a cohort summary."""
    import numpy as np

    usable = np.array([v for v in inflations if v is not None and math.isfinite(v)], dtype=float)
    return LeakageSummary(
        n_pairs=n_pairs,
        n_with_inflation=int(usable.size),
        median_inflation=float(np.median(usable)) if usable.size else None,
        p90_inflation=float(np.percentile(usable, 90)) if usable.size else None,
        n_retrospective_only=n_retrospective_only,
        n_notorious=n_notorious,
    )
