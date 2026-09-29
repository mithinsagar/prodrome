"""When a signal started, when the label followed, and the gap between them.

This module computes the project's dependent variable, and almost all of its
difficulty is in what to *exclude*.

Left truncation, the trap that invalidates the naive analysis
------------------------------------------------------------
The obvious calculation is: find the first label version mentioning the reaction,
subtract the quarter the signal first fired, report the difference. Applied to
every pair, it is wrong.

DailyMed's archive for a drug begins at some version 1 -- for Ozempic, 2017-12-06.
If a reaction is *already* described in that first archived version, the date it
was added is unknown and unknowable from this data: it happened before the
observation window opened. Such a pair is **prevalent**, not incident. Including
it assigns it a label date of "the start of the archive", which is arbitrary, and
because these pairs are overwhelmingly the well-established reactions of older
drugs, including them systematically biases lead time toward zero.

Survival analysis has a name for this -- left truncation -- and the correct
handling is to exclude prevalent pairs from the time-to-event analysis while
still counting them, so the exclusion is visible. :class:`PairStatus` makes each
pair's disposition explicit rather than letting it be implied by a filter.

Right censoring
---------------
A pair whose signal has fired but whose label has not changed by the end of the
data is censored, not negative. It may yet be labelled. Treating censored pairs as
"never labelled" would understate the labelling rate and inflate apparent lead
times for the pairs that did get labelled.

Persistence
-----------
A criterion can fire in one quarter on a handful of reports and stop firing in the
next. Published work has proposed quarterly persistence as a prioritisation
dimension in its own right (see ``docs/METHODS.md``). prodrome records both the
first quarter a criterion fired and the first quarter it fired and *kept* firing
for a configurable number of consecutive quarters, so the lead time each
definition buys can be compared rather than assumed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from prodrome.timeframe import Quarter


@dataclass(frozen=True, slots=True)
class SignalOnset:
    """When one criterion first fired for one pair.

    Attributes:
        criterion: registry key from :data:`prodrome.stats.criteria.CRITERIA`.
        first_fired: earliest quarter the criterion held at all.
        first_persistent: earliest quarter beginning a run of consecutive firing
            quarters of at least the required length. None when no such run exists.
        quarters_fired: total quarters in which the criterion held.
        quarters_observed: quarters of data available for the pair.
    """

    criterion: str
    first_fired: Quarter | None
    first_persistent: Quarter | None
    quarters_fired: int
    quarters_observed: int

    @property
    def ever_fired(self) -> bool:
        return self.first_fired is not None

    @property
    def firing_fraction(self) -> float:
        """Share of observed quarters in which the criterion held.

        A low fraction alongside an early onset is the signature of an unstable
        signal: it fired once and lapsed. This is the number that separates
        "detected early" from "got lucky once".
        """
        return self.quarters_fired / self.quarters_observed if self.quarters_observed else 0.0


#: Consecutive firing quarters required for a signal to count as persistent.
DEFAULT_PERSISTENCE = 2


def signal_onsets(
    fired_by_quarter: Mapping[str, Mapping[Quarter, bool]],
    *,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, SignalOnset]:
    """Compute onset per criterion from a point-in-time firing series.

    Args:
        fired_by_quarter: criterion key -> {quarter -> whether it held}.
        persistence: consecutive quarters required for `first_persistent`.

    Raises:
        ValueError: if `persistence` is below 1.
    """
    if persistence < 1:
        raise ValueError(f"persistence must be at least 1, got {persistence}")

    results: dict[str, SignalOnset] = {}
    for criterion, series in fired_by_quarter.items():
        quarters = sorted(series)
        flags = [bool(series[q]) for q in quarters]
        first = next((q for q, f in zip(quarters, flags, strict=True) if f), None)

        first_persistent = None
        for index in range(len(flags) - persistence + 1):
            if all(flags[index : index + persistence]):
                first_persistent = quarters[index]
                break

        results[criterion] = SignalOnset(
            criterion=criterion,
            first_fired=first,
            first_persistent=first_persistent,
            quarters_fired=sum(flags),
            quarters_observed=len(flags),
        )
    return results


@dataclass(frozen=True, slots=True)
class LabelOnset:
    """When a label first described a reaction, and whether that date is knowable.

    Attributes:
        first_labelled_date: authoritative date of the earliest version mentioning
            the reaction, or None if no version does.
        present_at_baseline: True when the earliest *archived* version already
            mentioned it -- so the addition date precedes the observation window
            and the pair is left-truncated.
        baseline_date: date of the earliest archived version, which bounds what
            this data can see.
        latest_assessed_date: date of the most recent version assessed, which is
            where a censored pair is censored.
        versions_assessed: how many versions carried a usable verdict.
    """

    first_labelled_date: dt.date | None
    present_at_baseline: bool
    baseline_date: dt.date | None
    latest_assessed_date: dt.date | None
    versions_assessed: int

    @property
    def first_labelled_quarter(self) -> Quarter | None:
        if self.first_labelled_date is None:
            return None
        return Quarter.containing(self.first_labelled_date)

    @property
    def ever_labelled(self) -> bool:
        return self.first_labelled_date is not None


def label_onset(
    mentions: Sequence[tuple[dt.date, bool, bool]],
) -> LabelOnset:
    """Reduce a label-mention history to an onset.

    Args:
        mentions: one tuple per label version, ``(authoritative_date, is_labelled,
            is_assessable)``, in any order. ``is_assessable`` is False for a version
            whose document could not be parsed; those are skipped rather than read
            as "not labelled", which would otherwise fabricate a label gap or,
            worse, a spurious un-labelling.

    Returns:
        The onset. With no assessable version, every field is empty and the caller
        must treat the pair as unobservable rather than unlabelled.
    """
    assessable = sorted((date, labelled) for date, labelled, ok in mentions if ok)
    if not assessable:
        return LabelOnset(None, False, None, None, 0)

    baseline_date, baseline_labelled = assessable[0]
    first_labelled = next((date for date, labelled in assessable if labelled), None)
    return LabelOnset(
        first_labelled_date=first_labelled,
        present_at_baseline=baseline_labelled,
        baseline_date=baseline_date,
        latest_assessed_date=assessable[-1][0],
        versions_assessed=len(assessable),
    )


class PairStatus(StrEnum):
    """A pair's disposition for the time-to-label analysis.

    Every pair gets exactly one of these, and only ``LABELLED_AFTER_SIGNAL`` and
    ``CENSORED`` enter the survival model. The others are counted and reported, so
    the size of each exclusion is visible in the output rather than buried in a
    ``WHERE`` clause.
    """

    LABELLED_AFTER_SIGNAL = "labelled_after_signal"
    """The case of interest: signal fired, label followed. Contributes an event."""

    CENSORED = "censored"
    """Signal fired, no label change yet. Contributes censored time."""

    LABELLED_BEFORE_SIGNAL = "labelled_before_signal"
    """The label changed before the criterion fired. Excluded from time-to-event
    (the event precedes the origin) but retained as a measure of how often
    disproportionality *lags* regulatory action."""

    PREVALENT_AT_BASELINE = "prevalent_at_baseline"
    """Already labelled in the earliest archived version. Left-truncated: the
    addition date is not knowable from this data."""

    NO_SIGNAL = "no_signal"
    """The criterion never fired. No origin, so no survival time."""

    UNOBSERVABLE = "unobservable"
    """No assessable label version. Cannot contribute either way."""


@dataclass(frozen=True, slots=True)
class PairOutcome:
    """The full disposition of one (drug, reaction, criterion) triple."""

    criterion: str
    status: PairStatus
    signal_quarter: Quarter | None
    label_quarter: Quarter | None
    #: Quarters from signal onset to label change, or to censoring. Positive means
    #: the signal preceded the label.
    lead_time_quarters: int | None
    #: True when the observation ended without a label change.
    censored: bool
    firing_fraction: float

    @property
    def in_survival_set(self) -> bool:
        return self.status in (PairStatus.LABELLED_AFTER_SIGNAL, PairStatus.CENSORED)

    @property
    def is_event(self) -> bool:
        return self.status is PairStatus.LABELLED_AFTER_SIGNAL

    def as_row(self) -> dict[str, object]:
        return {
            "criterion": self.criterion,
            "status": self.status.value,
            "signal_quarter": self.signal_quarter.label if self.signal_quarter else None,
            "label_quarter": self.label_quarter.label if self.label_quarter else None,
            "lead_time_quarters": self.lead_time_quarters,
            "censored": self.censored,
            "in_survival_set": self.in_survival_set,
            "is_event": self.is_event,
            "firing_fraction": round(self.firing_fraction, 4),
        }


def pair_outcome(
    onset: SignalOnset,
    label: LabelOnset,
    *,
    observation_end: Quarter,
    use_persistent_onset: bool = False,
) -> PairOutcome:
    """Classify one pair and compute its lead time.

    Args:
        onset: the criterion's firing history for this pair.
        label: the pair's label-mention history.
        observation_end: last quarter with data, where censoring happens.
        use_persistent_onset: take the signal origin from `first_persistent`
            instead of `first_fired`. The stricter definition trades lead time for
            fewer one-quarter flukes; reporting both is the point.
    """
    if label.versions_assessed == 0:
        return PairOutcome(
            onset.criterion,
            PairStatus.UNOBSERVABLE,
            None,
            None,
            None,
            False,
            onset.firing_fraction,
        )

    # Prevalence is checked before the signal, because a pair already labelled at
    # baseline is left-truncated regardless of what the statistics later did.
    if label.present_at_baseline:
        return PairOutcome(
            onset.criterion,
            PairStatus.PREVALENT_AT_BASELINE,
            onset.first_persistent if use_persistent_onset else onset.first_fired,
            label.first_labelled_quarter,
            None,
            False,
            onset.firing_fraction,
        )

    signal_quarter = onset.first_persistent if use_persistent_onset else onset.first_fired
    if signal_quarter is None:
        return PairOutcome(
            onset.criterion,
            PairStatus.NO_SIGNAL,
            None,
            label.first_labelled_quarter,
            None,
            False,
            onset.firing_fraction,
        )

    label_quarter = label.first_labelled_quarter
    if label_quarter is None:
        return PairOutcome(
            onset.criterion,
            PairStatus.CENSORED,
            signal_quarter,
            None,
            observation_end - signal_quarter,
            True,
            onset.firing_fraction,
        )

    lead = label_quarter - signal_quarter
    if lead < 0:
        return PairOutcome(
            onset.criterion,
            PairStatus.LABELLED_BEFORE_SIGNAL,
            signal_quarter,
            label_quarter,
            lead,
            False,
            onset.firing_fraction,
        )
    return PairOutcome(
        onset.criterion,
        PairStatus.LABELLED_AFTER_SIGNAL,
        signal_quarter,
        label_quarter,
        lead,
        False,
        onset.firing_fraction,
    )
