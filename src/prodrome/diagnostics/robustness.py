"""Is this signal real, or an artefact of who filed the reports?

Disproportionality does not distinguish a genuine pharmacological association from
a reporting pattern that merely looks like one. Four artefact mechanisms are
well described in the pharmacovigilance literature and all of them are visible in
the data prodrome already collects, so all four are measured and attached to every
signal. This is the layer that separates a queue a human can work from a list of
ratios.

**Reporter concentration.** A pair whose reports come overwhelmingly from one
country, or one quarter, is not showing a population-level association -- it is
showing a local reporting campaign, a single institution's audit, or a translation
artefact. Measured with a Herfindahl-Hirschman index over the reporting-country
distribution.

**Consumer and legal reporting.** ``primarysource.qualification`` distinguishes
physician (1), pharmacist (2), other health professional (3), lawyer (4) and
consumer (5) reports. Mass-tort litigation produces enormous volumes of
lawyer-sourced and consumer-sourced reports for a specific drug-event pair; the
resulting disproportionality is a fact about litigation, not about pharmacology.
A high lawyer share is the single most specific artefact marker available in FAERS.

**Volume spikes.** A signal built from one quarter's surge behaves differently from
one that accumulated steadily. The spike ratio compares the largest single
quarter's contribution to the median quarter's.

**Temporal stability.** A leave-one-quarter-out check: if removing a single quarter
drops the pair below its signalling threshold, the signal rests on that quarter
alone.

None of these are proof of artefact. They are the questions a reviewer would ask,
computed in advance so the reviewer can start from the answers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: openFDA's primarysource.qualification coding.
QUALIFICATION_LABELS: dict[str, str] = {
    "1": "physician",
    "2": "pharmacist",
    "3": "other health professional",
    "4": "lawyer",
    "5": "consumer or non-health professional",
}

#: Qualification codes that mark a report as arising outside clinical practice.
#: Lawyer-sourced reports are the strongest litigation marker FAERS offers.
NON_CLINICAL_QUALIFICATIONS = frozenset({"4", "5"})
LAWYER_QUALIFICATION = "4"


def herfindahl(counts: Sequence[int] | Mapping[str, int]) -> float:
    """Herfindahl-Hirschman concentration index, on 0..1.

    1.0 means every report came from a single category; a value near 0 means they
    are spread evenly across many. Chosen over "share of the largest category"
    because it responds to the whole distribution: two countries at 45% each is
    concentrated in a way that a top-share of 0.45 would understate.
    """
    values = list(counts.values()) if isinstance(counts, Mapping) else list(counts)
    total = sum(values)
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in values)


@dataclass(frozen=True, slots=True)
class RobustnessDiagnostics:
    """Artefact indicators for one drug-reaction pair."""

    reporter_concentration: float
    top_country: str | None
    top_country_share: float
    consumer_share: float
    lawyer_share: float
    spike_ratio: float
    spike_quarter: str | None
    quarters_with_reports: int
    #: Share of total reports removable by dropping the single largest quarter.
    single_quarter_dependence: float

    @property
    def is_geographically_concentrated(self) -> bool:
        """Whether one country dominates.

        0.5 corresponds to a single country supplying about 70% of reports. FAERS is
        a US database, so a US-dominated pair is unremarkable -- the flag matters
        when the dominant country is *not* the US, which the dashboard shows
        alongside it.
        """
        return self.reporter_concentration >= 0.5

    @property
    def is_litigation_shaped(self) -> bool:
        """Whether lawyer-sourced reporting is high enough to explain the signal.

        A 10% lawyer share is far above background: across FAERS as a whole,
        lawyer-sourced reports are a low single-digit percentage, so a pair at 10%
        is being driven by something other than clinical observation.
        """
        return self.lawyer_share >= 0.10

    @property
    def is_spike_driven(self) -> bool:
        """Whether one quarter supplies a disproportionate share of the reports."""
        return self.single_quarter_dependence >= 0.5

    @property
    def artefact_flags(self) -> tuple[str, ...]:
        """Machine-readable flags, for filtering and for the dashboard."""
        flags = []
        if self.is_geographically_concentrated:
            flags.append("geographic_concentration")
        if self.is_litigation_shaped:
            flags.append("litigation_pattern")
        if self.is_spike_driven:
            flags.append("single_quarter_spike")
        if self.consumer_share >= 0.60:
            flags.append("consumer_dominated")
        return tuple(flags)

    @property
    def robustness_score(self) -> float:
        """A 0..1 summary, 1 meaning no artefact indicator fired.

        A deliberately simple average of the four complements rather than a fitted
        weighting: the four mechanisms are not commensurable, nobody has ground
        truth for their relative importance, and a fitted weight would imply a
        precision this does not have. It is a sorting aid, and the component
        measures are always shown next to it.
        """
        penalties = [
            min(self.reporter_concentration, 1.0),
            min(self.lawyer_share * 5.0, 1.0),
            min(self.single_quarter_dependence, 1.0),
            min(max(self.consumer_share - 0.5, 0.0) * 2.0, 1.0),
        ]
        return 1.0 - sum(penalties) / len(penalties)

    def as_row(self) -> dict[str, object]:
        return {
            "reporter_concentration": round(self.reporter_concentration, 4),
            "top_country": self.top_country,
            "top_country_share": round(self.top_country_share, 4),
            "consumer_share": round(self.consumer_share, 4),
            "lawyer_share": round(self.lawyer_share, 4),
            "spike_ratio": round(self.spike_ratio, 3),
            "spike_quarter": self.spike_quarter,
            "quarters_with_reports": self.quarters_with_reports,
            "single_quarter_dependence": round(self.single_quarter_dependence, 4),
            "robustness_score": round(self.robustness_score, 4),
            "artefact_flags": ",".join(self.artefact_flags),
        }


def assess_robustness(
    country_counts: Mapping[str, int],
    qualification_counts: Mapping[str, int],
    quarterly_reports: Mapping[str, int],
) -> RobustnessDiagnostics:
    """Compute every artefact indicator for one pair.

    Args:
        country_counts: reporting country -> reports.
        qualification_counts: openFDA qualification code -> reports.
        quarterly_reports: quarter label -> *new* reports that quarter.
    """
    country_total = sum(country_counts.values())
    top_country, top_count = (
        max(country_counts.items(), key=lambda kv: kv[1]) if country_counts else (None, 0)
    )

    qualification_total = sum(qualification_counts.values())
    consumer = qualification_counts.get("5", 0)
    lawyer = qualification_counts.get(LAWYER_QUALIFICATION, 0)

    volumes = [v for v in quarterly_reports.values() if v > 0]
    report_total = sum(volumes)
    largest = max(volumes) if volumes else 0
    median = sorted(volumes)[len(volumes) // 2] if volumes else 0

    spike_quarter = (
        max(quarterly_reports.items(), key=lambda kv: kv[1])[0] if quarterly_reports else None
    )

    return RobustnessDiagnostics(
        reporter_concentration=herfindahl(country_counts),
        top_country=top_country,
        top_country_share=top_count / country_total if country_total else 0.0,
        consumer_share=consumer / qualification_total if qualification_total else 0.0,
        lawyer_share=lawyer / qualification_total if qualification_total else 0.0,
        spike_ratio=largest / median if median > 0 else (math.inf if largest else 0.0),
        spike_quarter=spike_quarter if largest else None,
        quarters_with_reports=len(volumes),
        single_quarter_dependence=largest / report_total if report_total else 0.0,
    )
