"""How prodrome identifies "reports about this drug", and why it is not obvious.

The finding that forced this module
----------------------------------
FAERS reports are free text from the field. openFDA attempts to harmonise each
reported drug against FDA's product database and, when it succeeds, attaches an
``openfda`` block carrying a UNII. Joining on that UNII is the precise, obviously
correct thing to do -- so this project did it first.

Measured against the live API, harmonisation coverage turns out to be *bimodal*:

==============  ==========  ========================  ==============
drug            by UNII     by active-substance name  UNII coverage
==============  ==========  ========================  ==============
pembrolizumab      104,614                   105,077          99.6%
semaglutide         73,001                   100,515          72.6%
osimertinib              0                    31,954           0.0%
esketamine               0                    18,640           0.0%
ubrogepant               0                     6,026           0.0%
==============  ==========  ========================  ==============

A UNII-only join silently discards osimertinib entirely -- 31,954 reports, none
of them joinable -- while reporting no error and producing a perfectly plausible
empty result. A name-only join, which is what most published FAERS analyses use,
is imprecise in the other direction and cannot distinguish a substance from a
brand that contains it.

The resolution
--------------
Identify a drug by the *union* of three exact-match clauses: its UNII, its active
substance names as coded by the reporter, and its brand names. Each component is
an exact match, so the union adds recall without adding fuzziness. The reporter's
``activesubstance.activesubstancename`` is the workhorse -- it is populated even
when openFDA's harmonisation failed completely.

Two properties make this safe rather than merely broader:

*The selector is pinned per drug in version-controlled configuration.* A selector
that changed between quarters would put a discontinuity into every time series
that looked exactly like a real signal.

*Coverage is measured and recorded.* Every result carries the share of its reports
that were UNII-harmonised, so a reader can see which drugs rest on name matching
and discount accordingly. That number is in the warehouse and on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

from prodrome.clients import query as q

FIELD_UNII = "patient.drug.openfda.unii.exact"
FIELD_SUBSTANCE = "patient.drug.activesubstance.activesubstancename.exact"
FIELD_BRAND = "patient.drug.medicinalproduct.exact"


@dataclass(frozen=True, slots=True)
class DrugSelector:
    """An openFDA search clause identifying every report about one drug.

    Attributes:
        unii: FDA substance identifier, used when openFDA harmonised the report.
        substance_names: active substance names as a reporter would code them.
            Uppercase, because the ``.exact`` index is case-sensitive.
        brand_names: brand names, as a further fallback.
    """

    unii: str | None
    substance_names: tuple[str, ...] = ()
    brand_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.unii and not self.substance_names and not self.brand_names:
            raise ValueError("a selector needs at least one of unii, substance or brand names")

    @property
    def clause(self) -> str:
        """The disjunctive search clause.

        Not wrapped in :func:`prodrome.clients.query.all_of`; callers combine it
        with a date window, and that function parenthesises each operand, which is
        what keeps this OR-group from binding loosely against a following AND.
        """
        parts: list[str] = []
        if self.unii:
            parts.append(q.field_is(FIELD_UNII, self.unii))
        if self.substance_names:
            parts.append(q.any_of(FIELD_SUBSTANCE, self.substance_names))
        if self.brand_names:
            parts.append(q.any_of(FIELD_BRAND, self.brand_names))
        return " OR ".join(parts)

    @property
    def unii_only_clause(self) -> str | None:
        """UNII clause alone, for measuring harmonisation coverage."""
        return q.field_is(FIELD_UNII, self.unii) if self.unii else None

    @property
    def components(self) -> tuple[str, ...]:
        """Which identification methods this selector uses, for the audit trail."""
        used = []
        if self.unii:
            used.append("unii")
        if self.substance_names:
            used.append("substance")
        if self.brand_names:
            used.append("brand")
        return tuple(used)


@dataclass(frozen=True, slots=True)
class SelectorCoverage:
    """Measured recall of each identification method for one drug.

    Recorded at cohort-resolution time and carried into the warehouse, so a
    published figure can be traced to how its population was identified.
    """

    union_reports: int
    unii_reports: int
    substance_reports: int
    brand_reports: int

    @property
    def unii_coverage(self) -> float:
        """Share of identified reports that openFDA harmonised to a UNII.

        1.0 means the obvious join would have worked; 0.0 means it would have
        returned nothing at all while looking perfectly healthy.
        """
        return self.unii_reports / self.union_reports if self.union_reports else 0.0

    @property
    def relies_on_names(self) -> bool:
        """True when name matching contributes materially to this drug's reports.

        The 0.95 bar is deliberately strict: below it, enough reports arrive
        through name matching that the drug's numbers should be read as
        name-identified rather than substance-identified.
        """
        return self.unii_coverage < 0.95

    def as_row(self) -> dict[str, float | int | bool]:
        return {
            "union_reports": self.union_reports,
            "unii_reports": self.unii_reports,
            "substance_reports": self.substance_reports,
            "brand_reports": self.brand_reports,
            "unii_coverage": round(self.unii_coverage, 4),
            "relies_on_names": self.relies_on_names,
        }
