"""Resolving drug names into a pinned, auditable cohort.

Why DailyMed is the primary source
----------------------------------
The obvious approach is to resolve identity through openFDA's label index, which
returns UNII and set id together. It was the first thing tried here, and it is
wrong in two ways that both silently corrupt the study population:

*It is incomplete.* Osimertinib, esketamine and ubrogepant return no label record
at all, under either brand or generic name. Resolving through it drops real drugs
from the cohort with no error.

*It surfaces the wrong set id.* Searching it for semaglutide yields the oral
tablet label -- 15 revisions -- and not the injection label, which has 19. The
injection label is the one carrying the 2023 ileus warning, so resolving through
openFDA would have removed this project's own validation case from its cohort.

DailyMed has neither problem: it is the authoritative SPL repository, it exposes
complete version history, and the SPL document itself carries the active
ingredients' UNIIs because FDA requires it. So identity and history both come from
DailyMed, and openFDA is consulted only to confirm a drug has enough adverse-event
volume to analyse.

Why the output is committed rather than recomputed
-------------------------------------------------
Two failure modes are invisible downstream. Picking a repackager's set id -- they
republish a label once and never revise it -- makes a drug appear never to have
had a label change, so every real label change becomes a false negative in the
outcome variable. Picking an excipient's UNII makes a drug appear to have no
adverse events. Neither raises an error, and no downstream check would catch
either. So the resolver ranks candidates and explains its choice, a human confirms
it once, and the result is version-controlled.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from prodrome.clients.dailymed import (
    DailyMedClient,
    active_ingredient_uniis,
    labeller_from_title,
)
from prodrome.clients.openfda import OpenFdaClient
from prodrome.selector import DrugSelector, SelectorCoverage

logger = logging.getLogger(__name__)

#: Labellers that republish somebody else's label without revising it. Matched
#: against the labeller in a DailyMed title, and used to *demote* a candidate
#: rather than exclude it -- for a few older products the only available label
#: genuinely is a relabeller's.
#: DailyMed titles are shaped "BRAND (GENERIC) DOSAGE FORM [LABELLER]". The
#: leading run of capitals before the parenthesis is the brand name.
_BRAND_IN_TITLE = re.compile(r"^([A-Z0-9][A-Z0-9 .,'/+-]*?)\s*\(")

_REPACKAGER = re.compile(
    r"\b(A-S MEDICATION|AIDAREX|PROFICIENT RX|QUALITY CARE|NUCARE|REDPHARM|"
    r"BRYANT RANCH|DIRECT[_ ]?RX|PD-RX|RPK PHARMACEUTICALS|ASCLEMED|"
    r"PREFERRED PHARMACEUTICALS|DENTON PHARMA|NORTHWIND|LAKE ERIE MEDICAL|"
    r"MEDSOURCE|CARDINAL HEALTH|MCKESSON|HF ACQUISITION|HENRY SCHEIN|"
    r"REMEDYREPACK|CLINICAL SOLUTIONS|ADVANCED RX|MODAVAR|BLUEPOINT|"
    r"GENERAL INJECTABLES|UNIT DOSE|SAFECOR|MAJOR PHARMACEUTICALS|"
    r"DOH CENTRAL PHARMACY|ST\.? MARY'S MEDICAL PARK)\b",
    re.IGNORECASE,
)

#: A drug with fewer revisions than this has too little label history to yield
#: either a usable positive or a meaningful censored observation.
MIN_LABEL_REVISIONS = 4

#: Below this, quarterly disproportionality is too unstable to be informative.
MIN_EVENT_REPORTS = 500


@dataclass(slots=True)
class ResolutionCandidate:
    """One DailyMed set id considered for a drug, with the evidence for it."""

    set_id: str
    title: str
    latest_version: int
    labeller: str = ""
    uniis: dict[str, str] = field(default_factory=dict)
    event_report_count: int = 0
    #: Which source answered the UNII lookup, recorded for the audit trail.
    unii_source: str = ""
    #: Measured recall of each identification method. See prodrome.selector.
    coverage: SelectorCoverage | None = None

    @property
    def looks_like_repackager(self) -> bool:
        return bool(_REPACKAGER.search(self.labeller or self.title))

    @property
    def is_combination(self) -> bool:
        return len(self.uniis) > 1

    @property
    def score(self) -> tuple[int, int]:
        """Ranking key.

        The latest version *number* dominates. It is a proxy for revision count,
        not the count itself -- DailyMed's search listing reports the current
        version number, and the two can differ because version numbers are not
        always contiguous. As a ranking signal the proxy is what matters: only an
        application holder accumulates versions at all, so a high number is strong
        evidence against a repackager. The exact revision count comes from
        ``history.json`` during the label backfill, where it is actually needed.
        """
        return (0 if self.looks_like_repackager else 1, self.latest_version)

    def primary_unii(self, wanted_name: str) -> tuple[str, str] | None:
        """Pick the UNII matching the requested drug name.

        A combination product carries several active UNIIs. Choosing by name match
        rather than by position is what keeps "sacubitril" from resolving to
        valsartan in the combination label.
        """
        if not self.uniis:
            return None
        target = wanted_name.strip().upper()
        for unii, name in self.uniis.items():
            upper = name.upper()
            if upper == target or target in upper or upper in target:
                return unii, name
        if len(self.uniis) == 1:
            return next(iter(self.uniis.items()))
        return None


@dataclass(slots=True)
class Resolution:
    """The outcome of resolving one requested drug name."""

    requested: str
    chosen: ResolutionCandidate | None = None
    unii: str | None = None
    substance_name: str | None = None
    selector: DrugSelector | None = None
    candidates: list[ResolutionCandidate] = field(default_factory=list)
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.chosen is not None and self.unii is not None and self.problem is None

    def to_cohort_entry(self) -> dict[str, object]:
        """The YAML fragment for ``conf/cohort.yml``."""
        if not self.ok or self.chosen is None or self.unii is None:
            raise ValueError(f"{self.requested} did not resolve")
        c = self.chosen
        selector = self.selector
        entry: dict[str, object] = {
            "unii": self.unii,
            "name": (self.substance_name or self.requested).title(),
            "spl_set_id": c.set_id,
        }
        if selector is not None:
            entry["substance_names"] = list(selector.substance_names)
            entry["brand_names"] = list(selector.brand_names)
        if c.coverage is not None:
            entry["unii_coverage"] = round(c.coverage.unii_coverage, 4)
        entry["notes"] = (
            f"{c.labeller or 'unknown labeller'}; SPL version {c.latest_version}; "
            f"{c.event_report_count:,} FAERS reports"
            + (
                f"; UNII harmonisation covers only "
                f"{c.coverage.unii_coverage:.0%} of them, so name matching carries "
                f"this drug"
                if c.coverage is not None and c.coverage.relies_on_names
                else ""
            )
            + (
                f"; combination product ({len(c.uniis)} active ingredients)"
                if c.is_combination
                else ""
            )
        )
        return entry


class CohortResolver:
    """Resolves drug names to reviewed cohort entries, DailyMed first."""

    def __init__(
        self,
        dailymed: DailyMedClient,
        openfda_events: OpenFdaClient,
        openfda_labels: OpenFdaClient | None = None,
        *,
        candidates_to_inspect: int = 3,
    ) -> None:
        self._dailymed = dailymed
        self._events = openfda_events
        #: Optional cheap source for UNII-by-set-id. When present it is tried
        #: first; the SPL archive is downloaded only when it comes back empty.
        self._labels = openfda_labels
        self._candidates_to_inspect = candidates_to_inspect

    def _uniis_for(self, candidate: ResolutionCandidate) -> dict[str, str]:
        """Active UNIIs for a candidate, cheapest source first."""
        if self._labels is not None:
            found = self._labels.active_uniis_by_set_id(candidate.set_id)
            if found:
                candidate.unii_source = "openfda"
                return found
        document = self._dailymed.fetch_version(candidate.set_id, candidate.latest_version)
        if document is None:
            return {}
        candidate.unii_source = "spl"
        return active_ingredient_uniis(document.xml)

    def resolve(self, drug_name: str) -> Resolution:
        """Resolve one drug name."""
        listings = self._dailymed.search_set_ids(drug_name, page_size=50)
        if not listings:
            return Resolution(drug_name, problem="no DailyMed label found for that name")

        candidates = [
            ResolutionCandidate(
                set_id=listing.set_id,
                title=listing.title,
                latest_version=listing.version_count,
                labeller=labeller_from_title(listing.title),
            )
            for listing in listings
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Inspecting a candidate costs a multi-megabyte archive download, so only
        # the few best are opened. The search listing's own version number is a
        # reliable enough pre-filter to make that safe.
        for candidate in candidates[: self._candidates_to_inspect]:
            candidate.uniis = self._uniis_for(candidate)

        inspected = [c for c in candidates[: self._candidates_to_inspect] if c.uniis]
        if not inspected:
            return Resolution(
                drug_name,
                candidates=candidates,
                problem="no candidate label yielded an active-ingredient UNII",
            )

        best = max(inspected, key=lambda c: c.score)
        picked = best.primary_unii(drug_name)
        if picked is None:
            return Resolution(
                drug_name,
                chosen=best,
                candidates=candidates,
                problem=(
                    f"label is a combination of {len(best.uniis)} actives and none "
                    f"matched the requested name; pin the UNII by hand"
                ),
            )
        unii, substance = picked

        # Build the union selector and measure what each component recalls. This
        # is the step that catches the drugs openFDA never harmonised: without it
        # osimertinib resolves to a valid label and zero adverse events.
        selector = DrugSelector(
            unii=unii,
            substance_names=_substance_name_variants(substance, drug_name),
            brand_names=_brand_name_variants(best.title),
        )
        best.coverage = self._events.measure_selector_coverage(selector)
        best.event_report_count = best.coverage.union_reports

        problem = None
        if best.latest_version < MIN_LABEL_REVISIONS:
            problem = (
                f"best set id is only at version {best.latest_version} "
                f"(need {MIN_LABEL_REVISIONS}); likely not the application holder's label"
            )
        elif best.event_report_count < MIN_EVENT_REPORTS:
            problem = (
                f"only {best.event_report_count:,} FAERS reports (need "
                f"{MIN_EVENT_REPORTS:,}) -- too sparse for quarterly analysis"
            )

        return Resolution(
            requested=drug_name,
            chosen=best,
            unii=unii,
            substance_name=substance,
            selector=selector,
            candidates=candidates,
            problem=problem,
        )

    def resolve_all(self, drug_names: list[str]) -> list[Resolution]:
        resolutions = []
        for name in drug_names:
            logger.info("resolving %s", name)
            try:
                resolutions.append(self.resolve(name))
            except Exception as exc:
                logger.exception("resolving %s failed", name)
                resolutions.append(Resolution(name, problem=f"{type(exc).__name__}: {exc}"))
        return resolutions


def _substance_name_variants(substance_name: str, requested: str) -> tuple[str, ...]:
    """Substance names to match on, including the salt-free form.

    SPL names the ingredient as formulated -- "ESKETAMINE HYDROCHLORIDE" -- while
    reporters usually code the base. Including both, plus the name the cohort
    asked for, is what keeps a salt form from halving a drug's report count.
    """
    variants = {substance_name.strip().upper(), requested.strip().upper()}
    # Strip a trailing salt or hydrate term to recover the base substance.
    for name in list(variants):
        parts = name.split()
        if len(parts) > 1 and parts[-1] in _SALT_WORDS:
            variants.add(" ".join(parts[:-1]))
    return tuple(sorted(v for v in variants if v))


#: Salt and hydrate words that appear as the final token of an SPL ingredient name.
_SALT_WORDS = frozenset(
    {
        "HYDROCHLORIDE",
        "SULFATE",
        "SODIUM",
        "POTASSIUM",
        "CALCIUM",
        "MALEATE",
        "MESYLATE",
        "TARTRATE",
        "CITRATE",
        "ACETATE",
        "PHOSPHATE",
        "FUMARATE",
        "SUCCINATE",
        "BESYLATE",
        "TOSYLATE",
        "BROMIDE",
        "CHLORIDE",
        "DIHYDRATE",
        "MONOHYDRATE",
        "HYDRATE",
        "ANHYDROUS",
        "HEMIFUMARATE",
        "DIMALEATE",
    }
)


def _brand_name_variants(title: str) -> tuple[str, ...]:
    """Brand names from a DailyMed title, which reads "BRAND (GENERIC) FORM [LABELLER]"."""
    match = _BRAND_IN_TITLE.match(title.strip())
    if match is None:
        return ()
    brand = match.group(1).strip().upper()
    # A title may list co-packaged brands, e.g. "OZEMPIC ... RYBELSUS ...".
    return tuple(sorted({b.strip() for b in brand.split(",") if b.strip()}))
