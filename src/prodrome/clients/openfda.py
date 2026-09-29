"""openFDA client: adverse-event counts and current drug labels.

The request-budget argument
--------------------------
A naive implementation asks openFDA one question per (drug, reaction, quarter)
cell. For a 40-drug cohort, 60 reactions each and 32 quarters that is 76,800
requests for the co-occurrence counts alone, before any marginals -- days of
work against the free quota, repeated on every backfill.

prodrome instead exploits a property verified against the live API: for a single
term, ``count=patient.reaction.reactionmeddrapt.exact`` returns exactly the
number of *reports* containing that term, identical to the ``meta.results.total``
of a targeted query. So one count request per (drug, quarter) yields the entire
row of co-occurrence counts at once, cutting the backfill by two orders of
magnitude.

The catch is truncation: count returns only the most frequent terms -- 100
without an API key, up to 1,000 with one -- and the interesting reactions are
often rare. Ileus for semaglutide in 2024Q1 has 21 reports and does not appear in
the top 100, and ileus is precisely the case this project was built to study.

:class:`CountResponse` therefore tracks whether a response was truncated, which
makes the inference exact rather than hopeful:

* fewer terms returned than the cap -> the response is **exhaustive**, and a
  term's absence is a true zero, recorded with no further request;
* exactly the cap returned -> the response is **truncated**, and any candidate
  term not present has an unknown count that must be fetched directly.

Missing that distinction is how a pipeline ends up quietly recording zeros for
its most important cells.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

from prodrome.clients import query as q
from prodrome.clients.base import ApiClient
from prodrome.selector import DrugSelector, SelectorCoverage
from prodrome.timeframe import Quarter

logger = logging.getLogger(__name__)

EVENT_ENDPOINT = "https://api.fda.gov/drug/event.json"
LABEL_ENDPOINT = "https://api.fda.gov/drug/label.json"

#: openFDA's own field names, named once so a rename upstream is a one-line fix.
FIELD_RECEIVE_DATE = "receivedate"
FIELD_DRUG_UNII = "patient.drug.openfda.unii.exact"
FIELD_REACTION_PT = "patient.reaction.reactionmeddrapt.exact"
FIELD_SERIOUS = "serious"
FIELD_QUALIFICATION = "primarysource.qualification"
FIELD_COUNTRY = "occurcountry"
FIELD_SEX = "patient.patientsex"
FIELD_DRUG_ROLE = "patient.drug.drugcharacterization"

#: Earliest receivedate worth querying. FAERS in its current form starts in 2004;
#: earlier records exist but are sparse and differently coded.
EPOCH = dt.date(2004, 1, 1)

#: Term cap the API applies to a count aggregation. 100 is the anonymous ceiling;
#: an API key raises it to 1,000, which is why prodrome nags about the key.
COUNT_LIMIT_ANONYMOUS = 100
COUNT_LIMIT_WITH_KEY = 1000


@dataclass(frozen=True, slots=True)
class CountResponse:
    """A count aggregation, carrying its own truncation status.

    ``truncated`` is the field that matters. See the module docstring.
    """

    counts: dict[str, int]
    truncated: bool
    limit_applied: int

    def get(self, term: str) -> int | None:
        """Report count for `term`, or None when genuinely unknown.

        Returns 0 -- a fact -- when the response was exhaustive and the term is
        absent. Returns None -- an admission -- when the response was truncated,
        so the caller knows to ask directly instead of recording a false zero.
        """
        if term in self.counts:
            return self.counts[term]
        return None if self.truncated else 0

    @property
    def total_occurrences(self) -> int:
        return sum(self.counts.values())


class OpenFdaClient:
    """Typed access to the two openFDA endpoints prodrome reads."""

    def __init__(self, client: ApiClient, *, has_api_key: bool) -> None:
        self._client = client
        self.has_api_key = has_api_key
        self.count_limit = COUNT_LIMIT_WITH_KEY if has_api_key else COUNT_LIMIT_ANONYMOUS
        if not has_api_key:
            logger.warning(
                "No openFDA API key: count aggregations are capped at %d terms instead of "
                "%d, and the daily quota is 1,000 requests instead of 120,000. Rare "
                "reactions will need individual queries, so a backfill will be slow and "
                "may not complete. Get a free key at "
                "https://open.fda.gov/apis/authentication/",
                COUNT_LIMIT_ANONYMOUS,
                COUNT_LIMIT_WITH_KEY,
            )

    # ---- low level -------------------------------------------------------

    def _total(self, search: str, *, on_error: str = "raise") -> int:
        """Report count for a search expression, via ``meta.results.total``."""
        payload = self._client.get_json("", {"search": search, "limit": 1}, on_error=on_error)
        if payload is None:
            return 0
        try:
            return int(payload["meta"]["results"]["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unexpected openFDA envelope for search={search!r}") from exc

    def _count(self, search: str, field: str, *, on_error: str = "raise") -> CountResponse:
        params: dict[str, Any] = {"search": search, "count": field}
        # Sending limit without a key is a 403, not a clamp, so only send it when
        # we can actually use it.
        if self.has_api_key:
            params["limit"] = self.count_limit
        payload = self._client.get_json("", params, on_error=on_error)
        if payload is None:
            return CountResponse({}, truncated=False, limit_applied=self.count_limit)
        results = payload.get("results") or []
        # openFDA labels the key differently by field type: term aggregations return
        # "term", date aggregations return "time". Undocumented, and the reason a
        # date-keyed count raises KeyError against a term-only reader.
        counts: dict[str, int] = {}
        for row in results:
            key = row.get("term", row.get("time"))
            if key is None:
                logger.warning("count row with neither 'term' nor 'time': %r", row)
                continue
            counts[str(key)] = int(row["count"])
        return CountResponse(
            counts=counts,
            truncated=len(results) >= self.count_limit,
            limit_applied=self.count_limit,
        )

    # ---- metadata --------------------------------------------------------

    def last_updated(self) -> dt.date:
        """The date openFDA says the event index is current to.

        Used to resolve ``last_quarter: auto``, so the final quarter of a run is
        never a half-populated stub.
        """
        payload = self._client.get_json("", {"limit": 1})
        if payload is None:
            raise ValueError("openFDA returned no metadata envelope")
        raw = payload["meta"]["last_updated"]
        return dt.date.fromisoformat(str(raw))

    def latest_complete_quarter(self) -> Quarter:
        """The most recent quarter that has fully closed in the index.

        The quarter containing ``last_updated`` is still accruing reports, so
        including it would show as a false decline in every count series.
        """
        return Quarter.containing(self.last_updated()).shift(-1)

    # ---- point-in-time windows ------------------------------------------

    @staticmethod
    def window_clause(as_of: Quarter, *, since: dt.date = EPOCH) -> str:
        """Cumulative point-in-time window: every report received up to `as_of`.

        Cumulative rather than per-quarter because that is how disproportionality
        is computed in practice -- a signal is assessed against the whole
        accumulated experience, not one quarter of it in isolation.
        """
        return q.date_range(FIELD_RECEIVE_DATE, since, as_of.end_date)

    # ---- the four counts a contingency table needs ----------------------

    def total_reports(self, as_of: Quarter) -> int:
        """`N`: all reports received by `as_of`."""
        return self._total(self.window_clause(as_of))

    def drug_reports(self, drug: DrugSelector, as_of: Quarter) -> int:
        """`a + b`: reports identifying the drug."""
        return self._total(q.all_of(drug.clause, self.window_clause(as_of)))

    def reaction_reports(self, reaction: str, as_of: Quarter) -> int:
        """`a + c`: reports naming the reaction, any drug."""
        return self._total(
            q.all_of(q.field_is(FIELD_REACTION_PT, reaction), self.window_clause(as_of))
        )

    def drug_reaction_reports(self, drug: DrugSelector, reaction: str, as_of: Quarter) -> int:
        """`a`: reports identifying both.

        One request per cell, so this is the expensive path. It exists only as the
        fallback for reactions a truncated count aggregation could not answer.
        """
        return self._total(
            q.all_of(
                drug.clause,
                q.field_is(FIELD_REACTION_PT, reaction),
                self.window_clause(as_of),
            )
        )

    # ---- bulk harvests ---------------------------------------------------

    def drug_reaction_counts(self, drug: DrugSelector, as_of: Quarter) -> CountResponse:
        """Every reaction count for one drug as of one quarter, in one request.

        This is the call that makes the backfill affordable: it replaces one
        request per (drug, reaction, quarter) cell with one per (drug, quarter).
        """
        return self._count(q.all_of(drug.clause, self.window_clause(as_of)), FIELD_REACTION_PT)

    def global_reaction_counts(self, as_of: Quarter) -> CountResponse:
        """Database-wide reaction counts as of one quarter, in one request.

        Supplies the `a + c` marginal for every reaction it covers, shared across
        the whole cohort.
        """
        return self._count(self.window_clause(as_of), FIELD_REACTION_PT)

    def stratum_counts(
        self, drug: DrugSelector, reaction: str, as_of: Quarter, field: str
    ) -> CountResponse:
        """Counts broken down by a stratifying field, for the adjusted analysis."""
        # Stratum breakdowns qualify a signal but no statistic depends on them, so
        # a persistent upstream failure degrades the diagnostics rather than ending
        # the run.
        return self._count(
            q.all_of(
                drug.clause,
                q.field_is(FIELD_REACTION_PT, reaction),
                self.window_clause(as_of),
            ),
            field,
            on_error="skip",
        )

    def drug_stratum_totals(self, drug: DrugSelector, as_of: Quarter, field: str) -> CountResponse:
        """Stratum totals for the drug, ignoring reaction -- the `a + b` per stratum."""
        return self._count(q.all_of(drug.clause, self.window_clause(as_of)), field)

    def global_stratum_totals(self, as_of: Quarter, field: str) -> CountResponse:
        """Database-wide stratum totals -- the `N` per stratum."""
        return self._count(self.window_clause(as_of), field)

    def reaction_stratum_totals(self, reaction: str, as_of: Quarter, field: str) -> CountResponse:
        """Stratum totals for the reaction across all drugs -- the `a + c` per stratum."""
        return self._count(
            q.all_of(q.field_is(FIELD_REACTION_PT, reaction), self.window_clause(as_of)),
            field,
        )

    def reporter_country_counts(
        self, drug: DrugSelector, reaction: str, as_of: Quarter
    ) -> CountResponse:
        """Reporting-country breakdown, for the concentration diagnostic."""
        return self.stratum_counts(drug, reaction, as_of, FIELD_COUNTRY)

    def qualification_counts(
        self, drug: DrugSelector, reaction: str, as_of: Quarter
    ) -> CountResponse:
        """Reporter-type breakdown (physician, pharmacist, consumer, lawyer).

        ``primarysource.qualification`` coded 5 is "consumer or non-health
        professional"; a pair dominated by it behaves very differently from a
        clinician-reported one, which the artefact diagnostics use.
        """
        return self.stratum_counts(drug, reaction, as_of, FIELD_QUALIFICATION)

    def measure_selector_coverage(self, drug: DrugSelector) -> SelectorCoverage:
        """Measure how much each identification method recalls for this drug.

        Four requests per drug, run once at cohort-resolution time. The result is
        pinned into ``conf/cohort.yml`` so it is not recomputed, and carried into
        the warehouse so every figure can be traced to how its population was
        identified.
        """
        from prodrome.selector import FIELD_BRAND, FIELD_SUBSTANCE

        union = self._total(drug.clause)
        unii_clause = drug.unii_only_clause
        return SelectorCoverage(
            union_reports=union,
            unii_reports=self._total(unii_clause) if unii_clause else 0,
            substance_reports=(
                self._total(q.any_of(FIELD_SUBSTANCE, drug.substance_names))
                if drug.substance_names
                else 0
            ),
            brand_reports=(
                self._total(q.any_of(FIELD_BRAND, drug.brand_names)) if drug.brand_names else 0
            ),
        )

    def quarterly_new_reports(self, drug: DrugSelector, reaction: str) -> dict[str, int]:
        """Per-quarter *new* report counts for one pair, for spike detection.

        Counts by ``receivedate`` with no window, then buckets into quarters.
        One request covers the entire history of the pair.
        """
        response = self._count(
            q.all_of(drug.clause, q.field_is(FIELD_REACTION_PT, reaction)),
            FIELD_RECEIVE_DATE,
            on_error="skip",
        )
        buckets: dict[str, int] = {}
        for raw, count in response.counts.items():
            try:
                day = dt.datetime.strptime(raw, "%Y%m%d").date()
            except ValueError:
                logger.debug("skipping unparseable receivedate bucket %r", raw)
                continue
            label = Quarter.containing(day).label
            buckets[label] = buckets.get(label, 0) + count
        return buckets

    # ---- labels ----------------------------------------------------------

    def current_label_by_name(self, drug_name: str) -> list[dict[str, Any]]:
        """Current SPL records matching a brand or generic name.

        Searches both name fields because cohort lists are written the way a
        clinician says them, which may be either. Used only by cohort resolution,
        whose output is reviewed by a human before it is trusted.
        """
        search = (
            f"{q.field_is('openfda.brand_name', drug_name)} "
            f"OR {q.field_is('openfda.generic_name', drug_name)}"
        )
        payload = self._client.get_json("", {"search": search, "limit": 100})
        if payload is None:
            return []
        return list(payload.get("results") or [])

    def active_uniis_by_set_id(self, set_id: str) -> dict[str, str]:
        """Active-ingredient UNIIs for a set id, if openFDA happens to index it.

        A cheap fast path for cohort resolution: this is a small JSON document,
        whereas the authoritative answer means downloading a multi-megabyte SPL
        archive. openFDA's label index is incomplete, so a caller must treat an
        empty result as "ask DailyMed" rather than as "no active ingredients".

        The returned names come from ``openfda.substance_name``, which is the
        harmonised active-substance list -- not ``generic_name``, which is the
        field that reports Ozempic as "ORAL SEMAGLUTIDE".
        """
        payload = self._client.get_json(
            "", {"search": q.field_is("openfda.spl_set_id", set_id), "limit": 1}
        )
        if payload is None:
            return {}
        results = payload.get("results") or []
        if not results:
            return {}
        openfda = results[0].get("openfda") or {}
        uniis = [str(u).upper() for u in openfda.get("unii", []) if u]
        names = [str(n) for n in openfda.get("substance_name", []) if n]
        if not uniis:
            return {}
        # The two lists are parallel when both are present; when they are not,
        # fall back to the UNII as its own label rather than mispairing them.
        if len(names) == len(uniis):
            return dict(zip(uniis, names, strict=True))
        return {unii: names[0] if names else unii for unii in uniis}

    def drug_report_count(self, drug: DrugSelector) -> int:
        """All-time report count, no date window.

        Used to screen a cohort candidate for sufficient volume before committing
        request budget to its quarterly series.
        """
        return self._total(drug.clause)

    def current_label(self, unii: str) -> list[dict[str, Any]]:
        """Current SPL documents for a substance, from openFDA's label index.

        Used to resolve which DailyMed set id belongs to the application holder
        and to read ``recent_major_changes``. The *historical* text comes from
        DailyMed, because openFDA only indexes the current version of a label.
        """
        payload = self._client.get_json(
            "",
            {"search": q.field_is("openfda.unii", unii), "limit": 100},
        )
        if payload is None:
            return []
        return list(payload.get("results") or [])
