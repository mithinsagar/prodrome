"""Harvesting point-in-time 2x2 tables within a free API quota.

The budget problem
------------------
A 2x2 table needs four counts. Asked naively, one request per count per (drug,
reaction, quarter) cell, a 55-drug cohort with 60 reactions each over 46 quarters
costs roughly 600,000 requests. At the anonymous quota of 1,000 per day that is
nearly two years; even with a key it is five days of continuous traffic against a
free public service, repeated on every backfill. The project would not exist.

What makes it affordable
------------------------
Three observations, each verified against the live API:

1. ``count=patient.reaction.reactionmeddrapt.exact`` returns, for a single term,
   exactly the number of *reports* containing it -- identical to a targeted query's
   ``meta.results.total``. So one request yields an entire row of co-occurrence
   counts. (Checked term by term: NAUSEA 537 = 537, PANCREATITIS 106 = 106.)
2. The reaction marginal `a + c` does not depend on the drug, so one global count
   request per quarter serves the whole cohort.
3. The grand total `N` is one request per quarter.

That reduces the backfill to roughly one request per (drug, quarter) plus two per
quarter, plus targeted fallbacks: order 3,000 requests rather than 600,000.

The correctness catch
---------------------
A count aggregation is truncated to the most frequent terms -- 100 without an API
key, 1,000 with one. A term's absence from a truncated response does not mean zero.
Ileus for semaglutide in 2024Q1 has 21 reports and is nowhere in the top 100, and
ileus is the pair this project was built to study.

So every cell records how its `a` was obtained:

``count_present``
    The term appeared in the aggregation. Exact.
``count_exhaustive``
    The aggregation returned fewer terms than the cap, so it was complete and the
    term's absence is a true zero. Exact, and free.
``targeted``
    The aggregation was truncated and the term was absent, so its count was
    fetched individually. Exact, and costs a request.

Without that distinction a pipeline silently writes zeros for its most important
cells, and every downstream statistic is wrong in a way nothing would detect.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from prodrome.clients.openfda import CountResponse, OpenFdaClient
from prodrome.config import CohortDrug, Config
from prodrome.selector import DrugSelector
from prodrome.stats.contingency import Contingency, ContingencyError
from prodrome.timeframe import Quarter

logger = logging.getLogger(__name__)


class ASource(str):
    """Provenance of a co-occurrence count. See the module docstring."""

    COUNT_PRESENT = "count_present"
    COUNT_EXHAUSTIVE = "count_exhaustive"
    TARGETED = "targeted"


@dataclass(frozen=True, slots=True)
class ContingencyCell:
    """One point-in-time 2x2 table with its provenance."""

    drug_unii: str
    reaction: str
    as_of_quarter: Quarter
    table: Contingency
    a_source: str

    def as_row(self, run_id: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "drug_unii": self.drug_unii,
            "reaction": self.reaction,
            "as_of_quarter": self.as_of_quarter.label,
            "a": self.table.a,
            "b": self.table.b,
            "c": self.table.c,
            "d": self.table.d,
            "a_source": self.a_source,
        }


@dataclass
class GlobalMarginals:
    """Per-quarter database-wide counts, fetched once and shared across the cohort.

    This is the object that turns a per-drug cost into a per-quarter one. It is
    mutable and long-lived on purpose: the reaction marginal for 2019Q2 is the same
    number for all 55 drugs, and fetching it 55 times would be the single largest
    waste in the pipeline.
    """

    client: OpenFdaClient
    _totals: dict[str, int] = field(default_factory=dict)
    _reaction_counts: dict[str, CountResponse] = field(default_factory=dict)
    _targeted: dict[tuple[str, str], int] = field(default_factory=dict)

    def grand_total(self, as_of: Quarter) -> int:
        """`N` for a quarter."""
        key = as_of.label
        if key not in self._totals:
            self._totals[key] = self.client.total_reports(as_of)
        return self._totals[key]

    def _counts(self, as_of: Quarter) -> CountResponse:
        key = as_of.label
        if key not in self._reaction_counts:
            self._reaction_counts[key] = self.client.global_reaction_counts(as_of)
        return self._reaction_counts[key]

    def reaction_total(self, reaction: str, as_of: Quarter) -> int:
        """`a + c` for a reaction, from the shared aggregation or a targeted query."""
        found = self._counts(as_of).get(reaction)
        if found is not None:
            return found
        cache_key = (reaction, as_of.label)
        if cache_key not in self._targeted:
            self._targeted[cache_key] = self.client.reaction_reports(reaction, as_of)
        return self._targeted[cache_key]


@dataclass(frozen=True, slots=True)
class HarvestStats:
    """How the harvest spent its request budget, for the run manifest."""

    cells: int
    from_count_present: int
    from_count_exhaustive: int
    from_targeted: int
    skipped_inconsistent: int

    @property
    def targeted_share(self) -> float:
        """Share of cells that needed their own request.

        The number to watch: a high share means the count aggregations are being
        truncated hard, which happens without an API key and makes the backfill
        an order of magnitude more expensive.
        """
        return self.from_targeted / self.cells if self.cells else 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "cells": self.cells,
            "from_count_present": self.from_count_present,
            "from_count_exhaustive": self.from_count_exhaustive,
            "from_targeted": self.from_targeted,
            "skipped_inconsistent": self.skipped_inconsistent,
            "targeted_share": round(self.targeted_share, 4),
        }


class ContingencyHarvester:
    """Builds point-in-time 2x2 tables for a cohort."""

    def __init__(self, client: OpenFdaClient, config: Config) -> None:
        self._client = client
        self._config = config
        self._globals = GlobalMarginals(client)
        self._counters = {
            ASource.COUNT_PRESENT: 0,
            ASource.COUNT_EXHAUSTIVE: 0,
            ASource.TARGETED: 0,
        }
        self._cells = 0
        self._skipped = 0

    @property
    def stats(self) -> HarvestStats:
        return HarvestStats(
            cells=self._cells,
            from_count_present=self._counters[ASource.COUNT_PRESENT],
            from_count_exhaustive=self._counters[ASource.COUNT_EXHAUSTIVE],
            from_targeted=self._counters[ASource.TARGETED],
            skipped_inconsistent=self._skipped,
        )

    def discover_reactions(self, drug: CohortDrug, as_of: Quarter) -> list[str]:
        """Choose which reactions to track for a drug.

        Ranked by the drug's own cumulative report count at the *latest* quarter,
        so the selection is made once on the fullest picture rather than drifting
        quarter to quarter. A selection that changed per quarter would put a
        discontinuity into every series.

        The consequence, stated plainly: reactions outside a drug's top
        ``max_reactions_per_drug`` are never evaluated, so the analysis is a study
        of a drug's *commonly reported* reactions. Extending it to the long tail is
        a matter of request budget, not of method.
        """
        response = self._client.drug_reaction_counts(drug.selector, as_of)
        ranked = sorted(response.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        chosen = [term for term, count in ranked if count >= self._config.min_reaction_reports][
            : self._config.max_reactions_per_drug
        ]
        if response.truncated:
            logger.debug(
                "reaction discovery for %s was truncated at %d terms; the tail below "
                "%d reports is not visible",
                drug.name,
                response.limit_applied,
                min((c for _, c in ranked), default=0),
            )
        return chosen

    def _co_occurrence(
        self,
        selector: DrugSelector,
        reaction: str,
        as_of: Quarter,
        counts: CountResponse,
    ) -> tuple[int, str]:
        """`a` for one cell, and how it was obtained."""
        found = counts.get(reaction)
        if found is not None:
            source = (
                ASource.COUNT_PRESENT if reaction in counts.counts else ASource.COUNT_EXHAUSTIVE
            )
            return found, source
        return self._client.drug_reaction_reports(selector, reaction, as_of), ASource.TARGETED

    def harvest_drug(
        self, drug: CohortDrug, reactions: Sequence[str], quarters: Sequence[Quarter]
    ) -> Iterator[ContingencyCell]:
        """Yield every point-in-time cell for one drug.

        Inconsistent marginals -- which happen when openFDA reindexes between the
        requests making up one table -- are logged and skipped rather than
        silently coerced. A coerced table produces a plausible ratio from
        impossible counts, which is worse than a gap.
        """
        selector = drug.selector
        for quarter in quarters:
            counts = self._client.drug_reaction_counts(selector, quarter)
            drug_total = self._client.drug_reports(selector, quarter)
            grand_total = self._globals.grand_total(quarter)
            if drug_total == 0 or grand_total == 0:
                logger.debug("%s has no reports as of %s; skipping", drug.name, quarter)
                continue

            for reaction in reactions:
                a, source = self._co_occurrence(selector, reaction, quarter, counts)
                if a == 0:
                    # A zero cell carries no disproportionality information and
                    # would only inflate the warehouse; the absence is recoverable
                    # from the drug's reaction list.
                    continue
                reaction_total = self._globals.reaction_total(reaction, quarter)
                try:
                    table = Contingency.from_marginals(
                        co_occurrence=a,
                        drug_total=drug_total,
                        reaction_total=reaction_total,
                        grand_total=grand_total,
                    )
                except ContingencyError as exc:
                    self._skipped += 1
                    logger.warning(
                        "skipping %s / %s as of %s: %s", drug.name, reaction, quarter, exc
                    )
                    continue

                self._cells += 1
                self._counters[source] += 1
                yield ContingencyCell(
                    drug_unii=drug.unii,
                    reaction=reaction,
                    as_of_quarter=quarter,
                    table=table,
                    a_source=source,
                )

    def estimate_requests(
        self, n_drugs: int, n_quarters: int, n_reactions: int, *, truncation_rate: float = 0.25
    ) -> int:
        """Predicted request count, for the pre-flight budget check.

        Deliberately reported before a run rather than discovered during one: a
        weekly job that exhausts the daily quota cannot be retried until tomorrow,
        which in practice means a missed week.
        """
        per_quarter_global = 2 * n_quarters
        per_drug_quarter = 2 * n_drugs * n_quarters
        targeted = int(n_drugs * n_quarters * n_reactions * truncation_rate)
        discovery = n_drugs
        return per_quarter_global + per_drug_quarter + targeted + discovery
