"""Stage 1: fetch everything, land it in the warehouse.

This is the only stage that spends API quota, so it is built to be interrupted and
resumed. Every upstream response is cached on disk by request, which means a run
that dies halfway -- quota exhausted, laptop closed, CI timeout -- costs nothing to
resume: the second run replays the cache for what already succeeded and spends
requests only on what did not.

The stage writes counts and verdicts, not statistics. Keeping the expensive network
stage free of analysis means a change to an estimator never requires re-downloading
anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from prodrome.clients.base import ApiClient, ApiError, QuotaExceededError
from prodrome.clients.dailymed import DailyMedClient
from prodrome.clients.openfda import (
    FIELD_COUNTRY,
    FIELD_QUALIFICATION,
    OpenFdaClient,
)
from prodrome.config import Config, Settings
from prodrome.ingest.contingency import ContingencyHarvester, HarvestStats
from prodrome.ingest.labels import LabelTimelineBuilder, build_null_pool
from prodrome.labelmatch.embed import build_embedder
from prodrome.timeframe import Quarter, quarters_between
from prodrome.warehouse import Warehouse

logger = logging.getLogger(__name__)

#: Reaction count per drug above which the diagnostics harvest is sampled rather
#: than exhaustive. The diagnostics need three extra requests per *pair*, which is
#: the one place in the pipeline where cost is quadratic in the cohort; sampling the
#: pairs that actually matter -- those with an open signal -- keeps it linear.
DIAGNOSTIC_PAIR_LIMIT = 25


@dataclass
class IngestResult:
    """What one ingest run landed, for the manifest and the CLI summary."""

    run_id: str
    quarters: list[Quarter] = field(default_factory=list)
    reactions_by_drug: dict[str, list[str]] = field(default_factory=dict)
    contingency_rows: int = 0
    label_version_rows: int = 0
    mention_rows: int = 0
    quarterly_report_rows: int = 0
    reporter_mix_rows: int = 0
    harvest: HarvestStats | None = None
    drugs_without_labels: list[str] = field(default_factory=list)
    #: Drugs skipped because openFDA failed persistently for them. Reported rather
    #: than silently absent: a missing drug looks identical to a drug with no
    #: signals in every downstream aggregate.
    failed_drugs: list[str] = field(default_factory=list)
    quota_exhausted: bool = False

    @property
    def total_reactions(self) -> int:
        return sum(len(v) for v in self.reactions_by_drug.values())

    def summary_lines(self) -> list[str]:
        lines = [
            f"run {self.run_id}",
            f"  quarters evaluated      {len(self.quarters)}"
            + (f"  ({self.quarters[0]} .. {self.quarters[-1]})" if self.quarters else ""),
            f"  drugs                   {len(self.reactions_by_drug)}",
            f"  reaction terms tracked  {self.total_reactions}",
            f"  contingency cells       {self.contingency_rows:,}",
            f"  label versions          {self.label_version_rows:,}",
            f"  label-mention verdicts  {self.mention_rows:,}",
        ]
        if self.harvest is not None:
            h = self.harvest
            lines.append(
                f"  count-derived cells     {h.from_count_present + h.from_count_exhaustive:,}"
                f"  ({1 - h.targeted_share:.1%} of cells needed no individual request)"
            )
            if h.skipped_inconsistent:
                lines.append(
                    f"  skipped (inconsistent)  {h.skipped_inconsistent:,}"
                    "  -- openFDA reindexed mid-table"
                )
        if self.drugs_without_labels:
            lines.append(
                f"  no label timeline       {len(self.drugs_without_labels)} drugs "
                f"(excluded from latency): {', '.join(self.drugs_without_labels[:5])}"
                + (" ..." if len(self.drugs_without_labels) > 5 else "")
            )
        if self.failed_drugs:
            lines.append(
                f"  upstream failures       {len(self.failed_drugs)} drugs skipped after "
                f"exhausted retries: {', '.join(self.failed_drugs[:5])}"
                + (" ..." if len(self.failed_drugs) > 5 else "")
            )
            lines.append(
                "    Responses are cached, so re-running costs only what failed. If this "
                "recurs for the same drug, check its selector rather than assuming a "
                "transient fault."
            )
        if self.quota_exhausted:
            lines.append(
                "  NOTE: request budget was exhausted; the warehouse is partial but "
                "valid. Re-run to continue from the cache."
            )
        return lines


def run_ingest(
    warehouse: Warehouse,
    config: Config,
    settings: Settings,
    *,
    run_id: str,
    events: OpenFdaClient,
    dailymed: DailyMedClient,
    transports: list[ApiClient],
    last_quarter: Quarter,
    skip_labels: bool = False,
    skip_diagnostics: bool = False,
) -> IngestResult:
    """Fetch and land everything the later stages need.

    Args:
        last_quarter: resolved final quarter of the evaluation grid.
        skip_labels: land only adverse-event counts. Useful for iterating on the
            disproportionality side without re-walking label archives.
        skip_diagnostics: skip the per-pair reporter-composition harvest, which is
            the most request-hungry part.
    """
    result = IngestResult(run_id=run_id)
    first = Quarter.parse(config.window.first_quarter)
    quarters = quarters_between(first, last_quarter)
    result.quarters = quarters

    warehouse.append_rows(
        "raw_cohort",
        [
            {
                "run_id": run_id,
                "drug_unii": drug.unii,
                "drug_name": drug.name,
                "spl_set_id": drug.spl_set_id,
                "substance_names": ",".join(drug.substance_names),
                "brand_names": ",".join(drug.brand_names),
                "unii_coverage": drug.unii_coverage,
                "notes": drug.notes,
            }
            for drug in config.cohort
        ],
    )

    harvester = ContingencyHarvester(events, config)

    try:
        # ---- reaction discovery, once per drug on the fullest window ----------
        for drug in config.cohort:
            try:
                reactions = harvester.discover_reactions(drug, last_quarter)
            except ApiError as exc:
                # One drug failing must not cost the other 54. The failure is
                # recorded and the drug skipped -- visibly, in the summary, because a
                # silently absent drug is indistinguishable from a drug with no
                # signals in every downstream aggregate.
                logger.error("reaction discovery failed for %s: %s", drug.name, exc)
                result.failed_drugs.append(drug.name)
                continue
            result.reactions_by_drug[drug.unii] = reactions
            logger.info("%s: tracking %d reaction terms", drug.name, len(reactions))

        # ---- point-in-time contingency tables --------------------------------
        for drug in config.cohort:
            reactions = result.reactions_by_drug.get(drug.unii, [])
            if not reactions:
                continue
            try:
                cells = list(harvester.harvest_drug(drug, reactions, quarters))
            except ApiError as exc:
                logger.error("contingency harvest failed for %s: %s", drug.name, exc)
                result.failed_drugs.append(drug.name)
                continue
            written = warehouse.append_batched(
                "raw_contingency", (cell.as_row(run_id) for cell in cells)
            )
            result.contingency_rows += written
            logger.info("%s: %d contingency cells", drug.name, written)

        # ---- diagnostics inputs ----------------------------------------------
        if not skip_diagnostics:
            result.quarterly_report_rows, result.reporter_mix_rows = _harvest_diagnostics(
                warehouse, events, config, run_id, result, last_quarter
            )

        # ---- label timelines and mention verdicts ----------------------------
        # DailyMed is a separate service, so label ingest is attempted even when the
        # event harvest degraded -- a partial warehouse with a full label timeline is
        # still useful, and the reverse is not.
        if not skip_labels:
            _harvest_labels(warehouse, dailymed, config, settings, run_id, result)

    except QuotaExceededError as exc:
        # A partial warehouse is useful and valid; an aborted run that discards
        # everything already fetched is not. The cache makes resumption cheap.
        logger.error("request budget exhausted: %s", exc)
        result.quota_exhausted = True

    result.harvest = harvester.stats
    _record_traffic(warehouse, run_id, transports)
    return result


def _harvest_diagnostics(
    warehouse: Warehouse,
    events: OpenFdaClient,
    config: Config,
    run_id: str,
    result: IngestResult,
    last_quarter: Quarter,
) -> tuple[int, int]:
    """Fetch reporter composition and quarterly volume for the top pairs.

    Restricted to each drug's most-reported reactions: this is the only part of the
    pipeline whose cost is proportional to (drugs x reactions) rather than
    (drugs + quarters), and running it exhaustively would dominate the whole budget
    for information that only matters where there is a signal to qualify.
    """
    quarterly_rows = 0
    mix_rows = 0
    for drug in config.cohort:
        reactions = result.reactions_by_drug.get(drug.unii, [])[:DIAGNOSTIC_PAIR_LIMIT]
        selector = drug.selector
        for reaction in reactions:
            buckets = events.quarterly_new_reports(selector, reaction)
            quarterly_rows += warehouse.append_rows(
                "raw_quarterly_reports",
                [
                    {
                        "run_id": run_id,
                        "drug_unii": drug.unii,
                        "reaction": reaction,
                        "quarter": quarter,
                        "new_reports": count,
                    }
                    for quarter, count in sorted(buckets.items())
                ],
            )
            for dimension, field_name in (
                ("country", FIELD_COUNTRY),
                ("qualification", FIELD_QUALIFICATION),
            ):
                response = events.stratum_counts(selector, reaction, last_quarter, field_name)
                mix_rows += warehouse.append_rows(
                    "raw_reporter_mix",
                    [
                        {
                            "run_id": run_id,
                            "drug_unii": drug.unii,
                            "reaction": reaction,
                            "as_of_quarter": last_quarter.label,
                            "dimension": dimension,
                            "category": category,
                            "reports": count,
                        }
                        for category, count in sorted(response.counts.items())
                    ],
                )
    return quarterly_rows, mix_rows


def _harvest_labels(
    warehouse: Warehouse,
    dailymed: DailyMedClient,
    config: Config,
    settings: Settings,
    run_id: str,
    result: IngestResult,
) -> None:
    """Reconstruct label timelines and assess every tracked reaction against them."""
    backend = config.labelmatch.embed_backend or settings.embed_backend
    embedder = build_embedder(backend, model_name=config.labelmatch.onnx_model)
    logger.info("label matching with embedding backend %s", embedder.name)

    builder = LabelTimelineBuilder(dailymed, embedder, config)
    null_terms = build_null_pool(result.reactions_by_drug)

    for drug in config.cohort:
        reactions = result.reactions_by_drug.get(drug.unii, [])
        if not reactions:
            continue
        records = builder.build_timeline(drug)
        if not records:
            result.drugs_without_labels.append(drug.name)
            continue
        result.label_version_rows += warehouse.append_rows(
            "raw_label_version", [r.as_row(run_id) for r in records]
        )
        # The null pool must exclude this drug's own tracked reactions, or the null
        # would contain terms the label genuinely mentions and be biased upward --
        # making real matches harder to detect for exactly the drugs with the most
        # reactions.
        drug_specific_null = [t for t in null_terms if t not in set(reactions)]
        result.mention_rows += warehouse.append_batched(
            "raw_label_mention",
            (
                record.as_row(run_id)
                for record in builder.assess_mentions(records, reactions, drug_specific_null)
            ),
        )
        logger.info("%s: %d label versions assessed", drug.name, len(records))


def _record_traffic(warehouse: Warehouse, run_id: str, transports: list[ApiClient]) -> None:
    """Fold every client's request counters into the run manifest."""
    warehouse.finish_run(
        run_id,
        requests=sum(t.stats.requests for t in transports),
        cache_hits=sum(t.stats.cache_hits for t in transports),
        retries=sum(t.stats.retries for t in transports),
    )
