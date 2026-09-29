"""Stage 3: join signals to labels, measure the gap, rank what is still open.

This stage produces the project's actual outputs:

* per-criterion lead time, as a survival curve -- how long labelling takes once a
  criterion fires, with censored pairs handled correctly;
* the leakage benchmark -- how much a conventional retrospective analysis would
  have overstated each signal;
* robustness diagnostics -- whether a signal looks like a reporting artefact;
* a calibrated, ranked queue of currently-unlabelled signals, which is the thing a
  signal-management team would actually use.

The stage is deliberately free of network access and of statistical estimation that
belongs upstream: it joins, classifies and fits. Everything it reads was written by
stages 1 and 2, so every number here is reproducible from the warehouse alone.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from prodrome.config import Config
from prodrome.diagnostics import assess_robustness
from prodrome.latency.leakage import compare_leakage, measure_notoriety, summarise_leakage
from prodrome.latency.onset import (
    LabelOnset,
    PairStatus,
    label_onset,
    pair_outcome,
    signal_onsets,
)
from prodrome.latency.survival import HAZARD_FEATURES, fit_hazard_model, kaplan_meier
from prodrome.stats.criteria import CRITERIA
from prodrome.timeframe import Quarter
from prodrome.warehouse import Warehouse

logger = logging.getLogger(__name__)

#: The criterion the priority queue and the headline figures use. The EMA interval
#: rule, because the design-time measurement on semaglutide/ileus showed it firing
#: roughly 16 months before the label change while the MHRA triple never fired at
#: all. Every criterion is still computed and reported side by side -- this is only
#: the default view.
PRIMARY_CRITERION = "ema_ror025"


@dataclass
class LatencyResult:
    """Summary of a latency pass."""

    run_id: str
    pair_outcomes: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    survival_curves: int = 0
    leakage_rows: int = 0
    diagnostics_rows: int = 0
    queue_rows: int = 0
    hazard_note: str = ""
    hazard_auc: float | None = None
    hazard_lift_at_50: float | None = None
    median_lead_by_criterion: dict[str, float | None] = field(default_factory=dict)
    median_inflation: float | None = None
    retrospective_only_share: float | None = None

    def summary_lines(self) -> list[str]:
        lines = [f"run {self.run_id}", f"  pair outcomes           {self.pair_outcomes:,}"]
        for status, count in sorted(self.status_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {status:24s} {count:,}")
        if self.median_lead_by_criterion:
            lines.append("  median quarters from signal to label change:")
            for criterion, median in self.median_lead_by_criterion.items():
                shown = f"{median:.1f}" if median is not None else "not reached"
                lines.append(f"    {criterion:24s} {shown}")
        if self.median_inflation is not None:
            lines.append(
                f"  leakage: a retrospective analysis reports a PRR "
                f"{self.median_inflation:.2f}x the value available at the label change"
            )
        if self.retrospective_only_share is not None:
            lines.append(
                f"  {self.retrospective_only_share:.1%} of labelled pairs signal only "
                f"retrospectively -- detections a conventional analysis claims but "
                f"could not have made"
            )
        if self.hazard_auc is not None:
            lines.append(
                f"  prioritisation model: AUC {self.hazard_auc:.3f}, "
                f"lift@50 {self.hazard_lift_at_50:.2f}x over base rate"
            )
        if self.hazard_note:
            lines.append(f"  model note: {self.hazard_note}")
        lines.append(f"  open signals queued     {self.queue_rows:,}")
        return lines


# ---------------------------------------------------------------------------
# DataFrame boundary helpers.
#
# pandas types groupby keys and itertuples attributes as Hashable or as a union of
# every dtype it supports, because a DataFrame carries no static schema. Rather
# than casting at each of two dozen call sites, values cross into domain types
# here, once, where the conversion can be checked and explained.
# ---------------------------------------------------------------------------


def _pair(unii: object, reaction: object) -> tuple[str, str]:
    """A (drug, reaction) key, from whatever pandas handed back."""
    return (str(unii), str(reaction))


def _text(value: object) -> str:
    return str(value)


def _count(value: object) -> int:
    """An integer count, treating a missing or unusable value as zero."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def run_latency(warehouse: Warehouse, config: Config, *, run_id: str) -> LatencyResult:  # noqa: PLR0912, PLR0915
    """Join scored series to label timelines and fit the models."""
    result = LatencyResult(run_id=run_id)

    cleared = warehouse.clear_runs(
        [
            "stat_pair_outcome",
            "stat_leakage",
            "stat_diagnostics",
            "model_survival_curve",
            "model_survival_points",
            "model_hazard_report",
            "model_hazard_coefficient",
            "model_calibration",
            "model_priority_queue",
        ],
        run_id,
    )
    if cleared:
        logger.info("cleared previous output for this run: %s", cleared)

    stats = warehouse.query_df(
        """
        SELECT drug_unii, reaction, as_of_quarter, a, expected, prr, ror, ror_ci_lower,
               chi2_yates, log2_oe_shrunk, ebgm, eb05, n_criteria_firing,
               fired_mhra_prr, fired_ema_ror025, fired_who_oe025, fired_dubious_prr_only
        FROM stat_disproportionality WHERE run_id = ?
        ORDER BY drug_unii, reaction, as_of_quarter
        """,
        [run_id],
    )
    if stats.empty:
        logger.warning("no scored statistics for run %s; run the score stage first", run_id)
        return result

    mentions = warehouse.query_df(
        """
        SELECT m.drug_unii, m.reaction, m.spl_version, m.authoritative_date,
               m.is_labelled, v.is_usable
        FROM raw_label_mention m
        JOIN raw_label_version v
          ON v.run_id = m.run_id AND v.drug_unii = m.drug_unii
         AND v.spl_version = m.spl_version
        WHERE m.run_id = ?
        """,
        [run_id],
    )
    drug_names = dict(
        warehouse.query("SELECT drug_unii, drug_name FROM raw_cohort WHERE run_id = ?", [run_id])
    )

    observation_end = Quarter.parse(str(stats["as_of_quarter"].max()))

    # ---- label onsets -----------------------------------------------------
    onset_by_pair: dict[tuple[str, str], LabelOnset] = {}
    if not mentions.empty:
        for (unii, reaction), group in mentions.groupby(["drug_unii", "reaction"]):
            onset_by_pair[_pair(unii, reaction)] = label_onset(
                [
                    (_as_date(row.authoritative_date), bool(row.is_labelled), bool(row.is_usable))
                    for row in group.itertuples()
                ]
            )

    # ---- signal onsets and pair outcomes ----------------------------------
    outcome_rows: list[dict[str, object]] = []
    status_counts: dict[str, int] = defaultdict(int)
    durations: dict[str, list[tuple[float, int]]] = defaultdict(list)
    signal_quarter_by_pair: dict[tuple[str, str], Quarter | None] = {}

    for (unii, reaction), group in stats.groupby(["drug_unii", "reaction"], sort=True):
        fired_series = {
            criterion.key: {
                Quarter.parse(str(q)): bool(v)
                for q, v in zip(
                    group["as_of_quarter"], group[f"fired_{criterion.key}"], strict=True
                )
            }
            for criterion in CRITERIA
        }
        onsets = signal_onsets(fired_series)
        label = onset_by_pair.get(_pair(unii, reaction))

        for criterion_key, onset in onsets.items():
            if label is None:
                # No label timeline for this drug: the pair cannot contribute an
                # outcome. Recorded as unobservable rather than dropped, so the
                # exclusion is visible in the status counts.
                status_counts[PairStatus.UNOBSERVABLE.value] += 1
                continue
            outcome = pair_outcome(onset, label, observation_end=observation_end)
            status_counts[outcome.status.value] += 1
            outcome_rows.append(
                {
                    "run_id": run_id,
                    "drug_unii": _text(unii),
                    "reaction": _text(reaction),
                    **outcome.as_row(),
                }
            )
            if outcome.in_survival_set and outcome.lead_time_quarters is not None:
                durations[criterion_key].append(
                    (float(max(outcome.lead_time_quarters, 0)), int(outcome.is_event))
                )
            if criterion_key == PRIMARY_CRITERION:
                signal_quarter_by_pair[_pair(unii, reaction)] = outcome.signal_quarter

    result.pair_outcomes = warehouse.append_batched("stat_pair_outcome", outcome_rows)
    result.status_counts = dict(status_counts)

    # ---- survival curves ---------------------------------------------------
    for criterion in CRITERIA:
        pairs = durations.get(criterion.key, [])
        curve = kaplan_meier(
            np.array([d for d, _ in pairs]),
            np.array([e for _, e in pairs]),
            criterion=criterion.key,
        )
        warehouse.append_rows("model_survival_curve", [{"run_id": run_id, **curve.as_row()}])
        if curve.timeline:
            warehouse.append_rows(
                "model_survival_points",
                [
                    {
                        "run_id": run_id,
                        "criterion": criterion.key,
                        "quarters": float(t),
                        "survival": float(s),
                    }
                    for t, s in zip(curve.timeline, curve.survival, strict=True)
                ],
            )
        result.survival_curves += 1
        result.median_lead_by_criterion[criterion.key] = curve.median_quarters_to_label

    # ---- leakage benchmark -------------------------------------------------
    result.leakage_rows, result.median_inflation, result.retrospective_only_share = _write_leakage(
        warehouse, run_id, stats, outcome_rows, signal_quarter_by_pair
    )

    # ---- robustness diagnostics -------------------------------------------
    result.diagnostics_rows = _write_diagnostics(warehouse, run_id)

    # ---- prioritisation model ---------------------------------------------
    panel = _build_panel(warehouse, run_id, stats, onset_by_pair, signal_quarter_by_pair)
    report, model = fit_hazard_model(
        panel, holdout_from_quarter=config.latency.holdout_from_quarter
    )
    warehouse.append_rows("model_hazard_report", [{"run_id": run_id, **report.as_row()}])
    if report.coefficients:
        warehouse.append_rows(
            "model_hazard_coefficient",
            [
                {"run_id": run_id, "feature": feature, "coefficient": value}
                for feature, value in report.coefficients.items()
            ],
        )
    if report.calibration:
        warehouse.append_rows(
            "model_calibration",
            [
                {
                    "run_id": run_id,
                    "decile": index,
                    "mean_predicted": predicted,
                    "observed_rate": observed,
                    "n_rows": n,
                }
                for index, (predicted, observed, n) in enumerate(report.calibration)
            ],
        )
    result.hazard_note = report.note
    if report.fitted and report.n_test_events:
        result.hazard_auc = report.roc_auc
        result.hazard_lift_at_50 = report.lift_at_50

    # ---- the operational queue --------------------------------------------
    result.queue_rows = _write_priority_queue(
        warehouse, run_id, panel, model, observation_end, drug_names
    )
    return result


def _as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _write_leakage(
    warehouse: Warehouse,
    run_id: str,
    stats: pd.DataFrame,
    outcome_rows: list[dict[str, object]],
    signal_quarters: Mapping[tuple[str, str], Quarter | None],
) -> tuple[int, float | None, float | None]:
    """Compare point-in-time and retrospective statistics for every labelled pair."""
    # Only *incident* pairs may enter the leakage benchmark. A pair already labelled
    # in the earliest archived label version is left-truncated: its recorded
    # label_quarter is the start of the archive, not the date the reaction was added,
    # so comparing statistics "at the label change" against it is meaningless. Left
    # in, these pairs dominate the cohort -- 92 of 144 in the first smoke run -- and
    # drag the measured inflation toward 1.0, understating the very effect this
    # benchmark exists to quantify.
    eligible_statuses = {
        PairStatus.LABELLED_AFTER_SIGNAL.value,
        PairStatus.LABELLED_BEFORE_SIGNAL.value,
    }
    label_quarters = {
        (row["drug_unii"], row["reaction"]): row["label_quarter"]
        for row in outcome_rows
        if row["criterion"] == PRIMARY_CRITERION
        and row["label_quarter"]
        and row["status"] in eligible_statuses
    }
    quarterly = warehouse.query_df(
        "SELECT drug_unii, reaction, quarter, new_reports FROM raw_quarterly_reports "
        "WHERE run_id = ?",
        [run_id],
    )
    volume_by_pair: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in quarterly.itertuples():
        volume_by_pair[_pair(row.drug_unii, row.reaction)][_text(row.quarter)] = _count(
            row.new_reports
        )

    rows: list[dict[str, object]] = []
    inflations: list[float] = []
    retrospective_only = 0
    notorious = 0

    for (unii, reaction), group in stats.groupby(["drug_unii", "reaction"], sort=True):
        label_quarter_label = label_quarters.get(_pair(unii, reaction))
        if not label_quarter_label:
            continue
        label_quarter = Quarter.parse(str(label_quarter_label))
        signal_quarter = signal_quarters.get(_pair(unii, reaction))

        series = {
            Quarter.parse(str(q)): float(v)
            for q, v in zip(group["as_of_quarter"], group["prr"], strict=True)
            if v is not None and np.isfinite(v)
        }
        if not series:
            continue
        comparison = compare_leakage(
            "prr", series, signal_quarter=signal_quarter, label_quarter=label_quarter
        )
        notoriety = measure_notoriety(
            volume_by_pair.get(_pair(unii, reaction), {}), label_quarter=label_quarter
        )
        if comparison.inflation_ratio is not None:
            inflations.append(comparison.inflation_ratio)
        # A pair that signals only retrospectively is one where the criterion had not
        # fired by the time the label changed, yet the full-data statistic clears the
        # signalling bar. These are detections a conventional analysis claims but
        # could not have made prospectively. The 2.0 bar is the PRR threshold shared
        # by the MHRA triple and the no-gate control, so the comparison is against a
        # published rule rather than an arbitrary cut.
        if signal_quarter is None or signal_quarter > label_quarter:
            retrospective = comparison.retrospective
            if np.isfinite(retrospective) and retrospective >= 2.0:
                retrospective_only += 1
        notorious += int(notoriety.is_notorious)

        rows.append(
            {
                "run_id": run_id,
                "drug_unii": _text(unii),
                "reaction": _text(reaction),
                **comparison.as_row(),
                **notoriety.as_row(),
            }
        )

    written = warehouse.append_batched("stat_leakage", rows)
    summary = summarise_leakage(
        inflations,
        n_pairs=len(rows),
        n_retrospective_only=retrospective_only,
        n_notorious=notorious,
    )
    return written, summary.median_inflation, summary.retrospective_only_share


def _write_diagnostics(warehouse: Warehouse, run_id: str) -> int:
    """Compute artefact diagnostics from the reporter-composition harvest."""
    mix = warehouse.query_df(
        "SELECT drug_unii, reaction, dimension, category, reports FROM raw_reporter_mix "
        "WHERE run_id = ?",
        [run_id],
    )
    quarterly = warehouse.query_df(
        "SELECT drug_unii, reaction, quarter, new_reports FROM raw_quarterly_reports "
        "WHERE run_id = ?",
        [run_id],
    )
    if mix.empty and quarterly.empty:
        logger.info("no diagnostics inputs for run %s (ingest ran with --skip-diagnostics)", run_id)
        return 0

    countries: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    qualifications: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in mix.itertuples():
        target = countries if row.dimension == "country" else qualifications
        target[_pair(row.drug_unii, row.reaction)][_text(row.category)] = _count(row.reports)

    volumes: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in quarterly.itertuples():
        volumes[_pair(row.drug_unii, row.reaction)][_text(row.quarter)] = _count(row.new_reports)

    pairs = set(countries) | set(qualifications) | set(volumes)
    rows = [
        {
            "run_id": run_id,
            "drug_unii": unii,
            "reaction": reaction,
            **assess_robustness(
                countries.get((unii, reaction), {}),
                qualifications.get((unii, reaction), {}),
                volumes.get((unii, reaction), {}),
            ).as_row(),
        }
        for unii, reaction in sorted(pairs)
    ]
    return warehouse.append_batched("stat_diagnostics", rows)


def _build_panel(
    warehouse: Warehouse,
    run_id: str,
    stats: pd.DataFrame,
    onset_by_pair: Mapping[tuple[str, str], LabelOnset],
    signal_quarters: Mapping[tuple[str, str], Quarter | None],
) -> pd.DataFrame:
    """Assemble the pair-quarter risk set for the discrete-time hazard model.

    One row per (drug, reaction, quarter) in which the pair is *at risk*: a signal
    has fired and no label change has happened yet. Rows stop at the quarter the
    label changes, which is the row carrying the event. Pairs already labelled at
    baseline never enter, because they are left-truncated.
    """
    diagnostics = warehouse.query_df(
        "SELECT drug_unii, reaction, reporter_concentration, consumer_share, "
        "spike_ratio FROM stat_diagnostics WHERE run_id = ?",
        [run_id],
    )
    diagnostic_index = {(row.drug_unii, row.reaction): row for row in diagnostics.itertuples()}
    firing_fraction = (
        warehouse.query_df(
            "SELECT drug_unii, reaction, firing_fraction FROM stat_pair_outcome "
            "WHERE run_id = ? AND criterion = ?",
            [run_id, PRIMARY_CRITERION],
        )
        .set_index(["drug_unii", "reaction"])["firing_fraction"]
        .to_dict()
    )
    drug_totals = stats.groupby("drug_unii")["a"].sum().to_dict()
    grand_total = max(sum(drug_totals.values()), 1)

    rows: list[dict[str, object]] = []
    for (unii, reaction), group in stats.groupby(["drug_unii", "reaction"], sort=True):
        signal_quarter = signal_quarters.get(_pair(unii, reaction))
        if signal_quarter is None:
            continue
        label = onset_by_pair.get(_pair(unii, reaction))
        if label is None or label.present_at_baseline:
            continue
        label_quarter = label.first_labelled_quarter
        diagnostic = diagnostic_index.get(_pair(unii, reaction))

        for row in group.sort_values("as_of_quarter").itertuples():
            quarter = Quarter.parse(str(row.as_of_quarter))
            if quarter < signal_quarter:
                continue
            if label_quarter is not None and quarter > label_quarter:
                break
            event = int(label_quarter is not None and quarter == label_quarter)
            rows.append(
                {
                    "drug_unii": _text(unii),
                    "reaction": _text(reaction),
                    "as_of_quarter": _text(row.as_of_quarter),
                    "labelled_this_quarter": event,
                    "log2_oe_shrunk": _num(row.log2_oe_shrunk),
                    "log_ebgm": _log1p(row.ebgm),
                    "log_eb05": _log1p(row.eb05),
                    "log_prr": _log1p(row.prr),
                    "log_ror_lower": _log1p(row.ror_ci_lower),
                    "chi2_yates": _num(row.chi2_yates),
                    "log_a": float(np.log1p(max(_count(row.a), 0))),
                    "log_expected": _log1p(row.expected),
                    "quarters_since_signal": quarter - signal_quarter,
                    "firing_fraction": float(
                        firing_fraction.get(_pair(unii, reaction), 0.0) or 0.0
                    ),
                    "reporter_concentration": (
                        _num(diagnostic.reporter_concentration) if diagnostic else 0.0
                    ),
                    "consumer_share": _num(diagnostic.consumer_share) if diagnostic else 0.0,
                    "spike_ratio": _num(diagnostic.spike_ratio) if diagnostic else 0.0,
                    "n_criteria_firing": _count(row.n_criteria_firing),
                    "drug_report_share": float(drug_totals.get(_text(unii), 0)) / grand_total,
                }
            )
            if event:
                break
    return pd.DataFrame(rows)


def _write_priority_queue(
    warehouse: Warehouse,
    run_id: str,
    panel: pd.DataFrame,
    model: Any,
    observation_end: Quarter,
    drug_names: dict[str, str],
) -> int:
    """Score and rank the signals that are still unlabelled at the latest quarter.

    This is the operational output. Ranking is by fitted hazard when a model was
    fitted; otherwise by the shrinkage observed-to-expected ratio, which is the most
    conservative single measure available and degrades gracefully rather than
    emitting nothing.

    Supporting tables are turned into plain keyed dicts rather than being indexed
    with ``.loc``. That is not only friendlier to type checking: a ``.loc`` lookup on
    a MultiIndex silently returns a Series or a scalar depending on how many rows
    match, so a duplicated key would change the shape of every field downstream.
    """
    if panel.empty:
        return 0
    latest = panel[
        (panel["as_of_quarter"] == observation_end.label) & (panel["labelled_this_quarter"] == 0)
    ].copy()
    if latest.empty:
        return 0

    features = [f for f in HAZARD_FEATURES if f in latest.columns]
    if model is not None and features:
        latest["hazard_probability"] = model.predict_proba(
            latest[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )[:, 1]
    else:
        latest["hazard_probability"] = np.nan

    sort_column = (
        "hazard_probability" if latest["hazard_probability"].notna().any() else "log2_oe_shrunk"
    )
    latest = latest.sort_values(sort_column, ascending=False).reset_index(drop=True)

    statistics = _index_by_pair(
        warehouse.query_df(
            "SELECT drug_unii, reaction, a, ebgm, eb05, ror, ror_ci_lower, prr, "
            "n_criteria_firing FROM stat_disproportionality "
            "WHERE run_id = ? AND as_of_quarter = ?",
            [run_id, observation_end.label],
        )
    )
    robustness = _index_by_pair(
        warehouse.query_df(
            "SELECT drug_unii, reaction, robustness_score, artefact_flags "
            "FROM stat_diagnostics WHERE run_id = ?",
            [run_id],
        )
    )
    onsets = _index_by_pair(
        warehouse.query_df(
            "SELECT drug_unii, reaction, signal_quarter FROM stat_pair_outcome "
            "WHERE run_id = ? AND criterion = ?",
            [run_id, PRIMARY_CRITERION],
        )
    )

    rows: list[dict[str, object]] = []
    for rank, row in enumerate(latest.itertuples(), start=1):
        key = _pair(row.drug_unii, row.reaction)
        stat = statistics.get(key, {})
        rob = robustness.get(key, {})
        onset = onsets.get(key, {})
        signal_quarter = onset.get("signal_quarter")
        rows.append(
            {
                "run_id": run_id,
                "drug_unii": key[0],
                "drug_name": drug_names.get(key[0], key[0]),
                "reaction": key[1],
                "as_of_quarter": observation_end.label,
                "hazard_probability": _num(row.hazard_probability),
                "a": _count(stat.get("a")) if stat else None,
                "ebgm": _num(stat.get("ebgm")),
                "eb05": _num(stat.get("eb05")),
                "ror": _num(stat.get("ror")),
                "ror_ci_lower": _num(stat.get("ror_ci_lower")),
                "prr": _num(stat.get("prr")),
                "signal_quarter": _text(signal_quarter) if signal_quarter is not None else None,
                "quarters_since_signal": _count(row.quarters_since_signal),
                "criteria_firing": str(_count(stat.get("n_criteria_firing"))) if stat else None,
                "robustness_score": _num(rob.get("robustness_score")),
                "artefact_flags": _text(rob.get("artefact_flags") or ""),
                "rank": rank,
            }
        )
    return warehouse.append_batched("model_priority_queue", rows)


def _index_by_pair(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    """Index a frame by (drug_unii, reaction) as plain dicts.

    Raises:
        ValueError: on a duplicated key. A duplicate here would mean two rows claim
            the same pair, and silently keeping one of them would make the published
            queue depend on row order.
    """
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for record in frame.to_dict(orient="records"):
        key = _pair(record.get("drug_unii"), record.get("reaction"))
        if key in indexed:
            raise ValueError(f"duplicate row for pair {key}")
        indexed[key] = {str(k): v for k, v in record.items()}
    return indexed


def _num(value: object) -> float:
    """Coerce to a finite float, mapping anything else to 0.0."""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def _log1p(value: object) -> float:
    """log1p of a non-negative measure, 0.0 for anything unusable.

    Ratio measures are heavily right-skewed; a logistic model on the raw scale is
    dominated by a handful of enormous values.
    """
    numeric = _num(value)
    return float(np.log1p(numeric)) if numeric > 0 else 0.0
