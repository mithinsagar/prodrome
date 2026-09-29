"""The evidence pack a brief is written from.

The rule this module enforces
-----------------------------
A language model is never shown the warehouse and never asked to compute anything.
It is handed a closed set of already-computed facts and asked only to *phrase*
them. Every quantity that may appear in the output exists in the pack first, and
:mod:`prodrome.brief.verify` rejects any output containing a number that is not in
it.

That ordering matters. The failure mode for LLM summaries of quantitative work is
not obvious nonsense -- it is a plausible number in the right units and the wrong
value, which reads as authoritative and is very hard to spot. Making the numbers
upstream of the prose, and mechanically checkable against it, removes the
opportunity rather than asking the model to be careful.

The pack is also a complete input to the deterministic template, so the brief is
generated with byte-identical numbers whether or not a model is configured. The
model changes the wording. It cannot change the facts.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from prodrome.warehouse import Warehouse

#: How many gaps the brief covers. A weekly brief is read; a list of 300 is not.
DEFAULT_GAP_COUNT = 8


@dataclass(frozen=True, slots=True)
class GapEvidence:
    """Every fact about one label gap that the brief may state."""

    drug_name: str
    drug_unii: str
    reaction: str
    reports: int
    expected: float
    ror: float | None
    ror_ci_lower: float | None
    prr: float | None
    ebgm: float | None
    eb05: float | None
    calibrated_p: float | None
    signal_quarter: str | None
    quarters_since_signal: int | None
    criteria_firing: int
    robustness_score: float | None
    artefact_flags: str
    hazard_probability: float | None
    relies_on_name_matching: bool

    @property
    def numeric_facts(self) -> dict[str, float]:
        """Every number the brief is permitted to state for this gap."""
        facts: dict[str, float] = {
            "reports": float(self.reports),
            "expected": float(self.expected),
            "criteria_firing": float(self.criteria_firing),
        }
        for name in (
            "ror",
            "ror_ci_lower",
            "prr",
            "ebgm",
            "eb05",
            "calibrated_p",
            "robustness_score",
            "hazard_probability",
            "quarters_since_signal",
        ):
            value = getattr(self, name)
            if value is not None:
                facts[name] = float(value)
        return facts

    @property
    def headline(self) -> str:
        return f"{self.drug_name} / {self.reaction.title()}"

    @property
    def caveats(self) -> list[str]:
        """Qualifications that must accompany this gap wherever it is reported."""
        notes: list[str] = []
        if self.artefact_flags:
            notes.append("reporting-pattern concern: " + self.artefact_flags.replace(",", ", "))
        if self.relies_on_name_matching:
            notes.append(
                "reports for this drug are identified largely by reporter-supplied "
                "name rather than FDA substance harmonisation"
            )
        if self.reports < 10:
            notes.append(f"based on only {self.reports} reports")
        if self.calibrated_p is not None and self.calibrated_p >= 0.05:
            notes.append("does not reach significance against the empirically calibrated null")
        return notes


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """The closed set of facts a brief may be written from."""

    generated_on: dt.date
    run_id: str
    as_of_quarter: str
    data_vintage: str | None
    cohort_size: int
    gaps: tuple[GapEvidence, ...]
    total_open_gaps: int
    #: Cohort-level context.
    median_lead_quarters_by_criterion: dict[str, float | None] = field(default_factory=dict)
    median_leakage_inflation: float | None = None
    model_precision_at_50: float | None = None
    model_base_rate: float | None = None
    model_lift_at_50: float | None = None

    @property
    def allowed_numbers(self) -> set[float]:
        """Every number that may appear in the brief.

        Includes integers as floats and both rounded and unrounded forms, because a
        writer will naturally say "2.4" for 2.41. Verification allows a tolerance,
        so this set defines *which* facts are quotable, not their exact spelling.
        """
        allowed: set[float] = {float(self.cohort_size), float(self.total_open_gaps)}
        for gap in self.gaps:
            allowed.update(gap.numeric_facts.values())
        for value in self.median_lead_quarters_by_criterion.values():
            if value is not None:
                allowed.add(float(value))
        for value in (
            self.median_leakage_inflation,
            self.model_precision_at_50,
            self.model_base_rate,
            self.model_lift_at_50,
        ):
            if value is not None:
                allowed.add(float(value))
        return allowed

    def as_prompt_payload(self) -> dict[str, Any]:
        """The pack as a JSON-serialisable structure for a model prompt."""
        return {
            "generated_on": self.generated_on.isoformat(),
            "as_of_quarter": self.as_of_quarter,
            "data_vintage": self.data_vintage,
            "cohort_size": self.cohort_size,
            "total_open_gaps": self.total_open_gaps,
            "median_lead_quarters_by_criterion": self.median_lead_quarters_by_criterion,
            "median_leakage_inflation": self.median_leakage_inflation,
            "model_precision_at_50": self.model_precision_at_50,
            "model_base_rate": self.model_base_rate,
            "model_lift_at_50": self.model_lift_at_50,
            "gaps": [
                {
                    "drug": gap.drug_name,
                    "reaction": gap.reaction,
                    "reports": gap.reports,
                    "expected": round(gap.expected, 2),
                    "ror": gap.ror,
                    "ror_lower_95": gap.ror_ci_lower,
                    "ebgm": gap.ebgm,
                    "eb05": gap.eb05,
                    "calibrated_p": gap.calibrated_p,
                    "signal_first_fired": gap.signal_quarter,
                    "quarters_since_signal": gap.quarters_since_signal,
                    "criteria_firing": gap.criteria_firing,
                    "robustness_score": gap.robustness_score,
                    "caveats": gap.caveats,
                }
                for gap in self.gaps
            ],
        }


def build_evidence(
    warehouse: Warehouse, *, run_id: str, gap_count: int = DEFAULT_GAP_COUNT
) -> EvidencePack:
    """Assemble the evidence pack from the marts.

    Raises:
        ValueError: when the marts are absent, which means dbt has not run. Better
            to say so than to emit a brief with nothing in it.
    """
    if not warehouse.table_exists("mart_label_gap"):
        raise ValueError(
            "mart_label_gap does not exist; build the dbt models first (cd dbt && dbt build)"
        )

    manifest = warehouse.read_table("mart_run_manifest").head(1)
    gaps_table = warehouse.find_table("mart_label_gap")
    gaps_frame = warehouse.query_df(
        f"""
        SELECT drug_name, drug_unii, reaction, reports, expected, ror, ror_ci_lower,
               prr, ebgm, eb05, calibrated_p, signal_quarter, n_criteria_firing,
               robustness_score, artefact_flags, hazard_probability,
               relies_on_name_matching, as_of_quarter
        FROM {gaps_table}
        ORDER BY coalesce(hazard_probability, 0) DESC, eb05 DESC NULLS LAST
        LIMIT ?
        """,
        [gap_count],
    )
    total_gaps = int(warehouse.scalar(f"SELECT count(*) FROM {gaps_table}") or 0)

    leadtimes: dict[str, float | None] = {}
    leadtime_table = warehouse.find_table("mart_criterion_leadtime")
    if leadtime_table is not None:
        for criterion, median in warehouse.query(
            f"SELECT criterion, median_lead_quarters FROM {leadtime_table}"
        ):
            leadtimes[str(criterion)] = float(median) if median is not None else None

    median_inflation = None
    leakage_table = warehouse.find_table("mart_leakage_benchmark")
    if leakage_table is not None:
        median_inflation = warehouse.scalar(
            f"SELECT median(inflation_ratio) FROM {leakage_table} "
            "WHERE inflation_ratio IS NOT NULL AND isfinite(inflation_ratio)"
        )

    as_of = (
        str(gaps_frame["as_of_quarter"].iloc[0])
        if not gaps_frame.empty
        else str(manifest["last_quarter"].iloc[0])
        if not manifest.empty
        else "unknown"
    )

    def _manifest_value(column: str) -> float | None:
        if manifest.empty or column not in manifest.columns:
            return None
        value = manifest[column].iloc[0]
        return None if value is None else float(value)

    quarters_since = {}
    queue_table = warehouse.find_table("model_priority_queue")
    if queue_table is not None:
        quarters_since = dict(
            warehouse.query(
                f"SELECT drug_unii || '|' || reaction, quarters_since_signal "
                f"FROM {queue_table} WHERE run_id = ?",
                [run_id],
            )
        )

    gaps = tuple(
        GapEvidence(
            drug_name=str(row.drug_name),
            drug_unii=str(row.drug_unii),
            reaction=str(row.reaction),
            reports=int(_opt(row.reports) or 0),
            expected=_opt(row.expected) or 0.0,
            ror=_opt(row.ror),
            ror_ci_lower=_opt(row.ror_ci_lower),
            prr=_opt(row.prr),
            ebgm=_opt(row.ebgm),
            eb05=_opt(row.eb05),
            calibrated_p=_opt(row.calibrated_p),
            signal_quarter=None if row.signal_quarter is None else str(row.signal_quarter),
            quarters_since_signal=_opt_int(quarters_since.get(f"{row.drug_unii}|{row.reaction}")),
            criteria_firing=int(_opt(row.n_criteria_firing) or 0),
            robustness_score=_opt(row.robustness_score),
            artefact_flags=str(row.artefact_flags or ""),
            hazard_probability=_opt(row.hazard_probability),
            relies_on_name_matching=bool(row.relies_on_name_matching),
        )
        for row in gaps_frame.itertuples()
    )

    return EvidencePack(
        generated_on=dt.date.today(),
        run_id=run_id,
        as_of_quarter=as_of,
        data_vintage=(
            str(manifest["openfda_last_updated"].iloc[0])[:10] if not manifest.empty else None
        ),
        cohort_size=int(_manifest_value("drugs") or 0),
        gaps=gaps,
        total_open_gaps=total_gaps,
        median_lead_quarters_by_criterion=leadtimes,
        median_leakage_inflation=(
            round(float(median_inflation), 3) if median_inflation is not None else None
        ),
        model_precision_at_50=_manifest_value("model_precision_at_50"),
        model_base_rate=_manifest_value("model_base_rate"),
        model_lift_at_50=_manifest_value("model_lift_at_50"),
    )


def _opt(value: Any) -> float | None:
    """A finite float, or None.

    Values read off a DataFrame are typed by pandas as a union of every dtype it
    supports, so the conversion goes through ``str`` -- which accepts all of them --
    rather than through ``float``, which the checker cannot prove is applicable.
    """
    import math

    if value is None:
        return None
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _opt_int(value: Any) -> int | None:
    numeric = _opt(value)
    return int(numeric) if numeric is not None else None
