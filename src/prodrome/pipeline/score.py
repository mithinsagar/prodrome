"""Stage 2: turn counts into calibrated disproportionality statistics.

Everything here is pure computation over what stage 1 landed, so it re-runs in
seconds. That matters: this is where a change to an estimator or a threshold lands,
and iterating on statistics must not require re-spending API quota.

Point-in-time discipline, applied to the priors too
---------------------------------------------------
Both shrinkage layers in this project learn from the data, which creates a second,
subtler leakage risk than the one in :mod:`prodrome.latency.leakage`. The
gamma-Poisson prior and the empirical null are *fitted from the cohort*, so fitting
them once on the final snapshot and applying them backwards would let 2026 reporting
behaviour shrink a 2018 estimate. The 2018 estimate would then contain information
from the future, which is precisely what the project exists to avoid.

So both are refitted per quarter, on that quarter's cells only. It costs more
compute and it is the only correct thing to do. The fitted parameters are written to
the warehouse per quarter, both because a shrunk estimate is uninterpretable without
the prior that produced it, and because drift in those parameters over time is
itself a finding worth plotting.

Where the empirical null comes from
-----------------------------------
Empirical calibration needs a set of drug-event pairs known not to be causally
associated. Curated negative-control sets exist for specific research questions but
not for an arbitrary 55-drug cohort, so prodrome supports two sources:

*Curated.* Pairs listed in ``conf/negative_controls.yml``. Preferred when available,
and the method is then exactly Schuemie et al. (2014).

*Empirical null on the central mass.* Absent a curated set, the null is fitted to the
trimmed centre of the log-ROR distribution across all pairs. The justification is
the same one that underpins Efron's empirical null and the noise component of
DuMouchel's mixture: of the tens of thousands of drug-event pairs in a cohort like
this, the overwhelming majority are not causal, so the bulk of the distribution *is*
the null. Trimming the tails keeps the genuine signals from dragging the fit.

The second is weaker than the first, and which one was used is recorded per quarter
rather than left to be inferred.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from prodrome.config import Config
from prodrome.stats.calibration import (
    InsufficientControlsError,
    NullDistribution,
    fit_null,
)
from prodrome.stats.contingency import Contingency
from prodrome.stats.criteria import CRITERIA, evaluate_criteria
from prodrome.stats.disproportionality import Z_95, score
from prodrome.stats.ebgm import GpsPrior, fit_prior, score_cells
from prodrome.warehouse import Warehouse

logger = logging.getLogger(__name__)

#: Quantile trimmed from each tail when fitting the empirical null to the central
#: mass. 0.15 removes the genuine signals at the top and the strongly protective
#: artefacts at the bottom while retaining the bulk that constitutes the null.
CENTRAL_MASS_TRIM = 0.15

#: A quarter with fewer cells than this cannot support either a mixture prior or a
#: null fit; its cells are still scored, just without shrinkage or calibration.
MIN_CELLS_PER_QUARTER = 50


@dataclass
class ScoreResult:
    """Summary of a scoring pass."""

    run_id: str
    quarters_scored: int = 0
    rows_written: int = 0
    priors_fitted: int = 0
    nulls_fitted: int = 0
    nulls_from_curated_controls: int = 0
    quarters_without_shrinkage: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [
            f"run {self.run_id}",
            f"  quarters scored           {self.quarters_scored}",
            f"  statistic rows            {self.rows_written:,}",
            f"  GPS priors fitted         {self.priors_fitted}",
            f"  empirical nulls fitted    {self.nulls_fitted}"
            f"  ({self.nulls_from_curated_controls} from curated negative controls, "
            f"{self.nulls_fitted - self.nulls_from_curated_controls} from central mass)",
        ]
        if self.quarters_without_shrinkage:
            lines.append(
                f"  no shrinkage (too sparse) {len(self.quarters_without_shrinkage)} quarters: "
                + ", ".join(self.quarters_without_shrinkage[:6])
            )
        return lines


def _log_ror_and_se(table: Contingency) -> tuple[float, float]:
    """log(ROR) and its standard error, or NaN when a cell is empty."""
    if 0 in (table.a, table.b, table.c, table.d):
        return math.nan, math.nan
    log_ror = math.log((table.a * table.d) / (table.b * table.c))
    se = math.sqrt(1 / table.a + 1 / table.b + 1 / table.c + 1 / table.d)
    return log_ror, se


def _fit_quarter_null(
    log_rors: np.ndarray,
    ses: np.ndarray,
    control_mask: np.ndarray,
) -> tuple[NullDistribution | None, bool]:
    """Fit the empirical null for one quarter.

    Returns the null and whether curated controls were used.
    """
    if control_mask.any():
        try:
            return fit_null(log_rors[control_mask], ses[control_mask]), True
        except InsufficientControlsError as exc:
            logger.info("curated controls unusable this quarter (%s); using central mass", exc)

    usable = np.isfinite(log_rors) & np.isfinite(ses) & (ses > 0)
    if usable.sum() < 20:
        return None, False
    values = log_rors[usable]
    low, high = np.quantile(values, [CENTRAL_MASS_TRIM, 1 - CENTRAL_MASS_TRIM])
    central = usable & (log_rors >= low) & (log_rors <= high)
    if central.sum() < 20:
        return None, False
    try:
        return fit_null(log_rors[central], ses[central]), False
    except InsufficientControlsError:
        return None, False


def run_score(warehouse: Warehouse, config: Config, *, run_id: str) -> ScoreResult:  # noqa: PLR0915
    """Score every ingested contingency cell, quarter by quarter."""
    from scipy import stats as sps

    result = ScoreResult(run_id=run_id)
    control_pairs = {(u.upper(), r.upper()) for u, r in config.negative_controls}

    # Re-running a stage must replace its output, not merge with it. See
    # Warehouse.clear_run for the bug that made this necessary.
    cleared = warehouse.clear_runs(
        ["stat_disproportionality", "raw_gps_prior", "raw_empirical_null"], run_id
    )
    if cleared:
        logger.info("cleared previous output for this run: %s", cleared)

    quarters = [
        row[0]
        for row in warehouse.query(
            "SELECT DISTINCT as_of_quarter FROM raw_contingency WHERE run_id = ? "
            "ORDER BY as_of_quarter",
            [run_id],
        )
    ]
    if not quarters:
        logger.warning("no contingency cells for run %s; nothing to score", run_id)
        return result

    criterion_keys = [c.key for c in CRITERIA]

    for quarter_label in quarters:
        cells = warehouse.query(
            "SELECT drug_unii, reaction, a, b, c, d FROM raw_contingency "
            "WHERE run_id = ? AND as_of_quarter = ? ORDER BY drug_unii, reaction",
            [run_id, quarter_label],
        )
        if not cells:
            continue

        tables = [Contingency(a=int(a), b=int(b), c=int(c), d=int(d)) for _, _, a, b, c, d in cells]
        scored = [score(t) for t in tables]
        counts = np.array([t.a for t in tables], dtype=float)
        expected = np.array([t.expected for t in tables], dtype=float)

        # ---- gamma-Poisson shrinkage, refitted on this quarter ---------------
        prior: GpsPrior | None = None
        ebgm = eb05 = eb95 = noise_weight = None
        if len(tables) >= MIN_CELLS_PER_QUARTER:
            try:
                prior = fit_prior(counts, expected)
                gps = score_cells(counts, expected, prior=prior)
                ebgm, eb05, eb95 = gps.ebgm, gps.eb05, gps.eb95
                noise_weight = gps.posterior_weight
                result.priors_fitted += 1
                warehouse.append_rows(
                    "raw_gps_prior",
                    [{"run_id": run_id, "as_of_quarter": quarter_label, **prior.as_row()}],
                )
            except (ValueError, FloatingPointError) as exc:
                logger.warning("GPS prior failed for %s: %s", quarter_label, exc)
        if prior is None:
            result.quarters_without_shrinkage.append(quarter_label)

        # ---- empirical calibration, refitted on this quarter -----------------
        log_rors = np.array([_log_ror_and_se(t)[0] for t in tables], dtype=float)
        ses = np.array([_log_ror_and_se(t)[1] for t in tables], dtype=float)
        control_mask = np.array(
            [(str(u).upper(), str(r).upper()) in control_pairs for u, r, *_ in cells]
        )
        null, from_curated = _fit_quarter_null(log_rors, ses, control_mask)
        if null is not None:
            result.nulls_fitted += 1
            result.nulls_from_curated_controls += int(from_curated)
            warehouse.append_rows(
                "raw_empirical_null",
                [
                    {
                        "run_id": run_id,
                        "as_of_quarter": quarter_label,
                        "mu": null.mu,
                        "sd": null.sd,
                        "n_controls": null.n_controls,
                        "converged": null.converged,
                    }
                ],
            )

        rows: list[dict[str, object]] = []
        for index, (drug_unii, reaction, *_rest) in enumerate(cells):
            measures = scored[index]
            fired = evaluate_criteria(measures)
            log_ror, se = float(log_rors[index]), float(ses[index])
            raw_p = (
                float(2 * sps.norm.sf(abs(log_ror) / se))
                if math.isfinite(log_ror) and math.isfinite(se) and se > 0
                else None
            )
            calibrated_p = null.calibrated_p(log_ror, se) if null is not None else None
            row: dict[str, object] = {
                "run_id": run_id,
                "drug_unii": drug_unii,
                "reaction": reaction,
                "as_of_quarter": quarter_label,
                "a": measures.a,
                "b": measures.b,
                "c": measures.c,
                "d": measures.d,
                "n": measures.n,
                "expected": measures.expected,
                "prr": _finite(measures.prr),
                "prr_ci_lower": _finite(measures.prr_ci.lower),
                "prr_ci_upper": _finite(measures.prr_ci.upper),
                "ror": _finite(measures.ror),
                "ror_ci_lower": _finite(measures.ror_ci.lower),
                "ror_ci_upper": _finite(measures.ror_ci.upper),
                "chi2_yates": _finite(measures.chi2_yates),
                "rrr": _finite(measures.rrr),
                "log2_oe_shrunk": _finite(measures.log2_oe_shrunk),
                "log2_oe_ci_lower": _finite(measures.log2_oe_ci.lower),
                "log2_oe_ci_upper": _finite(measures.log2_oe_ci.upper),
                "ebgm": _finite(ebgm[index]) if ebgm is not None else None,
                "eb05": _finite(eb05[index]) if eb05 is not None else None,
                "eb95": _finite(eb95[index]) if eb95 is not None else None,
                "gps_posterior_noise_weight": (
                    _finite(noise_weight[index]) if noise_weight is not None else None
                ),
                "calibrated_p": _finite(calibrated_p) if calibrated_p is not None else None,
                "raw_p": _finite(raw_p) if raw_p is not None else None,
                "degenerate": measures.degenerate,
                "n_criteria_firing": sum(fired.values()),
            }
            for key in criterion_keys:
                row[f"fired_{key}"] = fired[key]
            rows.append(row)

        result.rows_written += warehouse.append_batched("stat_disproportionality", rows)
        result.quarters_scored += 1
        logger.info(
            "%s: scored %d cells%s",
            quarter_label,
            len(rows),
            "" if prior is None else f", GPS noise weight {prior.noise_weight:.3f}",
        )

    return result


def _finite(value: float | None) -> float | None:
    """Map NaN and infinity to NULL.

    DuckDB stores NaN happily, but a NaN silently propagates through every
    downstream aggregate and comparison, turning one degenerate cell into a
    column of nulls-that-look-like-numbers. Converting at the boundary keeps
    "not computable" distinguishable from "computed as zero".
    """
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


__all__ = ["Z_95", "ScoreResult", "run_score"]
