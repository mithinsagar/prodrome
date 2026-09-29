"""Time-to-label survival analysis and the supervised prioritisation model.

Two questions, two methods
--------------------------
**"How long does labelling take, per criterion?"** is a survival question, answered
with Kaplan-Meier. Censored pairs -- signal fired, label not yet changed -- carry
real information about the labelling rate and must not be dropped or recoded as
negatives. KM handles them correctly and yields a median time-to-label per
criterion with a confidence band.

**"Which currently-unlabelled signals are most likely to be labelled soon?"** is a
prediction question, and it is the one a signal-management team actually needs,
because it turns a list of thousands of disproportionality hits into a ranked
queue. It is answered with a discrete-time hazard model: one row per
pair-quarter at risk, a binary outcome for "labelled this quarter", and a logistic
fit. That formulation handles censoring natively -- a pair simply stops
contributing rows once it is labelled or the data ends -- and yields a calibrated
per-quarter probability that composes into any horizon.

Why the split is temporal
-------------------------
A random train/test split would be catastrophically optimistic here. Pairs are
observed repeatedly across quarters, so random splitting puts later quarters of
the same pair in training and earlier ones in test -- the model sees the answer.
Worse, it leaks the *era*: FAERS reporting volume and FDA labelling behaviour both
drift, so a random split lets the model calibrate against the future.

prodrome splits on quarter: everything before ``holdout_from_quarter`` trains,
everything at or after it is held out. That is the only split that answers the
question actually being asked, which is whether the model would have been useful
prospectively.

What is reported
----------------
Discrimination (AUC) is reported but is not the headline. The headline is
**precision@k** at the horizon, because a reviewer works a queue of fixed length:
"of the 50 signals this ranked highest, how many were labelled within two years"
is the number that decides whether the model is worth using. Calibration is
reported alongside, since a probability that is not calibrated cannot be used to
set a threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Features the hazard model is allowed to use. Explicit rather than
#: "every numeric column", because the panel also carries outcome-derived and
#: identifying columns, and a wildcard would eventually leak one of them in.
HAZARD_FEATURES: tuple[str, ...] = (
    "log2_oe_shrunk",
    "log_ebgm",
    "log_eb05",
    "log_prr",
    "log_ror_lower",
    "chi2_yates",
    "log_a",
    "log_expected",
    "quarters_since_signal",
    "firing_fraction",
    "reporter_concentration",
    "consumer_share",
    "spike_ratio",
    "n_criteria_firing",
    "drug_report_share",
)

#: Minimum events required before a fit is attempted. Below this the coefficients
#: are noise and a fitted model would be worse than useless -- it would look like
#: a result.
MIN_EVENTS_TO_FIT = 30


@dataclass(frozen=True, slots=True)
class KaplanMeierCurve:
    """A survival curve for one criterion."""

    criterion: str
    n_pairs: int
    n_events: int
    n_censored: int
    median_quarters_to_label: float | None
    #: Cumulative probability of being labelled by 4, 8 and 12 quarters.
    labelled_by_1y: float
    labelled_by_2y: float
    labelled_by_3y: float
    timeline: tuple[float, ...] = field(default=())
    survival: tuple[float, ...] = field(default=())

    def as_row(self) -> dict[str, object]:
        return {
            "criterion": self.criterion,
            "n_pairs": self.n_pairs,
            "n_events": self.n_events,
            "n_censored": self.n_censored,
            "median_quarters_to_label": self.median_quarters_to_label,
            "labelled_by_1y": round(self.labelled_by_1y, 4),
            "labelled_by_2y": round(self.labelled_by_2y, 4),
            "labelled_by_3y": round(self.labelled_by_3y, 4),
        }


def kaplan_meier(durations: np.ndarray, events: np.ndarray, *, criterion: str) -> KaplanMeierCurve:
    """Fit a Kaplan-Meier curve to time-to-label data.

    Args:
        durations: quarters from signal onset to label change or censoring.
        events: 1 where a label change was observed, 0 where censored.
    """
    from lifelines import KaplanMeierFitter

    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    n_events = int(events.sum())

    if durations.size == 0 or n_events == 0:
        return KaplanMeierCurve(
            criterion=criterion,
            n_pairs=int(durations.size),
            n_events=n_events,
            n_censored=int(durations.size - n_events),
            median_quarters_to_label=None,
            labelled_by_1y=0.0,
            labelled_by_2y=0.0,
            labelled_by_3y=0.0,
        )

    fitter = KaplanMeierFitter()
    fitter.fit(durations, event_observed=events, label=criterion)
    median = float(fitter.median_survival_time_)

    def labelled_by(quarters: int) -> float:
        # predict() returns the survival probability; the complement is the
        # cumulative incidence of labelling, which is what a reader wants.
        return float(1.0 - fitter.predict(quarters))

    return KaplanMeierCurve(
        criterion=criterion,
        n_pairs=int(durations.size),
        n_events=n_events,
        n_censored=int(durations.size - n_events),
        median_quarters_to_label=median if np.isfinite(median) else None,
        labelled_by_1y=labelled_by(4),
        labelled_by_2y=labelled_by(8),
        labelled_by_3y=labelled_by(12),
        timeline=tuple(float(t) for t in fitter.timeline),
        survival=tuple(float(s) for s in fitter.survival_function_.iloc[:, 0]),
    )


@dataclass(frozen=True, slots=True)
class HazardModelReport:
    """Everything needed to judge whether the prioritisation model is usable."""

    n_train_rows: int
    n_test_rows: int
    n_train_events: int
    n_test_events: int
    holdout_from_quarter: str
    features: tuple[str, ...]
    coefficients: dict[str, float]
    roc_auc: float
    average_precision: float
    base_rate: float
    #: precision@k for a reviewer working a queue of k signals.
    precision_at_k: dict[int, float]
    #: Observed vs predicted event rate by predicted-probability decile.
    calibration: tuple[tuple[float, float, int], ...]
    fitted: bool
    note: str = ""

    @property
    def lift_at_50(self) -> float:
        """How much better than random the top 50 is.

        This is the single number that says whether the model earns its place: a
        lift of 1.0 means ranking by the model is no better than ranking at random,
        and the whole exercise reduces to the disproportionality score alone.
        """
        precision = self.precision_at_k.get(50, 0.0)
        return precision / self.base_rate if self.base_rate > 0 else 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_train_events": self.n_train_events,
            "n_test_events": self.n_test_events,
            "holdout_from_quarter": self.holdout_from_quarter,
            "roc_auc": round(self.roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "base_rate": round(self.base_rate, 5),
            "precision_at_10": round(self.precision_at_k.get(10, 0.0), 4),
            "precision_at_50": round(self.precision_at_k.get(50, 0.0), 4),
            "precision_at_100": round(self.precision_at_k.get(100, 0.0), 4),
            "lift_at_50": round(self.lift_at_50, 3),
            "fitted": self.fitted,
            "note": self.note,
        }


def _precision_at_k(
    y_true: np.ndarray, scores: np.ndarray, ks: tuple[int, ...]
) -> dict[int, float]:
    order = np.argsort(-scores, kind="stable")
    ranked = y_true[order]
    out: dict[int, float] = {}
    for k in ks:
        if k <= ranked.size:
            out[k] = float(ranked[:k].mean())
    return out


def _calibration_table(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> tuple[tuple[float, float, int], ...]:
    """(mean predicted, observed rate, n) per predicted-probability decile.

    Quantile bins rather than equal-width: predicted hazards are heavily
    right-skewed, so equal-width bins would put almost every row in the first one
    and say nothing.
    """
    if probabilities.size == 0:
        return ()
    ranks = pd.qcut(pd.Series(probabilities).rank(method="first"), q=bins, labels=False)
    rows: list[tuple[float, float, int]] = []
    for bin_index in range(bins):
        mask = (ranks == bin_index).to_numpy()
        if not mask.any():
            continue
        rows.append(
            (
                float(probabilities[mask].mean()),
                float(y_true[mask].mean()),
                int(mask.sum()),
            )
        )
    return tuple(rows)


def fit_hazard_model(
    panel: pd.DataFrame,
    *,
    holdout_from_quarter: str,
    features: tuple[str, ...] = HAZARD_FEATURES,
    outcome_column: str = "labelled_this_quarter",
    quarter_column: str = "as_of_quarter",
) -> tuple[HazardModelReport, Any]:
    """Fit the discrete-time hazard model on a pair-quarter panel.

    Args:
        panel: one row per (drug, reaction, quarter) at risk.
        holdout_from_quarter: quarter label; rows at or after it are held out.
        features: columns to use. Missing columns are dropped with a warning
            rather than raising, so a run with a diagnostic disabled still fits.

    Returns:
        The report and the fitted estimator (None when not fitted).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    available = [f for f in features if f in panel.columns]
    missing = [f for f in features if f not in panel.columns]
    if missing:
        logger.warning("hazard model: features absent from the panel, skipped: %s", missing)
    if not available:
        return (
            HazardModelReport(
                0,
                0,
                0,
                0,
                holdout_from_quarter,
                (),
                {},
                0.5,
                0.0,
                0.0,
                {},
                (),
                False,
                "no usable features in the panel",
            ),
            None,
        )

    frame = panel.copy()
    # A missing diagnostic is "no evidence of a problem", which on these features
    # means the neutral value 0 after standardisation, not a dropped row: dropping
    # rows would silently restrict the risk set.
    frame[available] = frame[available].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_mask = frame[quarter_column] < holdout_from_quarter
    train, test = frame[train_mask], frame[~train_mask]

    n_train_events = int(train[outcome_column].sum())
    if n_train_events < MIN_EVENTS_TO_FIT:
        return (
            HazardModelReport(
                len(train),
                len(test),
                n_train_events,
                int(test[outcome_column].sum()),
                holdout_from_quarter,
                tuple(available),
                {},
                0.5,
                0.0,
                float(test[outcome_column].mean()) if len(test) else 0.0,
                {},
                (),
                False,
                f"only {n_train_events} training events, need {MIN_EVENTS_TO_FIT}",
            ),
            None,
        )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            # Plain L2 logistic, deliberately *not* class-weighted. Balancing a
            # rare-event outcome is the reflex here, and it is wrong for this use:
            # it shifts the intercept so predicted probabilities no longer match
            # observed rates, and a probability that is not calibrated cannot be
            # used to set a review threshold -- which is the whole point of the
            # model. Ranking, and therefore precision@k, is invariant to the
            # intercept, so balancing would buy nothing and cost the calibration.
            ("logit", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
        ]
    )
    model.fit(train[available], train[outcome_column])

    coefficients = dict(
        zip(available, (float(c) for c in model.named_steps["logit"].coef_[0]), strict=True)
    )

    if len(test) == 0 or test[outcome_column].nunique() < 2:
        return (
            HazardModelReport(
                len(train),
                len(test),
                n_train_events,
                int(test[outcome_column].sum()),
                holdout_from_quarter,
                tuple(available),
                coefficients,
                0.5,
                0.0,
                float(test[outcome_column].mean()) if len(test) else 0.0,
                {},
                (),
                True,
                "holdout has no outcome variation; metrics not computable",
            ),
            model,
        )

    probabilities = model.predict_proba(test[available])[:, 1]
    y_true = test[outcome_column].to_numpy()

    return (
        HazardModelReport(
            n_train_rows=len(train),
            n_test_rows=len(test),
            n_train_events=n_train_events,
            n_test_events=int(y_true.sum()),
            holdout_from_quarter=holdout_from_quarter,
            features=tuple(available),
            coefficients=coefficients,
            roc_auc=float(roc_auc_score(y_true, probabilities)),
            average_precision=float(average_precision_score(y_true, probabilities)),
            base_rate=float(y_true.mean()),
            precision_at_k=_precision_at_k(y_true, probabilities, (10, 25, 50, 100, 200)),
            calibration=_calibration_table(y_true, probabilities),
            fitted=True,
        ),
        model,
    )
