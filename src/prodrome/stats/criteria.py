"""The registry of published signal-detection criteria.

This module is the conceptual centre of prodrome. Every other project that
computes PRR and ROR picks *one* threshold rule, applies it, and presents the
result as "the signals". But the rules disagree, and they disagree in a specific
direction: the odds-ratio rules fire earlier and noisier, the MHRA triple fires
later and cleaner. Whether that trade is worth making is an empirical question
about lead time, and it has never been answered on open data.

So criteria are first-class objects here. Each one is evaluated independently on
every drug-reaction-quarter cell, its own signal-onset quarter is recorded, and
the latency layer reports the lead time each rule bought against the label
change that eventually happened -- alongside how many false alarms it raised to
buy it.

Adding a criterion is a one-line registry entry and changes no other code.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from prodrome.stats.disproportionality import Disproportionality


@dataclass(frozen=True, slots=True)
class Criterion:
    """A named, citable threshold rule over a scored contingency table.

    Attributes:
        key: stable identifier used as a warehouse column and dashboard facet.
            Never rename one: onset history is keyed on it.
        label: short human name for the dashboard.
        source: the body or paper the thresholds come from. Every rule in this
            registry is somebody's published rule, not one prodrome invented.
        min_count: minimum co-occurrence count `a` the rule requires.
        rule: predicate over the scored table.
        rationale: why the rule is shaped the way it is, surfaced in the docs
            table so a reviewer can judge the comparison without reading code.
    """

    key: str
    label: str
    source: str
    min_count: int
    rule: Callable[[Disproportionality], bool]
    rationale: str

    def holds(self, d: Disproportionality) -> bool:
        """Evaluate the rule, enforcing ``min_count`` first.

        The count gate is applied here rather than inside each predicate so that
        a rule can never accidentally fire on a single report.
        """
        if d.a < self.min_count:
            return False
        try:
            result = self.rule(d)
        except (ValueError, ZeroDivisionError):
            return False
        return bool(result) and not _any_nan_in_use(d, self)


def _any_nan_in_use(d: Disproportionality, criterion: Criterion) -> bool:
    """Guard against a NaN estimator being read as a satisfied threshold.

    A comparison against NaN is False in Python, so ``prr >= 2`` fails closed
    already. The lower-bound rules are the dangerous ones -- this keeps the
    intent explicit rather than relying on that.
    """
    relevant = {
        "mhra_prr": (d.prr, d.chi2_yates),
        "ema_ror025": (d.ror_ci.lower,),
        "who_oe025": (d.log2_oe_ci.lower,),
        "dubious_prr_only": (d.prr,),
    }.get(criterion.key, ())
    return any(isinstance(v, float) and math.isnan(v) for v in relevant)


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="mhra_prr",
        label="MHRA triple",
        source="Evans, Waller & Davis (2001)",
        min_count=3,
        rule=lambda d: d.prr >= 2.0 and d.chi2_yates >= 4.0,
        rationale=(
            "Requires effect size and significance together, which suppresses the "
            "small-count noise that dominates spontaneous-report data. The cost is "
            "lead time, and quantifying that cost is the point of this project."
        ),
    ),
    Criterion(
        key="ema_ror025",
        label="EMA ROR lower bound",
        source="EMA/EudraVigilance signal-detection guidance",
        min_count=3,
        rule=lambda d: d.ror_ci.lower > 1.0,
        rationale=(
            "A single interval-based gate. Fires as soon as the data rule out no "
            "association at 95%, so it is systematically earlier than the MHRA "
            "triple and systematically less specific."
        ),
    ),
    Criterion(
        key="who_oe025",
        label="Shrinkage O/E lower bound",
        source="Noren et al. (2006), UMC practice",
        min_count=1,
        rule=lambda d: d.log2_oe_ci.lower > 0.0,
        rationale=(
            "The only rule here that is defined at a = 1, because shrinkage makes "
            "the estimate conservative instead of explosive. Included to test "
            "whether Bayesian shrinkage buys lead time without buying noise."
        ),
    ),
    Criterion(
        key="dubious_prr_only",
        label="PRR >= 2, no gate",
        source="common practice in the published FAERS literature",
        min_count=1,
        rule=lambda d: d.prr >= 2.0,
        rationale=(
            "Deliberately included as a negative control on method. A large share "
            "of published FAERS disproportionality analyses apply exactly this, "
            "with no count or significance gate. prodrome reports its false-alarm "
            "rate next to the others so the comparison is visible rather than "
            "asserted."
        ),
    ),
)

CRITERIA_BY_KEY: dict[str, Criterion] = {c.key: c for c in CRITERIA}


def evaluate_criteria(d: Disproportionality) -> dict[str, bool]:
    """Evaluate every registered criterion against one scored table."""
    return {c.key: c.holds(d) for c in CRITERIA}
