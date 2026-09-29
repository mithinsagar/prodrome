"""The 2x2 table every disproportionality measure is computed from.

Notation follows the pharmacovigilance literature throughout this package::

                     reaction R      not R
    drug D               a             b        a + b = reports mentioning D
    not D                c             d        c + d
                     -------       -------
                      a + c         b + d       N = a + b + c + d

`a` is the co-occurrence count. It is the only cell openFDA reports directly;
the other three are derived from three marginal totals, which is why
:meth:`Contingency.from_marginals` is the constructor the ingest layer uses.

A note on the unit of analysis
------------------------------
openFDA's ``count=patient.reaction.reactionmeddrapt.exact`` counts *reaction
occurrences*, not reports: a report listing five reactions contributes five.
The marginal totals from ``meta.results.total``, by contrast, count *reports*.
Mixing the two would silently inflate `a` relative to `a + b`.

prodrome therefore derives every cell from report-level counts obtained with
``limit=1`` + ``meta.results.total``, and uses the occurrence-level ``count``
aggregation only to *discover* which reaction terms are worth querying. See
``docs/METHODS.md`` for the measurement argument.
"""

from __future__ import annotations

from dataclasses import dataclass


class ContingencyError(ValueError):
    """A set of marginals cannot describe a real 2x2 table."""


@dataclass(frozen=True, slots=True)
class Contingency:
    """A validated 2x2 disproportionality table.

    Cells are report counts and therefore non-negative integers. Instances are
    frozen because downstream estimators cache on them.
    """

    a: int
    b: int
    c: int
    d: int

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise ContingencyError(f"cell {name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ContingencyError(f"cell {name} must be non-negative, got {value}")
        if self.n == 0:
            raise ContingencyError("table is empty: N == 0")

    @classmethod
    def from_marginals(
        cls,
        *,
        co_occurrence: int,
        drug_total: int,
        reaction_total: int,
        grand_total: int,
    ) -> Contingency:
        """Build a table from the four counts openFDA can actually answer.

        Args:
            co_occurrence: reports naming both the drug and the reaction (`a`).
            drug_total: reports naming the drug, any reaction (`a + b`).
            reaction_total: reports naming the reaction, any drug (`a + c`).
            grand_total: all reports in the window (`N`).

        Raises:
            ContingencyError: if the marginals are mutually inconsistent, which
                in practice means the four queries did not observe the same
                window -- openFDA reindexed between them, or a date filter was
                applied to one and not another. Failing loudly here is the
                point: a silently negative `b` produces a plausible-looking
                ratio that is pure noise.
        """
        b = drug_total - co_occurrence
        c = reaction_total - co_occurrence
        d = grand_total - co_occurrence - b - c
        problems = []
        if co_occurrence > drug_total:
            problems.append(f"a ({co_occurrence}) > drug_total ({drug_total})")
        if co_occurrence > reaction_total:
            problems.append(f"a ({co_occurrence}) > reaction_total ({reaction_total})")
        if d < 0:
            problems.append(
                f"d < 0: grand_total ({grand_total}) is smaller than the implied "
                f"union of the margins ({co_occurrence + b + c})"
            )
        if problems:
            raise ContingencyError("inconsistent marginals: " + "; ".join(problems))
        return cls(a=co_occurrence, b=b, c=c, d=d)

    @property
    def n(self) -> int:
        """Grand total `N`."""
        return self.a + self.b + self.c + self.d

    @property
    def drug_total(self) -> int:
        """`a + b`: reports naming the drug."""
        return self.a + self.b

    @property
    def reaction_total(self) -> int:
        """`a + c`: reports naming the reaction."""
        return self.a + self.c

    @property
    def expected(self) -> float:
        """`E`, the count expected if drug and reaction were independent.

        This is the baseline for every Bayesian measure in this package, and
        the denominator of the relative reporting ratio.
        """
        return self.drug_total * self.reaction_total / self.n

    @property
    def has_empty_margin(self) -> bool:
        """True when a margin is zero, so log-scale estimators are undefined.

        Callers should branch on this rather than relying on continuity
        corrections, which quietly turn "no information" into "no signal".
        """
        return 0 in (self.drug_total, self.reaction_total, self.b + self.d, self.c + self.d)
