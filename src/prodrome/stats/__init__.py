"""Disproportionality estimators and their calibration.

The public surface is deliberately small: build a :class:`Contingency`, hand it
to :func:`score`, and read a :class:`Disproportionality`. Everything else in
this package is either a specific estimator or a correction applied to one.
"""

from prodrome.stats.contingency import Contingency, ContingencyError
from prodrome.stats.criteria import CRITERIA, Criterion, evaluate_criteria
from prodrome.stats.disproportionality import Disproportionality, score

__all__ = [
    "CRITERIA",
    "Contingency",
    "ContingencyError",
    "Criterion",
    "Disproportionality",
    "evaluate_criteria",
    "score",
]
