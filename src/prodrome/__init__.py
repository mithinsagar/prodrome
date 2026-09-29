"""prodrome: a signal-to-label latency engine for FDA adverse event data.

Conventional pharmacovigilance tooling answers "which drug-reaction pairs are
reported disproportionately". prodrome answers a different question: **how much
warning does each signal-detection criterion actually buy you before the label
changes** -- measured point-in-time, so the answer is not contaminated by the
labelling event it is trying to predict.

See ``docs/METHODS.md`` for the statistical argument and ``ARCHITECTURE.md`` for
how the pieces fit together.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
