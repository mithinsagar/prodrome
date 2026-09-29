"""The three pipeline stages.

Split by what they cost and what they depend on, so that the expensive stage runs
rarely and the cheap ones iterate freely:

``ingest``
    Talks to the two APIs. The only stage that spends request quota, and the only
    one that can fail because of something outside this repository. Fully
    resumable: every response is cached on disk, so a re-run after a failure costs
    only the requests that had not yet succeeded.
``score``
    Pure computation over the ingested counts. Re-runnable in seconds, which
    matters because this is where a change to an estimator or a threshold lands.
``latency``
    Joins the scored series to the label timeline and fits the models. Depends on
    both of the above and on nothing external.

Keeping these separate is what makes it possible to re-derive every published
number without re-downloading anything.
"""

from prodrome.pipeline.ingest import IngestResult, run_ingest
from prodrome.pipeline.latency import LatencyResult, run_latency
from prodrome.pipeline.score import ScoreResult, run_score

__all__ = [
    "IngestResult",
    "LatencyResult",
    "ScoreResult",
    "run_ingest",
    "run_latency",
    "run_score",
]
