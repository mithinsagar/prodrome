# Runs and reproducibility

Everything under `data/` is derived and gitignored, because it is all reproducible
from the two public APIs. What makes a published figure auditable is not the data file
but the provenance recorded alongside it.

## What a run records

Every fact row carries a `run_id`. Every run carries:

| field | why it matters |
|---|---|
| `config_digest` | sha256 of the resolved analysis plan — the cohort, window and every threshold |
| `openfda_last_updated` | the data vintage. openFDA reindexes on its own cadence, so the same query returns different numbers on different days |
| `first_quarter` / `last_quarter` | the evaluation grid |
| `has_openfda_key` | determines whether count aggregations were capped at 100 terms or 1,000, which changes how many cells needed individual queries |
| `embed_backend` | the semantic threshold is calibrated per backend; one tuned on the other is meaningless |
| `requests`, `cache_hits`, `retries` | how much traffic produced this, and how much came from cache |

`mart_run_manifest` joins that to the cohort's identification coverage and the model's
held-out metrics, and the dashboard prints it in the header. A figure whose run and
data vintage are not on screen cannot be audited.

## Reproducing a run

```bash
git checkout <commit>        # the config is version-controlled with the code
make pipeline marts export
```

With a warm cache this re-derives every number without contacting either API. With a
cold cache the numbers may differ from the original run, because openFDA will have
reindexed — which is why `openfda_last_updated` is recorded rather than assumed.

## Rebuilding one historical run's marts

```bash
cd dbt && dbt build --vars '{run_id: <run_id>}'
```

`dbt` defaults to `run_id: latest`, which resolves to the most recent **finished**
run. Unfinished runs are excluded so a crashed or in-flight run is never published as
though it were complete.

## The cache

`data/cache/` holds every upstream response, content-addressed by request. It is the
most valuable thing to keep: it turns a multi-hour backfill into an incremental run
that fetches only the newest quarter, and it is what makes re-running after a failure
cheap. The weekly workflow persists it through GitHub Actions' cache, keyed per run
with a prefix restore so a new key still starts warm.

The API key is deliberately excluded from the cache key, so a cache built without one
is reused by a run that has one.

```bash
make clean        # build artefacts, keeps the cache
make clean-cache  # also drops the cache, forcing a full refetch
```

## A partial run is still a valid run

openFDA's event index fails intermittently. When the request budget is exhausted or a
drug fails persistently, the run is recorded as finished with what it got, and the
summary names the drugs that were skipped. That is deliberate: a partial warehouse is
useful, and a silently absent drug is indistinguishable from a drug with no signals in
every downstream aggregate — so the absence is reported rather than inferred.
