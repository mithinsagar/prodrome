# Architecture

Why the pieces are arranged this way, and what each boundary is protecting.

## The shape of the problem

The project has one hard constraint and one hard requirement.

**Constraint: it must be free.** Both data sources are free public APIs with quotas,
the warehouse must not need a server, and the scheduled job must fit GitHub Actions'
free tier. That rules out a hosted database, a managed orchestrator and any paid API.

**Requirement: every number must be reproducible and traceable.** This is a
quantitative claim about drug safety. A figure that cannot be traced to the
configuration, data vintage and API traffic that produced it is not usable, and a
result that changes between runs for unexplained reasons is worse than no result.

Almost every structural decision below follows from one of those two.

---

## Layers

```
conf/*.yml ─────────── the analysis plan, version-controlled
                       (cohort, window, thresholds, selectors)
     │
     ▼
clients/ ──────────── rate-limited · caching · retrying · circuit-broken HTTP
     │                 openfda.py  · dailymed.py · query.py (safe search builder)
     ▼
ingest/ ───────────── cohort resolution · contingency harvest · label timelines
     │                 selector.py · contingency.py · labels.py · cohort.py
     ▼
pipeline/ ─────────── three stages, separated by what they cost
     │                 ingest (network) → score (compute) → latency (compute)
     ▼
warehouse/ ────────── DuckDB: raw_* facts, stat_* statistics, model_* outputs
     │                 every row carries run_id and as_of_quarter
     ▼
dbt/ ──────────────── staging → intermediate → marts, plus 69 data quality tests
     │
     ▼
publish/ ──────────── Parquet · Tableau .hyper · dashboard JSON bundle
     │
     ├──► dashboard/  static page, no backend, GitHub Pages
     └──► brief/      evidence pack → optional LLM → numeric verification
```

The statistical core (`stats/`, `labelmatch/`, `latency/`, `diagnostics/`) is
deliberately **not** a layer in that stack. Those modules take plain data structures
and return plain data structures; they know nothing about HTTP, DuckDB or dbt. That is
what makes them testable by simulation recovery and against constructed cases with
known answers, which is how the numerics are actually validated.

---

## Decisions, and what each one is protecting against

### Three pipeline stages, not one

`ingest` spends API quota and takes hours on a cold cache. `score` and `latency` are
pure computation and take seconds. Separating them means a change to an estimator or a
threshold never requires re-downloading anything — which matters enormously, because
the statistics are where the iteration happens.

It also means the expensive stage can be interrupted and resumed. Every upstream
response is cached on disk by request, so a run that dies halfway — quota exhausted,
CI timeout, laptop closed — costs nothing to resume.

### DuckDB, not PostgreSQL

The working set is a few million rows of aggregates. DuckDB is a single file with no
server, so `git clone && make all` behaves identically on a laptop and in CI, and the
same file is read directly by dbt, pandas and the Tableau extract writer. A hosted
Postgres would add a credential, a cost and a failure mode for no analytical gain.

The cost is real and accepted: DuckDB is single-writer, so a read command during an
ingest fails. `WarehouseBusyError` exists to explain that rather than surface a lock
trace.

### Statistics in Python, reshaping in dbt

The split is "SQL where SQL is honest, Python where the statistics live". The
gamma-Poisson MLE, the gamma posterior quantiles behind EB05, and the empirical-null
fit have no SQL expression; forcing them into SQL would mean approximating them.
Conversely, the point-in-time label range join, the wide-to-long criterion unpivot and
the mart aggregations are set operations that SQL states more clearly than Python, and
dbt gives them lineage, documentation and tests for free.

dbt reads the Python-written tables as **sources**, not seeds, which keeps the
direction of dependency unambiguous.

### dbt in its own virtualenv

dbt resolves in a different dependency universe from the application. Installing it
alongside the app's pins produced conflicts on every attempt. Two virtualenvs is the
cheapest correct answer, and CI gives dbt its own job for the same reason.

### The cohort is a committed input, not a derived artefact

`tools/resolve_cohort.py` resolves drug names to UNIIs and SPL set ids, and writes
`conf/cohort.yml` for human review. It is not part of the pipeline.

Two failure modes force this. Picking a repackager's SPL set id makes a drug appear
never to have had a label change, turning every real label change into a false
negative in the outcome variable. Picking an excipient's UNII makes a drug appear to
have no adverse events. **Neither raises an error, and no downstream check would catch
either** — the pipeline would run to completion and report confident nonsense. So the
resolver ranks candidates, explains its choice, and a human confirms it once.

### Criteria as a registry, not as code paths

`stats/criteria.py` holds each published threshold rule as a first-class object.
Adding one is a one-line entry, and every criterion is then evaluated on every cell
and gets its own onset quarter automatically. If criteria were inline conditions
scattered through the scoring code, they would drift apart and the comparison between
them — which is the project's entire point — would quietly stop being like-for-like.

A dbt test recomputes two of the rules in SQL and asserts they match the Python
registry. The two implementations share no code, so a threshold changed in one place is
caught.

### The selector is an object, pinned per drug

`selector.py` exists because identifying a drug's reports turned out to be a
measurement problem rather than a lookup. Making it a pinned, version-controlled object
per drug guarantees the same definition is used for every quarter of that drug's
series — a selector that changed mid-series would put a discontinuity into the data
that looked exactly like a real signal.

### Point-in-time is enforced in one place

`timeframe.py` owns what "as of" means: cumulative, on `receivedate`, to a quarter
boundary. Every layer goes through it. Had each layer built its own window, the
`receivedate`/`receiptdate` distinction would eventually be got wrong somewhere, and
the resulting non-nested windows would corrupt every lead-time measurement invisibly.
The dbt monotonicity test is the backstop.

### Provenance columns on every fact

Every row carries `run_id`; every fact carries `as_of_quarter`; every contingency cell
carries `a_source`; every drug carries `unii_coverage`. Each of those answers a
question a reviewer will ask about a specific number, and none can be reconstructed
after the fact.

`mart_run_manifest` collects the run's config digest, data vintage, embedding backend
and request accounting into one row, and the dashboard prints it in the header.

### The brief's numbers come before its prose

`brief/evidence.py` builds a closed set of computed facts. The model — if configured at
all — is asked only to phrase them, and `brief/verify.py` extracts every number from
the output and matches it against that set, rejecting the whole brief if anything is
unaccounted for.

The failure mode being prevented is specific: not obvious nonsense, which is visible on
reading, but a plausible number in the right units and the wrong value. Making the
numbers upstream of the prose and mechanically checkable removes the opportunity rather
than asking the model to be careful. The deterministic template is the reference
output; the model is an optional improvement to its wording.

### The dashboard is static

No backend, so it deploys free to GitHub Pages and cannot break independently of the
data. Everything it needs is precomputed into one JSON bundle — aggregation left to the
browser would mean shipping the raw marts to it. Charts are hand-built SVG because the
forms needed are simple and a charting library's defaults (thick marks, heavy grids, a
colour cycle) are precisely what the visual design rejects.

The bundle is published by the same job that produced its data, so the page and its
data can never be a version apart.

---

## Handling upstream failure, and two lessons about diagnosing it

During development `drug/event` appeared to fail on roughly 40% of requests while
`drug/label`, `device/event` and `food/enforcement` stayed healthy. That pattern reads
unambiguously as an unhealthy index, and the response was to widen the retry budget to
nine attempts and build a circuit breaker. There were two separate causes and neither
was an unhealthy index.

**Cause one: a malformed query.** openFDA answers a `count` aggregation over an
analysed string field with HTTP 500, not 400 — so `count=occurcountry` failed every
single time, deterministically, while `count=occurcountry.exact` works. Mixed into a
sample with healthy query shapes, a deterministic failure on one shape is
statistically identical to a random failure across all of them. The retry machinery
then did its job perfectly and hid the bug.

**Cause two: sustained load.** openFDA documents 240 requests/minute, but that is a
burst ceiling rather than a throughput figure. A cold backfill at 200/minute drew 500s
that cost three checkpoint inhibitors from the run; the identical queries issued at
60/minute succeeded 30 times out of 30.

And the circuit breaker, built for the wrong reason, was also built the wrong way. Its
first version cut the retry budget when it opened, including for essential requests —
which turned a recoverable slowdown into permanent data loss, and is precisely how
those three drugs were dropped. **Backpressure and giving up are not the same thing.**

The current design, with each mechanism's job stated:

- **On-disk response caching** makes re-running the remedy: each pass costs only what
  the previous one lost, and a full re-derivation needs no network at all.
- **A sustained rate limit of 90/minute** answers "am I the problem". This is the
  setting that actually prevents load-induced failure, and the one to reach for first.
- **Jittered exponential backoff, five attempts**, answers "is this fault transient".
  Full jitter, so a burst of failures does not produce a synchronised retry storm.
- **A pre-flight field guard** (`openfda.validate_count_field`) answers "is this
  request even valid", before quota is spent on a guaranteed 500.
- **The circuit breaker** applies a cooldown scaling with the observed failure rate,
  and curtails retries **only** for requests marked `on_error="skip"`. Essential
  requests keep the full budget, because the alternative to waiting is a missing cell,
  and a missing cell is indistinguishable downstream from a real zero.
- **`RequestStats.failures`** is surfaced in the run summary, so a non-zero count
  against a healthy API reads as a defect to investigate rather than weather to endure.

A drug that fails persistently is skipped and **named**, because a silently absent drug
is indistinguishable from a drug with no signals in every downstream aggregate.

## Testing strategy

| layer | how it is tested | why that way |
|---|---|---|
| estimators | simulation recovery from a known generating process; constructed cases with known answers | there is no closed-form oracle for EBGM; recovering the parameters that generated the data is the real check |
| label matching | the semaglutide/ileus transition between real archived label versions | a ground-truth regulatory event with a known date |
| HTTP client | respx-mocked, including malformed `Retry-After`, 200-with-HTML, a tripped circuit breaker, and the analysed-field guard | the real API cannot be made to fail on demand, and CI must not depend on it being up |
| dbt models | a synthetic warehouse with known properties | lets tests assert that *bad* data fails, which real data cannot demonstrate |
| data quality | `make dbt-negative` runs the suite against planted violations and expects failure | a test never observed to fail is not known to work |
| live APIs | `tests/integration`, marked `network`, excluded from CI | run on the weekly schedule, where a failure is information rather than noise |

The synthetic warehouse is the load-bearing piece. It removes the network from CI, keeps
the repository small, and — most importantly — makes the data-quality tests meaningful,
because a test that passes on real data only proves the real data happened to be clean
that day.

---

## What would change at ten times the scale

Stated because the current choices are right for this size and would not be for a
larger one:

- **The contingency harvest is the bottleneck**, and it is bounded by API quota rather
  than compute. Ten times the cohort means openFDA's bulk quarterly JSON downloads
  instead of the count API, which changes the ingest layer and nothing else — the
  warehouse schema and everything above it are unaffected.
- **DuckDB would still be fine** for the aggregates; the raw event records would not
  fit and should not be landed.
- **The label backfill parallelises trivially** (independent per drug) and is currently
  serial only out of politeness to a free NLM service.
- **FAISS would start earning its place.** At the current corpus size the numpy matrix
  product is faster than building the index.
