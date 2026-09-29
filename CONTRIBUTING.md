# Contributing

## Setting up

```bash
make install install-dbt     # two virtualenvs; dbt resolves separately
make doctor                  # reports what is configured and what it costs
```

An openFDA API key is optional for development but needed for a full backfill:
<https://open.fda.gov/apis/authentication/>. Put it in `.env`.

## Before opening a pull request

```bash
make lint type test          # ruff, mypy --strict, pytest
make fixture marts           # dbt models and 69 data quality tests, no network
```

CI runs the same gates plus a synthetic-warehouse dbt build and an export check.

## Conventions this repository actually holds to

**Comments explain why, never what.** A comment restating the code is noise. A
comment recording *why a simpler version was wrong* is the most valuable line in the
file, because that reasoning is not recoverable from the code. Much of this codebase
is that kind of comment, and most of them exist because the simpler version was tried
first and measured.

**Numbers in docstrings are measurements, not illustrations.** Where a docstring
gives a coverage percentage, a cosine similarity or a request count, it was measured
against the live API and is reproducible. If you change a threshold, re-measure and
update the number.

**Point-in-time correctness is not negotiable.** Any code touching a query window
goes through `prodrome.timeframe`. Never introduce `receiptdate`: it moves forward
when a report is amended, so windows built on it do not nest and cumulative counts
can fall. The dbt test `assert_cumulative_counts_are_monotone` will catch it, and
that test is load-bearing.

**Failing closed beats failing plausibly.** A NaN estimator must not satisfy a
threshold; an unreadable label must not read as "not labelled"; a truncated
aggregation must not read as zero. Each of these has a named state in the code
(`degenerate`, `Verdict.UNKNOWN`, `a_source`) precisely so the distinction survives
into the warehouse.

**A new signal-detection criterion is a registry entry**, in
`prodrome/stats/criteria.py`. It will then be evaluated on every cell and get its own
onset quarter automatically. Add the matching column to `stat_disproportionality`, the
`accepted_values` test, and the SQL cross-check in
`dbt/tests/assert_criteria_flags_agree_with_thresholds.sql`.

**Tests assert behaviour, with the reason in the docstring.** A test named
`test_negative_retry_after_is_treated_as_malformed` whose docstring explains that a
negative value would raise in `time.sleep` is worth ten assertions on internals.

## Adding a drug to the cohort

Do not hand-edit `conf/cohort.yml`. Run the resolver and review its diff:

```bash
python tools/resolve_cohort.py --names "<drug>" --show-rejected
```

Check two things in the output before committing. The chosen SPL set id must belong to
the **application holder** — a repackager's label has one version and would make the
drug appear never to have had a label change. And the UNII must be the **active
moiety**, not an excipient or a salt: an excipient UNII produces a drug with no
adverse events, and neither failure raises an error.

## Changing the label-matching thresholds

The semantic threshold is a z-score against a per-label empirical null, not a raw
cosine — see `docs/METHODS.md` §10 for why a fixed cosine cannot work. If you change
it, re-run the ground-truth check:

```bash
pytest -m network -k GroundTruthLabelTransition
```

That asserts ileus is absent from Ozempic's 2022 label and present in its 2023 one,
which is the transition the whole project is calibrated against.

## If the API starts failing

Two causes have been seen, and neither is fixed by a bigger retry budget.

**A malformed query.** openFDA answers a `count` over an *analysed* string field with
**HTTP 500 rather than 400** — a client error disguised as a server error, which the
retry machinery absorbs and hides. `openfda.ANALYSED_STRING_FIELDS` lists the fields
needing `.exact`; `validate_count_field` enforces it. Coded and date fields
(`primarysource.qualification`, `patient.patientsex`, `serious`, `receivedate`) must
*not* carry it.

**Sustained load.** openFDA documents 240 requests/minute, but that is a burst ceiling.
A backfill at 200/minute drew 500s that cost three drugs; the same queries at 60/minute
succeeded 30/30. If failures appear during a long run, lower
`http.requests_per_minute` before touching anything else.

So, in order:

1. **Hold the query shape fixed and repeat it.** A deterministic failure on one shape
   looks exactly like a random failure across many when shapes are sampled together.
   That is precisely the mistake made here, and it cost hours.
2. **Check whether a counted field is analysed.** See above.
3. **Lower the request rate.** Backpressure and giving up are not the same thing: a
   retry budget answers "is this fault transient", a rate limit answers "am I the
   problem".
4. **Look at `RequestStats.failures`** in the run summary. A healthy run ends at zero;
   a non-zero count is a defect to investigate, not weather to endure.

Responses are cached on disk, so re-running a stage costs only what the previous pass
did not get. `prodrome status` shows the request accounting.
