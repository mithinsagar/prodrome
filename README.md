# prodrome

**A signal-to-label latency engine for FDA adverse event data.**

[![ci](https://github.com/mithinsagar/prodrome/actions/workflows/ci.yml/badge.svg)](https://github.com/mithinsagar/prodrome/actions/workflows/ci.yml)
[![weekly](https://github.com/mithinsagar/prodrome/actions/workflows/weekly.yml/badge.svg)](https://github.com/mithinsagar/prodrome/actions/workflows/weekly.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Conventional pharmacovigilance tooling answers *which drug–reaction pairs are
reported disproportionately*. FDA's own FAERS dashboard does it, OpenVigil does it,
and there are hundreds of published disproportionality analyses.

prodrome answers a question none of them can: **how much warning does each
signal-detection criterion actually buy you before the label changes?**

Answering it requires two things that do not exist in the standard toolkit — a
reconstructed history of what every drug's label said at every point in time, and
statistics computed *as of* each quarter rather than on the final snapshot. The
second turns out to matter far more than expected.

---

## The problem this exists to solve

A drug-safety team running disproportionality analysis gets thousands of hits and
no way to rank them beyond the size of the ratio. The obvious fix is to learn from
history: which past disproportionality signals actually went on to become label
changes? Build that outcome variable, and ranking becomes a prediction problem.

The obvious way to build it is wrong, and wrong in a way that flatters the result.

When a reaction is added to a drug's label, reporting of that reaction rises
sharply — clinicians who now know to look for it, report it. This is **notoriety
bias**, and it means a disproportionality statistic computed on the full cumulative
database is *partly caused by the label change it would be used to predict*. The
outcome leaks into the predictor.

Measured here on semaglutide and ileus — a real case, where FDA added ileus to
Ozempic's label in September 2023 — using cumulative counts at each quarter cutoff:

| data through | reports | PRR | χ² | MHRA triple fires? |
|---|---:|---:|---:|:--|
| 2022-12-31 | 17 | 1.68 | 4.0 | no |
| 2023-06-30 | 22 | 1.76 | 6.5 | no |
| 2023-09-30 | 25 | 1.72 | 6.8 | no |
| **← FDA adds ileus to the label, 2023-10-09** | | | | |
| 2024-12-31 | 160 | **6.92** | **792.5** | yes |

A retrospective analysis reports PRR 6.92 with χ² of 793 and implicitly credits the
method with finding it. Point-in-time, the MHRA triple **never fired at all** before
the label changed. Only the EMA interval criterion did — around 2022 H1, roughly 16
months early.

**So every statistic in prodrome is computed point-in-time, and the retrospective
value is computed too, purely to report the ratio between them.** That ratio is the
measurement: it says how much of the apparent signal in a conventional analysis
arrived *after*, and *because of*, the outcome being predicted.

---

## What it does

```
openFDA drug/event ──┐
                     ├─► point-in-time 2×2 tables ─► PRR · ROR · χ² · shrinkage O/E
                     │    (one per drug × reaction    · EBGM/EB05 (gamma-Poisson MLE)
                     │     × quarter, cumulative)     · Mantel-Haenszel adjusted ROR
                     │                                · empirically calibrated p
                     │                                       │
DailyMed SPL archive ┘                                       ▼
   ├─ version history per drug                    signal onset, per criterion
   ├─ every archived version's text                         │
   ├─ LOINC safety sections only                            ▼
   └─ label-mention verdict ──────────────► label onset ─► lead time
        lexical → synonym → embedding                        │
                                                            ▼
                                        survival curves · leakage benchmark
                                        artefact diagnostics · hazard model
                                                            │
                                                            ▼
                                     DuckDB ─► dbt ─► Tableau .hyper + dashboard
```

**Five things it produces**

1. **Lead time per criterion**, as a Kaplan-Meier curve with censoring handled
   properly — how long labelling takes once a criterion fires, and how many alarms
   it raised to get there.
2. **The leakage benchmark** — how much a conventional retrospective analysis
   overstates each signal, and how many of its "detections" it could not actually
   have made prospectively.
3. **A ranked queue of open label gaps**, scored by a discrete-time hazard model
   fitted on a strictly earlier period, so the ranking is prospective by
   construction. Reported with precision@k and a calibration curve, not just AUC.
4. **Artefact diagnostics** on every signal — reporter concentration, litigation
   pattern, volume spikes, single-quarter dependence. A signal driven by mass-tort
   reporting and one accumulated steadily across many countries are not the same
   finding.
5. **A weekly brief** whose every number is mechanically verified against a fixed
   evidence pack before publication.

---

## Five things that had to be got right

Each of these was found by measuring against the live APIs, and each silently
produces wrong numbers rather than an error.

### 1. openFDA's substance harmonisation has zero coverage for some drugs

The precise, obviously correct way to identify a drug's reports is its UNII, FDA's
substance identifier. Coverage turns out to be **bimodal**:

| drug | reports by UNII | by reported substance name | UNII coverage |
|---|---:|---:|---:|
| pembrolizumab | 104,614 | 105,077 | 99.6% |
| semaglutide | 73,001 | 100,515 | 72.6% |
| **osimertinib** | **0** | **31,954** | **0.0%** |
| **esketamine** | **0** | **18,640** | **0.0%** |
| **adalimumab** | 53,861 | 716,520 | **7.5%** |

A UNII-only join silently discards osimertinib entirely — 31,954 reports, none
joinable, no error raised, a perfectly plausible empty result. A name-only join,
which is what most published FAERS analyses use, is imprecise in the other
direction. prodrome uses the **union** of UNII, reported active-substance name and
brand name, pins it per drug in version control so it cannot drift between quarters,
and **records the measured coverage next to every result**. 14 of the 55 tracked
drugs rely materially on name matching. See [`src/prodrome/selector.py`](src/prodrome/selector.py).

### 2. openFDA's own generic-name field is wrong for Ozempic

`patient.drug.openfda.generic_name` reports `ORAL SEMAGLUTIDE` for a subcutaneous
injection. Name-based joins are not merely imprecise here; the harmonised name is
factually incorrect. That is why identity is resolved from the SPL document's own
ingredient block, filtered to `classCode="ACTIB"` — otherwise you key the cohort on
disodium phosphate.

### 3. A truncated aggregation is not a zero

`count=patient.reaction.reactionmeddrapt.exact` returns, for a single term, exactly
the report count for that term — verified term by term (`NAUSEA` 537 = 537,
`PANCREATITIS` 106 = 106). That makes one request yield a whole row of
co-occurrence counts and cuts the backfill from ~607,000 requests to ~13,000.

But the aggregation is truncated to the most frequent terms — 100 without an API
key, 1,000 with one. **Ileus for semaglutide in 2024Q1 has 21 reports and is not in
the top 100**, and ileus is the case this project was built to study. So every cell
records how its count was obtained: exhaustive response (absence is a real zero),
present in the aggregation, or fetched individually. Without that distinction the
pipeline writes false zeros into its most important cells.

### 4. MedDRA is British, US labels are American

MedDRA is maintained to British spelling; US prescribing information is American.
So the FAERS term is `Diarrhoea` and the label says `diarrhea`; `Oesophagitis`
against `esophagitis`; `Ischaemic stroke` against `ischemic stroke`. These are the
same word. Missing them systematically *inflates* the apparent label gap — in the
direction that makes the project's headline look better, which is the worst kind of
bug. Handled by a curated stem list rather than a blanket `oe → e` rule, which would
mangle "toe", "does" and "shoe" in the label text being searched.

### 5. A fixed cosine threshold cannot work here

Embedding similarity on short clinical terms, measured with `bge-small-en-v1.5`:

| pair | cosine |
|---|---:|
| thrombocytopenia / low platelet count | 0.754 |
| ileus / intestinal obstruction | 0.729 |
| ileus / blockage of the bowel | 0.689 |
| **ileus / kidney stone** | **0.611** |
| ileus / hair loss | 0.568 |

True and false pairs are separated by under 0.08, and on real labels the best-match
scores cluster at **mean 0.615, sd 0.037** — so any fixed cutoff either admits
kidney stones as a match for ileus or rejects genuine paraphrase. prodrome instead
scores a sample of *other* cohort reactions against the same label to build a
**per-label empirical null**, and accepts a semantic match only when it stands out
against that. Same reasoning as the empirical calibration applied to the
disproportionality statistics: the theoretical null is wrong, so measure the real one.

---

## Statistical methods

Four disproportionality families, because the central finding is that **they
disagree about when a signal starts**:

| measure | what it adds | source |
|---|---|---|
| PRR + χ² (Yates) | the MHRA/EMA workhorse | Evans, Waller & Davis (2001) |
| ROR with 95% CI | admits a stratified form, so it is the one adjusted for confounding | Rothman, Lanes & Sacks (2004) |
| shrinkage log₂ O/E | behaves sanely at *a* = 1, where PRR is enormous and worthless | Norén et al. (2006) |
| **EBGM / EB05** | gamma-Poisson shrinkage fitted by MLE — the method behind FDA's own Empirica Signal | DuMouchel (1999) |

Plus three corrections that are rare in published FAERS work:

- **Mantel-Haenszel adjusted ROR** with Robins-Breslow-Greenland variance, stratified
  on age, sex, report year and reporter type. Crude and adjusted are reported side by
  side, because the gap is itself a diagnostic: a signal that vanishes on adjustment
  was a confounding artefact.
- **Empirical calibration** after Schuemie et al. (2014) — a nominal *p* < 0.05 does
  not mean a 5% false-positive rate in spontaneous-report data. On a synthetic check,
  an estimate at raw *p* = 0.008 calibrates to *p* = 0.149.
- **Point-in-time priors.** Both shrinkage layers learn from the data, so fitting
  them once on the final snapshot would let 2026 reporting behaviour shrink a 2018
  estimate. Both are refitted per quarter.

Full derivations and every threshold's provenance: **[`docs/METHODS.md`](docs/METHODS.md)**.

---

## Quick start

Nothing here costs money. Both data sources are free public APIs, the warehouse is a
local file, and the scheduled job runs on GitHub Actions' free tier.

```bash
git clone https://github.com/mithinsagar/prodrome
cd prodrome
make install install-dbt        # two virtualenvs: dbt resolves separately
make doctor                     # checks the environment and tells you what is missing
```

**Get a free openFDA API key** (no card, instant, emailed immediately):
<https://open.fda.gov/apis/authentication/> → *Get an API key*. Then:

```bash
cp .env.example .env            # paste the key as PRODROME_OPENFDA_API_KEY
```

It is optional but it matters: without a key the quota is 1,000 requests/day instead
of 120,000, and count aggregations are capped at 100 terms instead of 1,000. A full
backfill will not complete without one.

```bash
make dry-run                    # estimate the request cost before spending it
make smoke                      # end-to-end on 3 drugs and a short window
```

Then the real thing:

```bash
make pipeline                   # ingest → score → latency
make marts                      # dbt models + data quality tests
make export                      # Parquet, Tableau .hyper, dashboard bundle
make brief                       # the weekly brief
make dashboard                   # serve it at http://localhost:8000
```

Every response is cached on disk, so an interrupted run resumes for free and a
re-run re-derives every number without re-downloading anything.

---

## Repository layout

```
src/prodrome/
  selector.py          drug identification, and the coverage data behind it
  timeframe.py         quarters, and the point-in-time contract
  clients/             rate-limited, caching, circuit-broken HTTP
    openfda.py         the request-budget argument lives here
    dailymed.py        SPL version history — the source that makes this possible
  stats/               contingency · PRR/ROR/χ² · EBGM · Mantel-Haenszel · calibration
    criteria.py        the registry of published threshold rules
  labelmatch/          sectioning · lexical · embeddings · FAISS · the decision
  latency/             signal onset · label onset · survival · the leakage benchmark
  diagnostics/         is this signal an artefact of how it was reported?
  brief/               evidence pack · providers · numeric verification
  pipeline/            the three stages
dbt/                   staging → intermediate → marts, with 69 tests
dashboard/             the static dashboard (no backend, deploys to Pages)
tools/                 cohort resolver · synthetic warehouse generator
docs/                  METHODS.md · ARCHITECTURE.md
```

---

## Reproducibility and provenance

Every fact row carries a `run_id`. Every run records its config digest, the openFDA
data vintage, the embedding backend, and its full request accounting. The
`mart_run_manifest` model exposes all of it, and the dashboard prints it in the
header — a figure whose run and data vintage are not on screen cannot be audited.

The cohort is **resolved by a tool and committed for review**, not recomputed at
analysis time. Two silent failure modes make that necessary: picking a repackager's
SPL set id makes a drug look as though its label never changed (turning every real
label change into a false negative), and picking an excipient's UNII makes a drug
look as though it has no adverse events. Neither raises an error.

---

## Data quality

69 dbt tests, including checks that are specific to this domain rather than generic
not-null assertions:

- **Cumulative counts must be monotone.** Every cell counts reports received on or
  before its quarter end, so a decrease is impossible and means the point-in-time
  windows are broken — a `receiptdate`/`receivedate` mix-up, or two quarters read
  from different index generations. This test caught a real bug in this repository's
  own test-data generator.
- **A label gap may never be asserted from an unreadable label.** Treating an
  unparseable document as "not labelled" would manufacture exactly the finding the
  project reports.
- **Lead-time sign must match the outcome status**, or the central claim inverts
  while every summary statistic still looks reasonable.
- **Only incident pairs may enter the survival set** — pairs already labelled at the
  earliest archived version are left-truncated and must be excluded.
- **Criterion flags are recomputed in SQL and cross-checked** against the Python
  registry. The two share no code, so a threshold changed in one place is caught.
- **The GPS posterior must be ordered** EB05 ≤ EBGM ≤ EB95, which a
  numerically-solved mixture quantile can violate if the root finder misbrackets.

`make dbt-negative` runs the suite against a warehouse with deliberately planted
violations and asserts that it fails. A test that has never been observed to fail is
not known to work.

---

## What this cannot tell you

- **These are reporting associations, not risks.** FAERS has no denominator, no
  control group, and no verification of causality. A disproportionate ratio is a
  hypothesis to evaluate, never evidence that a drug caused a reaction.
- **A label change is a proxy outcome.** It reflects FDA's assessment of evidence
  from many sources, most of them not FAERS. Predicting it is useful for prioritising
  review; it is not the same as predicting whether an association is real.
- **Left truncation is excluded, not solved.** Reactions already described in a
  drug's earliest archived label have an unknowable addition date and are dropped
  from the time-to-event analysis. The count is reported alongside every result.
- **MedDRA's hierarchy is licensed** and cannot be redistributed, so there is no
  System Organ Class grouping and the clinical synonym table is hand-built and
  incomplete. Reaction term strings themselves come from public FAERS records.
- **Only a drug's most-reported reactions are evaluated**, capped per drug to stay
  inside a free quota. Extending to the long tail is a matter of request budget, not
  of method.
- **FAERS reports reach the public database weeks after FDA receives them**, so a
  quarter-end cut reconstructs "reports with `receivedate` ≤ T", not literally what a
  reviewer could have seen on date T. The lag biases lead-time estimates *downward*,
  against this project's own headline.

---

## Data sources and terms

- **openFDA** drug adverse event and drug label APIs — public domain, no
  registration required, [terms](https://open.fda.gov/terms/).
- **DailyMed** (US National Library of Medicine) SPL archive — public domain.

Neither is validated for clinical use. From openFDA's own disclaimer: *do not rely on
openFDA to make decisions regarding medical care.*

## Licence

MIT — see [LICENSE](LICENSE).
