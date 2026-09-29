# Methods

Every threshold, formula and design decision in prodrome, with its provenance. Where
a choice was made by measurement rather than by convention, the measurement is here.

## Contents

1. [The point-in-time contract](#1-the-point-in-time-contract)
2. [Identifying a drug's reports](#2-identifying-a-drugs-reports)
3. [The 2×2 table and its provenance](#3-the-22-table-and-its-provenance)
4. [Disproportionality estimators](#4-disproportionality-estimators)
5. [Gamma-Poisson shrinkage](#5-gamma-poisson-shrinkage-ebgm--eb05)
6. [Confounding: Mantel-Haenszel](#6-confounding-mantel-haenszel)
7. [Empirical calibration](#7-empirical-calibration)
8. [Signal-detection criteria](#8-signal-detection-criteria)
9. [Reconstructing the label timeline](#9-reconstructing-the-label-timeline)
10. [Deciding whether a label describes a reaction](#10-deciding-whether-a-label-describes-a-reaction)
11. [Lead time, censoring and truncation](#11-lead-time-censoring-and-truncation)
12. [The leakage benchmark](#12-the-leakage-benchmark)
13. [Artefact diagnostics](#13-artefact-diagnostics)
14. [The prioritisation model](#14-the-prioritisation-model)
15. [Known limitations](#15-known-limitations)
16. [References](#16-references)

---

## 1. The point-in-time contract

Every statistic is computed **as of** a quarter boundary, from reports FDA had
received on or before that date. The window is cumulative — `receivedate:[20040101 TO
<quarter end>]` — because that is how disproportionality is assessed in practice: a
signal is evaluated against the whole accumulated experience, not one quarter of it.

### Why `receivedate` and never `receiptdate`

openFDA exposes both. `receiptdate` is the date of the *most recent version* of a
report and moves forward whenever a follow-up is filed. Filtering on it would let a
2019 report amended in 2024 appear in a 2024 window and vanish from the 2019 one, so
the windows would not nest and cumulative counts could fall. `receivedate` is the
date FDA first received the report and never changes. Only it supports a stable
point-in-time cut, so prodrome uses it everywhere and does not offer the alternative.

The dbt test `assert_cumulative_counts_are_monotone` exists to catch any violation of
this: a cumulative count that decreases means the windows are not nested.

### The residual approximation, stated plainly

FAERS reports reach the public database some weeks after FDA receives them. A
quarter-end cut therefore reconstructs "reports with `receivedate` ≤ T", not "what a
reviewer could literally have seen on date T". The lag is short relative to a quarter
and it biases lead-time estimates **downward** — against this project's own headline —
so it is a conservative approximation rather than a flattering one.

---

## 2. Identifying a drug's reports

FAERS reports are free text from the field. openFDA attempts to harmonise each
reported drug against FDA's product database and, on success, attaches an `openfda`
block carrying a UNII. Joining on that UNII is the precise, obviously correct
approach, and it was the first thing implemented here.

Measured against the live API, harmonisation coverage is **bimodal**:

| drug | by UNII | by reported substance name | by brand name | UNII coverage |
|---|---:|---:|---:|---:|
| pembrolizumab | 104,614 | 105,077 | 69,391 | 99.6% |
| semaglutide | 73,001 | 100,515 | 66,038 | 72.6% |
| osimertinib | 0 | 31,954 | 23,554 | 0.0% |
| esketamine | 0 | 18,640 | 17,694 | 0.0% |
| ubrogepant | 0 | 6,026 | 5,789 | 0.0% |

A UNII-only join discards osimertinib entirely — 31,954 reports, none joinable, no
error, a plausible-looking empty result. A name-only join, the common approach in
published work, is imprecise in the other direction and cannot separate a substance
from a brand containing it.

**Resolution.** A drug is identified by the disjunction of three exact-match clauses:

```
patient.drug.openfda.unii.exact:"<UNII>"
  OR patient.drug.activesubstance.activesubstancename.exact:"<SUBSTANCE>"
  OR patient.drug.medicinalproduct.exact:"<BRAND>"
```

Each component is an exact match, so the union adds recall without adding fuzziness.
`activesubstance.activesubstancename` is the workhorse: it is populated even when
harmonisation failed completely.

Two properties make this safe rather than merely broader:

- The selector is **pinned per drug in version control**. A selector that changed
  between quarters would put a discontinuity into every series that looked exactly
  like a real signal.
- Coverage is **measured and recorded**. `unii_coverage` travels into the warehouse
  and onto the dashboard, so a reader can see which drugs rest on name matching. 14
  of 55 tracked drugs do.

### Salt versus active moiety

SPL names the ingredient as formulated. Esketamine's label gives
`ESKETAMINE HYDROCHLORIDE` (UNII `L8P1H35P2Z`) whose nested `activeMoiety` is
`ESKETAMINE` (`50LFG02TXD`), and reporters code the base. The resolver includes both
the formulated name and its salt-stripped form.

### Excipients

An SPL ingredient block lists excipients alongside the active. Ozempic's carries
UNIIs for disodium phosphate, propylene glycol, phenol and water. Only
`classCode` in `{ACTIB, ACTIM, ACTIR}` is an active ingredient; keying a cohort on
an excipient would produce a drug with no adverse events and no error.

---

## 3. The 2×2 table and its provenance

```
                     reaction R      not R
    drug D               a             b        a + b = reports naming D
    not D                c             d        c + d
                     -------       -------
                      a + c         b + d       N = a + b + c + d
```

`a` is the only cell openFDA reports directly; the others are derived from three
marginals. `Contingency.from_marginals` validates consistency and **raises** rather
than coercing: openFDA serves the four counts from four separate requests, and if it
reindexes between them the margins disagree. A coerced table produces a plausible
ratio from impossible counts, which is worse than a gap.

### The request-budget argument

One request per count per cell would cost ≈ 607,000 requests for 55 drugs × 60
reactions × 46 quarters. Three observations, each verified against the live API,
reduce that to ≈ 13,000:

1. For a single term, `count=patient.reaction.reactionmeddrapt.exact` returns exactly
   the **report** count for that term, identical to a targeted query's
   `meta.results.total`. Verified term by term: `NAUSEA` 537 = 537, `VOMITING`
   328 = 328, `PANCREATITIS` 106 = 106, `DIARRHOEA` 313 = 313. So one request yields
   a whole row of co-occurrence counts.
2. The reaction marginal `a + c` does not depend on the drug — one global count
   request per quarter serves the whole cohort.
3. `N` is one request per quarter.

### Truncation is not zero

A count aggregation returns only the most frequent terms: **100 without an API key,
1,000 with one** (sending `limit` with `count` and no key is a 403, not a clamp).
Ileus for semaglutide in 2024Q1 has 21 reports and does **not** appear in the top
100 — and ileus is the case this project was built to study.

So every cell records how its `a` was obtained:

| `a_source` | meaning | exact? | cost |
|---|---|:--:|---|
| `count_present` | the term appeared in the aggregation | yes | free |
| `count_exhaustive` | the aggregation returned fewer terms than the cap, so it was complete and the absence is a true zero | yes | free |
| `targeted` | the aggregation was truncated and the term absent, so it was fetched individually | yes | one request |

Without that distinction a pipeline writes false zeros into precisely its most
important cells, and nothing downstream would detect it.

### Unit of analysis

The `count` aggregation counts reaction *occurrences*; the marginals from
`meta.results.total` count *reports*. For a single term these coincide (verified
above) because a report listing a PT twice is vanishingly rare. Summing `count`
across terms does **not** give a report total, and prodrome never does that.

---

## 4. Disproportionality estimators

All four are computed for every table, because the central finding is that **they
disagree about when a signal starts**.

**Proportional reporting ratio**

$$\mathrm{PRR} = \frac{a/(a+b)}{c/(c+d)}, \qquad
\operatorname{Var}(\ln \mathrm{PRR}) = \frac{1}{a} - \frac{1}{a+b} + \frac{1}{c} - \frac{1}{c+d}$$

**Reporting odds ratio**

$$\mathrm{ROR} = \frac{ad}{bc}, \qquad
\operatorname{Var}(\ln \mathrm{ROR}) = \frac1a + \frac1b + \frac1c + \frac1d$$

**Chi-squared with Yates' continuity correction**

$$\chi^2 = \frac{N\bigl(|ad - bc| - N/2\bigr)^2}{(a+b)(c+d)(a+c)(b+d)}$$

Returns 0 on a degenerate table rather than NaN, so the MHRA gate fails closed.

**Shrinkage log observed-to-expected**, with $E = (a+b)(a+c)/N$ and $\alpha = 0.5$:

$$\log_2 \frac{a + \alpha}{E + \alpha}$$

with a credibility interval from the $\mathrm{Gamma}(a + \alpha,\ \text{rate} = E + \alpha)$
posterior. Unlike the Wald intervals this stays finite and honest at $a \in \{0, 1\}$,
so it is the measure to quote for rare pairs.

Intervals are Wald on the log scale — not exact, but what the published criteria were
calibrated against. Minimum-count thresholds are part of each criterion rather than a
global filter, which is how the small-count regime is handled.

---

## 5. Gamma-Poisson shrinkage (EBGM / EB05)

DuMouchel's empirical Bayes estimator, the method behind Empirica Signal — the tool
FDA's own safety reviewers use. The amount of shrinkage applied to a cell is
*learned from the whole database* rather than fixed a priori.

$$n \mid E, \lambda \sim \mathrm{Poisson}(\lambda E), \qquad
\lambda \sim P\,\mathrm{Gamma}(\alpha_1, \beta_1) + (1-P)\,\mathrm{Gamma}(\alpha_2, \beta_2)$$

Integrating gives a two-component negative-binomial marginal with a closed form, so
the five prior parameters are fitted by maximum likelihood over every cell at once:

$$n \mid E \sim P\,\mathrm{NB}\!\left(\alpha_1, \tfrac{\beta_1}{\beta_1+E}\right)
+ (1-P)\,\mathrm{NB}\!\left(\alpha_2, \tfrac{\beta_2}{\beta_2+E}\right)$$

Two components are what make this work on spontaneous-report data: one absorbs the
enormous mass of noise cells, the other describes the genuinely elevated ones. A
single-gamma prior cannot do both and over-shrinks real signals toward 1.

Reported quantities: $\mathrm{EBGM} = \exp(\mathbb{E}[\log \lambda])$ under the
posterior, and $\mathrm{EB05}$, its 5th percentile. The posterior is a gamma mixture
with no closed-form quantile, so EB05 comes from Brent root-finding on the mixture
CDF, bracketed by the component quantiles.

**Validated by simulation recovery.** Drawing 8,000 cells from a known prior and
refitting recovers the component means and mixing weight to within 20%:

| parameter | true | fitted |
|---|---:|---:|
| component 1 prior mean | 0.500 | 0.496 |
| component 2 prior mean | 2.000 | 1.988 |
| mixing weight *P* | 0.150 | 0.156 |

Component *means* are the identifiable part of a gamma mixture — shape and rate trade
off along a ridge — and the means are what the posterior depends on.

**The behaviour that justifies the complexity:**

| n | E | RRR = n/E | EBGM | EB05 | verdict |
|---:|---:|---:|---:|---:|---|
| 1,218 | 82.7 | 14.73 | 14.71 | 14.03 | plenty of data: barely shrunk |
| 1 | 0.007 | **143.85** | **6.08** | 0.59 | one report: correctly not a signal |

A single report at RRR = 144 is a screaming signal under PRR. GPS discounts it to
EB05 = 0.59, well under the conventional `EB05 ≥ 2` bar.

**Point-in-time priors.** The prior is refitted on each quarter's cells. Fitting once
on the final snapshot and applying it backwards would let later reporting behaviour
shrink an earlier estimate — leaking the future into the past, which is exactly the
contamination this project measures. Fitted parameters are written per quarter,
because a shrunk estimate is uninterpretable without its prior.

---

## 6. Confounding: Mantel-Haenszel

A crude ROR treats the database as one exchangeable population. Three structural
confounders dominate FAERS: **age and sex** (reporting rates for whole reaction
classes differ, and drug populations are not balanced), **report year** (total volume
has grown by more than an order of magnitude and the reaction mix shifted with it),
and **reporter type and country** (consumer- and litigation-sourced reports have a
completely different reaction profile).

$$\mathrm{ROR}_{MH} = \frac{\sum_k a_k d_k / N_k}{\sum_k b_k c_k / N_k}$$

with the Robins-Breslow-Greenland variance, which is consistent both for few large
strata and for many sparse ones — the sparse regime being normal here, since
year × sex × age band produces many thin strata.

Strata with an empty margin carry no odds-ratio information and are dropped, counted
so the exclusion is visible.

Crude and adjusted are reported side by side plus their ratio, because **the gap is
itself a diagnostic**: a signal that vanishes on adjustment was a confounding
artefact, which is exactly the false alarm a review team wants filtered.

*Validated against a constructed dataset with a known common odds ratio of 2.0 and
deliberately unbalanced exposure prevalence: crude 1.56, MH 2.00.*

---

## 7. Empirical calibration

Every estimator assumes its own null: that under no association, $\log \mathrm{ROR}$
is centred on 0 with the Wald variance. In spontaneous-report data this is false.
Confounding by indication, channel effects, event co-reporting and duplicates all push
the null off centre and widen it, so a nominal *p* < 0.05 does not mean a 5%
false-positive rate.

Following Schuemie et al. (2014), the null is estimated **empirically**. For a pair
with no true association:

$$\log \mathrm{RR}_{\text{obs}} \sim \mathcal{N}(\mu,\ \sigma^2 + \mathrm{se}_{\text{obs}}^2)$$

with $\mu$ (systematic bias) and $\sigma$ (between-pair heterogeneity) fitted by MLE
across the controls, each control's own standard error entering as known measurement
error. A calibrated two-sided *p*-value follows directly.

**Effect, on a synthetic check** with a null biased to $\mu = 0.35$: an estimate of
$\log \mathrm{ROR} = 0.80,\ \mathrm{se} = 0.30$ has raw *p* = 0.0077 and calibrated
*p* = 0.149. A finding that looks solidly significant does not survive.

### Where the controls come from

Curated negative-control sets exist for specific research questions, not for an
arbitrary 55-drug cohort, so prodrome supports two sources and **records which was
used per quarter**:

- **Curated** — pairs listed in `conf/negative_controls.yml`. Preferred; the method is
  then exactly Schuemie et al.
- **Empirical null on the central mass** — the null fitted to the trimmed centre
  (15% from each tail) of the log-ROR distribution across all pairs. Justified by the
  same reasoning that underpins Efron's empirical null and the noise component of
  DuMouchel's mixture: of tens of thousands of pairs, the overwhelming majority are
  not causal, so the bulk of the distribution *is* the null. Weaker than a curated
  set, and labelled as such.

Both are refitted per quarter, so drift in $\mu$ over time is itself observable.

---

## 8. Signal-detection criteria

Criteria are first-class objects (`src/prodrome/stats/criteria.py`). Each is evaluated
independently on every cell and gets its own onset quarter, so the lead time each buys
is measured rather than assumed.

| key | rule | min *a* | source |
|---|---|:--:|---|
| `mhra_prr` | PRR ≥ 2 **and** χ² ≥ 4 | 3 | Evans, Waller & Davis (2001) |
| `ema_ror025` | lower 95% bound of ROR > 1 | 3 | EMA / EudraVigilance guidance |
| `who_oe025` | lower bound of shrinkage log₂ O/E > 0 | 1 | Norén et al. (2006), UMC practice |
| `dubious_prr_only` | PRR ≥ 2, no count or significance gate | 1 | **negative control on method** |

The last is included deliberately. A large share of published FAERS analyses apply
exactly that rule, and prodrome reports its false-alarm rate next to the others so the
comparison is visible rather than asserted.

The count gate is enforced in `Criterion.holds` rather than inside each predicate, so
a rule can never accidentally fire on a single report. NaN estimators fail closed.

### Persistence

A criterion can fire in one quarter on a handful of reports and stop in the next. Both
the first firing quarter and the first quarter beginning a run of ≥ 2 consecutive
firing quarters are recorded, so the lead time each definition buys can be compared.
Quarterly persistence has been proposed in the literature as a prioritisation
dimension in its own right; here it is measured against a supervised outcome.

---

## 9. Reconstructing the label timeline

**This is the part that does not exist in standard tooling.** openFDA's label endpoint
indexes only the *current* version of each label. That answers "is this reaction
labelled" and cannot answer "since when" — so it cannot distinguish a reaction FDA has
warned about for a decade from one added last quarter in response to the very reports
being analysed.

DailyMed publishes the full version history of every SPL: a version number and
publication date per revision, and the complete document for each.

```
GET /dailymed/services/v2/spls/{setid}/history.json      → versions + dates
GET /dailymed/getFile.cfm?setid={setid}&type=zip&version=N → that version's archive
```

Verified while designing this: Ozempic's application-holder label has 19 archived
versions from 2017-12-06 to 2026-06-10, and effective dates confirm distinct
documents (v1 → 2017-12-05, v6 → 2020-01-16, v13 → 2022-10-07, v20 → 2026-06-01).

### Three operational hazards

**Repackager labels.** Searching by name returns many set ids, most belonging to
repackagers who republish once and never revise. Ozempic has several with a single
version. Using one makes the drug appear never to have had a label change, so every
real change becomes a false negative in the outcome variable. Set ids are therefore
pinned in config after review, ranked by revision count — only an application holder
accumulates revisions.

**Success-shaped failures.** Requesting a nonexistent version returns **HTTP 200 with
an HTML error page**. Only a content check — the `PK` ZIP magic — distinguishes that
from a real archive.

**Non-contiguous version numbers.** Ozempic's history runs 1–9 then 11–20, skipping
10. Iterating `range(1, n+1)` requests a version that does not exist and, per the
above, appears to succeed.

### Which sections count

A US prescribing information document is mostly *not* safety labelling. Ozempic's 2023
label devotes **18,761 characters to CLINICAL STUDIES**, which names dozens of adverse
events as trial outcomes. Matching against the whole document would mark almost every
reaction as already labelled.

| tier | LOINC | section |
|---|---|---|
| **core** | 34066-1 | Boxed warning |
| | 34070-3 | Contraindications |
| | 43685-7 | Warnings and precautions |
| | 34084-4 | Adverse reactions |
| secondary | 34073-7 | Drug interactions |
| | 43684-0, 42228-7, 77290-5, 77291-3, 34081-0, 34082-8 | Use in specific populations and its subsections |
| | 34088-5 | Overdosage |
| | 42231-1, 34076-0 | Medication guide, patient information |

Excluded, each for a reason: **clinical studies** (names adverse events as trial
outcomes), **clinical pharmacology / mechanism / PK / PD** (mechanistic),
**nonclinical toxicology and carcinogenesis** (animal findings — "thyroid C-cell
tumours in mice" is not a labelled human adverse reaction), **indications and usage**
(therapeutic goals collide with adverse events: "weight decreased" is both for a GLP-1
agonist), and the procedural sections.

An unrecognised LOINC code is treated as excluded — the conservative direction, so a
new section type cannot silently start counting as evidence.

Text is attributed to the **deepest** coded section that owns it, because SPL nests
("5.7 Severe Gastrointestinal Adverse Reactions" inside "5 WARNINGS AND
PRECAUTIONS") and taking a parent's subtree would double-count its children.

A document with under 200 characters of core safety text is recorded **unusable**, not
"nothing labelled" — the distinction the dbt test
`assert_label_gap_is_never_asserted_from_an_unusable_label` enforces.

---

## 10. Deciding whether a label describes a reaction

Three layers, strongest evidence first.

### Orthography: MedDRA is British, US labels are American

MedDRA is maintained to British spelling conventions. So the FAERS term is
`Diarrhoea` and the label says `diarrhea`; `Oesophagitis`/`esophagitis`;
`Ischaemic stroke`/`ischemic stroke`; `Anaemia`/`anemia`;
`Leucopenia`/`leukopenia`. These are the same word, and missing them systematically
**inflates** the apparent label gap — in the direction that makes the project's own
headline look better.

Handled by a **curated stem list**, not a blanket `oe → e` rule: the general rule
mangles "toe", "does" and "shoe" in the label text being searched. ~70 medical stems
where the substitution is unambiguous.

### Register: Latinate term versus colloquial label

`Pyrexia`/"fever", `Pruritus`/"itching", `Dyspnoea`/"shortness of breath",
`Asthenia`/"weakness", `Thrombocytopenia`/"low platelet count". A curated synonym
table, inverted so matching works in both directions — `Ileus` must find "intestinal
obstruction" *and* `Intestinal obstruction` must find "ileus", since both occur in
FAERS.

MedDRA's own synonym files are licensed and cannot be redistributed, so this table is
hand-built and therefore incomplete. Stated as a limitation, not hidden.

### The semantic residue, and why a fixed threshold fails

For paraphrase — "intestinal obstruction" against "blockage of the bowel" — embeddings
are the right tool. But measured with `bge-small-en-v1.5` on short clinical terms:

| pair | cosine |
|---|---:|
| thrombocytopenia / low platelet count | 0.754 |
| ileus / intestinal obstruction | 0.729 |
| ileus / blockage of the bowel | 0.689 |
| pyrexia / fever | 0.709 |
| **ileus / kidney stone** | **0.611** |
| ileus / hair loss | 0.568 |

Under 0.08 separates true from false. On the real Ozempic label, best-match scores
across 124 passages cluster at **mean 0.615, sd 0.037**. Any fixed cutoff either
admits kidney stones as a match for ileus or rejects genuine paraphrase — which is why
published cosine thresholds do not transfer between corpora.

**The empirical-null decision.** For each label version, a sample of reaction terms
drawn from *other* cohort drugs is scored against the same passages, giving a
per-label null distribution of best-match scores. A candidate is accepted only when it
stands out against that null: `z ≥ 2.5` **and** raw cosine `≥ 0.55`. The null pool
excludes the drug's own tracked reactions, or it would contain terms the label
genuinely mentions and be biased upward.

The bar is deliberately strict in one direction: a false "already labelled" destroys a
real label gap — the finding the project reports — whereas a false "not labelled" is
caught downstream when no label change materialises to predict.

### Validation on the ground-truth case

FDA added ileus to Ozempic's label in September 2023. Parsing v13 (2022-10-07) and
v14 (2023-10-09):

| reaction | v13 | v14 | how decided |
|---|---|---|---|
| **Ileus** | `not_labelled` | **`labelled_core`** | exact |
| Intestinal obstruction | `not_labelled` | **`labelled_core`** | synonym → "ileus" |
| Pancreatitis | `labelled_core` | `labelled_core` | exact (already-labelled control) |
| Diarrhoea | `labelled_core` | `labelled_core` | spelling → "diarrhea" |
| Thyroid neoplasm | `labelled_core` | `labelled_core` | synonym → "thyroid tumor" |
| Alopecia | `not_labelled` | `not_labelled` | z = +0.08 |
| Nephrolithiasis | `not_labelled` | `not_labelled` | z = +2.05, below the 2.5 bar |

### Chunking and retrieval

Safety-section text is split into overlapping windows of 3 sentences, stride 2. Small
enough to keep one mention from being diluted across a 4,000-character section; large
enough to keep the context a human reviewer needs; overlapping so a mention spanning a
boundary is never split from its context. Sentence splitting handles the
abbreviations label prose is dense with — "0.5 mg", "5.1", "U.S.", "e.g." — and bullet
lists, which adverse-reaction sections often use with no terminal punctuation.

Retrieval is exact cosine: a FAISS flat inner-product index above 2,000 passages, a
numpy matrix product below. `IndexFlatIP` rather than an approximate index because a
label-match verdict must not depend on a FAISS build or random seed — two runs must not
disagree about whether a reaction is labelled.

### Embedding backends

- `hashed` — deterministic hashing vectoriser over word and character n-grams, no
  model download, byte-identical everywhere. Used in CI. Character n-grams give real
  morphological similarity (`thrombocytopenia`/`thrombocytopenic` = 0.863) but **no
  semantics** (`ileus`/`intestinal obstruction` = 0.061), which is why the ONNX
  backend is not decorative.
- `onnx` — `BAAI/bge-small-en-v1.5` via fastembed, ONNX Runtime, no PyTorch. ~130 MB
  once, then offline. Used for published figures. The bge query prefix is applied to
  the reaction term only, as the model was trained.

Backends are recorded per run, and the threshold is calibrated per backend — one
tuned on the other is meaningless.

---

## 11. Lead time, censoring and truncation

**Signal onset** is the earliest quarter a criterion held, per criterion. **Label
onset** is the authoritative date of the earliest version whose verdict is labelled,
where authoritative date is the document's own `effectiveTime` when present, else the
DailyMed publication date.

**Lead time** = label-onset quarter − signal-onset quarter. Positive means the signal
preceded the label.

### Left truncation: the trap that invalidates the naive analysis

DailyMed's archive for a drug begins at some version 1 — for Ozempic, 2017-12-06. If a
reaction is *already* described there, the date it was added is unknowable from this
data: it happened before the observation window opened. Such a pair is **prevalent**,
not incident. Including it assigns an arbitrary label date, and because these are
overwhelmingly the well-established reactions of older drugs, including them biases
lead time systematically toward zero.

Every pair is classified explicitly, so each exclusion is visible in the output rather
than implied by a `WHERE` clause:

| status | meaning | in survival set? |
|---|---|:--:|
| `labelled_after_signal` | signal fired, label followed | ✓ event |
| `censored` | signal fired, no label change yet | ✓ censored |
| `labelled_before_signal` | the label changed first — how often disproportionality *lags* regulatory action | ✗ |
| `prevalent_at_baseline` | already labelled in the earliest archived version — left-truncated | ✗ |
| `no_signal` | the criterion never fired, so there is no origin | ✗ |
| `unobservable` | no assessable label version | ✗ |

Enforced by the dbt test `assert_survival_set_is_well_formed`.

### Right censoring

A pair whose signal fired but whose label has not changed by the end of the data is
**censored**, not negative — it may yet be labelled. Kaplan-Meier handles it; recoding
censored pairs as "never labelled" would understate the labelling rate and inflate
apparent lead times for those that were labelled.

---

## 12. The leakage benchmark

For every pair that was eventually labelled, three values of each statistic:

- **at signal onset** — using only data available when the criterion fired;
- **at the label change** — using only data available when the label changed;
- **retrospective** — on the full cumulative database, which is what a conventional
  analysis reports.

$$\text{inflation ratio} = \frac{\text{retrospective}}{\text{value at the label change}}$$

Plus a **notoriety measurement**: new reports in the four quarters after the label
change divided by the four before. New rather than cumulative, since cumulative counts
only rise and cannot show a surge.

Two summary quantities:

- **`leakage_confirmed`** — reporting more than doubled after the label change *and*
  the retrospective statistic is at least twice the value available at it. Part of the
  evidence a retrospective analysis cites was created by the outcome it predicts.
- **`retrospective_only` signals** — pairs where no criterion had fired by the label
  change, yet the full-data statistic clears the bar. These are detections a
  conventional analysis claims but could not have made.

On the semaglutide/ileus case: PRR at the label change 1.72, retrospective 6.92,
**inflation 4.02×**, notoriety surge 5.9×.

---

## 13. Artefact diagnostics

Disproportionality does not distinguish a pharmacological association from a reporting
pattern that resembles one. Four documented mechanisms, all measurable from data
already collected:

| diagnostic | measure | flag |
|---|---|---|
| geographic concentration | Herfindahl index over reporting country | HHI ≥ 0.5 |
| litigation pattern | share of `primarysource.qualification` = 4 (lawyer) | ≥ 10% |
| consumer dominance | share of qualification = 5 | ≥ 60% |
| single-quarter dependence | largest quarter ÷ total new reports | ≥ 0.5 |

Herfindahl rather than top-category share because it responds to the whole
distribution: two countries at 45% each is concentrated in a way a top-share of 0.45
understates. FAERS is a US database, so US dominance is unremarkable — the flag that
matters is concentration in a single **non-US** country, which the marts separate.

A 10% lawyer share is far above background: across FAERS, lawyer-sourced reports are
low single digits, so a pair at 10% is driven by something other than clinical
observation.

`robustness_score` is a plain average of the four complements — deliberately not a
fitted weighting, since the mechanisms are not commensurable and nobody has ground
truth for their relative importance. It is a sorting aid, always shown beside its
components.

---

## 14. The prioritisation model

**Discrete-time hazard.** One row per (drug, reaction, quarter) *at risk*: a signal
has fired, no label change yet, and the pair is not prevalent at baseline. Rows stop
at the quarter of the label change, which carries the event. Censoring is handled
natively — a pair simply stops contributing rows.

Logistic regression on 15 standardised features: the shrinkage and frequentist
statistics, count and expected count on the log scale, quarters since signal, firing
stability, the artefact diagnostics, criteria firing, and the drug's share of reports.
Ratio measures enter as `log1p` because they are heavily right-skewed.

**Deliberately not class-weighted.** Balancing a rare-event outcome is the reflex and
it is wrong here: it shifts the intercept so predicted probabilities no longer match
observed rates, and an uncalibrated probability cannot be used to set a review
threshold — which is the whole point. Ranking, and therefore precision@k, is invariant
to the intercept, so balancing would buy nothing and cost the calibration. Verified:
with balancing, predicted 0.093 against observed 0.004; without, predicted 0.002–0.100
tracks observed 0.008–0.114 monotonically across deciles.

**The split is temporal**, at `holdout_from_quarter`. A random split would put later
quarters of the same pair in training and earlier ones in test, and would leak the era
— FAERS volume and FDA labelling behaviour both drift.

**Reported:** precision@k as the headline, because a reviewer works a queue of fixed
length — "of the 50 signals ranked highest, how many were labelled within two years"
is what decides whether the model is worth using. AUC and average precision alongside,
plus the calibration table, plus `lift_at_50` — a lift of 1.0 would mean the model
adds nothing over the disproportionality score alone.

A fit is refused below 30 training events, and the refusal is recorded rather than
silently producing coefficients that look like a result.

---

## 15. Known limitations

1. **Reporting associations, not risks.** FAERS has no denominator, no control group,
   no causality verification. A disproportionate ratio is a hypothesis.
2. **A label change is a proxy outcome**, reflecting FDA's assessment of evidence from
   many sources, most not FAERS.
3. **Left truncation is excluded, not solved.** Counts reported alongside results.
4. **MedDRA's hierarchy is licensed** — no SOC grouping; the synonym table is hand-built
   and incomplete.
5. **Only each drug's most-reported reactions** are evaluated, capped per drug for
   request budget.
6. **Publication lag** means a quarter-end cut approximates rather than reproduces what
   a reviewer could have seen. Biases lead time downward.
7. **Duplicate reports** are not de-duplicated beyond what openFDA does. FAERS
   duplication is well documented and inflates counts for heavily-reported pairs.
8. **The empirical null on the central mass is weaker** than curated negative controls.
   Which source was used is recorded per quarter.
9. **The cohort is 55 drugs**, chosen for label movement. Results do not generalise to
   drugs whose labels have never changed.
 10. **Two upstream failure modes, neither of them an unreliable index.** openFDA
     answers a `count` over an *analysed* string field with HTTP 500 rather than 400,
     and its documented 240 requests/minute is a burst ceiling rather than sustained
     capacity — a backfill at 200/minute lost three drugs where 60/minute succeeded
     30/30. Both presented together as a ~40% random failure rate on the whole index.
     Handled by a pre-flight field guard, a 90/minute sustained rate, and a circuit
     breaker that throttles rather than curtailing retries. A run that exhausts its
     request budget is recorded as partial rather than silently truncated. See
     ARCHITECTURE.md; the misdiagnosis is a better lesson than the bugs.

---

## 16. References

- DuMouchel, W. (1999). Bayesian data mining in large frequency tables, with an
  application to the FDA spontaneous reporting system. *The American Statistician*
  53(3), 177–190.
- DuMouchel, W. & Pregibon, D. (2001). Empirical Bayes screening for multi-item
  associations. *KDD '01*.
- Evans, S. J. W., Waller, P. C. & Davis, S. (2001). Use of proportional reporting
  ratios for signal generation from spontaneous adverse drug reaction reports.
  *Pharmacoepidemiology and Drug Safety* 10(6), 483–486.
- Mantel, N. & Haenszel, W. (1959). Statistical aspects of the analysis of data from
  retrospective studies of disease. *JNCI* 22(4), 719–748.
- Norén, G. N., Bate, A., Orre, R. & Edwards, I. R. (2006). Extending the methods used
  to screen the WHO drug safety database towards analysis of complex associations and
  improved accuracy for rare events. *Statistics in Medicine* 25(21), 3740–3757.
- Robins, J., Breslow, N. & Greenland, S. (1986). Estimators of the Mantel-Haenszel
  variance consistent in both sparse data and large-strata limiting models.
  *Biometrics* 42(2), 311–323.
- Rothman, K. J., Lanes, S. & Sacks, S. T. (2004). The reporting odds ratio and its
  advantages over the proportional reporting ratio. *Pharmacoepidemiology and Drug
  Safety* 13(8), 519–523.
- Schuemie, M. J., Ryan, P. B., DuMouchel, W., Suchard, M. A. & Madigan, D. (2014).
  Interpreting observational studies: why empirical calibration is needed to correct
  *p*-values. *Statistics in Medicine* 33(2), 209–218.

---

## Appendix: the leakage benchmark excludes left-truncated pairs

Worth stating separately because it is easy to get wrong, and the first
implementation here did.

The benchmark compares a statistic's value at the label change against its
retrospective value. That requires a meaningful "label change" date — which a
left-truncated pair does not have. A pair already described in a drug's earliest
archived label version has a recorded label date equal to the start of the archive,
which is an artefact of when DailyMed's history begins, not a regulatory event.

Left in, these pairs dominate: in a three-drug smoke run over 16 quarters, 92 of 144
pair-criterion observations were `prevalent_at_baseline`, because the most-reported
reactions of a GLP-1 agonist — nausea, vomiting, diarrhoea — have been labelled since
approval. Their "inflation ratio" is close to 1.0 by construction, and including them dragged
the measured median to 0.95× — understating the very effect the benchmark exists to
quantify.

Only `labelled_after_signal` and `labelled_before_signal` pairs are eligible. The
count of excluded pairs is reported alongside the result.
