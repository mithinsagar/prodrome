# Dashboard data

`dashboard.json` is the single bundle the page reads. It is **derived** — regenerated
by `prodrome export`, which stages it here — but a copy is committed so the dashboard
is viewable immediately after a clone, before anyone has spent hours of API quota:

```bash
make dashboard        # http://localhost:8000
```

The committed copy is from a three-drug smoke run over 2021Q1–2024Q4, so its numbers
are illustrative of the *shape* of the output rather than the published result. Two
consequences are visible in it and are correct rather than broken:

- **No lead times.** The top reactions of GLP-1 agonists (nausea, vomiting,
  diarrhoea) have been labelled since approval, so every pair in this sample is
  left-truncated and excluded from the time-to-event analysis by design. Lead time
  needs rarer reactions, which needs the deeper count aggregations an API key unlocks.
- **No fitted model.** With no observed label changes in the training window there are
  no events to fit on, so the ranking falls back to the shrinkage observed-to-expected
  ratio and the brief says so instead of reporting a precision of 0%.

The weekly workflow regenerates this from the full cohort and publishes it to GitHub
Pages, so the live dashboard reflects the real run.
