"""Generate a synthetic warehouse with known properties, for CI.

Why a synthetic warehouse rather than a committed real one
---------------------------------------------------------
Three reasons, in order of importance.

**It makes the data-quality tests meaningful.** A dbt test that passes on real data
tells you the real data happened to be clean today. This generator plants specific
violations behind a flag, so the suite can assert that a *bad* warehouse fails --
which is the only way to know a test works at all. ``--with-violations`` exists for
exactly that.

**It removes the network from CI.** openFDA's event index was measured returning
intermittent 500s on roughly 40% of requests. A CI job that depends on it is a CI
job that fails for reasons unrelated to the change under review, and a suite that
fails randomly is a suite people stop reading.

**It keeps the repository small.** A real warehouse covering 55 drugs over 46
quarters is hundreds of megabytes.

The generated data is deliberately *realistic in shape* rather than in value: the
count distributions are heavy-tailed, disproportionality is concentrated in a few
pairs, labels accumulate mentions over versions, and a known fraction of pairs are
left-truncated. That is what the models and tests need to exercise their branches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

from prodrome.stats.contingency import Contingency
from prodrome.stats.criteria import evaluate_criteria
from prodrome.stats.disproportionality import score
from prodrome.stats.ebgm import fit_prior, score_cells
from prodrome.timeframe import Quarter, quarters_between
from prodrome.warehouse import open_warehouse

RNG_SEED = 20240101

FIXTURE_DRUGS = [
    # (unii, name, unii_coverage) -- coverage values span the range measured on the
    # real cohort, including the 0.0 case that motivates the union selector.
    ("53AXN4NNHX", "Semaglutide", 0.726),
    ("OYN3CCI6QE", "Tirzepatide", 1.000),
    ("3C06JJ0Z2O", "Osimertinib", 0.000),
    ("U1O3J18SFL", "Montelukast", 0.991),
    ("9N7R477WCK", "Tramadol", 0.203),
    ("FYS6T7F842", "Adalimumab", 0.075),
]

FIXTURE_REACTIONS = [
    "NAUSEA",
    "VOMITING",
    "DIARRHOEA",
    "ILEUS",
    "PANCREATITIS",
    "HEADACHE",
    "FATIGUE",
    "RASH",
    "DIZZINESS",
    "HEPATIC FAILURE",
    "SEIZURE",
    "ANAEMIA",
    "THROMBOCYTOPENIA",
    "PNEUMONITIS",
    "SUICIDAL IDEATION",
    "GASTROPARESIS",
]


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/warehouse/prodrome.duckdb"))
    parser.add_argument("--quarters", type=int, default=20)
    parser.add_argument(
        "--with-violations",
        action="store_true",
        help=(
            "Plant data-quality violations so the dbt tests can be asserted to "
            "actually fail. Used by tests/integration/test_dbt_tests_catch_violations.py."
        ),
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(RNG_SEED)
    first = Quarter(2020, 1)
    quarters = quarters_between(first, first.shift(args.quarters - 1))

    if args.out.exists():
        args.out.unlink()

    with open_warehouse(args.out) as warehouse:
        run_id = warehouse.start_run(
            command="fixture",
            prodrome_version="0.1.0",
            config_digest="fixture" + ("-violating" if args.with_violations else ""),
            cohort_size=len(FIXTURE_DRUGS),
            first_quarter=quarters[0].label,
            last_quarter=quarters[-1].label,
            openfda_last_updated=dt.date(2026, 7, 30),
            has_openfda_key=True,
            embed_backend="hashed",
            notes="synthetic fixture; see tools/make_fixture_warehouse.py",
        )

        warehouse.append_rows(
            "raw_cohort",
            [
                {
                    "run_id": run_id,
                    "drug_unii": unii,
                    "drug_name": name,
                    "spl_set_id": f"set-{unii.lower()}",
                    "substance_names": name.upper(),
                    "brand_names": name.upper(),
                    "unii_coverage": coverage,
                    "notes": "synthetic",
                }
                for unii, name, coverage in FIXTURE_DRUGS
            ],
        )

        # ---- label timeline: one version every three quarters ----------------
        label_versions: dict[str, list[tuple[int, dt.date]]] = {}
        version_rows = []
        for unii, _name, _coverage in FIXTURE_DRUGS:
            versions = []
            for index, quarter in enumerate(quarters[::3], start=1):
                date = quarter.end_date
                versions.append((index, date))
                version_rows.append(
                    {
                        "run_id": run_id,
                        "drug_unii": unii,
                        "spl_set_id": f"set-{unii.lower()}",
                        "spl_version": index,
                        "published_date": date,
                        "effective_date": date - dt.timedelta(days=5),
                        "authoritative_date": date - dt.timedelta(days=5),
                        "core_char_count": int(rng.integers(4000, 22000)),
                        "section_count": int(rng.integers(18, 32)),
                        # One version per drug is deliberately unusable, so the
                        # "never assert a gap from an unreadable label" test has
                        # something to exercise.
                        "is_usable": index != 2,
                    }
                )
            label_versions[unii] = versions
        warehouse.append_rows("raw_label_version", version_rows)

        # ---- contingency, statistics, label mentions -------------------------
        grand_totals = {
            q.label: int(9_000_000 + 420_000 * i + rng.integers(0, 40_000))
            for i, q in enumerate(quarters)
        }
        contingency_rows = []
        mention_rows = []
        quarterly_rows = []
        mix_rows = []

        for unii, _name, _coverage in FIXTURE_DRUGS:
            drug_scale = float(rng.uniform(0.002, 0.03))
            for reaction_index, reaction in enumerate(FIXTURE_REACTIONS):
                # A few pairs are genuinely elevated; most are not. That is the
                # distribution the gamma-Poisson prior exists to learn.
                elevated = (reaction_index + hash(unii)) % 7 == 0
                effect = float(rng.uniform(2.5, 9.0)) if elevated else float(rng.uniform(0.6, 1.6))
                reaction_prevalence = float(rng.uniform(0.0004, 0.02))

                # Left truncation: a third of pairs are already labelled at the
                # earliest archived version, so they must be excluded from
                # time-to-event analysis.
                prevalent = (reaction_index * 3 + len(unii)) % 3 == 0
                # Of the rest, some acquire a label mention partway through.
                add_at_version = (
                    None if prevalent else int(rng.integers(2, len(label_versions[unii]) + 2))
                )

                # Counts must be CUMULATIVE and therefore monotone: each cell counts
                # reports received on or before the quarter end. Drawing an
                # independent Poisson per quarter produces a series that can fall,
                # which is physically impossible and which
                # assert_cumulative_counts_are_monotone correctly rejects -- the
                # first version of this generator did exactly that and the test
                # caught it. Increments are drawn, then accumulated.
                cumulative_a = 0
                for quarter_index, quarter in enumerate(quarters):
                    grand = grand_totals[quarter.label]
                    drug_total = max(30, int(grand * drug_scale * (0.35 + 0.05 * quarter_index)))
                    reaction_total = max(40, int(grand * reaction_prevalence))
                    expected = drug_total * reaction_total / grand
                    # Per-quarter increment, scaled so the cumulative total tracks
                    # the expected count under the pair's effect size.
                    increment = rng.poisson(max(expected * effect / len(quarters), 0.05))
                    cumulative_a = int(cumulative_a + increment)
                    a = max(1, min(cumulative_a, drug_total, reaction_total))
                    try:
                        table = Contingency.from_marginals(
                            co_occurrence=a,
                            drug_total=drug_total,
                            reaction_total=reaction_total,
                            grand_total=grand,
                        )
                    except ValueError:
                        continue
                    contingency_rows.append(
                        {
                            "run_id": run_id,
                            "drug_unii": unii,
                            "reaction": reaction,
                            "as_of_quarter": quarter.label,
                            "a": table.a,
                            "b": table.b,
                            "c": table.c,
                            "d": table.d,
                            "a_source": "count_present" if a > 5 else "targeted",
                        }
                    )
                    quarterly_rows.append(
                        {
                            "run_id": run_id,
                            "drug_unii": unii,
                            "reaction": reaction,
                            "quarter": quarter.label,
                            "new_reports": int(max(0, rng.poisson(max(a / 6, 0.4)))),
                        }
                    )

                for version, date in label_versions[unii]:
                    labelled = prevalent or (
                        add_at_version is not None and version >= add_at_version
                    )
                    mention_rows.append(
                        {
                            "run_id": run_id,
                            "drug_unii": unii,
                            "spl_version": version,
                            "authoritative_date": date - dt.timedelta(days=5),
                            "reaction": reaction,
                            "verdict": ("labelled_core" if labelled else "not_labelled")
                            if version != 2
                            else "unknown",
                            "is_labelled": bool(labelled) and version != 2,
                            "best_semantic_score": float(rng.uniform(0.5, 0.8)),
                            "best_semantic_z": float(rng.normal(0.5, 1.2)),
                            "evidence_method": "lexical:exact" if labelled else None,
                            "evidence_score": 1.0 if labelled else None,
                            "evidence_section_code": "34084-4" if labelled else None,
                            "evidence_section_name": "Adverse reactions" if labelled else None,
                            "evidence_tier": "core" if labelled else None,
                            "evidence_snippet": (
                                f"{reaction.lower()} has been reported" if labelled else None
                            ),
                            "evidence_via": reaction.lower() if labelled else None,
                        }
                    )

                for dimension, categories in (
                    ("country", ["US", "GB", "DE", "JP", "CA"]),
                    ("qualification", ["1", "2", "3", "4", "5"]),
                ):
                    weights = rng.dirichlet(np.full(len(categories), 0.7))
                    for category, weight in zip(categories, weights, strict=True):
                        reports = int(weight * rng.integers(40, 800))
                        if reports <= 0:
                            continue
                        mix_rows.append(
                            {
                                "run_id": run_id,
                                "drug_unii": unii,
                                "reaction": reaction,
                                "as_of_quarter": quarters[-1].label,
                                "dimension": dimension,
                                "category": category,
                                "reports": reports,
                            }
                        )

        warehouse.append_batched("raw_contingency", contingency_rows)
        warehouse.append_batched("raw_label_mention", mention_rows)
        warehouse.append_batched("raw_quarterly_reports", quarterly_rows)
        warehouse.append_batched("raw_reporter_mix", mix_rows)

        # ---- statistics, computed with the real estimators -------------------
        stat_rows = []
        for quarter in quarters:
            cells = [r for r in contingency_rows if r["as_of_quarter"] == quarter.label]
            if not cells:
                continue
            tables = [
                Contingency(a=int(r["a"]), b=int(r["b"]), c=int(r["c"]), d=int(r["d"]))
                for r in cells
            ]
            counts = np.array([t.a for t in tables], dtype=float)
            expected = np.array([t.expected for t in tables], dtype=float)
            gps = None
            if len(tables) >= 50:
                try:
                    gps = score_cells(counts, expected, prior=fit_prior(counts, expected))
                except ValueError:
                    gps = None
            for index, (cell, table) in enumerate(zip(cells, tables, strict=True)):
                measures = score(table)
                fired = evaluate_criteria(measures)
                row = {
                    "run_id": run_id,
                    "drug_unii": cell["drug_unii"],
                    "reaction": cell["reaction"],
                    "as_of_quarter": quarter.label,
                    **{
                        k: _finite(v)
                        for k, v in measures.as_row().items()
                        if k not in {"degenerate"}
                    },
                    "degenerate": measures.degenerate,
                    "ebgm": _finite(gps.ebgm[index]) if gps else None,
                    "eb05": _finite(gps.eb05[index]) if gps else None,
                    "eb95": _finite(gps.eb95[index]) if gps else None,
                    "gps_posterior_noise_weight": (
                        _finite(gps.posterior_weight[index]) if gps else None
                    ),
                    "raw_p": None,
                    "calibrated_p": None,
                    "n_criteria_firing": sum(fired.values()),
                }
                row.update({f"fired_{k}": v for k, v in fired.items()})
                stat_rows.append(row)
        warehouse.append_batched("stat_disproportionality", stat_rows)

        # Violations are planted last, and written directly, for two reasons. The
        # Contingency dataclass rejects a negative cell -- which is the guard working,
        # and which means a planted bad row cannot pass through the statistics loop --
        # and a violating row must not contaminate the fitted GPS prior, or the
        # "good" statistics would themselves be wrong.
        if args.with_violations:
            violations, bad_mentions = _violation_rows(run_id, quarters)
            warehouse.append_rows("raw_contingency", violations)
            warehouse.append_rows("raw_label_mention", bad_mentions)

        warehouse.finish_run(run_id, requests=len(stat_rows) // 4, cache_hits=len(stat_rows))

    print(
        f"wrote {args.out}\n"
        f"  {len(FIXTURE_DRUGS)} drugs x {len(FIXTURE_REACTIONS)} reactions "
        f"x {len(quarters)} quarters\n"
        f"  {len(contingency_rows):,} contingency cells, {len(stat_rows):,} statistic rows, "
        f"{len(mention_rows):,} label mentions"
        + (
            "\n  VIOLATIONS PLANTED -- dbt tests are expected to fail"
            if args.with_violations
            else ""
        )
    )
    return 0


def _violation_rows(
    run_id: str, quarters: list[Quarter]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """One violation per data-quality test, so each can be proven to fire.

    A test that has never been observed to fail is not known to work. These rows
    are what tests/integration/test_dbt_tests_catch_violations.py asserts against.
    """
    contingency_rows: list[dict[str, object]] = []
    mention_rows: list[dict[str, object]] = []
    # assert_contingency_margins_agree: a exceeding its own margin.
    contingency_rows.append(
        {
            "run_id": run_id,
            "drug_unii": "53AXN4NNHX",
            "reaction": "VIOLATION MARGIN",
            "as_of_quarter": quarters[0].label,
            "a": 500,
            "b": -400,
            "c": 10,
            "d": 1000,
            "a_source": "targeted",
        }
    )
    # assert_cumulative_counts_are_monotone: a cumulative count that falls.
    for index, quarter in enumerate(quarters[:3]):
        contingency_rows.append(
            {
                "run_id": run_id,
                "drug_unii": "53AXN4NNHX",
                "reaction": "VIOLATION MONOTONE",
                "as_of_quarter": quarter.label,
                "a": 400 - index * 150,
                "b": 5000,
                "c": 900,
                "d": 9_000_000,
                "a_source": "count_present",
            }
        )
    # assert_label_gap_is_never_asserted_from_an_unusable_label: a definite verdict
    # recorded against a version flagged unusable.
    mention_rows.append(
        {
            "run_id": run_id,
            "drug_unii": "53AXN4NNHX",
            "spl_version": 2,
            "authoritative_date": quarters[0].end_date,
            "reaction": "VIOLATION UNUSABLE",
            "verdict": "not_labelled",
            "is_labelled": False,
            "best_semantic_score": 0.4,
            "best_semantic_z": 0.1,
            "evidence_method": None,
            "evidence_score": None,
            "evidence_section_code": None,
            "evidence_section_name": None,
            "evidence_tier": None,
            "evidence_snippet": None,
            "evidence_via": None,
        }
    )
    return contingency_rows, mention_rows


def _finite(value: object) -> object:
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    sys.exit(main())
