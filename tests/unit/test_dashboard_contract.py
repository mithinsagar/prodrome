"""The contract between the exported bundle and the dashboard's JavaScript.

The dashboard is a static page with no backend, so a bundle missing a key is a
blank section for every viewer with no server-side error to notice. The page and the
exporter are written in different languages and cannot share a type, so the contract
is asserted here instead.

Every field listed below is read by name in `dashboard/app.js`. Adding a field to the
page means adding it here, which is the point: the test fails until the exporter
actually provides it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from prodrome.publish.export import build_dashboard_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_JS = REPO_ROOT / "dashboard" / "app.js"
INDEX_HTML = REPO_ROOT / "dashboard" / "index.html"

#: Top-level keys app.js reads off the bundle.
REQUIRED_BUNDLE_KEYS = {
    "schema_version",
    "manifest",
    "criteria",
    "drugs",
    "gaps",
    "leakage",
    "trend",
    "survival",
    "calibration",
    "coefficients",
    "queue",
}

#: Per-collection fields app.js reads. Kept explicit rather than derived, so a
#: rename on either side is caught rather than silently producing blanks.
REQUIRED_FIELDS: dict[str, set[str]] = {
    "criteria": {
        "criterion",
        "median_lead_quarters",
        "median_lead_months",
        "signalled_then_labelled",
        "signalled_not_yet_labelled",
        "labelled_before_signal",
        "never_signalled",
        "excluded_left_truncated",
        "mean_firing_fraction",
        "labelled_by_1y",
        "labelled_by_2y",
        "labelled_by_3y",
        "share_of_signals_labelled",
    },
    "gaps": {
        "drug_unii",
        "drug_name",
        "reaction",
        "reports",
        "expected",
        "prr",
        "ror",
        "ror_ci_lower",
        "eb05",
        "ebgm",
        "calibrated_p",
        "significance_lost_to_calibration",
        "n_criteria_firing",
        "signal_quarter",
        "hazard_probability",
        "robustness_score",
        "artefact_flags",
        "relies_on_name_matching",
    },
    "drugs": {
        "drug_name",
        "open_label_gaps",
        "signals_that_were_labelled",
        "open_signals",
        "median_lead_quarters",
        "reactions_tracked",
        "latest_drug_reports",
        "usable_label_versions",
        "unii_coverage",
        "relies_on_name_matching",
        "first_label_date",
        "latest_label_date",
        "already_labelled_at_baseline",
        "notes",
    },
    "trend": {
        "drug_unii",
        "drug_name",
        "reaction",
        "as_of_quarter",
        "reports",
        "expected",
        "prr",
        "ror",
        "ror_ci_lower",
        "eb05",
        "chi2_yates",
        "n_criteria_firing",
        "labelled_at_quarter",
        "transition_type",
    },
}


@pytest.fixture(scope="module")
def bundle() -> dict[str, object]:
    """A bundle built from minimal but structurally complete frames."""

    class _Stub:
        """Stands in for a Warehouse for the model_* tables the bundle also reads."""

        def table_exists(self, name: str) -> bool:
            return True

        def read_table(self, name: str) -> pd.DataFrame:
            return self.query_df("")

        def query_df(self, sql: str, params: object = None) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "criterion": "ema_ror025",
                        "quarters": 4.0,
                        "survival": 0.8,
                        "decile": 0,
                        "mean_predicted": 0.01,
                        "observed_rate": 0.012,
                        "n_rows": 100,
                        "feature": "log2_oe_shrunk",
                        "coefficient": 0.8,
                        "drug_unii": "U",
                        "reaction": "R",
                        "quarters_since_signal": 3,
                    }
                ]
            )

    frames = {
        "mart_run_manifest": pd.DataFrame(
            [
                {
                    "run_id": "abc",
                    "first_quarter": "2020Q1",
                    "last_quarter": "2024Q4",
                    "openfda_last_updated": "2026-07-30",
                    "drugs": 6,
                    "drugs_reliant_on_name_matching": 2,
                    "contingency_cells": 1000,
                    "quarters": 20,
                    "embed_backend": "onnx",
                    "requests": 500,
                    "prodrome_version": "0.1.0",
                    "model_lift_at_50": 4.2,
                    "model_precision_at_50": 0.18,
                    "model_base_rate": 0.03,
                    "total_open_gaps": 42,
                }
            ]
        ),
        "mart_criterion_leadtime": pd.DataFrame(
            [dict.fromkeys(REQUIRED_FIELDS["criteria"], 1.0) | {"criterion": "ema_ror025"}]
        ),
        "mart_drug_summary": pd.DataFrame(
            [dict.fromkeys(REQUIRED_FIELDS["drugs"], 1) | {"drug_name": "Semaglutide"}]
        ),
        "mart_label_gap": pd.DataFrame(
            [
                dict.fromkeys(REQUIRED_FIELDS["gaps"], 1.0)
                | {
                    "drug_unii": "U",
                    "reaction": "ILEUS",
                    "drug_name": "Semaglutide",
                    "artefact_flags": "",
                    "signal_quarter": "2022Q2",
                }
            ]
        ),
        "mart_leakage_benchmark": pd.DataFrame(
            [
                {
                    "drug_unii": "U",
                    "drug_name": "Semaglutide",
                    "reaction": "ILEUS",
                    "inflation_ratio": 4.02,
                    "is_notorious": True,
                    "materially_inflated": True,
                    "at_label_change": 1.72,
                    "retrospective": 6.92,
                }
            ]
        ),
        "mart_signal_trend": pd.DataFrame(
            [
                dict.fromkeys(REQUIRED_FIELDS["trend"], 1.0)
                | {
                    "drug_unii": "U",
                    "reaction": "ILEUS",
                    "drug_name": "Semaglutide",
                    "as_of_quarter": "2023Q3",
                    "transition_type": "added",
                }
            ]
        ),
    }
    return build_dashboard_bundle(frames, _Stub())  # type: ignore[arg-type]


def test_bundle_has_every_key_the_page_reads(bundle: dict[str, object]) -> None:
    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    assert not missing, f"the dashboard reads these keys and the bundle omits them: {missing}"


@pytest.mark.parametrize("collection", sorted(REQUIRED_FIELDS))
def test_collections_carry_every_field_the_page_reads(
    bundle: dict[str, object], collection: str
) -> None:
    rows = bundle[collection]
    assert isinstance(rows, list) and rows, f"{collection} should be a non-empty list"
    missing = REQUIRED_FIELDS[collection] - set(rows[0])
    assert not missing, f"{collection} rows omit fields the dashboard reads: {missing}"


def test_bundle_is_json_serialisable_without_nan(bundle: dict[str, object]) -> None:
    """NaN is not valid JSON.

    `json.dumps` emits a bare `NaN` by default, which `JSON.parse` rejects -- so a
    single non-finite float anywhere would make the whole dashboard fail to load.
    The exporter passes allow_nan=False for this reason; this asserts it holds.
    """
    json.dumps(bundle, allow_nan=False)


def test_leakage_summary_shape(bundle: dict[str, object]) -> None:
    leakage = bundle["leakage"]
    assert isinstance(leakage, dict)
    assert {"rows", "median_inflation", "n_pairs"} <= set(leakage)


def test_every_element_id_used_by_the_script_exists_in_the_html() -> None:
    """A getElementById on a missing id is a silent no-op that blanks a section."""
    script = APP_JS.read_text(encoding="utf-8")
    markup = INDEX_HTML.read_text(encoding="utf-8")
    referenced = set(re.findall(r"getElementById\(['\"]([\w-]+)['\"]\)", script))
    present = set(re.findall(r"""id=["']([\w-]+)["']""", markup))
    missing = referenced - present
    assert not missing, f"app.js references ids absent from index.html: {sorted(missing)}"


def test_the_page_declares_both_theme_scopes() -> None:
    """Dark values must be declared under the media query *and* the data-theme
    scope, or the toggle only works in one direction."""
    css = (REPO_ROOT / "dashboard" / "styles.css").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in css
    assert ':root[data-theme="dark"]' in css
    assert "--surface-1" in css and "--series-1" in css
