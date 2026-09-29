"""Exporting the marts for Tableau and for the static dashboard.

Three formats, each because something downstream genuinely needs it:

``.hyper``
    Tableau's native extract. Written with pantab, which talks to the same Hyper
    API Tableau Desktop uses, so the result opens as a proper extract with typed
    columns rather than a re-parsed CSV. Chosen over publishing CSVs and letting
    Tableau infer types because inference gets quarter labels wrong -- "2023Q1"
    parses as a string on one machine and a date on another, and the axis silently
    reorders.
``.parquet``
    For anyone who wants the marts in pandas, DuckDB or Polars without installing
    Tableau. Typed, compressed, and the format a reviewer reproducing a figure will
    reach for.
``.json``
    A single compact bundle for the static dashboard. Aggregated and rounded here
    rather than in the browser: the dashboard is a static page with no backend, so
    whatever it needs has to arrive precomputed, and shipping raw marts to a browser
    would mean a multi-megabyte download to render a dozen charts.

Every export carries the run manifest, so a published figure can always be traced
back to the run, the config digest and the data vintage that produced it.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prodrome.warehouse import Warehouse

logger = logging.getLogger(__name__)

#: Marts exported in full. Order is the order they appear in the Tableau extract.
EXPORTED_MARTS: tuple[str, ...] = (
    "mart_run_manifest",
    "mart_criterion_leadtime",
    "mart_drug_summary",
    "mart_label_gap",
    "mart_leakage_benchmark",
    "mart_signal_trend",
)

#: Rows of the trend mart beyond which the dashboard bundle samples rather than
#: embedding everything. A static page should not ship tens of megabytes.
DASHBOARD_TREND_LIMIT = 4_000


@dataclass
class ExportResult:
    """What was written, for the CLI summary and the CI artefact listing."""

    export_dir: Path
    hyper_path: Path | None = None
    parquet_paths: list[Path] = field(default_factory=list)
    dashboard_path: Path | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [f"exported to {self.export_dir}"]
        for name, count in self.row_counts.items():
            lines.append(f"  {name:28s} {count:>9,d} rows")
        if self.hyper_path:
            size = self.hyper_path.stat().st_size / 1_048_576
            lines.append(f"  Tableau extract             {self.hyper_path.name} ({size:.1f} MB)")
        if self.dashboard_path:
            size = self.dashboard_path.stat().st_size / 1024
            lines.append(f"  dashboard bundle           {self.dashboard_path.name} ({size:.0f} KB)")
        if self.skipped:
            lines.append(f"  skipped (absent)           {', '.join(self.skipped)}")
        return lines


def _read_mart(warehouse: Warehouse, name: str) -> pd.DataFrame | None:
    """Read a mart, or None when dbt has not built it yet.

    Goes through `read_table` so the dbt schema layout is resolved rather than
    assumed: marts land in ``main_marts``, not ``main``.
    """
    try:
        return warehouse.read_table(name)
    except KeyError:
        return None


def _clean_for_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Make a frame safe for Hyper and Parquet.

    Three problems, all of which produce a hard failure or a wrong axis rather than a
    warning.

    **An entirely-null column has no Arrow type.** pyarrow infers ``null`` for it, and
    Hyper rejects that with ``Unsupported Arrow type: na``. This is not a hypothetical:
    ``median_lead_quarters`` is legitimately NULL for every row when no criterion
    reached a median, which is exactly what happens on a cohort whose labelled pairs
    are all left-truncated. Such columns are cast to a concrete type -- float when the
    column was numeric, empty string otherwise -- so the column survives into the
    extract as an explicitly empty one rather than failing the export.

    **pd.NA promotes a float column to object.** Using it to null out infinities
    changes the column's dtype, which then re-enters the all-null problem above.
    ``np.nan`` keeps float64.

    **Infinity round-trips through Parquet but renders as a number in Tableau**, which
    would put "inf" on an axis. Non-finite floats become null, which Tableau draws as
    a gap.
    """
    cleaned = frame.copy()
    for column in cleaned.columns:
        series = cleaned[column]
        was_numeric = pd.api.types.is_numeric_dtype(series)
        if pd.api.types.is_float_dtype(series):
            # np.nan, not pd.NA: pd.NA would promote the column to object dtype.
            series = series.replace([math.inf, -math.inf], np.nan)
        elif series.dtype == object and not series.isna().all():
            series = series.astype("string").fillna("")
        if series.isna().all():
            series = (
                series.astype("float64")
                if was_numeric
                else pd.Series([""] * len(series), index=series.index, dtype="string")
            )
        cleaned[column] = series
    return cleaned


def write_hyper(frames: dict[str, pd.DataFrame], path: Path) -> Path | None:
    """Write a multi-table Tableau extract.

    Returns None when pantab is not installed, which is the normal state in CI --
    the extract is a publishing artefact, not something the test suite needs.
    """
    try:
        import pantab
    except ImportError:
        logger.info(
            "pantab not installed; skipping the Tableau extract (pip install -e '.[tableau]')"
        )
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # pantab appends to an existing extract, which would duplicate every row on
        # a re-run and quietly double the numbers in the workbook.
        path.unlink()
    pantab.frames_to_hyper({name: _clean_for_export(f) for name, f in frames.items()}, path)
    return path


def write_parquet(frames: dict[str, pd.DataFrame], directory: Path) -> list[Path]:
    """Write one Parquet file per mart."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in frames.items():
        target = directory / f"{name}.parquet"
        _clean_for_export(frame).to_parquet(target, index=False, compression="zstd")
        written.append(target)
    return written


def _round_records(frame: pd.DataFrame, digits: int = 4) -> list[dict[str, Any]]:
    """Frame to JSON records with floats rounded and non-finite values nulled."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for raw_key, value in row.items():
            # pandas types column labels as Hashable; JSON keys must be strings.
            key = str(raw_key)
            if isinstance(value, float):
                clean[key] = round(value, digits) if math.isfinite(value) else None
            elif pd.isna(value):
                clean[key] = None
            elif hasattr(value, "isoformat"):
                clean[key] = value.isoformat()[:10]
            else:
                clean[key] = value
        records.append(clean)
    return records


def build_dashboard_bundle(frames: dict[str, pd.DataFrame], warehouse: Warehouse) -> dict[str, Any]:
    """Assemble the single JSON document the static dashboard reads.

    Everything the page needs, precomputed. The page has no backend, so any
    aggregation left undone here would have to happen in the browser over a much
    larger payload.
    """
    bundle: dict[str, Any] = {"schema_version": 1}

    manifest = frames.get("mart_run_manifest")
    bundle["manifest"] = (
        _round_records(manifest)[0] if manifest is not None and not manifest.empty else {}
    )

    for key, mart in (
        ("criteria", "mart_criterion_leadtime"),
        ("drugs", "mart_drug_summary"),
    ):
        frame = frames.get(mart)
        bundle[key] = _round_records(frame) if frame is not None else []

    gaps = frames.get("mart_label_gap")
    bundle["gaps"] = _round_records(gaps) if gaps is not None else []

    leakage = frames.get("mart_leakage_benchmark")
    if leakage is not None and not leakage.empty:
        ratios = leakage["inflation_ratio"].dropna()
        ratios = ratios[ratios.apply(math.isfinite)]
        bundle["leakage"] = {
            "rows": _round_records(leakage.nlargest(200, "inflation_ratio")),
            "median_inflation": round(float(ratios.median()), 3) if len(ratios) else None,
            "p90_inflation": round(float(ratios.quantile(0.9)), 3) if len(ratios) else None,
            "n_pairs": len(leakage),
            "n_notorious": int(leakage["is_notorious"].fillna(False).sum()),
            "n_materially_inflated": int(leakage["materially_inflated"].fillna(False).sum()),
        }
    else:
        bundle["leakage"] = {"rows": [], "median_inflation": None, "n_pairs": 0}

    trend = frames.get("mart_signal_trend")
    if trend is not None and not trend.empty:
        # Keep every quarter of the pairs that matter rather than a random sample of
        # all of them: a truncated series would draw as a line that stops for no
        # visible reason.
        focus = trend.groupby(["drug_unii", "reaction"])["reports"].max().nlargest(80).index
        subset = trend.set_index(["drug_unii", "reaction"]).loc[focus].reset_index()
        bundle["trend"] = _round_records(subset.head(DASHBOARD_TREND_LIMIT))
    else:
        bundle["trend"] = []

    for key, table in (
        ("survival", "model_survival_points"),
        ("calibration", "model_calibration"),
        ("coefficients", "model_hazard_coefficient"),
        ("queue", "model_priority_queue"),
    ):
        frame = _read_mart(warehouse, table)
        bundle[key] = _round_records(frame) if frame is not None else []

    return bundle


def export_all(
    warehouse: Warehouse,
    export_dir: Path,
    *,
    write_tableau: bool = True,
    write_dashboard: bool = True,
) -> ExportResult:
    """Export every available mart."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(export_dir=export_dir)

    frames: dict[str, pd.DataFrame] = {}
    for name in EXPORTED_MARTS:
        frame = _read_mart(warehouse, name)
        if frame is None:
            result.skipped.append(name)
            continue
        frames[name] = frame
        result.row_counts[name] = len(frame)

    if not frames:
        logger.warning("no marts found. Build them first: cd dbt && dbt build")
        return result

    result.parquet_paths = write_parquet(frames, export_dir / "parquet")

    if write_tableau:
        result.hyper_path = write_hyper(frames, export_dir / "prodrome.hyper")

    if write_dashboard:
        bundle = build_dashboard_bundle(frames, warehouse)
        target = export_dir / "dashboard.json"
        target.write_text(
            json.dumps(bundle, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        result.dashboard_path = target

    return result
