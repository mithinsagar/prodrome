"""Command line interface.

Stages are separate commands rather than one monolith, because they have very
different costs: ``ingest`` spends API quota and takes hours on a cold cache,
``score`` and ``latency`` are pure computation and take seconds. Being able to
re-run the cheap ones without the expensive one is what makes the statistics
iterable.

``prodrome run`` chains all three for the scheduled weekly job.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from prodrome import __version__
from prodrome.brief.compose import compose_brief
from prodrome.brief.evidence import build_evidence
from prodrome.brief.llm import build_generator
from prodrome.clients.factory import build_dailymed, build_openfda
from prodrome.config import Config, Settings, get_settings, load_config, redact
from prodrome.pipeline import run_ingest, run_latency, run_score
from prodrome.publish import export_all
from prodrome.timeframe import Quarter
from prodrome.warehouse import WarehouseBusyError, open_warehouse
from prodrome.warehouse.duck import config_digest

app = typer.Typer(
    name="prodrome",
    help="Signal-to-label latency engine for FDA adverse event data.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)

CONFIG_OPTION = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to the analysis plan YAML.")
]
VERBOSE_OPTION = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # httpx logs every request at INFO, which buries the pipeline's own output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _load(config_path: Path | None) -> tuple[Config, Settings]:
    config = load_config(config_path)
    settings = get_settings()
    if not config.cohort:
        error_console.print(
            "[red]The cohort is empty.[/red] Resolve one first:\n  python tools/resolve_cohort.py"
        )
        raise typer.Exit(1)
    return config, settings


def _limit_cohort(config: Config, limit: int | None) -> Config:
    """Trim the cohort, for smoke tests and cheap iteration."""
    if limit is None or limit >= len(config.cohort):
        return config
    trimmed = config.model_copy(update={"cohort": config.cohort[:limit]})
    console.print(f"[yellow]Cohort limited to the first {limit} drugs.[/yellow]")
    return trimmed


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"prodrome {__version__}")


@app.command()
def doctor(config_path: CONFIG_OPTION = None) -> None:
    """Check the environment before a long run.

    Reports what is configured and what that costs, so the two failure modes that
    waste the most time -- a missing API key and a missing embedding backend -- are
    caught in a second rather than three hours in.
    """
    settings = get_settings()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        error_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title="prodrome environment", show_lines=False)
    table.add_column("check")
    table.add_column("value")
    table.add_column("consequence")

    has_key = bool(settings.openfda_api_key)
    table.add_row(
        "openFDA API key",
        redact(settings.openfda_api_key) if has_key else "[red]not set[/red]",
        "120,000 requests/day, 1,000-term aggregations"
        if has_key
        else "[yellow]1,000 requests/day, 100-term aggregations -- a full backfill "
        "will not complete[/yellow]",
    )
    backend = config.labelmatch.embed_backend or settings.embed_backend
    try:
        from prodrome.labelmatch.embed import build_embedder

        embedder = build_embedder(backend, model_name=config.labelmatch.onnx_model)
        embed_status, embed_note = embedder.name, "ready"
    except (ImportError, ValueError) as exc:
        embed_status, embed_note = f"[red]{backend}[/red]", str(exc)[:70]
    table.add_row("embedding backend", embed_status, embed_note)

    table.add_row("cohort size", str(len(config.cohort)), "drugs tracked")
    with_labels = sum(1 for d in config.cohort if d.spl_set_id)
    table.add_row(
        "with label timeline",
        f"{with_labels}/{len(config.cohort)}",
        "only these can contribute a latency outcome",
    )
    name_reliant = sum(1 for d in config.cohort if d.relies_on_name_matching)
    table.add_row(
        "reliant on name matching",
        str(name_reliant),
        "openFDA UNII harmonisation does not cover these; see prodrome.selector",
    )
    table.add_row("warehouse", str(settings.warehouse_path), "DuckDB file")
    table.add_row("cache", str(settings.cache_dir), "replayed on re-runs")
    table.add_row(
        "LLM brief",
        settings.llm_provider if settings.llm_enabled else "disabled",
        "narrative brief" if settings.llm_enabled else "deterministic template will be used",
    )
    console.print(table)
    if not has_key:
        console.print(
            "\nGet a free openFDA key (no card, instant): "
            "[cyan]https://open.fda.gov/apis/authentication/[/cyan]\n"
            "Then set PRODROME_OPENFDA_API_KEY in .env"
        )


@app.command()
def ingest(
    config_path: CONFIG_OPTION = None,
    cohort_limit: Annotated[
        int | None, typer.Option("--cohort-limit", help="Use only the first N drugs.")
    ] = None,
    last_quarter: Annotated[
        str | None, typer.Option("--last-quarter", help="Override the final quarter, e.g. 2024Q4.")
    ] = None,
    skip_labels: Annotated[bool, typer.Option("--skip-labels")] = False,
    skip_diagnostics: Annotated[bool, typer.Option("--skip-diagnostics")] = False,
    request_budget: Annotated[
        int | None, typer.Option("--request-budget", help="Cap requests for this run.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Estimate the request cost and stop.")
    ] = False,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Stage 1: fetch adverse-event counts and label history into the warehouse."""
    _setup_logging(verbose)
    config, settings = _load(config_path)
    config = _limit_cohort(config, cohort_limit)

    events, events_transport = build_openfda(config, settings, request_budget=request_budget)
    labels_client, labels_transport = build_dailymed(config, settings)

    # Precedence: explicit flag, then the config's window, whose "auto" resolves
    # against openFDA's own last_updated rather than today's date -- so the final
    # quarter is never a partially populated stub.
    if last_quarter:
        resolved_last = Quarter.parse(last_quarter)
    else:
        resolved_last = config.window.resolve_last(events.latest_complete_quarter())
    first = Quarter.parse(config.window.first_quarter)
    n_quarters = resolved_last - first + 1

    if dry_run:
        from prodrome.ingest.contingency import ContingencyHarvester

        harvester = ContingencyHarvester(events, config)
        estimate = harvester.estimate_requests(
            len(config.cohort), n_quarters, config.max_reactions_per_drug
        )
        quota = settings.openfda_daily_quota
        console.print(
            f"Window {first} .. {resolved_last} ({n_quarters} quarters), "
            f"{len(config.cohort)} drugs\n"
            f"Estimated openFDA requests: [bold]{estimate:,}[/bold]\n"
            f"Daily quota available:      {quota:,}\n"
            f"Days of traffic:            {estimate / quota:.1f}"
        )
        for transport in (events_transport, labels_transport):
            transport.close()
        raise typer.Exit(0)

    with open_warehouse(settings.warehouse_path) as warehouse:
        run_id = warehouse.start_run(
            command="ingest",
            prodrome_version=__version__,
            config_digest=config_digest(config.model_dump(mode="json")),
            cohort_size=len(config.cohort),
            first_quarter=first.label,
            last_quarter=resolved_last.label,
            openfda_last_updated=events.last_updated(),
            has_openfda_key=bool(settings.openfda_api_key),
            embed_backend=config.labelmatch.embed_backend or settings.embed_backend,
        )
        console.print(f"[bold]ingest[/bold] run {run_id}: {first} .. {resolved_last}")
        try:
            result = run_ingest(
                warehouse,
                config,
                settings,
                run_id=run_id,
                events=events,
                dailymed=labels_client,
                transports=[events_transport, labels_transport],
                last_quarter=resolved_last,
                skip_labels=skip_labels,
                skip_diagnostics=skip_diagnostics,
            )
        finally:
            for transport in (events_transport, labels_transport):
                transport.close()
        for line in result.summary_lines():
            console.print(line)


@app.command()
def score(
    config_path: CONFIG_OPTION = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Stage 2: compute calibrated disproportionality statistics."""
    _setup_logging(verbose)
    config, settings = _load(config_path)
    with open_warehouse(settings.warehouse_path) as warehouse:
        target = run_id or _latest_run(warehouse)
        console.print(f"[bold]score[/bold] run {target}")
        for line in run_score(warehouse, config, run_id=target).summary_lines():
            console.print(line)


@app.command()
def latency(
    config_path: CONFIG_OPTION = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Stage 3: measure lead time, leakage and fit the prioritisation model."""
    _setup_logging(verbose)
    config, settings = _load(config_path)
    with open_warehouse(settings.warehouse_path) as warehouse:
        target = run_id or _latest_run(warehouse)
        console.print(f"[bold]latency[/bold] run {target}")
        for line in run_latency(warehouse, config, run_id=target).summary_lines():
            console.print(line)


@app.command(name="run")
def run_all(
    config_path: CONFIG_OPTION = None,
    cohort_limit: Annotated[int | None, typer.Option("--cohort-limit")] = None,
    last_quarter: Annotated[str | None, typer.Option("--last-quarter")] = None,
    skip_labels: Annotated[bool, typer.Option("--skip-labels")] = False,
    skip_diagnostics: Annotated[bool, typer.Option("--skip-diagnostics")] = False,
    request_budget: Annotated[int | None, typer.Option("--request-budget")] = None,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Run all three stages. This is what the weekly job invokes."""
    ingest(
        config_path=config_path,
        cohort_limit=cohort_limit,
        last_quarter=last_quarter,
        skip_labels=skip_labels,
        skip_diagnostics=skip_diagnostics,
        request_budget=request_budget,
        dry_run=False,
        verbose=verbose,
    )
    score(config_path=config_path, run_id=None, verbose=verbose)
    latency(config_path=config_path, run_id=None, verbose=verbose)


@app.command()
def export(
    export_dir: Annotated[
        Path | None, typer.Option("--out", help="Directory to write into.")
    ] = None,
    no_tableau: Annotated[bool, typer.Option("--no-tableau")] = False,
    no_dashboard: Annotated[bool, typer.Option("--no-dashboard")] = False,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Export the dbt marts as Parquet, a Tableau .hyper extract and the dashboard bundle.

    Requires the dbt models to have been built: `cd dbt && dbt build`.
    """
    _setup_logging(verbose)
    settings = get_settings()
    target = export_dir or settings.export_dir
    with open_warehouse(settings.warehouse_path, read_only=True) as warehouse:
        result = export_all(
            warehouse,
            target,
            write_tableau=not no_tableau,
            write_dashboard=not no_dashboard,
        )
    for line in result.summary_lines():
        console.print(line)

    # The dashboard reads its bundle from its own directory, so a successful export
    # also stages it there -- otherwise `make dashboard` would serve stale data.
    if result.dashboard_path is not None:
        staged = Path("dashboard/data/dashboard.json")
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(result.dashboard_path.read_bytes())
        console.print(f"  staged for the dashboard    {staged}")


@app.command()
def brief(
    out: Annotated[Path | None, typer.Option("--out", help="Write the brief here.")] = None,
    gaps: Annotated[int, typer.Option("--gaps", help="How many label gaps to cover.")] = 8,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Write the weekly brief.

    Deterministic by default. If an LLM provider and key are configured, the model is
    asked to rewrite the same evidence and its output is published only if every
    number in it traces back to the evidence pack.
    """
    _setup_logging(verbose)
    settings = get_settings()
    with open_warehouse(settings.warehouse_path, read_only=True) as warehouse:
        run_id = _latest_run(warehouse)
        pack = build_evidence(warehouse, run_id=run_id, gap_count=gaps)
    generator = build_generator(settings.llm_provider, settings.llm_api_key, settings.llm_model)
    composed = compose_brief(pack, generator)

    target = out or (settings.export_dir / "brief.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(composed.markdown, encoding="utf-8")

    console.print(f"[bold]brief[/bold] written to {target}")
    console.print(f"  source                {composed.source}")
    if composed.verification is not None:
        console.print(f"  verification          {composed.verification.summary()}")
    if composed.rejection_reason:
        console.print(f"  [yellow]model output rejected[/yellow] {composed.rejection_reason}")


@app.command()
def status() -> None:
    """Show what the warehouse contains."""
    settings = get_settings()
    if not Path(settings.warehouse_path).exists():
        console.print(f"No warehouse at {settings.warehouse_path}. Run `prodrome ingest`.")
        raise typer.Exit(0)
    with open_warehouse(settings.warehouse_path, read_only=True) as warehouse:
        runs = warehouse.query(
            "SELECT run_id, command, started_at, finished_at, cohort_size, "
            "first_quarter, last_quarter, requests, cache_hits FROM runs "
            "ORDER BY started_at DESC LIMIT 10"
        )
        table = Table(title="recent runs")
        for column in (
            "run",
            "command",
            "started",
            "done",
            "drugs",
            "window",
            "requests",
            "cached",
        ):
            table.add_column(column)
        for r in runs:
            table.add_row(
                str(r[0])[:10],
                str(r[1]),
                str(r[2])[:16],
                "yes" if r[3] else "[yellow]no[/yellow]",
                str(r[4]),
                f"{r[5]}..{r[6]}",
                f"{r[7]:,}",
                f"{r[8]:,}",
            )
        console.print(table)

        counts = Table(title="table row counts")
        counts.add_column("table")
        counts.add_column("rows", justify="right")
        for name in (
            "raw_contingency",
            "raw_label_version",
            "raw_label_mention",
            "raw_quarterly_reports",
            "raw_reporter_mix",
            "stat_disproportionality",
            "stat_pair_outcome",
            "stat_leakage",
            "stat_diagnostics",
            "model_survival_curve",
            "model_priority_queue",
        ):
            if warehouse.table_exists(name):
                counts.add_row(name, f"{warehouse.row_count(name):,}")
        console.print(counts)


def _latest_run(warehouse: object) -> str:
    run_id = warehouse.latest_complete_run()  # type: ignore[attr-defined]
    if run_id is None:
        error_console.print(
            "[red]No completed run in the warehouse.[/red] Run `prodrome ingest` first."
        )
        raise typer.Exit(1)
    return str(run_id)


def main() -> None:
    """Entry point that turns expected operational failures into clean messages.

    A lock conflict and a missing config are normal conditions, not bugs; printing a
    rich traceback for either trains a reader to ignore tracebacks.
    """
    try:
        app()
    except WarehouseBusyError as exc:
        error_console.print(f"[yellow]warehouse busy:[/yellow] {exc}")
        raise SystemExit(1) from exc
    except FileNotFoundError as exc:
        error_console.print(f"[red]not found:[/red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
