"""DuckDB access layer.

DuckDB rather than PostgreSQL, deliberately. The whole point of this project is
that it costs nothing to run and that a reviewer can reproduce it: DuckDB is a
single file with no server, so ``git clone && make all`` works on a laptop and in
GitHub Actions identically, and the resulting warehouse can be attached directly
from dbt, from pandas, and from the Tableau extract writer. A hosted Postgres
would add a credential, a cost and a failure mode for no analytical gain -- the
working set is a few million rows of aggregates, which is comfortably in DuckDB's
sweet spot and nowhere near needing a cluster.

Writes go through :meth:`Warehouse.append_rows`, which registers an Arrow table
and inserts from it. Row-by-row ``executemany`` on a columnar engine is
pathologically slow; this keeps the whole load vectorised.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import uuid
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Warehouse:
    """A DuckDB connection with prodrome's schema applied."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def apply_schema(self) -> None:
        """Create tables if absent. Idempotent, so it runs on every open."""
        self._connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    def append_rows(self, table: str, rows: Sequence[dict[str, Any]]) -> int:
        """Insert rows, matching on column name rather than position.

        Column-name matching matters: the dataclasses that produce these rows
        evolve, and a positional insert would silently write values into the wrong
        columns the first time a field was added in the middle of a definition.

        Returns the number of rows inserted.
        """
        if not rows:
            return 0
        columns = list(rows[0].keys())
        for index, row in enumerate(rows[1:], start=1):
            if list(row.keys()) != columns:
                raise ValueError(
                    f"row {index} of the batch for {table} has different keys "
                    f"({sorted(row)}) than the first ({sorted(columns)})"
                )
        arrow = pa.Table.from_pylist(list(rows))
        self._connection.register("_incoming", arrow)
        try:
            quoted = ", ".join(f'"{c}"' for c in columns)
            self._connection.execute(
                f'INSERT OR REPLACE INTO "{table}" ({quoted}) SELECT {quoted} FROM _incoming'
            )
        finally:
            self._connection.unregister("_incoming")
        return len(rows)

    def clear_run(self, table: str, run_id: str) -> int:
        """Delete a run's rows from a table before rewriting them.

        Necessary for stage idempotency. ``INSERT OR REPLACE`` updates rows whose
        primary key recurs, but leaves behind any row the new pass no longer produces
        -- so re-running a stage after fixing a bug silently mixes old and new
        results. That is exactly what happened here: tightening the leakage benchmark
        to exclude left-truncated pairs left the excluded rows in place, and the
        published median was computed over both.

        Returns the number of rows deleted.
        """
        if not self.table_exists(table):
            return 0
        before = int(self.scalar(f'SELECT count(*) FROM "{table}" WHERE run_id = ?', [run_id]) or 0)
        if before:
            self._connection.execute(f'DELETE FROM "{table}" WHERE run_id = ?', [run_id])
        return before

    def clear_runs(self, tables: Sequence[str], run_id: str) -> dict[str, int]:
        """Clear a run's rows from several tables, reporting what was removed."""
        return {t: n for t in tables if (n := self.clear_run(t, run_id)) > 0}

    def append_batched(
        self, table: str, rows: Iterable[dict[str, Any]], *, batch_size: int = 20_000
    ) -> int:
        """Insert a stream of rows in batches, so memory stays bounded."""
        batch: list[dict[str, Any]] = []
        total = 0
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                total += self.append_rows(table, batch)
                batch.clear()
        if batch:
            total += self.append_rows(table, batch)
        return total

    def query(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        return self._connection.execute(sql, params or []).fetchall()

    def query_df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Run a query and return a pandas DataFrame."""
        frame: pd.DataFrame = self._connection.execute(sql, params or []).df()
        return frame

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        row = self._connection.execute(sql, params or []).fetchone()
        return row[0] if row else None

    def find_table(self, name: str) -> str | None:
        """Fully-qualified name of a table, searched across schemas.

        dbt materialises each layer into its own schema (``main_staging``,
        ``main_intermediate``, ``main_marts``), which is good practice and makes
        lineage legible -- but it means a mart is not reachable by its bare name from
        a plain connection. Resolving the schema here keeps the layer separation
        without every caller having to know about it, and keeps the code working if
        the schema convention changes.

        Returns None when no schema holds the table.
        """
        rows = self.query(
            "SELECT table_schema FROM information_schema.tables "
            "WHERE table_name = ? ORDER BY table_schema",
            [name],
        )
        if not rows:
            return None
        schemas = [str(r[0]) for r in rows]
        # Prefer the marts schema when a name somehow exists in several, since that
        # is the published layer.
        for preferred in ("main_marts", "main"):
            if preferred in schemas:
                return f'"{preferred}"."{name}"'
        return f'"{schemas[0]}"."{name}"'

    def table_exists(self, name: str) -> bool:
        return self.find_table(name) is not None

    def read_table(self, name: str) -> pd.DataFrame:
        """Read a whole table as a DataFrame, resolving its schema.

        Raises:
            KeyError: when no schema holds the table, which means dbt has not built
                it yet.
        """
        qualified = self.find_table(name)
        if qualified is None:
            raise KeyError(f"{name} does not exist in any schema; run `dbt build` first")
        return self.query_df(f"SELECT * FROM {qualified}")

    def row_count(self, table: str) -> int:
        return int(self.scalar(f'SELECT count(*) FROM "{table}"') or 0)

    # ---- run manifest ----------------------------------------------------

    def start_run(
        self,
        *,
        command: str,
        prodrome_version: str,
        config_digest: str,
        cohort_size: int,
        first_quarter: str,
        last_quarter: str,
        openfda_last_updated: dt.date | None,
        has_openfda_key: bool,
        embed_backend: str | None,
        notes: str | None = None,
    ) -> str:
        """Open a run and return its id.

        Every fact row written afterwards carries this id, which is what makes a
        published figure traceable to the configuration and the API traffic behind
        it. A pipeline that cannot answer "which run produced this number" cannot
        be audited.
        """
        run_id = uuid.uuid4().hex[:16]
        self.append_rows(
            "runs",
            [
                {
                    "run_id": run_id,
                    "started_at": dt.datetime.now(tz=dt.UTC).replace(tzinfo=None),
                    "finished_at": None,
                    "command": command,
                    "prodrome_version": prodrome_version,
                    "config_digest": config_digest,
                    "cohort_size": cohort_size,
                    "first_quarter": first_quarter,
                    "last_quarter": last_quarter,
                    "openfda_last_updated": openfda_last_updated,
                    "has_openfda_key": has_openfda_key,
                    "embed_backend": embed_backend,
                    "requests": 0,
                    "cache_hits": 0,
                    "retries": 0,
                    "notes": notes,
                }
            ],
        )
        return run_id

    def finish_run(
        self, run_id: str, *, requests: int = 0, cache_hits: int = 0, retries: int = 0
    ) -> None:
        self._connection.execute(
            "UPDATE runs SET finished_at = ?, requests = ?, cache_hits = ?, retries = ? "
            "WHERE run_id = ?",
            [
                dt.datetime.now(tz=dt.UTC).replace(tzinfo=None),
                requests,
                cache_hits,
                retries,
                run_id,
            ],
        )

    def latest_complete_run(self) -> str | None:
        """Most recent finished run, which is what the marts and exports read.

        Unfinished runs are excluded so a crashed or in-flight run cannot be
        published as if it were complete.
        """
        found = self.scalar(
            "SELECT run_id FROM runs WHERE finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT 1"
        )
        return str(found) if found is not None else None


class WarehouseBusyError(RuntimeError):
    """Another process holds the warehouse lock.

    DuckDB is single-writer, and a write connection excludes even readers. That is
    the right trade for this project -- the alternative is a database server, which
    would add a credential and a cost to something designed to run for free -- but
    it means a read command issued during a long ingest must explain itself rather
    than surfacing a lock trace.
    """


@contextmanager
def open_warehouse(path: Path | str, *, read_only: bool = False) -> Any:
    """Open the warehouse, applying the schema on write connections.

    Raises:
        WarehouseBusyError: another process holds the lock, which in practice
            means an ingest is running.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = duckdb.connect(str(resolved), read_only=read_only)
    except duckdb.IOException as exc:
        if "lock" in str(exc).lower():
            raise WarehouseBusyError(
                f"{resolved} is locked by another process -- a pipeline run is "
                f"probably in progress. DuckDB allows a single writer; wait for it "
                f"to finish."
            ) from exc
        raise
    try:
        warehouse = Warehouse(connection)
        if not read_only:
            warehouse.apply_schema()
        yield warehouse
    finally:
        connection.close()


def config_digest(payload: Any) -> str:
    """Stable sha256 of a configuration, for the run manifest.

    Sorted keys and a canonical separator so the digest depends on the
    configuration's content and not on how it happened to be serialised.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
