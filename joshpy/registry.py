"""DuckDB-backed registry for tracking parameter sweeps, job configurations, and run results.

This module provides a persistent registry for tracking Josh simulation experiments,
enabling users to:
- Track experiments by name (e.g., experiment_name="jotr_sensitivity")
- Link configs to their MD5 hashes for easy lookup
- Query runs by parameter values
- Get session summaries with success/failure counts

Example usage:
    from joshpy.jobs import JobConfig, JobExpander, run_sweep
    from joshpy.registry import RunRegistry

    # Setup
    config = JobConfig(
        template_path=Path("template.jshc.j2"),
        source_path=Path("simulation.josh"),
        simulation="JoshuaTreeSim",
        sweep=SweepConfig(config_parameters=[...]),
    )
    job_set = JobExpander().expand(config)

    registry = RunRegistry("experiment.duckdb")
    session_id = registry.create_session(
        config=config,
        experiment_name="jotr_sensitivity",
    )

    # Register configs
    for job in job_set.jobs:
        registry.register_run(session_id, job.run_hash, str(job.source_path), job.config_content, None, job.parameters)

    # Run with tracking (status management is automatic)
    cli = JoshCLI()
    results = run_sweep(cli, job_set, registry=registry, session_id=session_id)
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from joshpy.schema import CELL_DATA_CURRENT_VIEW_SQL, SCHEMA_SQL

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


class RegistryBusyError(RuntimeError):
    """Raised when the DuckDB registry file is locked by another process.

    DuckDB is single-writer: a second process opening the same registry file
    while another holds the write lock would otherwise surface a raw, low-level
    ``duckdb.IOException``. This typed error lets consumers catch one named
    exception instead of pattern-matching DuckDB's message string.
    """


def _check_duckdb() -> None:
    """Raise ImportError if duckdb is not available."""
    if not HAS_DUCKDB:
        raise ImportError(
            "duckdb is required for the registry module. Install with: pip install joshpy[registry]"
        )


def configure_s3(
    conn: Any,
    endpoint: str,
    access_key: str,
    secret_key: str,
    url_style: str = "path",
    use_ssl: bool | None = None,
) -> None:
    """Configure a DuckDB connection for S3/MinIO access via httpfs.

    Installs and loads the httpfs extension, then creates an S3 secret
    so ``read_csv_auto('s3://bucket/key.csv')`` works transparently.

    Credential resolution is the caller's responsibility -- this function
    takes explicit values.  ``ingest_results()`` resolves credentials from
    environment variables (``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``,
    ``MINIO_SECRET_KEY``) before calling here.

    DuckDB's httpfs expects ``ENDPOINT`` to be a bare hostname without a
    scheme (e.g. ``"storage.googleapis.com"``).  This function accepts
    either a bare hostname OR a full URL (``"https://storage.googleapis.com"``)
    so callers can forward the same ``MINIO_ENDPOINT`` env var they pass to
    the Josh JAR without needing to strip the scheme themselves.  When a
    scheme is present ``use_ssl`` is inferred from it unless explicitly set.

    Args:
        conn: DuckDB connection object.
        endpoint: S3-compatible endpoint.  Accepts bare hostname
            (``"storage.googleapis.com"``) or full URL
            (``"https://storage.googleapis.com"``).
        access_key: Access key / key ID.
        secret_key: Secret key.
        url_style: ``"path"`` (default, MinIO) or ``"vhost"`` (AWS).
        use_ssl: Use HTTPS.  If None (default), inferred from endpoint
            scheme when present, otherwise True.
    """
    host = endpoint
    inferred_ssl: bool | None = None
    if host.startswith("https://"):
        host = host[len("https://"):]
        inferred_ssl = True
    elif host.startswith("http://"):
        host = host[len("http://"):]
        inferred_ssl = False
    # Strip any trailing path / slash -- DuckDB expects just host[:port]
    host = host.rstrip("/").split("/", 1)[0]

    resolved_ssl = use_ssl if use_ssl is not None else (
        inferred_ssl if inferred_ssl is not None else True
    )

    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(
        """
        CREATE OR REPLACE SECRET (
            TYPE s3,
            KEY_ID ?,
            SECRET ?,
            ENDPOINT ?,
            URL_STYLE ?,
            USE_SSL ?
        )
        """,
        [access_key, secret_key, host, url_style, resolved_ssl],
    )


def _resolve_s3_credentials(
    endpoint: str | None,
    access_key: str | None,
    secret_key: str | None,
) -> tuple[str, str, str]:
    """Resolve S3/MinIO credentials from explicit args, falling back to env vars.

    Mirrors the resolution ``ingest_results()`` does internally
    (``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``, ``MINIO_SECRET_KEY``), shared
    here for :func:`RunRegistry.push_to_s3` / :func:`RunRegistry.pull_from_s3`
    / :func:`open_s3_registries`.

    Raises:
        RuntimeError: If any credential is missing after falling back to env vars.
    """
    import os

    endpoint = endpoint or os.environ.get("MINIO_ENDPOINT", "")
    access_key = access_key or os.environ.get("MINIO_ACCESS_KEY", "")
    secret_key = secret_key or os.environ.get("MINIO_SECRET_KEY", "")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError(
            "S3 credentials required: pass endpoint/access_key/secret_key, or "
            "set MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY."
        )
    return endpoint, access_key, secret_key


def _default_s3_prefix(db_path: Path | str) -> str:
    """Derive a default S3 prefix from a registry's own file name.

    Mirrors the local convention (one registry file per experiment, named
    after it) so the S3 layout needs no separate naming scheme to learn.

    Raises:
        ValueError: If db_path is ``":memory:"`` (no filename to derive from).
    """
    db_str = str(db_path)
    if db_str == ":memory:":
        raise ValueError(
            "An in-memory registry has no filename to derive an S3 prefix "
            "from -- pass prefix= explicitly."
        )
    return f"run-registries/{Path(db_str).stem}"


# Registry tables synced by push_to_s3()/pull_from_s3(), in parent-first
# (foreign-key-safe) order. run_tags has no FK and sorts anywhere, but is kept
# last for clarity since it's metadata, not run/result data.
REGISTRY_SYNC_TABLES: tuple[str, ...] = (
    "sweep_sessions",
    "job_configs",
    "session_configs",
    "config_parameters",
    "job_runs",
    "run_outputs",
    "cell_data",
    "run_tags",
)


@contextmanager
def open_s3_registries(
    names: list[str],
    *,
    bucket: str,
    prefix_template: str = "run-registries/{name}",
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    use_ssl: bool | None = None,
) -> Iterator[Any]:
    """Open multiple S3-published registries for read-only cross-registry queries.

    The recommended way to analyze several experiments together: each
    registry pushed via :meth:`RunRegistry.push_to_s3` is exposed as a schema
    of views (``<name>.<table>``) reading directly from its Parquet export --
    nothing is copied or merged locally. JOIN/UNION across the views instead
    of pulling multiple experiments' data into one registry, which risks the
    hash-collision ambiguity a reused registry already has (see
    "One registry per experiment" in the best-practices guide) -- merging
    genuinely different experiments' rows together has the same problem.

    Args:
        names: Registry names to open (as pushed -- i.e. the ``name`` each
            was published under via ``push_to_s3``).
        bucket: S3/MinIO bucket containing the published registries.
        prefix_template: Format string mapping a name to its S3 prefix.
            Default matches ``push_to_s3``'s own default layout.
        endpoint: S3 endpoint. Falls back to ``MINIO_ENDPOINT`` env var.
        access_key: S3 access key. Falls back to ``MINIO_ACCESS_KEY``.
        secret_key: S3 secret key. Falls back to ``MINIO_SECRET_KEY``.
        use_ssl: Use HTTPS. See :func:`configure_s3`.

    Yields:
        An in-memory DuckDB connection with each registry attached as a
        same-named schema of views.

    Examples:
        >>> with open_s3_registries(
        ...     ["jotr_baseline", "jotr_dry"], bucket="josh-batch-storage"
        ... ) as conn:
        ...     df = conn.execute('''
        ...         SELECT 'baseline' AS experiment, step, AVG(averageHeight)
        ...         FROM jotr_baseline.cell_data GROUP BY step
        ...         UNION ALL
        ...         SELECT 'dry' AS experiment, step, AVG(averageHeight)
        ...         FROM jotr_dry.cell_data GROUP BY step
        ...     ''').df()
    """
    _check_duckdb()
    endpoint, access_key, secret_key = _resolve_s3_credentials(endpoint, access_key, secret_key)
    conn = duckdb.connect(":memory:")
    configure_s3(conn, endpoint, access_key, secret_key, use_ssl=use_ssl)
    try:
        for name in names:
            prefix = prefix_template.format(name=name).rstrip("/")
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{name}"')
            for table in REGISTRY_SYNC_TABLES:
                remote = f"s3://{bucket}/{prefix}/{table}.parquet"
                try:
                    conn.execute(
                        f'CREATE VIEW "{name}".{table} AS '
                        f"SELECT * FROM read_parquet('{remote}')"
                    )
                except Exception:
                    # Table wasn't published (e.g. older export predating a
                    # schema addition, or genuinely empty/never pushed) --
                    # skip it, same as ProjectCatalog.open_registries() skips
                    # missing local registry files.
                    continue
        yield conn
    finally:
        conn.close()


def _get_git_hash() -> str | None:
    """Get current git HEAD hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        head = result.stdout.strip()
        if not head or result.returncode != 0:
            return None
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = dirty_result.stdout.strip()
        return f"{head[:12]}+dirty" if dirty else head[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# Sparsity warning thresholds (configurable module globals)
SPARSITY_WARN_COLUMN_NULL_PERCENT = 50  # Warn if column >50% NULL
SPARSITY_WARN_MIN_SPARSE_COLUMNS = 2  # Only warn if >=2 columns are sparse
SPARSITY_WARN_MIN_ROWS = 1000  # Don't warn for tiny datasets

# Core columns that exist in cell_data schema (not variable columns)
CELL_DATA_CORE_COLUMNS = frozenset(
    {
        "cell_id",
        "run_id",
        "run_hash",
        "step",
        "replicate",
        "position_x",
        "position_y",
        "longitude",
        "latitude",
        "entity_type",
        "geom",
    }
)


def _quote_identifier(name: str) -> str:
    """Quote an identifier for use in SQL.

    Uses double quotes to preserve original names with special characters
    like dots (e.g., "avg.height").

    Args:
        name: Original variable name.

    Returns:
        Quoted identifier safe for SQL (e.g., '"avg.height"').
    """
    # Escape any double quotes in the name by doubling them
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _infer_type(value: Any) -> str:
    """Infer SQL type from a single value.

    Used for both export variables (cell_data) and config parameters
    (config_parameters) to determine appropriate column types.

    Args:
        value: A single value to infer type from.

    Returns:
        'DOUBLE' if value is numeric (int, float, or numeric string),
        'VARCHAR' otherwise.
    """
    if value is None:
        return "VARCHAR"  # Can't infer from None, default to VARCHAR
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "DOUBLE"
    if isinstance(value, str):
        if value == "":
            return "VARCHAR"
        try:
            float(value)
            return "DOUBLE"
        except (ValueError, TypeError):
            return "VARCHAR"
    return "VARCHAR"


# Core columns in config_parameters table (not parameter columns)
CONFIG_PARAMS_CORE_COLUMNS = frozenset({"run_hash"})


@dataclass
class SessionInfo:
    """Information about a sweep session.

    Attributes:
        session_id: Unique session identifier.
        experiment_name: Name of the experiment for path templates.
        created_at: When the session was created.
        template_path: Path to the template file (if any).
        template_hash: Hash of the template content.
        simulation: Name of the simulation.
        total_jobs: Total number of job configurations.
        total_replicates: Total number of replicates across all jobs.
        status: Session status (pending, running, completed, failed).
        metadata: Additional metadata as a dictionary.
    """

    session_id: str
    experiment_name: str | None
    created_at: datetime
    template_path: str | None
    template_hash: str | None
    simulation: str | None
    total_jobs: int | None
    total_replicates: int | None
    status: str
    metadata: dict[str, Any] | None

    @property
    def job_config(self) -> Any:
        """Extract JobConfig from metadata if stored.

        Returns the JobConfig used to create this session, enabling
        session reconstruction patterns (re-expand jobs from stored config).

        Returns:
            JobConfig if metadata contains 'job_config' key, None otherwise.

        Examples:
            >>> session = registry.get_session(session_id)
            >>> if session.job_config:
            ...     # Re-expand jobs for execution
            ...     job_set = JobExpander().expand(session.job_config)
        """
        if self.metadata and "job_config" in self.metadata:
            from joshpy.jobs import JobConfig

            return JobConfig.from_dict(self.metadata["job_config"])
        return None


@dataclass
class ConfigInfo:
    """Information about a job configuration.

    Attributes:
        run_hash: MD5 hash of josh + config + file_mappings (12 chars).
        session_id: Session this config belongs to.
        josh_path: Path to the .josh script file.
        josh_content: Rendered .josh source content (may be None for legacy data).
        config_content: Full text content of the configuration.
        file_mappings: Dict mapping names to {"path": "...", "hash": "..."}.
        parameters: Parameter values used to generate this config.
        label: Optional human-readable label for this run.
        created_at: When the config was registered.
        status: Lifecycle status, one of ``{"active", "superseded", "bad"}``.
            ``None`` is stored for runs that have never been marked and is read
            as ``"active"``; prefer :attr:`effective_status` to normalize it.
        superseded_by: The run_hash that replaced this run, set when
            ``status == "superseded"``. ``None`` otherwise.
        status_reason: Free-text explanation recorded when the status was set.
        status_updated_at: When the status was last changed.
    """

    run_hash: str
    session_id: str
    josh_path: str | None
    josh_content: str | None
    config_content: str
    file_mappings: dict[str, dict[str, str]] | None
    parameters: dict[str, Any]
    label: str | None
    created_at: datetime
    status: str | None = None
    superseded_by: str | None = None
    status_reason: str | None = None
    status_updated_at: datetime | None = None

    @property
    def effective_status(self) -> str:
        """Status normalized so ``None`` reads as ``"active"``."""
        return self.status or "active"

    @property
    def is_current(self) -> bool:
        """Whether this run is current (i.e. :attr:`effective_status` is active)."""
        return self.effective_status == "active"


@dataclass
class RunStatus:
    """Lifecycle status of a single run.

    "Current" is not a stored flag — it is exactly ``status == "active"``. A run
    that has never been marked has ``status is None``, which is read as
    ``"active"`` (see :attr:`effective_status`). See REGISTRY_PROVENANCE.md for
    the full model.

    Attributes:
        run_hash: The run this status describes.
        status: One of ``{"active", "superseded", "bad"}`` or ``None`` (unmarked,
            treated as active).
        superseded_by: The run_hash that replaced this run, set when
            ``status == "superseded"``. ``None`` otherwise.
        reason: Free-text explanation recorded when the status was set.
        updated_at: When the status was last changed.
    """

    run_hash: str
    status: str | None
    superseded_by: str | None = None
    reason: str | None = None
    updated_at: datetime | None = None

    @property
    def effective_status(self) -> str:
        """Status normalized so ``None`` reads as ``"active"``."""
        return self.status or "active"

    @property
    def is_current(self) -> bool:
        """Whether the run is current (i.e. :attr:`effective_status` is active)."""
        return self.effective_status == "active"


@dataclass
class ConfigSourceInfo:
    """Result of resolving a config's original source file on disk.

    Attributes:
        path: Original file path, or None if not recoverable
            (e.g., templated configs where no config_path was set).
        exists: Whether the file currently exists on disk.
        content_matches: Whether the file's current content matches
            the config_content stored in the registry.
    """

    path: Path | None
    exists: bool
    content_matches: bool


@dataclass
class RunInfo:
    """Information about a job run.

    Attributes:
        run_id: Unique run identifier.
        run_hash: Run hash for this run.
        replicate: Replicate number (0-indexed).
        started_at: When the run started.
        completed_at: When the run completed.
        exit_code: Process exit code (0 = success).
        output_path: Path to output files.
        error_message: Error message if run failed.
        metadata: Additional metadata.
    """

    run_id: str
    run_hash: str
    replicate: int
    started_at: datetime | None
    completed_at: datetime | None
    exit_code: int | None
    output_path: str | None
    error_message: str | None
    metadata: dict[str, Any] | None


@dataclass
class RunDetail:
    """Aggregated detail for a single run.

    Combines a run's stored configuration, its recorded executions, and its
    replicate count into one structured object. This is the data-layer
    counterpart to :meth:`RunRegistry.describe_run` (which renders the same
    information as a human-readable string); use :meth:`RunRegistry.get_run_info`
    to obtain one.

    Attributes:
        config: The run's stored configuration (parameters, file mappings,
            josh path, label, ...).
        runs: All recorded executions for this run (typically one per replicate
            attempt). May be empty if no runs have been recorded yet.
        replicate_count: Distinct replicate count from ``cell_data`` -- the
            source of truth for how many replicates are loaded.
        replicates: The set of replicate indices present in ``cell_data``. The
            index is a collision-avoidance tag, so this set may be sparse (e.g.
            ``{1, 4, 5}``); only its size is meaningful.
    """

    config: ConfigInfo
    runs: list[RunInfo]
    replicate_count: int
    replicates: set[int] = field(default_factory=set)

    @property
    def run_hash(self) -> str:
        """The run's hash (delegates to :attr:`config`)."""
        return self.config.run_hash

    @property
    def label(self) -> str | None:
        """The run's label, if any (delegates to :attr:`config`)."""
        return self.config.label

    @property
    def parameters(self) -> dict[str, Any]:
        """The run's parameter values (delegates to :attr:`config`)."""
        return self.config.parameters

    @property
    def succeeded(self) -> int:
        """Number of recorded runs that exited successfully (exit code 0)."""
        return sum(1 for r in self.runs if r.exit_code == 0)

    @property
    def failed(self) -> int:
        """Number of recorded runs that exited with a non-zero code."""
        return sum(1 for r in self.runs if r.exit_code is not None and r.exit_code != 0)

    @property
    def pending(self) -> int:
        """Number of recorded runs with no exit code yet (still running)."""
        return sum(1 for r in self.runs if r.exit_code is None)


@dataclass
class DropSummary:
    """Rows removed by :meth:`RunRegistry.drop_run`, per table.

    Attributes:
        run_hash: The run hash that was dropped.
        label: The label that was dropped (if the run was labeled).
        rows: Mapping of table name -> number of rows deleted.
    """

    run_hash: str
    label: str | None
    rows: dict[str, int]

    @property
    def total(self) -> int:
        """Total rows deleted across all tables."""
        return sum(self.rows.values())


@dataclass
class ConsistencyIssue:
    """A run<->analysis consistency problem found by :meth:`RunRegistry.check_consistency`.

    Attributes:
        kind: Machine-readable issue type (e.g. ``"duplicate_replicate"``,
            ``"data_without_config"``, ``"orphan_cell_data"``, ``"ran_not_ingested"``).
        run_hash: The run hash the issue concerns (if applicable).
        detail: Human-readable description.
        severity: ``"error"`` (data integrity) or ``"warning"`` (informational,
            e.g. ran-but-not-ingested).
    """

    kind: str
    run_hash: str | None
    detail: str
    severity: str = "warning"


@dataclass
class ColumnStats:
    """Statistics for a single column in cell_data.

    Attributes:
        name: Column name (sanitized).
        dtype: SQL data type (DOUBLE or VARCHAR).
        total_rows: Total number of rows in the table.
        null_count: Number of NULL values in this column.
    """

    name: str
    dtype: str
    total_rows: int
    null_count: int

    @property
    def null_percent(self) -> float:
        """Percentage of NULL values (0-100)."""
        if self.total_rows == 0:
            return 0.0
        return (self.null_count / self.total_rows) * 100


@dataclass
class SparsityReport:
    """Report on column sparsity in cell_data.

    Used to detect when different simulation types are being mixed,
    which creates sparse columns that hurt query performance.

    Attributes:
        total_rows: Total rows in cell_data.
        column_stats: List of ColumnStats for each variable column.
        threshold_percent: NULL percentage threshold used for warnings.
    """

    total_rows: int
    column_stats: list[ColumnStats]
    threshold_percent: float = SPARSITY_WARN_COLUMN_NULL_PERCENT

    @property
    def sparse_columns(self) -> list[ColumnStats]:
        """Columns exceeding the sparsity threshold."""
        return [c for c in self.column_stats if c.null_percent > self.threshold_percent]

    @property
    def should_warn(self) -> bool:
        """True if enough sparse columns to warrant a warning."""
        return (
            self.total_rows >= SPARSITY_WARN_MIN_ROWS
            and len(self.sparse_columns) >= SPARSITY_WARN_MIN_SPARSE_COLUMNS
        )

    def __str__(self) -> str:
        """Human-readable sparsity report."""
        if not self.sparse_columns:
            return f"No sparse columns (threshold: {self.threshold_percent}% NULL)"

        lines = [
            f"SparsityWarning: {len(self.sparse_columns)} columns are "
            f">{self.threshold_percent}% NULL:"
        ]
        for col in self.sparse_columns:
            lines.append(
                f"  - {col.name}: {col.null_percent:.1f}% NULL "
                f"({col.null_count:,}/{col.total_rows:,})"
            )
        lines.append("")
        lines.append(
            "Consider removing unused columns or using a separate registry "
            "for different simulations."
        )
        return "\n".join(lines)


@dataclass
class SessionSummary:
    """Aggregated statistics for a session.

    Attributes:
        session_id: Session identifier.
        experiment_name: Name of the experiment.
        simulation: Simulation name.
        status: Session status.
        total_jobs: Total job configurations.
        total_replicates: Total expected replicates.
        runs_completed: Number of completed runs.
        runs_succeeded: Number of successful runs.
        runs_failed: Number of failed runs.
        runs_pending: Number of pending runs.
    """

    session_id: str
    experiment_name: str | None
    simulation: str | None
    status: str
    total_jobs: int
    total_replicates: int
    runs_completed: int
    runs_succeeded: int
    runs_failed: int
    runs_pending: int


@dataclass
class DataSummary:
    """Summary of data loaded in the registry.

    Attributes:
        sessions: Number of sweep sessions.
        configs: Number of job configurations.
        runs: Number of job runs.
        cell_data_rows: Number of rows in cell_data table.
        variables: List of variable names found in cell_data.
        entity_types: List of entity types found in cell_data.
        step_range: (min, max) step values, or None if no data.
        replicate_range: (min, max) replicate values, or None if no data.
        spatial_extent: Dict with 'lon' and 'lat' tuples, or None if no spatial data.
        parameters: List of parameter names found in job_configs.
    """

    sessions: int
    configs: int
    runs: int
    cell_data_rows: int
    variables: list[str]
    entity_types: list[str]
    step_range: tuple[int, int] | None
    replicate_range: tuple[int, int] | None
    spatial_extent: dict[str, tuple[float, float]] | None
    parameters: list[str]

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "Registry Data Summary",
            "=" * 40,
            f"Sessions: {self.sessions}",
            f"Configs:  {self.configs}",
            f"Runs:     {self.runs}",
            f"Rows:     {self.cell_data_rows:,}",
            "",
            f"Variables: {', '.join(self.variables) if self.variables else '(none)'}",
            f"Entity types: {', '.join(self.entity_types) if self.entity_types else '(none)'}",
            f"Parameters: {', '.join(self.parameters) if self.parameters else '(none)'}",
        ]
        if self.step_range:
            lines.append(f"Steps: {self.step_range[0]} - {self.step_range[1]}")
        if self.replicate_range:
            lines.append(f"Replicates: {self.replicate_range[0]} - {self.replicate_range[1]}")
        if self.spatial_extent:
            lon = self.spatial_extent.get("lon")
            lat = self.spatial_extent.get("lat")
            if lon and lat:
                lines.append(
                    f"Spatial extent: lon [{lon[0]:.2f}, {lon[1]:.2f}], "
                    f"lat [{lat[0]:.2f}, {lat[1]:.2f}]"
                )
        return "\n".join(lines)


def _generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


@dataclass
class RunRegistry:
    """DuckDB-backed registry for tracking parameter sweeps and job runs.

    Supports both file-based persistence and in-memory mode (using ":memory:").

    Attributes:
        db_path: Path to the DuckDB database file, or ":memory:" for in-memory.
        enable_spatial: If True, load spatial extension and create geometry column.
        _conn: DuckDB connection (created automatically).

    Examples:
        >>> # File-based (persistent)
        >>> registry = RunRegistry("experiment.duckdb")

        >>> # In-memory (for testing)
        >>> registry = RunRegistry(":memory:")

        >>> # With spatial support enabled (default)
        >>> registry = RunRegistry("experiment.duckdb", enable_spatial=True)

        >>> # Context manager
        >>> with RunRegistry("experiment.duckdb") as registry:
        ...     session_id = registry.create_session(...)
    """

    db_path: Path | str = "josh_runs.duckdb"
    enable_spatial: bool = True
    _conn: Any = field(default=None, repr=False)

    # Filter state for context managers
    _spatial_filter_bbox: tuple[float, float, float, float] | None = field(default=None, repr=False)
    _spatial_filter_geojson: str | dict | None = field(default=None, repr=False)
    _time_filter_range: tuple[int, int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize database connection and schema."""
        _check_duckdb()
        if self._conn is None:
            db_str = str(self.db_path)
            try:
                self._conn = duckdb.connect(db_str)
            except duckdb.IOException as e:
                msg = str(e)
                if "lock" in msg.lower():
                    raise RegistryBusyError(
                        f"Registry '{db_str}' is locked by another process — wait "
                        f"for it to finish, or use a different registry path. "
                        f"(DuckDB: {msg.splitlines()[0]})"
                    ) from e
                raise
            self._init_schema()
            self._migrate_schema()
            if self.enable_spatial:
                self._init_spatial()
            # Create the current view last: it does SELECT c.* over cell_data, so
            # it must be defined after _init_spatial() has (optionally) added the
            # geom column, or its captured column set would drift from the table.
            self._conn.execute(CELL_DATA_CURRENT_VIEW_SQL)

    @property
    def conn(self) -> Any:
        """Direct access to DuckDB connection for custom queries.

        Returns:
            DuckDB connection object.

        Examples:
            >>> df = registry.conn.execute("SELECT * FROM cell_data LIMIT 10").df()
        """
        return self._conn

    def _init_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        self._conn.execute(SCHEMA_SQL)

    def _migrate_schema(self) -> None:
        """Apply schema migrations for forward compatibility with older databases."""
        # Add label column to job_configs if missing (added in v0.X)
        try:
            self._conn.execute("SELECT label FROM job_configs LIMIT 0")
        except Exception:
            self._conn.execute("ALTER TABLE job_configs ADD COLUMN label VARCHAR")

        # Run lifecycle status columns (REGISTRY_PROVENANCE.md). Additive and
        # nullable; NULL status is read as 'active' everywhere via coalesce, so
        # existing rows keep their behavior. Probe one column and add the whole
        # set if it is missing.
        try:
            self._conn.execute("SELECT status FROM job_configs LIMIT 0")
        except Exception:
            self._conn.execute("ALTER TABLE job_configs ADD COLUMN status VARCHAR")
            self._conn.execute(
                "ALTER TABLE job_configs ADD COLUMN superseded_by VARCHAR"
            )
            self._conn.execute("ALTER TABLE job_configs ADD COLUMN status_reason TEXT")
            self._conn.execute(
                "ALTER TABLE job_configs ADD COLUMN status_updated_at TIMESTAMP"
            )

    def _init_spatial(self) -> None:
        """Initialize DuckDB spatial extension and geometry column."""
        try:
            self._conn.execute("INSTALL spatial; LOAD spatial;")
            # Add geometry column if it doesn't exist
            # Check if column exists first
            columns = self._conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'cell_data' AND column_name = 'geom'"
            ).fetchall()
            if not columns:
                self._conn.execute("ALTER TABLE cell_data ADD COLUMN geom GEOMETRY;")
        except Exception:
            # Spatial extension may not be available in all DuckDB builds
            # Silently continue without spatial support
            pass

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> RunRegistry:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close connection."""
        self.close()

    # ========== DuckDB Direct Access ==========

    def query(self, sql: str, params: list | None = None) -> Any:
        """Execute a SQL query with parameters.

        This provides direct access to DuckDB for custom queries beyond
        the pre-built methods. Use this when you need to run complex
        queries or explore the data in ways not covered by the API.

        Args:
            sql: SQL query with ? placeholders for parameters.
            params: List of parameter values.

        Returns:
            DuckDB relation (call .df() for DataFrame, .fetchall() for tuples).

        Examples:
            >>> # Get DataFrame
            >>> df = registry.query(
            ...     "SELECT * FROM cell_data WHERE step BETWEEN ? AND ?",
            ...     [0, 10]
            ... ).df()

            >>> # Get raw results
            >>> rows = registry.query(
            ...     "SELECT COUNT(*) FROM cell_data WHERE run_hash = ?",
            ...     ["abc123"]
            ... ).fetchone()
        """
        return self._conn.execute(sql, params or [])

    def to_parquet(self, path: str | Path, table: str = "cell_data") -> None:
        """Export a table to Parquet format.

        Parquet is recommended for R/Python analysis due to compression
        and type preservation.

        Args:
            path: Output file path.
            table: Table name to export (default: cell_data).

        Examples:
            >>> registry.to_parquet("results.parquet")

            In R:

            >>> # df <- arrow::read_parquet("results.parquet")

            In Python:

            >>> import pandas as pd
            >>> df = pd.read_parquet("results.parquet")
        """
        path_str = str(path)
        self._conn.execute(f"COPY {table} TO '{path_str}' (FORMAT PARQUET)")

    def to_csv(self, path: str | Path, table: str = "cell_data") -> None:
        """Export a table to CSV format.

        Args:
            path: Output file path.
            table: Table name to export (default: cell_data).

        Examples:
            >>> registry.to_csv("results.csv")

            In R:

            >>> # df <- readr::read_csv("results.csv")
        """
        path_str = str(path)
        self._conn.execute(f"COPY {table} TO '{path_str}' (FORMAT CSV, HEADER)")

    # ========== S3 Sync ==========

    def _describe_remote_columns(self, remote: str) -> dict[str, str]:
        """Column name -> DuckDB SQL type for a published table's Parquet file.

        Dict order matches the Parquet file's own column order.
        """
        rows = self.conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{remote}')").fetchall()
        return {row[0]: row[1] for row in rows}

    def _sync_table_columns(self, table: str, remote_cols: dict[str, str]) -> None:
        """Add any columns present in *remote_cols* but missing from *table* locally.

        ``config_parameters`` and ``cell_data`` have dynamically-added
        columns (sweep parameters, export variables respectively) that only
        exist once something has locally registered/ingested that specific
        column. A destination registry that's never seen a given parameter
        or variable needs it added before ``INSERT``/merge can reference it.
        """
        local_cols = {
            row[0]
            for row in self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                [table],
            ).fetchall()
        }
        for col_name, col_type in remote_cols.items():
            if col_name not in local_cols:
                quoted = _quote_identifier(col_name)
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {quoted} {col_type}")

    def push_to_s3(
        self,
        *,
        bucket: str,
        prefix: str | None = None,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        use_ssl: bool | None = None,
        tables: tuple[str, ...] | None = None,
    ) -> str:
        """Publish this registry's tables to S3 as Parquet, one file per table.

        Overwrites whatever is currently at the destination prefix with this
        registry's current state. This publishes *this* registry only -- it
        does not read or touch any other registry. Pair with
        :meth:`pull_from_s3` (same registry, restoring or adding replicates)
        or :func:`open_s3_registries` (querying several published registries
        together, read-only, without merging them).

        Args:
            bucket: S3/MinIO bucket to publish under.
            prefix: S3 key prefix. Defaults to ``run-registries/<name>``
                where ``<name>`` is this registry's own filename stem --
                matching the "one registry file per experiment, named after
                it" convention already used locally. Required if this
                registry is ``":memory:"``.
            endpoint: S3 endpoint. Falls back to ``MINIO_ENDPOINT`` env var.
            access_key: S3 access key. Falls back to ``MINIO_ACCESS_KEY``.
            secret_key: S3 secret key. Falls back to ``MINIO_SECRET_KEY``.
            use_ssl: Use HTTPS. See :func:`configure_s3`.
            tables: Tables to publish. Defaults to all of
                :data:`REGISTRY_SYNC_TABLES`.

        Returns:
            The ``s3://bucket/prefix/`` URI published to.

        Examples:
            >>> registry.push_to_s3(bucket="josh-batch-storage")
            's3://josh-batch-storage/run-registries/jotr_sensitivity/'
        """
        # Resolve the (purely local) prefix before touching credentials/network,
        # so a ":memory:" registry without an explicit prefix fails fast with
        # the right error instead of first demanding S3 credentials it'll
        # never use.
        resolved_prefix = (prefix or _default_s3_prefix(self.db_path)).strip("/")
        endpoint, access_key, secret_key = _resolve_s3_credentials(endpoint, access_key, secret_key)
        configure_s3(self.conn, endpoint, access_key, secret_key, use_ssl=use_ssl)

        for table in tables or REGISTRY_SYNC_TABLES:
            remote = f"s3://{bucket}/{resolved_prefix}/{table}.parquet"
            self.conn.execute(f"COPY (SELECT * FROM {table}) TO '{remote}' (FORMAT PARQUET)")

        return f"s3://{bucket}/{resolved_prefix}/"

    def pull_from_s3(
        self,
        *,
        bucket: str,
        prefix: str | None = None,
        mode: str = "restore",
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        use_ssl: bool | None = None,
        tables: tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        """Pull a registry's Parquet export from S3 back into this registry.

        Args:
            bucket: S3/MinIO bucket the registry was published to.
            prefix: S3 key prefix. Defaults to ``run-registries/<name>``,
                matching :meth:`push_to_s3`'s default. Required if this
                registry is ``":memory:"``.
            mode: ``"restore"`` (default) or ``"merge"``:

                - ``"restore"``: this registry's tables are emptied and
                  replaced with exactly what's at *prefix*. Use this to
                  rehydrate a local registry from its own prior
                  ``push_to_s3`` export -- e.g. on a new machine, or after
                  deleting the local file. The table schemas (constraints,
                  the ``cell_id`` sequence) are preserved; only the rows
                  change.
                - ``"merge"``: adds rows from *prefix* that this registry
                  doesn't already have (by primary key, or for ``cell_data``
                  by ``(run_hash, replicate)`` since ``cell_id`` is a local
                  surrogate with no meaning across registries); existing
                  local rows are never touched. **Only use this to bring in
                  more runs
                  of the *same* experiment** (e.g. a teammate's machine or a
                  batch worker publishing to the same prefix) -- not to
                  combine genuinely different experiments into one registry.
                  Per the one-registry-per-experiment guidance, merging
                  unrelated experiments risks the same hash-collision
                  ambiguity a reused registry already warns about. To
                  analyze multiple experiments together without merging
                  their data, use :func:`open_s3_registries` instead.
            endpoint: S3 endpoint. Falls back to ``MINIO_ENDPOINT`` env var.
            access_key: S3 access key. Falls back to ``MINIO_ACCESS_KEY``.
            secret_key: S3 secret key. Falls back to ``MINIO_SECRET_KEY``.
            use_ssl: Use HTTPS. See :func:`configure_s3`.
            tables: Tables to pull. Defaults to all of
                :data:`REGISTRY_SYNC_TABLES`.

        Returns:
            Dict mapping table name -> number of rows loaded from S3 (for
            ``"merge"``, only the newly-inserted rows; for ``"restore"``,
            the full row count now in the table).

        Raises:
            ValueError: If mode is not ``"restore"`` or ``"merge"``.
        """
        if mode not in ("restore", "merge"):
            raise ValueError(f"mode must be 'restore' or 'merge', got {mode!r}")

        # Same reasoning as push_to_s3: resolve the local prefix before
        # touching credentials/network.
        resolved_prefix = (prefix or _default_s3_prefix(self.db_path)).strip("/")
        endpoint, access_key, secret_key = _resolve_s3_credentials(endpoint, access_key, secret_key)
        configure_s3(self.conn, endpoint, access_key, secret_key, use_ssl=use_ssl)
        sync_tables = tables or REGISTRY_SYNC_TABLES

        if mode == "restore":
            # Child-first delete (FK order), same reasoning as drop_run().
            for table in reversed(sync_tables):
                self.conn.execute(f"DELETE FROM {table}")

        rows: dict[str, int] = {}
        for table in sync_tables:
            remote = f"s3://{bucket}/{resolved_prefix}/{table}.parquet"
            # config_parameters and cell_data have dynamically-added columns
            # (sweep parameters, export variables) that only exist locally
            # once something has registered/ingested that specific column --
            # a destination registry that's never seen a given parameter/
            # variable needs it added before the two sides' columns line up.
            remote_cols = self._describe_remote_columns(remote)
            self._sync_table_columns(table, remote_cols)

            if mode == "restore":
                col_list = ", ".join(_quote_identifier(c) for c in remote_cols)
                self.conn.execute(
                    f"INSERT INTO {table} ({col_list}) "
                    f"SELECT {col_list} FROM read_parquet('{remote}')"
                )
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            elif table == "cell_data":
                # cell_id is a per-registry surrogate (auto-incremented
                # locally) with no cross-registry meaning -- deduping on it
                # would either drop real rows or duplicate them depending on
                # accidental ID overlap between registries. Dedup instead on
                # (run_hash, replicate), the same identity loaded_replicates()
                # already uses everywhere else in this codebase, and let
                # cell_id be freshly assigned by the local sequence.
                cols = [
                    row[0]
                    for row in self.conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'cell_data' AND column_name != 'cell_id' "
                        "ORDER BY ordinal_position"
                    ).fetchall()
                ]
                quoted_cols = ", ".join(_quote_identifier(c) for c in cols)
                select_clause = ", ".join(
                    _quote_identifier(c) if c in remote_cols else f"NULL AS {_quote_identifier(c)}"
                    for c in cols
                )
                before = self.conn.execute("SELECT COUNT(*) FROM cell_data").fetchone()[0]
                self.conn.execute(
                    f"""
                    INSERT INTO cell_data ({quoted_cols})
                    SELECT {select_clause} FROM read_parquet('{remote}') AS src
                    WHERE NOT EXISTS (
                        SELECT 1 FROM cell_data cd
                        WHERE cd.run_hash = src.run_hash AND cd.replicate = src.replicate
                    )
                    """
                )
                after = self.conn.execute("SELECT COUNT(*) FROM cell_data").fetchone()[0]
                count = (after - before,)
            else:
                cols = [
                    row[0]
                    for row in self.conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = ? ORDER BY ordinal_position",
                        [table],
                    ).fetchall()
                ]
                quoted_cols = ", ".join(_quote_identifier(c) for c in cols)
                select_clause = ", ".join(
                    _quote_identifier(c) if c in remote_cols else f"NULL AS {_quote_identifier(c)}"
                    for c in cols
                )
                before = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                self.conn.execute(
                    f"INSERT OR IGNORE INTO {table} ({quoted_cols}) "
                    f"SELECT {select_clause} FROM read_parquet('{remote}')"
                )
                after = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                count = (after - before,)
            rows[table] = count[0] if count else 0

        if "cell_data" in sync_tables:
            # restore's INSERT above supplies explicit cell_id values from
            # the source; keep the local sequence ahead of them so future
            # inserts don't collide. (merge never carries cell_id over, so
            # this is a no-op there beyond the harmless MAX() check.)
            max_id = self.conn.execute("SELECT MAX(cell_id) FROM cell_data").fetchone()[0]
            if max_id is not None:
                current = self.conn.execute(
                    "SELECT last_value FROM duckdb_sequences() WHERE sequence_name = 'cell_id_seq'"
                ).fetchone()[0]
                current = current or 0
                if current < max_id:
                    self.conn.execute(
                        f"SELECT nextval('cell_id_seq') FROM range({int(max_id) - int(current)})"
                    )

        return rows

    # ========== Tags (free-form metadata) ==========
    #
    # `run_tags` is a sidecar table (scope, key) -> JSON, kept deliberately
    # separate from config_parameters/cell_data so tagging never mutates a
    # production table's schema. Three *validated* scopes correspond to real,
    # already-existing identifiers in this registry -- each checks the target
    # exists before writing, and each is documented below against exactly the
    # tables/columns it joins:
    #
    #   scope="run_hash"    -- joins job_configs.run_hash, config_parameters.run_hash,
    #                          job_runs.run_hash, cell_data.run_hash (all the same
    #                          value, denormalized on purpose). Set via
    #                          tag_by_run_hash(); DiagnosticQueries.get_parameter_comparison
    #                          auto-joins this scope when param_name isn't a
    #                          declared sweep parameter.
    #   scope="session_id"  -- joins sweep_sessions.session_id, job_runs.session_id,
    #                          session_configs.session_id. Set via tag_by_session_id().
    #   scope="run_id"      -- joins job_runs.run_id, cell_data.run_id, run_outputs.run_id
    #                          (via job_runs). Set via tag_by_run_id().
    #
    # Anything else (e.g. a site code shared by many run_hashes) is a
    # *synthetic* grouping with no corresponding column anywhere in the
    # schema -- there is no join for it, by definition. Use tag_custom() /
    # get_custom_tags() for that, and find_tagged() to recover the set of
    # keys sharing a tag value (e.g. every run_hash at a given site).

    #: Scopes with a validated tag_by_*()/get_tags_by_*() pair. tag_custom()
    #: refuses these -- use the dedicated method instead, which checks the
    #: target actually exists.
    _VALIDATED_TAG_SCOPES = ("run_hash", "session_id", "run_id")

    def _upsert_tags(self, scope: str, key: str, tags: dict[str, Any]) -> None:
        """Merge *tags* into whatever's already stored at (scope, key)."""
        if not tags:
            raise ValueError("at least one tag=value pair is required")

        existing = self.conn.execute(
            "SELECT tags FROM run_tags WHERE scope = ? AND key = ?", [scope, key]
        ).fetchone()
        merged = json.loads(existing[0]) if existing else {}
        merged.update(tags)

        self.conn.execute(
            """
            INSERT INTO run_tags (scope, key, tags, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (scope, key)
            DO UPDATE SET tags = excluded.tags, updated_at = excluded.updated_at
            """,
            [scope, key, json.dumps(merged)],
        )

    def _read_tags(self, scope: str, key: str) -> dict[str, Any]:
        result = self.conn.execute(
            "SELECT tags FROM run_tags WHERE scope = ? AND key = ?", [scope, key]
        ).fetchone()
        return json.loads(result[0]) if result else {}

    def tag_by_run_hash(self, run_hash: str, **tags: Any) -> None:
        """Attach free-form JSON metadata to a run_hash.

        Joins ``job_configs.run_hash`` / ``config_parameters.run_hash`` /
        ``job_runs.run_hash`` / ``cell_data.run_hash`` -- all the same value.
        ``DiagnosticQueries.get_parameter_comparison(variable, param_name)``
        picks these up automatically for any ``param_name`` that isn't a
        declared sweep parameter.

        Calling this again for the same run_hash merges into the existing
        tags (like ``dict.update``) rather than replacing them.

        Args:
            run_hash: The run to tag.
            **tags: Tag values to set, in whatever shape you like.

        Raises:
            KeyError: If run_hash isn't a registered run.
            ValueError: If no tags are given.

        Examples:
            >>> registry.tag_by_run_hash(run_hash, site="JOTR001", biome="desert")
        """
        if self.get_config_by_hash(run_hash) is None:
            raise KeyError(f"No run found with hash '{run_hash}'")
        self._upsert_tags("run_hash", run_hash, tags)

    def get_tags_by_run_hash(self, run_hash: str) -> dict[str, Any]:
        """Get the tags attached to a run_hash, or ``{}`` if none."""
        return self._read_tags("run_hash", run_hash)

    def tag_by_session_id(self, session_id: str, **tags: Any) -> None:
        """Attach free-form JSON metadata to a sweep session.

        Joins ``sweep_sessions.session_id`` / ``job_runs.session_id`` /
        ``session_configs.session_id``.

        Args:
            session_id: The session to tag.
            **tags: Tag values to set, in whatever shape you like.

        Raises:
            KeyError: If session_id isn't a registered session.
            ValueError: If no tags are given.
        """
        if self.get_session(session_id) is None:
            raise KeyError(f"No session found with id '{session_id}'")
        self._upsert_tags("session_id", session_id, tags)

    def get_tags_by_session_id(self, session_id: str) -> dict[str, Any]:
        """Get the tags attached to a session_id, or ``{}`` if none."""
        return self._read_tags("session_id", session_id)

    def tag_by_run_id(self, run_id: str, **tags: Any) -> None:
        """Attach free-form JSON metadata to a specific run execution.

        Distinct from ``tag_by_run_hash``: a run_hash is *what* was run and
        can have several run_ids (e.g. under the ``pool`` collision policy);
        a run_id is *which execution* produced a given replicate. Joins
        ``job_runs.run_id`` / ``cell_data.run_id`` / ``run_outputs.run_id``
        (the latter via ``job_runs``).

        Args:
            run_id: The run execution to tag.
            **tags: Tag values to set, in whatever shape you like.

        Raises:
            KeyError: If run_id isn't a recorded execution.
            ValueError: If no tags are given.
        """
        if self.get_run(run_id) is None:
            raise KeyError(f"No run execution found with id '{run_id}'")
        self._upsert_tags("run_id", run_id, tags)

    def get_tags_by_run_id(self, run_id: str) -> dict[str, Any]:
        """Get the tags attached to a run_id, or ``{}`` if none."""
        return self._read_tags("run_id", run_id)

    def tag_custom(self, key: str, *, scope: str, **tags: Any) -> None:
        """Attach free-form JSON metadata under a synthetic scope.

        For groupings that don't correspond to any column in the schema --
        e.g. a site code shared by many run_hashes. Unlike ``tag_by_run_hash``
        etc., there's no existence check (nothing to check against) and no
        automatic join anywhere: recover matching keys with ``find_tagged()``,
        then use them as a plain filter (e.g. ``run_hash IN (...)``) against
        whatever table you're actually querying.

        Args:
            key: The key to tag (e.g. ``"JOTR001"``).
            scope: Name for this synthetic grouping (e.g. ``"site"``). Must
                not be one of the validated scopes (``run_hash``,
                ``session_id``, ``run_id``) -- use the matching ``tag_by_*``
                method for those instead.
            **tags: Tag values to set, in whatever shape you like.

        Raises:
            ValueError: If scope is a validated scope, or no tags are given.

        Examples:
            >>> registry.tag_custom("JOTR001", scope="site", n_plots=12)
        """
        if scope in self._VALIDATED_TAG_SCOPES:
            raise ValueError(
                f"scope={scope!r} is one of joshpy's own validated scopes -- "
                f"use tag_by_{scope}() instead, which checks the target exists."
            )
        self._upsert_tags(scope, key, tags)

    def get_custom_tags(self, key: str, *, scope: str) -> dict[str, Any]:
        """Get the tags attached to *key* under a synthetic *scope*."""
        return self._read_tags(scope, key)

    def list_tag_keys(self, *, scope: str) -> list[str]:
        """List all distinct tag keys in use for a scope.

        Args:
            scope: Scope to inspect (e.g. ``"run_hash"``, or a custom scope
                like ``"site"``).

        Returns:
            Sorted list of tag keys (e.g. ``["biome", "site"]``).
        """
        rows = self.conn.execute(
            "SELECT DISTINCT json_keys(tags) FROM run_tags WHERE scope = ?",
            [scope],
        ).fetchall()
        keys: set[str] = set()
        for row in rows:
            keys.update(row[0] or [])
        return sorted(keys)

    def find_tagged(self, tag_key: str, value: Any, *, scope: str = "run_hash") -> list[str]:
        """Find every key in *scope* whose tags[tag_key] == value.

        The reverse of ``get_tags_by_*`` / ``get_custom_tags``: given a tag
        value (e.g. a site code), recover every key (e.g. run_hash) that
        carries it -- the many-to-one direction, since several keys can
        share the same tag value.

        Args:
            tag_key: The tag key to match on (e.g. ``"site"``).
            value: The value to match (compared as text).
            scope: Scope to search. Defaults to ``"run_hash"``.

        Returns:
            List of matching keys (e.g. run_hashes), in no particular order.

        Examples:
            >>> registry.tag_by_run_hash("hash1", site="JOTR001")
            >>> registry.tag_by_run_hash("hash2", site="JOTR001")
            >>> registry.find_tagged("site", "JOTR001")
            ['hash1', 'hash2']
        """
        json_path = f"$.{tag_key}"
        rows = self.conn.execute(
            "SELECT key FROM run_tags WHERE scope = ? AND json_extract_string(tags, ?) = ?",
            [scope, json_path, str(value)],
        ).fetchall()
        return [row[0] for row in rows]

    # ========== Filter Context Managers ==========

    @contextmanager
    def spatial_filter(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        geojson: str | dict | None = None,
    ) -> Iterator[None]:
        """Context manager for spatial filtering of queries.

        All DiagnosticQueries within this context will be spatially filtered.
        Can be nested with time_filter().

        Args:
            bbox: Bounding box as (min_lon, max_lon, min_lat, max_lat).
            geojson: GeoJSON polygon string or dict.

        Raises:
            ValueError: If both bbox and geojson are provided.

        Examples:
            >>> with registry.spatial_filter(bbox=(-116, -115, 33.5, 34.0)):
            ...     df = queries.get_timeseries("height", run_hash="abc123")

            >>> # Nested with time filter
            >>> with registry.spatial_filter(geojson=park_boundary):
            ...     with registry.time_filter(step_range=(0, 50)):
            ...         df = queries.get_timeseries("height", run_hash="abc123")
        """
        if bbox and geojson:
            raise ValueError("Specify either bbox or geojson, not both")

        # Save previous state (for nested calls)
        prev_bbox = self._spatial_filter_bbox
        prev_geojson = self._spatial_filter_geojson

        self._spatial_filter_bbox = bbox
        self._spatial_filter_geojson = geojson
        try:
            yield
        finally:
            self._spatial_filter_bbox = prev_bbox
            self._spatial_filter_geojson = prev_geojson

    @contextmanager
    def time_filter(
        self,
        step_range: tuple[int, int],
    ) -> Iterator[None]:
        """Context manager for temporal filtering of queries.

        All DiagnosticQueries within this context will be filtered to
        the specified step range. Can be nested with spatial_filter().

        Args:
            step_range: Tuple of (min_step, max_step) inclusive.

        Examples:
            >>> with registry.time_filter(step_range=(0, 50)):
            ...     df = queries.get_timeseries("height", run_hash="abc123")

            >>> # Nested with spatial filter
            >>> with registry.time_filter(step_range=(10, 20)):
            ...     with registry.spatial_filter(bbox=(-116, -115, 33.5, 34.0)):
            ...         df = queries.get_comparison("height", group_by="maxGrowth")
        """
        prev_range = self._time_filter_range
        self._time_filter_range = step_range
        try:
            yield
        finally:
            self._time_filter_range = prev_range

    def _get_filter_clauses(self) -> tuple[str, list]:
        """Get SQL WHERE clauses for all active filters.

        This is used internally by DiagnosticQueries to apply active
        spatial and temporal filters.

        Returns:
            Tuple of (where_clause_string, params_list).
            The where_clause_string starts with " AND " if there are clauses.
        """
        clauses = []
        params: list = []

        # Spatial filter
        if self._spatial_filter_bbox:
            min_lon, max_lon, min_lat, max_lat = self._spatial_filter_bbox
            clauses.append("longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?")
            params.extend([min_lon, max_lon, min_lat, max_lat])
        elif self._spatial_filter_geojson:
            geojson_str = (
                json.dumps(self._spatial_filter_geojson)
                if isinstance(self._spatial_filter_geojson, dict)
                else self._spatial_filter_geojson
            )
            clauses.append("ST_Within(geom, ST_GeomFromGeoJSON(?))")
            params.append(geojson_str)

        # Time filter
        if self._time_filter_range:
            min_step, max_step = self._time_filter_range
            clauses.append("step BETWEEN ? AND ?")
            params.extend([min_step, max_step])

        if clauses:
            return (" AND " + " AND ".join(clauses), params)
        return ("", [])

    # ========== Session Management ==========

    def create_session(
        self,
        config: Any,
        experiment_name: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create a new sweep session.

        Args:
            config: Job configuration containing simulation, template, and sweep info.
                Must have `simulation`, `template_path`, and `to_dict()` attributes
                (typically a JobConfig from joshpy.jobs).
            experiment_name: Name for the experiment. Defaults to config.simulation.
            session_id: Optional externally-provided session ID.
                        If None, generates a UUID. This allows the frontend/API
                        layer to manage session IDs (e.g., using project IDs).

        Returns:
            The session ID (generated or provided).

        Examples:
            >>> from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter
            >>> config = JobConfig(
            ...     template_path=Path("template.jshc.j2"),
            ...     source_path=Path("simulation.josh"),
            ...     simulation="Main",
            ...     sweep=SweepConfig(
            ...         config_parameters=[ConfigSweepParameter(name="maxGrowth", values=[10, 20, 30])]
            ...     ),
            ... )
            >>> session_id = registry.create_session(config=config)

        Note:
            total_jobs and total_replicates are computed from the JobSet
            after job expansion. Use job_set.total_jobs and job_set.total_replicates.
        """
        if session_id is None:
            session_id = _generate_id()

        # Extract fields from config
        simulation = getattr(config, "simulation", None)
        template_path = getattr(config, "template_path", None)
        template_path_str = str(template_path) if template_path else None

        # Use experiment_name if provided, otherwise default to simulation
        if experiment_name is None:
            experiment_name = simulation

        # Auto-compute metadata from config
        metadata = {"job_config": config.to_dict()}
        git_hash = _get_git_hash()
        if git_hash:
            metadata["git_hash"] = git_hash
        metadata_json = json.dumps(metadata)

        self.conn.execute(
            """
            INSERT INTO sweep_sessions
            (session_id, experiment_name, simulation,
             template_path, template_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                session_id,
                experiment_name,
                simulation,
                template_path_str,
                None,  # template_hash is no longer used
                metadata_json,
            ],
        )
        return session_id

    def update_session_status(self, session_id: str, status: str) -> None:
        """Update the status of a session.

        Args:
            session_id: The session to update.
            status: New status (e.g., 'running', 'completed', 'failed').
        """
        self.conn.execute(
            "UPDATE sweep_sessions SET status = ? WHERE session_id = ?",
            [status, session_id],
        )

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Get session information by ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            SessionInfo if found, None otherwise.
        """
        result = self.conn.execute(
            """
            SELECT session_id, experiment_name, created_at, template_path, template_hash,
                   simulation, total_jobs, total_replicates, status, metadata
            FROM sweep_sessions
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()

        if result is None:
            return None

        return SessionInfo(
            session_id=result[0],
            experiment_name=result[1],
            created_at=result[2],
            template_path=result[3],
            template_hash=result[4],
            simulation=result[5],
            total_jobs=result[6],
            total_replicates=result[7],
            status=result[8],
            metadata=json.loads(result[9]) if result[9] else None,
        )

    def list_sessions(
        self, experiment_name: str | None = None, limit: int = 100
    ) -> list[SessionInfo]:
        """List sessions, optionally filtered by experiment name.

        Args:
            experiment_name: Filter by experiment name (optional).
            limit: Maximum number of sessions to return.

        Returns:
            List of SessionInfo objects, ordered by creation time (newest first).
        """
        if experiment_name is not None:
            result = self.conn.execute(
                """
                SELECT session_id, experiment_name, created_at, template_path, template_hash,
                       simulation, total_jobs, total_replicates, status, metadata
                FROM sweep_sessions
                WHERE experiment_name = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [experiment_name, limit],
            ).fetchall()
        else:
            result = self.conn.execute(
                """
                SELECT session_id, experiment_name, created_at, template_path, template_hash,
                       simulation, total_jobs, total_replicates, status, metadata
                FROM sweep_sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

        return [
            SessionInfo(
                session_id=row[0],
                experiment_name=row[1],
                created_at=row[2],
                template_path=row[3],
                template_hash=row[4],
                simulation=row[5],
                total_jobs=row[6],
                total_replicates=row[7],
                status=row[8],
                metadata=json.loads(row[9]) if row[9] else None,
            )
            for row in result
        ]

    # ========== Run Registration ==========

    def register_run(
        self,
        session_id: str,
        run_hash: str,
        josh_path: str,
        config_content: str,
        file_mappings: dict[str, dict[str, str]] | None,
        parameters: dict[str, Any],
        josh_content: str | None = None,
    ) -> None:
        """Register a job configuration (run specification).

        Args:
            session_id: Session this config belongs to.
            run_hash: MD5 hash of josh + config + file_mappings (12 chars).
            josh_path: Path to the .josh script file.
            config_content: Full text of the rendered configuration.
            file_mappings: Dict mapping names to {"path": "...", "hash": "..."}.
            parameters: Parameter values used to generate this config.
            josh_content: Rendered .josh source content (optional).
        """
        file_mappings_json = json.dumps(file_mappings) if file_mappings else None

        # Use INSERT OR IGNORE to handle duplicate run hashes
        self.conn.execute(
            """
            INSERT OR IGNORE INTO job_configs
            (run_hash, session_id, josh_path, josh_content, config_content, file_mappings)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [run_hash, session_id, josh_path, josh_content, config_content, file_mappings_json],
        )

        # Always record the session<->config link (supports pooled runs
        # where the same run_hash is used across multiple sessions)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO session_configs (session_id, run_hash)
            VALUES (?, ?)
            """,
            [session_id, run_hash],
        )

        # Insert typed parameters into config_parameters table
        if parameters:
            # Infer types for each parameter and ensure columns exist
            param_columns = {name: _infer_type(value) for name, value in parameters.items()}
            self._ensure_config_columns(param_columns)

            # Build dynamic INSERT with all parameter columns
            param_names = list(parameters.keys())
            quoted_cols = ", ".join(_quote_identifier(n) for n in param_names)
            placeholders = ", ".join("?" for _ in param_names)
            values = [parameters[n] for n in param_names]

            # INSERT OR IGNORE for idempotency (same run_hash)
            self.conn.execute(
                f"""
                INSERT OR IGNORE INTO config_parameters
                (run_hash, {quoted_cols})
                VALUES (?, {placeholders})
                """,
                [run_hash] + values,
            )
        else:
            # No parameters, just insert the run_hash
            self.conn.execute(
                """
                INSERT OR IGNORE INTO config_parameters (run_hash)
                VALUES (?)
                """,
                [run_hash],
            )

    def _get_parameters_for_run_hash(self, run_hash: str) -> dict[str, Any]:
        """Get parameters dict from config_parameters table for a run_hash.

        Args:
            run_hash: The run hash to get parameters for.

        Returns:
            Dict of parameter name -> value, or empty dict if not found.
        """
        param_cols = self.list_config_columns()
        if not param_cols:
            return {}

        # Build query to select all parameter columns
        quoted_cols = ", ".join(_quote_identifier(c) for c in param_cols)
        result = self.conn.execute(
            f"""
            SELECT {quoted_cols}
            FROM config_parameters
            WHERE run_hash = ?
            """,
            [run_hash],
        ).fetchone()

        if result is None:
            return {}

        # Build parameters dict, excluding NULL values
        parameters = {}
        for i, col_name in enumerate(param_cols):
            if result[i] is not None:
                parameters[col_name] = result[i]
        return parameters

    def get_config_by_hash(self, run_hash: str) -> ConfigInfo | None:
        """Get config information by run hash.

        Args:
            run_hash: The run hash to look up.

        Returns:
            ConfigInfo if found, None otherwise.
        """
        result = self.conn.execute(
            """
            SELECT run_hash, session_id, josh_path, josh_content,
                   config_content, file_mappings, label, created_at,
                   status, superseded_by, status_reason, status_updated_at
            FROM job_configs
            WHERE run_hash = ?
            """,
            [run_hash],
        ).fetchone()

        if result is None:
            return None

        # Get parameters from typed columns
        parameters = self._get_parameters_for_run_hash(run_hash)

        return ConfigInfo(
            run_hash=result[0],
            session_id=result[1],
            josh_path=result[2],
            josh_content=result[3],
            config_content=result[4],
            file_mappings=json.loads(result[5]) if result[5] else None,
            parameters=parameters,
            label=result[6],
            created_at=result[7],
            status=result[8],
            superseded_by=result[9],
            status_reason=result[10],
            status_updated_at=result[11],
        )

    def get_file_mappings(
        self, label_or_hash: str
    ) -> dict[str, Path] | None:
        """Get the external data file mappings for a run.

        Args:
            label_or_hash: Run label or run_hash.

        Returns:
            Dict mapping data names to file paths, or None if no
            file mappings were registered for this run.

        Raises:
            KeyError: If the label or hash is not found.
        """
        run_hash = self._resolve_label_or_hash(label_or_hash)
        config = self.get_config_by_hash(run_hash)
        if config is None:
            raise KeyError(f"No run found for '{label_or_hash}'")
        if config.file_mappings is None:
            return None
        return {
            name: Path(info["path"])
            for name, info in config.file_mappings.items()
        }

    def _resolve_label_or_hash(self, label_or_hash: str) -> str:
        """Resolve a label-or-hash string to a valid run_hash.

        Args:
            label_or_hash: Label or run_hash.

        Returns:
            Resolved run_hash.

        Raises:
            KeyError: If no matching run exists.
        """
        try:
            run_hash = self.resolve_label(label_or_hash)
        except KeyError:
            run_hash = label_or_hash

        if self.get_config_by_hash(run_hash) is None:
            raise KeyError(f"No run found for '{label_or_hash}'")
        return run_hash

    def _resolve_run_id_for_hash(
        self,
        run_hash: str,
        run_id: str | None = None,
    ) -> str:
        """Resolve a run_id for a run_hash, defaulting to latest execution.

        Args:
            run_hash: Resolved run hash.
            run_id: Optional explicit run_id.

        Returns:
            A run_id associated with run_hash.

        Raises:
            KeyError: If no matching run execution exists.
            ValueError: If run_id exists but does not match run_hash.
        """
        if run_id is not None:
            row = self.conn.execute(
                "SELECT run_hash FROM job_runs WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if row is None:
                raise KeyError(f"No run execution found for run_id '{run_id}'")
            if row[0] != run_hash:
                raise ValueError(
                    f"run_id '{run_id}' belongs to run_hash '{row[0]}', "
                    f"not '{run_hash}'"
                )
            return run_id

        latest = self.conn.execute(
            """
            SELECT run_id
            FROM job_runs
            WHERE run_hash = ?
            ORDER BY started_at DESC NULLS LAST, completed_at DESC NULLS LAST, run_id DESC
            LIMIT 1
            """,
            [run_hash],
        ).fetchone()
        if latest is None:
            raise KeyError(f"No run executions found for run_hash '{run_hash}'")
        return latest[0]

    def get_configs_for_session(self, session_id: str) -> list[ConfigInfo]:
        """Get all configs for a session.

        Args:
            session_id: The session ID to get configs for.

        Returns:
            List of ConfigInfo objects.
        """
        # First get the run_hashes and basic config info
        # Join through session_configs to support pooled runs (same
        # run_hash registered across multiple sessions).
        result = self.conn.execute(
            """
            SELECT jc.run_hash, jc.session_id, jc.josh_path, jc.josh_content,
                   jc.config_content, jc.file_mappings, jc.label, jc.created_at,
                   jc.status, jc.superseded_by, jc.status_reason,
                   jc.status_updated_at
            FROM job_configs jc
            JOIN session_configs sc ON jc.run_hash = sc.run_hash
            WHERE sc.session_id = ?
            ORDER BY sc.created_at
            """,
            [session_id],
        ).fetchall()

        configs = []
        for row in result:
            run_hash = row[0]
            parameters = self._get_parameters_for_run_hash(run_hash)
            configs.append(
                ConfigInfo(
                    run_hash=run_hash,
                    session_id=row[1],
                    josh_path=row[2],
                    josh_content=row[3],
                    config_content=row[4],
                    file_mappings=json.loads(row[5]) if row[5] else None,
                    parameters=parameters,
                    label=row[6],
                    created_at=row[7],
                    status=row[8],
                    superseded_by=row[9],
                    status_reason=row[10],
                    status_updated_at=row[11],
                )
            )
        return configs

    # ========== Labels ==========

    def label_run(
        self,
        run_hash: str,
        label: str,
        force: bool = False,
        on_collision: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Assign a human-readable label to a run configuration.

        A label is a **pure alias** — the one handle you resolve to a run. It is
        not where currency or run attributes live: currency is
        :meth:`mark_run`/``status`` and attributes are tags (see
        REGISTRY_PROVENANCE.md). Labels are unique within a registry; on
        collision the behavior depends on ``force`` and ``on_collision``:

        - Default: raise ``ValueError``.
        - ``force=True``: silently drop the old label and reassign, leaving the
          old run's status untouched. For "I mislabeled it, just fix it."
        - ``on_collision="supersede"``: the old run **releases the label**
          (``label = NULL``) and is marked ``status = 'superseded'`` with
          ``superseded_by = run_hash`` and the given ``reason``; the new run
          takes the label. This records real provenance instead of mangling the
          old label with a timestamp suffix (the retired ``"timestamp"`` mode).

        Args:
            run_hash: The run hash to label.
            label: Human-readable label (e.g., "baseline", "high_mortality").
            force: If True, reassign the label even if already taken.
            on_collision: Collision strategy. ``"supersede"`` archives the old
                run via supersession. Mutually exclusive with ``force``.
            reason: Free-text explanation stored on the superseded run. Only
                meaningful with ``on_collision="supersede"``.

        Raises:
            KeyError: If run_hash does not exist.
            ValueError: If label is already assigned to a different run
                and neither force nor on_collision is set, or if both
                force and on_collision are set, or if on_collision has
                an invalid value.
        """
        if force and on_collision is not None:
            raise ValueError(
                "force and on_collision are mutually exclusive. "
                "Use force=True to drop the old label, or "
                "on_collision='supersede' to archive it."
            )
        if on_collision is not None and on_collision != "supersede":
            raise ValueError(
                f"Invalid on_collision value: {on_collision!r}. "
                "Must be 'supersede' or None."
            )

        # Verify run_hash exists
        existing = self.conn.execute(
            "SELECT run_hash FROM job_configs WHERE run_hash = ?", [run_hash]
        ).fetchone()
        if existing is None:
            raise KeyError(f"No run found with hash '{run_hash}'")

        # Check if label is already taken
        taken = self.conn.execute(
            "SELECT run_hash FROM job_configs WHERE label = ?", [label]
        ).fetchone()
        if taken is not None and taken[0] != run_hash:
            if on_collision == "supersede":
                # Old run releases the label and is marked superseded, pointing
                # at the run that replaced it. It then falls out of list_labels()
                # and current-only reads automatically — no timestamp suffix.
                self.mark_run(
                    taken[0],
                    status="superseded",
                    superseded_by=run_hash,
                    reason=reason,
                )
                self.conn.execute(
                    "UPDATE job_configs SET label = NULL WHERE run_hash = ?",
                    [taken[0]],
                )
            elif force:
                # Clear old assignment
                self.conn.execute(
                    "UPDATE job_configs SET label = NULL WHERE label = ?",
                    [label],
                )
            else:
                raise ValueError(
                    f"Label '{label}' already assigned to run {taken[0]}. "
                    f"Use force=True to reassign, or "
                    f"on_collision='supersede' to archive the old run."
                )

        self.conn.execute(
            "UPDATE job_configs SET label = ? WHERE run_hash = ?", [label, run_hash]
        )

    def list_labels(self) -> list[tuple[str, str]]:
        """List all labeled runs.

        Returns:
            List of (label, run_hash) tuples, sorted by label.
        """
        result = self.conn.execute(
            "SELECT label, run_hash FROM job_configs "
            "WHERE label IS NOT NULL ORDER BY label"
        ).fetchall()
        return [(row[0], row[1]) for row in result]

    def resolve_label(self, label: str) -> str:
        """Get the run_hash for a labeled run.

        Args:
            label: The label to look up.

        Returns:
            The run_hash associated with the label.

        Raises:
            KeyError: If no run has this label.
        """
        result = self.conn.execute(
            "SELECT run_hash FROM job_configs WHERE label = ?", [label]
        ).fetchone()
        if result is None:
            raise KeyError(f"No run found with label '{label}'")
        return result[0]

    # ========== Run status / supersession ==========

    #: Closed vocabulary for ``job_configs.status``. ``None`` (unmarked) is read
    #: as ``"active"`` everywhere via ``coalesce``.
    _RUN_STATUSES = frozenset({"active", "superseded", "bad"})

    def mark_run(
        self,
        run_hash: str,
        status: str,
        *,
        superseded_by: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Set a run's lifecycle status.

        ``status`` is a closed enum — ``"active"``, ``"superseded"``, or
        ``"bad"`` — and is the single source of truth for whether a run should be
        used. "Current" is exactly ``status == "active"`` (see
        REGISTRY_PROVENANCE.md); there is no separate currency flag.

        Args:
            run_hash: The run to mark. Also accepts a label (resolved first).
            status: One of ``{"active", "superseded", "bad"}``.
            superseded_by: The run_hash that replaces this one. Required when
                ``status == "superseded"`` and rejected otherwise. The target
                must exist.
            reason: Free-text explanation, stored alongside the status.

        Raises:
            KeyError: If ``run_hash`` (or ``superseded_by``) does not exist.
            ValueError: If ``status`` is not in the enum, if ``superseded_by`` is
                given without ``status="superseded"`` (or vice versa), or if a
                run is superseded by itself.
        """
        if status not in self._RUN_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. "
                f"Must be one of {sorted(self._RUN_STATUSES)}."
            )
        if status == "superseded" and superseded_by is None:
            raise ValueError(
                "status='superseded' requires superseded_by (the replacing run_hash)."
            )
        if status != "superseded" and superseded_by is not None:
            raise ValueError(
                "superseded_by is only valid with status='superseded'."
            )

        run_hash = self._resolve_label_or_hash(run_hash)

        if superseded_by is not None:
            if superseded_by == run_hash:
                raise ValueError("A run cannot supersede itself.")
            target = self.conn.execute(
                "SELECT run_hash FROM job_configs WHERE run_hash = ?",
                [superseded_by],
            ).fetchone()
            if target is None:
                raise KeyError(f"No run found with hash '{superseded_by}'")

        # active is the resting state: clear the supersession link when returning
        # a run to active so stale provenance doesn't linger.
        self.conn.execute(
            """
            UPDATE job_configs
            SET status = ?, superseded_by = ?, status_reason = ?,
                status_updated_at = CURRENT_TIMESTAMP
            WHERE run_hash = ?
            """,
            [status, superseded_by, reason, run_hash],
        )

    def get_run_status(self, label_or_hash: str) -> RunStatus:
        """Get a run's lifecycle status.

        Args:
            label_or_hash: Label or run_hash of the run.

        Returns:
            A :class:`RunStatus`. Unmarked runs report ``status=None``, which
            :attr:`RunStatus.effective_status` normalizes to ``"active"``.

        Raises:
            KeyError: If the label or hash is not found.
        """
        run_hash = self._resolve_label_or_hash(label_or_hash)
        row = self.conn.execute(
            """
            SELECT status, superseded_by, status_reason, status_updated_at
            FROM job_configs WHERE run_hash = ?
            """,
            [run_hash],
        ).fetchone()
        # _resolve_label_or_hash guarantees the row exists.
        assert row is not None
        return RunStatus(
            run_hash=run_hash,
            status=row[0],
            superseded_by=row[1],
            reason=row[2],
            updated_at=row[3],
        )

    def supersede(
        self,
        run_hash: str,
        replaces: str,
        reason: str | None = None,
    ) -> None:
        """Mark ``run_hash`` as the replacement for an existing run.

        Convenience wrapper over :meth:`mark_run` for the partial-rerun story:
        the replaced run becomes ``status='superseded'`` pointing at ``run_hash``.
        If the replaced run held a label, that label is released so it no longer
        resolves to the stale run — attach it to the new run with
        :meth:`label_run` if desired.

        Args:
            run_hash: The new, replacing run.
            replaces: Label or run_hash of the run being retired.
            reason: Free-text explanation stored on the superseded run.

        Raises:
            KeyError: If either run is not found.
            ValueError: If a run would supersede itself.
        """
        old_hash = self._resolve_label_or_hash(replaces)
        self.mark_run(
            old_hash, status="superseded", superseded_by=run_hash, reason=reason
        )
        self.conn.execute(
            "UPDATE job_configs SET label = NULL WHERE run_hash = ?", [old_hash]
        )

    def run_history(self, label_or_hash: str) -> list[RunDetail]:
        """Walk the supersession chain for a run, newest (current) first.

        Replaces the retired ``resolve_latest()`` prefix-matching. Starting from
        the given run, follows ``superseded_by`` backwards to reconstruct the
        lineage. Because supersession points from old → new, the chain is
        recovered by finding, at each step, the run that the previous entry
        supersedes.

        Args:
            label_or_hash: Label or run_hash of any run in the lineage. Typically
                the current (active) run.

        Returns:
            A list of :class:`RunDetail`, current run first, then each run it
            superseded, oldest last. Length 1 when nothing was superseded.

        Raises:
            KeyError: If the label or hash is not found.
        """
        head = self._resolve_label_or_hash(label_or_hash)
        history: list[RunDetail] = []
        seen: set[str] = set()
        current: str | None = head
        while current is not None and current not in seen:
            seen.add(current)
            history.append(self.get_run_info(current))
            prev = self.conn.execute(
                "SELECT run_hash FROM job_configs WHERE superseded_by = ?",
                [current],
            ).fetchone()
            current = prev[0] if prev is not None else None
        return history

    def export_config(self, label_or_hash: str, output_dir: str | Path) -> Path:
        """Export a run's config content to a file for IDE diffing.

        Resolves by label first, falls back to run_hash.

        Args:
            label_or_hash: Label or run_hash to look up.
            output_dir: Directory to write the config file to.

        Returns:
            Path to the written file.

        Raises:
            KeyError: If no matching run is found.
        """
        # Try label first, fall back to hash
        try:
            run_hash = self.resolve_label(label_or_hash)
            filename = f"{label_or_hash}.jshc"
        except KeyError:
            run_hash = label_or_hash
            filename = f"{run_hash}.jshc"

        config = self.get_config_by_hash(run_hash)
        if config is None:
            raise KeyError(f"No run found for '{label_or_hash}'")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        header = (
            f"# READ-ONLY snapshot exported from registry\n"
            f"# Run: {run_hash}\n"
            f"# Editing this file has no effect. To change parameters,\n"
            f"# edit your source .jshc file and re-run.\n\n"
        )
        output_path.write_text(header + config.config_content)
        return output_path

    def export_josh(self, label_or_hash: str, output_dir: str | Path) -> Path:
        """Export a run's josh source content to a file for IDE viewing/diffing.

        Resolves by label first, falls back to run_hash.

        Args:
            label_or_hash: Label or run_hash to look up.
            output_dir: Directory to write the josh file to.

        Returns:
            Path to the written file.

        Raises:
            KeyError: If no matching run is found or josh content is not stored.
        """
        try:
            run_hash = self.resolve_label(label_or_hash)
            filename = f"{label_or_hash}.josh"
        except KeyError:
            run_hash = label_or_hash
            filename = f"{run_hash}.josh"

        config = self.get_config_by_hash(run_hash)
        if config is None:
            raise KeyError(f"No run found for '{label_or_hash}'")
        if config.josh_content is None:
            raise KeyError(
                f"No josh content stored for run '{label_or_hash}'. "
                f"Re-register the run with josh_content to enable this feature."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        header = (
            f"# READ-ONLY snapshot exported from registry\n"
            f"# Run: {run_hash}\n"
            f"# Editing this file has no effect.\n\n"
        )
        output_path.write_text(header + config.josh_content)
        return output_path

    def resolve_config_source(self, run_hash: str) -> ConfigSourceInfo:
        """Locate the original .jshc file on disk and check if it still matches.

        Looks up the session metadata to find the original ``config_path``,
        then checks whether the file exists and whether its content has
        changed since it was registered.

        Args:
            run_hash: The run hash to look up.

        Returns:
            A :class:`ConfigSourceInfo` describing the file's status.
        """
        config_info = self.get_config_by_hash(run_hash)
        if config_info is None:
            return ConfigSourceInfo(path=None, exists=False, content_matches=False)

        session = self.get_session(config_info.session_id)
        if session is None:
            return ConfigSourceInfo(path=None, exists=False, content_matches=False)

        job_config = session.job_config
        if job_config is None or getattr(job_config, "config_path", None) is None:
            return ConfigSourceInfo(path=None, exists=False, content_matches=False)

        original_path = Path(job_config.config_path)
        if not original_path.exists():
            return ConfigSourceInfo(
                path=original_path, exists=False, content_matches=False
            )

        try:
            current_content = original_path.read_text()
        except OSError:
            return ConfigSourceInfo(
                path=original_path, exists=False, content_matches=False
            )

        matches = current_content == config_info.config_content
        return ConfigSourceInfo(
            path=original_path, exists=True, content_matches=matches
        )

    def resolve_josh_source(self, run_hash: str) -> ConfigSourceInfo:
        """Locate the original .josh file on disk and check if it still matches.

        Compares the file at ``josh_path`` against the stored ``josh_content``.

        Args:
            run_hash: The run hash to look up.

        Returns:
            A :class:`ConfigSourceInfo` describing the file's status.
        """
        config_info = self.get_config_by_hash(run_hash)
        if config_info is None:
            return ConfigSourceInfo(path=None, exists=False, content_matches=False)

        if config_info.josh_path is None or config_info.josh_content is None:
            return ConfigSourceInfo(path=None, exists=False, content_matches=False)

        original_path = Path(config_info.josh_path)
        if not original_path.exists():
            return ConfigSourceInfo(
                path=original_path, exists=False, content_matches=False
            )

        try:
            current_content = original_path.read_text()
        except OSError:
            return ConfigSourceInfo(
                path=original_path, exists=False, content_matches=False
            )

        matches = current_content == config_info.josh_content
        return ConfigSourceInfo(
            path=original_path, exists=True, content_matches=matches
        )

    def compare_configs(
        self,
        label_or_hash_1: str,
        label_or_hash_2: str,
        ide: str = "vscode",
    ) -> tuple[Path, Path]:
        """Export two run configs and open a side-by-side diff in an IDE.

        Convenience wrapper around :func:`joshpy.diff.open_diff`.

        Args:
            label_or_hash_1: Label or run_hash of the first run.
            label_or_hash_2: Label or run_hash of the second run.
            ide: IDE to open diff in (default: ``"vscode"``).
                Supported: ``"vscode"``, ``"cursor"``.

        Returns:
            Tuple of Paths to the exported config files.

        Raises:
            KeyError: If a label or hash is not found.
            RuntimeError: If the IDE CLI is not found in PATH.
        """
        from joshpy.inspect import open_diff

        return open_diff(self, label_or_hash_1, label_or_hash_2, ide=ide)

    def compare_josh(
        self,
        label_or_hash_1: str,
        label_or_hash_2: str,
        ide: str = "vscode",
    ) -> tuple[Path, Path]:
        """Export two runs' josh sources and open a side-by-side diff in an IDE.

        Convenience wrapper around :func:`joshpy.inspect.open_josh_diff`.

        Args:
            label_or_hash_1: Label or run_hash of the first run.
            label_or_hash_2: Label or run_hash of the second run.
            ide: IDE to open diff in (default: ``"vscode"``).
                Supported: ``"vscode"``, ``"cursor"``.

        Returns:
            Tuple of Paths to the exported josh files.

        Raises:
            KeyError: If a label or hash is not found, or josh content
                is not stored.
            RuntimeError: If the IDE CLI is not found in PATH.
        """
        from joshpy.inspect import open_josh_diff

        return open_josh_diff(self, label_or_hash_1, label_or_hash_2, ide=ide)

    # ========== Describe (human-readable summaries) ==========

    def describe_labels(self) -> str:
        """Human-readable listing of all labeled runs.

        Convenience wrapper around :func:`joshpy.inspect.format_labels`.

        Returns:
            A formatted table of labels with run_hash and creation time.
        """
        from joshpy.inspect import format_labels

        return format_labels(self)

    def describe_sessions(self) -> str:
        """Human-readable listing of all sweep sessions.

        Convenience wrapper around :func:`joshpy.inspect.format_sessions`.

        Returns:
            A formatted table of sessions with experiment name, status, and
            run counts.
        """
        from joshpy.inspect import format_sessions

        return format_sessions(self)

    def describe_run(self, label_or_hash: str) -> str:
        """Human-readable detail for a single run.

        Convenience wrapper around :func:`joshpy.inspect.format_run_info`.

        Args:
            label_or_hash: Label or run_hash of the run to describe.

        Returns:
            A multi-section detail string (parameters, data files, replicates,
            and per-run results).

        Raises:
            KeyError: If the label or hash is not found.
        """
        from joshpy.inspect import format_run_info

        return format_run_info(self, label_or_hash)

    def describe_summary(self) -> str:
        """Human-readable overview of everything in the registry.

        Convenience wrapper around :func:`joshpy.inspect.format_summary`.

        Returns:
            A high-level data summary for the whole registry.
        """
        from joshpy.inspect import format_summary

        return format_summary(self)

    # ========== Bottle ==========

    def bottle(
        self,
        label_or_hash: str,
        output_dir: str | Path = Path("bottles"),
        cli: Any | None = None,
        omit_jshd: bool = False,
    ) -> Path:
        """Create a self-contained bottle archive from a registered run.

        By default, copies data files into the archive and raises if any are
        missing. Use ``omit_jshd=True`` for lightweight archives when the
        recipient has the data locally.

        Args:
            label_or_hash: Run label or run_hash to bottle.
            output_dir: Directory for the archive. Default: ``./bottles/``.
            cli: Optional JoshCLI instance for JAR metadata.
            omit_jshd: If True, skip copying .jshd data files.

        Returns:
            Path to the created ``.tar.gz`` archive.

        Raises:
            KeyError: If the run is not found.
            ValueError: If josh content is not stored for the run.
            FileNotFoundError: If ``omit_jshd`` is False and a data file
                is missing.
        """
        from joshpy.bottle import create_bottle_from_registry

        return create_bottle_from_registry(
            registry=self,
            label_or_hash=label_or_hash,
            cli=cli,
            output_dir=output_dir,
            omit_jshd=omit_jshd,
        )

    # ========== Run Tracking ==========

    def start_run(
        self,
        run_hash: str,
        *,
        session_id: str,
        replicate: int = 0,
        output_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the start of a job run.

        Args:
            run_hash: Run hash for this run.
            session_id: Session that initiated this run.
            replicate: Replicate number (0-indexed).
            output_path: Path where output will be written.
            metadata: Additional metadata.

        Returns:
            The generated run ID.
        """
        run_id = _generate_id()
        metadata_json = json.dumps(metadata) if metadata else None

        self.conn.execute(
            """
            INSERT INTO job_runs
            (run_id, run_hash, session_id, replicate, started_at, output_path, metadata)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """,
            [run_id, run_hash, session_id, replicate, output_path, metadata_json],
        )
        return run_id

    def complete_run(
        self,
        run_id: str,
        exit_code: int,
        error_message: str | None = None,
    ) -> None:
        """Record the completion of a job run.

        Args:
            run_id: The run ID to update.
            exit_code: Process exit code (0 = success).
            error_message: Error message if run failed.
        """
        self.conn.execute(
            """
            UPDATE job_runs
            SET completed_at = CURRENT_TIMESTAMP, exit_code = ?, error_message = ?
            WHERE run_id = ?
            """,
            [exit_code, error_message, run_id],
        )

    def get_run(self, run_id: str) -> RunInfo | None:
        """Get run information by ID.

        Args:
            run_id: The run ID to look up.

        Returns:
            RunInfo if found, None otherwise.
        """
        result = self.conn.execute(
            """
            SELECT run_id, run_hash, replicate, started_at, completed_at,
                   exit_code, output_path, error_message, metadata
            FROM job_runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

        if result is None:
            return None

        return RunInfo(
            run_id=result[0],
            run_hash=result[1],
            replicate=result[2],
            started_at=result[3],
            completed_at=result[4],
            exit_code=result[5],
            output_path=result[6],
            error_message=result[7],
            metadata=json.loads(result[8]) if result[8] else None,
        )

    def get_runs_for_hash(self, run_hash: str) -> list[RunInfo]:
        """Get all runs for a run hash.

        Args:
            run_hash: The run hash to get runs for.

        Returns:
            List of RunInfo objects.
        """
        result = self.conn.execute(
            """
            SELECT run_id, run_hash, replicate, started_at, completed_at,
                   exit_code, output_path, error_message, metadata
            FROM job_runs
            WHERE run_hash = ?
            ORDER BY started_at
            """,
            [run_hash],
        ).fetchall()

        return [
            RunInfo(
                run_id=row[0],
                run_hash=row[1],
                replicate=row[2],
                started_at=row[3],
                completed_at=row[4],
                exit_code=row[5],
                output_path=row[6],
                error_message=row[7],
                metadata=json.loads(row[8]) if row[8] else None,
            )
            for row in result
        ]

    def get_replicate_count(self, run_hash: str) -> int:
        """Get the number of distinct replicates for a run hash from cell_data.

        This is the source-of-truth count, derived from actual loaded data
        rather than from job_runs metadata. Returns 0 if no data has been
        loaded yet.

        The replicate index is the identity of a replicate: counting distinct
        ``replicate`` values gives the number of replicates loaded for this run,
        regardless of which execution (run_id) produced each one. Pooled runs
        dispatch fresh, non-colliding indices, so distinct replicate == total
        replicates; re-ingesting an already-loaded index is a no-op (see
        :meth:`loaded_replicates`).

        Args:
            run_hash: The run hash to count replicates for.

        Returns:
            Number of distinct replicates in cell_data.
        """
        result = self.conn.execute(
            "SELECT COUNT(DISTINCT replicate) FROM cell_data WHERE run_hash = ?",
            [run_hash],
        ).fetchone()
        return result[0] if result else 0

    def loaded_replicates(self, run_hash: str) -> set[int]:
        """Return the set of replicate indices already loaded for a run hash.

        The single source of truth for "what's already ingested". Ingestion
        skips any replicate index already in this set (idempotent re-ingest);
        the replicate index is the dedup identity.

        Args:
            run_hash: The run hash to inspect.

        Returns:
            Set of distinct replicate indices present in cell_data.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT replicate FROM cell_data WHERE run_hash = ?",
            [run_hash],
        ).fetchall()
        return {row[0] for row in rows}

    def get_run_info(self, label_or_hash: str) -> RunDetail:
        """Get aggregated structured detail for a single run.

        Combines :meth:`get_config_by_hash`, :meth:`get_runs_for_hash`, and
        :meth:`get_replicate_count` into one :class:`RunDetail`. This is the
        structured, data-layer counterpart to :meth:`describe_run`, which formats
        this same information as a human-readable string.

        Args:
            label_or_hash: Label or run_hash of the run.

        Returns:
            A :class:`RunDetail` aggregating the config, recorded runs, and
            replicate count.

        Raises:
            KeyError: If the label or hash is not found.

        Examples:
            >>> detail = registry.get_run_info("baseline")
            >>> detail.parameters
            {'survivalProbAdult': 85}
            >>> detail.succeeded, detail.failed, detail.pending
            (3, 0, 0)
        """
        run_hash = self._resolve_label_or_hash(label_or_hash)
        config = self.get_config_by_hash(run_hash)
        # _resolve_label_or_hash guarantees a config exists for this hash.
        assert config is not None
        replicates = self.loaded_replicates(run_hash)
        return RunDetail(
            config=config,
            runs=self.get_runs_for_hash(run_hash),
            replicate_count=len(replicates),
            replicates=replicates,
        )

    def drop_run(self, label_or_hash: str) -> DropSummary:
        """Delete all registry state for a run, so its config can be redone.

        **This is the only operation that deletes or replaces existing run data.**
        All other registry writes are append-only (runs, cell_data) or metadata
        (labels, session status). Use this to clear a run before re-running it
        from scratch — e.g. when a sweep's outputs are bad and you want a clean
        slate rather than pooling more replicates onto them.

        Removes, in foreign-key order, everything tied to the run's hash:
        ``cell_data``, ``run_outputs``, ``job_runs``, ``config_parameters``,
        ``session_configs``, and the ``job_configs`` row (including its label).
        The owning session row is left intact (it may hold other runs).

        Args:
            label_or_hash: Label or run_hash of the run to drop.

        Returns:
            A :class:`DropSummary` with per-table deleted-row counts.

        Raises:
            KeyError: If the label or hash is not found.
        """
        run_hash = self._resolve_label_or_hash(label_or_hash)
        config = self.get_config_by_hash(run_hash)
        label = config.label if config is not None else None

        # Child -> parent order (DuckDB enforces foreign keys on delete).
        deletes: list[tuple[str, str, list[Any]]] = [
            ("cell_data", "DELETE FROM cell_data WHERE run_hash = ?", [run_hash]),
            (
                "run_outputs",
                "DELETE FROM run_outputs WHERE run_id IN "
                "(SELECT run_id FROM job_runs WHERE run_hash = ?)",
                [run_hash],
            ),
            ("job_runs", "DELETE FROM job_runs WHERE run_hash = ?", [run_hash]),
            ("config_parameters", "DELETE FROM config_parameters WHERE run_hash = ?", [run_hash]),
            ("session_configs", "DELETE FROM session_configs WHERE run_hash = ?", [run_hash]),
            ("job_configs", "DELETE FROM job_configs WHERE run_hash = ?", [run_hash]),
        ]

        # DuckDB foreign keys are check-only — it rejects ON DELETE CASCADE
        # ("FOREIGN KEY constraints cannot use CASCADE, SET NULL or SET DEFAULT"),
        # so there's no DB-level cascade to lean on; we delete child-first by hand.
        # Deletes also auto-commit per statement: DuckDB doesn't see a referencing
        # table's deletes within the *same* explicit transaction (a documented FK
        # limitation), so one BEGIN/COMMIT would raise a constraint error.
        # Child-first ordering keeps each step valid; a partial failure (rare) can
        # be finished by re-running drop_run.
        rows: dict[str, int] = {}
        for table, _del, params in deletes:
            where = _del.split("WHERE", 1)[1]
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {where}", params
            ).fetchone()
            rows[table] = count[0] if count else 0
            self._conn.execute(_del, params)

        return DropSummary(run_hash=run_hash, label=label, rows=rows)

    def check_consistency(
        self,
        run_hash: str | None = None,
        *,
        strict: bool = False,
    ) -> list[ConsistencyIssue]:
        """Detect run<->analysis drift between the execution and data tables.

        Guards the principle that the registry (analysis) must reflect what
        running produced. Checks (scoped to *run_hash* if given, else all runs):

        - ``data_without_config`` — ``cell_data`` for a run_hash with no
          ``job_configs`` row (orphaned data; **error**).
        - ``orphan_cell_data`` — ``cell_data`` whose ``run_id`` is absent from
          ``job_runs`` (the FK should prevent this; **error** if seen).
        - ``duplicate_replicate`` — a ``(run_hash, replicate)`` present under more
          than one ``run_id`` (row inflation from a pre-fix re-ingest; **error**).
        - ``ran_not_ingested`` — a run_hash with a succeeded ``job_runs`` row but
          no ``cell_data`` (ran but results never loaded; **warning**).

        Args:
            run_hash: Limit the check to one run, or None for the whole registry.
            strict: If True, raise ``RuntimeError`` when any issue is found.

        Returns:
            A list of :class:`ConsistencyIssue` (empty if consistent).

        Raises:
            RuntimeError: If *strict* and any issue is found.
        """
        and_c_hash = " AND c.run_hash = ?" if run_hash else ""
        and_hash = " AND run_hash = ?" if run_hash else ""
        hash_param: list[Any] = [run_hash] if run_hash else []
        issues: list[ConsistencyIssue] = []

        # data_without_config
        rows = self._conn.execute(
            "SELECT DISTINCT c.run_hash FROM cell_data c "
            "LEFT JOIN job_configs j ON c.run_hash = j.run_hash "
            f"WHERE j.run_hash IS NULL{and_c_hash}",
            hash_param,
        ).fetchall()
        for (h,) in rows:
            issues.append(ConsistencyIssue(
                "data_without_config", h,
                f"cell_data exists for run_hash {h} but it has no job_configs row.",
                "error",
            ))

        # orphan_cell_data
        rows = self._conn.execute(
            "SELECT DISTINCT c.run_hash FROM cell_data c "
            "LEFT JOIN job_runs r ON c.run_id = r.run_id "
            f"WHERE r.run_id IS NULL{and_c_hash}",
            hash_param,
        ).fetchall()
        for (h,) in rows:
            issues.append(ConsistencyIssue(
                "orphan_cell_data", h,
                f"cell_data for run_hash {h} references a run_id absent from job_runs.",
                "error",
            ))

        # duplicate_replicate: (run_hash, replicate) under >1 run_id
        rows = self._conn.execute(
            "SELECT run_hash, replicate, COUNT(DISTINCT run_id) AS n "
            "FROM cell_data "
            f"WHERE TRUE{and_hash} "
            "GROUP BY run_hash, replicate HAVING COUNT(DISTINCT run_id) > 1",
            hash_param,
        ).fetchall()
        for h, rep, n in rows:
            issues.append(ConsistencyIssue(
                "duplicate_replicate", h,
                f"run_hash {h} replicate {rep} loaded under {n} run_ids "
                f"(row inflation — drop_run() and re-ingest to fix).",
                "error",
            ))

        # ran_not_ingested: succeeded run with no cell_data
        rows = self._conn.execute(
            "SELECT DISTINCT r.run_hash FROM job_runs r "
            "WHERE r.exit_code = 0"
            f"{(' AND r.run_hash = ?' if run_hash else '')} "
            "AND NOT EXISTS (SELECT 1 FROM cell_data c WHERE c.run_hash = r.run_hash)",
            hash_param,
        ).fetchall()
        for (h,) in rows:
            issues.append(ConsistencyIssue(
                "ran_not_ingested", h,
                f"run_hash {h} has a succeeded run but no ingested cell_data "
                f"(run ahead of analysis — ingest results).",
                "warning",
            ))

        if strict and issues:
            summary = "\n".join(f"  [{i.severity}] {i.kind}: {i.detail}" for i in issues)
            raise RuntimeError(f"Registry consistency check failed:\n{summary}")
        return issues

    # ========== Output Tracking ==========

    def register_output(
        self,
        run_id: str,
        output_type: str,
        file_path: str,
        file_size: int | None = None,
        row_count: int | None = None,
    ) -> str:
        """Register an output file from a run.

        Args:
            run_id: The run this output belongs to.
            output_type: Type of output (e.g., 'csv', 'log', 'error').
            file_path: Path to the output file.
            file_size: Size of the file in bytes.
            row_count: Number of rows (for tabular data).

        Returns:
            The generated output ID.
        """
        output_id = _generate_id()

        self.conn.execute(
            """
            INSERT INTO run_outputs
            (output_id, run_id, output_type, file_path, file_size, row_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [output_id, run_id, output_type, file_path, file_size, row_count],
        )
        return output_id

    def get_debug_output_files(
        self,
        label_or_hash: str,
        *,
        run_id: str | None = None,
        entity_types: list[str] | None = None,
        existing_only: bool = True,
    ) -> list[Path]:
        """Get debug output file paths for a labeled/hashed run.

        Args:
            label_or_hash: Run label or run_hash.
            run_id: Optional explicit run execution ID. If omitted, uses the
                latest execution for the run hash.
            entity_types: Optional debug entity types to include (e.g.,
                ["organism", "patch"]). If omitted, includes all.
            existing_only: If True, return only paths that exist on disk.

        Returns:
            List of debug file paths for the selected run execution.

        Raises:
            KeyError: If the run or run execution is not found.
            ValueError: If the explicit run_id does not belong to the run hash.
        """
        run_hash = self._resolve_label_or_hash(label_or_hash)
        resolved_run_id = self._resolve_run_id_for_hash(run_hash, run_id=run_id)

        params: list[Any] = [resolved_run_id]
        where = ["run_id = ?", "output_type LIKE 'debug.%'"]

        if entity_types:
            wanted = [f"debug.{etype}" for etype in entity_types]
            placeholders = ", ".join(["?"] * len(wanted))
            where.append(f"output_type IN ({placeholders})")
            params.extend(wanted)

        rows = self.conn.execute(
            f"""
            SELECT output_type, file_path
            FROM run_outputs
            WHERE {' AND '.join(where)}
            ORDER BY output_type, file_path
            """,
            params,
        ).fetchall()

        seen: set[Path] = set()
        paths: list[Path] = []
        for _, file_path in rows:
            path = Path(file_path)
            if existing_only and not path.exists():
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)

        return paths

    def load_debug(
        self,
        label_or_hash: str,
        *,
        run_id: str | None = None,
        entity_types: list[str] | None = None,
        existing_only: bool = True,
    ) -> Any:
        """Load debug messages for a run from registered debug output files.

        Args:
            label_or_hash: Run label or run_hash.
            run_id: Optional explicit run execution ID. If omitted, uses latest.
            entity_types: Optional debug entity types to include.
            existing_only: If True, only load files that currently exist.

        Returns:
            DebugMessageStore with messages merged across all selected files.

        Raises:
            KeyError: If run/run execution is not found.
            ValueError: If no matching debug files are available.
            FileNotFoundError: If ``existing_only=False`` and any file is missing.
        """
        from joshpy.debug import DebugMessageStore, load_debug_file

        files = self.get_debug_output_files(
            label_or_hash,
            run_id=run_id,
            entity_types=entity_types,
            existing_only=existing_only,
        )
        if not files:
            raise ValueError(
                f"No debug output files found for '{label_or_hash}'. "
                "Ensure debugFiles are configured and outputs were registered."
            )

        store = DebugMessageStore()
        for path in files:
            file_store = load_debug_file(path)
            for msg in file_store.messages:
                store.add(msg)
            store.parse_errors += file_store.parse_errors
        return store

    # ========== Query Methods ==========

    def get_runs_by_parameters(self, **params: Any) -> list[dict[str, Any]]:
        """Query runs by parameter values.

        Args:
            **params: Parameter name-value pairs to filter by.

        Returns:
            List of dicts containing run info and parameters.
        """
        # Get run info first
        if not params:
            # No filters - return all runs
            result = self.conn.execute(
                """
                SELECT r.run_id, r.run_hash, r.replicate, r.started_at, r.completed_at,
                       r.exit_code, r.output_path, r.error_message
                FROM job_runs r
                ORDER BY r.started_at DESC
                """
            ).fetchall()
        else:
            # Build filter conditions using typed config_parameters columns
            conditions = []
            values = []
            for key, value in params.items():
                quoted_key = _quote_identifier(key)
                # Compare values appropriately - use the raw value for comparison
                # DuckDB handles type coercion for = comparisons
                conditions.append(f"cp.{quoted_key} = ?")
                values.append(value)

            where_clause = " AND ".join(conditions)
            result = self.conn.execute(
                f"""
                SELECT r.run_id, r.run_hash, r.replicate, r.started_at, r.completed_at,
                       r.exit_code, r.output_path, r.error_message
                FROM job_runs r
                JOIN config_parameters cp ON r.run_hash = cp.run_hash
                WHERE {where_clause}
                ORDER BY r.started_at DESC
                """,
                values,
            ).fetchall()

        # Build result with parameters from typed columns
        runs = []
        for row in result:
            run_hash = row[1]
            parameters = self._get_parameters_for_run_hash(run_hash)
            runs.append(
                {
                    "run_id": row[0],
                    "run_hash": run_hash,
                    "replicate": row[2],
                    "started_at": row[3],
                    "completed_at": row[4],
                    "exit_code": row[5],
                    "output_path": row[6],
                    "error_message": row[7],
                    "parameters": parameters,
                }
            )
        return runs

    def get_session_summary(self, session_id: str) -> SessionSummary | None:
        """Get aggregated statistics for a session.

        Args:
            session_id: The session ID to summarize.

        Returns:
            SessionSummary with counts, or None if session not found.
        """
        # Get session info
        session = self.get_session(session_id)
        if session is None:
            return None

        # Count configs (total_jobs) for this session via junction table
        configs_count = self.conn.execute(
            "SELECT COUNT(*) FROM session_configs WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]

        # Count runs by status (job_runs has session_id directly)
        result = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN completed_at IS NOT NULL THEN 1 END) as completed,
                COUNT(CASE WHEN exit_code = 0 THEN 1 END) as succeeded,
                COUNT(CASE WHEN exit_code IS NOT NULL AND exit_code != 0 THEN 1 END) as failed
            FROM job_runs
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()

        total_runs = result[0] if result else 0
        completed = result[1] if result else 0
        succeeded = result[2] if result else 0
        failed = result[3] if result else 0
        pending = total_runs - completed

        return SessionSummary(
            session_id=session_id,
            experiment_name=session.experiment_name,
            simulation=session.simulation,
            status=session.status,
            total_jobs=configs_count,
            total_replicates=total_runs,  # Use actual run count
            runs_completed=completed,
            runs_succeeded=succeeded,
            runs_failed=failed,
            runs_pending=pending,
        )

    def export_results_df(self, session_id: str) -> Any:
        """Export session results as a pandas DataFrame.

        Args:
            session_id: The session to export.

        Returns:
            pandas DataFrame with run results and parameters.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError(
                "pandas is required for export_results_df. Install with: pip install pandas"
            ) from err

        # Query all runs for this session
        result = self.conn.execute(
            """
            SELECT run_id, run_hash, replicate, started_at, completed_at,
                   exit_code, output_path, error_message
            FROM job_runs
            WHERE session_id = ?
            ORDER BY started_at
            """,
            [session_id],
        ).fetchall()

        rows = []
        for row in result:
            run_hash = row[1]
            params = self._get_parameters_for_run_hash(run_hash)
            row_dict = {
                "run_id": row[0],
                "run_hash": run_hash,
                "replicate": row[2],
                "started_at": row[3],
                "completed_at": row[4],
                "exit_code": row[5],
                "output_path": row[6],
                "error_message": row[7],
                **params,  # Flatten parameters into columns
            }
            rows.append(row_dict)

        return pd.DataFrame(rows)

    # ========== Discovery Methods ==========

    def list_variable_columns(self) -> list[str]:
        """List all variable column names in cell_data.

        Returns the dynamically-added variable columns. Column names preserve
        original names with special characters (e.g., 'avg.height').

        Returns:
            Sorted list of variable column names.

        Examples:
            >>> registry.list_variable_columns()
            ['averageAge', 'avg.height', 'treeCount']
        """
        result = self.conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'cell_data'
            ORDER BY column_name
            """
        ).fetchall()

        # Filter out core columns, return only variable columns
        all_cols = [row[0] for row in result]
        return sorted([c for c in all_cols if c not in CELL_DATA_CORE_COLUMNS])

    def list_export_variables(self, session_id: str | None = None) -> list[str]:
        """List all export variable names from simulation outputs.

        These are the variables exported by Josh simulations,
        stored as typed columns in the cell_data table. Variable names
        preserve original .josh names (e.g., 'avg.height').

        When session_id is provided, only returns variables that have at least
        one non-NULL value for runs in that session.

        Args:
            session_id: Optional session ID to filter by. If provided, only
                        returns variables with data in that session.

        Returns:
            Sorted list of variable column names.

        Examples:
            >>> registry.list_export_variables()
            ['averageAge', 'avg.height', 'treeCount']

            >>> registry.list_export_variables(session_id="abc123")
            ['treeCount']  # Only variables with data in this session
        """
        all_vars = self.list_variable_columns()

        if not session_id or not all_vars:
            return all_vars

        # Filter to variables that have non-NULL values in this session
        # Build a query that checks each column for non-NULL values
        vars_with_data = []
        for var_name in all_vars:
            quoted = _quote_identifier(var_name)
            result = self.conn.execute(
                f"""
                SELECT 1
                FROM cell_data cd
                JOIN session_configs sc ON cd.run_hash = sc.run_hash
                WHERE sc.session_id = ? AND {quoted} IS NOT NULL
                LIMIT 1
                """,
                [session_id],
            ).fetchone()
            if result:
                vars_with_data.append(var_name)

        return sorted(vars_with_data)

    # Alias for backward compatibility
    def list_variables(self, session_id: str | None = None) -> list[str]:
        """Alias for list_export_variables(). Deprecated, use list_export_variables()."""
        return self.list_export_variables(session_id)

    def _ensure_variable_columns(self, columns: dict[str, str]) -> None:
        """Ensure variable columns exist in cell_data table.

        Adds missing columns with the specified types. This is called by
        CellDataLoader when loading CSVs with new variables.

        Column names are preserved exactly as provided (with quotes for SQL).
        For example, 'avg.height' becomes column "avg.height".

        Args:
            columns: Dict mapping column name (original) to SQL type
                     (either 'DOUBLE' or 'VARCHAR').

        Raises:
            ValueError: If trying to add a column that exists with a different type.
        """
        existing = self.list_variable_columns()

        for col_name, col_type in columns.items():
            if col_name in existing:
                # Verify type matches
                type_result = self.conn.execute(
                    """
                    SELECT data_type
                    FROM information_schema.columns
                    WHERE table_name = 'cell_data' AND column_name = ?
                    """,
                    [col_name],
                ).fetchone()

                if type_result:
                    existing_type = type_result[0].upper()
                    requested_type = col_type.upper()
                    # Normalize type names for comparison
                    if existing_type in ("DOUBLE", "FLOAT", "REAL"):
                        existing_type = "DOUBLE"
                    if requested_type in ("DOUBLE", "FLOAT", "REAL"):
                        requested_type = "DOUBLE"

                    if existing_type != requested_type:
                        raise ValueError(
                            f"Column '{col_name}' exists as {existing_type} but "
                            f"new data has {requested_type}. This may indicate "
                            f"mixed simulation types. Use a separate registry."
                        )
            else:
                # Add new column with quoted identifier
                quoted = _quote_identifier(col_name)
                self.conn.execute(f"ALTER TABLE cell_data ADD COLUMN {quoted} {col_type}")

    def check_sparsity(self) -> SparsityReport:
        """Check for sparse columns in cell_data.

        Sparse columns (>50% NULL by default) often indicate that different
        simulation types are being mixed in the same registry, which hurts
        query performance.

        Returns:
            SparsityReport with statistics for each variable column.

        Examples:
            >>> report = registry.check_sparsity()
            >>> if report.should_warn:
            ...     print(report)
        """
        # Get total row count
        total_result = self.conn.execute("SELECT COUNT(*) FROM cell_data").fetchone()
        total_rows = total_result[0] if total_result else 0

        if total_rows == 0:
            return SparsityReport(total_rows=0, column_stats=[])

        # Get stats for each variable column
        variable_cols = self.list_variable_columns()
        column_stats = []

        for col_name in variable_cols:
            # Get column type
            type_result = self.conn.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = 'cell_data' AND column_name = ?
                """,
                [col_name],
            ).fetchone()

            dtype = type_result[0] if type_result else "UNKNOWN"

            # Count NULLs
            null_result = self.conn.execute(
                f'SELECT COUNT(*) FROM cell_data WHERE "{col_name}" IS NULL'
            ).fetchone()
            null_count = null_result[0] if null_result else 0

            column_stats.append(
                ColumnStats(
                    name=col_name,
                    dtype=dtype,
                    total_rows=total_rows,
                    null_count=null_count,
                )
            )

        return SparsityReport(
            total_rows=total_rows,
            column_stats=column_stats,
            threshold_percent=SPARSITY_WARN_COLUMN_NULL_PERCENT,
        )

    def list_config_columns(self) -> list[str]:
        """List all parameter column names in config_parameters.

        Returns the dynamically-added parameter columns. Column names preserve
        original names with special characters (e.g., 'soil.moisture').

        Returns:
            Sorted list of parameter column names.

        Examples:
            >>> registry.list_config_columns()
            ['maxGrowth', 'scenario', 'soil.moisture']
        """
        result = self.conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'config_parameters'
            ORDER BY column_name
            """
        ).fetchall()

        # Filter out core columns, return only parameter columns
        all_cols = [row[0] for row in result]
        return sorted([c for c in all_cols if c not in CONFIG_PARAMS_CORE_COLUMNS])

    def _ensure_config_columns(self, columns: dict[str, str]) -> None:
        """Ensure parameter columns exist in config_parameters table.

        Adds missing columns with the specified types. This is called by
        register_run() when registering configs with parameters.

        Column names are preserved exactly as provided (with quotes for SQL).
        For example, 'soil.moisture' becomes column "soil.moisture".

        Args:
            columns: Dict mapping column name (original) to SQL type
                     (either 'DOUBLE' or 'VARCHAR').

        Raises:
            ValueError: If trying to add a column that exists with a different type.
        """
        existing = self.list_config_columns()

        for col_name, col_type in columns.items():
            if col_name in existing:
                # Verify type matches
                type_result = self.conn.execute(
                    """
                    SELECT data_type
                    FROM information_schema.columns
                    WHERE table_name = 'config_parameters' AND column_name = ?
                    """,
                    [col_name],
                ).fetchone()

                if type_result:
                    existing_type = type_result[0].upper()
                    requested_type = col_type.upper()
                    # Normalize type names for comparison
                    if existing_type in ("DOUBLE", "FLOAT", "REAL"):
                        existing_type = "DOUBLE"
                    if requested_type in ("DOUBLE", "FLOAT", "REAL"):
                        requested_type = "DOUBLE"

                    if existing_type != requested_type:
                        raise ValueError(
                            f"Config parameter '{col_name}' exists as {existing_type} but "
                            f"new data has {requested_type}. This may indicate "
                            f"mixed sweep configurations. Use a separate registry."
                        )
            else:
                # Add new column with quoted identifier
                quoted = _quote_identifier(col_name)
                self.conn.execute(f"ALTER TABLE config_parameters ADD COLUMN {quoted} {col_type}")

    def list_config_parameters(self, session_id: str | None = None) -> list[str]:
        """List all config parameter names from sweep configurations.

        These are the parameters you defined in your JobConfig sweep,
        stored as typed columns in the config_parameters table.

        Args:
            session_id: Optional session ID to filter by.

        Returns:
            Sorted list of parameter names.

        Examples:
            >>> registry.list_config_parameters()
            ['maxGrowth', 'scenario', 'survivalProb']
        """
        all_params = self.list_config_columns()

        if not session_id or not all_params:
            return all_params

        # Filter to parameters that have non-NULL values in this session
        params_with_data = []
        for param_name in all_params:
            quoted = _quote_identifier(param_name)
            result = self.conn.execute(
                f"""
                SELECT 1
                FROM config_parameters cp
                JOIN session_configs sc ON cp.run_hash = sc.run_hash
                WHERE sc.session_id = ? AND {quoted} IS NOT NULL
                LIMIT 1
                """,
                [session_id],
            ).fetchone()
            if result:
                params_with_data.append(param_name)

        return sorted(params_with_data)

    # Alias for backward compatibility
    def list_parameters(self, session_id: str | None = None) -> list[str]:
        """Alias for list_config_parameters(). Deprecated, use list_config_parameters()."""
        return self.list_config_parameters(session_id)

    def list_entity_types(self, session_id: str | None = None) -> list[str]:
        """List all entity types found in cell_data.

        Args:
            session_id: Optional session ID to filter by.

        Returns:
            Sorted list of entity type names.
        """
        if session_id:
            result = self.conn.execute(
                """
                SELECT DISTINCT entity_type
                FROM cell_data cd
                JOIN session_configs sc ON cd.run_hash = sc.run_hash
                WHERE sc.session_id = ? AND entity_type IS NOT NULL
                ORDER BY entity_type
                """,
                [session_id],
            ).fetchall()
        else:
            result = self.conn.execute(
                """
                SELECT DISTINCT entity_type
                FROM cell_data
                WHERE entity_type IS NOT NULL
                ORDER BY entity_type
                """
            ).fetchall()

        return [row[0] for row in result]

    def get_data_summary(self, session_id: str | None = None) -> DataSummary:
        """Get summary of all data in registry.

        Provides counts, available variables, parameters, and data ranges
        for diagnostic purposes.

        Args:
            session_id: Optional session ID to filter by.

        Returns:
            DataSummary with counts and metadata.
        """
        # Get counts (use session_configs junction table for session filtering)
        if session_id:
            sessions_count = 1
            configs_count = self.conn.execute(
                "SELECT COUNT(*) FROM session_configs WHERE session_id = ?",
                [session_id],
            ).fetchone()[0]
            runs_count = self.conn.execute(
                "SELECT COUNT(*) FROM job_runs WHERE session_id = ?",
                [session_id],
            ).fetchone()[0]
            rows_count = self.conn.execute(
                """
                SELECT COUNT(*) FROM cell_data cd
                JOIN session_configs sc ON cd.run_hash = sc.run_hash
                WHERE sc.session_id = ?
                """,
                [session_id],
            ).fetchone()[0]
        else:
            sessions_count = self.conn.execute("SELECT COUNT(*) FROM sweep_sessions").fetchone()[0]
            configs_count = self.conn.execute("SELECT COUNT(*) FROM job_configs").fetchone()[0]
            runs_count = self.conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
            rows_count = self.conn.execute("SELECT COUNT(*) FROM cell_data").fetchone()[0]

        # Get variables, parameters, entity types
        variables = self.list_export_variables(session_id)
        parameters = self.list_config_parameters(session_id)
        entity_types = self.list_entity_types(session_id)

        # Get step/replicate ranges
        if session_id:
            range_result = self.conn.execute(
                """
                SELECT MIN(step), MAX(step), MIN(replicate), MAX(replicate)
                FROM cell_data cd
                JOIN session_configs sc ON cd.run_hash = sc.run_hash
                WHERE sc.session_id = ?
                """,
                [session_id],
            ).fetchone()
        else:
            range_result = self.conn.execute(
                "SELECT MIN(step), MAX(step), MIN(replicate), MAX(replicate) FROM cell_data"
            ).fetchone()

        step_range = None
        replicate_range = None
        if range_result and range_result[0] is not None:
            step_range = (range_result[0], range_result[1])
            replicate_range = (range_result[2], range_result[3])

        # Get spatial extent
        if session_id:
            spatial_result = self.conn.execute(
                """
                SELECT MIN(longitude), MAX(longitude), MIN(latitude), MAX(latitude)
                FROM cell_data cd
                JOIN session_configs sc ON cd.run_hash = sc.run_hash
                WHERE sc.session_id = ? AND longitude IS NOT NULL
                """,
                [session_id],
            ).fetchone()
        else:
            spatial_result = self.conn.execute(
                """
                SELECT MIN(longitude), MAX(longitude), MIN(latitude), MAX(latitude)
                FROM cell_data
                WHERE longitude IS NOT NULL
                """
            ).fetchone()

        spatial_extent = None
        if spatial_result and spatial_result[0] is not None:
            spatial_extent = {
                "lon": (spatial_result[0], spatial_result[1]),
                "lat": (spatial_result[2], spatial_result[3]),
            }

        return DataSummary(
            sessions=sessions_count,
            configs=configs_count,
            runs=runs_count,
            cell_data_rows=rows_count,
            variables=variables,
            entity_types=entity_types,
            step_range=step_range,
            replicate_range=replicate_range,
            spatial_extent=spatial_extent,
            parameters=parameters,
        )


@dataclass
class RegistryCallback:
    """Helper for recording CLI results in the registry.

    This class helps record the results of CLI executions in the registry,
    tracking run starts and completions.

    Attributes:
        registry: The RunRegistry to record runs in.
        session_id: The session ID for the current sweep.

    Examples:
        >>> from joshpy.cli import JoshCLI
        >>> from joshpy.jobs import JobExpander, to_run_config
        >>> registry = RunRegistry("experiment.duckdb")
        >>> session_id = registry.create_session(...)
        >>> callback = RegistryCallback(registry, session_id)
        >>> cli = JoshCLI()
        >>> for job in job_set:
        ...     run_config = to_run_config(job)
        ...     result = cli.run(run_config)
        ...     callback.record(job, result)
    """

    registry: RunRegistry
    session_id: str

    def record(self, job: Any, result: Any) -> str:
        """Record a job execution result in the registry.

        Args:
            job: ExpandedJob that was executed.
            result: CLIResult from the CLI execution.

        Returns:
            The run_id for the recorded run.
        """
        # Import here to avoid circular dependency
        from joshpy.cli import CLIResult
        from joshpy.jobs import ExpandedJob

        if not isinstance(job, ExpandedJob):
            raise TypeError(f"Expected ExpandedJob, got {type(job)}")
        if not isinstance(result, CLIResult):
            raise TypeError(f"Expected CLIResult, got {type(result)}")

        # Create run record (records both start and completion)
        run_id = self.registry.start_run(
            run_hash=job.run_hash,
            session_id=self.session_id,
            replicate=0,  # CLI runs all replicates at once
            output_path=str(job.config_path.parent) if job.config_path else None,
            metadata={
                "parameters": job.parameters,
                "replicates": job.replicates,
            },
        )

        # Complete the run with the result
        error_msg = result.stderr if not result.success else None
        self.registry.complete_run(
            run_id=run_id,
            exit_code=result.exit_code,
            error_message=error_msg,
        )

        return run_id
