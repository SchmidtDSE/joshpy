"""DuckDB-backed registry for tracking parameter sweeps, job configurations, and run results.

This module provides a persistent registry for tracking Josh simulation experiments,
enabling users to:
- Track experiments by name (e.g., experiment_name="jotr_sensitivity")
- Link configs to their MD5 hashes for easy lookup
- Query runs by parameter values
- Get session summaries with success/failure counts

Example usage:
    from joshpy.jobs import JobExpander, JobRunner
    from joshpy.registry import RunRegistry, RegistryCallback

    # Setup
    registry = RunRegistry("experiment.duckdb")
    session_id = registry.create_session(
        experiment_name="jotr_sensitivity",
        simulation="JoshuaTreeSim",
        total_jobs=len(job_set.jobs),
        total_replicates=sum(j.replicates for j in job_set.jobs),
    )

    # Register configs
    for job in job_set.jobs:
        registry.register_run(session_id, job.run_hash, str(job.source_path), job.config_content, None, job.parameters)

    # Run with tracking
    callback = RegistryCallback(registry, session_id)
    results = runner.run_all(job_set, on_complete=callback)

    registry.update_session_status(session_id, "completed")
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _check_duckdb() -> None:
    """Raise ImportError if duckdb is not available."""
    if not HAS_DUCKDB:
        raise ImportError(
            "duckdb is required for the registry module. Install with: pip install joshpy[registry]"
        )


# Sparsity warning thresholds (configurable module globals)
SPARSITY_WARN_COLUMN_NULL_PERCENT = 50  # Warn if column >50% NULL
SPARSITY_WARN_MIN_SPARSE_COLUMNS = 2    # Only warn if >=2 columns are sparse
SPARSITY_WARN_MIN_ROWS = 1000           # Don't warn for tiny datasets

# Core columns that exist in cell_data schema (not variable columns)
CELL_DATA_CORE_COLUMNS = frozenset({
    "cell_id", "run_id", "run_hash", "step", "replicate",
    "position_x", "position_y", "longitude", "latitude",
    "entity_type", "geom",
})


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


# SQL Schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sweep_sessions (
    session_id      VARCHAR PRIMARY KEY,
    experiment_name VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    template_path   VARCHAR,
    template_hash   VARCHAR(12),
    simulation      VARCHAR,
    total_jobs      INTEGER,
    total_replicates INTEGER,
    status          VARCHAR DEFAULT 'pending',
    metadata        JSON
);

CREATE TABLE IF NOT EXISTS job_configs (
    run_hash        VARCHAR(12) PRIMARY KEY,
    session_id      VARCHAR REFERENCES sweep_sessions(session_id),
    josh_path       TEXT,
    config_content  TEXT,
    file_mappings   JSON,
    parameters      JSON,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_runs (
    run_id          VARCHAR PRIMARY KEY,
    run_hash        VARCHAR(12) REFERENCES job_configs(run_hash),
    replicate       INTEGER,
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    exit_code       INTEGER,
    output_path     VARCHAR,
    error_message   TEXT,
    metadata        JSON
);

CREATE TABLE IF NOT EXISTS run_outputs (
    output_id       VARCHAR PRIMARY KEY,
    run_id          VARCHAR REFERENCES job_runs(run_id),
    output_type     VARCHAR,
    file_path       VARCHAR,
    file_size       BIGINT,
    row_count       INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS cell_id_seq START 1;

CREATE TABLE IF NOT EXISTS cell_data (
    cell_id         BIGINT PRIMARY KEY DEFAULT nextval('cell_id_seq'),
    run_id          VARCHAR REFERENCES job_runs(run_id),
    run_hash        VARCHAR(12),
    step            INTEGER NOT NULL,
    replicate       INTEGER NOT NULL,
    position_x      DOUBLE,
    position_y      DOUBLE,
    longitude       DOUBLE,
    latitude        DOUBLE,
    entity_type     VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_cell_run ON cell_data(run_id);
CREATE INDEX IF NOT EXISTS idx_cell_run_hash ON cell_data(run_hash);
CREATE INDEX IF NOT EXISTS idx_cell_step ON cell_data(step);
CREATE INDEX IF NOT EXISTS idx_cell_replicate ON cell_data(replicate);
CREATE INDEX IF NOT EXISTS idx_cell_spatial ON cell_data(longitude, latitude);
CREATE INDEX IF NOT EXISTS idx_cell_step_replicate ON cell_data(step, replicate);
"""


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

        Example:
            session = registry.get_session(session_id)
            if session.job_config:
                # Re-expand jobs for execution
                job_set = JobExpander().expand(session.job_config)
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
        config_content: Full text content of the configuration.
        file_mappings: Dict mapping names to {"path": "...", "hash": "..."}.
        parameters: Parameter values used to generate this config.
        created_at: When the config was registered.
    """

    run_hash: str
    session_id: str
    josh_path: str | None
    config_content: str
    file_mappings: dict[str, dict[str, str]] | None
    parameters: dict[str, Any]
    created_at: datetime


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

    Example:
        # File-based (persistent)
        registry = RunRegistry("experiment.duckdb")

        # In-memory (for testing)
        registry = RunRegistry(":memory:")

        # With spatial support enabled (default)
        registry = RunRegistry("experiment.duckdb", enable_spatial=True)

        # Context manager
        with RunRegistry("experiment.duckdb") as registry:
            session_id = registry.create_session(...)
    """

    db_path: Path | str = "josh_runs.duckdb"
    enable_spatial: bool = True
    _conn: Any = field(default=None, repr=False)

    # Filter state for context managers
    _spatial_filter_bbox: tuple[float, float, float, float] | None = field(
        default=None, repr=False
    )
    _spatial_filter_geojson: str | dict | None = field(default=None, repr=False)
    _time_filter_range: tuple[int, int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize database connection and schema."""
        _check_duckdb()
        if self._conn is None:
            db_str = str(self.db_path)
            self._conn = duckdb.connect(db_str)
            self._init_schema()
            if self.enable_spatial:
                self._init_spatial()

    @property
    def conn(self) -> Any:
        """Direct access to DuckDB connection for custom queries.

        Returns:
            DuckDB connection object.

        Example:
            df = registry.conn.execute("SELECT * FROM cell_data LIMIT 10").df()
        """
        return self._conn

    def _init_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        self._conn.execute(SCHEMA_SQL)

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
                self._conn.execute(
                    "ALTER TABLE cell_data ADD COLUMN geom GEOMETRY;"
                )
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

        Example:
            # Get DataFrame
            df = registry.query(
                "SELECT * FROM cell_data WHERE step BETWEEN ? AND ?",
                [0, 10]
            ).df()

            # Get raw results
            rows = registry.query(
                "SELECT COUNT(*) FROM cell_data WHERE run_hash = ?",
                ["abc123"]
            ).fetchone()
        """
        return self._conn.execute(sql, params or [])

    def to_parquet(self, path: str | Path, table: str = "cell_data") -> None:
        """Export a table to Parquet format.

        Parquet is recommended for R/Python analysis due to compression
        and type preservation.

        Args:
            path: Output file path.
            table: Table name to export (default: cell_data).

        Example:
            registry.to_parquet("results.parquet")

            # In R:
            # df <- arrow::read_parquet("results.parquet")

            # In Python:
            # import pandas as pd
            # df = pd.read_parquet("results.parquet")
        """
        path_str = str(path)
        self._conn.execute(f"COPY {table} TO '{path_str}' (FORMAT PARQUET)")

    def to_csv(self, path: str | Path, table: str = "cell_data") -> None:
        """Export a table to CSV format.

        Args:
            path: Output file path.
            table: Table name to export (default: cell_data).

        Example:
            registry.to_csv("results.csv")

            # In R:
            # df <- readr::read_csv("results.csv")
        """
        path_str = str(path)
        self._conn.execute(f"COPY {table} TO '{path_str}' (FORMAT CSV, HEADER)")

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

        Example:
            with registry.spatial_filter(bbox=(-116, -115, 33.5, 34.0)):
                df = queries.get_timeseries("height", run_hash="abc123")

            # Nested with time filter
            with registry.spatial_filter(geojson=park_boundary):
                with registry.time_filter(step_range=(0, 50)):
                    df = queries.get_timeseries("height", run_hash="abc123")
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

        Example:
            with registry.time_filter(step_range=(0, 50)):
                df = queries.get_timeseries("height", run_hash="abc123")

            # Nested with spatial filter
            with registry.time_filter(step_range=(10, 20)):
                with registry.spatial_filter(bbox=(-116, -115, 33.5, 34.0)):
                    df = queries.get_comparison("height", group_by="maxGrowth")
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
        experiment_name: str | None = None,
        simulation: str | None = None,
        template_path: str | None = None,
        template_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create a new sweep session.

        Args:
            experiment_name: Name for the experiment (used in path templates).
            simulation: Name of the simulation being run.
            template_path: Path to the Jinja template file.
            template_hash: Hash of the template content.
            metadata: Additional metadata dictionary.
            session_id: Optional externally-provided session ID.
                        If None, generates a UUID. This allows the frontend/API
                        layer to manage session IDs (e.g., using project IDs).

        Returns:
            The session ID (generated or provided).

        Note:
            total_jobs and total_replicates are computed from the JobSet
            after job expansion. Use job_set.total_jobs and job_set.total_replicates.
        """
        if session_id is None:
            session_id = _generate_id()
        metadata_json = json.dumps(metadata) if metadata else None

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
                template_path,
                template_hash,
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
    ) -> None:
        """Register a job configuration (run specification).

        Args:
            session_id: Session this config belongs to.
            run_hash: MD5 hash of josh + config + file_mappings (12 chars).
            josh_path: Path to the .josh script file.
            config_content: Full text of the rendered configuration.
            file_mappings: Dict mapping names to {"path": "...", "hash": "..."}.
            parameters: Parameter values used to generate this config.
        """
        parameters_json = json.dumps(parameters)
        file_mappings_json = json.dumps(file_mappings) if file_mappings else None

        # Use INSERT OR IGNORE to handle duplicate run hashes
        self.conn.execute(
            """
            INSERT OR IGNORE INTO job_configs
            (run_hash, session_id, josh_path, config_content, file_mappings, parameters)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [run_hash, session_id, josh_path, config_content, file_mappings_json, parameters_json],
        )

    def get_config_by_hash(self, run_hash: str) -> ConfigInfo | None:
        """Get config information by run hash.

        Args:
            run_hash: The run hash to look up.

        Returns:
            ConfigInfo if found, None otherwise.
        """
        result = self.conn.execute(
            """
            SELECT run_hash, session_id, josh_path, config_content, file_mappings, parameters, created_at
            FROM job_configs
            WHERE run_hash = ?
            """,
            [run_hash],
        ).fetchone()

        if result is None:
            return None

        return ConfigInfo(
            run_hash=result[0],
            session_id=result[1],
            josh_path=result[2],
            config_content=result[3],
            file_mappings=json.loads(result[4]) if result[4] else None,
            parameters=json.loads(result[5]) if result[5] else {},
            created_at=result[6],
        )

    def get_configs_for_session(self, session_id: str) -> list[ConfigInfo]:
        """Get all configs for a session.

        Args:
            session_id: The session ID to get configs for.

        Returns:
            List of ConfigInfo objects.
        """
        result = self.conn.execute(
            """
            SELECT run_hash, session_id, josh_path, config_content, file_mappings, parameters, created_at
            FROM job_configs
            WHERE session_id = ?
            ORDER BY created_at
            """,
            [session_id],
        ).fetchall()

        return [
            ConfigInfo(
                run_hash=row[0],
                session_id=row[1],
                josh_path=row[2],
                config_content=row[3],
                file_mappings=json.loads(row[4]) if row[4] else None,
                parameters=json.loads(row[5]) if row[5] else {},
                created_at=row[6],
            )
            for row in result
        ]

    # ========== Run Tracking ==========

    def start_run(
        self,
        run_hash: str,
        replicate: int = 0,
        output_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the start of a job run.

        Args:
            run_hash: Run hash for this run.
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
            (run_id, run_hash, replicate, started_at, output_path, metadata)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """,
            [run_id, run_hash, replicate, output_path, metadata_json],
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

    # ========== Query Methods ==========

    def get_runs_by_parameters(self, **params: Any) -> list[dict[str, Any]]:
        """Query runs by parameter values.

        Args:
            **params: Parameter name-value pairs to filter by.

        Returns:
            List of dicts containing run info and parameters.
        """
        if not params:
            # No filters - return all runs with their configs
            result = self.conn.execute(
                """
                SELECT r.run_id, r.run_hash, r.replicate, r.started_at, r.completed_at,
                       r.exit_code, r.output_path, r.error_message, c.parameters
                FROM job_runs r
                JOIN job_configs c ON r.run_hash = c.run_hash
                ORDER BY r.started_at DESC
                """
            ).fetchall()
        else:
            # Build filter conditions using JSON extraction
            conditions = []
            values = []
            for key, value in params.items():
                # Use DuckDB JSON extraction with type-appropriate comparison
                conditions.append(f"json_extract_string(c.parameters, '$.{key}') = ?")
                values.append(str(value))

            where_clause = " AND ".join(conditions)
            result = self.conn.execute(
                f"""
                SELECT r.run_id, r.run_hash, r.replicate, r.started_at, r.completed_at,
                       r.exit_code, r.output_path, r.error_message, c.parameters
                FROM job_runs r
                JOIN job_configs c ON r.run_hash = c.run_hash
                WHERE {where_clause}
                ORDER BY r.started_at DESC
                """,
                values,
            ).fetchall()

        return [
            {
                "run_id": row[0],
                "run_hash": row[1],
                "replicate": row[2],
                "started_at": row[3],
                "completed_at": row[4],
                "exit_code": row[5],
                "output_path": row[6],
                "error_message": row[7],
                "parameters": json.loads(row[8]) if row[8] else {},
            }
            for row in result
        ]

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

        # Count configs (total_jobs) for this session
        configs_count = self.conn.execute(
            "SELECT COUNT(*) FROM job_configs WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]

        # Count runs by status
        result = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN completed_at IS NOT NULL THEN 1 END) as completed,
                COUNT(CASE WHEN exit_code = 0 THEN 1 END) as succeeded,
                COUNT(CASE WHEN exit_code IS NOT NULL AND exit_code != 0 THEN 1 END) as failed
            FROM job_runs r
            JOIN job_configs c ON r.run_hash = c.run_hash
            WHERE c.session_id = ?
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

        # Query all runs for this session with their parameters
        result = self.conn.execute(
            """
            SELECT r.run_id, r.run_hash, r.replicate, r.started_at, r.completed_at,
                   r.exit_code, r.output_path, r.error_message, c.parameters
            FROM job_runs r
            JOIN job_configs c ON r.run_hash = c.run_hash
            WHERE c.session_id = ?
            ORDER BY r.started_at
            """,
            [session_id],
        ).fetchall()

        rows = []
        for row in result:
            params = json.loads(row[8]) if row[8] else {}
            row_dict = {
                "run_id": row[0],
                "run_hash": row[1],
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
            
        Example:
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

        Example:
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
                JOIN job_configs jc ON cd.run_hash = jc.run_hash
                WHERE jc.session_id = ? AND {quoted} IS NOT NULL
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
                    if existing_type in ('DOUBLE', 'FLOAT', 'REAL'):
                        existing_type = 'DOUBLE'
                    if requested_type in ('DOUBLE', 'FLOAT', 'REAL'):
                        requested_type = 'DOUBLE'
                    
                    if existing_type != requested_type:
                        raise ValueError(
                            f"Column '{col_name}' exists as {existing_type} but "
                            f"new data has {requested_type}. This may indicate "
                            f"mixed simulation types. Use a separate registry."
                        )
            else:
                # Add new column with quoted identifier
                quoted = _quote_identifier(col_name)
                self.conn.execute(
                    f"ALTER TABLE cell_data ADD COLUMN {quoted} {col_type}"
                )
    
    def check_sparsity(self) -> SparsityReport:
        """Check for sparse columns in cell_data.
        
        Sparse columns (>50% NULL by default) often indicate that different
        simulation types are being mixed in the same registry, which hurts
        query performance.
        
        Returns:
            SparsityReport with statistics for each variable column.
            
        Example:
            >>> report = registry.check_sparsity()
            >>> if report.should_warn:
            ...     print(report)
        """
        # Get total row count
        total_result = self.conn.execute(
            "SELECT COUNT(*) FROM cell_data"
        ).fetchone()
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
            
            dtype = type_result[0] if type_result else 'UNKNOWN'
            
            # Count NULLs
            null_result = self.conn.execute(
                f"SELECT COUNT(*) FROM cell_data WHERE \"{col_name}\" IS NULL"
            ).fetchone()
            null_count = null_result[0] if null_result else 0
            
            column_stats.append(ColumnStats(
                name=col_name,
                dtype=dtype,
                total_rows=total_rows,
                null_count=null_count,
            ))
        
        return SparsityReport(
            total_rows=total_rows,
            column_stats=column_stats,
            threshold_percent=SPARSITY_WARN_COLUMN_NULL_PERCENT,
        )

    def list_config_parameters(self, session_id: str | None = None) -> list[str]:
        """List all config parameter names from sweep configurations.

        These are the parameters you defined in your JobConfig sweep,
        stored in the job_configs table's parameters JSON column.

        Args:
            session_id: Optional session ID to filter by.

        Returns:
            Sorted list of parameter names.

        Example:
            >>> registry.list_config_parameters()
            ['maxGrowth', 'scenario', 'survivalProb']
        """
        if session_id:
            result = self.conn.execute(
                """
                SELECT DISTINCT unnest(json_keys(parameters)) as param_name
                FROM job_configs
                WHERE session_id = ?
                ORDER BY param_name
                """,
                [session_id],
            ).fetchall()
        else:
            result = self.conn.execute(
                """
                SELECT DISTINCT unnest(json_keys(parameters)) as param_name
                FROM job_configs
                ORDER BY param_name
                """
            ).fetchall()

        return [row[0] for row in result]

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
                JOIN job_configs jc ON cd.run_hash = jc.run_hash
                WHERE jc.session_id = ? AND entity_type IS NOT NULL
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
        # Get counts
        if session_id:
            sessions_count = 1
            configs_count = self.conn.execute(
                "SELECT COUNT(*) FROM job_configs WHERE session_id = ?",
                [session_id],
            ).fetchone()[0]
            runs_count = self.conn.execute(
                """
                SELECT COUNT(*) FROM job_runs r
            JOIN job_configs c ON r.run_hash = c.run_hash
            WHERE c.session_id = ?
                """,
                [session_id],
            ).fetchone()[0]
            rows_count = self.conn.execute(
                """
                SELECT COUNT(*) FROM cell_data cd
                JOIN job_configs jc ON cd.run_hash = jc.run_hash
                WHERE jc.session_id = ?
                """,
                [session_id],
            ).fetchone()[0]
        else:
            sessions_count = self.conn.execute(
                "SELECT COUNT(*) FROM sweep_sessions"
            ).fetchone()[0]
            configs_count = self.conn.execute(
                "SELECT COUNT(*) FROM job_configs"
            ).fetchone()[0]
            runs_count = self.conn.execute(
                "SELECT COUNT(*) FROM job_runs"
            ).fetchone()[0]
            rows_count = self.conn.execute(
                "SELECT COUNT(*) FROM cell_data"
            ).fetchone()[0]

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
                JOIN job_configs jc ON cd.run_hash = jc.run_hash
                WHERE jc.session_id = ?
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
                JOIN job_configs jc ON cd.run_hash = jc.run_hash
                WHERE jc.session_id = ? AND longitude IS NOT NULL
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

    Example:
        from joshpy.cli import JoshCLI
        from joshpy.jobs import JobExpander, to_run_config

        registry = RunRegistry("experiment.duckdb")
        session_id = registry.create_session(...)

        callback = RegistryCallback(registry, session_id)
        cli = JoshCLI()

        for job in job_set:
            run_config = to_run_config(job)
            result = cli.run(run_config)
            callback.record(job, result)
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
            replicate=0,  # CLI runs all replicates at once
            output_path=str(job.config_path.parent) if job.config_path else None,
            metadata={"parameters": job.parameters},
        )

        # Complete the run with the result
        error_msg = result.stderr if not result.success else None
        self.registry.complete_run(
            run_id=run_id,
            exit_code=result.exit_code,
            error_message=error_msg,
        )

        return run_id
