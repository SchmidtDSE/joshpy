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
        registry.register_config(session_id, job.config_hash, job.config_content, job.parameters)

    # Run with tracking
    callback = RegistryCallback(registry, session_id)
    results = runner.run_all(job_set, on_complete=callback)

    registry.update_session_status(session_id, "completed")
"""

from __future__ import annotations

import json
import uuid
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
    config_hash     VARCHAR(12) PRIMARY KEY,
    session_id      VARCHAR REFERENCES sweep_sessions(session_id),
    config_content  TEXT,
    parameters      JSON,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_runs (
    run_id          VARCHAR PRIMARY KEY,
    config_hash     VARCHAR(12) REFERENCES job_configs(config_hash),
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
    config_hash     VARCHAR(12),
    step            INTEGER NOT NULL,
    replicate       INTEGER NOT NULL,
    position_x      DOUBLE,
    position_y      DOUBLE,
    longitude       DOUBLE,
    latitude        DOUBLE,
    entity_type     VARCHAR,
    variables       JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cell_run ON cell_data(run_id);
CREATE INDEX IF NOT EXISTS idx_cell_config ON cell_data(config_hash);
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


@dataclass
class ConfigInfo:
    """Information about a job configuration.

    Attributes:
        config_hash: MD5 hash of the config content (12 chars).
        session_id: Session this config belongs to.
        config_content: Full text content of the configuration.
        parameters: Parameter values used to generate this config.
        created_at: When the config was registered.
    """

    config_hash: str
    session_id: str
    config_content: str
    parameters: dict[str, Any]
    created_at: datetime


@dataclass
class RunInfo:
    """Information about a job run.

    Attributes:
        run_id: Unique run identifier.
        config_hash: Config hash for this run.
        replicate: Replicate number (0-indexed).
        started_at: When the run started.
        completed_at: When the run completed.
        exit_code: Process exit code (0 = success).
        output_path: Path to output files.
        error_message: Error message if run failed.
        metadata: Additional metadata.
    """

    run_id: str
    config_hash: str
    replicate: int
    started_at: datetime | None
    completed_at: datetime | None
    exit_code: int | None
    output_path: str | None
    error_message: str | None
    metadata: dict[str, Any] | None


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


def _generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


@dataclass
class RunRegistry:
    """DuckDB-backed registry for tracking parameter sweeps and job runs.

    Supports both file-based persistence and in-memory mode (using ":memory:").

    Attributes:
        db_path: Path to the DuckDB database file, or ":memory:" for in-memory.
        conn: DuckDB connection (created automatically).

    Example:
        # File-based (persistent)
        registry = RunRegistry("experiment.duckdb")

        # In-memory (for testing)
        registry = RunRegistry(":memory:")

        # Context manager
        with RunRegistry("experiment.duckdb") as registry:
            session_id = registry.create_session(...)
    """

    db_path: Path | str = "josh_runs.duckdb"
    conn: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize database connection and schema."""
        _check_duckdb()
        if self.conn is None:
            db_str = str(self.db_path)
            self.conn = duckdb.connect(db_str)
            self._init_schema()

    def _init_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        self.conn.execute(SCHEMA_SQL)

    def close(self) -> None:
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> RunRegistry:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close connection."""
        self.close()

    # ========== Session Management ==========

    def create_session(
        self,
        experiment_name: str | None = None,
        simulation: str | None = None,
        total_jobs: int | None = None,
        total_replicates: int | None = None,
        template_path: str | None = None,
        template_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new sweep session.

        Args:
            experiment_name: Name for the experiment (used in path templates).
            simulation: Name of the simulation being run.
            total_jobs: Total number of job configurations.
            total_replicates: Total number of replicates across all jobs.
            template_path: Path to the Jinja template file.
            template_hash: Hash of the template content.
            metadata: Additional metadata dictionary.

        Returns:
            The generated session ID.
        """
        session_id = _generate_id()
        metadata_json = json.dumps(metadata) if metadata else None

        self.conn.execute(
            """
            INSERT INTO sweep_sessions
            (session_id, experiment_name, simulation, total_jobs, total_replicates,
             template_path, template_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                session_id,
                experiment_name,
                simulation,
                total_jobs,
                total_replicates,
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

    # ========== Config Management ==========

    def register_config(
        self,
        session_id: str,
        config_hash: str,
        config_content: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a job configuration.

        Args:
            session_id: Session this config belongs to.
            config_hash: MD5 hash of the config content (12 chars).
            config_content: Full text of the configuration.
            parameters: Parameter values used to generate this config.
        """
        parameters_json = json.dumps(parameters)

        # Use INSERT OR IGNORE to handle duplicate config hashes
        self.conn.execute(
            """
            INSERT OR IGNORE INTO job_configs
            (config_hash, session_id, config_content, parameters)
            VALUES (?, ?, ?, ?)
            """,
            [config_hash, session_id, config_content, parameters_json],
        )

    def get_config_by_hash(self, config_hash: str) -> ConfigInfo | None:
        """Get config information by hash.

        Args:
            config_hash: The config hash to look up.

        Returns:
            ConfigInfo if found, None otherwise.
        """
        result = self.conn.execute(
            """
            SELECT config_hash, session_id, config_content, parameters, created_at
            FROM job_configs
            WHERE config_hash = ?
            """,
            [config_hash],
        ).fetchone()

        if result is None:
            return None

        return ConfigInfo(
            config_hash=result[0],
            session_id=result[1],
            config_content=result[2],
            parameters=json.loads(result[3]) if result[3] else {},
            created_at=result[4],
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
            SELECT config_hash, session_id, config_content, parameters, created_at
            FROM job_configs
            WHERE session_id = ?
            ORDER BY created_at
            """,
            [session_id],
        ).fetchall()

        return [
            ConfigInfo(
                config_hash=row[0],
                session_id=row[1],
                config_content=row[2],
                parameters=json.loads(row[3]) if row[3] else {},
                created_at=row[4],
            )
            for row in result
        ]

    # ========== Run Tracking ==========

    def start_run(
        self,
        config_hash: str,
        replicate: int = 0,
        output_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the start of a job run.

        Args:
            config_hash: Config hash for this run.
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
            (run_id, config_hash, replicate, started_at, output_path, metadata)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """,
            [run_id, config_hash, replicate, output_path, metadata_json],
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
            SELECT run_id, config_hash, replicate, started_at, completed_at,
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
            config_hash=result[1],
            replicate=result[2],
            started_at=result[3],
            completed_at=result[4],
            exit_code=result[5],
            output_path=result[6],
            error_message=result[7],
            metadata=json.loads(result[8]) if result[8] else None,
        )

    def get_runs_for_config(self, config_hash: str) -> list[RunInfo]:
        """Get all runs for a config.

        Args:
            config_hash: The config hash to get runs for.

        Returns:
            List of RunInfo objects.
        """
        result = self.conn.execute(
            """
            SELECT run_id, config_hash, replicate, started_at, completed_at,
                   exit_code, output_path, error_message, metadata
            FROM job_runs
            WHERE config_hash = ?
            ORDER BY started_at
            """,
            [config_hash],
        ).fetchall()

        return [
            RunInfo(
                run_id=row[0],
                config_hash=row[1],
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
                SELECT r.run_id, r.config_hash, r.replicate, r.started_at, r.completed_at,
                       r.exit_code, r.output_path, r.error_message, c.parameters
                FROM job_runs r
                JOIN job_configs c ON r.config_hash = c.config_hash
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
                SELECT r.run_id, r.config_hash, r.replicate, r.started_at, r.completed_at,
                       r.exit_code, r.output_path, r.error_message, c.parameters
                FROM job_runs r
                JOIN job_configs c ON r.config_hash = c.config_hash
                WHERE {where_clause}
                ORDER BY r.started_at DESC
                """,
                values,
            ).fetchall()

        return [
            {
                "run_id": row[0],
                "config_hash": row[1],
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

        # Count runs by status
        result = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN completed_at IS NOT NULL THEN 1 END) as completed,
                COUNT(CASE WHEN exit_code = 0 THEN 1 END) as succeeded,
                COUNT(CASE WHEN exit_code IS NOT NULL AND exit_code != 0 THEN 1 END) as failed
            FROM job_runs r
            JOIN job_configs c ON r.config_hash = c.config_hash
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
            total_jobs=session.total_jobs or 0,
            total_replicates=session.total_replicates or 0,
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
            SELECT r.run_id, r.config_hash, r.replicate, r.started_at, r.completed_at,
                   r.exit_code, r.output_path, r.error_message, c.parameters
            FROM job_runs r
            JOIN job_configs c ON r.config_hash = c.config_hash
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
                "config_hash": row[1],
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
            config_hash=job.config_hash,
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
