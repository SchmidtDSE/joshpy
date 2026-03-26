"""Project-level experiment catalog for cross-experiment provenance tracking.

A ProjectCatalog sits above RunRegistry, tracking which models, configs, and
data were used in which experiments, and where results live. It does NOT store
cell-level simulation data — that stays in per-experiment RunRegistry files.

The core schema is josh-agnostic (any tool can produce/consume it). An optional
``orchestration`` JSON field stores joshpy-specific metadata (JobConfig, sweep
structure, template provenance) for richer features when available.

Example usage:
    from joshpy.catalog import ProjectCatalog

    catalog = ProjectCatalog("project.duckdb")

    # Register an experiment
    exp_id = catalog.register_experiment(
        config=job_config,
        registry_path="experiments/baseline.duckdb",
        name="baseline_dev_fine",
    )

    # Check if already run
    existing = catalog.find_experiment(job_config)
    if existing and existing.status == "completed":
        print(f"Already run: {existing.registry_path}")

    # List experiments
    for exp in catalog.list_experiments(model_name="canonical*"):
        print(f"{exp.name}: {exp.status}")

    # Cross-experiment query
    experiments = catalog.list_experiments(status="completed")
    with catalog.open_registries(experiments) as conn:
        df = conn.execute("SELECT ...").df()

Requires: duckdb
"""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


def _check_duckdb() -> None:
    """Raise ImportError if duckdb is not available."""
    if not HAS_DUCKDB:
        raise ImportError(
            "duckdb is required for the catalog module. "
            "Install with: pip install joshpy[registry]"
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CATALOG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    model_hash    VARCHAR PRIMARY KEY,
    name          VARCHAR,
    path          VARCHAR,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata      JSON
);

CREATE TABLE IF NOT EXISTS data_manifests (
    manifest_hash  VARCHAR PRIMARY KEY,
    name           VARCHAR,
    path           VARCHAR,
    file_count     INTEGER,
    total_size     BIGINT,
    file_inventory JSON,
    registered_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata       JSON
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       VARCHAR PRIMARY KEY,
    name                VARCHAR,
    model_hash          VARCHAR,
    config_hash         VARCHAR,
    data_manifest_hash  VARCHAR,
    simulation          VARCHAR,
    replicates          INTEGER,
    registry_path       VARCHAR,
    status              VARCHAR DEFAULT 'pending',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP,
    summary             JSON,
    orchestration       JSON
);
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Information about a registered model."""

    model_hash: str
    name: str | None
    path: str | None
    registered_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class DataManifestInfo:
    """Information about a registered data manifest."""

    manifest_hash: str
    name: str | None
    path: str | None
    file_count: int = 0
    total_size: int = 0
    file_inventory: dict[str, Any] | None = None
    registered_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ExperimentInfo:
    """Information about a registered experiment."""

    experiment_id: str
    name: str | None
    model_hash: str | None
    config_hash: str | None
    data_manifest_hash: str | None
    simulation: str | None
    replicates: int | None
    registry_path: str | None
    status: str = "pending"
    created_at: datetime | None = None
    completed_at: datetime | None = None
    summary: dict[str, Any] | None = None
    orchestration: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def _hash_file_content(path: Path) -> str:
    """Compute MD5 hash of file contents (12-char hex)."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def compute_model_hash(path: Path) -> str:
    """Compute hash for a model file (.josh).

    Args:
        path: Path to the .josh file (or rendered .josh from a template).

    Returns:
        12-character hex hash of file content.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return _hash_file_content(path)


def compute_config_hash(content: str) -> str:
    """Compute hash for config content (.jshc).

    Args:
        content: Rendered .jshc content string.

    Returns:
        12-character hex hash.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def compute_data_manifest_hash(file_mappings: dict[str, Path]) -> str:
    """Compute hash for a set of data files.

    Hashes the sorted (name, file_content_hash) pairs for determinism.

    Args:
        file_mappings: Dict mapping names to file paths.

    Returns:
        12-character hex hash.
    """
    hasher = hashlib.md5()
    for name in sorted(file_mappings.keys()):
        path = file_mappings[name]
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path} (name='{name}')")
        hasher.update(name.encode("utf-8"))
        hasher.update(_hash_file_content(path).encode("utf-8"))
    return hasher.hexdigest()[:12]


# ---------------------------------------------------------------------------
# ProjectCatalog
# ---------------------------------------------------------------------------

class ProjectCatalog:
    """DuckDB-backed catalog for tracking experiments across a project.

    Tracks which models, configs, and data were used in which experiments,
    and where results (RunRegistry files) live.

    Args:
        path: Path to DuckDB file, or ":memory:" for in-memory catalog.

    Examples:
        >>> catalog = ProjectCatalog("project.duckdb")
        >>> model_hash = catalog.register_model(Path("model.josh"))
        >>> exp_id = catalog.register_experiment(
        ...     config=job_config,
        ...     registry_path="experiments/run1.duckdb",
        ... )
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        _check_duckdb()
        self._path = str(path)
        self._conn = duckdb.connect(self._path)
        self._conn.execute(CATALOG_SCHEMA_SQL)

    @property
    def path(self) -> str:
        """Path to the catalog DuckDB file."""
        return self._path

    # -- Models -----------------------------------------------------------

    def register_model(
        self,
        path: Path,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a model file (.josh) in the catalog.

        Computes content hash and stores metadata. If the same hash already
        exists, the registration is a no-op (idempotent).

        Args:
            path: Path to the .josh file.
            name: Human-readable name (defaults to filename stem).
            metadata: Optional metadata dict.

        Returns:
            12-character model hash.
        """
        model_hash = compute_model_hash(path)
        if name is None:
            name = path.stem

        self._conn.execute(
            """
            INSERT OR IGNORE INTO models (model_hash, name, path, metadata)
            VALUES (?, ?, ?, ?)
            """,
            [model_hash, name, str(path), json.dumps(metadata) if metadata else None],
        )
        return model_hash

    def get_model(self, model_hash: str) -> ModelInfo | None:
        """Get model info by hash."""
        row = self._conn.execute(
            "SELECT * FROM models WHERE model_hash = ?", [model_hash]
        ).fetchone()
        if row is None:
            return None
        return ModelInfo(
            model_hash=row[0],
            name=row[1],
            path=row[2],
            registered_at=row[3],
            metadata=json.loads(row[4]) if row[4] else None,
        )

    # -- Data Manifests ---------------------------------------------------

    def register_data(
        self,
        file_mappings: dict[str, Path],
        *,
        name: str | None = None,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a data manifest (set of .jshd files) in the catalog.

        Computes a manifest hash from the sorted file content hashes.
        Idempotent: re-registering the same files is a no-op.

        Args:
            file_mappings: Dict mapping names to file paths.
            name: Human-readable name for this data set.
            path: Directory path for reference.
            metadata: Optional metadata (grid_size, bounds, crs, etc.).

        Returns:
            12-character manifest hash.
        """
        manifest_hash = compute_data_manifest_hash(file_mappings)

        file_inventory = {}
        total_size = 0
        for fname, fpath in sorted(file_mappings.items()):
            size = fpath.stat().st_size
            total_size += size
            file_inventory[fname] = {
                "path": str(fpath),
                "hash": _hash_file_content(fpath),
                "size_bytes": size,
            }

        self._conn.execute(
            """
            INSERT OR IGNORE INTO data_manifests
                (manifest_hash, name, path, file_count, total_size, file_inventory, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                manifest_hash,
                name,
                path,
                len(file_mappings),
                total_size,
                json.dumps(file_inventory),
                json.dumps(metadata) if metadata else None,
            ],
        )
        return manifest_hash

    def get_data_manifest(self, manifest_hash: str) -> DataManifestInfo | None:
        """Get data manifest info by hash."""
        row = self._conn.execute(
            "SELECT * FROM data_manifests WHERE manifest_hash = ?", [manifest_hash]
        ).fetchone()
        if row is None:
            return None
        return DataManifestInfo(
            manifest_hash=row[0],
            name=row[1],
            path=row[2],
            file_count=row[3],
            total_size=row[4],
            file_inventory=json.loads(row[5]) if row[5] else None,
            registered_at=row[6],
            metadata=json.loads(row[7]) if row[7] else None,
        )

    # -- Experiments ------------------------------------------------------

    def register_experiment(
        self,
        config: Any,  # JobConfig
        registry_path: str | Path,
        *,
        name: str | None = None,
        experiment_id: str | None = None,
    ) -> str:
        """Register an experiment from a JobConfig.

        Automatically computes model and data hashes, registers them if new,
        and creates the experiment entry with joshpy orchestration metadata.

        Args:
            config: A JobConfig instance.
            registry_path: Path to the RunRegistry .duckdb file.
            name: Human-readable experiment name.
            experiment_id: Optional explicit ID (auto-generated UUID if None).

        Returns:
            Experiment ID.
        """
        experiment_id = experiment_id or str(uuid.uuid4())[:12]

        # Compute model hash
        model_hash = None
        if config.source_path and config.source_path.exists():
            model_hash = self.register_model(config.source_path)

        # Compute data manifest hash
        data_manifest_hash = None
        if config.file_mappings:
            data_manifest_hash = self.register_data(config.file_mappings)

        # Compute config hash from first expanded job's content,
        # or from config_path if available
        config_hash = None
        if config.config_path and config.config_path.exists():
            config_hash = _hash_file_content(config.config_path)

        # Build orchestration metadata
        orchestration = {
            "tool": "joshpy",
            "job_config": config.to_dict(),
        }

        # Add sweep summary if present
        if config.sweep:
            param_names = [p.name for p in (config.sweep.config_parameters or [])]
            param_names += [p.name for p in (config.sweep.file_parameters or [])]
            param_names += [p.name for p in (config.sweep.compound_parameters or [])]
            strategy_name = type(config.sweep.strategy).__name__ if config.sweep.strategy else "CartesianStrategy"
            orchestration["sweep_summary"] = {
                "parameters": param_names,
                "strategy": strategy_name,
                "total_combinations": len(config.sweep),
            }

        # Add template sources if present
        template_sources: dict[str, str] = {}
        if config.source_template_path:
            template_sources["model_template"] = str(config.source_template_path)
        if config.template_path:
            template_sources["config_template"] = str(config.template_path)
        if config.template_vars:
            template_sources["template_vars"] = config.template_vars
        if template_sources:
            orchestration["template_sources"] = template_sources

        self._conn.execute(
            """
            INSERT INTO experiments
                (experiment_id, name, model_hash, config_hash, data_manifest_hash,
                 simulation, replicates, registry_path, status, orchestration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            [
                experiment_id,
                name,
                model_hash,
                config_hash,
                data_manifest_hash,
                config.simulation,
                config.replicates,
                str(registry_path),
                json.dumps(orchestration),
            ],
        )
        return experiment_id

    def find_experiment(
        self,
        config: Any,  # JobConfig
    ) -> ExperimentInfo | None:
        """Find an existing experiment matching a JobConfig.

        Matches on model_hash + config_hash + data_manifest_hash. Returns the
        most recently created match, or None.

        Args:
            config: A JobConfig instance to match against.

        Returns:
            ExperimentInfo if a match is found, None otherwise.
        """
        # Compute hashes for comparison
        model_hash = None
        if config.source_path and config.source_path.exists():
            model_hash = compute_model_hash(config.source_path)

        config_hash = None
        if config.config_path and config.config_path.exists():
            config_hash = _hash_file_content(config.config_path)

        data_manifest_hash = None
        if config.file_mappings:
            # Only compute if all files exist
            try:
                data_manifest_hash = compute_data_manifest_hash(config.file_mappings)
            except FileNotFoundError:
                pass

        # Build query with NULL-safe comparison
        conditions = []
        params: list[Any] = []

        if model_hash is not None:
            conditions.append("model_hash = ?")
            params.append(model_hash)
        else:
            conditions.append("model_hash IS NULL")

        if config_hash is not None:
            conditions.append("config_hash = ?")
            params.append(config_hash)
        else:
            conditions.append("config_hash IS NULL")

        if data_manifest_hash is not None:
            conditions.append("data_manifest_hash = ?")
            params.append(data_manifest_hash)
        else:
            conditions.append("data_manifest_hash IS NULL")

        where = " AND ".join(conditions)
        row = self._conn.execute(
            f"SELECT * FROM experiments WHERE {where} ORDER BY created_at DESC LIMIT 1",
            params,
        ).fetchone()

        if row is None:
            return None
        return self._row_to_experiment(row)

    def get_experiment(self, experiment_id: str) -> ExperimentInfo | None:
        """Get experiment by ID."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", [experiment_id]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experiment(row)

    def list_experiments(
        self,
        *,
        status: str | None = None,
        model_hash: str | None = None,
        model_name: str | None = None,
        data_name: str | None = None,
    ) -> list[ExperimentInfo]:
        """List experiments with optional filters.

        Args:
            status: Filter by status (pending/running/completed/failed).
            model_hash: Filter by exact model hash.
            model_name: Filter by model name (supports SQL LIKE with %).
            data_name: Filter by data manifest name (supports SQL LIKE with %).

        Returns:
            List of matching experiments, newest first.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if status is not None:
            conditions.append("e.status = ?")
            params.append(status)
        if model_hash is not None:
            conditions.append("e.model_hash = ?")
            params.append(model_hash)
        if model_name is not None:
            # Replace * with % for LIKE
            pattern = model_name.replace("*", "%")
            conditions.append("m.name LIKE ?")
            params.append(pattern)
        if data_name is not None:
            pattern = data_name.replace("*", "%")
            conditions.append("d.name LIKE ?")
            params.append(pattern)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Use LEFT JOINs so experiments without model/data still appear
        rows = self._conn.execute(
            f"""
            SELECT e.*
            FROM experiments e
            LEFT JOIN models m ON e.model_hash = m.model_hash
            LEFT JOIN data_manifests d ON e.data_manifest_hash = d.manifest_hash
            {where}
            ORDER BY e.created_at DESC
            """,
            params,
        ).fetchall()

        return [self._row_to_experiment(row) for row in rows]

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        *,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Update experiment status and optional summary.

        Args:
            experiment_id: Experiment to update.
            status: New status (pending/running/completed/failed).
            summary: Optional summary dict (total_jobs, succeeded, failed, etc.).
        """
        completed_at = (
            datetime.now(timezone.utc).isoformat()
            if status in ("completed", "failed")
            else None
        )
        self._conn.execute(
            """
            UPDATE experiments
            SET status = ?, completed_at = ?, summary = ?
            WHERE experiment_id = ?
            """,
            [
                status,
                completed_at,
                json.dumps(summary) if summary else None,
                experiment_id,
            ],
        )

    # -- Cross-experiment queries -----------------------------------------

    @contextmanager
    def open_registries(
        self,
        experiments: list[ExperimentInfo],
    ) -> Iterator[Any]:  # duckdb.DuckDBPyConnection
        """Open multiple experiment registries for cross-experiment queries.

        Uses DuckDB ATTACH to make multiple registry files queryable in a
        single connection. Each registry is attached as ``exp_<N>`` where N
        is 0-indexed.

        Args:
            experiments: List of experiments whose registries to open.

        Yields:
            DuckDB connection with registries attached.

        Examples:
            >>> experiments = catalog.list_experiments(status="completed")
            >>> with catalog.open_registries(experiments) as conn:
            ...     df = conn.execute('''
            ...         SELECT 'exp_0' as source, step, AVG(averageHeight)
            ...         FROM exp_0.cell_data GROUP BY step
            ...         UNION ALL
            ...         SELECT 'exp_1' as source, step, AVG(averageHeight)
            ...         FROM exp_1.cell_data GROUP BY step
            ...     ''').df()
        """
        _check_duckdb()
        conn = duckdb.connect(":memory:")
        attached: list[str] = []

        try:
            for i, exp in enumerate(experiments):
                if not exp.registry_path:
                    continue
                reg_path = Path(exp.registry_path)
                if not reg_path.exists():
                    continue
                alias = f"exp_{i}"
                conn.execute(
                    f"ATTACH '{reg_path}' AS {alias} (READ_ONLY)"
                )
                attached.append(alias)
            yield conn
        finally:
            for alias in attached:
                try:
                    conn.execute(f"DETACH {alias}")
                except Exception:
                    pass
            conn.close()

    # -- Lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the catalog connection."""
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> ProjectCatalog:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- Internal ---------------------------------------------------------

    def _row_to_experiment(self, row: tuple[Any, ...]) -> ExperimentInfo:
        """Convert a database row to ExperimentInfo."""
        return ExperimentInfo(
            experiment_id=row[0],
            name=row[1],
            model_hash=row[2],
            config_hash=row[3],
            data_manifest_hash=row[4],
            simulation=row[5],
            replicates=row[6],
            registry_path=row[7],
            status=row[8],
            created_at=row[9],
            completed_at=row[10],
            summary=json.loads(row[11]) if row[11] else None,
            orchestration=json.loads(row[12]) if row[12] else None,
        )
