"""Cell-level spatiotemporal data loading and querying for Josh simulation outputs.

This module provides tools for loading simulation export CSVs into DuckDB and
querying them for diagnostic plots and analysis. All query methods return pandas
DataFrames for seamless integration with the scientific Python stack.

Example usage:
    from joshpy.registry import RunRegistry
    from joshpy.cell_data import CellDataLoader, DiagnosticQueries

    registry = RunRegistry("experiment.duckdb")
    loader = CellDataLoader(registry)

    # Load CSV export
    loader.load_csv(
        csv_path=Path("/tmp/output.csv"),
        run_id="abc123",
        config_hash="a1b2c3d4e5f6",
    )

    # Query time series
    queries = DiagnosticQueries(registry)
    df = queries.get_cell_timeseries(
        longitude=-116.1,
        latitude=33.9,
        variable="treeCount",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _check_pydantic() -> None:
    """Raise ImportError if pydantic is not available."""
    if not HAS_PYDANTIC:
        raise ImportError(
            "pydantic is required for cell data models. Install with: pip install pydantic"
        )


def _check_pandas() -> None:
    """Raise ImportError if pandas is not available."""
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for cell data loading. Install with: pip install pandas"
        )


if HAS_PYDANTIC:

    class CellRecord(BaseModel):
        """A single cell observation from a simulation export.

        Attributes:
            run_id: The run this observation belongs to.
            config_hash: Config hash for this run.
            step: Simulation timestep.
            replicate: Replicate number (0-indexed).
            position_x: Grid x coordinate.
            position_y: Grid y coordinate.
            longitude: Earth longitude (if CRS mode).
            latitude: Earth latitude (if CRS mode).
            entity_type: Type of entity (e.g., 'patch', 'ForeverTree').
            variables: Dict of variable names to values (e.g., {'treeCount': 42}).
        """

        run_id: str
        config_hash: str
        step: int
        replicate: int
        position_x: float | None = None
        position_y: float | None = None
        longitude: float | None = None
        latitude: float | None = None
        entity_type: str = "patch"
        variables: dict[str, Any] = Field(default_factory=dict)

        class Config:
            frozen = True


class CellDataLoader:
    """Load Josh export CSVs into the cell_data table.

    This class handles parsing CSV exports from Josh simulations and inserting
    them into the DuckDB cell_data table for spatiotemporal analysis.

    Attributes:
        registry: The RunRegistry to load data into.

    Example:
        registry = RunRegistry("experiment.duckdb")
        loader = CellDataLoader(registry)

        # Load a CSV export
        rows_loaded = loader.load_csv(
            csv_path=Path("/tmp/output.csv"),
            run_id="abc123",
            config_hash="a1b2c3d4e5f6",
        )
        print(f"Loaded {rows_loaded} rows")
    """

    def __init__(self, registry: Any):
        """Initialize the loader.

        Args:
            registry: RunRegistry instance to load data into.
        """
        from joshpy.registry import RunRegistry

        if not isinstance(registry, RunRegistry):
            raise TypeError(f"Expected RunRegistry, got {type(registry)}")

        self.registry = registry

    def load_csv(
        self,
        csv_path: Path,
        run_id: str,
        config_hash: str,
        entity_type: str = "patch",
    ) -> int:
        """Load a CSV export into the cell_data table.

        The CSV is expected to have columns for simulation variables plus:
        - step: timestep
        - replicate: replicate number
        - position.x, position.y: grid coordinates
        - position.longitude, position.latitude: Earth coordinates (optional)

        Uses DuckDB's native CSV reader for optimal performance.

        Args:
            csv_path: Path to the CSV file.
            run_id: The run ID this data belongs to.
            config_hash: Config hash for this run.
            entity_type: Type of entity being exported (default: "patch").

        Returns:
            Number of rows loaded.

        Raises:
            FileNotFoundError: If csv_path doesn't exist.
            ValueError: If CSV is missing required columns.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        conn = self.registry.conn
        csv_path_str = str(csv_path.resolve())

        # Read CSV header to identify columns using DuckDB
        header_result = conn.execute(
            f"SELECT * FROM read_csv_auto('{csv_path_str}') LIMIT 0"
        )
        columns = [desc[0] for desc in header_result.description]

        # Check required columns
        if "step" not in columns or "replicate" not in columns:
            raise ValueError(
                f"CSV must have 'step' and 'replicate' columns. Found: {columns}"
            )

        # Identify metadata vs variable columns
        meta_cols = {
            "position.x",
            "position.y",
            "position.longitude",
            "position.latitude",
            "step",
            "replicate",
        }
        var_cols = [c for c in columns if c not in meta_cols]

        # Build JSON object expression for variable columns using DuckDB's json functions
        if var_cols:
            # Build key-value pairs for json_object, filtering NULLs per column
            json_pairs = ", ".join(
                f"'{col}', \"{col}\"" for col in var_cols
            )
            json_expr = f"json_object({json_pairs})"
        else:
            json_expr = "'{}'::JSON"

        # Build column expressions with proper NULL handling
        pos_x_expr = '"position.x"' if "position.x" in columns else "NULL"
        pos_y_expr = '"position.y"' if "position.y" in columns else "NULL"
        lon_expr = '"position.longitude"' if "position.longitude" in columns else "NULL"
        lat_expr = '"position.latitude"' if "position.latitude" in columns else "NULL"

        # Escape single quotes in string literals
        safe_run_id = run_id.replace("'", "''")
        safe_config_hash = config_hash.replace("'", "''")
        safe_entity_type = entity_type.replace("'", "''")

        # Insert directly from CSV using DuckDB's native CSV reader
        insert_sql = f"""
            INSERT INTO cell_data
            (run_id, config_hash, step, replicate, position_x, position_y,
             longitude, latitude, entity_type, variables)
            SELECT
                '{safe_run_id}' as run_id,
                '{safe_config_hash}' as config_hash,
                CAST(step AS INTEGER) as step,
                CAST(replicate AS INTEGER) as replicate,
                {pos_x_expr} as position_x,
                {pos_y_expr} as position_y,
                {lon_expr} as longitude,
                {lat_expr} as latitude,
                '{safe_entity_type}' as entity_type,
                {json_expr} as variables
            FROM read_csv_auto('{csv_path_str}')
        """

        conn.execute(insert_sql)

        # Get row count
        count_result = conn.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{csv_path_str}')"
        ).fetchone()

        return count_result[0] if count_result else 0

    def load_csv_batch(
        self,
        csv_paths: list[tuple[Path, str, str]],
        entity_type: str = "patch",
    ) -> int:
        """Load multiple CSV files in batch.

        Args:
            csv_paths: List of (csv_path, run_id, config_hash) tuples.
            entity_type: Type of entity being exported.

        Returns:
            Total number of rows loaded across all files.
        """
        total_rows = 0
        for csv_path, run_id, config_hash in csv_paths:
            rows = self.load_csv(csv_path, run_id, config_hash, entity_type)
            total_rows += rows
        return total_rows


class DiagnosticQueries:
    """Pre-built queries for common diagnostic plots and analysis.

    All methods return pandas DataFrames for easy integration with matplotlib,
    seaborn, and other scientific Python libraries.

    Attributes:
        registry: The RunRegistry to query.

    Example:
        queries = DiagnosticQueries(registry)

        # Time series at a location
        df = queries.get_cell_timeseries(
            longitude=-116.1,
            latitude=33.9,
            variable="treeCount",
        )

        # Spatial snapshot
        df = queries.get_spatial_snapshot(
            step=50,
            variable="treeCount",
            config_hash="a1b2c3d4e5f6",
        )
    """

    def __init__(self, registry: Any):
        """Initialize the query builder.

        Args:
            registry: RunRegistry instance to query.
        """
        from joshpy.registry import RunRegistry

        if not isinstance(registry, RunRegistry):
            raise TypeError(f"Expected RunRegistry, got {type(registry)}")

        self.registry = registry

    def get_cell_timeseries(
        self,
        longitude: float,
        latitude: float,
        variable: str,
        config_hash: str | None = None,
        tolerance: float = 0.01,
    ) -> Any:
        """Get time series for a specific location.

        Args:
            longitude: Longitude of the cell.
            latitude: Latitude of the cell.
            variable: Variable name to extract (e.g., "treeCount").
            config_hash: Optional filter by config hash.
            tolerance: Spatial tolerance in degrees (default: ~1km at equator).

        Returns:
            DataFrame with columns: step, replicate, value, config_hash.
        """
        _check_pandas()

        query = """
            SELECT
                step,
                replicate,
                CAST(json_extract_string(variables, ?) AS DOUBLE) as value,
                config_hash
            FROM cell_data
            WHERE longitude BETWEEN ? AND ?
              AND latitude BETWEEN ? AND ?
        """
        params = [
            f"$.{variable}",
            longitude - tolerance,
            longitude + tolerance,
            latitude - tolerance,
            latitude + tolerance,
        ]

        if config_hash:
            query += " AND config_hash = ?"
            params.append(config_hash)

        query += " ORDER BY config_hash, replicate, step"

        return self.registry.conn.execute(query, params).df()

    def get_spatial_snapshot(
        self,
        step: int,
        variable: str,
        config_hash: str,
        replicate: int = 0,
    ) -> Any:
        """Get spatial data for a single timestep.

        Useful for creating heatmaps or choropleth maps.

        Args:
            step: The timestep to query.
            variable: Variable name to extract.
            config_hash: Config hash to filter by.
            replicate: Replicate number (default: 0).

        Returns:
            DataFrame with columns: longitude, latitude, value.
        """
        _check_pandas()

        return self.registry.conn.execute(
            """
            SELECT
                longitude,
                latitude,
                CAST(json_extract_string(variables, ?) AS DOUBLE) as value
            FROM cell_data
            WHERE step = ?
              AND config_hash = ?
              AND replicate = ?
              AND longitude IS NOT NULL
            """,
            [f"$.{variable}", step, config_hash, replicate],
        ).df()

    def get_parameter_comparison(
        self,
        variable: str,
        param_name: str,
        step: int | None = None,
        aggregation: str = "AVG",
    ) -> Any:
        """Compare a variable across parameter values.

        Args:
            variable: Variable name to analyze.
            param_name: Parameter name to group by.
            step: Optional timestep filter (if None, groups by step).
            aggregation: Aggregation function (AVG, MIN, MAX, SUM).

        Returns:
            DataFrame with columns: param_value, step, mean_value, std_value, n_cells.
        """
        _check_pandas()

        step_filter = "AND cd.step = ?" if step else ""
        step_group = "" if step else ", cd.step"

        query = f"""
            SELECT
                json_extract_string(jc.parameters, '$.{param_name}') as param_value,
                cd.step,
                {aggregation}(CAST(json_extract_string(cd.variables, '$.{variable}') AS DOUBLE)) as mean_value,
                STDDEV(CAST(json_extract_string(cd.variables, '$.{variable}') AS DOUBLE)) as std_value,
                COUNT(*) as n_cells
            FROM cell_data cd
            JOIN job_configs jc ON cd.config_hash = jc.config_hash
            WHERE 1=1 {step_filter}
            GROUP BY json_extract_string(jc.parameters, '$.{param_name}'){step_group}
            ORDER BY param_value, cd.step
        """

        params = [step] if step else []
        return self.registry.conn.execute(query, params).df()

    def get_replicate_uncertainty(
        self,
        variable: str,
        config_hash: str,
        step: int | None = None,
    ) -> Any:
        """Get mean and confidence intervals across replicates.

        Args:
            variable: Variable name to analyze.
            config_hash: Config hash to filter by.
            step: Optional timestep filter (if None, aggregates across all steps).

        Returns:
            DataFrame with: step, mean, std, ci_low, ci_high, n_replicates.
        """
        _check_pandas()

        step_filter = "AND step = ?" if step else ""

        query = f"""
            WITH replicate_means AS (
                SELECT
                    step,
                    replicate,
                    AVG(CAST(json_extract_string(variables, '$.{variable}') AS DOUBLE)) as rep_mean
                FROM cell_data
                WHERE config_hash = ? {step_filter}
                GROUP BY step, replicate
            )
            SELECT
                step,
                AVG(rep_mean) as mean,
                STDDEV(rep_mean) as std,
                AVG(rep_mean) - 1.96 * STDDEV(rep_mean) / SQRT(COUNT(*)) as ci_low,
                AVG(rep_mean) + 1.96 * STDDEV(rep_mean) / SQRT(COUNT(*)) as ci_high,
                COUNT(*) as n_replicates
            FROM replicate_means
            GROUP BY step
            ORDER BY step
        """

        params = [config_hash, step] if step else [config_hash]
        return self.registry.conn.execute(query, params).df()

    def get_bbox_timeseries(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        variable: str,
        config_hash: str | None = None,
        aggregation: str = "AVG",
    ) -> Any:
        """Get aggregated time series for a bounding box region.

        Args:
            min_lon: Minimum longitude.
            max_lon: Maximum longitude.
            min_lat: Minimum latitude.
            max_lat: Maximum latitude.
            variable: Variable name to analyze.
            config_hash: Optional config hash filter.
            aggregation: Aggregation function (AVG, MIN, MAX, SUM).

        Returns:
            DataFrame with: step, replicate, config_hash, value, n_cells.
        """
        _check_pandas()

        query = f"""
            SELECT
                step,
                replicate,
                config_hash,
                {aggregation}(CAST(json_extract_string(variables, '$.{variable}') AS DOUBLE)) as value,
                COUNT(*) as n_cells
            FROM cell_data
            WHERE longitude BETWEEN ? AND ?
              AND latitude BETWEEN ? AND ?
        """
        params = [min_lon, max_lon, min_lat, max_lat]

        if config_hash:
            query += " AND config_hash = ?"
            params.append(config_hash)

        query += " GROUP BY step, replicate, config_hash ORDER BY config_hash, replicate, step"

        return self.registry.conn.execute(query, params).df()

    def get_all_variables_at_step(
        self,
        step: int,
        config_hash: str,
        replicate: int = 0,
    ) -> Any:
        """Get all variables for all cells at a specific timestep.

        Returns the raw JSON variables column unpacked into DataFrame columns.

        Args:
            step: The timestep to query.
            config_hash: Config hash to filter by.
            replicate: Replicate number (default: 0).

        Returns:
            DataFrame with position columns plus all variable columns.
        """
        _check_pandas()

        # Get raw data
        df = self.registry.conn.execute(
            """
            SELECT
                longitude,
                latitude,
                position_x,
                position_y,
                variables
            FROM cell_data
            WHERE step = ?
              AND config_hash = ?
              AND replicate = ?
            """,
            [step, config_hash, replicate],
        ).df()

        if df.empty:
            return df

        # Unpack JSON variables into columns
        variables_df = df["variables"].apply(json.loads).apply(pd.Series)
        result = pd.concat([df.drop(columns=["variables"]), variables_df], axis=1)

        return result
