"""Tests for cell data loading and diagnostic queries."""

import json
import tempfile
from pathlib import Path

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from joshpy.cell_data import CellDataLoader, DiagnosticQueries
from joshpy.registry import RunRegistry


def skip_if_no_pandas():
    """Skip test if pandas is not available."""
    if not HAS_PANDAS:
        pytest.skip("pandas not installed")


class TestCellDataLoader:
    """Tests for CellDataLoader class."""

    def test_init_requires_registry(self):
        """Test that CellDataLoader requires a RunRegistry."""
        skip_if_no_pandas()
        registry = RunRegistry(":memory:")
        loader = CellDataLoader(registry)
        assert loader.registry == registry

    def test_init_rejects_non_registry(self):
        """Test that CellDataLoader rejects non-Registry objects."""
        skip_if_no_pandas()
        with pytest.raises(TypeError):
            CellDataLoader("not a registry")  # type: ignore

    def test_load_csv_basic(self):
        """Test loading a basic CSV file."""
        skip_if_no_pandas()
        registry = RunRegistry(":memory:")
        loader = CellDataLoader(registry)

        # Create session and config
        session_id = registry.create_session(experiment_name="test", simulation="TestSim")
        registry.register_run(session_id, "abc123", "test.josh", "test", None, {})
        # Create run record (required for foreign key)
        registry.start_run("abc123")

        # Create a sample CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.x,position.y,position.longitude,position.latitude,treeCount,avgHeight\n")
            f.write("0,0,0.0,0.0,-116.0,34.0,10,5.0\n")
            f.write("1,0,0.0,0.0,-116.0,34.0,12,6.5\n")
            f.write("2,0,0.0,0.0,-116.0,34.0,15,8.0\n")
            csv_path = Path(f.name)

        try:
            rows_loaded = loader.load_csv(
                csv_path=csv_path,
                run_id=registry.start_run("abc123"),
                run_hash="abc123",
            )

            assert rows_loaded == 3

            # Verify data was inserted
            result = registry.conn.execute(
                "SELECT COUNT(*) FROM cell_data WHERE run_hash = 'abc123'"
            ).fetchone()
            assert result[0] == 3

            # Verify a specific row
            row = registry.conn.execute(
                """
                SELECT step, replicate, longitude, latitude, variables
                FROM cell_data
                WHERE run_hash = 'abc123' AND step = 1
                """
            ).fetchone()

            assert row[0] == 1  # step
            assert row[1] == 0  # replicate
            assert row[2] == -116.0  # longitude
            assert row[3] == 34.0  # latitude

            variables = json.loads(row[4])
            assert variables["treeCount"] == 12
            assert variables["avgHeight"] == 6.5

        finally:
            csv_path.unlink()

    def test_load_csv_missing_file(self):
        """Test that load_csv raises FileNotFoundError for missing files."""
        skip_if_no_pandas()
        registry = RunRegistry(":memory:")
        loader = CellDataLoader(registry)

        with pytest.raises(FileNotFoundError):
            loader.load_csv(
                csv_path=Path("/nonexistent/file.csv"),
                run_id="test",
                run_hash="test",
            )

    def test_load_csv_missing_required_columns(self):
        """Test that load_csv raises ValueError for missing required columns."""
        skip_if_no_pandas()
        registry = RunRegistry(":memory:")
        loader = CellDataLoader(registry)

        # Create CSV without required columns
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("treeCount,avgHeight\n")
            f.write("10,5.0\n")
            csv_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="must have 'step' and 'replicate'"):
                loader.load_csv(
                    csv_path=csv_path,
                    run_id="test",
                    run_hash="test",
                )
        finally:
            csv_path.unlink()

    def test_load_csv_with_nan_values(self):
        """Test loading CSV with NaN values."""
        skip_if_no_pandas()
        registry = RunRegistry(":memory:")
        loader = CellDataLoader(registry)

        # Create session, config, and run
        session_id = registry.create_session(experiment_name="test", simulation="TestSim")
        registry.register_run(session_id, "abc", "test.josh", "test", None, {})
        run_id = registry.start_run("abc")

        # Create CSV with some NaN values
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,treeCount\n")
            f.write("0,0,-116.0,34.0,10\n")
            f.write("1,0,,,12\n")  # Missing coordinates
            csv_path = Path(f.name)

        try:
            rows = loader.load_csv(csv_path, run_id, "abc")
            assert rows == 2

            # Check that NaN was converted to None
            row = registry.conn.execute(
                "SELECT longitude, latitude FROM cell_data WHERE step = 1"
            ).fetchone()
            assert row[0] is None
            assert row[1] is None

        finally:
            csv_path.unlink()

    def test_load_csv_batch(self):
        """Test loading multiple CSV files in batch."""
        skip_if_no_pandas()
        registry = RunRegistry(":memory:")
        loader = CellDataLoader(registry)

        # Create session, configs, and runs
        session_id = registry.create_session(experiment_name="test", simulation="TestSim")
        files = []
        for i in range(2):
            run_hash = f"hash_{i}"
            registry.register_run(session_id, run_hash, "test.josh", "test", None, {})
            run_id = registry.start_run(run_hash)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write("step,replicate,treeCount\n")
                f.write(f"{i},0,{i*10}\n")
                files.append((Path(f.name), run_id, run_hash))

        try:
            total = loader.load_csv_batch(files)
            assert total == 2

            # Verify both runs were loaded
            count = registry.conn.execute("SELECT COUNT(DISTINCT run_id) FROM cell_data").fetchone()[0]
            assert count == 2

        finally:
            for path, _, _ in files:
                path.unlink()


class TestDiagnosticQueries:
    """Tests for DiagnosticQueries class."""

    def setup_method(self):
        """Set up test data before each test."""
        skip_if_no_pandas()
        self.registry = RunRegistry(":memory:")
        self.loader = CellDataLoader(self.registry)
        self.queries = DiagnosticQueries(self.registry)

        # Create a session and config
        session_id = self.registry.create_session(
            experiment_name="test",
            simulation="TestSim",
        )
        self.registry.register_run(
            session_id=session_id,
            run_hash="abc123",
            josh_path="test.josh",
            config_content="test config",
            file_mappings=None,
            parameters={"maxGrowth": 10},
        )

        # Create run record (required for foreign key)
        self.run_id = self.registry.start_run("abc123")

        # Load some test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,treeCount,avgHeight\n")
            # Cell at (-116.0, 34.0) across 3 timesteps, 2 replicates
            for step in range(3):
                for rep in range(2):
                    f.write(f"{step},{rep},-116.0,34.0,{10 + step + rep},{5.0 + step}\n")
            # Another cell at (-116.1, 34.1)
            for step in range(3):
                f.write(f"{step},0,-116.1,34.1,{20 + step},10.0\n")
            self.csv_path = Path(f.name)

        self.loader.load_csv(self.csv_path, self.run_id, "abc123")

    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self, "csv_path") and self.csv_path.exists():
            self.csv_path.unlink()

    def test_init_requires_registry(self):
        """Test that DiagnosticQueries requires a RunRegistry."""
        skip_if_no_pandas()
        queries = DiagnosticQueries(self.registry)
        assert queries.registry == self.registry

    def test_init_rejects_non_registry(self):
        """Test that DiagnosticQueries rejects non-Registry objects."""
        skip_if_no_pandas()
        with pytest.raises(TypeError):
            DiagnosticQueries("not a registry")  # type: ignore

    def test_get_cell_timeseries(self):
        """Test getting time series for a specific location."""
        skip_if_no_pandas()
        df = self.queries.get_cell_timeseries(
            longitude=-116.0,
            latitude=34.0,
            variable="treeCount",
            run_hash="abc123",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 3 steps × 2 replicates
        assert list(df.columns) == ["step", "replicate", "value", "run_hash"]
        assert df["value"].min() == 10
        assert df["value"].max() == 13

    def test_get_cell_timeseries_with_tolerance(self):
        """Test time series with spatial tolerance."""
        skip_if_no_pandas()
        # Query with large tolerance should get both cells
        df = self.queries.get_cell_timeseries(
            longitude=-116.05,
            latitude=34.05,
            variable="treeCount",
            tolerance=0.1,  # ~10km
        )

        assert len(df) == 9  # Both cells: (3×2) + 3

    def test_get_spatial_snapshot(self):
        """Test getting spatial data for a single timestep."""
        skip_if_no_pandas()
        df = self.queries.get_spatial_snapshot(
            step=1,
            variable="treeCount",
            run_hash="abc123",
            replicate=0,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two cells
        assert list(df.columns) == ["longitude", "latitude", "value"]
        assert (-116.0, 34.0) in zip(df["longitude"], df["latitude"])
        assert (-116.1, 34.1) in zip(df["longitude"], df["latitude"])

    def test_get_parameter_comparison(self):
        """Test comparing variables across parameter values."""
        skip_if_no_pandas()
        # Add another config with different parameter
        session2 = self.registry.create_session(experiment_name="test2", simulation="TestSim")
        self.registry.register_run(
            session_id=session2,
            run_hash="def456",
            josh_path="test.josh",
            config_content="test config 2",
            file_mappings=None,
            parameters={"maxGrowth": 20},
        )
        run2 = self.registry.start_run("def456")

        # Load data for second config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,treeCount\n")
            f.write("0,0,-116.0,34.0,50\n")
            csv_path = Path(f.name)

        self.loader.load_csv(csv_path, run2, "def456")
        csv_path.unlink()

        df = self.queries.get_parameter_comparison(
            variable="treeCount",
            param_name="maxGrowth",
        )

        assert isinstance(df, pd.DataFrame)
        assert "param_value" in df.columns
        assert "mean_value" in df.columns

    def test_get_replicate_uncertainty(self):
        """Test getting uncertainty across replicates."""
        skip_if_no_pandas()
        df = self.queries.get_replicate_uncertainty(
            variable="treeCount",
            run_hash="abc123",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 timesteps
        assert list(df.columns) == ["step", "mean", "std", "ci_low", "ci_high", "n_replicates"]

        # Check that we have 2 replicates per step
        assert all(df["n_replicates"] == 2)

    def test_get_bbox_timeseries(self):
        """Test getting time series for a bounding box."""
        skip_if_no_pandas()
        df = self.queries.get_bbox_timeseries(
            min_lon=-116.2,
            max_lon=-115.9,
            min_lat=33.9,
            max_lat=34.2,
            variable="treeCount",
            run_hash="abc123",
        )

        assert isinstance(df, pd.DataFrame)
        assert "step" in df.columns
        assert "value" in df.columns
        assert "n_cells" in df.columns

    def test_get_all_variables_at_step(self):
        """Test getting all variables unpacked for a timestep."""
        skip_if_no_pandas()
        df = self.queries.get_all_variables_at_step(
            step=1,
            run_hash="abc123",
            replicate=0,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two cells
        # Should have unpacked variable columns
        assert "treeCount" in df.columns
        assert "avgHeight" in df.columns
        assert "longitude" in df.columns

    def test_empty_results(self):
        """Test that queries return empty DataFrames for no matches."""
        skip_if_no_pandas()
        df = self.queries.get_cell_timeseries(
            longitude=0.0,  # No data at this location
            latitude=0.0,
            variable="treeCount",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
