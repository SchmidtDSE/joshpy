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
from joshpy.jobs import JobConfig
from joshpy.registry import RunRegistry


def _make_config(simulation: str = "TestSim") -> JobConfig:
    """Create a minimal JobConfig for testing."""
    return JobConfig(simulation=simulation)


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
        session_id = registry.create_session(_make_config(), experiment_name="test")
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

            # Verify a specific row - now using typed columns instead of JSON
            row = registry.conn.execute(
                """
                SELECT step, replicate, longitude, latitude, treeCount, avgHeight
                FROM cell_data
                WHERE run_hash = 'abc123' AND step = 1
                """
            ).fetchone()

            assert row[0] == 1  # step
            assert row[1] == 0  # replicate
            assert row[2] == -116.0  # longitude
            assert row[3] == 34.0  # latitude
            assert row[4] == 12  # treeCount (typed column)
            assert row[5] == 6.5  # avgHeight (typed column)

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
        session_id = registry.create_session(_make_config(), experiment_name="test")
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
        session_id = registry.create_session(_make_config(), experiment_name="test")
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
            _make_config(),
            experiment_name="test",
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
        session2 = self.registry.create_session(_make_config(), experiment_name="test2")
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

    def test_get_replicate_uncertainty_confidence(self):
        """Test that confidence parameter affects CI width."""
        skip_if_no_pandas()
        # Get 95% CI (default)
        df_95 = self.queries.get_replicate_uncertainty(
            variable="treeCount",
            run_hash="abc123",
            confidence=0.95,
        )

        # Get 99% CI (wider)
        df_99 = self.queries.get_replicate_uncertainty(
            variable="treeCount",
            run_hash="abc123",
            confidence=0.99,
        )

        # Get 90% CI (narrower)
        df_90 = self.queries.get_replicate_uncertainty(
            variable="treeCount",
            run_hash="abc123",
            confidence=0.90,
        )

        # 99% CI should be wider than 95% CI
        ci_width_99 = (df_99["ci_high"] - df_99["ci_low"]).mean()
        ci_width_95 = (df_95["ci_high"] - df_95["ci_low"]).mean()
        ci_width_90 = (df_90["ci_high"] - df_90["ci_low"]).mean()

        assert ci_width_99 > ci_width_95, "99% CI should be wider than 95% CI"
        assert ci_width_95 > ci_width_90, "95% CI should be wider than 90% CI"

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


class TestGetReplicateCv:
    """Tests for get_replicate_cv method."""

    def setup_method(self):
        """Set up test data with multiple replicates and steps."""
        skip_if_no_pandas()
        self.registry = RunRegistry(":memory:")
        self.loader = CellDataLoader(self.registry)
        self.queries = DiagnosticQueries(self.registry)

        # Create a session and config
        session_id = self.registry.create_session(
            _make_config(),
            experiment_name="test",
        )
        self.registry.register_run(
            session_id=session_id,
            run_hash="cv_test",
            josh_path="test.josh",
            config_content="test config",
            file_mappings=None,
            parameters={"maxGrowth": 10},
        )
        self.run_id = self.registry.start_run("cv_test")

    def _load_cv_test_data(self, data_rows: list[tuple]):
        """Helper to load test CSV data.

        Args:
            data_rows: List of (step, replicate, value) tuples
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,cover\n")
            for step, rep, val in data_rows:
                f.write(f"{step},{rep},-116.0,34.0,{val}\n")
            csv_path = Path(f.name)

        try:
            self.loader.load_csv(csv_path, self.run_id, "cv_test")
        finally:
            csv_path.unlink()

    def test_computes_per_replicate_cv(self):
        """Test that CV is computed correctly per replicate."""
        skip_if_no_pandas()

        # Create data with known CV:
        # Replicate 0: values [10, 10, 10] -> CV = 0
        # Replicate 1: values [10, 20, 30] -> mean=20, std=10, CV=0.5
        data = [
            (0, 0, 10), (1, 0, 10), (2, 0, 10),  # Rep 0: constant
            (0, 1, 10), (1, 1, 20), (2, 1, 30),  # Rep 1: varying
        ]
        self._load_cv_test_data(data)

        result = self.queries.get_replicate_cv(
            variable="cover",
            run_hash="cv_test",
            burn_in=0,
        )

        assert result["n_replicates"] == 2
        assert result["n_timesteps"] == 3
        assert len(result["replicate_cvs"]) == 2
        assert result["extinct_replicates"] == []

        # Check per-replicate CVs
        # Rep 0 should have CV = 0 (constant values)
        assert abs(result["replicate_cvs"][0]) < 0.001

        # Rep 1 should have CV = std/mean = 10/20 = 0.5
        assert abs(result["replicate_cvs"][1] - 0.5) < 0.01

        # Mean CV should be (0 + 0.5) / 2 = 0.25
        assert abs(result["mean_cv"] - 0.25) < 0.01

    def test_burn_in_filters_steps(self):
        """Test that burn_in correctly filters early timesteps."""
        skip_if_no_pandas()

        # Create data where early steps are different from later steps
        # Steps 0-1 have different values, steps 2-4 are constant
        data = [
            (0, 0, 100), (1, 0, 50),  # Burn-in period (variable)
            (2, 0, 10), (3, 0, 10), (4, 0, 10),  # Post burn-in (constant)
        ]
        self._load_cv_test_data(data)

        # With burn_in=2, only steps 2,3,4 should be used -> CV should be 0
        result = self.queries.get_replicate_cv(
            variable="cover",
            run_hash="cv_test",
            burn_in=2,
        )

        assert result["n_timesteps"] == 3  # Steps 2, 3, 4
        assert abs(result["mean_cv"]) < 0.001  # Constant values -> CV = 0

    def test_returns_inf_for_extinction(self):
        """Test that extinction returns inf and warns."""
        skip_if_no_pandas()
        import warnings

        # Create data where mean is below extinction threshold
        data = [
            (0, 0, 0.001), (1, 0, 0.002), (2, 0, 0.001),  # Near-zero values
        ]
        self._load_cv_test_data(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.queries.get_replicate_cv(
                variable="cover",
                run_hash="cv_test",
                burn_in=0,
                extinction_threshold=0.01,
            )

            # Should have extinction warning
            extinction_warnings = [x for x in w if "Extinction" in str(x.message)]
            assert len(extinction_warnings) == 1

        assert result["mean_cv"] == float("inf")
        assert result["extinct_replicates"] == [0]
        assert result["replicate_cvs"] == [float("inf")]

    def test_partial_extinction_returns_inf(self):
        """Test that if ANY replicate goes extinct, mean_cv is inf."""
        skip_if_no_pandas()
        import warnings

        # Replicate 0: healthy values
        # Replicate 1: extinction (near-zero)
        data = [
            (0, 0, 10), (1, 0, 10), (2, 0, 10),  # Rep 0: healthy
            (0, 1, 0.001), (1, 1, 0.001), (2, 1, 0.001),  # Rep 1: extinct
        ]
        self._load_cv_test_data(data)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.queries.get_replicate_cv(
                variable="cover",
                run_hash="cv_test",
                burn_in=0,
                extinction_threshold=0.01,
            )

        # Even though rep 0 is healthy, mean_cv should be inf
        assert result["mean_cv"] == float("inf")
        assert 1 in result["extinct_replicates"]
        assert 0 not in result["extinct_replicates"]

    def test_warns_on_few_timesteps(self):
        """Test warning when n_timesteps < MIN_TIMESTEPS_FOR_CV."""
        skip_if_no_pandas()
        import warnings
        from joshpy.cell_data import MIN_TIMESTEPS_FOR_CV

        # Create data with only 3 timesteps (less than MIN_TIMESTEPS_FOR_CV=10)
        data = [(0, 0, 10), (1, 0, 10), (2, 0, 10)]
        self._load_cv_test_data(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.queries.get_replicate_cv(
                variable="cover",
                run_hash="cv_test",
                burn_in=0,
            )

            # Should warn about few timesteps
            timestep_warnings = [x for x in w if "timesteps" in str(x.message).lower()]
            assert len(timestep_warnings) == 1, f"Expected warning about timesteps, got: {[str(x.message) for x in w]}"

        assert result["n_timesteps"] == 3
        assert result["n_timesteps"] < MIN_TIMESTEPS_FOR_CV

    def test_empty_data_returns_inf(self):
        """Test that empty/no data returns inf with empty lists."""
        skip_if_no_pandas()

        # Don't load any data, just query
        result = self.queries.get_replicate_cv(
            variable="cover",
            run_hash="nonexistent",
            burn_in=0,
        )

        assert result["mean_cv"] == float("inf")
        assert result["replicate_cvs"] == []
        assert result["n_replicates"] == 0
        assert result["n_timesteps"] == 0
        assert result["extinct_replicates"] == []

    def test_custom_extinction_threshold(self):
        """Test that custom extinction_threshold is respected."""
        skip_if_no_pandas()
        import warnings

        # Values between 0.01 and 0.1
        data = [(0, 0, 0.05), (1, 0, 0.05), (2, 0, 0.05)]
        self._load_cv_test_data(data)

        # With default threshold 0.01, this should NOT be extinction
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_default = self.queries.get_replicate_cv(
                variable="cover",
                run_hash="cv_test",
                burn_in=0,
                extinction_threshold=0.01,
            )
        assert result_default["mean_cv"] != float("inf")
        assert result_default["extinct_replicates"] == []

        # Reload data for second test
        self.registry.conn.execute("DELETE FROM cell_data WHERE run_hash = 'cv_test'")
        self._load_cv_test_data(data)

        # With threshold 0.1, this SHOULD be extinction
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_strict = self.queries.get_replicate_cv(
                variable="cover",
                run_hash="cv_test",
                burn_in=0,
                extinction_threshold=0.1,
            )
        assert result_strict["mean_cv"] == float("inf")
        assert result_strict["extinct_replicates"] == [0]
