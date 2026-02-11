"""Tests for discovery methods and diagnostic plotting."""

import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from joshpy.cell_data import CellDataLoader
from joshpy.diagnostics import SimulationDiagnostics
from joshpy.registry import DataSummary, RunRegistry


class TestDataSummary:
    """Tests for DataSummary dataclass."""

    def test_str_with_all_data(self):
        """Test __str__ with complete data."""
        summary = DataSummary(
            sessions=2,
            configs=9,
            runs=27,
            cell_data_rows=10000,
            variables=["averageAge", "averageHeight"],
            entity_types=["patch"],
            step_range=(0, 10),
            replicate_range=(0, 2),
            spatial_extent={"lon": (-116.4, -115.4), "lat": (33.7, 34.0)},
            parameters=["maxGrowth", "survivalProb"],
        )

        result = str(summary)

        assert "Registry Data Summary" in result
        assert "Sessions: 2" in result
        assert "Configs:  9" in result
        assert "Runs:     27" in result
        assert "10,000" in result  # Formatted with comma
        assert "averageAge, averageHeight" in result
        assert "patch" in result
        assert "maxGrowth, survivalProb" in result
        assert "Steps: 0 - 10" in result
        assert "Replicates: 0 - 2" in result
        assert "lon [-116.40, -115.40]" in result
        assert "lat [33.70, 34.00]" in result

    def test_str_with_no_data(self):
        """Test __str__ with empty/no data."""
        summary = DataSummary(
            sessions=0,
            configs=0,
            runs=0,
            cell_data_rows=0,
            variables=[],
            entity_types=[],
            step_range=None,
            replicate_range=None,
            spatial_extent=None,
            parameters=[],
        )

        result = str(summary)

        assert "Sessions: 0" in result
        assert "Variables: (none)" in result
        assert "Entity types: (none)" in result
        assert "Parameters: (none)" in result
        # Should not contain step/replicate/spatial info
        assert "Steps:" not in result
        assert "Replicates:" not in result
        assert "Spatial extent:" not in result


class TestDiscoveryMethods:
    """Tests for RunRegistry discovery methods."""

    def setup_method(self):
        """Set up test registry with sample data."""
        self.registry = RunRegistry(":memory:")

        # Create session and config
        session_id = self.registry.create_session(
            experiment_name="test_experiment",
            simulation="TestSim",
        )
        self.session_id = session_id

        # Register configs with parameters
        self.registry.register_run(
            session_id=session_id,
            run_hash="abc123",
            josh_path="test.josh",
            config_content="test config 1",
            file_mappings=None,
            parameters={"maxGrowth": 10, "scenario": "baseline"},
        )
        self.registry.register_run(
            session_id=session_id,
            run_hash="def456",
            josh_path="test.josh",
            config_content="test config 2",
            file_mappings=None,
            parameters={"maxGrowth": 20, "scenario": "optimistic"},
        )

        # Create runs
        self.run_id1 = self.registry.start_run("abc123")
        self.run_id2 = self.registry.start_run("def456")

        # Load cell data
        loader = CellDataLoader(self.registry)

        # CSV for first config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,treeCount,avgHeight\n")
            for step in range(3):
                for rep in range(2):
                    f.write(f"{step},{rep},-116.0,34.0,{10 + step},{5.0 + step}\n")
            self.csv1 = Path(f.name)
        loader.load_csv(self.csv1, self.run_id1, "abc123", entity_type="patch")

        # CSV for second config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,treeCount,avgHeight\n")
            for step in range(3):
                f.write(f"{step},0,-116.1,34.1,{20 + step},{10.0 + step}\n")
            self.csv2 = Path(f.name)
        loader.load_csv(self.csv2, self.run_id2, "def456", entity_type="patch")

    def teardown_method(self):
        """Clean up temp files."""
        if hasattr(self, "csv1") and self.csv1.exists():
            self.csv1.unlink()
        if hasattr(self, "csv2") and self.csv2.exists():
            self.csv2.unlink()

    def test_list_variables(self):
        """Test listing available variables."""
        variables = self.registry.list_variables()

        assert "treeCount" in variables
        assert "avgHeight" in variables
        assert len(variables) == 2

    def test_list_variables_by_session(self):
        """Test listing variables filtered by session."""
        variables = self.registry.list_variables(session_id=self.session_id)

        assert "treeCount" in variables
        assert "avgHeight" in variables

    def test_list_parameters(self):
        """Test listing available parameters."""
        parameters = self.registry.list_parameters()

        assert "maxGrowth" in parameters
        assert "scenario" in parameters
        assert len(parameters) == 2

    def test_list_parameters_by_session(self):
        """Test listing parameters filtered by session."""
        parameters = self.registry.list_parameters(session_id=self.session_id)

        assert "maxGrowth" in parameters
        assert "scenario" in parameters

    def test_list_entity_types(self):
        """Test listing entity types."""
        entity_types = self.registry.list_entity_types()

        assert "patch" in entity_types
        assert len(entity_types) == 1

    def test_get_data_summary(self):
        """Test full data summary."""
        summary = self.registry.get_data_summary()

        assert summary.sessions == 1
        assert summary.configs == 2
        assert summary.runs == 2
        assert summary.cell_data_rows == 9  # 6 from csv1 + 3 from csv2
        assert "treeCount" in summary.variables
        assert "avgHeight" in summary.variables
        assert "patch" in summary.entity_types
        assert "maxGrowth" in summary.parameters
        assert "scenario" in summary.parameters
        assert summary.step_range == (0, 2)
        assert summary.replicate_range == (0, 1)
        assert summary.spatial_extent is not None
        assert summary.spatial_extent["lon"][0] == pytest.approx(-116.1, rel=0.01)
        assert summary.spatial_extent["lon"][1] == pytest.approx(-116.0, rel=0.01)

    def test_get_data_summary_by_session(self):
        """Test data summary filtered by session."""
        summary = self.registry.get_data_summary(session_id=self.session_id)

        assert summary.sessions == 1
        assert summary.configs == 2

    def test_get_data_summary_empty_registry(self):
        """Test data summary on empty registry."""
        empty_registry = RunRegistry(":memory:")
        summary = empty_registry.get_data_summary()

        assert summary.sessions == 0
        assert summary.configs == 0
        assert summary.runs == 0
        assert summary.cell_data_rows == 0
        assert summary.variables == []
        assert summary.entity_types == []
        assert summary.parameters == []
        assert summary.step_range is None
        assert summary.spatial_extent is None


class TestSimulationDiagnostics:
    """Tests for SimulationDiagnostics class."""

    def setup_method(self):
        """Set up test registry with sample data."""
        self.registry = RunRegistry(":memory:")

        # Create session and configs
        session_id = self.registry.create_session(
            experiment_name="test_experiment",
            simulation="TestSim",
        )
        self.session_id = session_id

        self.registry.register_run(
            session_id=session_id,
            run_hash="abc123",
            josh_path="test.josh",
            config_content="test config 1",
            file_mappings=None,
            parameters={"maxGrowth": 10},
        )
        self.registry.register_run(
            session_id=session_id,
            run_hash="def456",
            josh_path="test.josh",
            config_content="test config 2",
            file_mappings=None,
            parameters={"maxGrowth": 20},
        )

        # Create runs and load data
        loader = CellDataLoader(self.registry)

        # Data for first config - 2 spatial cells, 3 steps, 2 replicates
        self.run_id1 = self.registry.start_run("abc123")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,treeCount,avgHeight\n")
            for step in range(5):
                for rep in range(2):
                    # Cell 1
                    f.write(f"{step},{rep},-116.0,34.0,{10 + step + rep},{5.0 + step * 0.5}\n")
                    # Cell 2
                    f.write(f"{step},{rep},-116.1,34.1,{15 + step + rep},{6.0 + step * 0.5}\n")
            self.csv1 = Path(f.name)
        loader.load_csv(self.csv1, self.run_id1, "abc123", entity_type="patch")

        # Data for second config
        self.run_id2 = self.registry.start_run("def456")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("step,replicate,position.longitude,position.latitude,treeCount,avgHeight\n")
            for step in range(5):
                for rep in range(2):
                    f.write(f"{step},{rep},-116.0,34.0,{20 + step * 2 + rep},{10.0 + step}\n")
            self.csv2 = Path(f.name)
        loader.load_csv(self.csv2, self.run_id2, "def456", entity_type="patch")

        self.diag = SimulationDiagnostics(self.registry)

    def teardown_method(self):
        """Clean up temp files and close figures."""
        plt.close("all")
        if hasattr(self, "csv1") and self.csv1.exists():
            self.csv1.unlink()
        if hasattr(self, "csv2") and self.csv2.exists():
            self.csv2.unlink()

    def test_init_requires_registry(self):
        """Test that init requires a RunRegistry."""
        with pytest.raises(TypeError):
            SimulationDiagnostics("not a registry")  # type: ignore

    def test_validate_variable_raises_for_missing(self):
        """Test that missing variables raise ValueError."""
        with pytest.raises(ValueError, match="Export variable 'nonexistent' not found"):
            self.diag._validate_variable("nonexistent")

    def test_validate_parameter_raises_for_missing(self):
        """Test that missing parameters raise ValueError."""
        with pytest.raises(ValueError, match="Config parameter 'nonexistent' not found"):
            self.diag._validate_parameter("nonexistent")

    def test_validate_aggregation_raises_for_invalid(self):
        """Test that invalid aggregation modes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid aggregation"):
            self.diag._validate_aggregation("invalid")

    def test_plot_timeseries_returns_figure(self):
        """Test that plot_timeseries returns a Figure."""
        fig = self.diag.plot_timeseries(
            "treeCount", run_hash="abc123", show=False
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_timeseries_with_mean_aggregation(self):
        """Test time series with mean aggregation."""
        fig = self.diag.plot_timeseries(
            "treeCount", run_hash="abc123", aggregate="mean", show=False
        )

        assert isinstance(fig, plt.Figure)
        # Should have one axes
        assert len(fig.axes) == 1

    def test_plot_timeseries_with_sum_aggregation(self):
        """Test time series with sum aggregation."""
        fig = self.diag.plot_timeseries(
            "treeCount", run_hash="abc123", aggregate="sum", show=False
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_timeseries_with_none_aggregation_warns(self):
        """Test that aggregate='none' emits a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = self.diag.plot_timeseries(
                "treeCount", run_hash="abc123", aggregate="none", show=False
            )

            assert len(w) == 1
            assert "aggregate='none'" in str(w[0].message)
            assert isinstance(fig, plt.Figure)

    def test_plot_timeseries_with_step_range(self):
        """Test time series with step range filter."""
        fig = self.diag.plot_timeseries(
            "treeCount",
            run_hash="abc123",
            step_range=(1, 3),
            show=False,
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_timeseries_with_parameter_filter(self):
        """Test time series with parameter filter."""
        fig = self.diag.plot_timeseries(
            "treeCount", maxGrowth=10, show=False
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_timeseries_multiple_configs_warns(self):
        """Test that multiple matching configs emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = self.diag.plot_timeseries(
                "treeCount", session_id=self.session_id, show=False
            )

            # Should warn about multiple configs
            warning_messages = [str(warning.message) for warning in w]
            assert any("Matched 2 configs" in msg for msg in warning_messages)
            assert isinstance(fig, plt.Figure)

    def test_plot_timeseries_no_data_raises(self):
        """Test that no matching data raises ValueError."""
        with pytest.raises(ValueError, match="No data found"):
            self.diag.plot_timeseries(
                "treeCount", run_hash="nonexistent", show=False
            )

    def test_plot_timeseries_missing_variable_raises(self):
        """Test that missing variable raises ValueError."""
        with pytest.raises(ValueError, match="Export variable 'missing' not found"):
            self.diag.plot_timeseries("missing", run_hash="abc123", show=False)

    def test_plot_comparison_timeseries(self):
        """Test comparison plot as time series."""
        fig = self.diag.plot_comparison(
            "treeCount", group_by="maxGrowth", show=False
        )

        assert isinstance(fig, plt.Figure)
        # Should have legend
        ax = fig.axes[0]
        assert ax.get_legend() is not None

    def test_plot_comparison_bar_chart(self):
        """Test comparison plot as bar chart at specific step."""
        fig = self.diag.plot_comparison(
            "treeCount", group_by="maxGrowth", step=2, show=False
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_comparison_aggregate_none_raises(self):
        """Test that aggregate='none' raises for comparison."""
        with pytest.raises(ValueError, match="aggregate='none' is not supported"):
            self.diag.plot_comparison(
                "treeCount", group_by="maxGrowth", aggregate="none", show=False
            )

    def test_plot_comparison_missing_parameter_raises(self):
        """Test that missing parameter raises ValueError."""
        with pytest.raises(ValueError, match="'missing' not found"):
            self.diag.plot_comparison(
                "treeCount", group_by="missing", show=False
            )

    def test_plot_spatial_returns_figure(self):
        """Test that plot_spatial returns a Figure."""
        fig = self.diag.plot_spatial(
            "treeCount", step=2, run_hash="abc123", show=False
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_spatial_no_data_raises(self):
        """Test that no spatial data raises ValueError."""
        with pytest.raises(ValueError, match="No spatial data found"):
            self.diag.plot_spatial(
                "treeCount", step=2, run_hash="nonexistent", show=False
            )

    def test_plot_spatial_missing_variable_raises(self):
        """Test that missing variable raises ValueError."""
        with pytest.raises(ValueError, match="Export variable 'missing' not found"):
            self.diag.plot_spatial(
                "missing", step=2, run_hash="abc123", show=False
            )

    def test_plot_with_custom_title(self):
        """Test that custom title is applied."""
        fig = self.diag.plot_timeseries(
            "treeCount",
            run_hash="abc123",
            title="Custom Title",
            show=False,
        )

        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"


class TestDiagnosticsEdgeCases:
    """Tests for edge cases in diagnostics."""

    def test_empty_registry(self):
        """Test diagnostics with empty registry."""
        registry = RunRegistry(":memory:")
        diag = SimulationDiagnostics(registry)

        with pytest.raises(ValueError, match="Export variable .* not found"):
            diag.plot_timeseries("anything", show=False)

    def test_valid_aggregation_modes(self):
        """Test all valid aggregation modes are accepted."""
        registry = RunRegistry(":memory:")
        diag = SimulationDiagnostics(registry)

        for mode in ["none", "mean", "sum", "min", "max"]:
            diag._validate_aggregation(mode)  # Should not raise


class TestNumericSorting:
    """Tests for numeric sorting of parameter values."""

    def test_sort_key_numeric_values(self):
        """Test that numeric string values sort numerically."""
        from joshpy.diagnostics import _sort_key_numeric_then_string

        values = ["10", "100", "20", "2", "1"]
        sorted_values = sorted(values, key=_sort_key_numeric_then_string)
        assert sorted_values == ["1", "2", "10", "20", "100"]

    def test_sort_key_mixed_values(self):
        """Test that mixed numeric/string values sort correctly."""
        from joshpy.diagnostics import _sort_key_numeric_then_string

        values = ["baseline", "10", "optimistic", "2", "100"]
        sorted_values = sorted(values, key=_sort_key_numeric_then_string)
        # Numbers first (sorted numerically), then strings (sorted alphabetically)
        assert sorted_values == ["2", "10", "100", "baseline", "optimistic"]

    def test_sort_key_float_values(self):
        """Test that float string values sort numerically."""
        from joshpy.diagnostics import _sort_key_numeric_then_string

        values = ["0.1", "0.01", "1.0", "0.5"]
        sorted_values = sorted(values, key=_sort_key_numeric_then_string)
        assert sorted_values == ["0.01", "0.1", "0.5", "1.0"]

    def test_sort_key_negative_values(self):
        """Test that negative values sort correctly."""
        from joshpy.diagnostics import _sort_key_numeric_then_string

        values = ["-10", "0", "10", "-5"]
        sorted_values = sorted(values, key=_sort_key_numeric_then_string)
        assert sorted_values == ["-10", "-5", "0", "10"]
