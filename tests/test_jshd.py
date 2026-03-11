"""Unit tests for the jshd module."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from joshpy.jshd import JshdData, JshdMetadata, inspect_jshd, plot_jshd


class TestJshdMetadata(unittest.TestCase):
    """Tests for JshdMetadata dataclass."""

    def test_from_json(self):
        """from_json() should parse CLI JSON output correctly."""
        json_data = {
            "minX": 0,
            "maxX": 922,
            "minY": 0,
            "maxY": 1112,
            "minTimestep": 0,
            "maxTimestep": 1,
            "width": 923,
            "height": 1113,
            "units": "K",
            "csv": "/tmp/output.csv",
        }

        meta = JshdMetadata.from_json(json_data)

        self.assertEqual(meta.min_x, 0)
        self.assertEqual(meta.max_x, 922)
        self.assertEqual(meta.min_y, 0)
        self.assertEqual(meta.max_y, 1112)
        self.assertEqual(meta.min_timestep, 0)
        self.assertEqual(meta.max_timestep, 1)
        self.assertEqual(meta.width, 923)
        self.assertEqual(meta.height, 1113)
        self.assertEqual(meta.units, "K")
        self.assertEqual(meta.csv_path, Path("/tmp/output.csv"))

    def test_num_timesteps_property(self):
        """num_timesteps should return correct count."""
        meta = JshdMetadata(
            min_x=0,
            max_x=9,
            min_y=0,
            max_y=9,
            min_timestep=0,
            max_timestep=4,
            width=10,
            height=10,
            units="meters",
            csv_path=Path("/tmp/test.csv"),
        )
        self.assertEqual(meta.num_timesteps, 5)

    def test_num_timesteps_single(self):
        """num_timesteps should be 1 for single timestep."""
        meta = JshdMetadata(
            min_x=0,
            max_x=9,
            min_y=0,
            max_y=9,
            min_timestep=0,
            max_timestep=0,
            width=10,
            height=10,
            units="meters",
            csv_path=Path("/tmp/test.csv"),
        )
        self.assertEqual(meta.num_timesteps, 1)

    def test_num_cells_property(self):
        """num_cells should return width * height."""
        meta = JshdMetadata(
            min_x=0,
            max_x=9,
            min_y=0,
            max_y=19,
            min_timestep=0,
            max_timestep=0,
            width=10,
            height=20,
            units="meters",
            csv_path=Path("/tmp/test.csv"),
        )
        self.assertEqual(meta.num_cells, 200)

    def test_empty_units(self):
        """Should handle empty units string."""
        json_data = {
            "minX": 0,
            "maxX": 9,
            "minY": 0,
            "maxY": 9,
            "minTimestep": 0,
            "maxTimestep": 0,
            "width": 10,
            "height": 10,
            "units": "",
            "csv": "/tmp/test.csv",
        }

        meta = JshdMetadata.from_json(json_data)
        self.assertEqual(meta.units, "")


class TestJshdData(unittest.TestCase):
    """Tests for JshdData dataclass."""

    def setUp(self):
        """Create sample metadata and dataframe for tests."""
        self.metadata = JshdMetadata(
            min_x=0,
            max_x=2,
            min_y=0,
            max_y=1,
            min_timestep=0,
            max_timestep=1,
            width=3,
            height=2,
            units="percent",
            csv_path=Path("/tmp/test.csv"),
        )

        # Create CSV data in the expected order: timestep -> y -> x
        # For timestep=0: y=0,x=0,1,2 then y=1,x=0,1,2
        # For timestep=1: same pattern
        rows = []
        val = 0
        for t in range(2):  # timesteps 0, 1
            for y in range(2):  # y 0, 1
                for x in range(3):  # x 0, 1, 2
                    rows.append({"x": x, "y": y, "timestep": t, "value": val})
                    val += 1

        self.df = pd.DataFrame(rows)

    def test_to_array_single_timestep(self):
        """to_array(timestep) should return 2D array (height, width)."""
        data = JshdData(metadata=self.metadata, df=self.df)

        arr = data.to_array(timestep=0)

        self.assertEqual(arr.shape, (2, 3))  # height=2, width=3
        # First timestep values are 0-5
        # Row order in CSV: y=0 (x=0,1,2), y=1 (x=0,1,2)
        # So arr[0] = [0, 1, 2] (y=0), arr[1] = [3, 4, 5] (y=1)
        np.testing.assert_array_equal(arr[0], [0, 1, 2])
        np.testing.assert_array_equal(arr[1], [3, 4, 5])

    def test_to_array_second_timestep(self):
        """to_array(timestep=1) should return correct values."""
        data = JshdData(metadata=self.metadata, df=self.df)

        arr = data.to_array(timestep=1)

        self.assertEqual(arr.shape, (2, 3))
        # Second timestep values are 6-11
        np.testing.assert_array_equal(arr[0], [6, 7, 8])
        np.testing.assert_array_equal(arr[1], [9, 10, 11])

    def test_to_array_all_timesteps(self):
        """to_array() without timestep should return 3D array."""
        data = JshdData(metadata=self.metadata, df=self.df)

        arr = data.to_array()

        self.assertEqual(arr.shape, (2, 2, 3))  # (T, H, W)
        np.testing.assert_array_equal(arr[0, 0], [0, 1, 2])
        np.testing.assert_array_equal(arr[1, 0], [6, 7, 8])

    def test_to_array_invalid_timestep(self):
        """to_array() should raise ValueError for invalid timestep."""
        data = JshdData(metadata=self.metadata, df=self.df)

        with self.assertRaises(ValueError) as ctx:
            data.to_array(timestep=5)

        self.assertIn("out of range", str(ctx.exception))
        self.assertIn("[0, 1]", str(ctx.exception))

    def test_to_array_negative_timestep(self):
        """to_array() should raise ValueError for negative timestep."""
        data = JshdData(metadata=self.metadata, df=self.df)

        with self.assertRaises(ValueError):
            data.to_array(timestep=-1)

    def test_cleanup_removes_temp_dir(self):
        """cleanup() should remove temp directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix="jshd_test_"))
        (temp_dir / "test.csv").touch()

        data = JshdData(
            metadata=self.metadata,
            df=self.df,
            _temp_dir=temp_dir,
        )

        self.assertTrue(temp_dir.exists())
        data.cleanup()
        self.assertFalse(temp_dir.exists())
        self.assertIsNone(data._temp_dir)

    def test_cleanup_noop_without_temp_dir(self):
        """cleanup() should be safe to call when no temp dir."""
        data = JshdData(
            metadata=self.metadata,
            df=self.df,
            _temp_dir=None,
        )

        # Should not raise
        data.cleanup()


class TestInspectJshd(unittest.TestCase):
    """Tests for inspect_jshd function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        cli = MagicMock()

        with self.assertRaises(FileNotFoundError) as ctx:
            inspect_jshd(cli, Path("/nonexistent/file.jshd"))

        self.assertIn("not found", str(ctx.exception))

    @patch("joshpy.jshd.pd.read_csv")
    def test_cli_execution(self, mock_read_csv):
        """Should call CLI with correct arguments."""
        # Create a real temp file to pass the exists() check
        with tempfile.NamedTemporaryFile(suffix=".jshd", delete=False) as f:
            jshd_path = Path(f.name)

        try:
            cli = MagicMock()
            cli._execute.return_value = MagicMock(
                success=True,
                exit_code=0,
                stdout=json.dumps({
                    "minX": 0,
                    "maxX": 9,
                    "minY": 0,
                    "maxY": 9,
                    "minTimestep": 0,
                    "maxTimestep": 0,
                    "width": 10,
                    "height": 10,
                    "units": "percent",
                    "csv": "/tmp/dump.csv",
                }),
                stderr="",
            )
            mock_read_csv.return_value = pd.DataFrame({
                "x": [0],
                "y": [0],
                "timestep": [0],
                "value": [50.0],
            })

            inspect_jshd(cli, jshd_path)

            # Verify CLI was called with correct args
            call_args = cli._execute.call_args[0][0]
            self.assertEqual(call_args[0], "inspectJshd")
            # --to-csv uses '=' syntax: --to-csv=/path/to/file.csv
            self.assertTrue(any("--to-csv=" in arg for arg in call_args))
            self.assertEqual(call_args[2], "data")  # Default variable

        finally:
            jshd_path.unlink()

    @patch("joshpy.jshd.pd.read_csv")
    def test_cli_failure(self, mock_read_csv):
        """Should raise RuntimeError on CLI failure."""
        with tempfile.NamedTemporaryFile(suffix=".jshd", delete=False) as f:
            jshd_path = Path(f.name)

        try:
            cli = MagicMock()
            cli._execute.return_value = MagicMock(
                success=False,
                exit_code=6,
                stdout="",
                stderr="Variable not found in JSHD",
            )

            with self.assertRaises(RuntimeError) as ctx:
                inspect_jshd(cli, jshd_path)

            self.assertIn("exit 6", str(ctx.exception))
            self.assertIn("Variable not found", str(ctx.exception))

        finally:
            jshd_path.unlink()

    @patch("joshpy.jshd.pd.read_csv")
    def test_custom_variable(self, mock_read_csv):
        """Should pass custom variable name to CLI."""
        with tempfile.NamedTemporaryFile(suffix=".jshd", delete=False) as f:
            jshd_path = Path(f.name)

        try:
            cli = MagicMock()
            cli._execute.return_value = MagicMock(
                success=True,
                exit_code=0,
                stdout=json.dumps({
                    "minX": 0,
                    "maxX": 9,
                    "minY": 0,
                    "maxY": 9,
                    "minTimestep": 0,
                    "maxTimestep": 0,
                    "width": 10,
                    "height": 10,
                    "units": "K",
                    "csv": "/tmp/dump.csv",
                }),
                stderr="",
            )
            mock_read_csv.return_value = pd.DataFrame({
                "x": [0],
                "y": [0],
                "timestep": [0],
                "value": [273.15],
            })

            inspect_jshd(cli, jshd_path, variable="temperature")

            call_args = cli._execute.call_args[0][0]
            self.assertEqual(call_args[2], "temperature")

        finally:
            jshd_path.unlink()


class TestPlotJshd(unittest.TestCase):
    """Tests for plot_jshd function."""

    def setUp(self):
        """Create sample JshdData for tests."""
        self.metadata = JshdMetadata(
            min_x=0,
            max_x=4,
            min_y=0,
            max_y=3,
            min_timestep=0,
            max_timestep=1,
            width=5,
            height=4,
            units="percent",
            csv_path=Path("/tmp/test.csv"),
        )

        # Create gradient data (increasing with x)
        rows = []
        for t in range(2):
            for y in range(4):
                for x in range(5):
                    rows.append({
                        "x": x,
                        "y": y,
                        "timestep": t,
                        "value": x * 25.0 + t * 5,
                    })

        self.df = pd.DataFrame(rows)
        self.data = JshdData(
            metadata=self.metadata,
            df=self.df,
            source_path=Path("test.jshd"),
        )

    def test_creates_figure(self):
        """plot_jshd should return a Figure object."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing

        fig = plot_jshd(self.data, timestep=0, show=False)

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Main axes + colorbar

    def test_invalid_timestep(self):
        """Should raise ValueError for invalid timestep."""
        import matplotlib

        matplotlib.use("Agg")

        with self.assertRaises(ValueError) as ctx:
            plot_jshd(self.data, timestep=5, show=False)

        self.assertIn("out of range", str(ctx.exception))

    def test_custom_title(self):
        """Should use custom title when provided."""
        import matplotlib

        matplotlib.use("Agg")

        fig = plot_jshd(self.data, timestep=0, title="Custom Title", show=False)

        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Custom Title")

    def test_default_title(self):
        """Should generate default title from filename and timestep."""
        import matplotlib

        matplotlib.use("Agg")

        fig = plot_jshd(self.data, timestep=0, show=False)

        ax = fig.axes[0]
        self.assertIn("test.jshd", ax.get_title())
        self.assertIn("timestep 0", ax.get_title())

    def test_colorbar_label_with_units(self):
        """Colorbar should include units."""
        import matplotlib

        matplotlib.use("Agg")

        fig = plot_jshd(self.data, timestep=0, show=False)

        # Check colorbar label (second axes)
        cbar_ax = fig.axes[1]
        self.assertIn("percent", cbar_ax.get_ylabel())

    def test_colorbar_label_without_units(self):
        """Colorbar should handle empty units."""
        import matplotlib

        matplotlib.use("Agg")

        # Create data with no units
        meta_no_units = JshdMetadata(
            min_x=0,
            max_x=4,
            min_y=0,
            max_y=3,
            min_timestep=0,
            max_timestep=0,
            width=5,
            height=4,
            units="",
            csv_path=Path("/tmp/test.csv"),
        )
        data_no_units = JshdData(
            metadata=meta_no_units,
            df=self.df[self.df["timestep"] == 0].copy(),
            source_path=Path("test.jshd"),
        )

        fig = plot_jshd(data_no_units, timestep=0, show=False)

        cbar_ax = fig.axes[1]
        self.assertEqual(cbar_ax.get_ylabel(), "Value")


if __name__ == "__main__":
    unittest.main()
