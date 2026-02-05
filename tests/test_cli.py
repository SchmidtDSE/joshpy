"""Unit tests for the cli module."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from joshpy.cli import (
    CLIResult,
    DiscoverConfigConfig,
    InspectJshdConfig,
    JoshCLI,
    PreprocessConfig,
    RunConfig,
    RunRemoteConfig,
    ValidateConfig,
)


class TestCLIResult(unittest.TestCase):
    """Tests for CLIResult class."""

    def test_success_property_true(self):
        """success should be True for exit_code 0."""
        result = CLIResult(
            exit_code=0,
            stdout="output",
            stderr="",
            command=["java", "-jar", "test.jar"],
        )
        self.assertTrue(result.success)

    def test_success_property_false(self):
        """success should be False for non-zero exit_code."""
        result = CLIResult(
            exit_code=1,
            stdout="",
            stderr="error",
            command=["java", "-jar", "test.jar"],
        )
        self.assertFalse(result.success)

    def test_success_property_negative(self):
        """success should be False for negative exit_code (timeout)."""
        result = CLIResult(
            exit_code=-1,
            stdout="",
            stderr="timeout",
            command=["java", "-jar", "test.jar"],
        )
        self.assertFalse(result.success)


class TestRunConfig(unittest.TestCase):
    """Tests for RunConfig dataclass."""

    def test_basic_creation(self):
        """Basic config creation with required fields."""
        config = RunConfig(
            script=Path("simulation.josh"),
            simulation="Main",
        )
        self.assertEqual(config.script, Path("simulation.josh"))
        self.assertEqual(config.simulation, "Main")
        self.assertEqual(config.replicates, 1)
        self.assertFalse(config.use_float64)

    def test_all_fields(self):
        """Config with all fields set."""
        config = RunConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            replicates=5,
            data={"config": Path("config.jshc"), "data": Path("data.jshd")},
            custom_tags={"param1": "value1"},
            crs="EPSG:4326",
            output_format="csv",
            output=Path("output.csv"),
            parallel=True,
            use_float64=True,
            verbose=True,
            upload_source=True,
            upload_config=True,
            upload_data=True,
            output_steps="0-10,50",
            seed=42,
        )
        self.assertEqual(config.replicates, 5)
        self.assertEqual(config.crs, "EPSG:4326")
        self.assertTrue(config.parallel)
        self.assertEqual(config.seed, 42)

    def test_frozen(self):
        """Config should be immutable."""
        config = RunConfig(
            script=Path("simulation.josh"),
            simulation="Main",
        )
        with self.assertRaises(AttributeError):
            config.simulation = "Other"  # type: ignore


class TestRunRemoteConfig(unittest.TestCase):
    """Tests for RunRemoteConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields."""
        config = RunRemoteConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            api_key="test-api-key",
        )
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.replicates, 1)
        self.assertIsNone(config.endpoint)

    def test_with_endpoint(self):
        """Config with custom endpoint."""
        config = RunRemoteConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            api_key="test-api-key",
            endpoint="https://custom.josh.cloud",
        )
        self.assertEqual(config.endpoint, "https://custom.josh.cloud")


class TestPreprocessConfig(unittest.TestCase):
    """Tests for PreprocessConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields."""
        config = PreprocessConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            data_file=Path("temperature.nc"),
            variable="temp",
            units="K",
            output=Path("temperature.jshd"),
        )
        self.assertEqual(config.variable, "temp")
        self.assertEqual(config.units, "K")
        self.assertFalse(config.amend)

    def test_with_coordinates(self):
        """Config with coordinate options."""
        config = PreprocessConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            data_file=Path("data.nc"),
            variable="rainfall",
            units="mm/year",
            output=Path("data.jshd"),
            amend=True,
            x_coord="longitude",
            y_coord="latitude",
            time_coord="time",
        )
        self.assertTrue(config.amend)
        self.assertEqual(config.x_coord, "longitude")


class TestValidateConfig(unittest.TestCase):
    """Tests for ValidateConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields."""
        config = ValidateConfig(script=Path("simulation.josh"))
        self.assertFalse(config.verbose)
        self.assertFalse(config.upload_source)


class TestDiscoverConfigConfig(unittest.TestCase):
    """Tests for DiscoverConfigConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields."""
        config = DiscoverConfigConfig(script=Path("simulation.josh"))
        self.assertEqual(config.script, Path("simulation.josh"))


class TestInspectJshdConfig(unittest.TestCase):
    """Tests for InspectJshdConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with all required fields."""
        config = InspectJshdConfig(
            jshd_file=Path("data.jshd"),
            variable="temperature",
            timestep=5,
            x=10,
            y=20,
        )
        self.assertEqual(config.variable, "temperature")
        self.assertEqual(config.timestep, 5)
        self.assertEqual(config.x, 10)
        self.assertEqual(config.y, 20)


class TestJoshCLI(unittest.TestCase):
    """Tests for JoshCLI class."""

    # Use the local jar that exists in the repo
    JAR_PATH = Path(__file__).parent.parent / "jar" / "joshsim-fat.jar"

    def test_init_with_path(self):
        """CLI should accept explicit jar path."""
        cli = JoshCLI(josh_jar=self.JAR_PATH)
        self.assertEqual(cli._resolved_jar, self.JAR_PATH.resolve())

    def test_init_with_missing_jar_raises(self):
        """CLI should raise FileNotFoundError for missing jar."""
        with self.assertRaises(FileNotFoundError):
            JoshCLI(josh_jar=Path("/nonexistent/joshsim.jar"))

    @patch("subprocess.run")
    def test_run_basic(self, mock_run):
        """run() should build correct command."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="output",
            stderr="",
        )

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        result = cli.run(config)

        self.assertTrue(result.success)
        # Verify command structure
        cmd = mock_run.call_args[0][0]
        self.assertIn("java", cmd)
        self.assertIn("-jar", cmd)
        self.assertIn("run", cmd)
        self.assertIn("Main", cmd)
        self.assertTrue(any("simulation.josh" in c for c in cmd))

    @patch("subprocess.run")
    def test_run_with_replicates(self, mock_run):
        """run() should include --replicates when > 1."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            replicates=5,
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        replicates_idx = cmd.index("--replicates")
        self.assertEqual(cmd[replicates_idx + 1], "5")

    @patch("subprocess.run")
    def test_run_with_data_files(self, mock_run):
        """run() should include --data flags for data files."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data={"editor": Path("/path/to/editor.jshc")},
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--data", cmd)
        # Find the data value
        data_idx = cmd.index("--data")
        self.assertIn("editor=", cmd[data_idx + 1])

    @patch("subprocess.run")
    def test_run_with_custom_tags(self, mock_run):
        """run() should include --custom-tag flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            custom_tags={"param1": "value1", "param2": "value2"},
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        tag_indices = [i for i, c in enumerate(cmd) if c == "--custom-tag"]
        tags = [cmd[i + 1] for i in tag_indices]
        self.assertIn("param1=value1", tags)
        self.assertIn("param2=value2", tags)

    @patch("subprocess.run")
    def test_run_with_options(self, mock_run):
        """run() should include all optional flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            crs="EPSG:4326",
            output_format="csv",
            output=Path("/path/to/output.csv"),
            parallel=True,
            use_float64=True,
            verbose=True,
            output_steps="0-10",
            seed=42,
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--crs", cmd)
        self.assertIn("EPSG:4326", cmd)
        self.assertIn("--output-format", cmd)
        self.assertIn("csv", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("--parallel", cmd)
        self.assertIn("--use-float-64", cmd)
        self.assertIn("--verbose", cmd)
        self.assertIn("--output-steps", cmd)
        self.assertIn("0-10", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)

    @patch("subprocess.run")
    def test_run_remote(self, mock_run):
        """run_remote() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunRemoteConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            api_key="test-key",
            replicates=3,
        )

        result = cli.run_remote(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("runRemote", cmd)
        self.assertIn("--api-key", cmd)
        self.assertIn("test-key", cmd)
        self.assertIn("--replicates", cmd)
        self.assertIn("3", cmd)

    @patch("subprocess.run")
    def test_preprocess(self, mock_run):
        """preprocess() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = PreprocessConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data_file=Path("/path/to/temperature.nc"),
            variable="temp",
            units="K",
            output=Path("/path/to/temperature.jshd"),
            amend=True,
            x_coord="longitude",
        )

        result = cli.preprocess(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("preprocess", cmd)
        self.assertIn("temp", cmd)
        self.assertIn("K", cmd)
        self.assertIn("--amend", cmd)
        self.assertIn("--x-coord", cmd)
        self.assertIn("longitude", cmd)

    @patch("subprocess.run")
    def test_validate(self, mock_run):
        """validate() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = ValidateConfig(
            script=Path("/path/to/simulation.josh"),
            verbose=True,
        )

        result = cli.validate(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("validate", cmd)
        self.assertIn("--verbose", cmd)

    @patch("subprocess.run")
    def test_discover_config(self, mock_run):
        """discover_config() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="configVar1\nconfigVar2", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = DiscoverConfigConfig(script=Path("/path/to/simulation.josh"))

        result = cli.discover_config(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("discoverConfig", cmd)

    @patch("subprocess.run")
    def test_inspect_jshd(self, mock_run):
        """inspect_jshd() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="value: 25.5", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = InspectJshdConfig(
            jshd_file=Path("/path/to/data.jshd"),
            variable="temperature",
            timestep=5,
            x=10,
            y=20,
        )

        result = cli.inspect_jshd(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("inspectJshd", cmd)
        self.assertIn("temperature", cmd)
        self.assertIn("5", cmd)
        self.assertIn("10", cmd)
        self.assertIn("20", cmd)

    @patch("subprocess.run")
    def test_timeout_handling(self, mock_run):
        """CLI should handle timeout gracefully."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["java"], timeout=30)

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        result = cli.run(config, timeout=30)

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("timed out", result.stderr)

    @patch("subprocess.run")
    def test_exception_handling(self, mock_run):
        """CLI should handle exceptions gracefully."""
        mock_run.side_effect = OSError("Java not found")

        cli = JoshCLI(josh_jar=self.JAR_PATH)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        result = cli.run(config)

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("Java not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
