"""Unit tests for the cli module."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from joshpy.cli import (
    CLIResult,
    DiscoverConfigConfig,
    ExportFileInfo,
    ExportPaths,
    InspectExportsConfig,
    InspectJshdConfig,
    JfrConfig,
    JoshCLI,
    NetcdfPreprocessConfig,
    GeotiffPreprocessConfig,
    CsvPreprocessConfig,
    RunConfig,
    RunRemoteConfig,
    ValidateConfig,
)
from joshpy.jar import JarMode


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


class TestNetcdfPreprocessConfig(unittest.TestCase):
    """Tests for NetcdfPreprocessConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields and defaults."""
        config = NetcdfPreprocessConfig(
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
        # Check defaults
        self.assertEqual(config.x_coord, "lon")
        self.assertEqual(config.y_coord, "lat")
        self.assertEqual(config.time_coord, "time")
        self.assertIsNone(config.timestep)

    def test_with_custom_coordinates(self):
        """Config with custom coordinate names."""
        config = NetcdfPreprocessConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            data_file=Path("data.nc"),
            variable="rainfall",
            units="mm/year",
            output=Path("data.jshd"),
            amend=True,
            x_coord="longitude",
            y_coord="latitude",
            time_coord="calendar_year",
            timestep=5,
        )
        self.assertTrue(config.amend)
        self.assertEqual(config.x_coord, "longitude")
        self.assertEqual(config.y_coord, "latitude")
        self.assertEqual(config.time_coord, "calendar_year")
        self.assertEqual(config.timestep, 5)

    def test_invalid_extension(self):
        """Should raise ValueError for non-NetCDF file."""
        with self.assertRaises(ValueError) as ctx:
            NetcdfPreprocessConfig(
                script=Path("simulation.josh"),
                simulation="Main",
                data_file=Path("data.tif"),
                variable="temp",
                units="K",
                output=Path("out.jshd"),
            )
        self.assertIn(".nc", str(ctx.exception))


class TestGeotiffPreprocessConfig(unittest.TestCase):
    """Tests for GeotiffPreprocessConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields."""
        config = GeotiffPreprocessConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            data_file=Path("cover.tif"),
            band=0,
            units="percent",
            output=Path("cover.jshd"),
            timestep=0,
        )
        self.assertEqual(config.band, 0)
        self.assertEqual(config.timestep, 0)
        self.assertFalse(config.amend)

    def test_with_crs(self):
        """Config with CRS specified."""
        config = GeotiffPreprocessConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            data_file=Path("cover.tiff"),
            band=1,
            units="meters",
            output=Path("elev.jshd"),
            timestep=0,
            crs="EPSG:4326",
            amend=True,
        )
        self.assertEqual(config.band, 1)
        self.assertEqual(config.crs, "EPSG:4326")
        self.assertTrue(config.amend)

    def test_negative_band(self):
        """Should raise ValueError for negative band."""
        with self.assertRaises(ValueError) as ctx:
            GeotiffPreprocessConfig(
                script=Path("simulation.josh"),
                simulation="Main",
                data_file=Path("cover.tif"),
                band=-1,
                units="percent",
                output=Path("cover.jshd"),
                timestep=0,
            )
        self.assertIn("band", str(ctx.exception))

    def test_negative_timestep(self):
        """Should raise ValueError for negative timestep."""
        with self.assertRaises(ValueError) as ctx:
            GeotiffPreprocessConfig(
                script=Path("simulation.josh"),
                simulation="Main",
                data_file=Path("cover.tif"),
                band=0,
                units="percent",
                output=Path("cover.jshd"),
                timestep=-1,
            )
        self.assertIn("timestep", str(ctx.exception))

    def test_invalid_extension(self):
        """Should raise ValueError for non-GeoTIFF file."""
        with self.assertRaises(ValueError) as ctx:
            GeotiffPreprocessConfig(
                script=Path("simulation.josh"),
                simulation="Main",
                data_file=Path("data.nc"),
                band=0,
                units="percent",
                output=Path("out.jshd"),
                timestep=0,
            )
        self.assertIn(".tif", str(ctx.exception))


class TestCsvPreprocessConfig(unittest.TestCase):
    """Tests for CsvPreprocessConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields."""
        config = CsvPreprocessConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            data_file=Path("stations.csv"),
            variable="precipitation",
            units="mm",
            output=Path("precip.jshd"),
            timestep=0,
        )
        self.assertEqual(config.variable, "precipitation")
        self.assertEqual(config.timestep, 0)
        self.assertFalse(config.amend)

    def test_with_options(self):
        """Config with optional fields."""
        config = CsvPreprocessConfig(
            script=Path("simulation.josh"),
            simulation="Main",
            data_file=Path("points.csv"),
            variable="elevation",
            units="m",
            output=Path("elev.jshd"),
            timestep=5,
            amend=True,
            crs="EPSG:4326",
            parallel=True,
        )
        self.assertEqual(config.timestep, 5)
        self.assertTrue(config.amend)
        self.assertEqual(config.crs, "EPSG:4326")
        self.assertTrue(config.parallel)

    def test_negative_timestep(self):
        """Should raise ValueError for negative timestep."""
        with self.assertRaises(ValueError) as ctx:
            CsvPreprocessConfig(
                script=Path("simulation.josh"),
                simulation="Main",
                data_file=Path("data.csv"),
                variable="temp",
                units="K",
                output=Path("out.jshd"),
                timestep=-1,
            )
        self.assertIn("timestep", str(ctx.exception))

    def test_invalid_extension(self):
        """Should raise ValueError for non-CSV file."""
        with self.assertRaises(ValueError) as ctx:
            CsvPreprocessConfig(
                script=Path("simulation.josh"),
                simulation="Main",
                data_file=Path("data.nc"),
                variable="temp",
                units="K",
                output=Path("out.jshd"),
                timestep=0,
            )
        self.assertIn(".csv", str(ctx.exception))


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

    # Use the DEV jar mode for tests
    JAR_MODE = JarMode.DEV

    def test_init_with_path(self):
        """CLI should accept JarMode."""
        cli = JoshCLI(josh_jar=self.JAR_MODE)
        # Just verify it resolved to a path that exists
        self.assertTrue(cli._resolved_jar.exists())

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

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data={"editor": Path("/path/to/editor.jshc")},
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--data", cmd)
        # Find the data value — name gets extension appended when missing
        data_idx = cmd.index("--data")
        self.assertIn("editor.jshc=", cmd[data_idx + 1])

    @patch("subprocess.run")
    def test_run_with_custom_tags(self, mock_run):
        """run() should include --custom-tag flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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
    def test_run_with_enable_profiler(self, mock_run):
        """run() should include --enable-profiler when enabled."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            enable_profiler=True,
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--enable-profiler", cmd)

    @patch("subprocess.run")
    def test_run_without_enable_profiler(self, mock_run):
        """run() should not include --enable-profiler by default."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--enable-profiler", cmd)

    @patch("subprocess.run")
    def test_run_remote(self, mock_run):
        """run_remote() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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
    def test_preprocess_netcdf(self, mock_run):
        """preprocess() should build correct command for NetCDF."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = NetcdfPreprocessConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data_file=Path("/path/to/temperature.nc"),
            variable="temp",
            units="K",
            output=Path("/path/to/temperature.jshd"),
            amend=True,
            x_coord="longitude",
            y_coord="latitude",
            time_coord="time",
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
        self.assertIn("--y-coord", cmd)
        self.assertIn("latitude", cmd)
        self.assertIn("--time-dim", cmd)
        self.assertIn("time", cmd)
        # NetCDF without timestep should NOT have --timestep
        self.assertNotIn("--timestep", cmd)

    @patch("subprocess.run")
    def test_preprocess_netcdf_with_timestep(self, mock_run):
        """preprocess() should include --timestep for NetCDF when specified."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = NetcdfPreprocessConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data_file=Path("/path/to/temperature.nc"),
            variable="temp",
            units="K",
            output=Path("/path/to/temperature.jshd"),
            timestep=5,
        )

        result = cli.preprocess(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--timestep", cmd)
        self.assertIn("5", cmd)

    @patch("subprocess.run")
    def test_preprocess_geotiff(self, mock_run):
        """preprocess() should build correct command for GeoTIFF."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = GeotiffPreprocessConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data_file=Path("/path/to/cover.tif"),
            band=0,
            units="percent",
            output=Path("/path/to/cover.jshd"),
            timestep=0,
            crs="EPSG:4326",
        )

        result = cli.preprocess(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("preprocess", cmd)
        self.assertIn("0", cmd)  # band index
        self.assertIn("percent", cmd)
        self.assertIn("--timestep", cmd)  # Required for GeoTIFF
        self.assertIn("--crs", cmd)
        self.assertIn("EPSG:4326", cmd)
        # GeoTIFF should NOT have coord dimension flags
        self.assertNotIn("--x-coord", cmd)
        self.assertNotIn("--y-coord", cmd)
        self.assertNotIn("--time-dim", cmd)

    @patch("subprocess.run")
    def test_preprocess_csv(self, mock_run):
        """preprocess() should build correct command for CSV."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = CsvPreprocessConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data_file=Path("/path/to/stations.csv"),
            variable="precipitation",
            units="mm",
            output=Path("/path/to/precip.jshd"),
            timestep=0,
        )

        result = cli.preprocess(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("preprocess", cmd)
        self.assertIn("precipitation", cmd)  # variable name
        self.assertIn("mm", cmd)
        self.assertIn("--timestep", cmd)  # Required for CSV
        self.assertIn("0", cmd)

    @patch("subprocess.run")
    def test_validate(self, mock_run):
        """validate() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = DiscoverConfigConfig(script=Path("/path/to/simulation.josh"))

        result = cli.discover_config(config)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertIn("discoverConfig", cmd)

    @patch("subprocess.run")
    def test_inspect_jshd(self, mock_run):
        """inspect_jshd() should build correct command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="value: 25.5", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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

        cli = JoshCLI(josh_jar=self.JAR_MODE)
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

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        result = cli.run(config)

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("Java not found", result.stderr)


class TestJfrConfig(unittest.TestCase):
    """Tests for JfrConfig dataclass."""

    def test_defaults(self):
        """JfrConfig should have sensible defaults."""
        config = JfrConfig(output=Path("/tmp/recording.jfr"))
        self.assertEqual(config.output, Path("/tmp/recording.jfr"))
        self.assertEqual(config.settings, "profile")
        self.assertIsNone(config.maxsize)

    def test_custom_settings(self):
        """JfrConfig should accept custom settings and maxsize."""
        config = JfrConfig(
            output=Path("/tmp/recording.jfr"),
            settings="default",
            maxsize="500m",
        )
        self.assertEqual(config.settings, "default")
        self.assertEqual(config.maxsize, "500m")

    def test_frozen(self):
        """JfrConfig should be immutable."""
        config = JfrConfig(output=Path("/tmp/recording.jfr"))
        with self.assertRaises(AttributeError):
            config.settings = "default"  # type: ignore[misc]


class TestJfr(unittest.TestCase):
    """Tests for JFR integration in JoshCLI."""

    JAR_MODE = JarMode.DEV

    @patch("subprocess.run")
    def test_run_with_jfr(self, mock_run):
        """run() with JFR should insert JVM flags before -jar."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )
        jfr = JfrConfig(output=Path("/tmp/recording.jfr"))

        result = cli.run(config, jfr=jfr)

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]

        # Find the JFR flag
        jfr_flags = [c for c in cmd if c.startswith("-XX:StartFlightRecording")]
        self.assertEqual(len(jfr_flags), 1)
        jfr_flag = jfr_flags[0]

        # Verify contents
        self.assertIn("filename=", jfr_flag)
        self.assertIn("recording.jfr", jfr_flag)
        self.assertIn("settings=profile", jfr_flag)
        self.assertIn("dumponexit=true", jfr_flag)

        # Verify ordering: JFR flag must come before -jar
        jfr_idx = cmd.index(jfr_flag)
        jar_idx = cmd.index("-jar")
        self.assertLess(jfr_idx, jar_idx)

    @patch("subprocess.run")
    def test_run_without_jfr(self, mock_run):
        """run() without JFR should not have any -XX: flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        cli.run(config)

        cmd = mock_run.call_args[0][0]
        xx_flags = [c for c in cmd if c.startswith("-XX:")]
        self.assertEqual(len(xx_flags), 0)

    @patch("subprocess.run")
    def test_jfr_with_maxsize(self, mock_run):
        """JFR with maxsize should include it in the flag."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )
        jfr = JfrConfig(output=Path("/tmp/recording.jfr"), maxsize="500m")

        cli.run(config, jfr=jfr)

        cmd = mock_run.call_args[0][0]
        jfr_flag = [c for c in cmd if c.startswith("-XX:StartFlightRecording")][0]
        self.assertIn("maxsize=500m", jfr_flag)

    @patch("subprocess.run")
    def test_jfr_with_default_settings(self, mock_run):
        """JFR with settings='default' should use that in the flag."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )
        jfr = JfrConfig(output=Path("/tmp/recording.jfr"), settings="default")

        cli.run(config, jfr=jfr)

        cmd = mock_run.call_args[0][0]
        jfr_flag = [c for c in cmd if c.startswith("-XX:StartFlightRecording")][0]
        self.assertIn("settings=default", jfr_flag)

    @patch("subprocess.run")
    def test_preprocess_with_jfr(self, mock_run):
        """preprocess() with JFR should insert JVM flags before -jar."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = NetcdfPreprocessConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            data_file=Path("/path/to/data.nc"),
            variable="temp",
            units="K",
            output=Path("/path/to/output.jshd"),
        )
        jfr = JfrConfig(output=Path("/tmp/preprocess.jfr"))

        cli.preprocess(config, jfr=jfr)

        cmd = mock_run.call_args[0][0]
        jfr_flags = [c for c in cmd if c.startswith("-XX:StartFlightRecording")]
        self.assertEqual(len(jfr_flags), 1)

        # Verify ordering
        jfr_idx = cmd.index(jfr_flags[0])
        jar_idx = cmd.index("-jar")
        self.assertLess(jfr_idx, jar_idx)

    @patch("subprocess.run")
    def test_validate_with_jfr(self, mock_run):
        """validate() with JFR should insert JVM flags before -jar."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = ValidateConfig(script=Path("/path/to/simulation.josh"))
        jfr = JfrConfig(output=Path("/tmp/validate.jfr"))

        cli.validate(config, jfr=jfr)

        cmd = mock_run.call_args[0][0]
        jfr_flags = [c for c in cmd if c.startswith("-XX:StartFlightRecording")]
        self.assertEqual(len(jfr_flags), 1)

    @patch("subprocess.run")
    def test_summarize_jfr(self, mock_run):
        """summarize_jfr() should invoke 'jfr summary'."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Flight Recording summary",
            stderr="",
        )

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        result = cli.summarize_jfr(Path("/tmp/recording.jfr"))

        self.assertTrue(result.success)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[1], "summary")
        self.assertIn("recording.jfr", cmd[2])

    @patch("subprocess.run")
    def test_summarize_jfr_derives_bin_from_java_path(self, mock_run):
        """summarize_jfr() should derive jfr binary from java_path when in bin/."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE, java_path="/usr/lib/jvm/java-21/bin/java")
        cli.summarize_jfr(Path("/tmp/recording.jfr"))

        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "/usr/lib/jvm/java-21/bin/jfr")


class TestPerJobJfr(unittest.TestCase):
    """Tests for _per_job_jfr helper."""

    def test_inserts_run_hash(self):
        """_per_job_jfr should insert run_hash before extension."""
        from joshpy.jobs import _per_job_jfr

        jfr = JfrConfig(output=Path("/tmp/profile.jfr"))
        result = _per_job_jfr(jfr, "abc123def456")
        self.assertEqual(result.output, Path("/tmp/profile_abc123def456.jfr"))

    def test_preserves_settings(self):
        """_per_job_jfr should preserve settings and maxsize."""
        from joshpy.jobs import _per_job_jfr

        jfr = JfrConfig(output=Path("/tmp/profile.jfr"), settings="default", maxsize="500m")
        result = _per_job_jfr(jfr, "abc123")
        self.assertEqual(result.settings, "default")
        self.assertEqual(result.maxsize, "500m")

    def test_no_extension(self):
        """_per_job_jfr should default to .jfr when no extension."""
        from joshpy.jobs import _per_job_jfr

        jfr = JfrConfig(output=Path("/tmp/profile"))
        result = _per_job_jfr(jfr, "abc123")
        self.assertEqual(result.output, Path("/tmp/profile_abc123.jfr"))


class TestInspectExportsConfig(unittest.TestCase):
    """Tests for InspectExportsConfig dataclass."""

    def test_basic_creation(self):
        """Basic config with required fields."""
        config = InspectExportsConfig(
            script=Path("simulation.josh"),
            simulation="Main",
        )
        self.assertEqual(config.script, Path("simulation.josh"))
        self.assertEqual(config.simulation, "Main")
        self.assertTrue(config.json_output)

    def test_frozen(self):
        """Config should be immutable."""
        config = InspectExportsConfig(
            script=Path("simulation.josh"),
            simulation="Main",
        )
        with self.assertRaises(AttributeError):
            config.simulation = "Other"  # type: ignore


class TestExportFileInfo(unittest.TestCase):
    """Tests for ExportFileInfo dataclass."""

    def test_basic_creation(self):
        """Basic creation with all fields."""
        info = ExportFileInfo(
            raw='"file:///tmp/output.csv"',
            protocol="file",
            host="",
            path="/tmp/output.csv",
            file_type="csv",
        )
        self.assertEqual(info.protocol, "file")
        self.assertEqual(info.path, "/tmp/output.csv")
        self.assertEqual(info.file_type, "csv")


class TestExportPaths(unittest.TestCase):
    """Tests for ExportPaths dataclass."""

    def test_get_patch_path_when_present(self):
        """get_patch_path returns path when patch export is configured."""
        paths = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw='"file:///tmp/output.csv"',
                    protocol="file",
                    host="",
                    path="/tmp/output.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={
                "organism": None,
                "patch": None,
                "agent": None,
                "disturbance": None,
            },
        )
        self.assertEqual(paths.get_patch_path(), "/tmp/output.csv")

    def test_get_patch_path_when_missing(self):
        """get_patch_path returns None when patch export is not configured."""
        paths = ExportPaths(
            simulation="Main",
            export_files={"patch": None, "meta": None, "entity": None},
            debug_files={
                "organism": None,
                "patch": None,
                "agent": None,
                "disturbance": None,
            },
        )
        self.assertIsNone(paths.get_patch_path())

    def test_resolve_path_with_template_variables(self):
        """resolve_path correctly substitutes template variables."""
        paths = ExportPaths(
            simulation="Main",
            export_files={"patch": None, "meta": None, "entity": None},
            debug_files={
                "organism": None,
                "patch": None,
                "agent": None,
                "disturbance": None,
            },
        )
        resolved = paths.resolve_path(
            "/tmp/output_{maxGrowth}_{replicate}.csv",
            maxGrowth=50,
            replicate=0,
        )
        self.assertEqual(resolved, Path("/tmp/output_50_0.csv"))

    def test_resolve_path_missing_variable_raises(self):
        """resolve_path raises KeyError for missing template variables."""
        paths = ExportPaths(
            simulation="Main",
            export_files={"patch": None, "meta": None, "entity": None},
            debug_files={
                "organism": None,
                "patch": None,
                "agent": None,
                "disturbance": None,
            },
        )
        with self.assertRaises(KeyError):
            paths.resolve_path("/tmp/output_{maxGrowth}_{replicate}.csv", maxGrowth=50)


class TestInspectExports(unittest.TestCase):
    """Tests for JoshCLI.inspect_exports method."""

    # Use the DEV jar mode for tests
    JAR_MODE = JarMode.DEV
    EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

    @patch("subprocess.run")
    def test_inspect_exports_parses_json(self, mock_run):
        """inspect_exports correctly parses JSON output."""
        json_output = """{
  "simulation": "Main",
  "exportFiles": {
    "patch": {
      "raw": "\\"file:///tmp/hello_josh_{maxGrowth}_{replicate}.csv\\"",
      "protocol": "file",
      "host": "",
      "path": "/tmp/hello_josh_{maxGrowth}_{replicate}.csv",
      "fileType": "csv"
    },
    "meta": null,
    "entity": null
  },
  "debugFiles": {
    "organism": null,
    "patch": null,
    "agent": null,
    "disturbance": null
  }
}"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json_output,
            stderr="",
        )

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = InspectExportsConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        result = cli.inspect_exports(config)

        self.assertEqual(result.simulation, "Main")
        self.assertIsNotNone(result.export_files["patch"])
        self.assertEqual(
            result.export_files["patch"].path, "/tmp/hello_josh_{maxGrowth}_{replicate}.csv"
        )
        self.assertEqual(result.export_files["patch"].protocol, "file")
        self.assertEqual(result.export_files["patch"].file_type, "csv")
        self.assertIsNone(result.export_files["meta"])
        self.assertIsNone(result.export_files["entity"])
        self.assertIsNone(result.debug_files["organism"])

    @patch("subprocess.run")
    def test_inspect_exports_omits_json_flag_when_true(self, mock_run):
        """inspect_exports omits --json flag when json_output is True (default).

        Note: --json is a toggle flag. Default is true, so we don't pass it.
        Passing --json would toggle it OFF.
        """
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"simulation": "Main", "exportFiles": {}, "debugFiles": {}}',
            stderr="",
        )

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = InspectExportsConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            json_output=True,
        )

        cli.inspect_exports(config)

        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--json", cmd)  # Not included - default is true
        self.assertIn("inspect-exports", cmd)
        self.assertIn("Main", cmd)

    @patch("subprocess.run")
    def test_inspect_exports_includes_json_flag_when_false(self, mock_run):
        """inspect_exports includes --json flag when json_output is False.

        Note: --json is a toggle flag. Default is true, so passing --json toggles it OFF.
        """
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Simulation: Main\n...",  # Human-readable output
            stderr="",
        )

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = InspectExportsConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
            json_output=False,
        )

        # This will fail to parse JSON, but we're just testing the flag is present
        try:
            cli.inspect_exports(config)
        except Exception:
            pass  # Expected - can't parse human-readable as JSON

        cmd = mock_run.call_args[0][0]
        self.assertIn("--json", cmd)  # Included to toggle OFF JSON

    @patch("subprocess.run")
    def test_inspect_exports_raises_on_failure(self, mock_run):
        """inspect_exports raises RuntimeError on non-zero exit code."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: Simulation not found",
        )

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = InspectExportsConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="NonExistent",
        )

        with self.assertRaises(RuntimeError) as context:
            cli.inspect_exports(config)

        self.assertIn("inspect-exports failed", str(context.exception))
        self.assertIn("exit code 1", str(context.exception))

    @patch("subprocess.run")
    def test_inspect_exports_handles_all_exports_none(self, mock_run):
        """inspect_exports handles case where all exports are null."""
        json_output = """{
  "simulation": "Main",
  "exportFiles": {
    "patch": null,
    "meta": null,
    "entity": null
  },
  "debugFiles": {
    "organism": null,
    "patch": null,
    "agent": null,
    "disturbance": null
  }
}"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json_output,
            stderr="",
        )

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = InspectExportsConfig(
            script=Path("/path/to/simulation.josh"),
            simulation="Main",
        )

        result = cli.inspect_exports(config)

        self.assertEqual(result.simulation, "Main")
        self.assertIsNone(result.export_files["patch"])
        self.assertIsNone(result.export_files["meta"])
        self.assertIsNone(result.export_files["entity"])
        self.assertIsNone(result.get_patch_path())


class TestInspectExportsIntegration(unittest.TestCase):
    """Integration tests for inspect_exports using local jar."""

    # Use the DEV jar mode for tests
    JAR_MODE = JarMode.DEV
    EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
    JAVA_PATH = Path(__file__).parent.parent / ".pixi" / "envs" / "default" / "bin" / "java"

    def test_inspect_exports_real_file(self):
        """Integration test: inspect_exports parses real Josh file."""
        script_path = self.EXAMPLES_DIR / "hello_cli_configurable.josh"
        if not script_path.exists():
            self.skipTest(f"Example file not found: {script_path}")
        if not self.JAVA_PATH.exists():
            self.skipTest(f"Java not found at: {self.JAVA_PATH}")

        cli = JoshCLI(josh_jar=self.JAR_MODE, java_path=str(self.JAVA_PATH))
        config = InspectExportsConfig(
            script=script_path,
            simulation="Main",
        )

        result = cli.inspect_exports(config)

        self.assertEqual(result.simulation, "Main")
        self.assertIsNotNone(result.export_files["patch"])
        self.assertEqual(result.export_files["patch"].protocol, "file")
        self.assertEqual(result.export_files["patch"].file_type, "csv")
        self.assertIn("{maxGrowth}", result.export_files["patch"].path)
        self.assertIn("{replicate}", result.export_files["patch"].path)

    def test_inspect_exports_nonexistent_simulation_raises(self):
        """Integration test: inspect_exports raises for nonexistent simulation."""
        script_path = self.EXAMPLES_DIR / "hello_cli_configurable.josh"
        if not script_path.exists():
            self.skipTest(f"Example file not found: {script_path}")
        if not self.JAVA_PATH.exists():
            self.skipTest(f"Java not found at: {self.JAVA_PATH}")

        cli = JoshCLI(josh_jar=self.JAR_MODE, java_path=str(self.JAVA_PATH))
        config = InspectExportsConfig(
            script=script_path,
            simulation="NonExistentSimulation",
        )

        with self.assertRaises(RuntimeError):
            cli.inspect_exports(config)


class TestStreamOutput(unittest.TestCase):
    """Tests for stream_output parameter in JoshCLI."""

    JAR_MODE = JarMode.DEV

    @patch("subprocess.run")
    def test_default_uses_subprocess_run(self, mock_run):
        """stream_output=False (default) should use subprocess.run."""
        mock_run.return_value = MagicMock(returncode=0, stdout="out", stderr="")
        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(script=Path("/path/sim.josh"), simulation="Main")

        result = cli.run(config)

        mock_run.assert_called_once()
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "out")

    @patch("subprocess.Popen")
    def test_stream_output_uses_popen(self, mock_popen):
        """stream_output=True should use Popen, not subprocess.run."""
        # Set up a mock process whose stdout/stderr are iterable
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["Step 1 done\n", "Step 2 done\n"])
        mock_proc.stderr = iter([""])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(script=Path("/path/sim.josh"), simulation="Main")

        result = cli.run(config, stream_output=True)

        mock_popen.assert_called_once()
        self.assertTrue(result.success)
        self.assertIn("Step 1 done", result.stdout)
        self.assertIn("Step 2 done", result.stdout)

    @patch("subprocess.Popen")
    def test_stream_output_captures_stderr(self, mock_popen):
        """stream_output=True should capture stderr in CLIResult."""
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["ok\n"])
        mock_proc.stderr = iter(["warning: something\n"])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(script=Path("/path/sim.josh"), simulation="Main")

        result = cli.run(config, stream_output=True)

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("warning: something", result.stderr)

    @patch("subprocess.Popen")
    def test_stream_output_writes_to_sys_stdout(self, mock_popen):
        """stream_output=True should write lines to sys.stdout."""
        import io

        mock_proc = MagicMock()
        mock_proc.stdout = iter(["hello\n"])
        mock_proc.stderr = iter([])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(script=Path("/path/sim.josh"), simulation="Main")

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli.run(config, stream_output=True)

        self.assertIn("hello", captured.getvalue())

    @patch("subprocess.Popen")
    def test_stream_output_timeout(self, mock_popen):
        """stream_output=True with timeout should report timeout."""
        import subprocess as sp

        mock_proc = MagicMock()
        mock_proc.stdout = iter(["partial\n"])
        mock_proc.stderr = iter([])
        mock_proc.wait.side_effect = sp.TimeoutExpired(cmd=["java"], timeout=5)
        mock_proc.kill.return_value = None
        mock_popen.return_value = mock_proc

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunConfig(script=Path("/path/sim.josh"), simulation="Main")

        result = cli.run(config, stream_output=True, timeout=5)

        self.assertEqual(result.exit_code, -1)
        self.assertIn("timed out", result.stderr)

    @patch("subprocess.Popen")
    def test_stream_output_run_remote(self, mock_popen):
        """stream_output should also work with run_remote()."""
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["remote step\n"])
        mock_proc.stderr = iter([])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = RunRemoteConfig(
            script=Path("/path/sim.josh"),
            simulation="Main",
            api_key="key",
        )

        result = cli.run_remote(config, stream_output=True)

        mock_popen.assert_called_once()
        self.assertTrue(result.success)
        self.assertIn("remote step", result.stdout)


class TestStageFromMinioConfig(unittest.TestCase):
    """Tests for StageFromMinioConfig."""

    def test_defaults(self):
        from joshpy.cli import StageFromMinioConfig

        config = StageFromMinioConfig(
            output_dir=Path("/tmp/out"),
            prefix="batch-jobs/abc/inputs/",
        )
        self.assertEqual(config.output_dir, Path("/tmp/out"))
        self.assertEqual(config.prefix, "batch-jobs/abc/inputs/")
        self.assertIsNone(config.minio_endpoint)
        self.assertIsNone(config.minio_access_key)
        self.assertIsNone(config.minio_secret_key)
        self.assertIsNone(config.minio_bucket)

    def test_frozen(self):
        from joshpy.cli import StageFromMinioConfig

        config = StageFromMinioConfig(output_dir=Path("/tmp"), prefix="p/")
        with self.assertRaises(AttributeError):
            config.prefix = "other/"


class TestStageFromMinio(unittest.TestCase):
    """Tests for JoshCLI.stage_from_minio()."""

    JAR_MODE = JarMode.LOCAL

    @patch("joshpy.jar.JarManager.get_jar", return_value=Path("/fake/joshsim-fat.jar"))
    @patch("subprocess.run")
    def test_basic_args(self, mock_run, _mock_jar):
        """stage_from_minio() should build correct CLI args."""
        from joshpy.cli import StageFromMinioConfig

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        config = StageFromMinioConfig(
            output_dir=Path("/tmp/out"),
            prefix="batch-jobs/abc/inputs/",
        )
        cli.stage_from_minio(config)

        cmd = mock_run.call_args[0][0]
        self.assertIn("stageFromMinio", cmd)
        self.assertIn("--output-dir", cmd)
        self.assertIn("--prefix", cmd)
        prefix_idx = cmd.index("--prefix")
        self.assertEqual(cmd[prefix_idx + 1], "batch-jobs/abc/inputs/")

    @patch("joshpy.jar.JarManager.get_jar", return_value=Path("/fake/joshsim-fat.jar"))
    @patch("subprocess.run")
    def test_minio_flags_only_when_set(self, mock_run, _mock_jar):
        """Only non-None minio flags should be passed."""
        from joshpy.cli import StageFromMinioConfig

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = JoshCLI(josh_jar=self.JAR_MODE)

        # With no minio flags
        config_no_minio = StageFromMinioConfig(
            output_dir=Path("/tmp/out"), prefix="p/"
        )
        cli.stage_from_minio(config_no_minio)
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--minio-endpoint", cmd)
        self.assertNotIn("--minio-access-key", cmd)
        self.assertNotIn("--minio-secret-key", cmd)
        self.assertNotIn("--minio-bucket", cmd)

        # With all minio flags
        config_with_minio = StageFromMinioConfig(
            output_dir=Path("/tmp/out"),
            prefix="p/",
            minio_endpoint="https://storage.example.com",
            minio_access_key="AKID",
            minio_secret_key="SECRET",
            minio_bucket="my-bucket",
        )
        cli.stage_from_minio(config_with_minio)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--minio-endpoint", cmd)
        self.assertIn("--minio-access-key", cmd)
        self.assertIn("--minio-secret-key", cmd)
        self.assertIn("--minio-bucket", cmd)
        ep_idx = cmd.index("--minio-endpoint")
        self.assertEqual(cmd[ep_idx + 1], "https://storage.example.com")
        bucket_idx = cmd.index("--minio-bucket")
        self.assertEqual(cmd[bucket_idx + 1], "my-bucket")


if __name__ == "__main__":
    unittest.main()
