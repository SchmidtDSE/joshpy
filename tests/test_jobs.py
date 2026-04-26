"""Unit tests for the jobs module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

try:
    import duckdb  # noqa: F401

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from joshpy.jobs import (
    ConfigSweepParameter,
    FileSweepParameter,
    CompoundSweepParameter,
    SweepConfig,
    JobConfig,
    ExpandedJob,
    JobSet,
    JobExpander,
    SweepResult,
    _compute_run_hash,
    _hash_file,
    _normalize_values,
    to_run_config,
    to_run_remote_config,
    run_sweep,
    discover_jshd_files,
)
from joshpy.strategies import SweepExecutionError


class TestRunHash(unittest.TestCase):
    """Tests for _compute_run_hash and _hash_file functions."""

    def test_returns_12_char_string(self):
        """Hash should be 12 characters."""
        result = _compute_run_hash(None, "test content")
        self.assertEqual(len(result), 12)
        self.assertTrue(all(c in '0123456789abcdef' for c in result))

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        content = "survivalProbAdult = 85 %"
        hash1 = _compute_run_hash(None, content)
        hash2 = _compute_run_hash(None, content)
        self.assertEqual(hash1, hash2)

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        hash1 = _compute_run_hash(None, "survivalProbAdult = 85 %")
        hash2 = _compute_run_hash(None, "survivalProbAdult = 90 %")
        self.assertNotEqual(hash1, hash2)

    def test_includes_josh_content(self):
        """Hash should change when josh content changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            josh1 = Path(tmpdir) / "sim1.josh"
            josh2 = Path(tmpdir) / "sim2.josh"
            josh1.write_text("simulation Main { }")
            josh2.write_text("simulation Main { // different }")

            hash1 = _compute_run_hash(josh1, "config content")
            hash2 = _compute_run_hash(josh2, "config content")
            self.assertNotEqual(hash1, hash2)

    def test_includes_file_mappings(self):
        """Hash should change when file_mappings content changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = Path(tmpdir) / "data1.jshd"
            data2 = Path(tmpdir) / "data2.jshd"
            data1.write_bytes(b"data version 1")
            data2.write_bytes(b"data version 2")

            hash1 = _compute_run_hash(None, "config", {"data": data1})
            hash2 = _compute_run_hash(None, "config", {"data": data2})
            self.assertNotEqual(hash1, hash2)

    def test_raises_if_josh_missing(self):
        """Should raise FileNotFoundError if josh file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            _compute_run_hash(Path("/nonexistent/file.josh"), "config")

    def test_raises_if_file_mapping_missing(self):
        """Should raise FileNotFoundError if file in file_mappings doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            _compute_run_hash(None, "config", {"data": Path("/nonexistent/data.jshd")})

    def test_seed_none_matches_legacy_hash(self):
        """seed=None preserves the unseeded hash (registry backward-compat)."""
        legacy = _compute_run_hash(None, "config")
        with_explicit_none = _compute_run_hash(None, "config", seed=None)
        self.assertEqual(legacy, with_explicit_none)

    def test_seed_changes_hash(self):
        """Same content with different seeds produces different hashes."""
        hash_seed42 = _compute_run_hash(None, "config", seed=42)
        hash_seed99 = _compute_run_hash(None, "config", seed=99)
        self.assertNotEqual(hash_seed42, hash_seed99)

    def test_same_seed_same_hash(self):
        """Same content with the same seed reproduces the same hash."""
        hash1 = _compute_run_hash(None, "config", seed=42)
        hash2 = _compute_run_hash(None, "config", seed=42)
        self.assertEqual(hash1, hash2)

    def test_seed_set_differs_from_unseeded(self):
        """seed=42 and seed=None disambiguate (intentional per-trajectory hash)."""
        unseeded = _compute_run_hash(None, "config")
        seeded = _compute_run_hash(None, "config", seed=42)
        self.assertNotEqual(unseeded, seeded)

    def test_hash_file(self):
        """_hash_file should return consistent hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            path.write_bytes(b"test content")
            
            hash1 = _hash_file(path)
            hash2 = _hash_file(path)
            self.assertEqual(hash1, hash2)
            self.assertEqual(len(hash1), 32)  # Full MD5 hex digest


class TestNormalizeValues(unittest.TestCase):
    """Tests for _normalize_values function."""

    def test_list_passthrough(self):
        """Lists should pass through unchanged."""
        values = [1, 2, 3]
        result = _normalize_values(values)
        self.assertEqual(result, [1, 2, 3])

    def test_tuple_to_list(self):
        """Tuples should convert to lists."""
        values = (1, 2, 3)
        result = _normalize_values(values)
        self.assertEqual(result, [1, 2, 3])

    def test_single_value_wrapped(self):
        """Single values should be wrapped in list."""
        result = _normalize_values(42)
        self.assertEqual(result, [42])

    def test_range_with_step(self):
        """Range spec with step should expand like arange."""
        values = {"start": 0, "stop": 10, "step": 2}
        result = _normalize_values(values)
        self.assertEqual(result, [0, 2, 4, 6, 8])

    def test_range_with_num(self):
        """Range spec with num should expand like linspace."""
        values = {"start": 0, "stop": 10, "num": 3}
        result = _normalize_values(values)
        self.assertEqual(result, [0.0, 5.0, 10.0])

    def test_range_missing_stop_raises(self):
        """Range spec without stop should raise ValueError."""
        values = {"start": 0, "step": 2}
        with self.assertRaises(KeyError):
            _normalize_values(values)

    def test_range_missing_step_and_num_raises(self):
        """Range spec without step or num should raise ValueError."""
        values = {"start": 0, "stop": 10}
        with self.assertRaises(ValueError):
            _normalize_values(values)


class TestConfigSweepParameter(unittest.TestCase):
    """Tests for ConfigSweepParameter class."""

    def test_basic_creation(self):
        """Basic parameter creation with list."""
        param = ConfigSweepParameter(name="survival", values=[85, 90, 95])
        self.assertEqual(param.name, "survival")
        self.assertEqual(param.values, [85, 90, 95])

    def test_range_expansion(self):
        """Parameters with range spec should expand."""
        param = ConfigSweepParameter(
            name="survival",
            values={"start": 80, "stop": 90, "step": 2}
        )
        self.assertEqual(param.values, [80, 82, 84, 86, 88])

    def test_to_dict_with_values(self):
        """to_dict should include values."""
        param = ConfigSweepParameter(name="x", values=[1, 2, 3])
        result = param.to_dict()
        self.assertEqual(result, {"name": "x", "values": [1, 2, 3]})

    def test_to_dict_with_range(self):
        """to_dict should preserve range spec."""
        param = ConfigSweepParameter(
            name="x",
            values={"start": 0, "stop": 10, "step": 2}
        )
        result = param.to_dict()
        self.assertEqual(result["name"], "x")
        self.assertIn("range", result)
        self.assertEqual(result["range"]["start"], 0)

    def test_from_dict_with_values(self):
        """from_dict should create parameter with values."""
        data = {"name": "x", "values": [1, 2, 3]}
        param = ConfigSweepParameter.from_dict(data)
        self.assertEqual(param.name, "x")
        self.assertEqual(param.values, [1, 2, 3])

    def test_from_dict_with_range(self):
        """from_dict should create parameter with range."""
        data = {"name": "x", "range": {"start": 0, "stop": 6, "step": 2}}
        param = ConfigSweepParameter.from_dict(data)
        self.assertEqual(param.name, "x")
        self.assertEqual(param.values, [0, 2, 4])


class TestFileSweepParameter(unittest.TestCase):
    """Tests for FileSweepParameter class."""

    def test_basic_creation(self):
        """Basic file parameter creation with paths."""
        param = FileSweepParameter(name="climate", paths=[
            Path("data/rcp26.jshd"),
            Path("data/rcp45.jshd"),
        ])
        self.assertEqual(param.name, "climate")
        self.assertEqual(len(param.paths), 2)
        self.assertEqual(param.paths[0], Path("data/rcp26.jshd"))

    def test_string_to_path_conversion(self):
        """String paths should be converted to Path objects."""
        param = FileSweepParameter(name="data", paths=[
            "data/file1.jshd",
            "data/file2.jshd",
        ])
        self.assertIsInstance(param.paths[0], Path)
        self.assertEqual(param.paths[0], Path("data/file1.jshd"))

    def test_labels_from_stems(self):
        """Labels should be derived from filename stems."""
        param = FileSweepParameter(name="climate", paths=[
            Path("data/rcp26.jshd"),
            Path("data/rcp45.jshd"),
            Path("other/rcp85.jshd"),
        ])
        self.assertEqual(param.labels, ["rcp26", "rcp45", "rcp85"])

    def test_duplicate_stems_raises(self):
        """Duplicate filename stems should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            FileSweepParameter(name="climate", paths=[
                Path("dir1/data.jshd"),
                Path("dir2/data.jshd"),
            ])
        self.assertIn("duplicate stems", str(ctx.exception).lower())

    def test_to_dict(self):
        """to_dict should serialize paths as strings."""
        param = FileSweepParameter(name="climate", paths=[
            Path("a.jshd"),
            Path("b.jshd"),
        ])
        result = param.to_dict()
        self.assertEqual(result["name"], "climate")
        self.assertEqual(result["paths"], ["a.jshd", "b.jshd"])

    def test_from_dict(self):
        """from_dict should deserialize paths."""
        data = {"name": "climate", "paths": ["a.jshd", "b.jshd"]}
        param = FileSweepParameter.from_dict(data)
        self.assertEqual(param.name, "climate")
        self.assertEqual(param.paths, [Path("a.jshd"), Path("b.jshd")])


class TestSweepConfig(unittest.TestCase):
    """Tests for SweepConfig class."""

    def test_empty_expand(self):
        """Empty config should expand to single empty dict."""
        config = SweepConfig()
        result = config.expand()
        self.assertEqual(result, [{}])

    def test_single_param_expand(self):
        """Single parameter should expand to list of single-key dicts."""
        config = SweepConfig(config_parameters=[
            ConfigSweepParameter(name="x", values=[1, 2, 3])
        ])
        result = config.expand()
        self.assertEqual(result, [{"x": 1}, {"x": 2}, {"x": 3}])

    def test_multiple_param_expand(self):
        """Multiple parameters should produce cartesian product."""
        config = SweepConfig(config_parameters=[
            ConfigSweepParameter(name="a", values=[1, 2]),
            ConfigSweepParameter(name="b", values=["x", "y"]),
        ])
        result = config.expand()
        expected = [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "x"},
            {"a": 2, "b": "y"},
        ]
        self.assertEqual(result, expected)

    def test_len(self):
        """Length should be product of parameter value counts."""
        config = SweepConfig(config_parameters=[
            ConfigSweepParameter(name="a", values=[1, 2, 3]),  # 3
            ConfigSweepParameter(name="b", values=["x", "y"]),  # 2
        ])
        self.assertEqual(len(config), 6)

    def test_empty_len(self):
        """Empty config should have length 1."""
        config = SweepConfig()
        self.assertEqual(len(config), 1)

    def test_to_dict(self):
        """to_dict should serialize config_parameters."""
        config = SweepConfig(config_parameters=[
            ConfigSweepParameter(name="x", values=[1, 2])
        ])
        result = config.to_dict()
        self.assertEqual(result["config_parameters"][0]["name"], "x")

    def test_from_dict(self):
        """from_dict should deserialize parameters."""
        data = {"parameters": [{"name": "x", "values": [1, 2]}]}
        config = SweepConfig.from_dict(data)
        self.assertEqual(len(config.parameters), 1)
        self.assertEqual(config.parameters[0].name, "x")

    def test_from_dict_new_style(self):
        """from_dict should deserialize config_parameters."""
        data = {"config_parameters": [{"name": "x", "values": [1, 2]}]}
        config = SweepConfig.from_dict(data)
        self.assertEqual(len(config.config_parameters), 1)
        self.assertEqual(config.config_parameters[0].name, "x")

    def test_parameters_property_returns_config_parameters(self):
        """parameters property should return config_parameters for backward compat."""
        config = SweepConfig(config_parameters=[
            ConfigSweepParameter(name="x", values=[1, 2])
        ])
        self.assertEqual(config.parameters, config.config_parameters)

    def test_file_param_expand(self):
        """File parameters should expand with path and label."""
        config = SweepConfig(file_parameters=[
            FileSweepParameter(name="climate", paths=[
                Path("data/rcp45.jshd"),
                Path("data/rcp85.jshd"),
            ])
        ])
        result = config.expand()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["climate"]["path"], Path("data/rcp45.jshd"))
        self.assertEqual(result[0]["climate"]["label"], "rcp45")
        self.assertEqual(result[1]["climate"]["path"], Path("data/rcp85.jshd"))
        self.assertEqual(result[1]["climate"]["label"], "rcp85")

    def test_mixed_param_expand(self):
        """Config and file parameters should expand as cartesian product."""
        config = SweepConfig(
            config_parameters=[
                ConfigSweepParameter(name="x", values=[1, 2]),
            ],
            file_parameters=[
                FileSweepParameter(name="data", paths=[
                    Path("a.jshd"),
                    Path("b.jshd"),
                ])
            ]
        )
        result = config.expand()
        self.assertEqual(len(result), 4)  # 2 x 2

        # Check first combo: x=1, data=a.jshd
        self.assertEqual(result[0]["x"], 1)
        self.assertEqual(result[0]["data"]["path"], Path("a.jshd"))
        self.assertEqual(result[0]["data"]["label"], "a")

        # Check last combo: x=2, data=b.jshd
        self.assertEqual(result[3]["x"], 2)
        self.assertEqual(result[3]["data"]["path"], Path("b.jshd"))
        self.assertEqual(result[3]["data"]["label"], "b")

    def test_len_with_file_params(self):
        """Length should include file parameter counts."""
        config = SweepConfig(
            config_parameters=[
                ConfigSweepParameter(name="x", values=[1, 2, 3]),  # 3
            ],
            file_parameters=[
                FileSweepParameter(name="data", paths=[
                    Path("a.jshd"),
                    Path("b.jshd"),
                ]),  # 2
            ]
        )
        self.assertEqual(len(config), 6)  # 3 x 2

    def test_to_dict_with_file_params(self):
        """to_dict should serialize file_parameters."""
        config = SweepConfig(file_parameters=[
            FileSweepParameter(name="data", paths=[Path("a.jshd")])
        ])
        result = config.to_dict()
        self.assertEqual(result["file_parameters"][0]["name"], "data")
        self.assertEqual(result["file_parameters"][0]["paths"], ["a.jshd"])

    def test_from_dict_with_file_params(self):
        """from_dict should deserialize file_parameters."""
        data = {
            "config_parameters": [{"name": "x", "values": [1, 2]}],
            "file_parameters": [{"name": "data", "paths": ["a.jshd", "b.jshd"]}]
        }
        config = SweepConfig.from_dict(data)
        self.assertEqual(len(config.config_parameters), 1)
        self.assertEqual(len(config.file_parameters), 1)
        self.assertEqual(config.file_parameters[0].name, "data")
        self.assertEqual(config.file_parameters[0].paths, [Path("a.jshd"), Path("b.jshd")])


class TestJobConfig(unittest.TestCase):
    """Tests for JobConfig class."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = JobConfig()
        self.assertEqual(config.simulation, "Main")
        self.assertEqual(config.replicates, 1)
        self.assertFalse(config.use_float64)

    def test_to_dict_minimal(self):
        """to_dict with defaults should produce minimal dict."""
        config = JobConfig()
        result = config.to_dict()
        # Only non-default values should be in dict
        self.assertNotIn("simulation", result)  # "Main" is default
        self.assertNotIn("replicates", result)  # 1 is default

    def test_to_dict_with_values(self):
        """to_dict should include non-default values."""
        config = JobConfig(
            simulation="ForestSim",
            replicates=5,
            template_string="test = {{ x }}",
        )
        result = config.to_dict()
        self.assertEqual(result["simulation"], "ForestSim")
        self.assertEqual(result["replicates"], 5)
        self.assertEqual(result["template_string"], "test = {{ x }}")

    def test_from_dict(self):
        """from_dict should create config correctly."""
        data = {
            "simulation": "TestSim",
            "replicates": 3,
            "template_string": "var = {{ val }}",
        }
        config = JobConfig.from_dict(data)
        self.assertEqual(config.simulation, "TestSim")
        self.assertEqual(config.replicates, 3)
        self.assertEqual(config.template_string, "var = {{ val }}")

    def test_to_yaml_from_yaml_roundtrip(self):
        """YAML serialization should round-trip correctly."""
        original = JobConfig(
            simulation="TestSim",
            replicates=3,
            template_string="var = {{ val }}",
            sweep=SweepConfig(config_parameters=[
                ConfigSweepParameter(name="val", values=[1, 2, 3])
            ])
        )
        yaml_str = original.to_yaml()
        restored = JobConfig.from_yaml(yaml_str)

        self.assertEqual(restored.simulation, original.simulation)
        self.assertEqual(restored.replicates, original.replicates)
        self.assertEqual(restored.template_string, original.template_string)
        self.assertEqual(len(restored.sweep.config_parameters), 1)
        self.assertEqual(restored.sweep.config_parameters[0].values, [1, 2, 3])


class TestJobExpander(unittest.TestCase):
    """Tests for JobExpander class."""

    def test_expand_no_sweep(self):
        """Expanding without sweep should produce single job."""
        config = JobConfig(
            template_string="value = 42",
            simulation="Main",
        )
        expander = JobExpander()
        job_set = expander.expand(config)

        self.assertEqual(len(job_set), 1)
        self.assertEqual(job_set.jobs[0].config_content, "value = 42")
        self.assertEqual(job_set.jobs[0].parameters, {})

    def test_expand_with_sweep(self):
        """Expanding with sweep should produce multiple jobs."""
        config = JobConfig(
            template_string="value = {{ x }}",
            sweep=SweepConfig(config_parameters=[
                ConfigSweepParameter(name="x", values=[1, 2, 3])
            ])
        )
        expander = JobExpander()
        job_set = expander.expand(config)

        self.assertEqual(len(job_set), 3)
        self.assertEqual(job_set.jobs[0].config_content, "value = 1")
        self.assertEqual(job_set.jobs[1].config_content, "value = 2")
        self.assertEqual(job_set.jobs[2].config_content, "value = 3")

    def test_expand_creates_config_files(self):
        """Expanded jobs should have config files written."""
        config = JobConfig(
            template_string="x = {{ val }}",
            sweep=SweepConfig(config_parameters=[
                ConfigSweepParameter(name="val", values=[10, 20])
            ])
        )
        expander = JobExpander()
        job_set = expander.expand(config)

        try:
            for job in job_set.jobs:
                self.assertTrue(job.config_path.exists())
                content = job.config_path.read_text()
                self.assertEqual(content, job.config_content)
        finally:
            job_set.cleanup()

    def test_expand_custom_tags(self):
        """Expanded jobs should have custom_tags from parameters."""
        config = JobConfig(
            template_string="a={{ a }}, b={{ b }}",
            sweep=SweepConfig(config_parameters=[
                ConfigSweepParameter(name="a", values=[1]),
                ConfigSweepParameter(name="b", values=["test"]),
            ])
        )
        expander = JobExpander()
        job_set = expander.expand(config)

        job = job_set.jobs[0]
        self.assertEqual(job.custom_tags["a"], "1")
        self.assertEqual(job.custom_tags["b"], "test")
        self.assertIn("run_hash", job.custom_tags)

        job_set.cleanup()

    def test_expand_to_specific_dir(self):
        """Expanding to specific directory should use that directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "configs"

            config = JobConfig(
                template_string="value = 1",
            )
            expander = JobExpander()
            job_set = expander.expand(config, output_dir=output_dir)

            self.assertTrue(output_dir.exists())
            self.assertIsNone(job_set.temp_dir)  # No temp dir created

    def test_expand_no_config_source(self):
        """Expanding without any config source should produce empty config."""
        config = JobConfig()
        expander = JobExpander()
        job_set = expander.expand(config)

        try:
            self.assertEqual(len(job_set), 1)
            self.assertEqual(job_set.jobs[0].config_content, "")
        finally:
            job_set.cleanup()

    def test_expand_with_file_params(self):
        """Expanding with file parameters should update file_mappings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file_a = Path(tmpdir) / "scenario_a.jshd"
            file_b = Path(tmpdir) / "scenario_b.jshd"
            file_a.write_bytes(b"data a")
            file_b.write_bytes(b"data b")

            config = JobConfig(
                template_string="test config",
                sweep=SweepConfig(file_parameters=[
                    FileSweepParameter(name="climate", paths=[file_a, file_b])
                ]),
                source_path=Path(tmpdir) / "sim.josh",
            )
            # Create source file for hash computation
            config.source_path.write_text("simulation")

            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 2)

                # First job should have file_a in file_mappings
                job_a = job_set.jobs[0]
                self.assertEqual(job_a.file_mappings["climate"], file_a)
                self.assertEqual(job_a.custom_tags["climate"], "scenario_a")
                self.assertEqual(job_a.custom_tags["climate_file"], "scenario_a.jshd")
                # File label should also be in parameters for SQL queryability
                self.assertEqual(job_a.parameters["climate"], "scenario_a")

                # Second job should have file_b in file_mappings
                job_b = job_set.jobs[1]
                self.assertEqual(job_b.file_mappings["climate"], file_b)
                self.assertEqual(job_b.custom_tags["climate"], "scenario_b")
                self.assertEqual(job_b.custom_tags["climate_file"], "scenario_b.jshd")
                # File label should also be in parameters for SQL queryability
                self.assertEqual(job_b.parameters["climate"], "scenario_b")
            finally:
                job_set.cleanup()

    def test_expand_preserves_default_file_mappings(self):
        """File params should override while preserving other file_mappings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            sweep_file_a = Path(tmpdir) / "sweep_a.jshd"
            sweep_file_b = Path(tmpdir) / "sweep_b.jshd"
            default_file = Path(tmpdir) / "default.jshd"
            sweep_file_a.write_bytes(b"data a")
            sweep_file_b.write_bytes(b"data b")
            default_file.write_bytes(b"default data")

            config = JobConfig(
                template_string="test config",
                file_mappings={"other": default_file},  # This should be preserved
                sweep=SweepConfig(file_parameters=[
                    FileSweepParameter(name="climate", paths=[sweep_file_a, sweep_file_b])
                ]),
                source_path=Path(tmpdir) / "sim.josh",
            )
            config.source_path.write_text("simulation")

            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                for job in job_set:
                    # Default file_mapping should be preserved
                    self.assertEqual(job.file_mappings["other"], default_file)
                    # Swept file should be set
                    self.assertIn(job.file_mappings["climate"], [sweep_file_a, sweep_file_b])
            finally:
                job_set.cleanup()

    def test_expand_mixed_config_and_file_params(self):
        """Mixed config and file params should create cartesian product."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = Path(tmpdir) / "a.jshd"
            file_b = Path(tmpdir) / "b.jshd"
            file_a.write_bytes(b"data a")
            file_b.write_bytes(b"data b")

            config = JobConfig(
                template_string="x = {{ x }}",
                sweep=SweepConfig(
                    config_parameters=[
                        ConfigSweepParameter(name="x", values=[1, 2]),
                    ],
                    file_parameters=[
                        FileSweepParameter(name="data", paths=[file_a, file_b]),
                    ],
                ),
                source_path=Path(tmpdir) / "sim.josh",
            )
            config.source_path.write_text("simulation")

            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 4)  # 2 x 2

                # Jobs should have both config params and file labels in parameters
                for job in job_set:
                    self.assertIn("x", job.parameters)
                    # File parameter labels should now be in parameters
                    self.assertIn("data", job.parameters)
                    # The value should be the filename stem (label)
                    self.assertIn(job.parameters["data"], ["a", "b"])
            finally:
                job_set.cleanup()

    def test_multiple_file_params_in_parameters(self):
        """Multiple file parameter labels should all appear in parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files for two different file parameters
            precip_low = Path(tmpdir) / "precip_low.jshd"
            precip_high = Path(tmpdir) / "precip_high.jshd"
            temp_cool = Path(tmpdir) / "temp_cool.jshd"
            temp_warm = Path(tmpdir) / "temp_warm.jshd"
            for f in [precip_low, precip_high, temp_cool, temp_warm]:
                f.write_bytes(b"test data")

            config = JobConfig(
                template_string="# config",
                sweep=SweepConfig(
                    file_parameters=[
                        FileSweepParameter(name="precip", paths=[precip_low, precip_high]),
                        FileSweepParameter(name="temp", paths=[temp_cool, temp_warm]),
                    ],
                ),
                source_path=Path(tmpdir) / "sim.josh",
            )
            config.source_path.write_text("simulation")

            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                # 2 x 2 = 4 jobs
                self.assertEqual(len(job_set), 4)

                for job in job_set:
                    # Both file parameter labels should be in parameters
                    self.assertIn("precip", job.parameters)
                    self.assertIn("temp", job.parameters)
                    self.assertIn(job.parameters["precip"], ["precip_low", "precip_high"])
                    self.assertIn(job.parameters["temp"], ["temp_cool", "temp_warm"])

                    # Also verify they're still in file_mappings
                    self.assertIn("precip", job.file_mappings)
                    self.assertIn("temp", job.file_mappings)
            finally:
                job_set.cleanup()


class TestJobSet(unittest.TestCase):
    """Tests for JobSet class."""

    def test_len(self):
        """Length should match number of jobs."""
        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content="",
                config_path=Path("/tmp/test"),
                config_name="test",
                run_hash="abc",
                parameters={},
                simulation="Main",
                replicates=1,
            )
        ])
        self.assertEqual(len(job_set), 1)

    def test_total_jobs_property(self):
        """total_jobs should return number of job configurations."""
        jobs = [
            ExpandedJob(
                config_content=f"val={i}",
                config_path=Path(f"/tmp/test{i}"),
                config_name="test",
                run_hash=f"hash{i}",
                parameters={"i": i},
                simulation="Main",
                replicates=3,
            )
            for i in range(5)
        ]
        job_set = JobSet(jobs=jobs)
        self.assertEqual(job_set.total_jobs, 5)

    def test_total_replicates_property(self):
        """total_replicates should sum replicates across all jobs."""
        jobs = [
            ExpandedJob(
                config_content="val=1",
                config_path=Path("/tmp/test1"),
                config_name="test",
                run_hash="hash1",
                parameters={},
                simulation="Main",
                replicates=3,
            ),
            ExpandedJob(
                config_content="val=2",
                config_path=Path("/tmp/test2"),
                config_name="test",
                run_hash="hash2",
                parameters={},
                simulation="Main",
                replicates=5,
            ),
        ]
        job_set = JobSet(jobs=jobs)
        self.assertEqual(job_set.total_replicates, 8)

    def test_total_replicates_empty(self):
        """total_replicates should be 0 for empty JobSet."""
        job_set = JobSet()
        self.assertEqual(job_set.total_replicates, 0)
        self.assertEqual(job_set.total_jobs, 0)

    def test_iteration(self):
        """Should be iterable over jobs."""
        jobs = [
            ExpandedJob(
                config_content=f"val={i}",
                config_path=Path(f"/tmp/test{i}"),
                config_name="test",
                run_hash=f"hash{i}",
                parameters={"i": i},
                simulation="Main",
                replicates=1,
            )
            for i in range(3)
        ]
        job_set = JobSet(jobs=jobs)

        collected = list(job_set)
        self.assertEqual(len(collected), 3)

    def test_cleanup(self):
        """Cleanup should remove temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "sweep"
            temp_path.mkdir()
            (temp_path / "test.txt").write_text("test")

            job_set = JobSet(temp_dir=temp_path)
            self.assertTrue(temp_path.exists())

            job_set.cleanup()
            self.assertFalse(temp_path.exists())
            self.assertIsNone(job_set.temp_dir)


class TestToRunConfig(unittest.TestCase):
    """Tests for to_run_config function."""

    def test_basic_conversion(self):
        """Basic job should convert to RunConfig."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
        )

        run_config = to_run_config(job)

        self.assertEqual(run_config.script, Path("/path/to/source.josh"))
        self.assertEqual(run_config.simulation, "Main")
        self.assertEqual(run_config.replicates, 1)
        self.assertIn("editor", run_config.data)
        self.assertEqual(run_config.data["editor"], Path("/tmp/editor.jshc"))

    def test_with_replicates(self):
        """Job with replicates should convert correctly."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=5,
            source_path=Path("/path/to/source.josh"),
        )

        run_config = to_run_config(job)

        self.assertEqual(run_config.replicates, 5)

    def test_with_custom_tags(self):
        """Job with custom tags should include them in config."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
            custom_tags={"x": "1", "y": "test"},
        )

        run_config = to_run_config(job)

        self.assertEqual(run_config.custom_tags["x"], "1")
        self.assertEqual(run_config.custom_tags["y"], "test")

    def test_with_options(self):
        """Job with options should include them in config."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
            seed=42,
            crs="EPSG:4326",
            use_float64=True,
            output_steps="0-10",
        )

        run_config = to_run_config(job)

        self.assertEqual(run_config.seed, 42)
        self.assertEqual(run_config.crs, "EPSG:4326")
        self.assertTrue(run_config.use_float64)
        self.assertEqual(run_config.output_steps, "0-10")

    def test_with_file_mappings(self):
        """Job with file mappings should include them in data dict."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
            file_mappings={"data": Path("/path/to/data.jshd")},
        )

        run_config = to_run_config(job)

        self.assertIn("editor", run_config.data)
        self.assertIn("data", run_config.data)
        self.assertEqual(run_config.data["data"], Path("/path/to/data.jshd"))

    def test_missing_source_path_raises(self):
        """Job without source_path should raise ValueError."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=None,
        )

        with self.assertRaises(ValueError):
            to_run_config(job)


class TestToRunRemoteConfig(unittest.TestCase):
    """Tests for to_run_remote_config function."""

    def test_basic_conversion(self):
        """Basic job should convert to RunRemoteConfig."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
        )

        run_config = to_run_remote_config(job, api_key="test-key")

        self.assertEqual(run_config.script, Path("/path/to/source.josh"))
        self.assertEqual(run_config.simulation, "Main")
        self.assertEqual(run_config.api_key, "test-key")
        self.assertIsNone(run_config.endpoint)

    def test_with_endpoint(self):
        """Job should convert with custom endpoint."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
        )

        run_config = to_run_remote_config(
            job, api_key="test-key", endpoint="https://custom.josh.cloud"
        )

        self.assertEqual(run_config.endpoint, "https://custom.josh.cloud")

    def test_with_custom_tags(self):
        """Job with custom tags should include them in config."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
            custom_tags={"x": "1", "param": "value"},
        )

        run_config = to_run_remote_config(job, api_key="test-key")

        self.assertEqual(run_config.custom_tags["x"], "1")
        self.assertEqual(run_config.custom_tags["param"], "value")

    def test_missing_source_path_raises(self):
        """Job without source_path should raise ValueError."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=None,
        )

        with self.assertRaises(ValueError):
            to_run_remote_config(job, api_key="test-key")


class TestSweepResult(unittest.TestCase):
    """Tests for SweepResult class."""

    def test_empty_result(self):
        """Empty result should have zero counts."""
        result = SweepResult()
        self.assertEqual(result.succeeded, 0)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result), 0)

    def test_iteration(self):
        """Should be iterable over job_results."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/test.jshc"),
            config_name="test",
            run_hash="abc123",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
        )
        result = SweepResult(
            job_results=[(job, {"success": True})],
            succeeded=1,
            failed=0,
        )

        collected = list(result)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0][0], job)

    def test_len(self):
        """Length should match number of job_results."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/test.jshc"),
            config_name="test",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
        )
        result = SweepResult(
            job_results=[(job, None), (job, None), (job, None)],
            succeeded=2,
            failed=1,
        )
        self.assertEqual(len(result), 3)


class TestRunSweep(unittest.TestCase):
    """Tests for run_sweep function."""

    def test_dry_run_returns_empty(self):
        """Dry run should return empty SweepResult."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content="test",
                config_path=Path("/tmp/test.jshc"),
                config_name="test",
                run_hash="abc123",
                parameters={"x": 1},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
        ])

        result = run_sweep(cli, job_set, dry_run=True, quiet=True)

        self.assertEqual(result.succeeded, 0)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result), 0)
        # CLI.run should not be called
        cli.run.assert_not_called()

    def test_remote_without_api_key(self):
        """remote=True without api_key should work (for local servers)."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        cli.run_remote.return_value = MagicMock(success=True, exit_code=0)

        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content="test",
                config_path=Path("/tmp/test.jshc"),
                config_name="test",
                run_hash="abc123",
                parameters={},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
        ])

        # api_key is now optional (for local servers), so this should NOT raise
        result = run_sweep(cli, job_set, remote=True, quiet=True)
        self.assertEqual(result.succeeded, 1)
        cli.run_remote.assert_called_once()

    def test_remote_with_api_key_uses_run_remote(self):
        """remote=True with api_key should use cli.run_remote()."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        cli.run_remote.return_value = MagicMock(success=True, exit_code=0)

        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content="test",
                config_path=Path("/tmp/test.jshc"),
                config_name="test",
                run_hash="abc123",
                parameters={"x": 1},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
        ])

        result = run_sweep(cli, job_set, remote=True, api_key="test-key", quiet=True)

        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 0)
        # run_remote should be called, not run
        cli.run_remote.assert_called_once()
        cli.run.assert_not_called()

    def test_remote_with_endpoint(self):
        """remote=True with custom endpoint should pass it to run_remote()."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        cli.run_remote.return_value = MagicMock(success=True, exit_code=0)

        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content="test",
                config_path=Path("/tmp/test.jshc"),
                config_name="test",
                run_hash="abc123",
                parameters={},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
        ])

        run_sweep(
            cli, job_set,
            remote=True,
            api_key="test-key",
            endpoint="https://custom.josh.cloud",
            quiet=True,
        )

        # Verify run_remote was called with a config containing the endpoint
        cli.run_remote.assert_called_once()
        run_config = cli.run_remote.call_args[0][0]
        self.assertEqual(run_config.endpoint, "https://custom.josh.cloud")
        self.assertEqual(run_config.api_key, "test-key")

    def test_local_run_does_not_use_run_remote(self):
        """remote=False (default) should use cli.run(), not run_remote()."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        cli.run.return_value = MagicMock(success=True, exit_code=0)

        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content="test",
                config_path=Path("/tmp/test.jshc"),
                config_name="test",
                run_hash="abc123",
                parameters={},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
        ])

        result = run_sweep(cli, job_set, quiet=True)

        self.assertEqual(result.succeeded, 1)
        # run should be called, not run_remote
        cli.run.assert_called_once()
        cli.run_remote.assert_not_called()

    def test_run_with_callback(self):
        """Callback should be called for each job."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        cli.run.return_value = MagicMock(success=True, exit_code=0)

        callback = MagicMock()

        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content="test",
                config_path=Path("/tmp/test.jshc"),
                config_name="test",
                run_hash="abc123",
                parameters={"x": 1},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
        ])

        result = run_sweep(cli, job_set, on_complete=callback, quiet=True)

        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(callback.call_count, 1)

    @unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
    def test_registry_registers_export_and_debug_outputs(self):
        """run_sweep should register resolved export/debug outputs in run_outputs."""
        from unittest.mock import MagicMock

        from joshpy.cli import CLIResult, ExportFileInfo, ExportPaths
        from joshpy.registry import RunRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.josh"
            source_path.write_text("start simulation Main\nend simulation\n")

            # Create actual files so file_size is recorded
            patch_0 = Path(tmpdir) / "patch_hashabc123_0.csv"
            patch_1 = Path(tmpdir) / "patch_hashabc123_1.csv"
            dbg_0 = Path(tmpdir) / "debug_hashabc123_0.txt"
            dbg_1 = Path(tmpdir) / "debug_hashabc123_1.txt"
            patch_0.write_text("step,replicate,value\n0,0,1\n")
            patch_1.write_text("step,replicate,value\n0,1,1\n")
            dbg_0.write_text("[Step 0, organism @ aaa (1.0, 2.0)] init\n")
            dbg_1.write_text("[Step 1, organism @ aaa (1.0, 2.0)] step\n")

            cli = MagicMock()
            cli.run.return_value = CLIResult(
                exit_code=0,
                stdout="",
                stderr="",
                command=["java", "-jar", "josh.jar", "run"],
            )
            cli.inspect_exports.return_value = ExportPaths(
                simulation="Main",
                export_files={
                    "patch": ExportFileInfo(
                        raw=f"file://{tmpdir}/patch_{{run_hash}}_{{replicate}}.csv",
                        protocol="file",
                        host="",
                        path=f"{tmpdir}/patch_{{run_hash}}_{{replicate}}.csv",
                        file_type="csv",
                    ),
                    "meta": None,
                    "entity": None,
                },
                debug_files={
                    "organism": ExportFileInfo(
                        raw=f"file://{tmpdir}/debug_{{run_hash}}_{{replicate}}.txt",
                        protocol="file",
                        host="",
                        path=f"{tmpdir}/debug_{{run_hash}}_{{replicate}}.txt",
                        file_type="txt",
                    ),
                    "patch": None,
                    "agent": None,
                    "disturbance": None,
                },
            )

            job = ExpandedJob(
                config_content="test",
                config_path=Path(tmpdir) / "test.jshc",
                config_name="test",
                run_hash="hashabc123",
                parameters={},
                simulation="Main",
                replicates=2,
                source_path=source_path,
                custom_tags={"run_hash": "hashabc123"},
            )
            job_set = JobSet(jobs=[job])

            registry = RunRegistry(":memory:")
            try:
                session_id = registry.create_session(config=JobConfig(simulation="Main"))
                registry.register_run(
                    session_id=session_id,
                    run_hash="hashabc123",
                    josh_path=str(source_path),
                    config_content="test",
                    file_mappings=None,
                    parameters={},
                )

                result = run_sweep(
                    cli,
                    job_set,
                    registry=registry,
                    session_id=session_id,
                    quiet=True,
                )

                self.assertEqual(result.succeeded, 1)
                self.assertEqual(cli.inspect_exports.call_count, 1)

                run_id = result.run_ids["hashabc123"]
                rows = registry.conn.execute(
                    """
                    SELECT output_type, file_path
                    FROM run_outputs
                    WHERE run_id = ?
                    ORDER BY output_type, file_path
                    """,
                    [run_id],
                ).fetchall()

                self.assertEqual(len(rows), 4)
                self.assertEqual(
                    rows,
                    [
                        ("debug.organism", str(dbg_0)),
                        ("debug.organism", str(dbg_1)),
                        ("export.patch", str(patch_0)),
                        ("export.patch", str(patch_1)),
                    ],
                )
            finally:
                registry.close()

    def test_stop_on_failure_raises_error(self):
        """Should raise SweepExecutionError on first failure when stop_on_failure=True."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        # First call succeeds, second fails
        cli.run.side_effect = [
            MagicMock(success=True, exit_code=0),
            MagicMock(success=False, exit_code=143, stderr="Process killed"),
            MagicMock(success=True, exit_code=0),
        ]

        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content=f"test{i}",
                config_path=Path(f"/tmp/test{i}.jshc"),
                config_name="test",
                run_hash=f"hash{i}",
                parameters={"i": i},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
            for i in range(3)
        ])

        # Should raise SweepExecutionError on failure
        with self.assertRaises(SweepExecutionError) as ctx:
            run_sweep(cli, job_set, stop_on_failure=True, quiet=True)

        # Check exception details
        self.assertEqual(ctx.exception.trial_num, 1)  # Second job (index 1)
        self.assertEqual(ctx.exception.succeeded_before, 1)
        self.assertEqual(ctx.exception.result.exit_code, 143)
        self.assertEqual(ctx.exception.job.parameters, {"i": 1})

    def test_stop_on_failure_false_continues(self):
        """Should continue and return results when stop_on_failure=False."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        # First call succeeds, second fails, third succeeds
        cli.run.side_effect = [
            MagicMock(success=True, exit_code=0),
            MagicMock(success=False, exit_code=1, stderr="Error"),
            MagicMock(success=True, exit_code=0),
        ]

        job_set = JobSet(jobs=[
            ExpandedJob(
                config_content=f"test{i}",
                config_path=Path(f"/tmp/test{i}.jshc"),
                config_name="test",
                run_hash=f"hash{i}",
                parameters={"i": i},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
            for i in range(3)
        ])

        result = run_sweep(cli, job_set, stop_on_failure=False, quiet=True)

        # Should complete all jobs
        self.assertEqual(len(result), 3)
        self.assertEqual(result.succeeded, 2)
        self.assertEqual(result.failed, 1)


class TestRunSweepStatusManagement(unittest.TestCase):
    """Tests for run_sweep() automatic session status management."""

    def _make_test_job_set(self, num_jobs=1):
        """Create a test JobSet."""
        return JobSet(jobs=[
            ExpandedJob(
                config_content=f"test{i}",
                config_path=Path(f"/tmp/test{i}.jshc"),
                config_name="test",
                run_hash=f"hash{i:04d}",
                parameters={"i": i},
                simulation="Main",
                replicates=1,
                source_path=Path("/tmp/source.josh"),
            )
            for i in range(num_jobs)
        ])

    def _make_cli_result(self, success=True, exit_code=0):
        """Create a CLIResult for testing."""
        from joshpy.cli import CLIResult
        return CLIResult(
            exit_code=exit_code,
            stdout="",
            stderr="" if success else "Error occurred",
            command=["java", "-jar", "test.jar"],
        )

    def test_sets_running_status_at_start(self):
        """run_sweep should set status to 'running' at start when manage_status=True."""
        from unittest.mock import MagicMock, call
        
        cli = MagicMock()
        cli.run.return_value = self._make_cli_result(success=True)
        
        registry = MagicMock()
        job_set = self._make_test_job_set()
        
        run_sweep(cli, job_set, registry=registry, session_id="test-session", quiet=True)
        
        # Check that update_session_status was called with "running"
        calls = registry.update_session_status.call_args_list
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], call("test-session", "running"))

    def test_sets_completed_status_on_success(self):
        """run_sweep should set status to 'completed' when all jobs succeed."""
        from unittest.mock import MagicMock, call
        
        cli = MagicMock()
        cli.run.return_value = self._make_cli_result(success=True)
        
        registry = MagicMock()
        job_set = self._make_test_job_set(num_jobs=2)
        
        run_sweep(cli, job_set, registry=registry, session_id="test-session", quiet=True)
        
        # Check final status is "completed"
        calls = registry.update_session_status.call_args_list
        self.assertEqual(calls[-1], call("test-session", "completed"))

    def test_sets_failed_status_on_job_failure(self):
        """run_sweep should set status to 'failed' when any job fails."""
        from unittest.mock import MagicMock, call
        
        cli = MagicMock()
        # First job succeeds, second job fails
        cli.run.side_effect = [
            self._make_cli_result(success=True, exit_code=0),
            self._make_cli_result(success=False, exit_code=1),
        ]
        
        registry = MagicMock()
        job_set = self._make_test_job_set(num_jobs=2)
        
        # With stop_on_failure=True (default), an exception is raised
        with self.assertRaises(SweepExecutionError):
            run_sweep(cli, job_set, registry=registry, session_id="test-session", quiet=True)
        
        # Check final status is "failed"
        calls = registry.update_session_status.call_args_list
        self.assertEqual(calls[-1], call("test-session", "failed"))
    
    def test_sets_failed_status_on_job_failure_stop_false(self):
        """run_sweep should set status to 'failed' when any job fails with stop_on_failure=False."""
        from unittest.mock import MagicMock, call
        
        cli = MagicMock()
        # First job succeeds, second job fails
        cli.run.side_effect = [
            self._make_cli_result(success=True, exit_code=0),
            self._make_cli_result(success=False, exit_code=1),
        ]
        
        registry = MagicMock()
        job_set = self._make_test_job_set(num_jobs=2)
        
        # With stop_on_failure=False, no exception - just returns with status
        run_sweep(
            cli, job_set, 
            registry=registry, 
            session_id="test-session", 
            stop_on_failure=False,
            quiet=True
        )
        
        # Check final status is "failed"
        calls = registry.update_session_status.call_args_list
        self.assertEqual(calls[-1], call("test-session", "failed"))

    def test_sets_failed_status_on_exception(self):
        """run_sweep should set status to 'failed' on exception."""
        from unittest.mock import MagicMock, call
        
        cli = MagicMock()
        cli.run.side_effect = RuntimeError("Simulation crashed")
        
        registry = MagicMock()
        job_set = self._make_test_job_set()
        
        with self.assertRaises(RuntimeError):
            run_sweep(cli, job_set, registry=registry, session_id="test-session", quiet=True)
        
        # Check that status was set to "failed"
        calls = registry.update_session_status.call_args_list
        self.assertEqual(len(calls), 2)  # "running" then "failed"
        self.assertEqual(calls[-1], call("test-session", "failed"))

    def test_manage_status_false_skips_updates(self):
        """run_sweep should not update status when manage_status=False."""
        from unittest.mock import MagicMock
        
        cli = MagicMock()
        cli.run.return_value = self._make_cli_result(success=True)
        
        registry = MagicMock()
        job_set = self._make_test_job_set()
        
        run_sweep(
            cli, job_set,
            registry=registry,
            session_id="test-session",
            manage_status=False,
            quiet=True,
        )
        
        # update_session_status should NOT be called
        registry.update_session_status.assert_not_called()

    def test_manage_status_default_is_true(self):
        """manage_status should default to True."""
        from unittest.mock import MagicMock
        
        cli = MagicMock()
        cli.run.return_value = self._make_cli_result(success=True)
        
        registry = MagicMock()
        job_set = self._make_test_job_set()
        
        # Don't pass manage_status explicitly
        run_sweep(cli, job_set, registry=registry, session_id="test-session", quiet=True)
        
        # update_session_status should be called (default manage_status=True)
        self.assertGreaterEqual(registry.update_session_status.call_count, 2)

    def test_no_status_update_without_registry(self):
        """run_sweep should not call update_session_status without registry."""
        from unittest.mock import MagicMock
        
        cli = MagicMock()
        cli.run.return_value = self._make_cli_result(success=True)
        
        job_set = self._make_test_job_set()
        
        # Should not raise even without registry
        result = run_sweep(cli, job_set, quiet=True)
        
        self.assertEqual(result.succeeded, 1)
        # No registry means no status updates (no error)


class TestCompoundSweepParameter(unittest.TestCase):
    """Tests for CompoundSweepParameter class."""

    def test_basic_creation_with_config_params(self):
        """Basic compound creation with config parameters."""
        compound = CompoundSweepParameter(
            name="growth_regime",
            parameters=[
                ConfigSweepParameter(name="growthRate", values=[0.1, 0.5, 1.0]),
                ConfigSweepParameter(name="mortalityRate", values=[0.05, 0.1, 0.2]),
            ],
            labels=["slow", "medium", "fast"],
        )
        self.assertEqual(compound.name, "growth_regime")
        self.assertEqual(len(compound), 3)
        self.assertEqual(compound.labels, ["slow", "medium", "fast"])

    def test_basic_creation_with_file_params(self):
        """Basic compound creation with file parameters."""
        compound = CompoundSweepParameter(
            name="climate_scenario",
            parameters=[
                FileSweepParameter(name="temp", paths=["temp_ssp126.jshd", "temp_ssp245.jshd"]),
                FileSweepParameter(name="precip", paths=["precip_ssp126.jshd", "precip_ssp245.jshd"]),
            ],
            labels=["ssp126", "ssp245"],
        )
        self.assertEqual(compound.name, "climate_scenario")
        self.assertEqual(len(compound), 2)

    def test_mixed_config_and_file_params(self):
        """Compound with mixed config and file parameters."""
        compound = CompoundSweepParameter(
            name="scenario",
            parameters=[
                FileSweepParameter(name="climate", paths=["rcp26.jshd", "rcp85.jshd"]),
                ConfigSweepParameter(name="adaptation", values=["none", "aggressive"]),
            ],
            labels=["optimistic", "pessimistic"],
        )
        self.assertEqual(len(compound), 2)
        self.assertEqual(len(compound.parameters), 2)

    def test_auto_labels_when_not_provided(self):
        """Labels should auto-generate as indices when not provided."""
        compound = CompoundSweepParameter(
            name="test",
            parameters=[
                ConfigSweepParameter(name="x", values=[1, 2, 3]),
            ],
        )
        self.assertEqual(compound.labels, ["0", "1", "2"])

    def test_mismatched_lengths_raises(self):
        """Mismatched parameter lengths should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CompoundSweepParameter(
                name="test",
                parameters=[
                    ConfigSweepParameter(name="a", values=[1, 2, 3]),
                    ConfigSweepParameter(name="b", values=[10, 20]),  # Different length
                ],
            )
        self.assertIn("same number of values", str(ctx.exception))

    def test_wrong_labels_count_raises(self):
        """Wrong number of labels should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CompoundSweepParameter(
                name="test",
                parameters=[
                    ConfigSweepParameter(name="x", values=[1, 2, 3]),
                ],
                labels=["a", "b"],  # Only 2 labels for 3 values
            )
        self.assertIn("3 scenarios", str(ctx.exception))
        self.assertIn("2 labels", str(ctx.exception))

    def test_invalid_parameter_type_raises(self):
        """Invalid parameter types should raise TypeError."""
        with self.assertRaises(TypeError) as ctx:
            CompoundSweepParameter(
                name="test",
                parameters=["not a parameter"],  # type: ignore
            )
        self.assertIn("ConfigSweepParameter or FileSweepParameter", str(ctx.exception))

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        compound = CompoundSweepParameter(
            name="regime",
            parameters=[
                ConfigSweepParameter(name="a", values=[1, 2]),
                FileSweepParameter(name="data", paths=["x.jshd", "y.jshd"]),
            ],
            labels=["low", "high"],
        )
        result = compound.to_dict()
        
        self.assertEqual(result["name"], "regime")
        self.assertEqual(result["labels"], ["low", "high"])
        self.assertEqual(len(result["parameters"]), 2)
        self.assertEqual(result["parameters"][0]["_type"], "config")
        self.assertEqual(result["parameters"][1]["_type"], "file")

    def test_to_dict_skips_numeric_labels(self):
        """to_dict should skip auto-generated numeric labels."""
        compound = CompoundSweepParameter(
            name="test",
            parameters=[
                ConfigSweepParameter(name="x", values=[1, 2]),
            ],
        )
        result = compound.to_dict()
        
        self.assertNotIn("labels", result)

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        data = {
            "name": "regime",
            "parameters": [
                {"_type": "config", "name": "a", "values": [1, 2]},
                {"_type": "file", "name": "data", "paths": ["x.jshd", "y.jshd"]},
            ],
            "labels": ["low", "high"],
        }
        compound = CompoundSweepParameter.from_dict(data)
        
        self.assertEqual(compound.name, "regime")
        self.assertEqual(compound.labels, ["low", "high"])
        self.assertEqual(len(compound.parameters), 2)
        self.assertIsInstance(compound.parameters[0], ConfigSweepParameter)
        self.assertIsInstance(compound.parameters[1], FileSweepParameter)

    def test_from_dict_infers_type_from_paths(self):
        """from_dict should infer file type from paths key."""
        data = {
            "name": "test",
            "parameters": [
                {"name": "a", "values": [1, 2]},  # No _type, but has values
                {"name": "data", "paths": ["x.jshd", "y.jshd"]},  # No _type, but has paths
            ],
        }
        compound = CompoundSweepParameter.from_dict(data)
        
        self.assertIsInstance(compound.parameters[0], ConfigSweepParameter)
        self.assertIsInstance(compound.parameters[1], FileSweepParameter)


class TestSweepConfigWithCompoundParams(unittest.TestCase):
    """Tests for SweepConfig with compound_parameters."""

    def test_expand_compound_only(self):
        """Compound parameters should expand to zipped combinations."""
        config = SweepConfig(
            compound_parameters=[
                CompoundSweepParameter(
                    name="regime",
                    parameters=[
                        ConfigSweepParameter(name="a", values=[1, 2, 3]),
                        ConfigSweepParameter(name="b", values=[10, 20, 30]),
                    ],
                    labels=["low", "medium", "high"],
                ),
            ],
        )
        result = config.expand()
        
        # Should have 3 combos (zipped), NOT 9 (cartesian)
        self.assertEqual(len(result), 3)
        
        # Check first combo
        self.assertEqual(result[0]["regime"], "low")
        self.assertEqual(result[0]["a"], 1)
        self.assertEqual(result[0]["b"], 10)
        
        # Check last combo
        self.assertEqual(result[2]["regime"], "high")
        self.assertEqual(result[2]["a"], 3)
        self.assertEqual(result[2]["b"], 30)

    def test_expand_compound_with_files(self):
        """Compound with file parameters should include path and label."""
        config = SweepConfig(
            compound_parameters=[
                CompoundSweepParameter(
                    name="climate",
                    parameters=[
                        FileSweepParameter(name="temp", paths=[
                            Path("temp_ssp126.jshd"),
                            Path("temp_ssp245.jshd"),
                        ]),
                        FileSweepParameter(name="precip", paths=[
                            Path("precip_ssp126.jshd"),
                            Path("precip_ssp245.jshd"),
                        ]),
                    ],
                    labels=["ssp126", "ssp245"],
                ),
            ],
        )
        result = config.expand()
        
        self.assertEqual(len(result), 2)
        
        # Check first combo (ssp126)
        self.assertEqual(result[0]["climate"], "ssp126")
        self.assertEqual(result[0]["temp"]["path"], Path("temp_ssp126.jshd"))
        self.assertEqual(result[0]["temp"]["label"], "temp_ssp126")
        self.assertEqual(result[0]["precip"]["path"], Path("precip_ssp126.jshd"))
        self.assertEqual(result[0]["precip"]["label"], "precip_ssp126")
        
        # Check second combo (ssp245)
        self.assertEqual(result[1]["climate"], "ssp245")
        self.assertEqual(result[1]["temp"]["path"], Path("temp_ssp245.jshd"))

    def test_expand_compound_with_config_params(self):
        """Compound parameters should cartesian with config_parameters."""
        config = SweepConfig(
            config_parameters=[
                ConfigSweepParameter(name="x", values=[1, 2]),
            ],
            compound_parameters=[
                CompoundSweepParameter(
                    name="regime",
                    parameters=[
                        ConfigSweepParameter(name="a", values=[10, 20]),
                        ConfigSweepParameter(name="b", values=[100, 200]),
                    ],
                    labels=["low", "high"],
                ),
            ],
        )
        result = config.expand()
        
        # 2 (x values) * 2 (compound scenarios) = 4 combos
        self.assertEqual(len(result), 4)
        
        # Should have cartesian of x with compound
        x_values = sorted(set(r["x"] for r in result))
        regimes = sorted(set(r["regime"] for r in result))
        self.assertEqual(x_values, [1, 2])
        self.assertEqual(regimes, ["high", "low"])

    def test_expand_multiple_compounds(self):
        """Multiple compound parameters should cartesian with each other."""
        config = SweepConfig(
            compound_parameters=[
                CompoundSweepParameter(
                    name="climate",
                    parameters=[
                        ConfigSweepParameter(name="temp", values=[20, 25]),
                    ],
                    labels=["cool", "warm"],
                ),
                CompoundSweepParameter(
                    name="soil",
                    parameters=[
                        ConfigSweepParameter(name="ph", values=[6, 7, 8]),
                    ],
                    labels=["acidic", "neutral", "alkaline"],
                ),
            ],
        )
        result = config.expand()
        
        # 2 (climate) * 3 (soil) = 6 combos
        self.assertEqual(len(result), 6)

    def test_len_with_compound_params(self):
        """Length should include compound parameter scenarios."""
        config = SweepConfig(
            config_parameters=[
                ConfigSweepParameter(name="x", values=[1, 2, 3]),  # 3
            ],
            compound_parameters=[
                CompoundSweepParameter(
                    name="regime",
                    parameters=[
                        ConfigSweepParameter(name="a", values=[10, 20]),
                    ],
                    labels=["low", "high"],
                ),  # 2
            ],
        )
        self.assertEqual(len(config), 6)  # 3 * 2

    def test_bool_with_compound_params(self):
        """Boolean check should include compound parameters."""
        empty = SweepConfig()
        self.assertFalse(bool(empty.config_parameters) or bool(empty.compound_parameters))
        
        with_compound = SweepConfig(
            compound_parameters=[
                CompoundSweepParameter(
                    name="test",
                    parameters=[ConfigSweepParameter(name="x", values=[1])],
                ),
            ],
        )
        self.assertTrue(bool(with_compound))

    def test_to_dict_with_compound_params(self):
        """to_dict should serialize compound_parameters."""
        config = SweepConfig(
            compound_parameters=[
                CompoundSweepParameter(
                    name="regime",
                    parameters=[
                        ConfigSweepParameter(name="a", values=[1, 2]),
                    ],
                    labels=["low", "high"],
                ),
            ],
        )
        result = config.to_dict()
        
        self.assertIn("compound_parameters", result)
        self.assertEqual(len(result["compound_parameters"]), 1)
        self.assertEqual(result["compound_parameters"][0]["name"], "regime")

    def test_from_dict_with_compound_params(self):
        """from_dict should deserialize compound_parameters."""
        data = {
            "config_parameters": [{"name": "x", "values": [1, 2]}],
            "compound_parameters": [
                {
                    "name": "regime",
                    "parameters": [
                        {"_type": "config", "name": "a", "values": [10, 20]},
                    ],
                    "labels": ["low", "high"],
                },
            ],
        }
        config = SweepConfig.from_dict(data)
        
        self.assertEqual(len(config.config_parameters), 1)
        self.assertEqual(len(config.compound_parameters), 1)
        self.assertEqual(config.compound_parameters[0].name, "regime")


class TestJobExpanderWithCompoundParams(unittest.TestCase):
    """Tests for JobExpander with compound_parameters."""

    def test_expand_with_compound_config_params(self):
        """Expanding with compound config parameters should create correct jobs."""
        config = JobConfig(
            template_string="a={{ a }}, b={{ b }}",
            sweep=SweepConfig(
                compound_parameters=[
                    CompoundSweepParameter(
                        name="regime",
                        parameters=[
                            ConfigSweepParameter(name="a", values=[1, 2]),
                            ConfigSweepParameter(name="b", values=[10, 20]),
                        ],
                        labels=["low", "high"],
                    ),
                ],
            ),
        )
        expander = JobExpander()
        job_set = expander.expand(config)
        
        try:
            self.assertEqual(len(job_set), 2)
            
            # Check first job (low)
            job_low = job_set.jobs[0]
            self.assertEqual(job_low.config_content, "a=1, b=10")
            self.assertEqual(job_low.parameters["regime"], "low")
            self.assertEqual(job_low.parameters["a"], 1)
            self.assertEqual(job_low.parameters["b"], 10)
            self.assertEqual(job_low.custom_tags["regime"], "low")
            
            # Check second job (high)
            job_high = job_set.jobs[1]
            self.assertEqual(job_high.config_content, "a=2, b=20")
            self.assertEqual(job_high.parameters["regime"], "high")
        finally:
            job_set.cleanup()

    def test_expand_with_compound_file_params(self):
        """Expanding with compound file parameters should update file_mappings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            temp_ssp126 = Path(tmpdir) / "temp_ssp126.jshd"
            temp_ssp245 = Path(tmpdir) / "temp_ssp245.jshd"
            precip_ssp126 = Path(tmpdir) / "precip_ssp126.jshd"
            precip_ssp245 = Path(tmpdir) / "precip_ssp245.jshd"
            for f in [temp_ssp126, temp_ssp245, precip_ssp126, precip_ssp245]:
                f.write_bytes(b"test data")

            config = JobConfig(
                template_string="# climate config",
                source_path=Path(tmpdir) / "sim.josh",
                sweep=SweepConfig(
                    compound_parameters=[
                        CompoundSweepParameter(
                            name="climate",
                            parameters=[
                                FileSweepParameter(name="temp", paths=[temp_ssp126, temp_ssp245]),
                                FileSweepParameter(name="precip", paths=[precip_ssp126, precip_ssp245]),
                            ],
                            labels=["ssp126", "ssp245"],
                        ),
                    ],
                ),
            )
            config.source_path.write_text("simulation")

            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 2)

                # First job should have ssp126 files
                job_126 = job_set.jobs[0]
                self.assertEqual(job_126.parameters["climate"], "ssp126")
                self.assertEqual(job_126.file_mappings["temp"], temp_ssp126)
                self.assertEqual(job_126.file_mappings["precip"], precip_ssp126)

                # Second job should have ssp245 files
                job_245 = job_set.jobs[1]
                self.assertEqual(job_245.parameters["climate"], "ssp245")
                self.assertEqual(job_245.file_mappings["temp"], temp_ssp245)
                self.assertEqual(job_245.file_mappings["precip"], precip_ssp245)
            finally:
                job_set.cleanup()

    def test_expand_compound_cartesian_with_config(self):
        """Compound should cartesian with regular config_parameters."""
        config = JobConfig(
            template_string="x={{ x }}, a={{ a }}",
            sweep=SweepConfig(
                config_parameters=[
                    ConfigSweepParameter(name="x", values=[1, 2]),
                ],
                compound_parameters=[
                    CompoundSweepParameter(
                        name="regime",
                        parameters=[
                            ConfigSweepParameter(name="a", values=[10, 20]),
                        ],
                        labels=["low", "high"],
                    ),
                ],
            ),
        )
        expander = JobExpander()
        job_set = expander.expand(config)

        try:
            # 2 (x) * 2 (regime) = 4 jobs
            self.assertEqual(len(job_set), 4)

            # Verify all combinations exist
            combos = [(j.parameters["x"], j.parameters["regime"]) for j in job_set]
            expected = [(1, "low"), (1, "high"), (2, "low"), (2, "high")]
            self.assertEqual(sorted(combos), sorted(expected))
        finally:
            job_set.cleanup()


class TestConfigPath(unittest.TestCase):
    """Tests for config_path support (raw .jshc without templating)."""

    def test_expand_with_config_path(self):
        """config_path should produce single job with file content and auto-parsed params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "baseline.jshc"
            config_file.write_text("maxGrowth = 50 meters\nfireYear = 75 count")

            config = JobConfig(
                config_path=config_file,
                simulation="Main",
            )
            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 1)
                job = job_set.jobs[0]
                self.assertEqual(job.config_content, "maxGrowth = 50 meters\nfireYear = 75 count")
                # Parameters are auto-parsed from .jshc content
                self.assertEqual(job.parameters, {"maxGrowth": 50, "fireYear": 75})
                # config_name is the logical Josh namespace (with .jshc appended)
                self.assertEqual(job.config_name, "sweep_config.jshc")
            finally:
                job_set.cleanup()

    def test_expand_config_path_with_file_sweep(self):
        """config_path should work with FileSweepParameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.jshc"
            config_file.write_text("value = 42")
            source_file = Path(tmpdir) / "sim.josh"
            source_file.write_text("simulation")
            file_a = Path(tmpdir) / "scenario_a.jshd"
            file_b = Path(tmpdir) / "scenario_b.jshd"
            file_a.write_bytes(b"data a")
            file_b.write_bytes(b"data b")

            config = JobConfig(
                config_path=config_file,
                source_path=source_file,
                sweep=SweepConfig(file_parameters=[
                    FileSweepParameter(name="climate", paths=[file_a, file_b])
                ]),
            )
            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 2)
                # Config content is the same for both (raw file, no rendering)
                for job in job_set:
                    self.assertEqual(job.config_content, "value = 42")
                # But file_mappings differ
                self.assertEqual(job_set.jobs[0].file_mappings["climate"], file_a)
                self.assertEqual(job_set.jobs[1].file_mappings["climate"], file_b)
            finally:
                job_set.cleanup()

    def test_config_path_mutual_exclusivity(self):
        """config_path should be mutually exclusive with template_path/template_string."""
        with self.assertRaises(ValueError):
            JobConfig(
                config_path=Path("config.jshc"),
                template_path=Path("template.jshc.j2"),
            )

        with self.assertRaises(ValueError):
            JobConfig(
                config_path=Path("config.jshc"),
                template_string="value = {{ x }}",
            )

    def test_config_path_serialization_roundtrip(self):
        """config_path should survive to_dict/from_dict roundtrip."""
        config = JobConfig(
            config_path=Path("/some/path/config.jshc"),
            simulation="Test",
            replicates=3,
        )
        d = config.to_dict()
        self.assertEqual(d["config_path"], "/some/path/config.jshc")

        restored = JobConfig.from_dict(d)
        self.assertEqual(restored.config_path, Path("/some/path/config.jshc"))
        self.assertEqual(restored.simulation, "Test")
        self.assertEqual(restored.replicates, 3)


class TestSourceTemplatePath(unittest.TestCase):
    """Tests for source_template_path and template_vars support."""

    def test_source_template_rendering(self):
        """source_template_path should render .josh.j2 with template_vars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create josh template
            josh_template = Path(tmpdir) / "model.josh.j2"
            josh_template.write_text(
                "start simulation {{ sim_name }}\n"
                "  grid.size = {{ grid_size }} m\n"
                "end simulation"
            )
            config_file = Path(tmpdir) / "config.jshc"
            config_file.write_text("value = 42")

            config = JobConfig(
                source_template_path=josh_template,
                template_vars={"sim_name": "TestSim", "grid_size": 30},
                config_path=config_file,
            )
            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 1)
                job = job_set.jobs[0]
                # source_path should be the rendered .josh file
                self.assertIsNotNone(job.source_path)
                rendered_content = job.source_path.read_text()
                self.assertIn("start simulation TestSim", rendered_content)
                self.assertIn("grid.size = 30 m", rendered_content)
            finally:
                job_set.cleanup()

    def test_source_template_mutual_exclusivity(self):
        """source_template_path should be mutually exclusive with source_path."""
        with self.assertRaises(ValueError):
            JobConfig(
                source_path=Path("model.josh"),
                source_template_path=Path("model.josh.j2"),
            )

    def test_template_vars_passed_to_config_template(self):
        """template_vars should be available in .jshc.j2 rendering too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            josh_template = Path(tmpdir) / "model.josh.j2"
            josh_template.write_text("simulation {{ sim_name }}")

            config_template = Path(tmpdir) / "config.jshc.j2"
            config_template.write_text(
                "patchSize = {{ grid_size }} meters\n"
                "maxGrowth = {{ maxGrowth }} meters"
            )

            config = JobConfig(
                source_template_path=josh_template,
                template_vars={"sim_name": "Test", "grid_size": 30},
                template_path=config_template,
                sweep=SweepConfig(config_parameters=[
                    ConfigSweepParameter(name="maxGrowth", values=[10, 50]),
                ]),
            )
            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 2)
                # template_vars (grid_size) should be in rendered config
                self.assertIn("patchSize = 30 meters", job_set.jobs[0].config_content)
                # sweep params should vary
                self.assertIn("maxGrowth = 10 meters", job_set.jobs[0].config_content)
                self.assertIn("maxGrowth = 50 meters", job_set.jobs[1].config_content)
            finally:
                job_set.cleanup()

    def test_sweep_params_override_template_vars(self):
        """Sweep parameters should override template_vars of the same name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_template = Path(tmpdir) / "config.jshc.j2"
            config_template.write_text("x = {{ x }}")

            config = JobConfig(
                template_path=config_template,
                template_vars={"x": "default_value"},
                sweep=SweepConfig(config_parameters=[
                    ConfigSweepParameter(name="x", values=["swept_value"]),
                ]),
            )
            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                self.assertEqual(len(job_set), 1)
                # Sweep param should win over template_var
                self.assertEqual(job_set.jobs[0].config_content, "x = swept_value")
            finally:
                job_set.cleanup()

    def test_source_template_serialization_roundtrip(self):
        """source_template_path and template_vars should survive serialization."""
        config = JobConfig(
            source_template_path=Path("/models/canonical.josh.j2"),
            template_vars={"grid_size": 30, "debug": True},
            config_path=Path("/configs/baseline.jshc"),
        )
        d = config.to_dict()
        self.assertEqual(d["source_template_path"], "/models/canonical.josh.j2")
        self.assertEqual(d["template_vars"], {"grid_size": 30, "debug": True})

        restored = JobConfig.from_dict(d)
        self.assertEqual(restored.source_template_path, Path("/models/canonical.josh.j2"))
        self.assertEqual(restored.template_vars, {"grid_size": 30, "debug": True})

    def test_josh_j2_extension_stripped(self):
        """Rendered .josh file should strip .j2 extension from name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            josh_template = Path(tmpdir) / "model.josh.j2"
            josh_template.write_text("simulation Main")
            config_file = Path(tmpdir) / "config.jshc"
            config_file.write_text("value = 1")

            config = JobConfig(
                source_template_path=josh_template,
                config_path=config_file,
            )
            expander = JobExpander()
            job_set = expander.expand(config)

            try:
                job = job_set.jobs[0]
                # The rendered file should be named model.josh, not model.josh.j2
                self.assertTrue(job.source_path.name.endswith(".josh"))
                self.assertFalse(job.source_path.name.endswith(".j2"))
            finally:
                job_set.cleanup()


class TestDiscoverJshdFiles(unittest.TestCase):
    """Tests for discover_jshd_files utility."""

    def test_flat_directory(self):
        """Should discover .jshd files in a flat directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "cover.jshd").write_bytes(b"data")
            (Path(tmpdir) / "fire_rbr.jshd").write_bytes(b"data")
            (Path(tmpdir) / "readme.txt").write_text("not a jshd")

            result = discover_jshd_files(tmpdir)
            self.assertEqual(len(result), 2)
            self.assertIn("cover", result)
            self.assertIn("fire_rbr", result)
            self.assertTrue(result["cover"].is_absolute())

    def test_recursive(self):
        """Should discover .jshd files in subdirectories when recursive=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "cover.jshd").write_bytes(b"data")
            monthly = Path(tmpdir) / "monthly"
            monthly.mkdir()
            (monthly / "tas_jan.jshd").write_bytes(b"data")
            (monthly / "tas_feb.jshd").write_bytes(b"data")

            # Non-recursive should only find top-level
            result_flat = discover_jshd_files(tmpdir, recursive=False)
            self.assertEqual(len(result_flat), 1)

            # Recursive should find all
            result_recursive = discover_jshd_files(tmpdir, recursive=True)
            self.assertEqual(len(result_recursive), 3)
            self.assertIn("tas_jan", result_recursive)

    def test_missing_directory(self):
        """Should raise FileNotFoundError for non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            discover_jshd_files("/nonexistent/path")

    def test_duplicate_stems(self):
        """Should raise ValueError for duplicate stems in recursive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "cover.jshd").write_bytes(b"data1")
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "cover.jshd").write_bytes(b"data2")

            with self.assertRaises(ValueError):
                discover_jshd_files(tmpdir, recursive=True)

    def test_empty_directory(self):
        """Should return empty dict for directory with no .jshd files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_jshd_files(tmpdir)
            self.assertEqual(result, {})


class TestToBatchRemoteConfig(unittest.TestCase):
    """Tests for to_batch_remote_config()."""

    def test_basic_conversion(self):
        from joshpy.jobs import to_batch_remote_config

        job = ExpandedJob(
            config_content="x = 1 count",
            config_path=Path("/tmp/config.jshc"),
            config_name="config.jshc",
            run_hash="abc123",
            parameters={"x": 1},
            simulation="Main",
            replicates=3,
            source_path=Path("/path/to/sim.josh"),
        )

        config = to_batch_remote_config(job, "gke-test", "sweeps/s1/jobs/abc123/")

        self.assertEqual(config.simulation, "Main")
        self.assertEqual(config.target, "gke-test")
        self.assertEqual(config.minio_prefix, "sweeps/s1/jobs/abc123/")
        self.assertEqual(config.replicates, 3)
        self.assertFalse(config.no_wait)
        # require_prestaged defaults to True (safe default for sweeps)
        self.assertTrue(config.require_prestaged)
        # stage_from_local_dir intentionally not exposed in joshpy
        self.assertFalse(hasattr(config, "stage_from_local_dir"))
        # custom_tags passthrough (empty here; tested in test_custom_tags_propagated)
        self.assertEqual(config.custom_tags, {})
        # replicate_start defaults to 0
        self.assertEqual(config.replicate_start, 0)

    def test_custom_tags_propagated_from_job(self):
        """to_batch_remote_config passes job.custom_tags through unchanged."""
        from joshpy.jobs import to_batch_remote_config

        job = ExpandedJob(
            config_content="x = 1 count",
            config_path=Path("/tmp/config.jshc"),
            config_name="config.jshc",
            run_hash="abc123def456",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/sim.josh"),
            custom_tags={"run_hash": "abc123def456", "label": "exp1", "x": "1"},
        )

        config = to_batch_remote_config(job, "gke-test", "sweeps/s1/jobs/abc123def456/")

        self.assertEqual(config.custom_tags["run_hash"], "abc123def456")
        self.assertEqual(config.custom_tags["label"], "exp1")
        self.assertEqual(config.custom_tags["x"], "1")

    def test_custom_tags_copied_not_shared(self):
        """Mutating the returned custom_tags must not affect job.custom_tags."""
        from joshpy.jobs import to_batch_remote_config

        job = ExpandedJob(
            config_content="x = 1 count",
            config_path=Path("/tmp/config.jshc"),
            config_name="config.jshc",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/sim.josh"),
            custom_tags={"run_hash": "abc123"},
        )

        config = to_batch_remote_config(job, "gke-test", "sweeps/s1/jobs/abc123/")
        config.custom_tags["injected"] = "bad"

        self.assertNotIn("injected", job.custom_tags)

    def test_replicate_start_passthrough(self):
        """to_batch_remote_config accepts replicate_start and forwards it."""
        from joshpy.jobs import to_batch_remote_config

        job = ExpandedJob(
            config_content="x = 1 count",
            config_path=Path("/tmp/config.jshc"),
            config_name="config.jshc",
            run_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=5,
            source_path=Path("/path/to/sim.josh"),
        )

        config = to_batch_remote_config(
            job, "gke-test", "sweeps/s1/jobs/abc123/", replicate_start=10,
        )

        self.assertEqual(config.replicate_start, 10)

    def test_no_wait_mode(self):
        from joshpy.jobs import to_batch_remote_config

        job = ExpandedJob(
            config_content="x = 1 count",
            config_path=Path("/tmp/config.jshc"),
            config_name="config.jshc",
            run_hash="abc123",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/sim.josh"),
        )

        config = to_batch_remote_config(
            job, "gke-test", "sweeps/s1/jobs/abc123/",
            no_wait=True, timeout=600,
        )

        self.assertTrue(config.no_wait)
        self.assertEqual(config.timeout, 600)

    def test_source_path_required(self):
        from joshpy.jobs import to_batch_remote_config

        job = ExpandedJob(
            config_content="x = 1 count",
            config_path=Path("/tmp/config.jshc"),
            config_name="config.jshc",
            run_hash="abc123",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
            source_path=None,
        )

        with self.assertRaises(ValueError, msg="source_path is required"):
            to_batch_remote_config(job, "gke-test", "sweeps/s1/jobs/abc123/")


class TestRunSweepBatchRemote(unittest.TestCase):
    """Tests for run_sweep() batch_remote mode (stage + dispatch)."""

    def _make_real_job(self, tmp: Path, run_hash: str = "abc123") -> ExpandedJob:
        """Build an ExpandedJob backed by a real on-disk .josh file."""
        src = tmp / "sim.josh"
        src.write_text("start simulation Main\nend simulation\n")
        return ExpandedJob(
            config_content="x = 1 count",
            config_path=tmp / "config.jshc",
            config_name="config.jshc",
            run_hash=run_hash,
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
            source_path=src,
        )

    def test_batch_remote_requires_target(self):
        from joshpy.jobs import run_sweep

        with self.assertRaises(ValueError, msg="target is required"):
            run_sweep(
                MagicMock(),
                MagicMock(total_jobs=1, total_replicates=1, __iter__=lambda s: iter([])),
                batch_remote=True,
                target=None,
            )

    def test_batch_remote_exclusive_with_remote(self):
        from joshpy.jobs import run_sweep

        with self.assertRaises(ValueError, msg="mutually exclusive"):
            run_sweep(
                MagicMock(),
                MagicMock(total_jobs=1, total_replicates=1, __iter__=lambda s: iter([])),
                batch_remote=True,
                remote=True,
                target="gke-test",
            )

    def test_stage_then_batch_remote_with_require_prestaged(self):
        """Blocking batch_remote path stages first, then dispatches with
        require_prestaged=True."""
        from joshpy.jobs import run_sweep

        mock_cli = MagicMock()
        mock_cli.stage_to_minio.return_value = MagicMock(
            success=True, exit_code=0, stdout="", stderr="",
        )
        mock_cli.batch_remote.return_value = MagicMock(
            success=True, exit_code=0, stdout="", stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            job = self._make_real_job(tmp)
            job_set = MagicMock(
                total_jobs=1, total_replicates=1,
                __iter__=lambda s: iter([job]),
            )

            result = run_sweep(
                mock_cli, job_set,
                batch_remote=True,
                target="gke-test",
                session_id="s1",
                quiet=True,
            )

        # stage happened before dispatch
        mock_cli.stage_to_minio.assert_called_once()
        mock_cli.batch_remote.assert_called_once()

        # dispatch config uses the same prefix staged to, with require_prestaged=True
        stage_cfg = mock_cli.stage_to_minio.call_args[0][0]
        dispatch_cfg = mock_cli.batch_remote.call_args[0][0]
        self.assertEqual(stage_cfg.prefix, dispatch_cfg.minio_prefix)
        self.assertTrue(dispatch_cfg.require_prestaged)
        # Per-job prefix is keyed on run_hash
        self.assertIn(job.run_hash, dispatch_cfg.minio_prefix)
        self.assertIn("sweeps/s1/", dispatch_cfg.minio_prefix)

        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 0)

    def test_stage_failure_short_circuits_dispatch(self):
        """If stage_to_minio fails, batch_remote should not be called."""
        from joshpy.jobs import run_sweep

        mock_cli = MagicMock()
        mock_cli.stage_to_minio.return_value = MagicMock(
            success=False, exit_code=1, stdout="", stderr="bucket denied",
        )

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            job = self._make_real_job(tmp)
            job_set = MagicMock(
                total_jobs=1, total_replicates=1,
                __iter__=lambda s: iter([job]),
            )

            result = run_sweep(
                mock_cli, job_set,
                batch_remote=True,
                target="gke-test",
                session_id="s1",
                quiet=True,
                stop_on_failure=False,
            )

        mock_cli.stage_to_minio.assert_called_once()
        mock_cli.batch_remote.assert_not_called()
        self.assertEqual(result.failed, 1)

    def test_async_dispatch_parses_json(self):
        """Async batch_remote should still stage first, then parse --no-wait JSON."""
        import json
        from joshpy.jobs import run_sweep

        dispatch_json = json.dumps({
            "jobId": "test-job-123",
            "target": "gke-test",
            "statusPath": "batch-status/test-job-123/status.json",
        })

        mock_cli = MagicMock()
        mock_cli.stage_to_minio.return_value = MagicMock(
            success=True, exit_code=0, stdout="", stderr="",
        )
        mock_cli.batch_remote.return_value = MagicMock(
            success=True, exit_code=0, stdout=dispatch_json, stderr="",
        )
        mock_cli.poll_batch.return_value = MagicMock(
            success=True, exit_code=0,
            stdout='{"status":"complete"}', stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            job = self._make_real_job(tmp)
            job_set = MagicMock(
                total_jobs=1, total_replicates=1,
                __iter__=lambda s: iter([job]),
            )

            result = run_sweep(
                mock_cli, job_set,
                batch_remote=True,
                target="gke-test",
                batch_no_wait=True,
                session_id="s1",
                quiet=True,
            )

        mock_cli.stage_to_minio.assert_called_once()
        mock_cli.batch_remote.assert_called_once()
        mock_cli.poll_batch.assert_called()
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 0)

    def test_bottle_records_batch_metadata(self):
        """bottle=first_failure should embed target + minio_prefix in manifest."""
        import json as _json
        import tarfile
        from joshpy.jobs import run_sweep

        mock_cli = MagicMock()
        mock_cli._resolved_jar = Path("/fake/joshsim-fat.jar")
        mock_cli.java_path = "java"
        # Force a stage failure so the bottle records the failing job
        mock_cli.stage_to_minio.return_value = MagicMock(
            success=False, exit_code=1, stdout="",
            stderr="bucket denied", command=["stageToMinio"],
        )

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            job = self._make_real_job(tmp, run_hash="metaabc12345")
            job_set = MagicMock(
                total_jobs=1, total_replicates=1,
                __iter__=lambda s: iter([job]),
            )
            bottle_dir = tmp / "bottles"

            with self.assertRaises(Exception):
                run_sweep(
                    mock_cli, job_set,
                    batch_remote=True,
                    target="gke-test",
                    session_id="s-meta",
                    quiet=True,
                    stop_on_failure=True,
                    bottle="first_failure",
                    bottle_dir=bottle_dir,
                    bottle_omit_jshd=True,
                )

            # Locate the bottle archive and parse its manifest
            archives = list(bottle_dir.glob("bottle_*.tar.gz"))
            self.assertEqual(len(archives), 1, f"got {archives}")
            with tarfile.open(archives[0], "r:gz") as tar:
                manifest_member = next(
                    m for m in tar.getmembers()
                    if m.name.endswith("/manifest.json")
                )
                f = tar.extractfile(manifest_member)
                assert f is not None
                manifest = _json.loads(f.read().decode("utf-8"))

            self.assertIn("batch", manifest)
            self.assertEqual(manifest["batch"]["target"], "gke-test")
            self.assertIn(
                "metaabc12345", manifest["batch"]["minio_prefix"],
            )
            self.assertEqual(
                manifest["batch"]["stage_prefix_root"], "sweeps/s-meta/",
            )

    def _make_auto_ingest_mocks(self):
        """Build the mock surface run_sweep needs to reach the auto_ingest hook.

        Uses MagicMock for registry + a MagicMock export_paths whose resolve_path
        returns a fake non-existent Path (so _register_job_outputs's file-stat
        checks fall through cleanly).
        """
        from joshpy.cli import CLIResult

        mock_cli = MagicMock()
        mock_cli._resolved_jar = Path("/fake/joshsim-fat.jar")
        mock_cli.java_path = "java"
        mock_cli.stage_to_minio.return_value = CLIResult(
            exit_code=0, stdout="", stderr="", command=["stageToMinio"],
        )
        mock_cli.batch_remote.return_value = CLIResult(
            exit_code=0, stdout="", stderr="", command=["batchRemote"],
        )

        export_info = MagicMock()
        export_info.protocol = "minio"
        export_info.path = "minio://bucket/e2e/output_{replicate}.csv"
        export_info.host = "bucket"
        mock_export_paths = MagicMock()
        mock_export_paths.export_files = {"patch": export_info}
        mock_export_paths.debug_files = {}
        # resolve_path returns a definitely-nonexistent local Path so the
        # _register_job_outputs file_size branch falls through.
        mock_export_paths.resolve_path.side_effect = (
            lambda t, **kw: Path("/tmp/_pr7_test_nonexistent_output.csv")
        )
        mock_cli.inspect_exports.return_value = mock_export_paths

        registry = MagicMock()
        registry.start_run.return_value = "run-id"
        # RegistryCallback calls registry methods; MagicMock absorbs them
        return mock_cli, mock_export_paths, registry

    def test_auto_ingest_calls_ingest_results(self):
        """Successful batch_remote jobs with auto_ingest=True should invoke
        ingest_results for each, passing the cached ExportPaths."""
        from unittest.mock import patch as _patch
        from joshpy.jobs import run_sweep

        mock_cli, mock_export_paths, registry = self._make_auto_ingest_mocks()

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            job = self._make_real_job(tmp, run_hash="ai0000000001")
            job_set = MagicMock(
                total_jobs=1, total_replicates=1,
                __iter__=lambda s: iter([job]),
            )

            with _patch("joshpy.sweep.ingest_results", return_value=100) as mock_ingest:
                run_sweep(
                    mock_cli, job_set,
                    registry=registry,
                    session_id="s-ai",
                    batch_remote=True,
                    target="gke-test",
                    auto_ingest=True,
                    quiet=True,
                    manage_status=False,
                )

            mock_ingest.assert_called_once()
            kwargs = mock_ingest.call_args.kwargs
            # Verify the cached ExportPaths was forwarded (avoids N+1 subprocess)
            self.assertIs(kwargs.get("export_paths"), mock_export_paths)

    def test_auto_ingest_false_skips_ingest(self):
        """auto_ingest=False should not invoke ingest_results."""
        from unittest.mock import patch as _patch
        from joshpy.jobs import run_sweep

        mock_cli, _mock_export_paths, registry = self._make_auto_ingest_mocks()

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            job = self._make_real_job(tmp, run_hash="ai0000000002")
            job_set = MagicMock(
                total_jobs=1, total_replicates=1,
                __iter__=lambda s: iter([job]),
            )

            with _patch("joshpy.sweep.ingest_results") as mock_ingest:
                run_sweep(
                    mock_cli, job_set,
                    registry=registry,
                    session_id="s-ai",
                    batch_remote=True,
                    target="gke-test",
                    auto_ingest=False,
                    quiet=True,
                    manage_status=False,
                )

            mock_ingest.assert_not_called()

    def test_auto_ingest_exception_doesnt_abort_sweep(self):
        """If ingest_results raises, the sweep still returns cleanly."""
        from unittest.mock import patch as _patch
        from joshpy.jobs import run_sweep

        mock_cli, _mock_export_paths, registry = self._make_auto_ingest_mocks()

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            job = self._make_real_job(tmp, run_hash="ai0000000003")
            job_set = MagicMock(
                total_jobs=1, total_replicates=1,
                __iter__=lambda s: iter([job]),
            )

            with _patch(
                "joshpy.sweep.ingest_results",
                side_effect=RuntimeError("simulated ingest failure"),
            ):
                result = run_sweep(
                    mock_cli, job_set,
                    registry=registry,
                    session_id="s-ai",
                    batch_remote=True,
                    target="gke-test",
                    auto_ingest=True,
                    quiet=True,
                    manage_status=False,
                )

            # Sweep should report success despite the ingest failure
            self.assertEqual(result.succeeded, 1)
            self.assertEqual(result.failed, 0)


if __name__ == '__main__':
    unittest.main()
