"""Unit tests for the jobs module."""

import tempfile
import unittest
from pathlib import Path

from joshpy.jobs import (
    SweepParameter,
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
)


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


class TestSweepParameter(unittest.TestCase):
    """Tests for SweepParameter class."""

    def test_basic_creation(self):
        """Basic parameter creation with list."""
        param = SweepParameter(name="survival", values=[85, 90, 95])
        self.assertEqual(param.name, "survival")
        self.assertEqual(param.values, [85, 90, 95])

    def test_range_expansion(self):
        """Parameters with range spec should expand."""
        param = SweepParameter(
            name="survival",
            values={"start": 80, "stop": 90, "step": 2}
        )
        self.assertEqual(param.values, [80, 82, 84, 86, 88])

    def test_to_dict_with_values(self):
        """to_dict should include values."""
        param = SweepParameter(name="x", values=[1, 2, 3])
        result = param.to_dict()
        self.assertEqual(result, {"name": "x", "values": [1, 2, 3]})

    def test_to_dict_with_range(self):
        """to_dict should preserve range spec."""
        param = SweepParameter(
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
        param = SweepParameter.from_dict(data)
        self.assertEqual(param.name, "x")
        self.assertEqual(param.values, [1, 2, 3])

    def test_from_dict_with_range(self):
        """from_dict should create parameter with range."""
        data = {"name": "x", "range": {"start": 0, "stop": 6, "step": 2}}
        param = SweepParameter.from_dict(data)
        self.assertEqual(param.name, "x")
        self.assertEqual(param.values, [0, 2, 4])


class TestSweepConfig(unittest.TestCase):
    """Tests for SweepConfig class."""

    def test_empty_expand(self):
        """Empty config should expand to single empty dict."""
        config = SweepConfig()
        result = config.expand()
        self.assertEqual(result, [{}])

    def test_single_param_expand(self):
        """Single parameter should expand to list of single-key dicts."""
        config = SweepConfig(parameters=[
            SweepParameter(name="x", values=[1, 2, 3])
        ])
        result = config.expand()
        self.assertEqual(result, [{"x": 1}, {"x": 2}, {"x": 3}])

    def test_multiple_param_expand(self):
        """Multiple parameters should produce cartesian product."""
        config = SweepConfig(parameters=[
            SweepParameter(name="a", values=[1, 2]),
            SweepParameter(name="b", values=["x", "y"]),
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
        config = SweepConfig(parameters=[
            SweepParameter(name="a", values=[1, 2, 3]),  # 3
            SweepParameter(name="b", values=["x", "y"]),  # 2
        ])
        self.assertEqual(len(config), 6)

    def test_empty_len(self):
        """Empty config should have length 1."""
        config = SweepConfig()
        self.assertEqual(len(config), 1)

    def test_to_dict(self):
        """to_dict should serialize parameters."""
        config = SweepConfig(parameters=[
            SweepParameter(name="x", values=[1, 2])
        ])
        result = config.to_dict()
        self.assertEqual(result["parameters"][0]["name"], "x")

    def test_from_dict(self):
        """from_dict should deserialize parameters."""
        data = {"parameters": [{"name": "x", "values": [1, 2]}]}
        config = SweepConfig.from_dict(data)
        self.assertEqual(len(config.parameters), 1)
        self.assertEqual(config.parameters[0].name, "x")


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
            sweep=SweepConfig(parameters=[
                SweepParameter(name="val", values=[1, 2, 3])
            ])
        )
        yaml_str = original.to_yaml()
        restored = JobConfig.from_yaml(yaml_str)

        self.assertEqual(restored.simulation, original.simulation)
        self.assertEqual(restored.replicates, original.replicates)
        self.assertEqual(restored.template_string, original.template_string)
        self.assertEqual(len(restored.sweep.parameters), 1)
        self.assertEqual(restored.sweep.parameters[0].values, [1, 2, 3])


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
            sweep=SweepConfig(parameters=[
                SweepParameter(name="x", values=[1, 2, 3])
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
            sweep=SweepConfig(parameters=[
                SweepParameter(name="val", values=[10, 20])
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
            sweep=SweepConfig(parameters=[
                SweepParameter(name="a", values=[1]),
                SweepParameter(name="b", values=["test"]),
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

    def test_expand_requires_template(self):
        """Expanding without template should raise ValueError."""
        config = JobConfig()
        expander = JobExpander()

        with self.assertRaises(ValueError):
            expander.expand(config)


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

        result = run_sweep(cli, job_set, callback=callback, quiet=True)

        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(callback.call_count, 1)

    def test_stop_on_failure(self):
        """Should stop on first failure when stop_on_failure=True."""
        from unittest.mock import MagicMock

        cli = MagicMock()
        # First call succeeds, second fails
        cli.run.side_effect = [
            MagicMock(success=True, exit_code=0),
            MagicMock(success=False, exit_code=1),
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

        result = run_sweep(cli, job_set, stop_on_failure=True, quiet=True)

        # Should stop after the second job (which failed)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 1)


if __name__ == '__main__':
    unittest.main()
