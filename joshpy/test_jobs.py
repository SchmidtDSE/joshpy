"""Unit tests for the jobs module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from joshpy.jobs import (
    SweepParameter,
    SweepConfig,
    JobConfig,
    ExpandedJob,
    JobSet,
    JobExpander,
    JobRunner,
    JobResult,
    compute_config_hash,
    _normalize_values,
    run_sweep,
)


class TestComputeConfigHash(unittest.TestCase):
    """Tests for compute_config_hash function."""

    def test_returns_12_char_string(self):
        """Hash should be 12 characters."""
        result = compute_config_hash("test content")
        self.assertEqual(len(result), 12)
        self.assertTrue(all(c in '0123456789abcdef' for c in result))

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        content = "survivalProbAdult = 85 %"
        hash1 = compute_config_hash(content)
        hash2 = compute_config_hash(content)
        self.assertEqual(hash1, hash2)

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        hash1 = compute_config_hash("survivalProbAdult = 85 %")
        hash2 = compute_config_hash("survivalProbAdult = 90 %")
        self.assertNotEqual(hash1, hash2)


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
        self.assertIn("config_hash", job.custom_tags)

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
                config_hash="abc",
                parameters={},
                simulation="Main",
                replicates=1,
            )
        ])
        self.assertEqual(len(job_set), 1)

    def test_iteration(self):
        """Should be iterable over jobs."""
        jobs = [
            ExpandedJob(
                config_content=f"val={i}",
                config_path=Path(f"/tmp/test{i}"),
                config_name="test",
                config_hash=f"hash{i}",
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


class TestJobRunner(unittest.TestCase):
    """Tests for JobRunner class."""

    # Use the local jar that exists in the repo
    JAR_PATH = Path(__file__).parent.parent / "jar" / "joshsim-fat.jar"

    def test_build_command_basic(self):
        """Build command should include basic options."""
        runner = JobRunner(josh_jar=self.JAR_PATH)
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            config_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            source_path=Path("/path/to/source.josh"),
        )

        cmd = runner.build_command(job)

        self.assertIn("java", cmd)
        self.assertIn("-jar", cmd)
        self.assertIn("run", cmd)
        self.assertIn("Main", cmd)
        # Source path should be resolved to absolute
        self.assertTrue(any("source.josh" in c for c in cmd))

    def test_build_command_with_replicates(self):
        """Build command should include replicates when > 1."""
        runner = JobRunner(josh_jar=self.JAR_PATH)
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            config_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=5,
        )

        cmd = runner.build_command(job)

        replicates_idx = cmd.index("--replicates")
        self.assertEqual(cmd[replicates_idx + 1], "5")

    def test_build_command_with_custom_tags(self):
        """Build command should include custom tags."""
        runner = JobRunner(josh_jar=self.JAR_PATH)
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            config_hash="abc123",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
            custom_tags={"x": "1", "y": "test"},
        )

        cmd = runner.build_command(job)

        # Check custom tags are included
        self.assertIn("--custom-tag", cmd)
        tag_indices = [i for i, c in enumerate(cmd) if c == "--custom-tag"]
        tags = [cmd[i + 1] for i in tag_indices]
        self.assertIn("x=1", tags)
        self.assertIn("y=test", tags)

    def test_build_command_with_options(self):
        """Build command should include all options."""
        runner = JobRunner(josh_jar=self.JAR_PATH)
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            config_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
            seed=42,
            crs="EPSG:4326",
            use_float64=True,
            output_steps="0-10",
        )

        cmd = runner.build_command(job)

        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)
        self.assertIn("--crs", cmd)
        self.assertIn("EPSG:4326", cmd)
        self.assertIn("--use-float-64", cmd)
        self.assertIn("--output-steps", cmd)
        self.assertIn("0-10", cmd)

    @patch('subprocess.run')
    def test_run_success(self, mock_run):
        """Run should return success result on exit code 0."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="output",
            stderr=""
        )

        runner = JobRunner(josh_jar=self.JAR_PATH)
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            config_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
        )

        result = runner.run(job)

        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "output")

    @patch('subprocess.run')
    def test_run_failure(self, mock_run):
        """Run should return failure result on non-zero exit code."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error message"
        )

        runner = JobRunner(josh_jar=self.JAR_PATH)
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/editor.jshc"),
            config_name="editor",
            config_hash="abc123",
            parameters={},
            simulation="Main",
            replicates=1,
        )

        result = runner.run(job)

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stderr, "error message")


class TestJobResult(unittest.TestCase):
    """Tests for JobResult class."""

    def test_success_property(self):
        """success should be True for exit_code 0."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/test"),
            config_name="test",
            config_hash="abc",
            parameters={},
            simulation="Main",
            replicates=1,
        )
        result = JobResult(
            job=job,
            exit_code=0,
            stdout="",
            stderr="",
            command=["java", "-jar", "test.jar"],
        )
        self.assertTrue(result.success)

    def test_failure_property(self):
        """success should be False for non-zero exit_code."""
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/test"),
            config_name="test",
            config_hash="abc",
            parameters={},
            simulation="Main",
            replicates=1,
        )
        result = JobResult(
            job=job,
            exit_code=1,
            stdout="",
            stderr="",
            command=["java", "-jar", "test.jar"],
        )
        self.assertFalse(result.success)


if __name__ == '__main__':
    unittest.main()
