"""Unit tests for the strategies module."""

import unittest
from pathlib import Path

from joshpy.strategies import (
    SweepStrategy,
    CartesianStrategy,
    OptunaStrategy,
    strategy_from_dict,
    sample_params_from_trial,
    _load_objective_from_path,
)
from joshpy.jobs import ConfigSweepParameter, FileSweepParameter


class TestCartesianStrategy(unittest.TestCase):
    """Tests for CartesianStrategy."""

    def test_is_not_adaptive(self):
        """Cartesian strategy should not be adaptive."""
        strategy = CartesianStrategy()
        self.assertFalse(strategy.is_adaptive)

    def test_expand_empty(self):
        """Empty params should yield single empty dict."""
        strategy = CartesianStrategy()
        result = list(strategy.expand([], []))
        self.assertEqual(result, [{}])

    def test_expand_single_config_param(self):
        """Single config param should yield one combo per value."""
        strategy = CartesianStrategy()
        params = [ConfigSweepParameter(name="x", values=[1, 2, 3])]
        result = list(strategy.expand(params, []))
        self.assertEqual(result, [{"x": 1}, {"x": 2}, {"x": 3}])

    def test_expand_multiple_config_params(self):
        """Multiple config params should yield cartesian product."""
        strategy = CartesianStrategy()
        params = [
            ConfigSweepParameter(name="a", values=[1, 2]),
            ConfigSweepParameter(name="b", values=[10, 20]),
        ]
        result = list(strategy.expand(params, []))
        expected = [
            {"a": 1, "b": 10},
            {"a": 1, "b": 20},
            {"a": 2, "b": 10},
            {"a": 2, "b": 20},
        ]
        self.assertEqual(result, expected)

    def test_expand_file_params(self):
        """File params should yield path/label dicts."""
        strategy = CartesianStrategy()
        file_params = [
            FileSweepParameter(name="climate", paths=[Path("a.jshd"), Path("b.jshd")])
        ]
        result = list(strategy.expand([], file_params))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["climate"]["path"], Path("a.jshd"))
        self.assertEqual(result[0]["climate"]["label"], "a")
        self.assertEqual(result[1]["climate"]["path"], Path("b.jshd"))
        self.assertEqual(result[1]["climate"]["label"], "b")

    def test_expand_mixed_params(self):
        """Mixed config and file params should produce cartesian product."""
        strategy = CartesianStrategy()
        config_params = [ConfigSweepParameter(name="x", values=[1, 2])]
        file_params = [
            FileSweepParameter(name="climate", paths=[Path("a.jshd"), Path("b.jshd")])
        ]
        result = list(strategy.expand(config_params, file_params))
        self.assertEqual(len(result), 4)  # 2 x 2

    def test_to_dict(self):
        """Should serialize to dict with type."""
        strategy = CartesianStrategy()
        result = strategy.to_dict()
        self.assertEqual(result, {"type": "cartesian"})

    def test_from_dict(self):
        """Should deserialize from dict."""
        strategy = CartesianStrategy.from_dict({})
        self.assertIsInstance(strategy, CartesianStrategy)


class TestOptunaStrategy(unittest.TestCase):
    """Tests for OptunaStrategy."""

    def test_is_adaptive(self):
        """Optuna strategy should be adaptive."""
        strategy = OptunaStrategy()
        self.assertTrue(strategy.is_adaptive)

    def test_expand_raises(self):
        """expand() should raise RuntimeError for adaptive strategies."""
        strategy = OptunaStrategy()
        with self.assertRaises(RuntimeError) as ctx:
            list(strategy.expand([], []))
        self.assertIn("adaptive", str(ctx.exception).lower())

    def test_default_values(self):
        """Default values should be set."""
        strategy = OptunaStrategy()
        self.assertEqual(strategy.n_trials, 100)
        self.assertEqual(strategy.direction, "minimize")
        self.assertEqual(strategy.sampler, "tpe")
        self.assertIsNone(strategy.objective)

    def test_custom_values(self):
        """Custom values should be set."""
        strategy = OptunaStrategy(
            n_trials=50,
            direction="maximize",
            sampler="random",
        )
        self.assertEqual(strategy.n_trials, 50)
        self.assertEqual(strategy.direction, "maximize")
        self.assertEqual(strategy.sampler, "random")

    def test_invalid_direction(self):
        """Invalid direction should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            OptunaStrategy(direction="invalid")
        self.assertIn("direction", str(ctx.exception))

    def test_invalid_sampler(self):
        """Invalid sampler should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            OptunaStrategy(sampler="invalid")
        self.assertIn("sampler", str(ctx.exception))

    def test_get_objective_raises_if_none(self):
        """get_objective() should raise if no objective configured."""
        strategy = OptunaStrategy()
        with self.assertRaises(ValueError) as ctx:
            strategy.get_objective()
        self.assertIn("objective", str(ctx.exception).lower())

    def test_get_objective_with_callable(self):
        """get_objective() should return callable objective."""
        def my_obj(reg, h, j):
            return 0.0

        strategy = OptunaStrategy(objective=my_obj)
        self.assertIs(strategy.get_objective(), my_obj)

    def test_to_dict_basic(self):
        """Should serialize basic fields to dict."""
        strategy = OptunaStrategy(n_trials=50, direction="maximize", sampler="random")
        result = strategy.to_dict()
        self.assertEqual(result["type"], "optuna")
        self.assertEqual(result["n_trials"], 50)
        self.assertEqual(result["direction"], "maximize")
        self.assertEqual(result["sampler"], "random")

    def test_to_dict_with_string_objective(self):
        """Should include string objective in serialization."""
        # String objectives are stored but not resolved until get_objective() is called
        strategy = OptunaStrategy(objective="my_module:my_func")
        result = strategy.to_dict()
        self.assertEqual(result["objective"], "my_module:my_func")
        
    def test_string_objective_resolved_lazily(self):
        """String objective should not be resolved until get_objective() is called."""
        # This should NOT raise - module doesn't exist but we're not calling get_objective()
        strategy = OptunaStrategy(objective="nonexistent_module:func")
        # But calling get_objective should raise
        with self.assertRaises(ImportError):
            strategy.get_objective()

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "n_trials": 50,
            "direction": "maximize",
            "sampler": "random",
        }
        strategy = OptunaStrategy.from_dict(data)
        self.assertEqual(strategy.n_trials, 50)
        self.assertEqual(strategy.direction, "maximize")
        self.assertEqual(strategy.sampler, "random")


class TestStrategyFromDict(unittest.TestCase):
    """Tests for strategy_from_dict factory."""

    def test_cartesian(self):
        """Should create CartesianStrategy."""
        strategy = strategy_from_dict({"type": "cartesian"})
        self.assertIsInstance(strategy, CartesianStrategy)

    def test_optuna(self):
        """Should create OptunaStrategy."""
        strategy = strategy_from_dict({"type": "optuna", "n_trials": 50})
        self.assertIsInstance(strategy, OptunaStrategy)
        self.assertEqual(strategy.n_trials, 50)

    def test_default_is_cartesian(self):
        """No type should default to cartesian."""
        strategy = strategy_from_dict({})
        self.assertIsInstance(strategy, CartesianStrategy)

    def test_unknown_type_raises(self):
        """Unknown type should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            strategy_from_dict({"type": "unknown"})
        self.assertIn("unknown", str(ctx.exception).lower())


class TestLoadObjectiveFromPath(unittest.TestCase):
    """Tests for _load_objective_from_path."""

    def test_invalid_format_raises(self):
        """Should raise for invalid path format."""
        with self.assertRaises(ValueError) as ctx:
            _load_objective_from_path("no_colon_here")
        self.assertIn("module:function", str(ctx.exception))

    def test_loads_builtin(self):
        """Should load a builtin function."""
        fn = _load_objective_from_path("builtins:len")
        self.assertEqual(fn([1, 2, 3]), 3)

    def test_import_error_propagates(self):
        """ImportError should propagate for non-existent module."""
        with self.assertRaises(ImportError):
            _load_objective_from_path("nonexistent_module_xyz:func")


class TestYamlRoundTrip(unittest.TestCase):
    """Tests for YAML serialization round-trips."""

    def test_cartesian_strategy_roundtrip(self):
        """CartesianStrategy should round-trip through dict."""
        original = CartesianStrategy()
        data = original.to_dict()
        restored = strategy_from_dict(data)
        self.assertIsInstance(restored, CartesianStrategy)

    def test_optuna_strategy_roundtrip(self):
        """OptunaStrategy should round-trip through dict."""
        original = OptunaStrategy(n_trials=75, direction="maximize", sampler="cmaes")
        data = original.to_dict()
        restored = strategy_from_dict(data)
        self.assertIsInstance(restored, OptunaStrategy)
        self.assertEqual(restored.n_trials, 75)
        self.assertEqual(restored.direction, "maximize")
        self.assertEqual(restored.sampler, "cmaes")


class TestSweepConfigWithStrategy(unittest.TestCase):
    """Tests for SweepConfig integration with strategies."""

    def test_default_strategy_is_cartesian(self):
        """SweepConfig should default to CartesianStrategy."""
        from joshpy.jobs import SweepConfig

        config = SweepConfig(
            config_parameters=[ConfigSweepParameter(name="x", values=[1, 2, 3])]
        )
        self.assertIsInstance(config.strategy, CartesianStrategy)

    def test_explicit_strategy(self):
        """SweepConfig should use explicit strategy."""
        from joshpy.jobs import SweepConfig

        strategy = OptunaStrategy(n_trials=50)
        config = SweepConfig(
            config_parameters=[ConfigSweepParameter(name="x", values=[1, 2, 3])],
            strategy=strategy,
        )
        self.assertIs(config.strategy, strategy)

    def test_expand_delegates_to_strategy(self):
        """SweepConfig.expand() should delegate to strategy."""
        from joshpy.jobs import SweepConfig

        config = SweepConfig(
            config_parameters=[ConfigSweepParameter(name="x", values=[1, 2])]
        )
        result = config.expand()
        self.assertEqual(result, [{"x": 1}, {"x": 2}])

    def test_len_raises_for_adaptive(self):
        """len(SweepConfig) should raise for adaptive strategies."""
        from joshpy.jobs import SweepConfig

        config = SweepConfig(
            config_parameters=[ConfigSweepParameter(name="x", values=[1, 2, 3])],
            strategy=OptunaStrategy(n_trials=50),
        )
        with self.assertRaises(ValueError) as ctx:
            len(config)
        self.assertIn("adaptive", str(ctx.exception).lower())

    def test_to_dict_includes_strategy(self):
        """SweepConfig.to_dict() should include non-default strategies."""
        from joshpy.jobs import SweepConfig

        config = SweepConfig(
            config_parameters=[ConfigSweepParameter(name="x", values=[1, 2, 3])],
            strategy=OptunaStrategy(n_trials=50),
        )
        data = config.to_dict()
        self.assertIn("strategy", data)
        self.assertEqual(data["strategy"]["type"], "optuna")
        self.assertEqual(data["strategy"]["n_trials"], 50)

    def test_to_dict_omits_default_strategy(self):
        """SweepConfig.to_dict() should omit default CartesianStrategy."""
        from joshpy.jobs import SweepConfig

        config = SweepConfig(
            config_parameters=[ConfigSweepParameter(name="x", values=[1, 2, 3])]
        )
        data = config.to_dict()
        self.assertNotIn("strategy", data)

    def test_from_dict_with_strategy(self):
        """SweepConfig.from_dict() should restore strategy."""
        from joshpy.jobs import SweepConfig

        data = {
            "config_parameters": [{"name": "x", "values": [1, 2, 3]}],
            "strategy": {"type": "optuna", "n_trials": 50},
        }
        config = SweepConfig.from_dict(data)
        self.assertIsInstance(config.strategy, OptunaStrategy)
        self.assertEqual(config.strategy.n_trials, 50)

    def test_from_dict_without_strategy(self):
        """SweepConfig.from_dict() should default to CartesianStrategy."""
        from joshpy.jobs import SweepConfig

        data = {
            "config_parameters": [{"name": "x", "values": [1, 2, 3]}],
        }
        config = SweepConfig.from_dict(data)
        self.assertIsInstance(config.strategy, CartesianStrategy)


class TestAdaptiveSweepResult(unittest.TestCase):
    """Tests for AdaptiveSweepResult."""

    def test_is_adaptive_property(self):
        """AdaptiveSweepResult.is_adaptive should be True."""
        from joshpy.jobs import AdaptiveSweepResult

        result = AdaptiveSweepResult()
        self.assertTrue(result.is_adaptive)

    def test_total_trials_from_study(self):
        """total_trials should return study trial count when available."""
        from joshpy.jobs import AdaptiveSweepResult
        from unittest.mock import MagicMock

        mock_study = MagicMock()
        mock_study.trials = [MagicMock(), MagicMock(), MagicMock()]

        result = AdaptiveSweepResult(study=mock_study)
        self.assertEqual(result.total_trials, 3)

    def test_total_trials_from_job_results(self):
        """total_trials should fall back to job_results length when no study."""
        from joshpy.jobs import AdaptiveSweepResult
        from unittest.mock import MagicMock

        mock_jobs = [(MagicMock(), MagicMock()) for _ in range(5)]
        result = AdaptiveSweepResult(job_results=mock_jobs)
        self.assertEqual(result.total_trials, 5)

    def test_trial_metrics_filters_inf(self):
        """trial_metrics should exclude inf values."""
        from joshpy.jobs import AdaptiveSweepResult
        from unittest.mock import MagicMock

        mock_trials = []
        for value in [1.0, 2.0, float("inf"), 3.0, None]:
            t = MagicMock()
            t.value = value
            mock_trials.append(t)

        mock_study = MagicMock()
        mock_study.trials = mock_trials

        result = AdaptiveSweepResult(study=mock_study)
        self.assertEqual(result.trial_metrics, [1.0, 2.0, 3.0])

    def test_trial_metrics_empty_without_study(self):
        """trial_metrics should return empty list without study."""
        from joshpy.jobs import AdaptiveSweepResult

        result = AdaptiveSweepResult()
        self.assertEqual(result.trial_metrics, [])

    def test_get_trial_summary(self):
        """get_trial_summary should return dict with statistics."""
        from joshpy.jobs import AdaptiveSweepResult
        from unittest.mock import MagicMock

        mock_trials = []
        for value in [1.0, 2.0, 3.0]:
            t = MagicMock()
            t.value = value
            mock_trials.append(t)

        mock_study = MagicMock()
        mock_study.trials = mock_trials

        result = AdaptiveSweepResult(
            study=mock_study, best_value=1.0, best_params={"x": 10}
        )
        summary = result.get_trial_summary()

        self.assertEqual(summary["n_trials"], 3)
        self.assertEqual(summary["n_completed"], 3)
        self.assertEqual(summary["n_failed"], 0)
        self.assertEqual(summary["best_value"], 1.0)
        self.assertEqual(summary["best_params"], {"x": 10})
        self.assertAlmostEqual(summary["mean_value"], 2.0)
        self.assertIsNotNone(summary["std_value"])

    def test_get_trial_summary_empty_without_study(self):
        """get_trial_summary should return empty dict without study."""
        from joshpy.jobs import AdaptiveSweepResult

        result = AdaptiveSweepResult()
        self.assertEqual(result.get_trial_summary(), {})

    def test_get_best_job(self):
        """get_best_job should return job with best run_hash."""
        from joshpy.jobs import AdaptiveSweepResult, ExpandedJob
        from unittest.mock import MagicMock
        from pathlib import Path

        # Create mock jobs
        job1 = ExpandedJob(
            config_content="",
            config_path=Path("/tmp/a"),
            config_name="sweep_config",
            run_hash="hash1",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
        )
        job2 = ExpandedJob(
            config_content="",
            config_path=Path("/tmp/b"),
            config_name="sweep_config",
            run_hash="hash2",
            parameters={"x": 2},
            simulation="Main",
            replicates=1,
        )

        # Create mock study with best trial
        mock_trial = MagicMock()
        mock_trial.user_attrs = {"run_hash": "hash2"}
        mock_study = MagicMock()
        mock_study.best_trial = mock_trial

        result = AdaptiveSweepResult(
            job_results=[(job1, MagicMock()), (job2, MagicMock())], study=mock_study
        )

        best_job = result.get_best_job()
        self.assertEqual(best_job, job2)
        self.assertEqual(best_job.parameters, {"x": 2})

    def test_get_best_job_returns_none_without_study(self):
        """get_best_job should return None without study."""
        from joshpy.jobs import AdaptiveSweepResult

        result = AdaptiveSweepResult()
        self.assertIsNone(result.get_best_job())

    def test_iteration(self):
        """AdaptiveSweepResult should be iterable."""
        from joshpy.jobs import AdaptiveSweepResult
        from unittest.mock import MagicMock

        mock_jobs = [(MagicMock(), MagicMock()) for _ in range(3)]
        result = AdaptiveSweepResult(job_results=mock_jobs)

        items = list(result)
        self.assertEqual(len(items), 3)

    def test_len(self):
        """len(AdaptiveSweepResult) should return job count."""
        from joshpy.jobs import AdaptiveSweepResult
        from unittest.mock import MagicMock

        mock_jobs = [(MagicMock(), MagicMock()) for _ in range(4)]
        result = AdaptiveSweepResult(job_results=mock_jobs)

        self.assertEqual(len(result), 4)


def _has_optuna() -> bool:
    """Check if optuna is available."""
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_optuna(), "optuna not installed")
class TestRunAdaptiveSweepValidation(unittest.TestCase):
    """Tests for run_adaptive_sweep input validation."""

    def test_requires_sweep_config(self):
        """Should raise ValueError if config.sweep is None."""
        from joshpy.strategies import run_adaptive_sweep
        from joshpy.jobs import JobConfig
        from unittest.mock import MagicMock

        config = JobConfig(template_string="test", sweep=None)
        mock_cli = MagicMock()
        mock_registry = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            run_adaptive_sweep(
                mock_cli,
                config,
                registry=mock_registry,
                session_id="test",
            )
        self.assertIn("config.sweep is required", str(ctx.exception))

    def test_requires_optuna_strategy(self):
        """Should raise TypeError if strategy is not OptunaStrategy."""
        from joshpy.strategies import run_adaptive_sweep
        from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter
        from unittest.mock import MagicMock

        config = JobConfig(
            template_string="test",
            sweep=SweepConfig(
                config_parameters=[ConfigSweepParameter(name="x", values=[1, 2])]
                # Default CartesianStrategy
            ),
        )
        mock_cli = MagicMock()
        mock_registry = MagicMock()

        with self.assertRaises(TypeError) as ctx:
            run_adaptive_sweep(
                mock_cli,
                config,
                registry=mock_registry,
                session_id="test",
            )
        self.assertIn("requires OptunaStrategy", str(ctx.exception))


class TestCreateSingleJob(unittest.TestCase):
    """Tests for _create_single_job helper."""

    def test_creates_job_with_config_params(self):
        """Should create ExpandedJob with config parameters."""
        from joshpy.strategies import _create_single_job
        from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter
        from pathlib import Path
        import tempfile
        import os

        # Create temp josh file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".josh", delete=False) as f:
            f.write("start simulation Main\nend simulation\n")
            josh_path = Path(f.name)

        try:
            config = JobConfig(
                template_string="maxGrowth = {{ x }} meters",
                source_path=josh_path,
                simulation="Main",
                replicates=3,
                sweep=SweepConfig(
                    config_parameters=[ConfigSweepParameter(name="x", values=[1, 2, 3])],
                    strategy=OptunaStrategy(n_trials=10, objective="math:sin"),
                ),
            )

            params = {"x": 42}
            job = _create_single_job(config, params, trial_num=5)

            self.assertEqual(job.parameters, {"x": 42})
            self.assertEqual(job.simulation, "Main")
            self.assertEqual(job.replicates, 3)
            self.assertIn("maxGrowth = 42 meters", job.config_content)
            self.assertIn("x", job.custom_tags)
            self.assertEqual(job.custom_tags["x"], "42")
            self.assertIsNotNone(job.run_hash)
            self.assertTrue(job.config_path.exists())
        finally:
            os.unlink(josh_path)

    def test_creates_job_with_file_params(self):
        """Should create ExpandedJob with file parameters in file_mappings."""
        from joshpy.strategies import _create_single_job
        from joshpy.jobs import JobConfig, SweepConfig, FileSweepParameter
        from pathlib import Path
        import tempfile
        import os

        # Create temp josh file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".josh", delete=False) as f:
            f.write("start simulation Main\nend simulation\n")
            josh_path = Path(f.name)

        # Create temp data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jshd", delete=False) as f:
            f.write("test data")
            data_path = Path(f.name)

        try:
            config = JobConfig(
                template_string="# config",
                source_path=josh_path,
                simulation="Main",
                sweep=SweepConfig(
                    file_parameters=[
                        FileSweepParameter(name="climate", paths=[str(data_path)])
                    ],
                    strategy=OptunaStrategy(n_trials=10, objective="math:sin"),
                ),
            )

            params = {"climate": {"path": data_path, "label": data_path.stem}}
            job = _create_single_job(config, params, trial_num=0)

            self.assertIn("climate", job.file_mappings)
            self.assertEqual(job.file_mappings["climate"], data_path)
            self.assertIn("climate", job.custom_tags)
            self.assertEqual(job.custom_tags["climate"], data_path.stem)
        finally:
            os.unlink(josh_path)
            os.unlink(data_path)


class TestSweepExecutionError(unittest.TestCase):
    """Tests for SweepExecutionError exception."""

    def test_error_message_includes_exit_code(self):
        """Error message should include exit code."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10, "y": 20}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 1
        result.stderr = None

        error = SweepExecutionError(
            job=job, result=result, trial_num=5, succeeded_before=3
        )

        self.assertIn("exit_code=1", str(error))
        self.assertIn("Trial 6", str(error))  # trial_num is 0-indexed
        self.assertIn("3 trial(s) succeeded", str(error))
        self.assertIn("abc123", str(error))

    def test_error_message_includes_stderr(self):
        """Error message should include stderr when present."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 1
        result.stderr = "Error: Configuration not found\nStack trace..."

        error = SweepExecutionError(
            job=job, result=result, trial_num=0, succeeded_before=0
        )

        self.assertIn("STDERR:", str(error))
        self.assertIn("Configuration not found", str(error))

    def test_error_attributes_accessible(self):
        """Error should expose job, result, and counts as attributes."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 42
        result.stderr = "some error"

        error = SweepExecutionError(
            job=job, result=result, trial_num=7, succeeded_before=5
        )

        self.assertIs(error.job, job)
        self.assertIs(error.result, result)
        self.assertEqual(error.trial_num, 7)
        self.assertEqual(error.succeeded_before, 5)


class TestStopOnFailureBehavior(unittest.TestCase):
    """Tests for stop_on_failure parameter in run_adaptive_sweep."""

    def test_stop_on_failure_true_raises_on_cli_failure(self):
        """Should raise SweepExecutionError when CLI returns non-zero."""
        from joshpy.strategies import (
            run_adaptive_sweep,
            OptunaStrategy,
            SweepExecutionError,
        )
        from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter
        from unittest.mock import MagicMock, patch
        from pathlib import Path
        import tempfile
        import os

        # Create temp josh file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".josh", delete=False) as f:
            f.write("start simulation Main\nend simulation\n")
            josh_path = Path(f.name)

        try:
            config = JobConfig(
                template_string="x = {{ x }} meters",
                source_path=josh_path,
                simulation="Main",
                replicates=1,
                sweep=SweepConfig(
                    config_parameters=[ConfigSweepParameter(name="x", values=[1, 2])],
                    strategy=OptunaStrategy(n_trials=2, direction="minimize"),
                ),
            )

            # Mock CLI to fail
            mock_cli = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.exit_code = 1
            mock_result.stderr = "Test error message"
            mock_cli.run.return_value = mock_result
            mock_cli.inspect_exports.return_value = MagicMock(patch=None)

            # Mock registry
            mock_registry = MagicMock()
            mock_registry.get_runs_for_hash.return_value = []

            with self.assertRaises(SweepExecutionError) as ctx:
                run_adaptive_sweep(
                    mock_cli,
                    config,
                    registry=mock_registry,
                    session_id="test",
                    objective=lambda r, h, j: 0.0,
                    stop_on_failure=True,
                    quiet=True,
                )

            error = ctx.exception
            self.assertEqual(error.result.exit_code, 1)
            self.assertIn("Test error message", str(error))

        finally:
            os.unlink(josh_path)

    def test_stop_on_failure_false_continues_on_cli_failure(self):
        """Should continue and return results when stop_on_failure=False."""
        from joshpy.strategies import run_adaptive_sweep, OptunaStrategy
        from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter
        from unittest.mock import MagicMock
        from pathlib import Path
        import tempfile
        import os

        # Create temp josh file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".josh", delete=False) as f:
            f.write("start simulation Main\nend simulation\n")
            josh_path = Path(f.name)

        try:
            config = JobConfig(
                template_string="x = {{ x }} meters",
                source_path=josh_path,
                simulation="Main",
                replicates=1,
                sweep=SweepConfig(
                    config_parameters=[ConfigSweepParameter(name="x", values=[1, 2])],
                    strategy=OptunaStrategy(n_trials=3, direction="minimize"),
                ),
            )

            # Mock CLI to fail
            mock_cli = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.exit_code = 1
            mock_result.stderr = "Test error"
            mock_cli.run.return_value = mock_result
            mock_cli.inspect_exports.return_value = MagicMock(patch=None)

            # Mock registry - need to return proper values for _store_study_outcomes
            mock_registry = MagicMock()
            mock_registry.get_runs_for_hash.return_value = []
            # Return a mock session with None metadata (so it doesn't try to serialize MagicMock)
            mock_session = MagicMock()
            mock_session.metadata = None
            mock_registry.get_session.return_value = mock_session

            # Should NOT raise - just return results with failures
            result = run_adaptive_sweep(
                mock_cli,
                config,
                registry=mock_registry,
                session_id="test",
                objective=lambda r, h, j: 0.0,
                stop_on_failure=False,
                quiet=True,
            )

            # All 3 trials should have run (and failed)
            self.assertEqual(result.failed, 3)
            self.assertEqual(result.succeeded, 0)
            self.assertEqual(len(result.job_results), 3)

        finally:
            os.unlink(josh_path)


class TestCvObjective(unittest.TestCase):
    """Tests for cv_objective built-in."""

    def test_returns_callable(self):
        """cv_objective should return a callable."""
        from joshpy.strategies import cv_objective

        obj = cv_objective("totalCover", burn_in=50)
        self.assertTrue(callable(obj))

    def test_computes_mean_cv_across_replicates(self):
        """Should return mean CV from get_replicate_cv."""
        from joshpy.strategies import cv_objective
        from unittest.mock import MagicMock, patch

        mock_registry = MagicMock()
        mock_job = MagicMock()

        # Mock get_replicate_cv to return a known mean_cv
        mock_result = {
            "mean_cv": 0.15,
            "replicate_cvs": [0.1, 0.15, 0.2],
            "n_replicates": 3,
            "n_timesteps": 50,
            "extinct_replicates": [],
        }

        with patch('joshpy.cell_data.DiagnosticQueries') as mock_queries_cls:
            mock_queries = MagicMock()
            mock_queries.get_replicate_cv.return_value = mock_result
            mock_queries_cls.return_value = mock_queries

            obj = cv_objective("totalCover", burn_in=50)
            result = obj(mock_registry, "test_hash", mock_job)

            # Should return the mean_cv from get_replicate_cv
            self.assertAlmostEqual(result, 0.15, places=5)

            # Verify get_replicate_cv was called with correct args
            mock_queries.get_replicate_cv.assert_called_once_with(
                variable="totalCover",
                run_hash="test_hash",
                burn_in=50,
                extinction_threshold=0.01,
            )

    def test_returns_inf_for_extinction(self):
        """Should return inf when any replicate goes extinct."""
        from joshpy.strategies import cv_objective
        from unittest.mock import MagicMock, patch

        mock_registry = MagicMock()
        mock_job = MagicMock()

        # Mock get_replicate_cv returning inf due to extinction
        mock_result = {
            "mean_cv": float("inf"),
            "replicate_cvs": [0.1, float("inf"), 0.2],
            "n_replicates": 3,
            "n_timesteps": 50,
            "extinct_replicates": [1],
        }

        with patch('joshpy.cell_data.DiagnosticQueries') as mock_queries_cls:
            mock_queries = MagicMock()
            mock_queries.get_replicate_cv.return_value = mock_result
            mock_queries_cls.return_value = mock_queries

            obj = cv_objective("totalCover", burn_in=0)
            result = obj(mock_registry, "test_hash", mock_job)

            self.assertEqual(result, float('inf'))

    def test_passes_extinction_threshold(self):
        """Should pass custom extinction_threshold to get_replicate_cv."""
        from joshpy.strategies import cv_objective
        from unittest.mock import MagicMock, patch

        mock_registry = MagicMock()
        mock_job = MagicMock()

        mock_result = {
            "mean_cv": 0.2,
            "replicate_cvs": [0.2],
            "n_replicates": 1,
            "n_timesteps": 50,
            "extinct_replicates": [],
        }

        with patch('joshpy.cell_data.DiagnosticQueries') as mock_queries_cls:
            mock_queries = MagicMock()
            mock_queries.get_replicate_cv.return_value = mock_result
            mock_queries_cls.return_value = mock_queries

            obj = cv_objective("totalCover", burn_in=10, extinction_threshold=0.05)
            obj(mock_registry, "test_hash", mock_job)

            # Verify custom extinction_threshold was passed
            mock_queries.get_replicate_cv.assert_called_once_with(
                variable="totalCover",
                run_hash="test_hash",
                burn_in=10,
                extinction_threshold=0.05,
            )


if __name__ == "__main__":
    unittest.main()
