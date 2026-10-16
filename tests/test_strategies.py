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


def _cli_result(exit_code=0, stderr=""):
    from joshpy.cli import CLIResult

    return CLIResult(exit_code=exit_code, stdout="", stderr=stderr, command=[])


class _AdaptiveContextCase(unittest.TestCase):
    """Fixture: a tiny model, an in-memory registry and a fake CLI."""

    def setUp(self):
        import tempfile
        from unittest.mock import MagicMock

        from joshpy.registry import RunRegistry

        self.tmp = Path(tempfile.mkdtemp())
        self.josh_path = self.tmp / "model.josh"
        self.josh_path.write_text("start simulation Main\nend simulation\n")
        self.registry = RunRegistry(":memory:")
        self.cli = MagicMock()
        self.cli.run.return_value = _cli_result()
        self.contexts = []

    def tearDown(self):
        import shutil

        for ctx in self.contexts:
            ctx.cleanup()
        self.registry.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_config(self, **overrides):
        from joshpy.jobs import JobConfig, SweepConfig

        kwargs = dict(
            template_string="maxGrowth = {{ x }} meters",
            source_path=self.josh_path,
            simulation="Main",
            sweep=SweepConfig(
                config_parameters=[ConfigSweepParameter(name="x", values=[1, 2, 3])],
                strategy=OptunaStrategy(n_trials=1, objective="math:sin"),
            ),
        )
        kwargs.update(overrides)
        return JobConfig(**kwargs)

    def make_context(self, config=None, **overrides):
        from joshpy.strategies import AdaptiveContext

        config = config or self.make_config()
        session_id = self.registry.create_session(config=config)
        kwargs = dict(
            cli=self.cli,
            config=config,
            registry=self.registry,
            session_id=session_id,
            quiet=True,
            stop_on_failure=False,
        )
        kwargs.update(overrides)
        ctx = AdaptiveContext(**kwargs)
        self.contexts.append(ctx)
        return ctx

    def patch_loading(self, return_value=1, side_effect=None):
        from unittest.mock import patch

        patcher = patch(
            "joshpy.sweep.load_job_results",
            return_value=return_value,
            side_effect=side_effect,
        )
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock


class TestAdaptiveContextExpansion(_AdaptiveContextCase):
    """evaluate() expands rows exactly like a batch sweep."""

    def test_expands_config_params(self):
        self.patch_loading()
        config = self.make_config(replicates=3)
        batch = self.make_context(config).evaluate([{"x": 42}])

        job = batch.jobs[0]
        self.assertEqual(job.parameters, {"x": 42})
        self.assertEqual(job.simulation, "Main")
        self.assertEqual(job.replicates, 3)
        self.assertIn("maxGrowth = 42 meters", job.config_content)
        self.assertEqual(job.custom_tags["x"], "42")
        self.assertTrue(job.config_path.exists())

    def test_expands_file_params(self):
        from joshpy.jobs import SweepConfig

        self.patch_loading()
        data_path = self.tmp / "climate_a.jshd"
        data_path.write_text("test data")
        config = self.make_config(
            template_string="# config",
            sweep=SweepConfig(
                file_parameters=[FileSweepParameter(name="climate", paths=[str(data_path)])],
                strategy=OptunaStrategy(n_trials=1, objective="math:sin"),
            ),
        )
        batch = self.make_context(config).evaluate(
            [{"climate": {"path": data_path, "label": data_path.stem}}]
        )

        job = batch.jobs[0]
        self.assertEqual(job.file_mappings["climate"], data_path)
        self.assertEqual(job.custom_tags["climate"], data_path.stem)

    def test_run_hash_matches_batch_expansion(self):
        """Adaptive and batch sweeps hash the same spec identically (seed included)."""
        from joshpy.jobs import JobExpander, SweepConfig

        self.patch_loading()
        config = self.make_config(seed=7)
        batch = self.make_context(config).evaluate([{"x": 2}])

        batch_config = self.make_config(
            seed=7,
            config_name="sweep_config",
            sweep=SweepConfig(config_parameters=[ConfigSweepParameter(name="x", values=[2])]),
        )
        job_set = JobExpander().expand(batch_config)
        try:
            self.assertEqual(batch.jobs[0].run_hash, job_set.jobs[0].run_hash)
        finally:
            job_set.cleanup()

    def test_seed_changes_run_hash(self):
        self.patch_loading()
        a = self.make_context(self.make_config(seed=1)).evaluate([{"x": 2}])
        b = self.make_context(self.make_config(seed=2)).evaluate([{"x": 2}])
        self.assertNotEqual(a.jobs[0].run_hash, b.jobs[0].run_hash)

    def test_config_name_defaults_to_template_stem(self):
        self.patch_loading()
        template = self.tmp / "adaptive_config.jshc.j2"
        template.write_text("maxGrowth = {{ x }} meters")
        config = self.make_config(template_string=None, template_path=template)
        batch = self.make_context(config).evaluate([{"x": 1}])
        self.assertEqual(batch.jobs[0].config_name, "adaptive_config.jshc")

    def test_explicit_config_name_wins(self):
        self.patch_loading()
        template = self.tmp / "adaptive_config.jshc.j2"
        template.write_text("maxGrowth = {{ x }} meters")
        config = self.make_config(
            template_string=None, template_path=template, config_name="forevertree"
        )
        batch = self.make_context(config).evaluate([{"x": 1}])
        self.assertEqual(batch.jobs[0].config_name, "forevertree.jshc")

    def test_jobs_get_sweep_labels(self):
        self.patch_loading()
        batch = self.make_context(self.make_config(label="exp")).evaluate([{"x": 1}])
        job = batch.jobs[0]
        self.assertEqual(job.custom_tags["label"], f"exp_{job.run_hash}")

    def test_batches_share_one_source_path(self):
        self.patch_loading()
        ctx = self.make_context()
        first = ctx.evaluate([{"x": 1}])
        second = ctx.evaluate([{"x": 2}])
        self.assertEqual(first.jobs[0].source_path, second.jobs[0].source_path)
        self.assertEqual(len(ctx.job_set()), 2)
        self.assertEqual(ctx.job_set().temp_dir, ctx.work_dir)


class TestAdaptiveContextEvaluate(_AdaptiveContextCase):
    """evaluate() registers, runs, loads and scores each batch."""

    def test_registers_each_job_with_attributes(self):
        self.patch_loading()
        batch = self.make_context().evaluate(
            [{"x": 1}, {"x": 2}], attributes={"generation": 0}
        )
        for run_hash in batch.run_hashes:
            self.assertIsNotNone(self.registry.get_config_by_hash(run_hash))
            self.assertEqual(self.registry.get_attributes(run_hash)["generation"], 0)

    def test_values_follow_row_order(self):
        self.patch_loading()
        batch = self.make_context().evaluate(
            [{"x": 3}, {"x": 1}, {"x": 2}],
            score=lambda registry, run_hash, job: job.parameters["x"] * 10,
        )
        self.assertEqual(batch.values, [30, 10, 20])
        self.assertEqual(batch.failures, [])

    def test_duplicate_rows_run_once(self):
        self.patch_loading()
        batch = self.make_context().evaluate([{"x": 1}, {"x": 1}, {"x": 2}])
        self.assertEqual(self.cli.run.call_count, 2)
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch.jobs[0].run_hash, batch.jobs[1].run_hash)

    def test_records_runs_and_counts(self):
        self.patch_loading()
        ctx = self.make_context()
        batch = ctx.evaluate([{"x": 1}, {"x": 2}])
        self.assertEqual(ctx.succeeded, 2)
        self.assertEqual(set(batch.run_ids), set(batch.run_hashes))
        self.assertEqual(ctx.run_ids, batch.run_ids)
        for run_hash in batch.run_hashes:
            self.assertEqual(len(self.registry.get_runs_for_hash(run_hash)), 1)

    def _fail_when_x_is(self, value):
        def run(run_config, **kwargs):
            failing = run_config.custom_tags.get("x") == str(value)
            return _cli_result(exit_code=1 if failing else 0, stderr="boom" if failing else "")

        self.cli.run.side_effect = run

    def test_failed_run_is_reported_not_scored(self):
        load = self.patch_loading()
        self._fail_when_x_is(2)
        ctx = self.make_context()
        batch = ctx.evaluate(
            [{"x": 1}, {"x": 2}], score=lambda registry, run_hash, job: 1.0
        )
        self.assertEqual(batch.values, [1.0, None])
        self.assertEqual(batch.failures, [1])
        self.assertEqual((ctx.succeeded, ctx.failed), (1, 1))
        self.assertEqual(load.call_count, 1)

    def test_stop_on_failure_raises_after_batch(self):
        from joshpy.strategies import SweepExecutionError

        load = self.patch_loading()
        self._fail_when_x_is(2)
        ctx = self.make_context(stop_on_failure=True)
        ctx.evaluate([{"x": 3}])  # one earlier evaluation, for global numbering

        with self.assertRaises(SweepExecutionError) as raised:
            ctx.evaluate([{"x": 1}, {"x": 2}, {"x": 4}])

        self.assertEqual(self.cli.run.call_count, 4)  # the whole batch ran
        self.assertEqual(load.call_count, 3)  # successes were still loaded
        self.assertEqual(raised.exception.trial_num, 2)
        self.assertEqual(raised.exception.succeeded_before, 2)

    def test_loading_error_is_reported(self):
        self.patch_loading(side_effect=RuntimeError("no csv"))
        batch = self.make_context().evaluate(
            [{"x": 1}], score=lambda registry, run_hash, job: 1.0
        )
        self.assertIsNone(batch.values[0])
        self.assertIn("no csv", batch.errors[0])
        self.assertEqual(batch.failures, [0])

    def test_score_error_is_reported(self):
        self.patch_loading()

        def bad_score(registry, run_hash, job):
            raise ValueError("bad metric")

        batch = self.make_context().evaluate([{"x": 1}], score=bad_score)
        self.assertEqual(batch.errors[0], "objective_error: bad metric")

    def test_empty_batch(self):
        batch = self.make_context().evaluate([])
        self.assertEqual(len(batch), 0)
        self.cli.run.assert_not_called()


class TestAdaptiveContextResume(_AdaptiveContextCase):
    """The registry is the run store: completed runs are reused, submitted
    no-wait jobs reattached, everything else run."""

    def loading_writes_cell_data(self, fail_first=0):
        """Patch loading with an idempotent fake that writes one row per replicate."""
        calls = {"n": 0}

        def load(cli, job, registry, export_paths, *, run_id, **kwargs):
            calls["n"] += 1
            if calls["n"] <= fail_first:
                raise RuntimeError("csv not there yet")
            loaded = registry.loaded_replicates(job.run_hash)
            rows = 0
            for rep in range(job.replicates):
                if rep not in loaded:
                    registry.conn.execute(
                        "INSERT INTO cell_data (run_id, run_hash, step, replicate) "
                        "VALUES (?, ?, 0, ?)",
                        [run_id, job.run_hash, rep],
                    )
                    rows += 1
            return rows

        return self.patch_loading(side_effect=load)

    @staticmethod
    def score(registry, run_hash, job):
        return job.parameters["x"] * 10

    def test_completed_run_is_reused(self):
        self.loading_writes_cell_data()
        ctx = self.make_context()
        first = ctx.evaluate([{"x": 1}], score=self.score)
        second = ctx.evaluate([{"x": 1}, {"x": 2}], score=self.score)

        self.assertEqual(first.sources, ["ran"])
        self.assertEqual(second.sources, ["reused", "ran"])
        self.assertEqual(self.cli.run.call_count, 2)
        self.assertEqual(second.values, [10, 20])
        self.assertEqual(second.rows_loaded, [0, 1])
        self.assertEqual(second.run_ids[first.run_hashes[0]], first.run_ids[first.run_hashes[0]])

    def test_reused_after_restart_in_new_session(self):
        self.loading_writes_cell_data()
        self.make_context().evaluate([{"x": 1}])
        batch = self.make_context().evaluate([{"x": 1}], score=self.score)
        self.assertEqual(batch.sources, ["reused"])
        self.assertEqual(self.cli.run.call_count, 1)
        self.assertEqual(batch.values, [10])

    def test_completed_but_unloaded_run_is_loaded_not_rerun(self):
        self.loading_writes_cell_data(fail_first=1)
        ctx = self.make_context()
        first = ctx.evaluate([{"x": 1}], score=self.score)
        self.assertIn(0, first.errors)  # loading failed the first time

        second = ctx.evaluate([{"x": 1}], score=self.score)
        self.assertEqual(second.sources, ["reused"])
        self.assertEqual(self.cli.run.call_count, 1)
        self.assertEqual(second.values, [10])

    def test_failed_run_is_rerun(self):
        self.loading_writes_cell_data()
        self.cli.run.side_effect = [_cli_result(exit_code=1, stderr="boom"), _cli_result()]
        ctx = self.make_context()
        ctx.evaluate([{"x": 1}])
        batch = ctx.evaluate([{"x": 1}])
        self.assertEqual(batch.sources, ["ran"])
        self.assertEqual(self.cli.run.call_count, 2)

    def test_bad_run_is_rerun(self):
        self.loading_writes_cell_data()
        ctx = self.make_context()
        first = ctx.evaluate([{"x": 1}])
        self.registry.mark_run(first.run_hashes[0], "bad", reason="wrong inputs")
        batch = ctx.evaluate([{"x": 1}])
        self.assertEqual(batch.sources, ["ran"])
        self.assertEqual(self.cli.run.call_count, 2)

    def test_reuse_can_be_disabled(self):
        self.loading_writes_cell_data()
        ctx = self.make_context(reuse_runs=False)
        ctx.evaluate([{"x": 1}])
        batch = ctx.evaluate([{"x": 1}])
        self.assertEqual(batch.sources, ["ran"])
        self.assertEqual(self.cli.run.call_count, 2)

    def test_submitted_job_is_reattached(self):
        from unittest.mock import MagicMock

        from joshpy.jobs import JobExpander
        from joshpy.registry import RegistryCallback
        from joshpy.sweep import register_job

        self.loading_writes_cell_data()
        self.cli.inspect_exports.return_value = MagicMock(export_files={}, debug_files={})
        self.cli.poll_batch.return_value = _cli_result()
        ctx = self.make_context(batch_remote=True, target="gke-test", poll_interval=0)

        # A no-wait dispatch from an earlier (crashed) process.
        job = JobExpander().expand(
            ctx._config_for_expansion(), output_dir=ctx.work_dir, combinations=[{"x": 1}]
        ).jobs[0]
        register_job(self.registry, ctx.session_id, job)
        submitted_run = RegistryCallback(self.registry, ctx.session_id).record_dispatch(
            job, batch_job_id="job-9", target="gke-test"
        )

        batch = ctx.evaluate([{"x": 1}], score=self.score)

        self.assertEqual(batch.sources, ["reattached"])
        self.assertEqual(batch.values, [10])
        self.cli.batch_remote.assert_not_called()
        self.cli.stage_to_minio.assert_not_called()
        self.assertEqual(self.cli.poll_batch.call_args.args[0].job_id, "job-9")
        run = self.registry.get_run(submitted_run)
        self.assertEqual(run.exit_code, 0)
        self.assertIsNotNone(run.completed_at)

    @unittest.skipUnless(_has_optuna(), "optuna not installed")
    def test_seeded_optuna_rerun_reuses_completed_trials(self):
        """A seeded study replays its proposals; completed trials are not rerun."""
        from joshpy.jobs import JobConfig, SweepConfig
        from joshpy.strategies import run_adaptive_sweep

        self.loading_writes_cell_data()

        def run(n_trials):
            config = JobConfig(
                template_string="maxGrowth = {{ x }} meters",
                source_path=self.josh_path,
                sweep=SweepConfig(
                    config_parameters=[
                        ConfigSweepParameter(name="x", values=list(range(1, 50)))
                    ],
                    strategy=OptunaStrategy(n_trials=n_trials, seed=11),
                ),
            )
            session_id = self.registry.create_session(config=config)
            result = run_adaptive_sweep(
                self.cli, config, registry=self.registry, session_id=session_id,
                objective=self.score, quiet=True,
            )
            result.job_set.cleanup()
            return result

        interrupted = run(4)
        runs_before = self.cli.run.call_count
        resumed = run(6)

        first_params = [t.params for t in interrupted.study.trials]
        self.assertEqual([t.params for t in resumed.study.trials[:4]], first_params)
        new_hashes = {
            job.run_hash for job, _ in resumed.job_results[4:]
        } - {job.run_hash for job, _ in interrupted.job_results}
        self.assertEqual(self.cli.run.call_count - runs_before, len(new_hashes))


class _RecordingStrategy(SweepStrategy):
    """Minimal adaptive strategy: evaluates fixed batches, returns the context."""

    def __init__(self, batches):
        self.batches = batches

    @property
    def is_adaptive(self):
        return True

    def expand(self, config_params, file_params, compound_params=None):
        raise RuntimeError("adaptive")

    def run_adaptive(self, ctx):
        from joshpy.jobs import AdaptiveSweepResult

        for batch in self.batches:
            ctx.evaluate(batch)
        return AdaptiveSweepResult(
            job_results=list(ctx.job_results),
            succeeded=ctx.succeeded,
            failed=ctx.failed,
            run_ids=dict(ctx.run_ids),
            job_set=ctx.job_set(),
        )

    def to_dict(self):
        return {"type": "recording"}

    @classmethod
    def from_dict(cls, data):
        return cls([])


class TestRunAdaptiveSweepDispatch(_AdaptiveContextCase):
    """run_adaptive_sweep hands any adaptive strategy a context."""

    def test_non_optuna_strategy_runs_its_own_loop(self):
        from joshpy.jobs import JobConfig, SweepConfig
        from joshpy.strategies import run_adaptive_sweep

        self.patch_loading()
        config = JobConfig(
            template_string="maxGrowth = {{ x }} meters",
            source_path=self.josh_path,
            sweep=SweepConfig(strategy=_RecordingStrategy([[{"x": 1}, {"x": 2}], [{"x": 3}]])),
        )
        session_id = self.registry.create_session(config=config)
        result = run_adaptive_sweep(
            self.cli, config, registry=self.registry, session_id=session_id, quiet=True
        )
        try:
            self.assertEqual(result.succeeded, 3)
            self.assertEqual(len(result.job_set), 3)
            self.assertEqual(self.registry.get_session(session_id).status, "completed")
        finally:
            result.job_set.cleanup()

    def test_strategy_error_marks_session_failed(self):
        from joshpy.jobs import JobConfig, SweepConfig
        from joshpy.strategies import run_adaptive_sweep

        class Exploding(_RecordingStrategy):
            def run_adaptive(self, ctx):
                raise RuntimeError("strategy blew up")

        config = JobConfig(
            template_string="# config",
            source_path=self.josh_path,
            sweep=SweepConfig(strategy=Exploding([])),
        )
        session_id = self.registry.create_session(config=config)
        with self.assertRaises(RuntimeError):
            run_adaptive_sweep(
                self.cli, config, registry=self.registry, session_id=session_id, quiet=True
            )
        self.assertEqual(self.registry.get_session(session_id).status, "failed")

    def test_non_adaptive_strategy_has_no_loop(self):
        with self.assertRaises(NotImplementedError):
            CartesianStrategy().run_adaptive(None)


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

        self.assertIn("Exit code: 1", str(error))
        self.assertIn("Job 6", str(error))  # trial_num is 0-indexed
        self.assertIn("3 job(s) succeeded", str(error))
        self.assertIn("abc123", str(error))
        self.assertIn("General error", str(error))  # diagnostic for exit code 1
        self.assertIn("Suggestion:", str(error))

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

    def test_oom_exit_code_137_diagnostic(self):
        """Exit code 137 (SIGKILL/OOM) should have helpful diagnostic."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 137
        result.stderr = ""

        error = SweepExecutionError(
            job=job, result=result, trial_num=0, succeeded_before=0
        )

        self.assertIn("out of memory", str(error).lower())
        self.assertIn("--use-float-64", str(error))
        self.assertEqual(error.diagnostic, "Process killed (SIGKILL) - likely out of memory")

    def test_sigterm_exit_code_143_diagnostic(self):
        """Exit code 143 (SIGTERM) should have helpful diagnostic."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 143
        result.stderr = ""

        error = SweepExecutionError(
            job=job, result=result, trial_num=0, succeeded_before=0
        )

        self.assertIn("terminated", str(error).lower())
        self.assertIn("--use-float-64", str(error))
        self.assertEqual(
            error.diagnostic,
            "Process terminated (SIGTERM) - likely out of memory or system timeout"
        )

    def test_unknown_exit_code_has_generic_message(self):
        """Unknown exit codes should still have a message."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 255
        result.stderr = ""

        error = SweepExecutionError(
            job=job, result=result, trial_num=0, succeeded_before=0
        )

        self.assertIn("Exit code: 255", str(error))
        self.assertIn("signal 127", str(error).lower())  # 255 - 128 = 127

    def test_jar_error_lifted_into_diagnostic(self):
        """The JAR's real error line should become the diagnostic, not the
        generic exit-code message."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 1  # would otherwise map to "General error"
        result.stderr = (
            "java.lang.IllegalArgumentException: "
            "Config value not found: sweep_config.mastYearIntervalTYPO\n"
            "\tat org.joshsim.lang.interpret.machine.Foo.lambda$pushConfig$7(Foo.java:1)\n"
            "\t... 59 more"
        )

        error = SweepExecutionError(
            job=job, result=result, trial_num=0, succeeded_before=0
        )

        self.assertIn("Config value not found: sweep_config.mastYearIntervalTYPO", error.diagnostic)
        self.assertNotEqual(error.diagnostic, "General error")
        # The exit-code interpretation is still available separately.
        self.assertEqual(error.exit_code_diagnostic, "General error")
        # Full stderr is preserved in the message.
        self.assertIn("59 more", str(error))

    def test_batch_remote_failure_lifted_not_missing_arg(self):
        """A 'batchRemote failed' line should surface instead of the misleading
        'Missing required argument' for exit code 100."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 100  # maps to "Missing required argument"
        result.stderr = "batchRemote failed: Access denied. (exit code 100)"

        error = SweepExecutionError(
            job=job, result=result, trial_num=0, succeeded_before=0
        )

        self.assertEqual(error.diagnostic, "batchRemote failed: Access denied. (exit code 100)")
        self.assertEqual(error.exit_code_diagnostic, "Missing required argument")

    def test_falls_back_to_exit_code_diagnostic_when_stderr_unhelpful(self):
        """When stderr is empty or only stack frames, fall back to the
        exit-code-based diagnostic."""
        from joshpy.strategies import SweepExecutionError
        from unittest.mock import MagicMock

        job = MagicMock()
        job.parameters = {"x": 10}
        job.run_hash = "abc123"

        result = MagicMock()
        result.exit_code = 1
        result.stderr = ""

        error = SweepExecutionError(
            job=job, result=result, trial_num=0, succeeded_before=0
        )
        self.assertEqual(error.diagnostic, "General error")


class TestExtractJarError(unittest.TestCase):
    """Tests for extract_jar_error helper."""

    def test_lifts_java_exception_line(self):
        from joshpy.strategies import extract_jar_error

        trace = (
            "java.lang.IllegalArgumentException: Config value not found: foo\n"
            "\tat org.joshsim.Foo.bar(Foo.java:1)\n"
            "\t... 59 more"
        )
        self.assertEqual(
            extract_jar_error(trace),
            "java.lang.IllegalArgumentException: Config value not found: foo",
        )

    def test_lifts_caused_by_exception(self):
        from joshpy.strategies import extract_jar_error

        trace = "Caused by: java.io.IOException: connection reset\n\tat Foo.bar(Foo.java:1)"
        self.assertEqual(
            extract_jar_error(trace), "java.io.IOException: connection reset"
        )

    def test_lifts_failed_line(self):
        from joshpy.strategies import extract_jar_error

        self.assertEqual(
            extract_jar_error("batchRemote failed: Access denied. (exit code 100)"),
            "batchRemote failed: Access denied. (exit code 100)",
        )

    def test_returns_none_for_empty(self):
        from joshpy.strategies import extract_jar_error

        self.assertIsNone(extract_jar_error(""))
        self.assertIsNone(extract_jar_error(None))

    def test_returns_none_for_frames_only(self):
        from joshpy.strategies import extract_jar_error

        self.assertIsNone(
            extract_jar_error("\tat Foo.bar(Foo.java:1)\n\t... 3 more")
        )

    def test_first_non_frame_line_fallback(self):
        from joshpy.strategies import extract_jar_error

        trace = "Something went wrong\n\tat Foo.bar(Foo.java:1)"
        self.assertEqual(extract_jar_error(trace), "Something went wrong")


class TestGetExitCodeDiagnostic(unittest.TestCase):
    """Tests for get_exit_code_diagnostic function."""

    def test_known_exit_code(self):
        """Known exit codes should return specific messages."""
        from joshpy.strategies import get_exit_code_diagnostic

        desc, suggestion = get_exit_code_diagnostic(137)
        self.assertIn("memory", desc.lower())
        self.assertIn("float-64", suggestion)

    def test_signal_based_exit_code(self):
        """Signal-based codes (> 128) should mention signal number."""
        from joshpy.strategies import get_exit_code_diagnostic

        desc, suggestion = get_exit_code_diagnostic(130)  # 128 + 2 = SIGINT
        self.assertIn("signal 2", desc.lower())

    def test_unknown_exit_code(self):
        """Unknown exit codes should return generic message."""
        from joshpy.strategies import get_exit_code_diagnostic

        desc, suggestion = get_exit_code_diagnostic(42)
        self.assertIn("42", desc)
        self.assertIn("stderr", suggestion.lower())


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
            from joshpy.cli import CLIResult

            mock_cli = MagicMock()
            mock_cli.run.return_value = CLIResult(
                exit_code=1, stdout="", stderr="Test error message", command=[]
            )
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
            from joshpy.cli import CLIResult

            mock_cli = MagicMock()
            mock_cli.run.return_value = CLIResult(
                exit_code=1, stdout="", stderr="Test error", command=[]
            )
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
