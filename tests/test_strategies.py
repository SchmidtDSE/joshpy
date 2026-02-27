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


if __name__ == "__main__":
    unittest.main()
