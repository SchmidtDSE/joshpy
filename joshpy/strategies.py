"""Sweep expansion strategies for parameter optimization.

This module defines the strategy pattern for parameter sweep expansion:
- `CartesianStrategy`: Full cartesian product of all parameters (default)
- `OptunaStrategy`: Bayesian optimization using Optuna (requires [adaptive] extra)

The strategy pattern enables different approaches to parameter space exploration
without changing the SweepConfig interface.

Example usage:
    from joshpy.strategies import CartesianStrategy, OptunaStrategy
    from joshpy.jobs import SweepConfig, ConfigSweepParameter

    # Default: exhaustive cartesian product
    config = SweepConfig(
        config_parameters=[
            ConfigSweepParameter(name="maxGrowth", values=[10, 50, 100]),
        ],
        strategy=CartesianStrategy(),
    )

    # Intelligent search with Optuna (requires pip install joshpy[adaptive])
    config = SweepConfig(
        config_parameters=[
            ConfigSweepParameter(name="maxGrowth", values=[10, 50, 100]),
        ],
        strategy=OptunaStrategy(
            n_trials=50,
            direction="minimize",
            objective="my_module:stability_metric",
        ),
    )
"""

from __future__ import annotations

import importlib
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from joshpy.jobs import ConfigSweepParameter, ExpandedJob, FileSweepParameter
    from joshpy.registry import RunRegistry

# Type alias for objective functions
# Signature: (registry, run_hash, job) -> float
ObjectiveFn = Callable[["RunRegistry", str, "ExpandedJob"], float]

# Strategy type registry for YAML serialization
_STRATEGY_TYPES: dict[str, type[SweepStrategy]] = {}


def register_strategy(name: str) -> Callable[[type[SweepStrategy]], type[SweepStrategy]]:
    """Decorator to register a strategy type for YAML serialization.

    Args:
        name: The type name to use in YAML (e.g., "cartesian", "optuna").

    Returns:
        Decorator that registers the strategy class.
    """

    def decorator(cls: type[SweepStrategy]) -> type[SweepStrategy]:
        _STRATEGY_TYPES[name] = cls
        return cls

    return decorator


def strategy_from_dict(data: dict[str, Any]) -> SweepStrategy:
    """Create a strategy from a dict (YAML deserialization).

    Args:
        data: Dict with "type" key and strategy-specific fields.

    Returns:
        Instantiated strategy.

    Raises:
        ValueError: If type is unknown.

    Examples:
        >>> strategy_from_dict({"type": "cartesian"})
        CartesianStrategy()
        >>> strategy_from_dict({"type": "optuna", "n_trials": 50})
        OptunaStrategy(n_trials=50, ...)
    """
    strategy_type = data.get("type", "cartesian")

    if strategy_type not in _STRATEGY_TYPES:
        available = ", ".join(_STRATEGY_TYPES.keys())
        raise ValueError(f"Unknown strategy type '{strategy_type}'. Available: {available}")

    cls = _STRATEGY_TYPES[strategy_type]
    # Remove "type" key before passing to from_dict
    strategy_data = {k: v for k, v in data.items() if k != "type"}
    return cls.from_dict(strategy_data)


class SweepStrategy(ABC):
    """Base class for sweep expansion strategies.

    Strategies define how parameter combinations are generated:
    - Non-adaptive strategies (like CartesianStrategy) yield all combinations upfront
    - Adaptive strategies (like OptunaStrategy) generate combinations on-demand
      based on results from previous runs

    Subclasses must implement:
    - `is_adaptive` property
    - `expand()` method (for non-adaptive) or indicate adaptive behavior
    - `to_dict()` and `from_dict()` for serialization
    """

    @property
    @abstractmethod
    def is_adaptive(self) -> bool:
        """True if strategy needs results before suggesting next params.

        Non-adaptive strategies can expand all combinations upfront.
        Adaptive strategies require incremental execution with result feedback.
        """
        ...

    @abstractmethod
    def expand(
        self,
        config_params: list[ConfigSweepParameter],
        file_params: list[FileSweepParameter],
    ) -> Iterator[dict[str, Any]]:
        """Yield parameter combinations to try.

        For non-adaptive strategies, yields all combinations upfront.
        For adaptive strategies, this raises an error - use run_adaptive_sweep() instead.

        Args:
            config_params: Configuration parameters to sweep over.
            file_params: File parameters to sweep over.

        Returns:
            Iterator of dicts, each containing parameter values for one combination.
            Config params: {"name": value}
            File params: {"name": {"path": Path, "label": str}}
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for YAML/JSON serialization.

        Must include a "type" key for deserialization.
        """
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepStrategy:
        """Create from dict (YAML/JSON deserialization).

        Args:
            data: Dict WITHOUT the "type" key (already consumed by strategy_from_dict).
        """
        ...


@register_strategy("cartesian")
@dataclass
class CartesianStrategy(SweepStrategy):
    """Full cartesian product of all parameters (default strategy).

    Generates every possible combination of parameter values. This is the
    traditional "grid search" approach.

    For N parameters with values [v1, v2, ...], generates:
    - Total combinations = len(v1) * len(v2) * ... * len(vN)

    Examples:
        >>> strategy = CartesianStrategy()
        >>> config = SweepConfig(
        ...     config_parameters=[
        ...         ConfigSweepParameter(name="a", values=[1, 2]),
        ...         ConfigSweepParameter(name="b", values=[10, 20]),
        ...     ],
        ...     strategy=strategy,
        ... )
        >>> # Generates: [{a:1, b:10}, {a:1, b:20}, {a:2, b:10}, {a:2, b:20}]
    """

    @property
    def is_adaptive(self) -> bool:
        """Cartesian strategy is not adaptive - all combinations known upfront."""
        return False

    def expand(
        self,
        config_params: list[ConfigSweepParameter],
        file_params: list[FileSweepParameter],
    ) -> Iterator[dict[str, Any]]:
        """Generate cartesian product of all parameter values.

        Args:
            config_params: Configuration parameters to sweep over.
            file_params: File parameters to sweep over.

        Returns:
            Iterator of dicts, one for each parameter combination.
        """
        # Build all param names and value lists
        names: list[str] = []
        value_lists: list[list[Any]] = []
        file_param_names: set[str] = set()

        for cp in config_params:
            names.append(cp.name)
            value_lists.append(cp.values)

        for fp in file_params:
            names.append(fp.name)
            # (path, label) tuples for file params
            value_lists.append(list(zip(fp.paths, fp.labels)))
            file_param_names.add(fp.name)

        if not names:
            yield {}
            return

        for combo in itertools.product(*value_lists):
            result: dict[str, Any] = {}
            for name, value in zip(names, combo):
                if name in file_param_names:
                    path, label = value
                    result[name] = {"path": path, "label": label}
                else:
                    result[name] = value
            yield result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {"type": "cartesian"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CartesianStrategy:
        """Create from dict (no additional fields)."""
        return cls()


def _check_optuna() -> None:
    """Raise ImportError if optuna is not available."""
    try:
        import optuna  # noqa: F401
    except ImportError:
        raise ImportError(
            "optuna is required for adaptive sweeps. Install with: pip install joshpy[adaptive]"
        )


def _load_objective_from_path(path: str) -> ObjectiveFn:
    """Load function from 'module.submodule:function_name' format.

    Args:
        path: Import path in "module:function" format.

    Returns:
        The loaded callable.

    Raises:
        ValueError: If path format is invalid or target is not callable.
        ImportError: If module cannot be imported.
        AttributeError: If function doesn't exist in module.

    Examples:
        >>> fn = _load_objective_from_path("my_project.objectives:stability_metric")
        >>> metric = fn(registry, run_hash, job)
    """
    if ":" not in path:
        raise ValueError(f"Invalid objective path '{path}'. Expected 'module:function' format.")
    module_path, func_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise ValueError(f"{path} is not callable")
    return fn


@register_strategy("optuna")
@dataclass
class OptunaStrategy(SweepStrategy):
    """Bayesian optimization using Optuna.

    Uses Optuna's intelligent sampling to efficiently explore the parameter
    space, learning from previous results to focus on promising regions.

    This is an adaptive strategy - it requires incremental execution where
    each trial's results inform the next suggestion. Use run_adaptive_sweep()
    instead of the standard expand() method, or use SweepManager which handles
    this automatically.

    Attributes:
        n_trials: Maximum number of trials to run.
        direction: Optimization direction ("minimize" or "maximize").
        sampler: Sampling algorithm ("tpe", "gp", "cmaes", "random").
        objective: Objective function (callable or "module:function" path).

    Examples:
        >>> # Define objective function
        >>> def stability_metric(registry, run_hash, job):
        ...     df = queries.get_timeseries("population", run_hash=run_hash)
        ...     return df["value"].std() / df["value"].mean()  # CV

        >>> strategy = OptunaStrategy(
        ...     n_trials=50,
        ...     direction="minimize",
        ...     objective=stability_metric,
        ... )

        >>> # Or specify via YAML path
        >>> strategy = OptunaStrategy(
        ...     n_trials=50,
        ...     direction="minimize",
        ...     objective="my_project.objectives:stability_metric",
        ... )
    """

    n_trials: int = 100
    direction: str = "minimize"  # "minimize" or "maximize"
    sampler: str = "tpe"  # "tpe", "gp", "cmaes", "random"
    objective: ObjectiveFn | str | None = None

    # Resolved objective (populated during validation or get_objective)
    _objective_fn: ObjectiveFn | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate strategy configuration.

        Note: Objective function resolution is deferred to get_objective() to allow
        YAML serialization/deserialization of strategy configs that reference
        modules not available at parse time.
        """
        # Validate direction
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(
                f"Invalid direction '{self.direction}'. Must be 'minimize' or 'maximize'."
            )

        # Validate sampler
        valid_samplers = ("tpe", "gp", "cmaes", "random")
        if self.sampler not in valid_samplers:
            raise ValueError(f"Invalid sampler '{self.sampler}'. Must be one of: {valid_samplers}")

        # Cache callable objectives immediately; defer string path resolution
        if callable(self.objective) and not isinstance(self.objective, str):
            self._objective_fn = self.objective
        # String paths are resolved lazily in get_objective()

    @property
    def is_adaptive(self) -> bool:
        """Optuna strategy is adaptive - requires result feedback."""
        return True

    def expand(
        self,
        config_params: list[ConfigSweepParameter],
        file_params: list[FileSweepParameter],
    ) -> Iterator[dict[str, Any]]:
        """Not supported for adaptive strategies, because, during a given run,
        the next parameters depend on the results of previous runs.

        Raises:
            RuntimeError: Always - use run_adaptive_sweep() instead.
        """
        raise RuntimeError(
            "OptunaStrategy is adaptive - use run_adaptive_sweep() instead of expand(). "
            "Adaptive strategies require incremental execution with result feedback."
        )

    def get_objective(self) -> ObjectiveFn:
        """Get the resolved objective function.

        Resolves string path objectives on first call (lazy loading).

        Returns:
            The objective function callable.

        Raises:
            ValueError: If no objective function is configured.
            ImportError: If string path references non-existent module.
        """
        # Resolve string path objectives lazily
        if self._objective_fn is None and isinstance(self.objective, str):
            self._objective_fn = _load_objective_from_path(self.objective)

        if self._objective_fn is None:
            raise ValueError(
                "No objective function configured for OptunaStrategy. "
                "Provide objective= in constructor or pass to run_adaptive_sweep()."
            )
        return self._objective_fn

    def create_study(self) -> Any:
        """Create an Optuna study with configured sampler.

        Returns:
            optuna.Study configured with this strategy's settings.

        Raises:
            ImportError: If optuna is not installed.
        """
        _check_optuna()
        import optuna

        sampler = self._create_sampler()
        return optuna.create_study(direction=self.direction, sampler=sampler)

    def _create_sampler(self) -> Any:
        """Create Optuna sampler based on configuration.

        Returns:
            Optuna sampler instance.
        """
        _check_optuna()
        import optuna

        if self.sampler == "tpe":
            return optuna.samplers.TPESampler()
        elif self.sampler == "gp":
            # GP sampler requires optuna-integration or specific setup
            # Fall back to TPE with a warning
            try:
                from optuna_integration import BoTorchSampler

                return BoTorchSampler()
            except ImportError:
                import warnings

                warnings.warn(
                    "GP sampler requires optuna-integration[botorch]. Falling back to TPE."
                )
                return optuna.samplers.TPESampler()
        elif self.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler()
        elif self.sampler == "random":
            return optuna.samplers.RandomSampler()
        else:
            return optuna.samplers.TPESampler()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Note: Callable objectives cannot be serialized - only string paths.
        """
        result: dict[str, Any] = {
            "type": "optuna",
            "n_trials": self.n_trials,
            "direction": self.direction,
            "sampler": self.sampler,
        }

        # Only serialize objective if it's a string path
        if isinstance(self.objective, str):
            result["objective"] = self.objective
        elif self.objective is not None:
            # Callable - try to get qualified name
            if hasattr(self.objective, "__module__") and hasattr(self.objective, "__name__"):
                result["objective"] = f"{self.objective.__module__}:{self.objective.__name__}"
            # Otherwise skip - user will need to provide at runtime

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptunaStrategy:
        """Create from dict (YAML/JSON deserialization)."""
        return cls(
            n_trials=data.get("n_trials", 100),
            direction=data.get("direction", "minimize"),
            sampler=data.get("sampler", "tpe"),
            objective=data.get("objective"),
        )


def sample_params_from_trial(
    trial: Any,  # optuna.Trial
    config_params: list[ConfigSweepParameter],
    file_params: list[FileSweepParameter],
) -> dict[str, Any]:
    """Sample parameters from an Optuna trial.

    Called by run_adaptive_sweep() to get parameter suggestions from Optuna.

    Args:
        trial: Optuna trial object.
        config_params: Configuration parameters to sample.
        file_params: File parameters to sample.

    Returns:
        Dict of sampled parameter values, matching the format from expand().
    """
    params: dict[str, Any] = {}

    for cp in config_params:
        # Use suggest_categorical for discrete values from the sweep
        params[cp.name] = trial.suggest_categorical(cp.name, cp.values)

    for fp in file_params:
        # Files are categorical choices
        idx = trial.suggest_categorical(f"_file_{fp.name}", list(range(len(fp.paths))))
        params[fp.name] = {"path": fp.paths[idx], "label": fp.labels[idx]}

    return params
