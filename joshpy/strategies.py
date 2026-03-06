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


def cv_objective(
    variable: str, burn_in: int = 0, extinction_threshold: float = 0.01
) -> ObjectiveFn:
    """Create objective minimizing coefficient of variation.

    Computes CV separately for each replicate's time series (after burn-in),
    then returns the mean CV across replicates. This correctly detects
    instability even when replicates oscillate out of phase.

    Lower CV = more stable dynamics. Common for equilibrium optimization.

    Args:
        variable: Export variable to analyze (e.g., "totalCover")
        burn_in: Steps to skip before measuring (default: 0)
        extinction_threshold: Mean below this returns inf (default: 0.01)

    Returns:
        Objective function: (registry, run_hash, job) -> float
        Returns float('inf') if any replicate goes extinct or has insufficient data.

    Example:
        >>> strategy = OptunaStrategy(
        ...     n_trials=30,
        ...     direction="minimize",
        ...     objective=cv_objective("totalCover", burn_in=50),
        ... )
    """

    def objective(registry: RunRegistry, run_hash: str, job: ExpandedJob) -> float:
        from joshpy.cell_data import DiagnosticQueries

        queries = DiagnosticQueries(registry)
        result = queries.get_replicate_cv(
            variable=variable,
            run_hash=run_hash,
            burn_in=burn_in,
            extinction_threshold=extinction_threshold,
        )
        return result["mean_cv"]

    return objective


class SweepExecutionError(Exception):
    """Raised when a sweep trial fails and stop_on_failure=True.

    This exception provides detailed information about the failed trial,
    including the job parameters, run hash, and stderr output from the
    CLI execution.

    Attributes:
        job: The ExpandedJob that failed.
        result: The CLIResult with exit code and stderr.
        trial_num: Zero-indexed trial number that failed.
        succeeded_before: Number of trials that succeeded before this failure.

    Examples:
        >>> try:
        ...     result = manager.run(objective=my_objective)
        ... except SweepExecutionError as e:
        ...     print(f"Trial {e.trial_num + 1} failed")
        ...     print(f"Parameters: {e.job.parameters}")
        ...     print(f"Error: {e.result.stderr}")
    """

    def __init__(
        self,
        job: Any,  # ExpandedJob
        result: Any,  # CLIResult
        trial_num: int,
        succeeded_before: int,
    ) -> None:
        self.job = job
        self.result = result
        self.trial_num = trial_num
        self.succeeded_before = succeeded_before

        # Build informative message
        message = (
            f"Trial {trial_num + 1} failed with exit_code={result.exit_code}. "
            f"{succeeded_before} trial(s) succeeded before this failure.\n"
            f"Parameters: {job.parameters}\n"
            f"Run hash: {job.run_hash}"
        )
        if result.stderr:
            message += f"\n\nSTDERR:\n{result.stderr}"

        super().__init__(message)


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
    return fn  # type: ignore[return-value]


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


def run_adaptive_sweep(
    cli: Any,  # JoshCLI
    config: Any,  # JobConfig
    *,
    registry: RunRegistry,
    session_id: str,
    objective: ObjectiveFn | None = None,
    remote: bool = False,
    api_key: str | None = None,
    endpoint: str | None = None,
    quiet: bool = False,
    load_config: Any | None = None,  # LoadConfig
    stop_on_failure: bool = True,
) -> Any:  # AdaptiveSweepResult
    """Run an adaptive sweep using Optuna.

    Unlike run_sweep(), this generates jobs on-demand based on Optuna's
    suggestions, loading results after each run to inform the next trial.

    The lifecycle for each trial:
    1. Sample parameters from Optuna trial
    2. Create and register job for these params
    3. Execute CLI for all replicates
    4. Wait briefly for export writes to settle
    5. Load CSV results with retries
    6. Evaluate objective function
    7. Report metric back to Optuna and store in registry

    Note: Optuna runs trials serially (one at a time). Josh maintains
    parallel replicate execution internally. Metrics wait for all
    replicates to finish before evaluation.

    Args:
        cli: JoshCLI instance to use for execution.
        config: JobConfig with OptunaStrategy configured.
        registry: RunRegistry for tracking (required).
        session_id: Session ID in registry (required).
        objective: Optional objective function override. If not provided,
            uses the objective from the strategy.
        remote: If True, use run_remote() for cloud execution.
        api_key: API key for remote execution (optional for local servers).
        endpoint: Custom endpoint URL for remote execution.
        quiet: If True, suppress progress output.
        load_config: LoadConfig for result loading behavior.
        stop_on_failure: If True (default), raise SweepExecutionError on first
            trial failure. If False, continue with remaining trials and report
            failures in the result.

    Returns:
        AdaptiveSweepResult with trial outcomes and best params.

    Raises:
        TypeError: If config.sweep.strategy is not OptunaStrategy.
        ValueError: If no objective function is configured.
        ImportError: If optuna is not installed.
        SweepExecutionError: If stop_on_failure=True and a trial fails.

    Examples:
        >>> from joshpy.strategies import run_adaptive_sweep, OptunaStrategy
        >>> from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter
        >>> from joshpy.registry import RunRegistry
        >>> from joshpy.cli import JoshCLI
        >>>
        >>> def stability_metric(registry, run_hash, job):
        ...     # Lower = more stable
        ...     df = queries.get_timeseries("population", run_hash=run_hash)
        ...     return df["value"].std() / df["value"].mean()
        >>>
        >>> config = JobConfig(
        ...     template_path=Path("template.jshc.j2"),
        ...     source_path=Path("simulation.josh"),
        ...     sweep=SweepConfig(
        ...         config_parameters=[
        ...             ConfigSweepParameter(name="maxGrowth", values=[10, 50, 100]),
        ...         ],
        ...         strategy=OptunaStrategy(n_trials=50, direction="minimize"),
        ...     ),
        ... )
        >>>
        >>> registry = RunRegistry("experiment.duckdb")
        >>> session_id = registry.create_session(config=config)
        >>> cli = JoshCLI()
        >>>
        >>> result = run_adaptive_sweep(
        ...     cli, config,
        ...     registry=registry,
        ...     session_id=session_id,
        ...     objective=stability_metric,
        ... )
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Best value: {result.best_value}")
    """
    _check_optuna()

    # Import here to avoid circular dependencies
    from joshpy.cli import InspectExportsConfig
    from joshpy.jobs import (
        AdaptiveSweepResult,
        to_run_config,
        to_run_remote_config,
    )
    from joshpy.sweep import LoadConfig, load_job_results

    if load_config is None:
        load_config = LoadConfig()

    # Validate strategy
    if config.sweep is None:
        raise ValueError("config.sweep is required for adaptive sweeps")

    base_strategy = config.sweep.strategy
    if not isinstance(base_strategy, OptunaStrategy):
        raise TypeError(
            f"run_adaptive_sweep requires OptunaStrategy, got {type(base_strategy).__name__}. "
            "For non-adaptive strategies, use run_sweep() instead."
        )

    # After isinstance check, we know it's OptunaStrategy
    strategy: OptunaStrategy = base_strategy

    # Get objective function (from arg or strategy)
    obj_fn = objective or strategy.get_objective()

    # Create Optuna study
    study = strategy.create_study()

    # Get export paths once (for result loading)
    export_paths = cli.inspect_exports(
        InspectExportsConfig(
            script=config.source_path,
            simulation=config.simulation,
        )
    )

    # Initialize tracking
    job_results: list[tuple[ExpandedJob, Any]] = []
    succeeded = 0
    failed = 0
    n_trials = strategy.n_trials

    if not quiet:
        print(f"Running adaptive sweep with {n_trials} trials")
        print(f"  Direction: {strategy.direction}")
        print(f"  Sampler: {strategy.sampler}")

    # Set session status to running
    registry.update_session_status(session_id, "running")

    try:
        for trial_num in range(n_trials):
            # 1. Ask Optuna for next params
            trial = study.ask()

            # 2. Sample parameters from trial
            params = sample_params_from_trial(
                trial,
                config.sweep.config_parameters,
                config.sweep.file_parameters,
            )

            if not quiet:
                # Format params for display
                display_params = {
                    k: v["label"] if isinstance(v, dict) and "label" in v else v
                    for k, v in params.items()
                }
                print(f"[{trial_num + 1}/{n_trials}] Params: {display_params}")

            # 3. Create job for these params
            job = _create_single_job(config, params, trial_num)

            # 4. Register job in registry
            registry.register_run(
                session_id=session_id,
                run_hash=job.run_hash,
                josh_path=str(job.source_path) if job.source_path else "",
                config_content=job.config_content,
                file_mappings=_convert_file_mappings(job.file_mappings),
                parameters=job.parameters,
            )

            # 5. Execute CLI
            if remote:
                run_config = to_run_remote_config(job, api_key=api_key, endpoint=endpoint)
                result = cli.run_remote(run_config)
            else:
                run_config = to_run_config(job)
                result = cli.run(run_config)

            job_results.append((job, result))

            # 6. Record the run result in the registry (needed for load_job_results)
            run_id = registry.start_run(
                run_hash=job.run_hash,
                replicate=0,  # CLI runs all replicates at once
                output_path=str(job.config_path.parent) if job.config_path else None,
                metadata={"parameters": job.parameters, "trial": trial_num},
            )
            error_msg = result.stderr if not result.success else None
            registry.complete_run(
                run_id=run_id,
                exit_code=result.exit_code,
                error_message=error_msg,
            )

            if result.success:
                succeeded += 1

                # 7. Wait for writes to settle and load results
                import time

                time.sleep(load_config.settle_delay)

                try:
                    rows_loaded = load_job_results(
                        cli,
                        job,
                        registry,
                        export_paths,
                        quiet=True,
                        load_config=load_config,
                        succeeded_before=succeeded - 1,
                    )
                except Exception as e:
                    # Log but don't fail the whole sweep
                    if not quiet:
                        print(f"  [WARN] Result loading failed: {e}")
                    # Set user attrs BEFORE tell() since trial is finished after tell()
                    trial.set_user_attr("error", str(e))
                    trial.set_user_attr("run_hash", job.run_hash)
                    study.tell(trial, float("inf"))
                    continue

                # 7. Compute objective
                try:
                    metric = obj_fn(registry, job.run_hash, job)
                except Exception as e:
                    if not quiet:
                        print(f"  [WARN] Objective evaluation failed: {e}")
                    # Set user attrs BEFORE tell() since trial is finished after tell()
                    trial.set_user_attr("error", f"objective_error: {e}")
                    trial.set_user_attr("run_hash", job.run_hash)
                    study.tell(trial, float("inf"))
                    continue

                # 8. Store traceability info BEFORE tell() since trial is finished after
                trial.set_user_attr("run_hash", job.run_hash)
                trial.set_user_attr("metric", metric)
                trial.set_user_attr("rows_loaded", rows_loaded)

                # Report to Optuna (finishes the trial)
                study.tell(trial, metric)

                if not quiet:
                    print(f"  [OK] metric={metric:.4f} (rows={rows_loaded})")

            else:
                failed += 1
                # Set user attrs BEFORE tell() since trial is finished after tell()
                trial.set_user_attr("run_hash", job.run_hash)
                trial.set_user_attr("error", f"exit_code={result.exit_code}")
                # Failed trial gets worst possible value
                study.tell(trial, float("inf"))

                if not quiet:
                    print(f"  [FAIL] exit_code={result.exit_code}")
                    if result.stderr:
                        print(f"\n  STDERR:\n{result.stderr}")

                if stop_on_failure:
                    # Update session status before raising
                    registry.update_session_status(session_id, "failed")
                    raise SweepExecutionError(
                        job=job,
                        result=result,
                        trial_num=trial_num,
                        succeeded_before=succeeded,
                    )

        # Store study outcomes in registry
        _store_study_outcomes(registry, session_id, study)

        # Update session status
        final_status = "completed" if failed == 0 else "failed"
        registry.update_session_status(session_id, final_status)

        if not quiet:
            print("\nAdaptive sweep complete:")
            print(f"  Trials: {n_trials} ({succeeded} succeeded, {failed} failed)")
            if study.best_trial is not None:
                print(f"  Best value: {study.best_value}")
                print(f"  Best params: {study.best_params}")

        return AdaptiveSweepResult(
            job_results=job_results,
            succeeded=succeeded,
            failed=failed,
            study=study,
            best_params=study.best_params if study.best_trial else None,
            best_value=study.best_value if study.best_trial else None,
        )

    except Exception:
        # Set status to failed on exception
        registry.update_session_status(session_id, "failed")
        raise


def _create_single_job(
    config: Any,  # JobConfig
    params: dict[str, Any],
    trial_num: int,
) -> Any:  # ExpandedJob
    """Create a single ExpandedJob from sampled parameters.

    Used by run_adaptive_sweep() to create jobs dynamically for each trial.

    Args:
        config: The JobConfig with template and settings.
        params: Parameter values sampled from Optuna trial.
        trial_num: Trial number (for unique directory naming).

    Returns:
        ExpandedJob ready for execution.
    """
    import tempfile
    from pathlib import Path

    # Import here to avoid circular dependencies
    from joshpy.jobs import ExpandedJob, _compute_run_hash

    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError(
            "jinja2 is required for job templating. Install with: pip install joshpy[jobs]"
        )

    # Load template
    if config.template_path:
        template_dir = config.template_path.parent
        template_name = config.template_path.name
        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
        template = env.get_template(template_name)
    elif config.template_string:
        env = Environment(autoescape=False)
        template = env.from_string(config.template_string)
    else:
        raise ValueError("Either template_path or template_string must be provided")

    # Separate config params from file params
    config_params: dict[str, Any] = {}
    file_mappings = config.file_mappings.copy()
    custom_tags: dict[str, str] = {}

    for key, value in params.items():
        if isinstance(value, dict) and "path" in value and "label" in value:
            # File parameter
            file_path = value["path"]
            file_label = value["label"]
            file_mappings[key] = file_path
            config_params[key] = file_label
            custom_tags[key] = file_label
            if hasattr(file_path, "name"):
                custom_tags[f"{key}_file"] = file_path.name
            else:
                custom_tags[f"{key}_file"] = str(file_path).split("/")[-1]
        else:
            # Config parameter
            config_params[key] = value
            custom_tags[str(key)] = str(value)

    # Render template
    rendered = template.render(**config_params)

    # Compute run_hash
    run_hash = _compute_run_hash(
        josh_path=config.source_path,
        config_content=rendered,
        file_mappings=file_mappings if file_mappings else None,
    )

    # Add run_hash as custom tag
    custom_tags["run_hash"] = run_hash

    # Derive config name from template path (e.g., "adaptive_config.jshc.j2" -> "adaptive_config.jshc")
    # This ensures the config name matches what the Josh file references via "config <name>.xxx"
    if config.template_path:
        # Remove .j2 extension to get .jshc filename
        config_name = config.template_path.stem  # "adaptive_config.jshc"
        if not config_name.endswith(".jshc"):
            config_name = config_name + ".jshc"
    else:
        config_name = "sweep_config.jshc"

    # Write config file to temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="josh_adaptive_"))
    config_subdir = temp_dir / f"trial_{trial_num:04d}_{run_hash}"
    config_subdir.mkdir(parents=True, exist_ok=True)
    config_path = config_subdir / config_name
    config_path.write_text(rendered)

    return ExpandedJob(
        config_content=rendered,
        config_path=config_path,
        config_name=config_name,
        run_hash=run_hash,
        parameters=config_params,
        simulation=config.simulation,
        replicates=config.replicates,
        source_path=config.source_path,
        file_mappings=file_mappings,
        custom_tags=custom_tags,
        upload_source_path=config.upload_source_path,
        upload_config_path=config.upload_config_path,
        upload_data_path=config.upload_data_path,
        output_steps=config.output_steps,
        seed=config.seed,
        crs=config.crs,
        use_float64=config.use_float64,
    )


def _convert_file_mappings(
    file_mappings: dict[str, Any],
) -> dict[str, dict[str, str]] | None:
    """Convert file_mappings from {name: Path} to registry format.

    Args:
        file_mappings: Dict mapping names to Path objects.

    Returns:
        Dict in {name: {path, hash}} format, or None if empty.
    """
    if not file_mappings:
        return None

    from pathlib import Path

    from joshpy.jobs import _hash_file

    result: dict[str, dict[str, str]] = {}
    for name, path in file_mappings.items():
        path = Path(path) if not isinstance(path, Path) else path
        result[name] = {
            "path": str(path),
            "hash": _hash_file(path) if path.exists() else "",
        }
    return result


def _store_study_outcomes(
    registry: RunRegistry,
    session_id: str,
    study: Any,  # optuna.Study
) -> None:
    """Store Optuna study outcomes in registry metadata.

    This enables querying best params/value without Optuna installed.

    Args:
        registry: RunRegistry instance.
        session_id: Session ID to update.
        study: Completed Optuna study.
    """
    import json

    # Build metadata dict
    metadata: dict[str, Any] = {
        "optuna_n_trials": len(study.trials),
        "optuna_direction": study.direction.name,
    }

    if study.best_trial is not None:
        metadata["optuna_best_params"] = study.best_params
        metadata["optuna_best_value"] = study.best_value
        metadata["optuna_best_trial_number"] = study.best_trial.number

    # Get existing session metadata and merge
    session = registry.get_session(session_id)
    if session and session.metadata:
        existing = session.metadata
        existing.update(metadata)
        metadata = existing

    # Update session metadata via direct SQL (registry doesn't have update_session_metadata)
    registry.conn.execute(
        "UPDATE sweep_sessions SET metadata = ? WHERE session_id = ?",
        [json.dumps(metadata), session_id],
    )
