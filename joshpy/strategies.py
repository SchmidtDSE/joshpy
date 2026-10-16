"""Sweep expansion strategies for parameter optimization.

This module defines the strategy pattern for parameter sweep expansion:
- `CartesianStrategy`: Full cartesian product of all parameters (default)
- `OptunaStrategy`: Bayesian optimization using Optuna (requires [adaptive] extra)

The strategy pattern enables different approaches to parameter space exploration
without changing the SweepConfig interface.

Non-adaptive strategies yield every combination from `expand()`. Adaptive
strategies implement `run_adaptive(ctx)`: they own their search loop and call
`ctx.evaluate(rows)` on an `AdaptiveContext`, which expands, registers, runs,
loads and scores each batch of proposed parameter sets.

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
import re
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from joshpy.jobs import (
        CompoundSweepParameter,
        ConfigSweepParameter,
        ExpandedJob,
        FileSweepParameter,
        JobSet,
    )
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


# Exit code to human-readable diagnostic mapping
EXIT_CODE_DIAGNOSTICS: dict[int, tuple[str, str]] = {
    # (short_description, suggestion)
    137: (
        "Process killed (SIGKILL) - likely out of memory",
        "Try reducing grid size, number of replicates, or using --use-float-64 for lower memory usage.",
    ),
    143: (
        "Process terminated (SIGTERM) - likely out of memory or system timeout",
        "Try reducing grid size, number of replicates, or using --use-float-64 for lower memory usage.",
    ),
    # Josh-specific exit codes (from JoshSimCommander)
    1: ("General error", "Check stderr for details."),
    2: ("Invalid arguments", "Check your command-line arguments and configuration."),
    3: ("Compilation error", "Check your .josh script for syntax errors."),
    4: ("Runtime error", "Check stderr for simulation runtime errors."),
    100: ("Missing required argument", "Check required arguments in your configuration."),
    101: ("File not found", "Check that all input files exist and paths are correct."),
    102: ("Invalid configuration", "Check your .jshc configuration file for errors."),
    103: ("Validation failed", "Check your .josh script for validation errors."),
    404: ("Simulation not found", "Check that the simulation name matches one defined in your .josh file."),
}


# Regex for a Java exception header line, e.g.
#   java.lang.IllegalArgumentException: Config value not found: sweep_config.foo
#   Caused by: java.io.IOException: connection reset
_JAVA_EXCEPTION_RE = re.compile(
    r"^(?:Caused by: )?(?P<cls>[\w.$]+(?:Exception|Error)): (?P<msg>.+)$"
)
# Regex for joshpy/JAR "<something> failed: <reason>" lines, e.g.
#   batchRemote failed: Access denied. (exit code 100)
_FAILED_RE = re.compile(r"^\s*\w+ failed: .+", re.IGNORECASE)


def extract_jar_error(stderr: str) -> str | None:
    """Extract the single most meaningful line from JAR stderr.

    JAR failures dump a Java stack trace where the genuinely useful line (the
    exception message) is buried under dozens of stack frames. This lifts that
    line out so it can be surfaced as a diagnostic.

    Args:
        stderr: Raw stderr captured from the JAR subprocess.

    Returns:
        The most informative line (exception class + message, a "... failed: ..."
        line, or the first non-frame line), or None if nothing useful is found.
    """
    if not stderr:
        return None

    lines = stderr.splitlines()

    # 1. Prefer an explicit Java exception header (class + message).
    for line in lines:
        match = _JAVA_EXCEPTION_RE.match(line.strip())
        if match:
            return f"{match.group('cls')}: {match.group('msg')}".strip()

    # 2. Otherwise a "<something> failed: <reason>" line (e.g. batchRemote failed).
    for line in lines:
        if _FAILED_RE.match(line):
            return line.strip()

    # 3. Otherwise the first non-empty line that is not a stack frame.
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith("at ")
            or stripped.startswith("... ")
            or stripped.startswith("Exception in thread")
        ):
            continue
        return stripped

    return None


def get_exit_code_diagnostic(exit_code: int) -> tuple[str, str]:
    """Get human-readable diagnostic for an exit code.

    Args:
        exit_code: The process exit code.

    Returns:
        Tuple of (description, suggestion). Returns generic messages for unknown codes.
    """
    if exit_code in EXIT_CODE_DIAGNOSTICS:
        return EXIT_CODE_DIAGNOSTICS[exit_code]

    # Check for signal-based exit codes (128 + signal_number)
    if exit_code > 128:
        signal_num = exit_code - 128
        return (
            f"Process killed by signal {signal_num}",
            "The process was terminated by a system signal. Check system logs for details.",
        )

    return (
        f"Process exited with code {exit_code}",
        "Check stderr for details.",
    )


class SweepExecutionError(Exception):
    """Raised when a sweep trial fails and stop_on_failure=True.

    This exception provides detailed information about the failed trial,
    including the job parameters, run hash, and stderr output from the
    CLI execution. Exit codes are mapped to human-readable diagnostics
    with actionable suggestions.

    Attributes:
        job: The ExpandedJob that failed.
        result: The CLIResult with exit code and stderr.
        trial_num: Zero-indexed trial number that failed.
        succeeded_before: Number of trials that succeeded before this failure.
        diagnostic: Human-readable description of the failure. Set to the JAR's
            actual error line (lifted from stderr) when available, otherwise the
            exit-code-based description.
        exit_code_diagnostic: The exit-code-based description (always set), kept
            separately so callers can still see the exit-code interpretation.
        suggestion: Actionable suggestion for resolving the error.

    Examples:
        >>> try:
        ...     result = manager.run(objective=my_objective)
        ... except SweepExecutionError as e:
        ...     print(f"Trial {e.trial_num + 1} failed")
        ...     print(f"Diagnostic: {e.diagnostic}")
        ...     print(f"Suggestion: {e.suggestion}")
        ...     print(f"Parameters: {e.job.parameters}")
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

        # Get human-readable diagnostic. Prefer the JAR's actual error line
        # (lifted out of the stack trace) over the generic exit-code guess,
        # which can be misleading when the JAR reuses exit codes.
        code_diagnostic, self.suggestion = get_exit_code_diagnostic(result.exit_code)
        self.exit_code_diagnostic = code_diagnostic
        jar_error = extract_jar_error(getattr(result, "stderr", "") or "")
        self.diagnostic = jar_error or code_diagnostic

        # Build informative message
        message = (
            f"Job {trial_num + 1} failed: {self.diagnostic}\n"
            f"{succeeded_before} job(s) succeeded before this failure.\n"
            f"\n"
            f"Suggestion: {self.suggestion}\n"
            f"\n"
            f"Parameters: {job.parameters}\n"
            f"Run hash: {job.run_hash}\n"
            f"Exit code: {result.exit_code}"
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
    - `expand()` method (for non-adaptive) or `run_adaptive()` (for adaptive)
    - `to_dict()` and `from_dict()` for serialization
    """

    def run_adaptive(self, ctx: AdaptiveContext) -> Any:
        """Run this strategy's search loop (adaptive strategies only).

        The strategy owns its loop and its own state. It proposes parameter
        sets and calls ``ctx.evaluate(params, score=...)`` for each batch;
        joshpy expands, registers, runs, loads and scores them. Called by
        :func:`run_adaptive_sweep`, which manages session status.

        Args:
            ctx: The AdaptiveContext for this run.

        Returns:
            A result object for the run. It should expose ``job_results``,
            ``succeeded``, ``failed``, ``run_ids`` and ``job_set`` (see
            :class:`~joshpy.jobs.AdaptiveSweepResult`) so SweepManager can
            load results after the run.

        Raises:
            NotImplementedError: For non-adaptive strategies.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not adaptive; expand() it and use run_sweep()."
        )

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
        compound_params: list[CompoundSweepParameter] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield parameter combinations to try.

        For non-adaptive strategies, yields all combinations upfront.
        For adaptive strategies, this raises an error - use run_adaptive_sweep() instead.

        Args:
            config_params: Configuration parameters to sweep over.
            file_params: File parameters to sweep over.
            compound_params: Groups of parameters that should co-vary (zipped internally).

        Returns:
            Iterator of dicts, each containing parameter values for one combination.
            Config params: {"name": value}
            File params: {"name": {"path": Path, "label": str}}
            Compound params: {"group_name": label, plus all inner param entries}
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

    Compound parameters are handled specially: their internal parameters are
    zipped together (not cartesian), but the compound as a whole participates
    in the cartesian product with other parameters.

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

        >>> # With compound parameters:
        >>> config = SweepConfig(
        ...     config_parameters=[ConfigSweepParameter(name="a", values=[1, 2])],
        ...     compound_parameters=[
        ...         CompoundSweepParameter(
        ...             name="scenario",
        ...             parameters=[
        ...                 FileSweepParameter(name="temp", paths=["t1.jshd", "t2.jshd"]),
        ...                 FileSweepParameter(name="precip", paths=["p1.jshd", "p2.jshd"]),
        ...             ],
        ...             labels=["low", "high"],
        ...         ),
        ...     ],
        ... )
        >>> # Generates 2 * 2 = 4 combinations (a × scenario), NOT 2 * 2 * 2 = 8
    """

    @property
    def is_adaptive(self) -> bool:
        """Cartesian strategy is not adaptive - all combinations known upfront."""
        return False

    def expand(
        self,
        config_params: list[ConfigSweepParameter],
        file_params: list[FileSweepParameter],
        compound_params: list[CompoundSweepParameter] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Generate cartesian product of all parameter values.

        Args:
            config_params: Configuration parameters to sweep over.
            file_params: File parameters to sweep over.
            compound_params: Groups of parameters that should co-vary.

        Returns:
            Iterator of dicts, one for each parameter combination.
        """
        if compound_params is None:
            compound_params = []

        # Build all param names and value lists for cartesian product
        names: list[str] = []
        value_lists: list[list[Any]] = []
        file_param_names: set[str] = set()
        compound_param_names: set[str] = set()

        # Config parameters
        for cp in config_params:
            names.append(cp.name)
            value_lists.append(cp.values)

        # File parameters
        for fp in file_params:
            names.append(fp.name)
            # (path, label) tuples for file params
            value_lists.append(list(zip(fp.paths, fp.labels)))
            file_param_names.add(fp.name)

        # Compound parameters - each compound becomes one "axis" in the cartesian product
        for compound in compound_params:
            names.append(compound.name)
            compound_param_names.add(compound.name)
            # Build list of zipped scenario dicts
            scenarios = self._expand_compound(compound)
            value_lists.append(scenarios)

        if not names:
            yield {}
            return

        for combo in itertools.product(*value_lists):
            result: dict[str, Any] = {}
            for name, value in zip(names, combo):
                if name in compound_param_names:
                    # Compound: value is a dict with label + all inner params
                    # Merge the compound dict into result
                    result.update(value)
                elif name in file_param_names:
                    path, label = value
                    result[name] = {"path": path, "label": label}
                else:
                    result[name] = value
            yield result

    def _expand_compound(self, compound: CompoundSweepParameter) -> list[dict[str, Any]]:
        """Expand a compound parameter into a list of scenario dicts.

        Each scenario dict contains:
        - The compound's name mapped to its label (for SQL queries)
        - All inner parameters with their values for that scenario index

        Args:
            compound: The compound parameter to expand.

        Returns:
            List of dicts, one per scenario (zipped index).
        """
        from joshpy.jobs import ConfigSweepParameter, FileSweepParameter

        scenarios: list[dict[str, Any]] = []
        n_scenarios = len(compound)

        for i in range(n_scenarios):
            scenario: dict[str, Any] = {}

            # Add the compound label
            assert compound.labels is not None  # Set in __post_init__
            scenario[compound.name] = compound.labels[i]

            # Add each inner parameter's value at this index
            for param in compound.parameters:
                if isinstance(param, ConfigSweepParameter):
                    scenario[param.name] = param.values[i]
                elif isinstance(param, FileSweepParameter):
                    scenario[param.name] = {
                        "path": param.paths[i],
                        "label": param.labels[i],
                    }

            scenarios.append(scenario)

        return scenarios

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
        compound_params: list[CompoundSweepParameter] | None = None,
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

    def run_adaptive(self, ctx: AdaptiveContext) -> Any:
        """Run the Optuna ask/tell loop, one trial per ``ctx.evaluate`` batch.

        Each trial samples parameters, evaluates them as a batch of one, and
        reports the objective back to the study. Failed runs, and runs whose
        results could not be loaded or scored, are reported as ``inf``.

        Args:
            ctx: The AdaptiveContext for this run.

        Returns:
            AdaptiveSweepResult with trial outcomes and best params.
        """
        from joshpy.jobs import AdaptiveSweepResult

        obj_fn = ctx.objective or self.get_objective()
        study = self.create_study()
        sweep = ctx.config.sweep
        n_trials = self.n_trials

        if not ctx.quiet:
            print(f"Running adaptive sweep with {n_trials} trials")
            print(f"  Direction: {self.direction}")
            print(f"  Sampler: {self.sampler}")

        for trial_num in range(n_trials):
            trial = study.ask()
            params = sample_params_from_trial(
                trial,
                sweep.config_parameters,
                sweep.file_parameters,
                sweep.compound_parameters,
            )

            if not ctx.quiet:
                display_params = {
                    k: v["label"] if isinstance(v, dict) and "label" in v else v
                    for k, v in params.items()
                }
                print(f"[{trial_num + 1}/{n_trials}] Params: {display_params}")

            batch = ctx.evaluate([params], score=obj_fn)
            job, result = batch.jobs[0], batch.results[0]

            # Set user attrs BEFORE tell() since the trial is finished after tell()
            trial.set_user_attr("run_hash", job.run_hash)

            if not result.success:
                trial.set_user_attr("error", f"exit_code={result.exit_code}")
                study.tell(trial, float("inf"))
                if not ctx.quiet:
                    print(f"  [FAIL] exit_code={result.exit_code}")
                    if result.stderr:
                        print(f"\n  STDERR:\n{result.stderr}")
                continue

            if 0 in batch.errors:
                trial.set_user_attr("error", batch.errors[0])
                study.tell(trial, float("inf"))
                continue

            metric = batch.values[0]
            trial.set_user_attr("metric", metric)
            trial.set_user_attr("rows_loaded", batch.rows_loaded[0])
            study.tell(trial, metric)

            if not ctx.quiet:
                print(f"  [OK] metric={metric:.4f} (rows={batch.rows_loaded[0]})")

        _store_study_outcomes(ctx.registry, ctx.session_id, study)

        if not ctx.quiet:
            print("\nAdaptive sweep complete:")
            print(f"  Trials: {n_trials} ({ctx.succeeded} succeeded, {ctx.failed} failed)")
            if study.best_trial is not None:
                print(f"  Best value: {study.best_value}")
                print(f"  Best params: {study.best_params}")

        return AdaptiveSweepResult(
            job_results=list(ctx.job_results),
            succeeded=ctx.succeeded,
            failed=ctx.failed,
            run_ids=dict(ctx.run_ids),
            job_set=ctx.job_set(),
            study=study,
            best_params=study.best_params if study.best_trial else None,
            best_value=study.best_value if study.best_trial else None,
        )


def sample_params_from_trial(
    trial: Any,  # optuna.Trial
    config_params: list[ConfigSweepParameter],
    file_params: list[FileSweepParameter],
    compound_params: list[CompoundSweepParameter] | None = None,
) -> dict[str, Any]:
    """Sample parameters from an Optuna trial.

    Called by run_adaptive_sweep() to get parameter suggestions from Optuna.

    Args:
        trial: Optuna trial object.
        config_params: Configuration parameters to sample.
        file_params: File parameters to sample.
        compound_params: Compound parameters to sample (picks one scenario per compound).

    Returns:
        Dict of sampled parameter values, matching the format from expand().
    """
    from joshpy.jobs import ConfigSweepParameter, FileSweepParameter

    if compound_params is None:
        compound_params = []

    params: dict[str, Any] = {}

    for cp in config_params:
        # Use suggest_categorical for discrete values from the sweep
        params[cp.name] = trial.suggest_categorical(cp.name, cp.values)

    for fp in file_params:
        # Files are categorical choices
        idx = trial.suggest_categorical(f"_file_{fp.name}", list(range(len(fp.paths))))
        params[fp.name] = {"path": fp.paths[idx], "label": fp.labels[idx]}

    for compound in compound_params:
        # Compound parameters: select one scenario index
        n_scenarios = len(compound)
        idx = trial.suggest_categorical(f"_compound_{compound.name}", list(range(n_scenarios)))

        # Add the compound's label
        assert compound.labels is not None  # Set in __post_init__
        params[compound.name] = compound.labels[idx]

        # Add each inner parameter's value at this index
        for inner_param in compound.parameters:
            if isinstance(inner_param, ConfigSweepParameter):
                params[inner_param.name] = inner_param.values[idx]
            elif isinstance(inner_param, FileSweepParameter):
                params[inner_param.name] = {
                    "path": inner_param.paths[idx],
                    "label": inner_param.labels[idx],
                }

    return params


@dataclass
class BatchResult:
    """Outcome of evaluating one batch of parameter sets.

    Returned by :meth:`AdaptiveContext.evaluate`. The lists are aligned with
    the requested rows. Rows with identical parameters resolve to the same job
    (same ``run_hash``) and are run once.

    Attributes:
        params: The requested parameter rows.
        jobs: Expanded job per row.
        results: CLIResult per row (shared by rows that share a job).
        values: Score per row (``score(registry, run_hash, job)``), or None
            when the run failed, its results could not be loaded or scored,
            or no score function was given.
        rows_loaded: Rows loaded into ``cell_data`` for each row's job.
        errors: Row index -> message for rows whose run succeeded but whose
            results could not be loaded or scored.
        run_ids: run_hash -> run_id of this batch's execution of that job.

    Examples:
        >>> batch = ctx.evaluate([{"x": 1.0}, {"x": 2.0}], score=my_objective)
        >>> for job, value in zip(batch.jobs, batch.values):
        ...     print(job.run_hash, value)
        >>> batch.failures  # row indices that failed or could not be scored
    """

    params: list[dict[str, Any]] = field(default_factory=list)
    jobs: list[ExpandedJob] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    rows_loaded: list[int] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)
    run_ids: dict[str, str] = field(default_factory=dict)

    @property
    def run_hashes(self) -> list[str]:
        """run_hash per row."""
        return [job.run_hash for job in self.jobs]

    @property
    def failures(self) -> list[int]:
        """Row indices whose run failed or whose results could not be loaded or scored."""
        return [
            i
            for i, result in enumerate(self.results)
            if not result.success or i in self.errors
        ]

    def __len__(self) -> int:
        return len(self.jobs)


def _adaptive_config_name(config: Any) -> str:
    """Default config namespace for adaptive sweeps when ``config_name`` is unset.

    Adaptive sweeps have always named the rendered config after the template
    file (``adaptive_config.jshc.j2`` -> ``config adaptive_config.*``), while
    batch sweeps default to ``sweep_config``. Both defaults are relied on by
    existing models, so each path keeps its own; an explicit
    ``JobConfig.config_name`` wins on both.
    """
    if config.template_path is not None:
        stem = Path(config.template_path).stem
        return stem[: -len(".jshc")] if stem.endswith(".jshc") else stem
    return "sweep_config"


@dataclass
class AdaptiveContext:
    """What an adaptive strategy runs its loop against.

    Built by :func:`run_adaptive_sweep` and passed to
    :meth:`SweepStrategy.run_adaptive`. The strategy owns its loop and its own
    in-memory state (an Optuna study, an sbi sampler, ...). The context owns
    everything joshpy does, through one method, :meth:`evaluate`: expand a
    batch of parameter sets, register each job, run them with
    :func:`~joshpy.jobs.run_sweep` in the configured execution mode (local,
    Josh Cloud, or batch remote), load their results into the registry, and
    score them.

    Every evaluated job is registered and recorded, whether or not the
    strategy later uses its result.

    Attributes:
        cli: JoshCLI used for execution.
        config: The JobConfig being swept (its ``sweep.strategy`` is the
            adaptive strategy).
        registry: Registry every job is recorded in.
        session_id: Registry session for this run.
        objective: Optional per-run score function override, passed through
            from ``SweepManager.run(objective=...)``.
        job_results: (job, CLIResult) for every evaluation, in order.
        run_ids: run_hash -> run_id of its latest execution.
        succeeded: Number of evaluations that ran successfully.
        failed: Number of evaluations that failed.
    """

    cli: Any  # JoshCLI
    config: Any  # JobConfig
    registry: RunRegistry
    session_id: str
    objective: ObjectiveFn | None = None
    remote: bool = False
    api_key: str | None = None
    endpoint: str | None = None
    batch_remote: bool = False
    target: str | None = None
    batch_no_wait: bool = False
    poll_interval: int = 10
    batch_timeout: int | None = None
    collision_policy: str = "fail"
    on_complete: Callable[[Any, Any], None] | None = None
    quiet: bool = False
    load_config: Any | None = None  # LoadConfig
    stop_on_failure: bool = True
    jfr: Any | None = None  # JfrConfig
    enable_profiler: bool = False
    stream_output: bool = False

    job_results: list[tuple[ExpandedJob, Any]] = field(default_factory=list)
    run_ids: dict[str, str] = field(default_factory=dict)
    succeeded: int = 0
    failed: int = 0

    _jobs: dict[str, ExpandedJob] = field(default_factory=dict, repr=False)
    _work_dir: Path | None = field(default=None, repr=False)
    _expansion_config: Any | None = field(default=None, repr=False)
    _export_paths: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)

    @property
    def work_dir(self) -> Path:
        """Directory every batch is expanded into (created on first use).

        One directory for the whole run keeps a templated or multi-file
        model's rendered entry at a single path, so all evaluated jobs share
        one ``source_path`` (``recover_sweep_results`` requires that).
        """
        if self._work_dir is None:
            self._work_dir = Path(tempfile.mkdtemp(prefix="josh_adaptive_"))
        return self._work_dir

    def evaluate(
        self,
        params: list[dict[str, Any]],
        *,
        score: Callable[[RunRegistry, str, ExpandedJob], Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> BatchResult:
        """Run a batch of parameter sets and return their results.

        Each row is expanded through :class:`~joshpy.jobs.JobExpander` (the
        same rendering and run hashing as a batch sweep), registered, and run
        as one :func:`~joshpy.jobs.run_sweep` call. Successful runs are loaded
        into ``cell_data`` and, if *score* is given, scored.

        Args:
            params: Parameter rows, shaped like ``SweepConfig.expand()`` output.
            score: Optional ``(registry, run_hash, job) -> Any`` evaluated for
                each successfully loaded job (an objective, or a dict of
                summary statistics).
            attributes: Extra run-level attributes recorded on this batch's
                jobs at registration (e.g. ``{"generation": 3}``).

        Returns:
            BatchResult aligned with *params*.

        Raises:
            SweepExecutionError: If ``stop_on_failure`` is set and a run
                failed. Raised after the whole batch has run and the
                successful runs have been loaded.
        """
        from joshpy.cli import CLIResult
        from joshpy.jobs import JobExpander, JobSet, run_sweep
        from joshpy.sweep import LoadConfig, apply_sweep_labels, register_job

        rows = [dict(p) for p in params]
        if not rows:
            return BatchResult()

        config = self._config_for_expansion()
        row_jobs = JobExpander().expand(
            config, output_dir=self.work_dir, combinations=rows
        ).jobs
        apply_sweep_labels(row_jobs, config.label)
        if attributes:
            for job in row_jobs:
                job.attributes.update(attributes)

        # Rows with identical parameters share a run_hash; run each job once.
        unique: dict[str, ExpandedJob] = {}
        for job in row_jobs:
            unique.setdefault(job.run_hash, job)
        batch_jobs = list(unique.values())

        for job in batch_jobs:
            register_job(self.registry, self.session_id, job)

        show_progress = not self.quiet and len(batch_jobs) > 1
        done = 0

        def _progress(job: ExpandedJob, result: Any) -> None:
            nonlocal done
            done += 1
            if show_progress:
                status = "OK  " if result.success else "FAIL"
                print(f"  [{done}/{len(batch_jobs)}] {status} {job.run_hash}")
            if self.on_complete is not None:
                self.on_complete(job, result)

        if show_progress:
            print(f"Evaluating {len(batch_jobs)} parameter sets")

        evaluated_before = self.succeeded + self.failed
        succeeded_before = self.succeeded
        sweep_result = run_sweep(
            self.cli,
            JobSet(jobs=batch_jobs),
            registry=self.registry,
            session_id=self.session_id,
            manage_status=False,
            remote=self.remote,
            api_key=self.api_key,
            endpoint=self.endpoint,
            batch_remote=self.batch_remote,
            target=self.target,
            batch_no_wait=self.batch_no_wait,
            poll_interval=self.poll_interval,
            batch_timeout=self.batch_timeout,
            # Loaded explicitly below in every mode, not just batch remote.
            auto_ingest=False,
            collision_policy=self.collision_policy,
            on_complete=_progress,
            stop_on_failure=False,
            quiet=True,
            jfr=self.jfr,
            enable_profiler=self.enable_profiler,
            stream_output=self.stream_output,
        )

        result_by_hash = {job.run_hash: result for job, result in sweep_result.job_results}
        for job in batch_jobs:
            result = result_by_hash.get(job.run_hash)
            if result is None:
                result = CLIResult(
                    exit_code=1,
                    stdout="",
                    stderr="No result was recorded for this job.",
                    command=[],
                )
                result_by_hash[job.run_hash] = result
            self.job_results.append((job, result))
            self._jobs.setdefault(job.run_hash, job)
            if result.success:
                self.succeeded += 1
            else:
                self.failed += 1
        self.run_ids.update(sweep_result.run_ids)

        # Load and score the successful runs.
        load_config = self.load_config or LoadConfig()
        succeeded_jobs = [j for j in batch_jobs if result_by_hash[j.run_hash].success]
        if succeeded_jobs:
            time.sleep(load_config.settle_delay)

        rows_by_hash: dict[str, int] = {}
        values_by_hash: dict[str, Any] = {}
        errors_by_hash: dict[str, str] = {}
        for k, job in enumerate(succeeded_jobs):
            try:
                rows_by_hash[job.run_hash] = self._load(
                    job,
                    sweep_result.run_ids.get(job.run_hash),
                    load_config,
                    succeeded_before=succeeded_before + k,
                )
            except Exception as e:
                errors_by_hash[job.run_hash] = str(e)
                if not self.quiet:
                    print(f"  [WARN] Result loading failed: {e}")
                continue

            if score is None:
                continue
            try:
                values_by_hash[job.run_hash] = score(self.registry, job.run_hash, job)
            except Exception as e:
                errors_by_hash[job.run_hash] = f"objective_error: {e}"
                if not self.quiet:
                    print(f"  [WARN] Objective evaluation failed: {e}")

        if self.stop_on_failure:
            for k, job in enumerate(batch_jobs):
                result = result_by_hash[job.run_hash]
                if not result.success:
                    raise SweepExecutionError(
                        job=job,
                        result=result,
                        trial_num=evaluated_before + k,
                        succeeded_before=succeeded_before + sum(
                            result_by_hash[j.run_hash].success for j in batch_jobs[:k]
                        ),
                    )

        return BatchResult(
            params=rows,
            jobs=row_jobs,
            results=[result_by_hash[j.run_hash] for j in row_jobs],
            values=[values_by_hash.get(j.run_hash) for j in row_jobs],
            rows_loaded=[rows_by_hash.get(j.run_hash, 0) for j in row_jobs],
            errors={
                i: errors_by_hash[j.run_hash]
                for i, j in enumerate(row_jobs)
                if j.run_hash in errors_by_hash
            },
            run_ids={
                h: sweep_result.run_ids[h] for h in unique if h in sweep_result.run_ids
            },
        )

    def job_set(self) -> JobSet:
        """The jobs evaluated so far (one per run_hash), owning the work directory."""
        from joshpy.jobs import JobSet

        return JobSet(
            jobs=list(self._jobs.values()),
            temp_dir=self._work_dir,
            template_path=self.config.template_path,
        )

    def cleanup(self) -> None:
        """Remove the work directory."""
        if self._work_dir is not None and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None

    def _config_for_expansion(self) -> Any:
        """The JobConfig every batch expands from, resolved once per run.

        Discovers a multi-file model's import closure once (instead of one
        ``inspect-imports`` JVM call per batch) and applies the adaptive
        config-name default.
        """
        if self._expansion_config is not None:
            return self._expansion_config

        from joshpy.jobs import (
            _entry_has_imports,
            _render_entry,
            _source_root,
            discover_import_files,
        )

        config = self.config
        if not config.import_files and _source_root(config) is not None:
            with tempfile.TemporaryDirectory(prefix="josh_adaptive_entry_") as tmp:
                entry = _render_entry(config, Path(tmp))
                has_imports = entry is not None and _entry_has_imports(entry)
            if has_imports:
                config = replace(config, import_files=discover_import_files(self.cli, config))
        if config.config_name is None:
            config = replace(config, config_name=_adaptive_config_name(config))

        self._expansion_config = config
        return config

    def _load(
        self,
        job: ExpandedJob,
        run_id: str | None,
        load_config: Any,
        *,
        succeeded_before: int,
    ) -> int:
        """Load one job's results into ``cell_data``; returns rows loaded."""
        from joshpy.cli import InspectExportsConfig
        from joshpy.sweep import load_job_results

        if run_id is None:
            raise RuntimeError(f"No run was recorded for {job.run_hash}")
        if job.source_path is None:
            raise RuntimeError(f"Job {job.run_hash} has no source_path to inspect exports")

        key = (str(job.source_path), job.simulation)
        if key not in self._export_paths:
            self._export_paths[key] = self.cli.inspect_exports(
                InspectExportsConfig(script=job.source_path, simulation=job.simulation)
            )
        return load_job_results(
            self.cli,
            job,
            self.registry,
            self._export_paths[key],
            run_id=run_id,
            quiet=True,
            load_config=load_config,
            succeeded_before=succeeded_before,
        )


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
    batch_remote: bool = False,
    target: str | None = None,
    batch_no_wait: bool = False,
    poll_interval: int = 10,
    batch_timeout: int | None = None,
    auto_ingest: bool = True,
    collision_policy: str = "fail",
    on_complete: Callable[[Any, Any], None] | None = None,
    quiet: bool = False,
    load_config: Any | None = None,  # LoadConfig
    stop_on_failure: bool = True,
    jfr: Any | None = None,  # JfrConfig
    enable_profiler: bool = False,
    stream_output: bool = False,
) -> Any:
    """Run an adaptive sweep: the strategy runs its own loop against joshpy.

    Builds an :class:`AdaptiveContext` and hands it to
    ``config.sweep.strategy.run_adaptive(ctx)``. The strategy proposes
    parameter sets and calls ``ctx.evaluate(...)`` for each batch; joshpy
    expands, registers, runs, loads and scores them. For
    :class:`OptunaStrategy` each trial is a batch of one.

    Session status is set to "running", then "completed" (or "failed" if any
    evaluation failed or the strategy raised).

    Args:
        cli: JoshCLI instance to use for execution.
        config: JobConfig whose ``sweep.strategy`` is adaptive.
        registry: RunRegistry for tracking (required).
        session_id: Session ID in registry (required).
        objective: Optional per-run score function override. If not provided,
            the strategy uses its own (e.g. ``OptunaStrategy.objective``).
        remote: If True, use run_remote() for Josh Cloud execution.
        api_key: API key for remote execution (optional for local servers).
        endpoint: Custom endpoint URL for remote execution.
        batch_remote: If True, run each batch through batchRemote.
        target: Target profile name (required if batch_remote=True).
        batch_no_wait: If True, dispatch a batch with ``--no-wait`` then poll.
        poll_interval: Seconds between polls in no-wait mode.
        batch_timeout: Per-job timeout in seconds for no-wait polling.
        auto_ingest: Accepted for signature compatibility. Adaptive sweeps
            always load results, since the strategy needs them.
        collision_policy: Collision policy for each batch (see ``run_sweep``).
        on_complete: Optional callback invoked after each job completes.
        quiet: If True, suppress progress output.
        load_config: LoadConfig for result loading behavior.
        stop_on_failure: If True (default), raise SweepExecutionError when a
            run fails (after its batch finishes). If False, failed runs are
            reported to the strategy and the sweep continues.
        jfr: Optional JFR profiling configuration (one recording per job).
        enable_profiler: Enable Josh evaluation profiler (--enable-profiler).
        stream_output: Stream JAR stdout/stderr to the terminal.

    Returns:
        The strategy's result (AdaptiveSweepResult for OptunaStrategy). It
        carries ``run_ids`` and a ``job_set`` owning the work directory; call
        ``result.job_set.cleanup()`` when done (SweepManager does this).

    Raises:
        ValueError: If config.sweep is missing, or batch_remote without target.
        TypeError: If config.sweep.strategy is not adaptive.
        SweepExecutionError: If stop_on_failure=True and a run fails.

    Examples:
        >>> from joshpy.strategies import run_adaptive_sweep, OptunaStrategy
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
        >>> registry = RunRegistry("experiment.duckdb")
        >>> session_id = registry.create_session(config=config)
        >>> result = run_adaptive_sweep(
        ...     JoshCLI(), config,
        ...     registry=registry,
        ...     session_id=session_id,
        ...     objective=stability_metric,
        ... )
        >>> print(f"Best params: {result.best_params}")
    """
    if config.sweep is None:
        raise ValueError("config.sweep is required for adaptive sweeps")

    strategy = config.sweep.strategy
    if strategy is None or not strategy.is_adaptive:
        raise TypeError(
            "run_adaptive_sweep requires OptunaStrategy or another adaptive strategy, "
            f"got {type(strategy).__name__}. "
            "For non-adaptive strategies, use run_sweep() instead."
        )
    if batch_remote and target is None:
        raise ValueError("target is required when batch_remote=True")

    ctx = AdaptiveContext(
        cli=cli,
        config=config,
        registry=registry,
        session_id=session_id,
        objective=objective,
        remote=remote,
        api_key=api_key,
        endpoint=endpoint,
        batch_remote=batch_remote,
        target=target,
        batch_no_wait=batch_no_wait,
        poll_interval=poll_interval,
        batch_timeout=batch_timeout,
        collision_policy=collision_policy,
        on_complete=on_complete,
        quiet=quiet,
        load_config=load_config,
        stop_on_failure=stop_on_failure,
        jfr=jfr,
        enable_profiler=enable_profiler,
        stream_output=stream_output,
    )

    registry.update_session_status(session_id, "running")
    try:
        result = strategy.run_adaptive(ctx)
    except Exception:
        registry.update_session_status(session_id, "failed")
        ctx.cleanup()
        raise

    registry.update_session_status(session_id, "completed" if ctx.failed == 0 else "failed")
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
