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

import functools
import importlib
import itertools
import re
import shutil
import sys
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
        path: Import path in "module:function" format, or "module:Class.attribute"
            for a function on a module-level class (e.g. a staticmethod).

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
    fn: Any = importlib.import_module(module_path)
    for attribute in func_name.split("."):
        fn = getattr(fn, attribute)
    if not callable(fn):
        raise ValueError(f"{path} is not callable")
    return fn  # type: ignore[no-any-return]


def _callable_path(fn: Any) -> str | None:
    """The "module:qualname" path that loads *fn* back, or None if there is none.

    A path is returned only when importing it would give back *fn* itself: a
    module-level function, or a function on a module-level class. Lambdas,
    nested functions and closures, bound methods, partials and callable
    instances have no such path, and neither does anything defined in
    ``__main__`` (a script, notebook or docs page), which another process
    cannot import.
    """
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module_name or not qualname or module_name == "__main__" or "<" in qualname:
        return None
    target: Any = sys.modules.get(module_name)
    try:
        for attribute in qualname.split("."):
            target = getattr(target, attribute)
        same = target is fn or bool(target == fn)  # == matches classmethods, rebound each access
    except Exception:
        return None
    return f"{module_name}:{qualname}" if same else None


def _callable_name(fn: Any) -> str:
    """A readable "module:qualname" name for *fn*, for provenance only."""
    if isinstance(fn, functools.partial):
        return f"functools.partial({_callable_name(fn.func)})"
    module_name = getattr(fn, "__module__", None) or type(fn).__module__
    qualname = getattr(fn, "__qualname__", None) or f"{type(fn).__qualname__} instance"
    return f"{module_name}:{qualname}"


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
        seed: Optional sampler seed. A seeded study proposes the same trials
            given the same results, so rerunning an interrupted sweep against
            the same registry reuses its completed runs.

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
    seed: int | None = None  # sampler seed; with stored results, replays reach the same state

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

        seed = self.seed
        if self.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        elif self.sampler == "gp":
            # GP sampler requires optuna-integration or specific setup
            # Fall back to TPE with a warning
            try:
                from optuna_integration import BoTorchSampler

                return BoTorchSampler(seed=seed)
            except ImportError:
                import warnings

                warnings.warn(
                    "GP sampler requires optuna-integration[botorch]. Falling back to TPE."
                )
                return optuna.samplers.TPESampler(seed=seed)
        elif self.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=seed)
        elif self.sampler == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        else:
            return optuna.samplers.TPESampler(seed=seed)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        A callable objective is saved as its "module:qualname" path when that
        path loads it back. Otherwise (a lambda, closure, bound method, or a
        function defined in ``__main__``) its name is kept under
        ``objective_not_importable``, for provenance only, and ``from_dict``
        leaves the objective to be provided at runtime.
        """
        result: dict[str, Any] = {
            "type": "optuna",
            "n_trials": self.n_trials,
            "direction": self.direction,
            "sampler": self.sampler,
        }
        if self.seed is not None:
            result["seed"] = self.seed

        if isinstance(self.objective, str):
            result["objective"] = self.objective
        elif self.objective is not None:
            path = _callable_path(self.objective)
            if path is not None:
                result["objective"] = path
            else:
                result["objective_not_importable"] = _callable_name(self.objective)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptunaStrategy:
        """Create from dict (YAML/JSON deserialization)."""
        return cls(
            n_trials=data.get("n_trials", 100),
            direction=data.get("direction", "minimize"),
            sampler=data.get("sampler", "tpe"),
            objective=data.get("objective"),
            seed=data.get("seed"),
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


# Summary-statistics function for SbiStrategy:
# (registry, run_hash, job) -> {"summary_name": value, ...}
SummaryFn = Callable[["RunRegistry", str, "ExpandedJob"], dict[str, float]]

_SBI_METHODS = ("smcabc", "mcabc")
_SBI_PRIOR_KINDS = ("uniform", "normal")
# Options that make sbi return something other than the simulated particles
# (a KDE object, or regression-adjusted parameters that were never run).
_SBI_UNSUPPORTED_KWARGS = ("kde", "kde_kwargs", "kde_sample_weights", "lra", "lra_with_weights",
                           "sass", "sass_fraction", "sass_expansion_degree", "return_summary")


def _check_sbi() -> None:
    """Raise ImportError if sbi (and torch) are not available."""
    try:
        import sbi  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "sbi is required for SbiStrategy. Install with: pip install joshpy[sbi] "
            "(for CPU-only torch, first: pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu)"
        ) from None


def _sbi_param_value(value: float) -> float:
    """Normalize an sbi draw (a float32) to its shortest round-tripping decimal.

    The rendered ``.jshc`` then reads ``1.0372891`` rather than the float64
    expansion of the float32, and the same draw always renders (and hashes)
    the same way, which replay-based resume relies on.
    """
    import numpy as np

    return float(np.format_float_positional(np.float32(value), unique=True, trim="-"))


def _count_generation_simulations(
    populations: list[dict[str, Any]],
    call_of: dict[str, int],
    call_sizes: list[int],
) -> None:
    """Add ``simulations`` and ``carried_over`` to each population, in place.

    sbi simulates generations in order, each over one or more simulator calls,
    and a generation ends with the call that accepted its last particle. So a
    generation's calls run from the call after the previous generation's last
    one to the latest call among its own new particles; the final generation
    also owns any calls after that (a budget that ran out mid-generation).
    Particles from earlier calls were carried over from the previous population.
    """
    boundary = -1  # last simulator call of the previous generation
    for i, pop in enumerate(populations):
        calls = [call_of.get(h, -1) if h is not None else -1 for h in pop["run_hashes"]]
        new = [c for c in calls if c > boundary]
        last = len(call_sizes) - 1 if i == len(populations) - 1 else max(new, default=boundary)
        pop["simulations"] = sum(call_sizes[boundary + 1 : last + 1])
        pop["carried_over"] = len(calls) - len(new)
        boundary = last


@dataclass
class SbiResult:
    """Results from an SbiStrategy run.

    Attributes:
        job_results: (ExpandedJob, CLIResult) for every simulation, in order.
        succeeded: Number of simulations that ran successfully.
        failed: Number of simulations that failed.
        run_ids: run_hash -> run_id of its latest execution.
        job_set: The evaluated jobs (one per run_hash), owning the work dir.
        method: The sbi method that ran ("smcabc" or "mcabc").
        param_names: Parameter order of ``samples`` columns.
        samples: Accepted parameter sets of the final population, a
            ``torch.Tensor`` of shape ``[N, len(param_names)]``.
        weights: Normalized importance weights of ``samples`` (uniform for
            MCABC), a ``torch.Tensor`` of shape ``[N]``.
        run_hashes: The run each sample came from (registry join key).
        populations: Per generation: ``generation``, ``epsilon`` (the target
            tolerance), ``run_hashes``, ``weights``, ``distances``,
            ``simulations`` (simulator evaluations the generation used, reused
            runs included) and ``carried_over`` (particles taken from the
            previous population: sbi fills the last population this way when
            the budget runs out mid-generation, so their distances may exceed
            ``epsilon``). One population for MCABC.
        observed: The observed summary statistics.
        summary: sbi's raw ``return_summary`` dictionary.
        load_errors: run_hash -> message, for runs that succeeded but whose
            results could not be loaded. Their summaries were NaN, so sbi
            never accepted them; they are not counted in ``failed``.

    Examples:
        >>> result = manager.run()
        >>> df = result.to_dataframe()   # one row per accepted particle
        >>> df.groupby("run_hash").weight.sum()
    """

    job_results: list[tuple[ExpandedJob, Any]] = field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    run_ids: dict[str, str] = field(default_factory=dict)
    job_set: Any = None  # JobSet
    method: str = "smcabc"
    param_names: list[str] = field(default_factory=list)
    samples: Any = None  # torch.Tensor
    weights: Any = None  # torch.Tensor
    run_hashes: list[str | None] = field(default_factory=list)
    populations: list[dict[str, Any]] = field(default_factory=list)
    observed: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    load_errors: dict[str, str] = field(default_factory=dict)

    @property
    def is_adaptive(self) -> bool:
        """True - this result came from an adaptive strategy."""
        return True

    def to_dataframe(self) -> Any:
        """Final population as a DataFrame: one column per parameter, plus
        ``weight`` and ``run_hash``.

        Parameter values are those the runs were rendered with (each float32
        draw's shortest decimal, as in ``job.parameters``), so running a
        particle's parameters again reproduces its ``run_hash``.
        """
        import pandas as pd

        if self.samples is None:
            return pd.DataFrame(columns=[*self.param_names, "weight", "run_hash"])
        df = pd.DataFrame(
            [[_sbi_param_value(v) for v in row] for row in self.samples.tolist()],
            columns=self.param_names,
        )
        df["weight"] = self.weights.tolist()
        df["run_hash"] = self.run_hashes
        return df

    def __iter__(self) -> Iterator[tuple[ExpandedJob, Any]]:
        """Iterate over (job, result) tuples."""
        return iter(self.job_results)

    def __len__(self) -> int:
        """Return total number of job results."""
        return len(self.job_results)


@register_strategy("sbi")
@dataclass
class SbiStrategy(SweepStrategy):
    """Approximate Bayesian computation with the sbi package.

    The search is sbi's own: rejection ABC (``"mcabc"``) or sequential Monte
    Carlo ABC (``"smcabc"``, Toni et al. 2009; Beaumont et al. 2009). joshpy
    supplies the simulator: each batch of parameter sets sbi proposes becomes
    one :meth:`AdaptiveContext.evaluate` call, so every simulation is
    registered, run in the configured mode (local, remote, batch remote),
    loaded, and summarized with ``summaries``. joshpy implements no statistics.

    Requires the ``sbi`` extra (``pip install joshpy[sbi]``).

    Attributes:
        prior: Parameter -> distribution, in the order sbi sees them, e.g.
            ``{"maxNewGrowth": {"uniform": [0.2, 2.0]}}``. Supported:
            ``{"uniform": [low, high]}`` and ``{"normal": [mean, std]}``.
            Parameter names are the template variables.
        summaries: ``(registry, run_hash, job) -> dict[str, float]``, or a
            ``"module:function"`` path. Returns the summary statistics of one
            run; its keys must cover ``observed``. Return NaN statistics to
            reject a run. An exception stops the search once its batch has
            finished (it usually means a bug); after a fix, a rerun reuses
            the completed runs.
        observed: Observed summary statistics (dict), a ``"module:function"``
            path returning that dict, or a path to a JSON file. Its key order
            fixes the order of the summary vector.
        method: ``"smcabc"`` (default) or ``"mcabc"``.
        method_kwargs: Passed to the sbi call, e.g. for SMC-ABC
            ``num_particles``, ``num_initial_pop``, ``num_simulations``,
            ``epsilon_decay``; for MC-ABC ``num_simulations`` and ``quantile``
            or ``eps``. KDE/LRA/SASS options are not supported (they return
            parameters that were never simulated).
        distance: sbi distance name (default ``"l2"``). Standardize your
            summaries so no single statistic dominates.
        seed: Seeds torch and numpy before sbi runs. With the same seed, sbi
            and torch versions, rerunning an interrupted sweep against the same
            registry replays the same proposals and reuses completed runs.

    Examples:
        >>> strategy = SbiStrategy(
        ...     prior={
        ...         "maxNewGrowth": {"uniform": [0.2, 2.0]},
        ...         "minPrecipImpactPct": {"uniform": [0.0, 100.0]},
        ...     },
        ...     summaries="my_project.summaries:final_height",
        ...     observed={"mean_height": 7.1, "height_cv": 0.31},
        ...     method_kwargs={"num_particles": 50, "num_initial_pop": 100,
        ...                    "num_simulations": 600, "epsilon_decay": 0.5},
        ... )
        >>> config = JobConfig(..., sweep=SweepConfig(strategy=strategy))
        >>> result = SweepManager.builder(config).with_registry("abc.duckdb").build().run()
        >>> result.to_dataframe()
    """

    prior: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    summaries: SummaryFn | str | None = None
    observed: dict[str, float] | str | None = None
    method: str = "smcabc"
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    distance: str = "l2"
    seed: int = 0

    _summaries_fn: SummaryFn | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the method, prior spec and sbi options."""
        if self.method not in _SBI_METHODS:
            raise ValueError(f"Invalid method '{self.method}'. Must be one of: {_SBI_METHODS}")
        if not self.prior:
            raise ValueError("SbiStrategy needs a prior with at least one parameter")
        for name, spec in self.prior.items():
            if not isinstance(spec, dict) or len(spec) != 1:
                raise ValueError(
                    f"Prior for '{name}' must be a one-key dict like "
                    "{'uniform': [low, high]} or {'normal': [mean, std]}"
                )
            ((kind, args),) = spec.items()
            if kind not in _SBI_PRIOR_KINDS:
                raise ValueError(
                    f"Unsupported prior '{kind}' for '{name}'. Supported: {_SBI_PRIOR_KINDS}"
                )
            if len(args) != 2:
                raise ValueError(f"Prior '{kind}' for '{name}' takes two numbers, got {args}")
            a, b = (float(v) for v in args)
            if kind == "uniform" and not a < b:
                raise ValueError(f"Uniform prior for '{name}' needs low < high, got {args}")
            if kind == "normal" and not b > 0:
                raise ValueError(f"Normal prior for '{name}' needs std > 0, got {args}")
        unsupported = sorted(set(self.method_kwargs) & set(_SBI_UNSUPPORTED_KWARGS))
        if unsupported:
            raise ValueError(
                f"Unsupported sbi options {unsupported}: SbiStrategy returns the simulated "
                "particles; fit a KDE or regression adjustment on result.samples instead."
            )
        if callable(self.summaries) and not isinstance(self.summaries, str):
            self._summaries_fn = self.summaries

    @property
    def is_adaptive(self) -> bool:
        """SbiStrategy is adaptive - sbi proposes parameters from results."""
        return True

    @property
    def param_names(self) -> list[str]:
        """Parameter names, in the order of sbi's parameter vector."""
        return list(self.prior)

    def expand(
        self,
        config_params: list[ConfigSweepParameter],
        file_params: list[FileSweepParameter],
        compound_params: list[CompoundSweepParameter] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Not supported: sbi proposes parameters as the run progresses.

        Raises:
            RuntimeError: Always - use run_adaptive_sweep() or SweepManager.run().
        """
        raise RuntimeError(
            "SbiStrategy is adaptive - use run_adaptive_sweep() or SweepManager.run() "
            "instead of expand()."
        )

    def get_summaries(self) -> SummaryFn:
        """The summary-statistics function, resolving a "module:function" path."""
        if self._summaries_fn is None and isinstance(self.summaries, str):
            self._summaries_fn = _load_objective_from_path(self.summaries)  # type: ignore[assignment]
        if self._summaries_fn is None:
            raise ValueError(
                "No summaries function configured for SbiStrategy. Provide summaries= "
                "in the constructor or pass objective= to SweepManager.run()."
            )
        return self._summaries_fn

    def get_observed(self) -> dict[str, float]:
        """The observed summary statistics, resolving a path or "module:function"."""
        observed: Any = self.observed
        if isinstance(observed, str):
            if ":" in observed and not Path(observed).exists():
                observed_fn: Callable[[], Any] = _load_objective_from_path(observed)  # type: ignore[assignment]
                observed = observed_fn()
            else:
                import json

                observed = json.loads(Path(observed).read_text())
        if not isinstance(observed, dict) or not observed:
            raise ValueError("SbiStrategy needs observed summary statistics (a non-empty dict)")
        return {str(k): float(v) for k, v in observed.items()}

    def build_prior(self) -> Any:
        """The prior as a torch distribution sbi accepts.

        All-uniform priors become ``sbi.utils.BoxUniform``; mixed priors become
        ``sbi.utils.MultipleIndependent`` of per-parameter distributions.
        """
        _check_sbi()
        import torch
        from sbi.utils import BoxUniform, MultipleIndependent

        specs = [next(iter(spec.items())) for spec in self.prior.values()]
        if all(kind == "uniform" for kind, _ in specs):
            low = torch.tensor([float(a) for _, (a, _) in specs])
            high = torch.tensor([float(b) for _, (_, b) in specs])
            return BoxUniform(low=low, high=high)

        dists: list[Any] = []
        for kind, (a, b) in specs:
            a_t, b_t = torch.tensor([float(a)]), torch.tensor([float(b)])
            if kind == "uniform":
                dists.append(torch.distributions.Uniform(a_t, b_t))
            else:
                dists.append(torch.distributions.Normal(a_t, b_t))
        return MultipleIndependent(dists)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        A callable ``summaries`` is saved as its "module:qualname" path when
        that path loads it back. Otherwise (a lambda, closure, bound method,
        or a function defined in ``__main__``) its name is kept under
        ``summaries_not_importable``, for provenance only, and ``from_dict``
        leaves ``summaries`` unset.
        """
        result: dict[str, Any] = {
            "type": "sbi",
            "method": self.method,
            "prior": self.prior,
            "distance": self.distance,
            "seed": self.seed,
        }
        if self.method_kwargs:
            result["method_kwargs"] = self.method_kwargs
        if isinstance(self.summaries, str):
            result["summaries"] = self.summaries
        elif self.summaries is not None:
            path = _callable_path(self.summaries)
            if path is not None:
                result["summaries"] = path
            else:
                result["summaries_not_importable"] = _callable_name(self.summaries)
        if self.observed is not None:
            result["observed"] = self.observed
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SbiStrategy:
        """Create from dict (YAML/JSON deserialization)."""
        return cls(
            prior=data.get("prior", {}),
            summaries=data.get("summaries"),
            observed=data.get("observed"),
            method=data.get("method", "smcabc"),
            method_kwargs=data.get("method_kwargs", {}),
            distance=data.get("distance", "l2"),
            seed=data.get("seed", 0),
        )

    def run_adaptive(self, ctx: AdaptiveContext) -> SbiResult:
        """Run sbi with joshpy as the simulator.

        Each simulator call (one batch of proposals) is one ``ctx.evaluate``.
        A run that fails, or whose results cannot be loaded, returns NaN
        summaries, which sbi never accepts; load errors are reported in
        ``SbiResult.load_errors`` and a warning. An exception from
        ``summaries``, or SMC-ABC's initial population having fewer than
        ``num_particles`` usable runs, stops the search. Afterwards each
        population member gets ``sbi_generation``, ``sbi_weight``,
        ``sbi_distance`` and ``sbi_accepted`` run attributes, and the session
        metadata gets the settings, versions and per-generation populations.

        Args:
            ctx: The AdaptiveContext for this run.

        Returns:
            SbiResult with the final population.
        """
        _check_sbi()
        import numpy as np
        import torch
        from sbi.inference import MCABC, SMCABC

        summaries_fn = ctx.objective or self.get_summaries()
        observed = self.get_observed()
        keys = list(observed)
        names = self.param_names
        prior = self.build_prior()
        x_o = torch.tensor([[observed[k] for k in keys]], dtype=torch.float32)
        settings = self._session_settings(observed)
        self._warn_on_shared_seed(ctx)
        self._warn_on_version_drift(ctx, settings)
        ctx.registry.update_session_metadata(ctx.session_id, sbi=settings)

        hash_by_theta: dict[tuple[float, ...], str] = {}
        call_of: dict[str, int] = {}  # run_hash -> simulator call that ran it
        call_sizes: list[int] = []
        n_calls = 0

        load_errors: dict[str, str] = {}  # run_hash -> why its results could not be loaded

        def simulator(theta: Any) -> Any:
            nonlocal n_calls
            rows = [
                {name: _sbi_param_value(v) for name, v in zip(names, row, strict=True)}
                for row in theta.tolist()
            ]
            batch = ctx.evaluate(rows, score=summaries_fn, attributes={"sbi_call": n_calls})
            call_sizes.append(len(rows))
            if batch.score_exceptions:
                i, error = next(iter(batch.score_exceptions.items()))
                raise RuntimeError(
                    f"summaries raised for {len(batch.score_exceptions)} of {len(rows)} runs "
                    f"in this batch, first for {batch.jobs[i].run_hash}: {error!r}. This "
                    "usually means a bug in the summaries function: fix it and rerun, and "
                    "completed runs are reused. To reject a run instead, return NaN statistics."
                ) from error

            failed = set(batch.failures)
            x = []
            for i, (row, job) in enumerate(zip(rows, batch.jobs, strict=True)):
                hash_by_theta[tuple(row[n] for n in names)] = job.run_hash
                call_of[job.run_hash] = n_calls
                if i in failed:
                    if i in batch.errors:
                        load_errors[job.run_hash] = batch.errors[i]
                    x.append([float("nan")] * len(keys))
                    continue
                value = batch.values[i]
                if value is None:
                    raise ValueError(
                        f"summaries returned None for {job.run_hash}; return a dict of "
                        "statistics (NaN statistics reject a run)"
                    )
                missing = [k for k in keys if k not in value]
                if missing:
                    raise ValueError(
                        f"summaries returned {sorted(value)} for {job.run_hash}, "
                        f"missing observed statistics {missing}"
                    )
                x.append([float(value[k]) for k in keys])
            x_batch = torch.tensor(x, dtype=torch.float32)

            # sbi keeps the initial population's num_particles closest runs, sorting NaN
            # last, so with too few usable runs it would keep runs without statistics and
            # replace the non-finite tolerance with 1e8.
            needed = self.method_kwargs.get("num_particles")
            if self.method == "smcabc" and n_calls == 0 and needed is not None:
                usable = int(torch.isfinite(x_batch).all(dim=1).sum())
                if usable < needed:
                    crashed = sum(not result.success for result in batch.results)
                    unloaded = len(batch.errors)
                    raise RuntimeError(
                        f"Only {usable} of {len(rows)} initial SMC-ABC simulations have "
                        f"finite summaries, fewer than num_particles={needed} ({crashed} "
                        f"failed, {unloaded} could not be loaded, "
                        f"{len(rows) - usable - crashed - unloaded} returned non-finite "
                        "statistics). Check those runs, or raise num_initial_pop."
                    )
            n_calls += 1
            return x_batch

        def lookup(particles: Any) -> list[str | None]:
            return [
                hash_by_theta.get(tuple(_sbi_param_value(v) for v in row))
                for row in particles.tolist()
            ]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        common = dict(
            simulator=simulator,
            prior=prior,
            distance=self.distance,
            simulation_batch_size=None,  # each proposal round is one batch
            num_workers=1,  # parallelism lives in Josh, not in sbi processes
            show_progress_bars=False,
        )
        populations: list[dict[str, Any]] = []
        if self.method == "smcabc":
            samples, summary = SMCABC(**common)(
                x_o=x_o, return_summary=True, **self.method_kwargs
            )
            for t, (particles, log_w, eps, dists) in enumerate(
                zip(
                    summary["particles"],
                    summary["weights"],
                    summary["epsilons"],
                    summary["distances"],
                    strict=True,
                )
            ):
                w = log_w.exp()
                populations.append(
                    {
                        "generation": t,
                        "epsilon": float(eps),
                        "run_hashes": lookup(particles),
                        "weights": (w / w.sum()).tolist(),
                        "distances": dists.tolist(),
                    }
                )
        else:
            samples, summary = MCABC(**common)(
                x_o=x_o, return_summary=True, **self.method_kwargs
            )
            dists = summary["distances"]
            populations.append(
                {
                    "generation": 0,
                    "epsilon": float(dists.max()) if len(dists) else None,
                    "run_hashes": lookup(samples),
                    "weights": [1.0 / len(samples)] * len(samples) if len(samples) else [],
                    "distances": dists.tolist(),
                }
            )
        _count_generation_simulations(populations, call_of, call_sizes)

        final = populations[-1]
        run_hashes = lookup(samples)
        if run_hashes == final["run_hashes"]:
            weights = torch.tensor(final["weights"])
        else:
            weights = torch.full((len(samples),), 1.0 / max(len(samples), 1))

        for pop in populations:
            for run_hash, w, d in zip(
                pop["run_hashes"], pop["weights"], pop["distances"], strict=True
            ):
                if run_hash is not None:
                    ctx.registry.set_attributes(
                        run_hash,
                        sbi_accepted=True,
                        sbi_generation=pop["generation"],
                        sbi_weight=w,
                        sbi_distance=d,
                    )

        settings.update(
            populations=populations,
            epsilons=[pop["epsilon"] for pop in populations],
            simulations=ctx.succeeded + ctx.failed,
            failed_simulations=ctx.failed,
            load_errors=load_errors,
            final_run_hashes=run_hashes,
        )
        ctx.registry.update_session_metadata(ctx.session_id, sbi=settings)
        self._warn_on_unusable_runs(load_errors, final)

        if not ctx.quiet:
            print(
                f"sbi {self.method} complete: {len(populations)} population(s), "
                f"{ctx.succeeded + ctx.failed} simulations ({ctx.failed} failed, "
                f"{len(load_errors)} not loaded), final epsilon {final['epsilon']}"
            )

        return SbiResult(
            job_results=list(ctx.job_results),
            succeeded=ctx.succeeded,
            failed=ctx.failed,
            run_ids=dict(ctx.run_ids),
            job_set=ctx.job_set(),
            method=self.method,
            param_names=names,
            samples=samples,
            weights=weights,
            run_hashes=run_hashes,
            populations=populations,
            observed=observed,
            summary=summary,
            load_errors=load_errors,
        )

    @staticmethod
    def _warn_on_unusable_runs(load_errors: dict[str, str], final: dict[str, Any]) -> None:
        """Warn, even when quiet, about runs whose NaN summaries sbi saw.

        A run whose results could not be loaded is never accepted, so it drops
        out of the search unnoticed. A quantile-based MCABC keeps its share of
        the simulations whether or not they have usable statistics, so its
        accepted samples can include such runs.
        """
        import math
        import warnings

        if load_errors:
            run_hash, message = next(iter(load_errors.items()))
            warnings.warn(
                f"{len(load_errors)} simulation(s) ran but their results could not be "
                f"loaded, so their summaries were NaN and sbi never accepted them (first: "
                f"{run_hash}: {message}). They are listed in SbiResult.load_errors and "
                "the session metadata.",
                stacklevel=2,
            )
        unusable = sum(not math.isfinite(d) for d in final["distances"])
        if unusable:
            warnings.warn(
                f"{unusable} of {len(final['distances'])} accepted samples have non-finite "
                "summaries (failed, unloaded or NaN-returning runs): a quantile keeps its "
                "share of the simulations whether or not they have usable statistics.",
                stacklevel=2,
            )

    def _session_settings(self, observed: dict[str, float]) -> dict[str, Any]:
        """Settings recorded in session metadata, with a replay fingerprint."""
        import hashlib
        import json

        import sbi
        import torch

        settings: dict[str, Any] = {
            "method": self.method,
            "prior": self.prior,
            "param_names": self.param_names,
            "observed": observed,
            "summary_keys": list(observed),
            "method_kwargs": self.method_kwargs,
            "distance": self.distance,
            "seed": self.seed,
        }
        settings["fingerprint"] = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()[:12]
        settings["versions"] = {
            "sbi": sbi.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        }
        return settings

    @staticmethod
    def _warn_on_shared_seed(ctx: AdaptiveContext) -> None:
        """Warn if JobConfig.seed gives every simulation the same random draws.

        ABC compares one stochastic simulation per proposal with the data. With
        a fixed Josh seed every proposal reuses one noise realization, so the
        posterior conditions on that realization and ignores the model's noise.
        """
        import warnings

        if ctx.config.seed is not None:
            warnings.warn(
                f"JobConfig.seed={ctx.config.seed} gives every simulation the same Josh "
                "random draws, so ABC conditions on one noise realization and the "
                "posterior ignores the model's stochasticity. Leave seed unset for "
                "SbiStrategy runs and seed only the run that produces the observed data.",
                stacklevel=2,
            )

    def _warn_on_version_drift(self, ctx: AdaptiveContext, settings: dict[str, Any]) -> None:
        """Warn if an earlier session with the same settings ran other versions.

        Replay-based resume needs identical proposals; a different sbi or torch
        version may propose differently, so completed runs may not be reused.
        """
        import json
        import warnings

        rows = ctx.registry.conn.execute(
            "SELECT session_id, metadata FROM sweep_sessions WHERE session_id != ?",
            [ctx.session_id],
        ).fetchall()
        for session_id, raw in rows:
            earlier = (json.loads(raw) if raw else {}).get("sbi") or {}
            if (
                earlier.get("fingerprint") == settings["fingerprint"]
                and earlier.get("versions") != settings["versions"]
            ):
                warnings.warn(
                    f"Session {session_id} ran these sbi settings under "
                    f"{earlier.get('versions')}; this run uses {settings['versions']}. "
                    "Proposals may differ, so its completed runs may not be reused.",
                    stacklevel=2,
                )


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
        score_exceptions: Row index -> the exception ``score`` raised, for
            rows whose results loaded but could not be scored (a subset of
            ``errors``).
        run_ids: run_hash -> run_id of the run each job's results come from.
        sources: Per row, where its result came from: ``"ran"`` (dispatched
            in this batch), ``"reused"`` (an earlier completed, loaded run of
            the same run_hash) or ``"reattached"`` (an earlier no-wait batch
            job polled to completion).

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
    sources: list[str] = field(default_factory=list)
    score_exceptions: dict[int, Exception] = field(default_factory=dict)

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
    strategy later uses its result. The registry doubles as the run store:
    with ``reuse_runs`` (default), a run_hash that already completed and was
    loaded is reused rather than run again, and a submitted no-wait batch job
    is polled rather than re-dispatched, so an interrupted sweep resumes
    from its last finished simulation.

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
    reuse_runs: bool = True

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

        evaluated_before = self.succeeded + self.failed
        succeeded_before = self.succeeded
        load_config = self.load_config or LoadConfig()

        # The registry is the run store: reuse runs that already completed and
        # loaded, reattach to submitted no-wait jobs, and run only the rest.
        # A crashed or interrupted run therefore resumes from its last
        # finished simulation.
        result_by_hash: dict[str, Any] = {}
        run_id_by_hash: dict[str, str] = {}
        source_by_hash: dict[str, str] = {}
        to_attach: list[tuple[ExpandedJob, Any]] = []
        to_run: list[ExpandedJob] = []
        for job in batch_jobs:
            state, prior = (
                self._prior_run(job, load_config) if self.reuse_runs else ("new", None)
            )
            if state == "done":
                assert prior is not None
                result_by_hash[job.run_hash] = CLIResult(
                    exit_code=0, stdout="", stderr="", command=["reuse", prior.run_id]
                )
                run_id_by_hash[job.run_hash] = prior.run_id
                source_by_hash[job.run_hash] = "reused"
            elif state == "submitted":
                to_attach.append((job, prior))
            else:
                to_run.append(job)

        n_reused = len(batch_jobs) - len(to_attach) - len(to_run)
        if not self.quiet and (n_reused or to_attach):
            print(
                f"  Reusing {n_reused} completed run(s), reattaching to "
                f"{len(to_attach)}, running {len(to_run)}"
            )

        if to_attach:
            for run_hash, (run_id, result) in self._reattach(to_attach).items():
                result_by_hash[run_hash] = result
                run_id_by_hash[run_hash] = run_id
                source_by_hash[run_hash] = "reattached"

        if to_run:
            self._check_distinct_exports(to_run)
            show_progress = not self.quiet and len(to_run) > 1
            done = 0

            def _progress(job: ExpandedJob, result: Any) -> None:
                nonlocal done
                done += 1
                if show_progress:
                    status = "OK  " if result.success else "FAIL"
                    print(f"  [{done}/{len(to_run)}] {status} {job.run_hash}")
                if self.on_complete is not None:
                    self.on_complete(job, result)

            if show_progress:
                print(f"Evaluating {len(to_run)} parameter sets")

            sweep_result = run_sweep(
                self.cli,
                JobSet(jobs=to_run),
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
            for job, result in sweep_result.job_results:
                result_by_hash[job.run_hash] = result
                source_by_hash[job.run_hash] = "ran"
            run_id_by_hash.update(sweep_result.run_ids)

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
                source_by_hash.setdefault(job.run_hash, "ran")
            self.job_results.append((job, result))
            self._jobs.setdefault(job.run_hash, job)
            if result.success:
                self.succeeded += 1
            else:
                self.failed += 1
        self.run_ids.update(run_id_by_hash)

        # Load and score the successful runs (loading is idempotent, so reused
        # runs load nothing new).
        succeeded_jobs = [j for j in batch_jobs if result_by_hash[j.run_hash].success]
        if any(source_by_hash[j.run_hash] != "reused" for j in succeeded_jobs):
            time.sleep(load_config.settle_delay)

        rows_by_hash: dict[str, int] = {}
        values_by_hash: dict[str, Any] = {}
        errors_by_hash: dict[str, str] = {}
        score_exceptions_by_hash: dict[str, Exception] = {}
        for k, job in enumerate(succeeded_jobs):
            try:
                rows_by_hash[job.run_hash] = self._load(
                    job,
                    run_id_by_hash.get(job.run_hash),
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
                score_exceptions_by_hash[job.run_hash] = e
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
            run_ids={h: run_id_by_hash[h] for h in unique if h in run_id_by_hash},
            sources=[source_by_hash[j.run_hash] for j in row_jobs],
            score_exceptions={
                i: score_exceptions_by_hash[j.run_hash]
                for i, j in enumerate(row_jobs)
                if j.run_hash in score_exceptions_by_hash
            },
        )

    def _prior_run(self, job: ExpandedJob, load_config: Any) -> tuple[str, Any]:
        """Classify *job* by its latest recorded run.

        Returns ``("done", run)`` for a successful run whose replicates are
        loaded (loading it now if it completed but was never loaded),
        ``("submitted", run)`` for a no-wait batch job that was dispatched but
        never completed, and ``("new", None)`` otherwise: never run, failed,
        not current (``bad`` / ``superseded``), or an interrupted blocking run.
        """
        config = self.registry.get_config_by_hash(job.run_hash)
        if config is None or config.effective_status != "active":
            return "new", None

        runs = self.registry.get_runs_for_hash(job.run_hash)
        if not runs:
            return "new", None
        latest = runs[-1]
        metadata = latest.metadata or {}

        if latest.completed_at is None:
            if metadata.get("batch_job_id") and (metadata.get("target") or self.target):
                return "submitted", latest
            return "new", None
        if latest.exit_code != 0:
            return "new", None

        if len(self.registry.loaded_replicates(job.run_hash)) < job.replicates:
            # The replicate count isn't part of run_hash. A run with fewer
            # replicates can't supply the rest, so rerun it without waiting for
            # their files; loading keeps the replicates already loaded.
            if metadata.get("replicates", job.replicates) < job.replicates:
                return "new", None
            # Completed but never (fully) loaded, e.g. interrupted before loading.
            try:
                self._load(job, latest.run_id, load_config, succeeded_before=0)
            except Exception:
                return "new", None
            if len(self.registry.loaded_replicates(job.run_hash)) < job.replicates:
                return "new", None
        return "done", latest

    def _reattach(
        self, pending: list[tuple[ExpandedJob, Any]]
    ) -> dict[str, tuple[str, Any]]:
        """Poll submitted no-wait jobs from an earlier process until they finish.

        Completes each job's run in the registry and registers its outputs.
        A job that times out is left submitted (it may still be running).

        Returns:
            run_hash -> (run_id, CLIResult of the poll).
        """
        from joshpy.jobs import _register_job_outputs, poll_batch_jobs
        from joshpy.registry import RegistryCallback

        by_job_id = {run.metadata["batch_job_id"]: (job, run) for job, run in pending}
        # _prior_run only reports "submitted" when a target is known.
        targets = {
            job_id: str(run.metadata.get("target") or self.target)
            for job_id, (_, run) in by_job_id.items()
        }
        recorder = RegistryCallback(self.registry, self.session_id)
        finished: dict[str, tuple[str, Any]] = {}

        def _on_polled(job_id: str, result: Any) -> None:
            job, run = by_job_id[job_id]
            if result.exit_code != -1:
                recorder.record_completion(run.run_id, result)
                _register_job_outputs(
                    self.cli,
                    self.registry,
                    job,
                    run.run_id,
                    export_paths=self._exports_for(job),
                    quiet=True,
                )
            finished[job.run_hash] = (run.run_id, result)
            if not self.quiet:
                print(f"  [REATTACHED] {job_id} exit_code={result.exit_code}")
            if self.on_complete is not None:
                self.on_complete(job, result)

        if not self.quiet:
            print(f"Reattaching to {len(by_job_id)} submitted batch job(s)")
        poll_batch_jobs(
            self.cli,
            targets,
            poll_interval=self.poll_interval,
            timeout=self.batch_timeout,
            on_finished=_on_polled,
        )
        return finished

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

    def _check_distinct_exports(self, jobs: list[ExpandedJob]) -> None:
        """Refuse a multi-job batch whose jobs would all write the same patch file.

        evaluate() runs a whole batch before loading it, so jobs sharing one
        export path would overwrite each other and the wrong rows would load.
        Only ``{run_hash}`` and ``{label}`` are unique per job.

        Raises:
            ValueError: If the patch export path has neither placeholder.
        """
        if len(jobs) < 2 or jobs[0].source_path is None:
            return
        try:
            info = self._exports_for(jobs[0]).export_files.get("patch")
        except Exception:
            return
        path = getattr(info, "path", None)
        if not isinstance(path, str) or "{run_hash}" in path or "{label}" in path:
            return
        raise ValueError(
            f"exportFiles.patch ({path!r}) has no {{run_hash}} or {{label}}, so all "
            f"{len(jobs)} jobs in this batch would write the same file and their "
            "results could not be told apart. Add {run_hash} to the export path, "
            'e.g. "file:///tmp/results_{run_hash}_{replicate}.csv".'
        )

    def _exports_for(self, job: ExpandedJob) -> Any:
        """inspect-exports result for *job*'s script, cached per (script, simulation)."""
        from joshpy.cli import InspectExportsConfig

        if job.source_path is None:
            raise RuntimeError(f"Job {job.run_hash} has no source_path to inspect exports")
        key = (str(job.source_path), job.simulation)
        if key not in self._export_paths:
            self._export_paths[key] = self.cli.inspect_exports(
                InspectExportsConfig(script=job.source_path, simulation=job.simulation)
            )
        return self._export_paths[key]

    def _load(
        self,
        job: ExpandedJob,
        run_id: str | None,
        load_config: Any,
        *,
        succeeded_before: int,
    ) -> int:
        """Load one job's results into ``cell_data``; returns rows loaded."""
        from joshpy.sweep import load_job_results

        if run_id is None:
            raise RuntimeError(f"No run was recorded for {job.run_hash}")

        return load_job_results(
            self.cli,
            job,
            self.registry,
            self._exports_for(job),
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
    reuse_runs: bool = True,
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
        reuse_runs: If True (default), a parameter set whose run_hash already
            has a completed, loaded run in the registry is not run again, and a
            submitted no-wait batch job is reattached to instead of
            re-dispatched. This is what makes an interrupted sweep resumable.
            Set False to always run.

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
        reuse_runs=reuse_runs,
    )

    registry.update_session_status(session_id, "running")
    try:
        result = strategy.run_adaptive(ctx)
    except BaseException:
        # Includes KeyboardInterrupt: an interrupted sweep is marked failed;
        # rerunning it resumes from the runs already in the registry.
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
    # Build metadata dict
    metadata: dict[str, Any] = {
        "optuna_n_trials": len(study.trials),
        "optuna_direction": study.direction.name,
    }

    if study.best_trial is not None:
        metadata["optuna_best_params"] = study.best_params
        metadata["optuna_best_value"] = study.best_value
        metadata["optuna_best_trial_number"] = study.best_trial.number

    registry.update_session_metadata(session_id, **metadata)
