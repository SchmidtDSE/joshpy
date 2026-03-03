"""Sweep orchestration and result collection for Josh simulations.

This module provides high-level tools for managing parameter sweeps:
- `recover_sweep_results()` - Load CSV results from completed sweep jobs into registry
- `load_job_results()` - Load CSV results for a single job with retry logic
- `LoadConfig` - Configuration for result loading behavior
- `ResultLoadError` - Error raised when result loading fails
- `SweepManager` - Convenience orchestrator for parameter sweeps
- `SweepManagerBuilder` - Builder pattern for flexible SweepManager configuration

Example usage:
    from joshpy.cli import JoshCLI, InspectExportsConfig
    from joshpy.jobs import JobConfig, JobExpander
    from joshpy.registry import RunRegistry
    from joshpy.sweep import recover_sweep_results, SweepManager

    # Setup
    cli = JoshCLI()
    registry = RunRegistry("experiment.duckdb")
    config = JobConfig.from_yaml_file(Path("sweep_config.yaml"))
    job_set = JobExpander().expand(config)

    # After jobs complete, recover results
    rows = recover_sweep_results(
        cli=cli,
        job_set=job_set,
        registry=registry,
    )
    print(f"Loaded {rows} rows")

    # Or use SweepManager for end-to-end workflow
    with SweepManager.from_config(config, registry=":memory:") as manager:
        results = manager.run()
        manager.load_results()
        df = manager.query("averageHeight", group_by="maxGrowth")
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from joshpy.cell_data import CellDataLoader, DiagnosticQueries
from joshpy.cli import ExportPaths, InspectExportsConfig, JoshCLI
from joshpy.jobs import (
    ExpandedJob,
    JobConfig,
    JobExpander,
    JobSet,
    SweepResult,
    run_sweep,
)
from joshpy.registry import RunRegistry


@dataclass
class LoadConfig:
    """Configuration for result loading behavior.

    Controls retry logic and timing for loading CSV results from completed jobs.
    Used by `load_job_results()` and `recover_sweep_results()`.

    Attributes:
        max_retries: Maximum number of retry attempts for file operations.
        retry_delay: Seconds to wait between retry attempts.
        settle_delay: Seconds to wait after file appears before reading.
            Helps avoid partial reads when files are still being written.
        raise_on_missing: If True, raise ResultLoadError when CSV not found
            after all retries. If False, silently skip missing files.

    Examples:
        >>> # Default configuration
        >>> config = LoadConfig()

        >>> # Aggressive retries for slow filesystems
        >>> config = LoadConfig(max_retries=5, retry_delay=1.0, settle_delay=0.5)

        >>> # Fail fast on missing files
        >>> config = LoadConfig(raise_on_missing=True)
    """

    max_retries: int = 3
    retry_delay: float = 0.5  # seconds
    settle_delay: float = 0.2  # seconds to wait after file appears
    raise_on_missing: bool = False  # raise if CSV not found after retries


class ResultLoadError(Exception):
    """Raised when result loading fails after retries.

    Contains context about which job failed and how many jobs succeeded
    before the failure, useful for debugging and recovery.

    Attributes:
        job: The job that failed to load.
        succeeded_before: Number of jobs that loaded successfully before this failure.
        message: Description of what went wrong.

    Examples:
        >>> try:
        ...     load_job_results(cli, job, registry, export_paths,
        ...                      load_config=LoadConfig(raise_on_missing=True))
        ... except ResultLoadError as e:
        ...     print(f"Failed after {e.succeeded_before} successful jobs")
        ...     print(f"Job: {e.job.run_hash}")
    """

    def __init__(self, job: ExpandedJob, succeeded_before: int, message: str) -> None:
        """Initialize ResultLoadError.

        Args:
            job: The job that failed to load.
            succeeded_before: Number of jobs that succeeded before this failure.
            message: Description of what went wrong.
        """
        self.job = job
        self.succeeded_before = succeeded_before
        self.message = message
        super().__init__(
            f"Failed to load results for job {job.run_hash}: {message}. "
            f"{succeeded_before} jobs succeeded before this failure."
        )


def _wait_for_file(path: Path, config: LoadConfig) -> bool:
    """Wait for file to exist and settle.

    Implements retry logic for file existence checks, with a settle delay
    to avoid reading partially-written files.

    Args:
        path: Path to the file to wait for.
        config: Load configuration with retry settings.

    Returns:
        True if file exists and is ready (non-empty), False otherwise.
    """
    for attempt in range(config.max_retries):
        if path.exists():
            # Brief pause to let writes complete (avoid partial reads)
            time.sleep(config.settle_delay)
            # Verify file is non-empty
            try:
                if path.stat().st_size > 0:
                    return True
            except OSError:
                # File might have been removed between exists() and stat()
                pass
        if attempt < config.max_retries - 1:
            time.sleep(config.retry_delay)
    return False


def load_job_results(
    cli: JoshCLI,
    job: ExpandedJob,
    registry: RunRegistry,
    export_paths: ExportPaths,
    *,
    export_type: str = "patch",
    quiet: bool = False,
    load_config: LoadConfig | None = None,
    succeeded_before: int = 0,
) -> int:
    """Load CSV results for a single job into the registry.

    Extracts result loading logic from recover_sweep_results() for reuse
    in adaptive sweep runners. Includes retry logic for robustness against
    filesystem delays and partial writes.

    Args:
        cli: JoshCLI instance (needed for path resolution).
        job: The completed job to load results for.
        registry: Registry to load results into.
        export_paths: Export path configuration from inspect_exports().
        export_type: Type of export ("patch", "meta", "entity").
        quiet: Suppress output.
        load_config: Retry and timing configuration. Uses defaults if None.
        succeeded_before: Number of jobs that succeeded before this one.
            Used for error context in ResultLoadError.

    Returns:
        Number of rows loaded.

    Raises:
        ResultLoadError: If raise_on_missing=True and CSV not found after retries,
            or if CSV load fails after retries.

    Examples:
        >>> # Basic usage
        >>> export_paths = cli.inspect_exports(InspectExportsConfig(...))
        >>> rows = load_job_results(cli, job, registry, export_paths)

        >>> # With retry configuration
        >>> config = LoadConfig(max_retries=5, raise_on_missing=True)
        >>> rows = load_job_results(cli, job, registry, export_paths,
        ...                         load_config=config)
    """
    if load_config is None:
        load_config = LoadConfig()

    # Get path template for requested export type
    path_template = _get_export_path(export_paths, export_type)
    if not path_template:
        return 0

    # Get run_id from registry
    runs = registry.get_runs_for_hash(job.run_hash)
    if not runs:
        return 0

    run_id = runs[0].run_id
    loader = CellDataLoader(registry)
    total_rows = 0
    simulation = job.simulation

    for rep in range(job.replicates):
        # Build template variables from job parameters and custom tags
        template_vars = {
            "simulation": simulation,
            "replicate": rep,
            **job.parameters,
            **job.custom_tags,
        }

        try:
            csv_path = export_paths.resolve_path(path_template, **template_vars)
        except KeyError:
            continue

        # Wait for file with retries
        if not _wait_for_file(csv_path, load_config):
            if load_config.raise_on_missing:
                raise ResultLoadError(
                    job=job,
                    succeeded_before=succeeded_before,
                    message=f"CSV not found after {load_config.max_retries} retries: {csv_path}",
                )
            continue

        # Load with retries for transient errors
        last_error: Exception | None = None
        for attempt in range(load_config.max_retries):
            try:
                rows = loader.load_csv(
                    csv_path=csv_path,
                    run_id=run_id,
                    run_hash=job.run_hash,
                )
                total_rows += rows
                if not quiet:
                    print(f"  Loaded {rows} rows from {csv_path.name}")
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < load_config.max_retries - 1:
                    time.sleep(load_config.retry_delay)

        if last_error is not None:
            if load_config.raise_on_missing:
                raise ResultLoadError(
                    job=job,
                    succeeded_before=succeeded_before,
                    message=f"CSV load failed after {load_config.max_retries} retries: {last_error}",
                )
            elif not quiet:
                print(f"  Error loading {csv_path}: {last_error}")

    return total_rows


def recover_sweep_results(
    cli: JoshCLI,
    job_set: JobSet,
    registry: RunRegistry,
    *,
    export_type: str = "patch",
    quiet: bool = False,
    load_config: LoadConfig | None = None,
) -> int:
    """Load CSV results from completed sweep jobs into the registry.

    Uses inspect_exports() to discover export path templates from the Josh file,
    then resolves template variables for each job's parameters, custom_tags,
    and replicates.

    This is a convenience wrapper around load_job_results() for batch loading.

    Args:
        cli: JoshCLI instance (needed for inspect_exports).
        job_set: The expanded jobs to collect results for.
        registry: Registry to load results into.
        export_type: Type of export to load ("patch", "meta", "entity").
        quiet: Suppress progress output.
        load_config: Retry and timing configuration. Uses defaults if None.

    Returns:
        Total number of rows loaded.

    Raises:
        ValueError: If jobs have different source_paths or no source_path.
        RuntimeError: If no export path configured for export_type.
        ResultLoadError: If load_config.raise_on_missing=True and loading fails.

    Examples:
        >>> from joshpy.sweep import recover_sweep_results
        >>> rows = recover_sweep_results(
        ...     cli=cli,
        ...     job_set=job_set,
        ...     registry=registry,
        ...     export_type="patch",
        ... )
        >>> print(f"Loaded {rows} rows from completed jobs")

        >>> # With retry configuration
        >>> from joshpy.sweep import LoadConfig
        >>> config = LoadConfig(max_retries=5, settle_delay=0.5)
        >>> rows = recover_sweep_results(cli, job_set, registry, load_config=config)
    """
    if load_config is None:
        load_config = LoadConfig()

    # Validate jobs have consistent source paths
    source_paths = {job.source_path for job in job_set if job.source_path}
    if len(source_paths) == 0:
        raise ValueError("All jobs must have a source_path set")
    if len(source_paths) != 1:
        raise ValueError(f"All jobs must have the same source_path. Found: {source_paths}")

    source_path = source_paths.pop()
    simulation = job_set.jobs[0].simulation

    # Get export path template from josh file
    export_paths = cli.inspect_exports(
        InspectExportsConfig(
            script=source_path,
            simulation=simulation,
        )
    )

    # Get path template for requested export type
    path_template = _get_export_path(export_paths, export_type)
    if not path_template:
        raise RuntimeError(
            f"No {export_type} export configured in {source_path}. "
            f"Check that exportFiles.{export_type} is set in your simulation."
        )

    if not quiet:
        print(f"Loading {export_type} results from: {path_template}")

    # Load CSV for each job using load_job_results()
    total_rows = 0
    total_jobs = len(job_set.jobs)
    jobs_loaded = 0
    jobs_not_in_registry = 0
    succeeded_jobs = 0

    for job in job_set:
        # Check if job is in registry
        runs = registry.get_runs_for_hash(job.run_hash)
        if not runs:
            jobs_not_in_registry += 1
            if not quiet:
                params_str = ", ".join(f"{k}={v}" for k, v in job.parameters.items())
                print(f"  No run recorded for job ({params_str or 'no params'})")
            continue

        # Load results for this job
        job_rows = load_job_results(
            cli=cli,
            job=job,
            registry=registry,
            export_paths=export_paths,
            export_type=export_type,
            quiet=quiet,
            load_config=load_config,
            succeeded_before=succeeded_jobs,
        )

        if job_rows > 0:
            total_rows += job_rows
            jobs_loaded += 1
        succeeded_jobs += 1

    if not quiet:
        print("\nResults:")
        print(f"  Jobs in sweep: {total_jobs}")
        print(f"  Jobs with results loaded: {jobs_loaded}")
        if jobs_not_in_registry > 0:
            print(f"  Jobs not yet executed (no run in registry): {jobs_not_in_registry}")
        print(f"  Total rows loaded: {total_rows}")

    return total_rows


def _get_export_path(export_paths: ExportPaths, export_type: str) -> str | None:
    """Get the export path template for a given export type.

    Args:
        export_paths: ExportPaths object from inspect_exports.
        export_type: Type of export ("patch", "meta", "entity").

    Returns:
        Path template string, or None if not configured.
    """
    if export_type == "patch":
        return export_paths.get_patch_path()
    elif export_type == "meta":
        info = export_paths.export_files.get("meta")
        return info.path if info else None
    elif export_type == "entity":
        info = export_paths.export_files.get("entity")
        return info.path if info else None
    else:
        raise ValueError(f"Unknown export_type: {export_type}. Use 'patch', 'meta', or 'entity'.")


@dataclass
class SweepManager:
    """Convenience orchestrator for parameter sweeps.

    Encapsulates the common workflow of expanding, running, and collecting
    sweep results. For more control, use the underlying components directly.

    Attributes:
        config: The job configuration.
        registry: DuckDB registry for tracking.
        cli: Josh CLI for execution.
        job_set: Expanded jobs.
        session_id: Session ID in registry.

    Examples:
        >>> # Simple usage with defaults
        >>> with SweepManager.from_config(config, registry=":memory:") as manager:
        ...     results = manager.run()
        ...     manager.load_results()
        ...     df = manager.query("averageHeight", group_by="maxGrowth")

        >>> # Builder pattern for more control
        >>> manager = (
        ...     SweepManager.builder(config)
        ...     .with_registry("experiment.duckdb", experiment_name="my_sweep")
        ...     .with_cli(jar_path=Path("josh.jar"))
        ...     .build()
        ... )
    """

    config: JobConfig
    registry: RunRegistry
    cli: JoshCLI
    job_set: JobSet
    session_id: str
    _owns_registry: bool = field(default=False, repr=False)
    _owns_cli: bool = field(default=False, repr=False)

    # Entry points
    @classmethod
    def builder(cls, config: JobConfig) -> SweepManagerBuilder:
        """Create a builder for complex configuration.

        Args:
            config: The job configuration to expand and run.

        Returns:
            A SweepManagerBuilder for fluent configuration.

        Examples:
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_registry("experiment.duckdb")
            ...     .with_cli(jar_path=Path("josh.jar"))
            ...     .build()
            ... )
        """
        return SweepManagerBuilder(config)

    @classmethod
    def from_config(cls, config: JobConfig, **kwargs: Any) -> SweepManager:
        """Create from a JobConfig.

        Args:
            config: The job configuration.
            **kwargs: Passed to with_defaults() (registry, experiment_name, jar_path).

        Returns:
            Configured SweepManager ready for use.

        Examples:
            >>> config = JobConfig(
            ...     template_path=Path("template.jshc.j2"),
            ...     source_path=Path("simulation.josh"),
            ...     simulation="Main",
            ...     replicates=3,
            ...     sweep=SweepConfig(config_parameters=[...]),
            ... )
            >>> manager = SweepManager.from_config(config, registry=":memory:")
        """
        return cls.builder(config).with_defaults(**kwargs).build()

    @classmethod
    def from_yaml(cls, path: Path, **kwargs: Any) -> SweepManager:
        """Create from a YAML config file.

        Args:
            path: Path to YAML file containing JobConfig.
            **kwargs: Passed to with_defaults() (registry, experiment_name, jar_path).

        Returns:
            Configured SweepManager ready for use.

        Examples:
            >>> manager = SweepManager.from_yaml(Path("sweep.yaml"), registry=":memory:")
        """
        config = JobConfig.from_yaml_file(path)
        return cls.builder(config).with_defaults(**kwargs).build()

    # Core operations
    def run(
        self,
        *,
        remote: bool = False,
        api_key: str | None = None,
        endpoint: str | None = None,
        stop_on_failure: bool = True,
        dry_run: bool = False,
        quiet: bool = False,
        on_complete: Callable[[ExpandedJob, Any], None] | None = None,
        objective: Any | None = None,
    ) -> SweepResult:
        """Execute all jobs in the sweep.

        Automatically detects adaptive vs batch strategy and dispatches
        to the appropriate runner. Runs are automatically recorded in the registry.

        For adaptive strategies (OptunaStrategy), uses run_adaptive_sweep()
        which generates jobs on-demand based on Optuna's suggestions.

        For non-adaptive strategies (CartesianStrategy), uses run_sweep()
        which executes all pre-expanded jobs.

        Args:
            remote: If True, use run_remote() for remote execution.
            api_key: API key for authentication (optional for local servers).
            endpoint: Custom endpoint URL (optional).
            stop_on_failure: If True (default), stop on first job failure. For
                adaptive strategies, raises SweepExecutionError with full error
                details. For batch strategies, returns partial results.
            dry_run: If True, print plan without executing (batch only).
            quiet: If True, suppress progress output.
            on_complete: Optional additional callback invoked after each job.
                Signature: callback(job, result) -> None. Called after
                registry recording. Use for progress reporting, logging, etc.
            objective: Objective function for adaptive strategies. If not provided,
                uses the objective from the strategy configuration.

        Returns:
            SweepResult for batch strategies, AdaptiveSweepResult for adaptive.
            Both are compatible (same base interface).

        Examples:
            >>> # Local execution (auto-detects strategy)
            >>> results = manager.run()

            >>> # Remote execution (Josh Cloud)
            >>> results = manager.run(remote=True, api_key="your-api-key")

            >>> # Remote execution (local server, no API key)
            >>> results = manager.run(remote=True, endpoint="http://localhost:8080")

            >>> # Dry run to see what would be executed (batch only)
            >>> results = manager.run(dry_run=True)

            >>> # Adaptive sweep with custom objective
            >>> def my_objective(registry, run_hash, job):
            ...     return registry.query("SELECT AVG(value) FROM cell_data WHERE run_hash=?", [run_hash]).fetchone()[0]
            >>> results = manager.run(objective=my_objective)
        """
        # Check if strategy is adaptive
        strategy = self.config.sweep.strategy if self.config.sweep else None

        if strategy is not None and strategy.is_adaptive:
            # Use adaptive runner
            from joshpy.strategies import run_adaptive_sweep

            if dry_run:
                # Adaptive doesn't support dry_run directly
                if not quiet:
                    n_trials = getattr(strategy, "n_trials", "?")
                    print(f"Would run adaptive sweep with {n_trials} trials")
                    print("  Dry run - no jobs will be executed")
                return SweepResult()

            return run_adaptive_sweep(
                cli=self.cli,
                config=self.config,
                registry=self.registry,
                session_id=self.session_id,
                objective=objective,
                remote=remote,
                api_key=api_key,
                endpoint=endpoint,
                quiet=quiet,
                stop_on_failure=stop_on_failure,
            )
        else:
            # Use batch runner
            return run_sweep(
                cli=self.cli,
                job_set=self.job_set,
                registry=self.registry,
                session_id=self.session_id,
                remote=remote,
                api_key=api_key,
                endpoint=endpoint,
                on_complete=on_complete,
                stop_on_failure=stop_on_failure,
                dry_run=dry_run,
                quiet=quiet,
            )

    def load_results(
        self,
        *,
        export_type: str = "patch",
        quiet: bool = False,
    ) -> int:
        """Load CSV results into registry.

        Args:
            export_type: Type of export to load ("patch", "meta", "entity").
            quiet: Suppress progress output.

        Returns:
            Total number of rows loaded.

        Examples:
            >>> rows = manager.load_results()
            >>> print(f"Loaded {rows} rows")
        """
        return recover_sweep_results(
            cli=self.cli,
            job_set=self.job_set,
            registry=self.registry,
            export_type=export_type,
            quiet=quiet,
        )

    def query(
        self,
        variable: str,
        *,
        group_by: str | None = None,
        step: int | None = None,
        **filters: Any,
    ) -> pd.DataFrame:
        """Query results with optional grouping and filtering.

        Args:
            variable: Variable name to query (e.g., "averageHeight").
            group_by: Optional parameter name to group by.
            step: Optional timestep filter.
            **filters: Additional parameter filters.

        Returns:
            DataFrame with query results.

        Examples:
            >>> # Get parameter comparison
            >>> df = manager.query("averageHeight", group_by="maxGrowth")

            >>> # Get comparison at specific step
            >>> df = manager.query("averageHeight", group_by="maxGrowth", step=50)
        """
        queries = DiagnosticQueries(self.registry)
        if group_by:
            return queries.get_parameter_comparison(
                variable=variable,
                param_name=group_by,
                step=step,
                **filters,
            )
        # Return all data as a basic timeseries
        return queries.get_replicate_uncertainty(
            variable=variable,
            run_hash=self.job_set.jobs[0].run_hash if self.job_set.jobs else "",
            step=step,
        )

    # Cleanup
    def cleanup(self) -> None:
        """Clean up temporary files.

        Removes temporary config files created during job expansion.
        Should be called when you're done with the SweepManager.

        Examples:
            >>> manager = SweepManager.from_yaml(Path("sweep.yaml"))
            >>> try:
            ...     manager.run()
            ...     manager.load_results()
            ... finally:
            ...     manager.cleanup()
        """
        if self.job_set:
            self.job_set.cleanup()

    def close(self) -> None:
        """Close registry connection if owned.

        Only closes the registry if it was created by the builder,
        not if an existing registry was passed in.

        Examples:
            >>> manager = SweepManager.from_yaml(Path("sweep.yaml"))
            >>> try:
            ...     manager.run()
            ... finally:
            ...     manager.close()
        """
        if self._owns_registry and self.registry:
            self.registry.close()

    def __enter__(self) -> SweepManager:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - cleanup and close."""
        self.cleanup()
        self.close()


class SweepManagerBuilder:
    """Builder for SweepManager with flexible configuration.

    The builder pattern allows for fluent configuration of SweepManager
    with sensible defaults and resource ownership tracking.

    Examples:
        >>> manager = (
        ...     SweepManagerBuilder(config)
        ...     .with_registry("experiment.duckdb", experiment_name="my_sweep")
        ...     .with_cli(jar_path=Path("josh.jar"))
        ...     .build()
        ... )
    """

    def __init__(self, config: JobConfig) -> None:
        """Initialize builder with a JobConfig.

        Args:
            config: The job configuration to expand and run.
        """
        self._config = config
        self._registry: RunRegistry | None = None
        self._cli: JoshCLI | None = None
        self._experiment_name: str | None = None
        self._session_id: str | None = None
        self._owns_registry = False
        self._owns_cli = False

    def with_registry(
        self,
        registry_or_path: str | Path | RunRegistry,
        *,
        experiment_name: str | None = None,
        session_id: str | None = None,
    ) -> SweepManagerBuilder:
        """Configure the registry.

        Args:
            registry_or_path: Path to create new registry, or existing instance.
            experiment_name: Name for new session (when creating registry).
            session_id: Use existing session ID (to resume a previous session).

        Returns:
            Self for chaining.

        Examples:
            >>> # Create new registry from path
            >>> builder.with_registry("experiment.duckdb", experiment_name="my_sweep")

            >>> # Use existing registry
            >>> builder.with_registry(existing_registry, session_id="abc-123")
        """
        if isinstance(registry_or_path, RunRegistry):
            self._registry = registry_or_path
            self._owns_registry = False
        else:
            self._registry = RunRegistry(registry_or_path)
            self._owns_registry = True

        self._experiment_name = experiment_name
        self._session_id = session_id
        return self

    def with_cli(
        self,
        cli: JoshCLI | None = None,
        *,
        jar_path: Path | None = None,
        java_path: str = "java",
    ) -> SweepManagerBuilder:
        """Configure the CLI.

        Args:
            cli: Existing JoshCLI instance.
            jar_path: Path to josh jar (creates new CLI).
            java_path: Path to java executable (default: "java").

        Returns:
            Self for chaining.

        Examples:
            >>> # Use existing CLI
            >>> builder.with_cli(existing_cli)

            >>> # Create new CLI with custom jar path
            >>> builder.with_cli(jar_path=Path("josh.jar"))
        """
        if cli is not None:
            self._cli = cli
            self._owns_cli = False
        else:
            self._cli = JoshCLI(josh_jar=jar_path, java_path=java_path)
            self._owns_cli = True
        return self

    def with_defaults(
        self,
        *,
        registry: str | Path | RunRegistry = ":memory:",
        experiment_name: str | None = None,
        jar_path: Path | None = None,
    ) -> SweepManagerBuilder:
        """Apply default configuration for simple cases.

        Convenience method that sets up both registry and CLI with defaults.

        Args:
            registry: Registry path or instance (default: in-memory).
            experiment_name: Name for the experiment session.
            jar_path: Path to josh jar file.

        Returns:
            Self for chaining.

        Examples:
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_defaults(registry=":memory:")
            ...     .build()
            ... )
        """
        self.with_registry(registry, experiment_name=experiment_name)
        self.with_cli(jar_path=jar_path)
        return self

    def build(self) -> SweepManager:
        """Build the SweepManager instance.

        Expands jobs, creates or verifies session, and registers runs.

        For adaptive strategies (OptunaStrategy), job expansion is deferred
        to run time - the JobSet will be empty initially.

        Returns:
            Configured SweepManager ready for use.

        Raises:
            ValueError: If session_id provided but hashes don't match registered runs.

        Examples:
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_registry("experiment.duckdb")
            ...     .with_cli()
            ...     .build()
            ... )
        """
        # Defaults
        if self._registry is None:
            self._registry = RunRegistry(":memory:")
            self._owns_registry = True

        if self._cli is None:
            self._cli = JoshCLI()
            self._owns_cli = True

        # Check if strategy is adaptive
        strategy = self._config.sweep.strategy if self._config.sweep else None
        is_adaptive = strategy is not None and strategy.is_adaptive

        if is_adaptive:
            # For adaptive strategies, jobs are created on-demand during run()
            # Don't pre-expand - just create session
            job_set = JobSet(jobs=[])
            session_id = self._session_id or self._registry.create_session(
                config=self._config,
                experiment_name=self._experiment_name,
            )
        else:
            # For non-adaptive strategies, expand jobs upfront
            expander = JobExpander()
            job_set = expander.expand(self._config)

            # Create or use session
            if self._session_id:
                session_id = self._session_id
                # Verify hashes match
                self._verify_hashes(job_set, session_id)
            else:
                session_id = self._registry.create_session(
                    config=self._config,
                    experiment_name=self._experiment_name,
                )
                # Register runs
                for job in job_set:
                    self._registry.register_run(
                        session_id=session_id,
                        run_hash=job.run_hash,
                        josh_path=str(job.source_path) if job.source_path else "",
                        config_content=job.config_content,
                        file_mappings=self._convert_file_mappings(job.file_mappings),
                        parameters=job.parameters,
                    )

        return SweepManager(
            config=self._config,
            registry=self._registry,
            cli=self._cli,
            job_set=job_set,
            session_id=session_id,
            _owns_registry=self._owns_registry,
            _owns_cli=self._owns_cli,
        )

    def _convert_file_mappings(
        self,
        file_mappings: dict[str, Path],
    ) -> dict[str, dict[str, str]] | None:
        """Convert file_mappings from {name: Path} to {name: {path, hash}} format.

        The registry stores file_mappings with both path and hash for each file.
        This allows verification and audit trailing.

        Args:
            file_mappings: Dict mapping names to Path objects.

        Returns:
            Dict in registry format, or None if no mappings.
        """
        if not file_mappings:
            return None

        from joshpy.jobs import _hash_file

        result = {}
        for name, path in file_mappings.items():
            result[name] = {
                "path": str(path),
                "hash": _hash_file(path) if path.exists() else "",
            }
        return result

    def _verify_hashes(self, job_set: JobSet, session_id: str) -> None:
        """Verify expanded job hashes match registered runs.

        Args:
            job_set: The expanded jobs to verify.
            session_id: Session ID to check against.

        Raises:
            ValueError: If hashes don't match (indicates non-deterministic expansion).
        """
        if self._registry is None:
            raise ValueError("Registry not configured")

        registered = self._registry.get_configs_for_session(session_id)
        registered_hashes = {r.run_hash for r in registered}
        expanded_hashes = {j.run_hash for j in job_set}

        if registered_hashes != expanded_hashes:
            missing = registered_hashes - expanded_hashes
            extra = expanded_hashes - registered_hashes
            raise ValueError(
                f"Job expansion hash mismatch for session {session_id}. "
                f"Missing: {missing}, Extra: {extra}. "
                "This indicates non-deterministic expansion - investigate!"
            )
