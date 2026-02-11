"""Sweep orchestration and result collection for Josh simulations.

This module provides high-level tools for managing parameter sweeps:
- `recover_sweep_results()` - Load CSV results from completed sweep jobs into registry
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
    manager = SweepManager.from_yaml(Path("sweep_config.yaml"), registry=":memory:")
    results = manager.run()
    manager.load_results()
    df = manager.query("averageHeight", group_by="maxGrowth")
"""

from __future__ import annotations

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
    to_run_config,
    to_run_remote_config,
)
from joshpy.registry import RunRegistry


def recover_sweep_results(
    cli: JoshCLI,
    job_set: JobSet,
    registry: RunRegistry,
    *,
    export_type: str = "patch",
    quiet: bool = False,
) -> int:
    """Load CSV results from completed sweep jobs into the registry.

    Uses inspect_exports() to discover export path templates from the Josh file,
    then resolves template variables for each job's parameters, custom_tags,
    and replicates.

    Args:
        cli: JoshCLI instance (needed for inspect_exports).
        job_set: The expanded jobs to collect results for.
        registry: Registry to load results into.
        export_type: Type of export to load ("patch", "meta", "entity").
        quiet: Suppress progress output.

    Returns:
        Total number of rows loaded.

    Raises:
        ValueError: If jobs have different source_paths or no source_path.
        RuntimeError: If no export path configured for export_type.

    Example:
        from joshpy.sweep import recover_sweep_results

        rows = recover_sweep_results(
            cli=cli,
            job_set=job_set,
            registry=registry,
            export_type="patch",
        )
        print(f"Loaded {rows} rows from completed jobs")
    """
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

    # Load CSV for each job and replicate
    loader = CellDataLoader(registry)
    total_rows = 0
    total_jobs = len(job_set.jobs)
    jobs_loaded = 0
    jobs_not_in_registry = 0
    files_missing = 0

    for job in job_set:
        # Get run_id from registry
        runs = registry.get_runs_for_hash(job.run_hash)
        if not runs:
            jobs_not_in_registry += 1
            if not quiet:
                params_str = ", ".join(f"{k}={v}" for k, v in job.parameters.items())
                print(f"  No run recorded for job ({params_str or 'no params'})")
            continue

        run_id = runs[0].run_id

        # Load each replicate - resolve path with all available variables
        job_rows = 0
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
            except KeyError as e:
                if not quiet:
                    print(f"  Warning: Cannot resolve path template - missing variable {e}")
                continue

            if csv_path.exists():
                try:
                    rows = loader.load_csv(
                        csv_path=csv_path,
                        run_id=run_id,
                        config_hash=job.run_hash,
                    )
                    job_rows += rows
                    if not quiet:
                        print(f"  Loaded {rows} rows from {csv_path.name}")
                except Exception as e:
                    if not quiet:
                        print(f"  Error loading {csv_path}: {e}")
            else:
                files_missing += 1
                if not quiet:
                    print(f"  File not found: {csv_path}")

        if job_rows > 0:
            total_rows += job_rows
            jobs_loaded += 1

    if not quiet:
        print("\nResults:")
        print(f"  Jobs in sweep: {total_jobs}")
        print(f"  Jobs with results loaded: {jobs_loaded}")
        if jobs_not_in_registry > 0:
            print(f"  Jobs not yet executed (no run in registry): {jobs_not_in_registry}")
        if files_missing > 0:
            print(f"  Output files not found: {files_missing}")
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

    Example:
        # Simple usage with defaults
        manager = SweepManager.from_yaml(Path("sweep.yaml"), registry=":memory:")
        results = manager.run()
        manager.load_results()
        df = manager.query("averageHeight", group_by="maxGrowth")

        # Builder pattern for more control
        manager = (
            SweepManager.builder(config)
            .with_registry("experiment.duckdb", experiment_name="my_sweep")
            .with_cli(jar_path=Path("josh.jar"))
            .build()
        )
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

        Example:
            manager = (
                SweepManager.builder(config)
                .with_registry("experiment.duckdb")
                .with_cli(jar_path=Path("josh.jar"))
                .build()
            )
        """
        return SweepManagerBuilder(config)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> SweepManager:
        """Create from a config dictionary.

        Args:
            config_dict: Dictionary representation of JobConfig.
            **kwargs: Passed to with_defaults() (registry, experiment_name, jar_path).

        Returns:
            Configured SweepManager ready for use.

        Example:
            manager = SweepManager.from_dict(config_dict, registry=":memory:")
        """
        config = JobConfig.from_dict(config_dict)
        return cls.builder(config).with_defaults(**kwargs).build()

    @classmethod
    def from_yaml(cls, path: Path, **kwargs: Any) -> SweepManager:
        """Create from a YAML config file.

        Args:
            path: Path to YAML file containing JobConfig.
            **kwargs: Passed to with_defaults() (registry, experiment_name, jar_path).

        Returns:
            Configured SweepManager ready for use.

        Example:
            manager = SweepManager.from_yaml(Path("sweep.yaml"), registry=":memory:")
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
        stop_on_failure: bool = False,
        dry_run: bool = False,
        quiet: bool = False,
        on_complete: Callable[[ExpandedJob, Any], None] | None = None,
    ) -> SweepResult:
        """Execute all jobs in the sweep.

        Args:
            remote: If True, use run_remote() for cloud execution.
            api_key: Josh Cloud API key (required if remote=True).
            endpoint: Custom Josh Cloud endpoint URL (optional).
            stop_on_failure: If True, stop on first failure.
            dry_run: If True, print plan without executing.
            quiet: If True, suppress progress output.
            on_complete: Optional callback invoked after each job completes.
                Signature: callback(job, result) -> None.

        Returns:
            SweepResult with all job outcomes.

        Raises:
            ValueError: If remote=True but api_key not provided.

        Example:
            # Local execution
            results = manager.run()

            # Cloud execution
            results = manager.run(remote=True, api_key="your-api-key")

            # Dry run to see what would be executed
            results = manager.run(dry_run=True)
        """
        if remote:
            if not api_key:
                raise ValueError("api_key required for remote execution")
            return self._run_remote(
                api_key=api_key,
                endpoint=endpoint,
                stop_on_failure=stop_on_failure,
                dry_run=dry_run,
                quiet=quiet,
                on_complete=on_complete,
            )
        else:
            return run_sweep(
                cli=self.cli,
                job_set=self.job_set,
                callback=on_complete,
                stop_on_failure=stop_on_failure,
                dry_run=dry_run,
                quiet=quiet,
            )

    def _run_remote(
        self,
        api_key: str,
        endpoint: str | None,
        stop_on_failure: bool,
        dry_run: bool,
        quiet: bool,
        on_complete: Callable[[ExpandedJob, Any], None] | None,
    ) -> SweepResult:
        """Execute all jobs remotely on Josh Cloud.

        Internal method used by run() when remote=True.
        """
        total_jobs = self.job_set.total_jobs
        total_replicates = self.job_set.total_replicates

        if not quiet:
            print(f"Running {total_jobs} jobs ({total_replicates} total replicates) remotely")

        if dry_run:
            if not quiet:
                print("Dry run - no jobs will be executed")
                for i, job in enumerate(self.job_set):
                    print(f"  [{i + 1}/{total_jobs}] {job.parameters}")
            return SweepResult()

        job_results: list[tuple[ExpandedJob, Any]] = []
        succeeded = 0
        failed = 0

        for i, job in enumerate(self.job_set):
            if not quiet:
                print(f"[{i + 1}/{total_jobs}] Running remotely: {job.parameters}")

            run_config = to_run_remote_config(job, api_key=api_key, endpoint=endpoint)
            result = self.cli.run_remote(run_config)

            job_results.append((job, result))

            if result.success:
                succeeded += 1
                if not quiet:
                    print("  [OK] Completed successfully")
            else:
                failed += 1
                if not quiet:
                    print(f"  [FAIL] Exit code: {result.exit_code}")

            if on_complete is not None:
                on_complete(job, result)

            if stop_on_failure and not result.success:
                if not quiet:
                    print("Stopping due to failure (stop_on_failure=True)")
                break

        if not quiet:
            print(f"Completed: {succeeded} succeeded, {failed} failed")

        return SweepResult(
            job_results=job_results,
            succeeded=succeeded,
            failed=failed,
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

        Example:
            rows = manager.load_results()
            print(f"Loaded {rows} rows")
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

        Example:
            # Get parameter comparison
            df = manager.query("averageHeight", group_by="maxGrowth")

            # Get comparison at specific step
            df = manager.query("averageHeight", group_by="maxGrowth", step=50)
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
            config_hash=self.job_set.jobs[0].run_hash if self.job_set.jobs else "",
            step=step,
        )

    # Cleanup
    def cleanup(self) -> None:
        """Clean up temporary files.

        Removes temporary config files created during job expansion.
        Should be called when you're done with the SweepManager.

        Example:
            manager = SweepManager.from_yaml(Path("sweep.yaml"))
            try:
                manager.run()
                manager.load_results()
            finally:
                manager.cleanup()
        """
        if self.job_set:
            self.job_set.cleanup()

    def close(self) -> None:
        """Close registry connection if owned.

        Only closes the registry if it was created by the builder,
        not if an existing registry was passed in.

        Example:
            manager = SweepManager.from_yaml(Path("sweep.yaml"))
            try:
                manager.run()
            finally:
                manager.close()
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

    Example:
        manager = (
            SweepManagerBuilder(config)
            .with_registry("experiment.duckdb", experiment_name="my_sweep")
            .with_cli(jar_path=Path("josh.jar"))
            .build()
        )
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

        Example:
            # Create new registry from path
            builder.with_registry("experiment.duckdb", experiment_name="my_sweep")

            # Use existing registry
            builder.with_registry(existing_registry, session_id="abc-123")
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

        Example:
            # Use existing CLI
            builder.with_cli(existing_cli)

            # Create new CLI with custom jar path
            builder.with_cli(jar_path=Path("josh.jar"))
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

        Example:
            manager = (
                SweepManager.builder(config)
                .with_defaults(registry=":memory:")
                .build()
            )
        """
        self.with_registry(registry, experiment_name=experiment_name)
        self.with_cli(jar_path=jar_path)
        return self

    def build(self) -> SweepManager:
        """Build the SweepManager instance.

        Expands jobs, creates or verifies session, and registers runs.

        Returns:
            Configured SweepManager ready for use.

        Raises:
            ValueError: If session_id provided but hashes don't match registered runs.

        Example:
            manager = (
                SweepManager.builder(config)
                .with_registry("experiment.duckdb")
                .with_cli()
                .build()
            )
        """
        # Defaults
        if self._registry is None:
            self._registry = RunRegistry(":memory:")
            self._owns_registry = True

        if self._cli is None:
            self._cli = JoshCLI()
            self._owns_cli = True

        # Expand jobs
        expander = JobExpander()
        job_set = expander.expand(self._config)

        # Create or use session
        if self._session_id:
            session_id = self._session_id
            # Verify hashes match
            self._verify_hashes(job_set, session_id)
        else:
            session_id = self._registry.create_session(
                experiment_name=self._experiment_name or self._config.simulation,
                simulation=self._config.simulation,
                metadata={"job_config": self._config.to_dict()},
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
