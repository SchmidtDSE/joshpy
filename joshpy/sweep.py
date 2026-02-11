"""Sweep orchestration and result collection for Josh simulations.

This module provides high-level tools for managing parameter sweeps:
- `recover_sweep_results()` - Load CSV results from completed sweep jobs into registry

Example usage:
    from joshpy.cli import JoshCLI, InspectExportsConfig
    from joshpy.jobs import JobConfig, JobExpander
    from joshpy.registry import RunRegistry
    from joshpy.sweep import recover_sweep_results

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
"""

from __future__ import annotations

from joshpy.cell_data import CellDataLoader
from joshpy.cli import ExportPaths, InspectExportsConfig, JoshCLI
from joshpy.jobs import JobSet
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
