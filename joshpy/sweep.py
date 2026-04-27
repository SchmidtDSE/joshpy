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

import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from joshpy.cell_data import CellDataLoader, DiagnosticQueries
from joshpy.cli import ExportPaths, InspectExportsConfig, JoshCLI, StageFromMinioConfig
from joshpy.jobs import (
    ExpandedJob,
    JobConfig,
    JobExpander,
    JobSet,
    SweepResult,
    run_sweep,
)
from joshpy.registry import RunRegistry, configure_s3

# Sentinel for SweepManager.run() kwargs so the builder-stashed defaults
# (set via with_batch_remote) can be distinguished from an explicit
# caller-provided value.
_UNSET: Any = object()


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


class SweepCollisionError(Exception):
    """Raised when a batch-remote sweep would silently overwrite prior MinIO outputs.

    The static check fires when:
    - the export path template lacks both ``{timestamp}`` and ``{run_hash}``, AND
    - the registry already has runs for one or more jobs in this sweep, AND
    - the user did not pass ``force=True`` to ``SweepManager.run()``.

    Attributes:
        conflicts: List of ``(job, path_template, prior_runs)`` triples — one
            entry per job in the sweep that would collide.
    """

    def __init__(
        self,
        conflicts: list[tuple[ExpandedJob, str, list[Any]]],
    ) -> None:
        self.conflicts = conflicts
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        n = len(self.conflicts)
        first_template = self.conflicts[0][1]
        per_run_lines = "\n".join(
            f"  - run_hash {job.run_hash}: {len(priors)} prior run(s)"
            for job, _template, priors in self.conflicts
        )
        return (
            f"{n} job(s) in this sweep would silently overwrite prior MinIO outputs.\n\n"
            f"{per_run_lines}\n\n"
            f"Your export path template ({first_template!r}) doesn't include "
            f"{{timestamp}} or {{run_hash}} — re-dispatching will overwrite the "
            f"existing CSVs while the registry still references them. The "
            f"registry would report more replicates than MinIO actually contains.\n\n"
            "Fix one of:\n"
            "  1. Add {timestamp} to your export path (recommended — every dispatch\n"
            "     gets a fresh folder):\n"
            '         exportFiles.patch = "minio://bucket/{timestamp}/output_{replicate}.csv"\n\n'
            "  2. Once batchRemote --custom-tag passthrough lands and joshpy auto-injects\n"
            "     run_hash, use {run_hash} for deterministic per-simulation paths:\n"
            '         exportFiles.patch = "minio://bucket/{run_hash}/output_{replicate}.csv"\n\n'
            "  3. Drop the prior run(s) from the registry if you intend a fresh re-run.\n\n"
            "  4. Pass force=True to SweepManager.run() to proceed anyway (you accept\n"
            "     that subsequent ingest() calls will count duplicate replicates)."
        )


def _check_export_path_safety(
    cli: JoshCLI,
    job_set: JobSet,
    registry: RunRegistry,
    *,
    export_paths_cache: dict[tuple[str, str], ExportPaths] | None = None,
    quiet: bool = False,
) -> dict[tuple[str, str], ExportPaths]:
    """Raise SweepCollisionError if a batch-remote sweep would overwrite prior MinIO outputs.

    For each job in ``job_set``, compute its export path template (caching by
    ``(source_path, simulation)``) and check three conditions:

    1. The patch export protocol is ``minio://`` — local paths are out of scope.
    2. The path template lacks ``{timestamp}`` and ``{run_hash}`` placeholders —
       either is sufficient to disambiguate dispatches.
    3. The registry has prior runs for the job's ``run_hash``.

    If all three are true for any job, raises ``SweepCollisionError`` listing the
    conflicting hashes.

    Args:
        cli: Josh CLI for ``inspect_exports``.
        job_set: The expanded jobs to check.
        registry: Registry to query for prior runs.
        export_paths_cache: Optional pre-populated cache. New entries are added
            in place. Returned for callers (e.g., the auto_ingest hook) to reuse.
        quiet: Suppress progress output (currently unused; reserved).

    Returns:
        The updated export-paths cache (same dict that was passed in, if any).
    """
    if export_paths_cache is None:
        export_paths_cache = {}

    conflicts: list[tuple[ExpandedJob, str, list[Any]]] = []

    for job in job_set:
        if job.source_path is None:
            continue
        cache_key = (str(job.source_path), job.simulation)
        export_paths = export_paths_cache.get(cache_key)
        if export_paths is None:
            export_paths = cli.inspect_exports(
                InspectExportsConfig(script=job.source_path, simulation=job.simulation)
            )
            export_paths_cache[cache_key] = export_paths

        export_info = export_paths.export_files.get("patch")
        if export_info is None or export_info.protocol != "minio":
            continue

        path_template = export_info.path
        if "{timestamp}" in path_template or "{run_hash}" in path_template:
            continue

        prior_runs = registry.get_runs_for_hash(job.run_hash)
        if prior_runs:
            conflicts.append((job, path_template, list(prior_runs)))

    if conflicts:
        raise SweepCollisionError(conflicts)

    return export_paths_cache


# Collision policies for batch-remote sweeps. See SweepManagerBuilder.with_collision_policy.
COLLISION_POLICIES: tuple[str, ...] = ("fail", "pool", "skip", "overwrite")


class _PartialFormat(dict):
    """Dict subclass for ``str.format_map`` — unknown keys stay as ``{key}``.

    Used to resolve known template variables (e.g. ``{run_hash}``, ``{label}``)
    in an export path while leaving ``{replicate}`` (and any other unknown
    placeholders) intact for downstream pattern matching.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _list_existing_replicates_minio(
    cli: JoshCLI,
    registry: RunRegistry,
    export_info: Any,
    known_vars: dict[str, Any],
    *,
    quiet: bool = False,
) -> set[int]:
    """List existing MinIO/S3 objects and extract replicate indices from filenames.

    Resolves all known template variables via ``known_vars`` (keeping unknown
    placeholders like ``{replicate}`` intact), globs the resulting S3 pattern
    via DuckDB's ``glob()``, then extracts the integer replicate index from
    each filename.

    Returns an empty set if:
    - the template has no ``{replicate}`` placeholder (single-file output);
    - the template contains unresolved placeholders beyond ``{replicate}`` and
      ``{timestamp}`` (e.g. ``{step}``, ``{variable}`` — multi-file patterns
      aren't handled by the MVP);
    - listing fails for any reason (treated as "nothing prior").

    Args:
        cli: JoshCLI (needed by ``_configure_minio_access``).
        registry: RunRegistry whose DuckDB conn is used for the glob query.
        export_info: ``ExportFileInfo`` — ``protocol`` must be ``"minio"``.
        known_vars: Resolved template variables (``run_hash``, ``label``,
            custom tags, simulation parameters). ``{replicate}`` stays literal.
        quiet: Suppress progress/warning output.

    Returns:
        Set of replicate indices found on MinIO.
    """
    try:
        resolved = export_info.path.format_map(_PartialFormat(known_vars))
    except (IndexError, ValueError):
        return set()

    if "{replicate}" not in resolved:
        # Without {replicate}, all replicates land in one consolidated CSV —
        # the file count can't tell us how many replicates exist. Caller
        # treats empty set as "dispatch fresh".
        return set()

    if "{timestamp}" in resolved:
        # {timestamp} means the user explicitly chose per-dispatch isolation
        # (each run writes to a fresh folder). Listing across dispatches
        # would find unrelated runs; safer to treat as no prior outputs.
        return set()

    bucket = export_info.host
    try:
        _configure_minio_access(
            cli, registry, export_info, resolved,
            download=False, output_dir=None, minio_bucket=None, quiet=quiet,
        )
    except Exception as e:
        if not quiet:
            print(f"  [POLICY] MinIO access not configured: {e}")
        return set()

    # Build glob: replace {replicate} and any other surviving placeholders
    # with '*'. (After the runtime-vars guard above, "other" placeholders
    # in practice mean unsupported / aspirational template vars rather
    # than multi-file fanout — Josh writes one CSV per replicate per
    # export type regardless.)
    glob_resolved = re.sub(r"\{[^{}]+\}", "*", resolved)
    prefix_for_glob = glob_resolved.lstrip("/")
    glob_pattern = f"s3://{bucket}/{prefix_for_glob}"

    try:
        rows = registry.conn.execute(
            "SELECT file FROM glob(?)", [glob_pattern]
        ).fetchall()
    except Exception as e:
        if not quiet:
            print(f"  [POLICY] glob() failed ({e}); treating as no prior outputs.")
        return set()

    # Build extraction regex: {replicate} → (\d+) capture, anything else → .*?.
    # Anchor on the s3 URL with bucket prefix so we don't accidentally match
    # outside the user's intended folder.
    pattern_parts: list[str] = [re.escape(f"s3://{bucket}")]
    cursor = 0
    placeholder_re = re.compile(r"\{([^{}]+)\}")
    for m in placeholder_re.finditer(resolved):
        pattern_parts.append(re.escape(resolved[cursor:m.start()]))
        if m.group(1) == "replicate":
            pattern_parts.append(r"(\d+)")
        else:
            pattern_parts.append(r".*?")
        cursor = m.end()
    pattern_parts.append(re.escape(resolved[cursor:]))
    name_pattern = re.compile("".join(pattern_parts) + r"$")

    found: set[int] = set()
    for (name,) in rows:
        name_match = name_pattern.match(name)
        if name_match:
            found.add(int(name_match.group(1)))
    return found


@dataclass
class _CollisionAction:
    """Outcome of applying a collision policy to an existing-replicates set.

    Attributes:
        action: ``"dispatch"`` (proceed with dispatch), ``"skip"`` (no-op),
            or ``"fail"`` (raise — caller turns this into SweepCollisionError).
        replicate_start: Offset for the batch-remote dispatch (ignored unless
            action == "dispatch").
        replicates: Number of replicates to dispatch (ignored unless
            action == "dispatch").
        existing: The pre-existing replicate indices observed on MinIO.
    """

    action: str
    replicate_start: int = 0
    replicates: int = 0
    existing: frozenset[int] = field(default_factory=frozenset)


def _apply_collision_policy(
    policy: str,
    existing: set[int],
    n_requested: int,
) -> _CollisionAction:
    """Compute the dispatch plan for a job given policy + pre-existing state.

    Policies:
    - ``"fail"`` (default) — if any prior replicates exist, return ``action="fail"``.
      The caller converts this into :class:`SweepCollisionError`. This path is
      primarily reached through :func:`_check_export_path_safety` (Item 6);
      include here so the code path is symmetric.
    - ``"pool"`` — fill the gap between existing and requested. Dispatches
      replicates ``max(existing)+1 .. n_requested-1`` at an offset. If already
      complete (``max(existing)+1 >= n_requested``), returns ``action="skip"``.
    - ``"skip"`` — idempotent: if *any* prior replicate exists, return
      ``action="skip"`` (treat as complete for CI re-runs). Otherwise dispatch
      normally.
    - ``"overwrite"`` — always dispatch ``0..n_requested-1`` over whatever's
      there. MinIO PUT replaces existing files at the same paths. NOTE: when
      the new dispatch is *smaller* than the existing run, replicates beyond
      ``n_requested-1`` remain as orphans — joshpy currently has no MinIO
      delete primitive to clean them up. Caller must accept this or use
      ``{timestamp}`` paths if cleanliness matters.

    Args:
        policy: One of :data:`COLLISION_POLICIES`.
        existing: Pre-existing replicate indices on MinIO.
        n_requested: Total replicates the user asked for (``job.replicates``).

    Returns:
        A ``_CollisionAction`` describing what to do.

    Raises:
        ValueError: if ``policy`` is not a recognized value.
    """
    if policy not in COLLISION_POLICIES:
        raise ValueError(
            f"Unknown collision policy {policy!r}; must be one of {COLLISION_POLICIES}"
        )

    existing_fs = frozenset(existing)

    # "overwrite" always dispatches 0..N-1, ignoring whatever's there.
    if policy == "overwrite":
        return _CollisionAction(
            action="dispatch",
            replicate_start=0,
            replicates=n_requested,
            existing=existing_fs,
        )

    if not existing_fs:
        return _CollisionAction(
            action="dispatch",
            replicate_start=0,
            replicates=n_requested,
            existing=existing_fs,
        )

    max_existing = max(existing_fs)

    if policy == "fail":
        return _CollisionAction(action="fail", existing=existing_fs)

    if policy == "skip":
        return _CollisionAction(action="skip", existing=existing_fs)

    # policy == "pool"
    k = max_existing + 1
    if k >= n_requested:
        return _CollisionAction(action="skip", existing=existing_fs)
    return _CollisionAction(
        action="dispatch",
        replicate_start=k,
        replicates=n_requested - k,
        existing=existing_fs,
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
    run_id: str,
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
        run_id: The run_id to attribute cell_data rows to. Must be the
            run_id from the specific CLI invocation that produced these
            results (from registry.start_run() or RegistryCallback.record()).
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
        >>> run_id = registry.start_run(run_hash=job.run_hash)
        >>> result = cli.run(to_run_config(job))
        >>> registry.complete_run(run_id, exit_code=result.exit_code)
        >>> export_paths = cli.inspect_exports(InspectExportsConfig(...))
        >>> rows = load_job_results(cli, job, registry, export_paths, run_id=run_id)
    """
    if load_config is None:
        load_config = LoadConfig()

    # Route minio:// exports through the S3-aware loader. Local paths
    # continue through the filesystem retry loop below.
    export_info = export_paths.export_files.get(export_type)
    if export_info is not None and export_info.protocol == "minio":
        return _load_job_results_minio(
            cli=cli,
            registry=registry,
            job=job,
            export_paths=export_paths,
            export_info=export_info,
            run_id=run_id,
            export_type=export_type,
            quiet=quiet,
        )

    # Get path template for requested export type
    path_template = _get_export_path(export_paths, export_type)
    if not path_template:
        return 0
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


def _load_job_results_minio(
    cli: JoshCLI,
    registry: RunRegistry,
    job: ExpandedJob,
    export_paths: ExportPaths,
    export_info: Any,  # ExportFileInfo
    run_id: str,
    export_type: str,
    quiet: bool,
) -> int:
    """Per-job loader for ``minio://`` export targets.

    Mirrors the S3 path in :func:`ingest_results` / :func:`_load_ingest_replicates`
    but works from an already-known (job, run_id) pair instead of a
    registry lookup by label/hash. Called from :func:`load_job_results`
    when it detects ``export_info.protocol == "minio"``.
    """
    from types import SimpleNamespace

    bucket, dl_dir = _configure_minio_access(
        cli, registry, export_info, export_info.path,
        download=False, output_dir=None, minio_bucket=None, quiet=quiet,
    )

    meta = _IngestMetadata(
        run_hash=job.run_hash,
        config=SimpleNamespace(parameters=job.parameters),
        simulation=job.simulation,
        total_replicates=job.replicates,
        label=job.label,
        custom_tags=dict(job.custom_tags),
    )

    return _load_ingest_replicates(
        registry, export_paths, export_info.path,
        is_minio=True, download=False, bucket=bucket, dl_dir=dl_dir,
        meta=meta, run_id=run_id,
        export_type=export_type, quiet=quiet,
    )


def recover_sweep_results(
    cli: JoshCLI,
    job_set: JobSet,
    registry: RunRegistry,
    *,
    run_ids: dict[str, str],
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
        run_ids: Mapping of run_hash to run_id. Each job's cell_data is
            attributed to the run_id for its run_hash. Obtain from
            SweepResult.run_ids after calling run_sweep().
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
        job_run_id = run_ids.get(job.run_hash)
        if job_run_id is None:
            jobs_not_in_registry += 1
            if not quiet:
                params_str = ", ".join(f"{k}={v}" for k, v in job.parameters.items())
                print(f"  No run_id for job ({params_str or 'no params'})")
            continue

        # Load results for this job
        job_rows = load_job_results(
            cli=cli,
            job=job,
            registry=registry,
            export_paths=export_paths,
            run_id=job_run_id,
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
class _IngestMetadata:
    """Resolved metadata for an ingest operation."""

    run_hash: str
    config: Any
    simulation: str
    total_replicates: int
    label: str | None
    custom_tags: dict[str, str] | None = None


def _resolve_ingest_metadata(
    registry: RunRegistry,
    label_or_hash: str,
    *,
    quiet: bool = False,
) -> _IngestMetadata:
    """Resolve a label or hash to the run metadata needed for ingestion."""
    run_hash = registry._resolve_label_or_hash(label_or_hash)
    config = registry.get_config_by_hash(run_hash)
    if config is None:
        raise KeyError(f"No config found for run hash: {run_hash}")

    session = registry.get_session(config.session_id)
    if session is None:
        raise KeyError(f"No session found: {config.session_id}")

    simulation = session.simulation

    # Determine replicate count: session metadata > job_runs count > fallback to 1
    total_replicates = session.total_replicates
    if not total_replicates:
        job_config = session.job_config
        if job_config is not None:
            total_replicates = getattr(job_config, "replicates", None)
    if not total_replicates:
        runs = registry.get_runs_for_hash(run_hash)
        total_replicates = len(runs) if runs else 1

    if not quiet:
        label_str = f" ({config.label})" if config.label else ""
        print(f"Ingesting results for {run_hash}{label_str}")
        print(f"  Simulation: {simulation}, Replicates: {total_replicates}")

    return _IngestMetadata(
        run_hash=run_hash,
        config=config,
        simulation=simulation,
        total_replicates=total_replicates,
        label=config.label,
    )


def _get_josh_source(config: Any, run_hash: str) -> tuple[Path, str | None]:
    """Get josh source file on disk, creating a temp file if needed.

    Returns:
        ``(josh_path, temp_file_path_or_None)``.  Caller must clean up
        the temp file when non-None.
    """
    if config.josh_path and Path(config.josh_path).exists():
        return Path(config.josh_path), None

    if config.josh_content:
        fd, temp_path = tempfile.mkstemp(suffix=".josh")
        os.close(fd)
        Path(temp_path).write_text(config.josh_content)
        return Path(temp_path), temp_path

    raise RuntimeError(
        f"Cannot inspect exports: no josh source available for {run_hash}. "
        "Neither josh_path exists on disk nor josh_content stored in registry."
    )


def _configure_minio_access(
    cli: JoshCLI,
    registry: RunRegistry,
    export_info: Any,
    path_template: str,
    *,
    download: bool,
    output_dir: Path | None,
    minio_bucket: str | None,
    quiet: bool,
) -> tuple[str, Path | None]:
    """Configure S3 direct read or download from MinIO.

    Returns:
        ``(bucket_name, download_dir_or_None)``.
    """
    bucket = minio_bucket or export_info.host

    if not download:
        endpoint = os.environ.get("MINIO_ENDPOINT", "")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "")

        if not endpoint or not access_key or not secret_key:
            raise RuntimeError(
                "MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY "
                "environment variables are required for S3 reads."
            )

        configure_s3(registry.conn, endpoint, access_key, secret_key)

        if not quiet:
            print(f"  Reading directly from S3 (bucket: {bucket})")

        return bucket, None

    # download=True: stage files locally via stageFromMinio
    prefix = str(Path(path_template).parent).lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    dl_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="joshpy-ingest-"))
    dl_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"  Downloading from minio://{bucket}/{prefix} to {dl_dir}")

    stage_result = cli.stage_from_minio(
        StageFromMinioConfig(
            output_dir=dl_dir,
            prefix=prefix,
            minio_bucket=bucket,
        )
    )
    if not stage_result.success:
        raise RuntimeError(
            f"stageFromMinio failed (exit {stage_result.exit_code}): "
            f"{stage_result.stderr}"
        )

    return bucket, dl_dir


def _load_ingest_replicates(
    registry: RunRegistry,
    export_paths: ExportPaths,
    path_template: str,
    *,
    is_minio: bool,
    download: bool,
    bucket: str,
    dl_dir: Path | None,
    meta: _IngestMetadata,
    run_id: str,
    export_type: str,
    quiet: bool,
) -> int:
    """Load CSVs for each replicate into the registry."""
    loader = CellDataLoader(registry)
    total_rows = 0
    loaded = 0
    skipped = 0

    template_vars_base: dict[str, Any] = {}
    if meta.custom_tags:
        template_vars_base.update(meta.custom_tags)
    template_vars_base["simulation"] = meta.simulation
    template_vars_base["run_hash"] = meta.run_hash
    if meta.config.parameters:
        template_vars_base.update(meta.config.parameters)
    if meta.label:
        template_vars_base["label"] = meta.label

    for rep in range(meta.total_replicates):
        template_vars = {**template_vars_base, "replicate": rep}

        try:
            resolved = export_paths.resolve_path(path_template, **template_vars)
        except KeyError:
            if not quiet:
                print(f"  Replicate {rep}: template variable missing, skipping")
            skipped += 1
            continue

        # Determine the actual path/URL to load
        if is_minio and not download:
            resolved_path = resolved.as_posix().lstrip("/")
            csv_target: Path | str = f"s3://{bucket}/{resolved_path}"
        elif is_minio and download:
            csv_target = dl_dir / resolved.name
        else:
            csv_target = resolved

        # Try loading -- skip gracefully if missing
        try:
            rows = loader.load_csv(
                csv_path=csv_target,
                run_id=run_id,
                run_hash=meta.run_hash,
                entity_type=export_type,
            )
            total_rows += rows
            loaded += 1
            if not quiet:
                print(f"  Replicate {rep}: {rows:,} rows loaded")
        except FileNotFoundError:
            skipped += 1
            if not quiet:
                print(f"  Replicate {rep}: not found, skipping")
        except Exception as e:
            skipped += 1
            if not quiet:
                err_str = str(e)
                # DuckDB raises IOException for missing S3 objects
                if "HTTP 404" in err_str or "NoSuchKey" in err_str:
                    print(f"  Replicate {rep}: not found in S3, skipping")
                else:
                    print(f"  Replicate {rep}: error loading: {e}")

    if not quiet:
        print(f"\nDone: {total_rows:,} rows loaded ({loaded} replicates, {skipped} skipped)")

    return total_rows


def ingest_results(
    cli: JoshCLI,
    registry: RunRegistry,
    label_or_hash: str,
    *,
    export_type: str = "patch",
    download: bool = False,
    output_dir: Path | None = None,
    minio_bucket: str | None = None,
    export_paths: ExportPaths | None = None,
    quiet: bool = False,
) -> int:
    """Recover and ingest results into the registry by label or run hash.

    Looks up the run in the registry, discovers export paths via
    ``inspect_exports``, and loads CSVs into the ``cell_data`` table.

    For ``minio://`` export paths the default behaviour reads CSVs directly
    from S3 into DuckDB via ``httpfs`` (no local download).  Set
    ``download=True`` to download via ``stageFromMinio`` first.

    Missing CSVs (e.g. from an OOM'd replicate) are skipped gracefully.

    Args:
        cli: JoshCLI instance.
        registry: RunRegistry where results will be loaded.
        label_or_hash: Human-readable label or 12-char run hash.
        export_type: Type of export to load (``"patch"``, ``"meta"``, ``"entity"``).
        download: If True, download CSVs locally via ``stageFromMinio``
            instead of reading directly from S3.
        output_dir: Local directory for downloads (temp dir if None).
            Only used when ``download=True``.
        minio_bucket: Override the MinIO bucket (default: parsed from
            the ``minio://`` export path).
        export_paths: Pre-computed ExportPaths. When provided, skips the
            ``cli.inspect_exports`` subprocess call. Callers inside a hot
            sweep loop (e.g. ``run_sweep`` auto-ingest) can pass the cached
            value from ``_register_job_outputs``.
        quiet: Suppress progress output.

    Returns:
        Total number of rows loaded.

    Raises:
        KeyError: If label/hash not found in registry.
        RuntimeError: If no export path configured for *export_type*, or
            if ``inspect_exports`` fails.

    Examples:
        >>> # Recover results for a labeled run (reads from S3)
        >>> rows = ingest_results(cli, registry, "my-label")

        >>> # Download locally first, then load
        >>> rows = ingest_results(cli, registry, "my-label", download=True)
    """
    meta = _resolve_ingest_metadata(registry, label_or_hash, quiet=quiet)
    josh_path, temp_josh = _get_josh_source(meta.config, meta.run_hash)

    try:
        if export_paths is None:
            export_paths = cli.inspect_exports(
                InspectExportsConfig(script=josh_path, simulation=meta.simulation)
            )

        export_info = export_paths.export_files.get(export_type)
        if export_info is None:
            raise RuntimeError(
                f"No {export_type} export configured in {josh_path}. "
                f"Check that exportFiles.{export_type} is set in your simulation."
            )

        path_template = export_info.path
        is_minio = export_info.protocol == "minio"

        if not quiet:
            proto = f"minio://{export_info.host}" if is_minio else "local"
            print(f"  Export path: {proto}{path_template}")

        bucket: str = ""
        dl_dir: Path | None = None
        if is_minio:
            bucket, dl_dir = _configure_minio_access(
                cli, registry, export_info, path_template,
                download=download, output_dir=output_dir,
                minio_bucket=minio_bucket, quiet=quiet,
            )

        run_id = registry._resolve_run_id_for_hash(meta.run_hash)

        return _load_ingest_replicates(
            registry, export_paths, path_template,
            is_minio=is_minio, download=download, bucket=bucket,
            dl_dir=dl_dir, meta=meta, run_id=run_id,
            export_type=export_type, quiet=quiet,
        )
    finally:
        if temp_josh:
            Path(temp_josh).unlink(missing_ok=True)


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
    _catalog: Any | None = field(default=None, repr=False)  # ProjectCatalog
    _experiment_id: str | None = field(default=None, repr=False)
    _last_run_ids: dict[str, str] = field(default_factory=dict, repr=False)
    # Builder-configured batch remote defaults (used by .run() when kwargs omitted)
    _batch_remote_target: str | None = field(default=None, repr=False)
    _batch_no_wait: bool = field(default=False, repr=False)
    _batch_poll_interval: int = field(default=10, repr=False)
    _batch_timeout: int | None = field(default=None, repr=False)
    _batch_auto_ingest: bool = field(default=True, repr=False)
    _collision_policy: str = field(default="fail", repr=False)

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
        batch_remote: bool | Any = _UNSET,
        target: str | None | Any = _UNSET,
        batch_no_wait: bool | Any = _UNSET,
        poll_interval: int | Any = _UNSET,
        batch_timeout: int | None | Any = _UNSET,
        auto_ingest: bool | Any = _UNSET,
        stop_on_failure: bool = True,
        dry_run: bool = False,
        quiet: bool = False,
        on_complete: Callable[[ExpandedJob, Any], None] | None = None,
        objective: Any | None = None,
        jfr: Any | None = None,  # JfrConfig
        enable_profiler: bool = False,
        stream_output: bool = False,
        bottle: str | None = None,
        bottle_dir: Path | None = None,
        bottle_omit_jshd: bool = False,
        force: bool = False,
        collision_policy: str | Any = _UNSET,
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
            jfr: Optional JFR profiling configuration. When provided, each job
                gets its own recording file with the run_hash in the filename.
            enable_profiler: Enable Josh evaluation profiler (--enable-profiler).
            stream_output: If True, stream JAR stdout/stderr to the terminal
                in real time while still capturing them in CLIResult.
            force: If True, bypass the static collision check that raises
                ``SweepCollisionError`` when a batch-remote sweep would
                silently overwrite prior MinIO outputs. Default False.
            collision_policy: Override the builder-configured collision policy
                for this call. One of :data:`COLLISION_POLICIES`: ``"fail"``
                (default) triggers the static check; ``"pool"`` fills the gap
                between existing and requested replicates via
                ``replicate_start``; ``"skip"`` is idempotent (no-op if any
                prior replicates exist).

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
        # Resolve batch settings: caller overrides > builder defaults
        if batch_remote is _UNSET:
            batch_remote = self._batch_remote_target is not None
        if target is _UNSET:
            target = self._batch_remote_target
        if batch_no_wait is _UNSET:
            batch_no_wait = self._batch_no_wait
        if poll_interval is _UNSET:
            poll_interval = self._batch_poll_interval
        if batch_timeout is _UNSET:
            batch_timeout = self._batch_timeout
        if auto_ingest is _UNSET:
            auto_ingest = self._batch_auto_ingest
        if collision_policy is _UNSET:
            collision_policy = self._collision_policy
        if collision_policy not in COLLISION_POLICIES:
            raise ValueError(
                f"Unknown collision policy {collision_policy!r}; "
                f"must be one of {COLLISION_POLICIES}"
            )

        # force=True is the legacy "dispatch over whatever's there" escape hatch.
        # Translate it into collision_policy="overwrite" so it bypasses BOTH
        # the static check (Item 6) AND the runtime listing check (Item 5).
        if force:
            collision_policy = "overwrite"

        # Pre-dispatch static collision check. Only fires for policy="fail"
        # (the default) — "pool" / "skip" / "overwrite" do their own runtime
        # handling at dispatch time inside run_sweep. dry_run bypasses.
        if (
            batch_remote
            and not dry_run
            and collision_policy == "fail"
        ):
            _check_export_path_safety(
                self.cli, self.job_set, self.registry, quiet=quiet,
            )

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
                batch_remote=batch_remote,
                target=target,
                batch_no_wait=batch_no_wait,
                poll_interval=poll_interval,
                batch_timeout=batch_timeout,
                auto_ingest=auto_ingest,
                quiet=quiet,
                stop_on_failure=stop_on_failure,
                jfr=jfr,
                enable_profiler=enable_profiler,
                stream_output=stream_output,
            )
        else:
            # Use batch runner
            result = run_sweep(
                cli=self.cli,
                job_set=self.job_set,
                registry=self.registry,
                session_id=self.session_id,
                remote=remote,
                api_key=api_key,
                endpoint=endpoint,
                batch_remote=batch_remote,
                target=target,
                batch_no_wait=batch_no_wait,
                poll_interval=poll_interval,
                batch_timeout=batch_timeout,
                auto_ingest=auto_ingest,
                collision_policy=collision_policy,
                on_complete=on_complete,
                stop_on_failure=stop_on_failure,
                dry_run=dry_run,
                quiet=quiet,
                jfr=jfr,
                enable_profiler=enable_profiler,
                stream_output=stream_output,
                bottle=bottle,
                bottle_dir=bottle_dir,
                bottle_omit_jshd=bottle_omit_jshd,
            )

            # Store run_ids for use by load_results()
            self._last_run_ids = result.run_ids

            # Update catalog status if configured
            if self._catalog is not None and self._experiment_id is not None and not dry_run:
                status = "completed" if result.failed == 0 else "failed"
                self._catalog.update_experiment_status(
                    self._experiment_id,
                    status,
                    summary={
                        "total_jobs": result.succeeded + result.failed,
                        "succeeded": result.succeeded,
                        "failed": result.failed,
                    },
                )

            return result

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
        if not self._last_run_ids:
            raise RuntimeError(
                "No run_ids available. Call run() before load_results(), "
                "or use recover_sweep_results() with explicit run_ids."
            )
        return recover_sweep_results(
            cli=self.cli,
            job_set=self.job_set,
            registry=self.registry,
            run_ids=self._last_run_ids,
            export_type=export_type,
            quiet=quiet,
        )

    def ingest(
        self,
        *,
        export_type: str = "patch",
        download: bool = False,
        output_dir: Path | None = None,
        minio_bucket: str | None = None,
        quiet: bool = False,
    ) -> int:
        """Recover and ingest results from MinIO (or local) by label.

        Uses ``ingest_results()`` to look up the run by label, discover
        export paths, and load CSVs into the registry.  Unlike
        ``load_results()`` this does not require a prior ``run()`` call --
        it works from the registry alone.

        Args:
            export_type: Type of export to load ("patch", "meta", "entity").
            download: If True, download CSVs locally instead of S3 direct read.
            output_dir: Download destination (only used with download=True).
            minio_bucket: Override MinIO bucket name.
            quiet: Suppress progress output.

        Returns:
            Total number of rows loaded.

        Examples:
            >>> manager.ingest()  # reads directly from S3
            >>> manager.ingest(download=True, output_dir=Path("./local"))
        """
        label = self._label if hasattr(self, "_label") and self._label else None
        identifier = label or self.job_set.jobs[0].run_hash
        return ingest_results(
            cli=self.cli,
            registry=self.registry,
            label_or_hash=identifier,
            export_type=export_type,
            download=download,
            output_dir=output_dir,
            minio_bucket=minio_bucket,
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
        self._catalog: Any | None = None  # ProjectCatalog
        self._catalog_experiment_name: str | None = None
        self._label: str | None = None
        self._label_force: bool = False
        self._label_on_collision: str | None = None
        # Batch-remote dispatch defaults (set via with_batch_remote()).
        self._batch_remote_target: str | None = None
        self._batch_no_wait: bool = False
        self._batch_poll_interval: int = 10
        self._batch_timeout: int | None = None
        self._batch_auto_ingest: bool = True
        self._collision_policy: str = "fail"

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

    def with_catalog(
        self,
        catalog: Any,  # ProjectCatalog
        *,
        experiment_name: str | None = None,
    ) -> SweepManagerBuilder:
        """Configure an optional ProjectCatalog for cross-experiment tracking.

        When a catalog is configured, the experiment is automatically registered
        on build() and its status is updated after run().

        Args:
            catalog: A ProjectCatalog instance.
            experiment_name: Human-readable name for the experiment in the catalog.

        Returns:
            Self for chaining.

        Examples:
            >>> from joshpy.catalog import ProjectCatalog
            >>> catalog = ProjectCatalog("project.duckdb")
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_registry("experiment.duckdb")
            ...     .with_catalog(catalog, experiment_name="baseline_dev_fine")
            ...     .build()
            ... )
        """
        self._catalog = catalog
        self._catalog_experiment_name = experiment_name
        return self

    def with_label(
        self,
        label: str,
        force: bool = False,
        on_collision: str | None = None,
    ) -> SweepManagerBuilder:
        """Set a label for this run (single-job configs only).

        The label is applied at build() time when the job is registered.
        For multi-job sweeps, raises ValueError at build() time.

        Args:
            label: Human-readable label for this run.
            force: If True, reassign the label even if already taken
                (drops the old label).
            on_collision: Collision strategy. ``"timestamp"`` archives the
                old label with a timestamp suffix so the bare label always
                points to the latest run. Mutually exclusive with ``force``.

        Returns:
            Self for chaining.

        Examples:
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_registry("experiment.duckdb")
            ...     .with_label("baseline")
            ...     .build()
            ... )

            >>> # Re-run with same label, archive the old one
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_registry("experiment.duckdb")
            ...     .with_label("baseline", on_collision="timestamp")
            ...     .build()
            ... )
        """
        self._label = label
        self._label_force = force
        self._label_on_collision = on_collision
        return self

    def with_batch_remote(
        self,
        target: str,
        *,
        no_wait: bool = False,
        poll_interval: int = 10,
        timeout: int | None = None,
        auto_ingest: bool = True,
    ) -> SweepManagerBuilder:
        """Pre-configure batch remote dispatch for the built SweepManager.

        After calling this, ``SweepManager.run()`` defaults to
        ``batch_remote=True`` with the supplied settings. Callers may still
        override any individual setting at ``.run()`` call time.

        Args:
            target: Target profile name (loaded from ``~/.josh/targets/<name>.json``).
            no_wait: If True, dispatch and return without polling. Polling
                is then the caller's responsibility (or a later
                ``SweepManager.ingest()`` call).
            poll_interval: Seconds between poll attempts when ``no_wait=True``.
            timeout: Overall timeout per job in seconds.
            auto_ingest: If True, auto-call ``ingest_results()`` after each
                successful batch job.

        Returns:
            Self for chaining.

        Examples:
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_registry("experiment.duckdb", experiment_name="my_sweep")
            ...     .with_batch_remote("gke-test")
            ...     .build()
            ... )
            >>> # Equivalent to manager.run(batch_remote=True, target="gke-test")
            >>> results = manager.run()
        """
        self._batch_remote_target = target
        self._batch_no_wait = no_wait
        self._batch_poll_interval = poll_interval
        self._batch_timeout = timeout
        self._batch_auto_ingest = auto_ingest
        return self

    def with_collision_policy(self, policy: str) -> SweepManagerBuilder:
        """Configure how batch-remote sweeps handle prior MinIO outputs.

        Args:
            policy: One of:

                - ``"fail"`` (default) — raise :class:`SweepCollisionError` at
                  ``.run()`` time if the export path template lacks both
                  ``{timestamp}`` and ``{run_hash}`` and any job's ``run_hash``
                  already has runs in the registry. Equivalent to the static
                  check; safest default.
                - ``"pool"`` — list existing MinIO objects and dispatch only
                  the missing replicates via
                  :attr:`BatchRemoteConfig.replicate_start`. Enables
                  "grow a sweep" workflows: first run N=5 succeeds, re-run
                  with N=10 only dispatches replicates 5..9.
                - ``"skip"`` — idempotent: if *any* matching MinIO object
                  exists, skip the dispatch entirely. Useful for CI reruns.
                - ``"overwrite"`` — always dispatch replicates 0..N-1 over
                  whatever's there; MinIO PUT replaces files at matching
                  paths. Note: shrinking sweeps (old N=10, new N=5) leaves
                  orphans 5..9 since joshpy has no MinIO delete primitive.
                  Use ``{timestamp}`` paths when orphan-cleanliness matters.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If ``policy`` is not one of :data:`COLLISION_POLICIES`.

        Examples:
            >>> manager = (
            ...     SweepManager.builder(config)
            ...     .with_registry("experiment.duckdb")
            ...     .with_batch_remote("gke-test")
            ...     .with_collision_policy("pool")
            ...     .build()
            ... )

        Notes:
            ``"pool"`` and ``"skip"`` enumerate prior outputs by listing MinIO
            and matching the export-path template. The template's
            ``{replicate}`` slot becomes the integer capture; any other
            unresolved placeholder (rare in practice — Josh writes one CSV
            per replicate per export type regardless of which template
            variables appear) becomes a wildcard. ``{timestamp}`` is the one
            special case: its presence signals per-dispatch isolation, so
            listing returns empty (no pooling across timestamps).
        """
        if policy not in COLLISION_POLICIES:
            raise ValueError(
                f"Unknown collision policy {policy!r}; "
                f"must be one of {COLLISION_POLICIES}"
            )
        self._collision_policy = policy
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
                    josh_content = None
                    if job.source_path and job.source_path.exists():
                        josh_content = job.source_path.read_text()

                    self._registry.register_run(
                        session_id=session_id,
                        run_hash=job.run_hash,
                        josh_path=str(job.source_path) if job.source_path else "",
                        config_content=job.config_content,
                        file_mappings=self._convert_file_mappings(job.file_mappings),
                        parameters=job.parameters,
                        josh_content=josh_content,
                    )

                # Apply label if set (with_label() takes precedence over
                # JobConfig.label; both register with the registry and
                # inject the label as a --custom-tag for export paths).
                effective_label = self._label or self._config.label
                if effective_label is not None:
                    if len(job_set.jobs) != 1:
                        raise ValueError(
                            f"with_label() requires a single-job config, but this "
                            f"config expands to {len(job_set.jobs)} jobs. Label "
                            f"individual runs via registry.label_run()."
                        )
                    self._registry.label_run(
                        job_set.jobs[0].run_hash,
                        effective_label,
                        force=self._label_force,
                        on_collision=self._label_on_collision,
                    )
                    job_set.jobs[0].label = effective_label
                    job_set.jobs[0].custom_tags["label"] = effective_label

        # Register with catalog if configured
        experiment_id = None
        if self._catalog is not None:
            registry_path = str(self._registry.db_path) if hasattr(self._registry, "db_path") else ""
            experiment_id = self._catalog.register_experiment(
                config=self._config,
                registry_path=registry_path,
                name=self._catalog_experiment_name,
            )
            self._catalog.update_experiment_status(experiment_id, "running")

        return SweepManager(
            config=self._config,
            registry=self._registry,
            cli=self._cli,
            job_set=job_set,
            session_id=session_id,
            _owns_registry=self._owns_registry,
            _owns_cli=self._owns_cli,
            _catalog=self._catalog,
            _experiment_id=experiment_id,
            _batch_remote_target=self._batch_remote_target,
            _batch_no_wait=self._batch_no_wait,
            _batch_poll_interval=self._batch_poll_interval,
            _batch_timeout=self._batch_timeout,
            _batch_auto_ingest=self._batch_auto_ingest,
            _collision_policy=self._collision_policy,
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
