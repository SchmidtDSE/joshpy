"""Self-contained reproducibility archives for Josh simulation runs.

A bottle is a ``.tar.gz`` archive containing everything needed to reproduce
Josh runs without Python or joshpy — just Java and the JAR.

Single-job bottle (``first_failure`` / ``first_success``)::

    bottle_{run_hash}/
        simulation.josh
        sweep_config.jshc
        data/
        run.sh
        manifest.json

Sweep bottle (``all`` / ``all_failures``)::

    bottle_sweep_{timestamp}/
        data/                        # shared .jshd files (copied once)
        jobs/
            {run_hash_1}/
                simulation.josh
                sweep_config.jshc
                run.sh               # --data paths → ../../data/
            {run_hash_2}/
                ...
        manifest.json                # metadata for all jobs

Usage::

    # During execution
    results = manager.run(bottle="first_failure")  # single-job bottle
    results = manager.run(bottle="all")            # sweep bottle

    # After the fact (single run)
    registry.bottle("baseline", cli=cli)

    # Standalone
    from joshpy.bottle import create_bottle, create_sweep_bottle
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tarfile
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from joshpy.cli import CLIResult, JoshCLI
    from joshpy.jobs import ExpandedJob
    from joshpy.registry import RunRegistry

BottleMode = Literal["first_failure", "all_failures", "first_success", "all"]
BOTTLE_MODES: set[str] = {"first_failure", "all_failures", "first_success", "all"}


def _get_joshpy_version() -> str:
    """Get joshpy version from package metadata."""
    try:
        from importlib.metadata import version

        return version("joshpy")
    except Exception:
        return "unknown"


def _get_git_hash() -> str | None:
    """Get current git HEAD hash, or None if not in a git repo."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        head = result.stdout.strip()
        if not head or result.returncode != 0:
            return None
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = dirty_result.stdout.strip()
        return f"{head[:12]}+dirty" if dirty else head[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _build_run_sh(
    *,
    simulation: str,
    replicates: int,
    config_name: str,
    data_files: dict[str, str],
    custom_tags: dict[str, str],
    run_hash: str,
    jar_version: str | None,
    jar_sha256: str | None,
    seed: int | None = None,
    crs: str | None = None,
    use_float64: bool = False,
    output_steps: str | None = None,
) -> str:
    """Generate the run.sh script content."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "#!/bin/bash",
        f"# Bottled by joshpy v{_get_joshpy_version()}",
        f"# JAR SHA256: {jar_sha256 or 'unknown'}",
        f"# JAR version: {jar_version or 'unknown'}",
        f"# Original run hash: {run_hash}",
        f"# Bottled at: {now}",
        "",
        "set -euo pipefail",
        "",
    ]

    # Build java command
    cmd_parts = [
        'java -jar "${1:?Usage: ./run.sh /path/to/joshsim-fat.jar}"',
        "run simulation.josh",
        simulation,
    ]

    # Config data
    cmd_parts.append(f"--data {config_name}={config_name}")

    # External data files
    for name, rel_path in sorted(data_files.items()):
        cmd_parts.append(f"--data {name}={rel_path}")

    # Replicates
    if replicates > 1:
        cmd_parts.append(f"--replicates {replicates}")

    # Custom tags
    for tag_name, tag_value in sorted(custom_tags.items()):
        cmd_parts.append(f"--custom-tag {tag_name}={tag_value}")

    # Optional flags
    if crs:
        cmd_parts.append(f"--crs {crs}")
    if use_float64:
        cmd_parts.append("--use-float-64")
    if output_steps:
        cmd_parts.append(f"--output-steps {output_steps}")
    if seed is not None:
        cmd_parts.append(f"--seed {seed}")

    # Join with continuation lines
    lines.append(" \\\n    ".join(cmd_parts))
    lines.append("")

    return "\n".join(lines)


def _build_manifest(
    *,
    job: ExpandedJob | None = None,
    cli_result: CLIResult | None = None,
    cli: JoshCLI | None = None,
    run_hash: str,
    simulation: str,
    replicates: int,
    parameters: dict[str, Any],
    original_josh_path: str | None = None,
    original_data_paths: dict[str, str] | None = None,
    batch_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest.json content."""
    from joshpy.jar import get_jar_hash, get_jar_version

    jar_version = None
    jar_sha256 = None
    if cli is not None:
        try:
            jar_sha256 = get_jar_hash(cli._resolved_jar)
            jar_version = get_jar_version(
                cli._resolved_jar, java_path=cli.java_path
            )
        except Exception:
            pass

    manifest: dict[str, Any] = {
        "joshpy_version": _get_joshpy_version(),
        "jar_version": jar_version,
        "jar_sha256": jar_sha256,
        "run_hash": run_hash,
        "simulation": simulation,
        "replicates": replicates,
        "parameters": parameters,
        "python_version": sys.version,
        "platform": platform.platform(),
        "original_josh_path": original_josh_path,
        "original_data_paths": original_data_paths or {},
        "git_hash": _get_git_hash(),
        "bottled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if cli_result is not None:
        manifest["exit_code"] = cli_result.exit_code
        manifest["stderr"] = cli_result.stderr
        manifest["stdout"] = cli_result.stdout

    if batch_metadata:
        manifest["batch"] = batch_metadata

    return manifest


def create_bottle(
    job: ExpandedJob,
    cli_result: CLIResult | None = None,
    cli: JoshCLI | None = None,
    output_dir: str | Path = Path("bottles"),
    omit_jshd: bool = False,
    batch_metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a self-contained bottle archive from an ExpandedJob.

    Args:
        job: The expanded job to bottle.
        cli_result: Optional CLI result (exit code, stderr, stdout).
        cli: Optional JoshCLI instance for JAR metadata.
        output_dir: Directory for the archive. Default: ``./bottles/``.
        omit_jshd: If True, skip copying .jshd data files into the archive.
            The ``run.sh`` still lists ``--data`` flags so the recipient knows
            what files to provide. Default False — missing data files raise
            ``FileNotFoundError``.
        batch_metadata: Optional dict embedded under the manifest's ``batch``
            key. Use this when bottling a batch-remote job so reviewers can
            see the target and MinIO prefix used (e.g.
            ``{"target": "gke-test", "minio_prefix": "sweeps/.../jobs/abc/"}``).

    Returns:
        Path to the created ``.tar.gz`` archive.

    Raises:
        FileNotFoundError: If ``omit_jshd`` is False and a data file is missing.
    """
    from joshpy.jar import get_jar_hash, get_jar_version

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if job.label:
        bottle_name = f"bottle_{job.label}_{job.run_hash}"
    else:
        bottle_name = f"bottle_{job.run_hash}"
    staging = Path(tempfile.mkdtemp()) / bottle_name

    try:
        staging.mkdir(parents=True)

        # 1. simulation.josh
        if job.source_path and job.source_path.exists():
            shutil.copy2(job.source_path, staging / "simulation.josh")
        else:
            warnings.warn(
                f"Josh source not found at {job.source_path}; "
                f"simulation.josh will be missing from bottle",
                stacklevel=2,
            )

        # 2. Config file (rendered .jshc)
        config_name = job.config_name
        if not config_name.endswith(".jshc"):
            config_name = config_name + ".jshc"
        (staging / config_name).write_text(job.config_content)

        # 3. Data files
        data_staging = staging / "data"
        data_rel_paths: dict[str, str] = {}
        original_data_paths: dict[str, str] = {}
        for name, src_path in job.file_mappings.items():
            original_data_paths[name] = str(src_path)
            data_rel_paths[name] = f"data/{src_path.name}"
            if omit_jshd:
                continue
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Data file '{name}' not found at {src_path}. "
                    f"Use omit_jshd=True to create a bottle without data files."
                )
            data_staging.mkdir(exist_ok=True)
            shutil.copy2(src_path, data_staging / src_path.name)

        # 4. JAR metadata
        jar_version = None
        jar_sha256 = None
        if cli is not None:
            try:
                jar_sha256 = get_jar_hash(cli._resolved_jar)
                jar_version = get_jar_version(
                    cli._resolved_jar, java_path=cli.java_path
                )
            except Exception:
                pass

        # 5. run.sh
        run_sh = _build_run_sh(
            simulation=job.simulation,
            replicates=job.replicates,
            config_name=config_name,
            data_files=data_rel_paths,
            custom_tags=job.custom_tags,
            run_hash=job.run_hash,
            jar_version=jar_version,
            jar_sha256=jar_sha256,
            seed=job.seed,
            crs=job.crs,
            use_float64=job.use_float64,
            output_steps=job.output_steps,
        )
        run_sh_path = staging / "run.sh"
        run_sh_path.write_text(run_sh)
        run_sh_path.chmod(0o755)

        # 6. manifest.json
        manifest = _build_manifest(
            job=job,
            cli_result=cli_result,
            cli=cli,
            run_hash=job.run_hash,
            simulation=job.simulation,
            replicates=job.replicates,
            parameters=job.parameters,
            original_josh_path=str(job.source_path) if job.source_path else None,
            original_data_paths=original_data_paths,
            batch_metadata=batch_metadata,
        )
        manifest["omit_jshd"] = omit_jshd
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )

        # 7. Create .tar.gz
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_name = f"{bottle_name}_{timestamp}.tar.gz"
        archive_path = output_dir / archive_name

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging, arcname=bottle_name)

        return archive_path

    finally:
        shutil.rmtree(staging.parent, ignore_errors=True)


def create_bottle_from_registry(
    registry: RunRegistry,
    label_or_hash: str,
    cli: JoshCLI | None = None,
    output_dir: str | Path = Path("bottles"),
    omit_jshd: bool = False,
) -> Path:
    """Create a bottle archive from data stored in the registry.

    Args:
        registry: RunRegistry to look up run data.
        label_or_hash: Run label or run_hash.
        cli: Optional JoshCLI instance for JAR metadata.
        output_dir: Directory for the archive. Default: ``./bottles/``.
        omit_jshd: If True, skip copying .jshd data files. Default False —
            missing data files raise ``FileNotFoundError``.

    Returns:
        Path to the created ``.tar.gz`` archive.

    Raises:
        KeyError: If the run is not found in the registry.
        ValueError: If josh content is not stored for the run.
        FileNotFoundError: If ``omit_jshd`` is False and a data file is missing.
    """
    from joshpy.jar import get_jar_hash, get_jar_version

    # Resolve label → hash
    try:
        run_hash = registry.resolve_label(label_or_hash)
    except KeyError:
        run_hash = label_or_hash

    config = registry.get_config_by_hash(run_hash)
    if config is None:
        raise KeyError(f"No run found for '{label_or_hash}'")

    if config.josh_content is None:
        raise ValueError(
            f"No josh content stored for run '{label_or_hash}'. "
            f"Cannot create bottle without the josh source."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.label:
        bottle_name = f"bottle_{config.label}_{run_hash}"
    else:
        bottle_name = f"bottle_{run_hash}"
    staging = Path(tempfile.mkdtemp()) / bottle_name

    try:
        staging.mkdir(parents=True)

        # 1. simulation.josh (from stored content)
        (staging / "simulation.josh").write_text(config.josh_content)

        # 2. Config file
        config_name = "sweep_config.jshc"
        (staging / config_name).write_text(config.config_content)

        # 3. Data files (from file_mappings paths)
        data_staging = staging / "data"
        data_rel_paths: dict[str, str] = {}
        original_data_paths: dict[str, str] = {}
        if config.file_mappings:
            for name, info in config.file_mappings.items():
                src_path = Path(info["path"])
                original_data_paths[name] = str(src_path)
                data_rel_paths[name] = f"data/{src_path.name}"
                if omit_jshd:
                    continue
                if not src_path.exists():
                    raise FileNotFoundError(
                        f"Data file '{name}' not found at {src_path}. "
                        f"Use omit_jshd=True to create a bottle without data files."
                    )
                data_staging.mkdir(exist_ok=True)
                shutil.copy2(src_path, data_staging / src_path.name)

        # 4. JAR metadata
        jar_version = None
        jar_sha256 = None
        if cli is not None:
            try:
                jar_sha256 = get_jar_hash(cli._resolved_jar)
                jar_version = get_jar_version(
                    cli._resolved_jar, java_path=cli.java_path
                )
            except Exception:
                pass

        # 5. Get run info for exit_code/stderr
        session = registry.get_session(config.session_id)
        simulation = session.simulation if session else "Main"
        replicates = (session.total_replicates if session and session.total_replicates else 1)

        # Look up latest run result
        runs = registry.get_runs_for_hash(run_hash)
        exit_code = None
        stderr = None
        stdout = None
        if runs:
            latest = runs[-1]
            exit_code = latest.exit_code
            stderr = latest.error_message

        # 6. run.sh
        run_sh = _build_run_sh(
            simulation=simulation,
            replicates=replicates,
            config_name=config_name,
            data_files=data_rel_paths,
            custom_tags={
                **(
                    {k: str(v) for k, v in config.parameters.items()}
                    if config.parameters
                    else {}
                ),
                # SweepManager injects these at runtime; replicate here
                "run_hash": run_hash,
                **({"label": config.label} if config.label else {}),
            },
            run_hash=run_hash,
            jar_version=jar_version,
            jar_sha256=jar_sha256,
        )
        run_sh_path = staging / "run.sh"
        run_sh_path.write_text(run_sh)
        run_sh_path.chmod(0o755)

        # 7. manifest.json
        manifest: dict[str, Any] = {
            "joshpy_version": _get_joshpy_version(),
            "jar_version": jar_version,
            "jar_sha256": jar_sha256,
            "run_hash": run_hash,
            "simulation": simulation,
            "replicates": replicates,
            "parameters": config.parameters,
            "python_version": sys.version,
            "platform": platform.platform(),
            "original_josh_path": config.josh_path,
            "original_data_paths": original_data_paths,
            "git_hash": _get_git_hash(),
            "bottled_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        manifest["omit_jshd"] = omit_jshd
        if exit_code is not None:
            manifest["exit_code"] = exit_code
        if stderr is not None:
            manifest["stderr"] = stderr

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )

        # 8. Create .tar.gz
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_name = f"{bottle_name}_{timestamp}.tar.gz"
        archive_path = output_dir / archive_name

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging, arcname=bottle_name)

        return archive_path

    finally:
        shutil.rmtree(staging.parent, ignore_errors=True)


def create_sweep_bottle(
    job_results: list[tuple[ExpandedJob, CLIResult]],
    cli: JoshCLI | None = None,
    output_dir: str | Path = Path("bottles"),
    omit_jshd: bool = False,
    batch_metadata: dict[str, Any] | None = None,
    per_job_batch_metadata: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Create a single bottle archive containing multiple sweep jobs.

    Data files are shared across jobs (copied once into ``data/``).
    Each job gets its own subdirectory under ``jobs/`` with its rendered
    source, config, and ``run.sh``.

    Archive structure::

        bottle_sweep_{timestamp}/
            data/                        # shared .jshd files (copied once)
            jobs/
                {run_hash_1}/
                    simulation.josh
                    sweep_config.jshc
                    run.sh               # --data paths point to ../../data/
                {run_hash_2}/
                    ...
            manifest.json                # metadata for all jobs

    Args:
        job_results: List of (ExpandedJob, CLIResult) tuples.
        cli: Optional JoshCLI instance for JAR metadata.
        output_dir: Directory for the archive. Default: ``./bottles/``.
        omit_jshd: If True, skip copying .jshd data files.
        batch_metadata: Optional dict embedded under the top-level manifest's
            ``batch`` key. Use for sweep-wide context (e.g.
            ``{"target": "gke-test", "stage_prefix_root": "sweeps/s1/"}``).
        per_job_batch_metadata: Optional dict mapping ``run_hash`` to a
            per-job batch dict. When set, each job manifest entry gets a
            ``batch`` sub-key (useful for recording per-job MinIO prefixes).

    Returns:
        Path to the created ``.tar.gz`` archive.

    Raises:
        FileNotFoundError: If ``omit_jshd`` is False and a data file is missing.
        ValueError: If job_results is empty.
    """
    from joshpy.jar import get_jar_hash, get_jar_version

    if not job_results:
        raise ValueError("Cannot create sweep bottle from empty job_results")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bottle_name = f"bottle_sweep_{timestamp}"
    staging = Path(tempfile.mkdtemp()) / bottle_name

    try:
        staging.mkdir(parents=True)
        data_staging = staging / "data"
        jobs_staging = staging / "jobs"
        jobs_staging.mkdir()

        # JAR metadata (shared across all jobs)
        jar_version = None
        jar_sha256 = None
        if cli is not None:
            try:
                jar_sha256 = get_jar_hash(cli._resolved_jar)
                jar_version = get_jar_version(
                    cli._resolved_jar, java_path=cli.java_path
                )
            except Exception:
                pass

        # Collect all unique data files across jobs (deduplicate by path)
        all_data_files: dict[str, Path] = {}  # name -> src_path
        for job, _ in job_results:
            for name, src_path in job.file_mappings.items():
                if name not in all_data_files:
                    all_data_files[name] = src_path

        # Copy shared data files once
        original_data_paths: dict[str, str] = {}
        for name, src_path in all_data_files.items():
            original_data_paths[name] = str(src_path)
            if omit_jshd:
                continue
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Data file '{name}' not found at {src_path}. "
                    f"Use omit_jshd=True to create a bottle without data files."
                )
            data_staging.mkdir(exist_ok=True)
            shutil.copy2(src_path, data_staging / src_path.name)

        # Create per-job directories (deduplicate by run_hash)
        seen_hashes: set[str] = set()
        job_manifests: list[dict[str, Any]] = []
        for job, cli_result in job_results:
            if job.run_hash in seen_hashes:
                continue
            seen_hashes.add(job.run_hash)
            job_dir = jobs_staging / job.run_hash
            job_dir.mkdir()

            # simulation.josh
            if job.source_path and job.source_path.exists():
                shutil.copy2(job.source_path, job_dir / "simulation.josh")

            # Config
            config_name = job.config_name
            if not config_name.endswith(".jshc"):
                config_name = config_name + ".jshc"
            (job_dir / config_name).write_text(job.config_content)

            # Data file relative paths (point back to shared ../../data/)
            data_rel_paths: dict[str, str] = {}
            for name, src_path in job.file_mappings.items():
                data_rel_paths[name] = f"../../data/{src_path.name}"

            # run.sh
            run_sh = _build_run_sh(
                simulation=job.simulation,
                replicates=job.replicates,
                config_name=config_name,
                data_files=data_rel_paths,
                custom_tags=job.custom_tags,
                run_hash=job.run_hash,
                jar_version=jar_version,
                jar_sha256=jar_sha256,
                seed=job.seed,
                crs=job.crs,
                use_float64=job.use_float64,
                output_steps=job.output_steps,
            )
            run_sh_path = job_dir / "run.sh"
            run_sh_path.write_text(run_sh)
            run_sh_path.chmod(0o755)

            # Per-job manifest entry
            entry: dict[str, Any] = {
                "run_hash": job.run_hash,
                "parameters": job.parameters,
                "exit_code": cli_result.exit_code,
                "success": cli_result.success,
            }
            if not cli_result.success:
                entry["stderr"] = cli_result.stderr
            if per_job_batch_metadata and job.run_hash in per_job_batch_metadata:
                entry["batch"] = per_job_batch_metadata[job.run_hash]
            job_manifests.append(entry)

        # Top-level manifest
        first_job = job_results[0][0]
        manifest: dict[str, Any] = {
            "joshpy_version": _get_joshpy_version(),
            "jar_version": jar_version,
            "jar_sha256": jar_sha256,
            "simulation": first_job.simulation,
            "total_jobs": len(job_results),
            "succeeded": sum(1 for _, r in job_results if r.success),
            "failed": sum(1 for _, r in job_results if not r.success),
            "omit_jshd": omit_jshd,
            "original_data_paths": original_data_paths,
            "jobs": job_manifests,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_hash": _get_git_hash(),
            "bottled_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        if batch_metadata:
            manifest["batch"] = batch_metadata
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )

        # Create .tar.gz
        archive_name = f"{bottle_name}.tar.gz"
        archive_path = output_dir / archive_name

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging, arcname=bottle_name)

        return archive_path

    finally:
        shutil.rmtree(staging.parent, ignore_errors=True)


def _should_bottle(
    mode: str,
    success: bool,
    already_bottled_failure: bool,
    already_bottled_success: bool,
) -> bool:
    """Determine whether to bottle a job result given the mode and state."""
    if mode == "first_failure":
        return not success and not already_bottled_failure
    elif mode == "all_failures":
        return not success
    elif mode == "first_success":
        return success and not already_bottled_success
    elif mode == "all":
        return True
    return False


def _resolve_data_mappings(
    original_data_paths: dict[str, str],
    data_dir: Path | None,
    extracted_data_dir: Path,
) -> dict[str, Path]:
    """Resolve data file mappings from original paths, data_dir, or extracted data."""
    file_mappings: dict[str, Path] = {}

    if data_dir is not None:
        if original_data_paths:
            import os

            orig_paths = list(original_data_paths.values())
            try:
                common = Path(os.path.commonpath(orig_paths))
            except ValueError:
                common = Path("/")

            # commonpath of a single file returns the file itself;
            # we need the containing directory as the root
            if not common.is_dir():
                common = common.parent

            for name, orig_str in original_data_paths.items():
                orig = Path(orig_str)
                try:
                    rel = orig.relative_to(common)
                except ValueError:
                    rel = Path(orig.name)
                file_mappings[name] = data_dir / rel
    elif extracted_data_dir.exists():
        for name, orig_str in original_data_paths.items():
            local = extracted_data_dir / Path(orig_str).name
            if local.exists():
                file_mappings[name] = local

    return file_mappings


def unbottle(
    archive: str | Path,
    extract_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    run_hash: str | None = None,
) -> list[Any]:
    """Unpack a bottle archive into a list of JobConfigs ready for joshpy.

    Always returns a list, even for single-job bottles (which return a
    one-element list). This makes the return type predictable regardless
    of whether the archive is a single-job or sweep bottle.

    Args:
        archive: Path to the ``.tar.gz`` bottle archive.
        extract_dir: Directory to extract into. Defaults to a temp dir.
        data_dir: Local directory containing ``.jshd`` data files. When
            provided, replaces the original data root: the common prefix
            of all original paths is swapped for ``data_dir``, preserving
            subdirectory structure. When omitted, uses data files from
            the extracted ``data/`` subdirectory (if present).
        run_hash: For sweep bottles, select a specific job by run hash.
            Returns a one-element list. Ignored for single-job bottles.

    Returns:
        List of ``JobConfig`` objects. One element for single-job bottles
        or when ``run_hash`` filters to one job; multiple for sweep bottles.

    Raises:
        FileNotFoundError: If the archive does not exist.
        ValueError: If the archive has no manifest.json.
        KeyError: If ``run_hash`` is specified but not found in a sweep bottle.
    """
    from joshpy.jobs import JobConfig

    archive = Path(archive)
    if not archive.exists():
        raise FileNotFoundError(f"Bottle archive not found: {archive}")

    if extract_dir is None:
        extract_dir = Path(tempfile.mkdtemp(prefix="joshpy_unbottle_"))
    else:
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
        data_dir = Path(data_dir)

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)

    # Find the bottle root directory
    entries = list(extract_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        bottle_dir = entries[0]
    else:
        bottle_dir = extract_dir

    # Read manifest
    manifest_path = bottle_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"No manifest.json found in bottle at {bottle_dir}")
    manifest = json.loads(manifest_path.read_text())

    simulation = manifest.get("simulation", "Main")
    original_data_paths = manifest.get("original_data_paths", {})
    is_sweep = (bottle_dir / "jobs").is_dir()

    if is_sweep:
        # Sweep bottle: shared data + per-job directories
        shared_data = bottle_dir / "data"
        data_mappings = _resolve_data_mappings(
            original_data_paths, data_dir, shared_data
        )

        jobs_dir = bottle_dir / "jobs"
        job_dirs = sorted(d for d in jobs_dir.iterdir() if d.is_dir())

        if run_hash is not None:
            # Select a specific job
            match = [d for d in job_dirs if d.name == run_hash]
            if not match:
                available = [d.name for d in job_dirs]
                raise KeyError(
                    f"Run hash '{run_hash}' not found in sweep bottle. "
                    f"Available: {available}"
                )
            job_dirs = match

        configs = []
        for job_dir in job_dirs:
            source = job_dir / "simulation.josh"
            config_files = list(job_dir.glob("*.jshc"))
            configs.append(
                JobConfig(
                    source_path=source if source.exists() else None,
                    config_path=config_files[0] if config_files else None,
                    simulation=simulation,
                    file_mappings=dict(data_mappings),
                )
            )

        return configs

    else:
        # Single-job bottle
        source_path = bottle_dir / "simulation.josh"
        config_files = list(bottle_dir.glob("*.jshc"))
        config_path = config_files[0] if config_files else None

        extracted_data = bottle_dir / "data"
        file_mappings = _resolve_data_mappings(
            original_data_paths, data_dir, extracted_data
        )

        return [
            JobConfig(
                source_path=source_path if source_path.exists() else None,
                config_path=config_path,
                simulation=simulation,
                file_mappings=file_mappings,
            )
        ]
