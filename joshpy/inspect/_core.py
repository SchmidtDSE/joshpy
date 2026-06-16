"""Implementation of viewing and diffing utilities for run configurations and josh sources."""

from __future__ import annotations

import difflib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from joshpy.registry import RunRegistry

# Map IDE names to their CLI commands.
IDE_COMMANDS: dict[str, str] = {
    "vscode": "code",
    "cursor": "cursor",
}


def _launch_ide(ide: str, args: list[str]) -> None:
    """Validate and launch an IDE with the given arguments.

    Raises:
        ValueError: If *ide* is not a supported IDE name.
        RuntimeError: If the IDE CLI is not found in PATH.
    """
    cli_name = IDE_COMMANDS.get(ide)
    if cli_name is None:
        supported = ", ".join(sorted(IDE_COMMANDS.keys()))
        raise ValueError(f"Unknown IDE '{ide}'. Supported: {supported}")

    cli_path = shutil.which(cli_name)
    if cli_path is None:
        raise RuntimeError(
            f"'{cli_name}' CLI not found in PATH. "
            f"To install the VS Code CLI: open VS Code, then "
            f"Cmd/Ctrl+Shift+P > 'Shell Command: Install code command "
            f"in PATH'."
        )

    subprocess.run([cli_path, *args], check=False)


def _resolve_label(registry: RunRegistry, label_or_hash: str) -> str:
    """Resolve a label to a run_hash, falling back to treating it as a hash."""
    try:
        return registry.resolve_label(label_or_hash)
    except KeyError:
        return label_or_hash


def _make_output_dir(output_dir: str | Path | None) -> Path:
    """Resolve or create an output directory."""
    if output_dir is None:
        return Path(tempfile.mkdtemp(prefix="joshpy_inspect_"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ========== Config (.jshc) ==========


def _resolve_path_for_run(
    registry: RunRegistry,
    label_or_hash: str,
    run_hash: str,
    output_dir: Path,
) -> Path:
    """Resolve the best file path to open for a run.

    If the original .jshc file still exists on disk with matching content,
    returns that path directly. If it exists but content has drifted, falls
    back to a temp export and prints a note to stderr. If not found, falls
    back to a temp export silently.
    """
    source = registry.resolve_config_source(run_hash)

    if source.exists and source.content_matches:
        return source.path

    if source.exists and not source.content_matches:
        # Export to temp, but swap the header to mention the real file
        config_info = registry.get_config_by_hash(run_hash)
        if config_info is None:
            raise KeyError(f"No run found for '{label_or_hash}'")
        try:
            registry.resolve_label(label_or_hash)
            filename = f"{label_or_hash}.jshc"
        except KeyError:
            filename = f"{run_hash}.jshc"
        header = (
            f"# READ-ONLY snapshot exported from registry\n"
            f"# Run: {run_hash}\n"
            f"#\n"
            f"# Note: source file still exists at {source.path}\n"
            f"# but has been modified since this run was registered.\n"
            f"# This is the stored version. Edit the source file to make changes.\n\n"
        )
        out_path = output_dir / filename
        out_path.write_text(header + config_info.config_content)
        return out_path

    return registry.export_config(label_or_hash, output_dir)


def view_config(
    registry: RunRegistry,
    label_or_hash: str,
) -> str:
    """Retrieve the stored config content for a run.

    Args:
        registry: RunRegistry instance.
        label_or_hash: Run label or run_hash.

    Returns:
        The config content as a string.

    Raises:
        KeyError: If the label or hash is not found in the registry.
    """
    run_hash = _resolve_label(registry, label_or_hash)

    config = registry.get_config_by_hash(run_hash)
    if config is None:
        raise KeyError(f"No run found for '{label_or_hash}'")
    return config.config_content


def open_view(
    registry: RunRegistry,
    label_or_hash: str,
    ide: str = "vscode",
    output_dir: str | Path | None = None,
) -> Path:
    """Open a run's config in the IDE.

    If the original ``.jshc`` file still exists on disk with matching
    content, opens it directly (editable). Otherwise exports a read-only
    snapshot to a temp file.

    Args:
        registry: RunRegistry instance.
        label_or_hash: Run label or run_hash.
        ide: IDE to open file in (``"vscode"`` or ``"cursor"``).
        output_dir: Directory for exported file if needed. Uses a tempdir if None.

    Returns:
        Path that was opened (real file or temp export).

    Raises:
        KeyError: If the label or hash is not found in the registry.
        ValueError: If *ide* is not a supported IDE name.
        RuntimeError: If the IDE CLI is not found in PATH.
    """
    run_hash = _resolve_label(registry, label_or_hash)
    out = _make_output_dir(output_dir)
    path = _resolve_path_for_run(registry, label_or_hash, run_hash, out)
    _launch_ide(ide, [str(path)])
    return path


def export_pair(
    registry: RunRegistry,
    label_or_hash_1: str,
    label_or_hash_2: str,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export two run configs to files for diffing.

    Always writes to temp files with READ-ONLY headers (explicit snapshot).

    Args:
        registry: RunRegistry instance.
        label_or_hash_1: First run label or run_hash.
        label_or_hash_2: Second run label or run_hash.
        output_dir: Directory for exported files. Uses a tempdir if None.

    Returns:
        Tuple of (path1, path2) to the exported config files.

    Raises:
        KeyError: If a label or hash is not found in the registry.
    """
    out = _make_output_dir(output_dir)
    path1 = registry.export_config(label_or_hash_1, out)
    path2 = registry.export_config(label_or_hash_2, out)
    return path1, path2


def text_diff(
    registry: RunRegistry,
    label_or_hash_1: str,
    label_or_hash_2: str,
) -> str:
    """Produce a unified text diff of two runs' stored configs.

    Works headless (terminal, CI, SSH) — no IDE required.

    Args:
        registry: RunRegistry instance.
        label_or_hash_1: First run label or run_hash.
        label_or_hash_2: Second run label or run_hash.

    Returns:
        Unified diff as a string (empty if the two configs are identical).

    Raises:
        KeyError: If a label or hash is not found in the registry.
    """
    a = view_config(registry, label_or_hash_1).splitlines(keepends=True)
    b = view_config(registry, label_or_hash_2).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(a, b, fromfile=label_or_hash_1, tofile=label_or_hash_2)
    )


def open_diff(
    registry: RunRegistry,
    label_or_hash_1: str,
    label_or_hash_2: str,
    ide: str = "vscode",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Open a side-by-side diff of two run configs in the IDE.

    For each run, if the original ``.jshc`` file still exists on disk with
    matching content, uses that path directly. Otherwise exports a read-only
    snapshot.

    Args:
        registry: RunRegistry instance.
        label_or_hash_1: First run label or run_hash.
        label_or_hash_2: Second run label or run_hash.
        ide: IDE to open diff in (``"vscode"`` or ``"cursor"``).
        output_dir: Directory for exported files if needed. Uses a tempdir if None.

    Returns:
        Tuple of (path1, path2) to the opened config files.

    Raises:
        KeyError: If a label or hash is not found in the registry.
        ValueError: If *ide* is not a supported IDE name.
        RuntimeError: If the IDE CLI is not found in PATH.
    """
    out = _make_output_dir(output_dir)
    run_hash_1 = _resolve_label(registry, label_or_hash_1)
    run_hash_2 = _resolve_label(registry, label_or_hash_2)
    path1 = _resolve_path_for_run(registry, label_or_hash_1, run_hash_1, out)
    path2 = _resolve_path_for_run(registry, label_or_hash_2, run_hash_2, out)
    _launch_ide(ide, ["--diff", str(path1), str(path2)])
    return path1, path2


# ========== Josh source (.josh) ==========


def _resolve_josh_path_for_run(
    registry: RunRegistry,
    label_or_hash: str,
    run_hash: str,
    output_dir: Path,
) -> Path:
    """Resolve the best file path to open for a run's josh source.

    If the original .josh file still exists on disk with matching content,
    returns that path directly. If it exists but content has drifted, falls
    back to a temp export. If not found, falls back to a temp export.
    """
    source = registry.resolve_josh_source(run_hash)

    if source.exists and source.content_matches:
        return source.path

    if source.exists and not source.content_matches:
        config_info = registry.get_config_by_hash(run_hash)
        if config_info is None:
            raise KeyError(f"No run found for '{label_or_hash}'")
        try:
            registry.resolve_label(label_or_hash)
            filename = f"{label_or_hash}.josh"
        except KeyError:
            filename = f"{run_hash}.josh"
        header = (
            f"# READ-ONLY snapshot exported from registry\n"
            f"# Run: {run_hash}\n"
            f"#\n"
            f"# Note: source file still exists at {source.path}\n"
            f"# but has been modified since this run was registered.\n"
            f"# This is the stored version.\n\n"
        )
        out_path = output_dir / filename
        out_path.write_text(header + config_info.josh_content)
        return out_path

    return registry.export_josh(label_or_hash, output_dir)


def view_josh(
    registry: RunRegistry,
    label_or_hash: str,
) -> str:
    """Retrieve the stored josh source content for a run.

    Args:
        registry: RunRegistry instance.
        label_or_hash: Run label or run_hash.

    Returns:
        The josh source content as a string.

    Raises:
        KeyError: If the label or hash is not found, or josh content
            is not stored.
    """
    run_hash = _resolve_label(registry, label_or_hash)

    config = registry.get_config_by_hash(run_hash)
    if config is None:
        raise KeyError(f"No run found for '{label_or_hash}'")
    if config.josh_content is None:
        raise KeyError(f"No josh content stored for run '{label_or_hash}'")
    return config.josh_content


def open_josh_view(
    registry: RunRegistry,
    label_or_hash: str,
    ide: str = "vscode",
    output_dir: str | Path | None = None,
) -> Path:
    """Open a run's josh source in the IDE.

    If the original ``.josh`` file still exists on disk with matching
    content, opens it directly. Otherwise exports a read-only snapshot.

    Args:
        registry: RunRegistry instance.
        label_or_hash: Run label or run_hash.
        ide: IDE to open file in (``"vscode"`` or ``"cursor"``).
        output_dir: Directory for exported file if needed. Uses a tempdir if None.

    Returns:
        Path that was opened (real file or temp export).

    Raises:
        KeyError: If the label or hash is not found, or josh content
            is not stored.
        ValueError: If *ide* is not a supported IDE name.
        RuntimeError: If the IDE CLI is not found in PATH.
    """
    run_hash = _resolve_label(registry, label_or_hash)
    out = _make_output_dir(output_dir)
    path = _resolve_josh_path_for_run(registry, label_or_hash, run_hash, out)
    _launch_ide(ide, [str(path)])
    return path


def export_josh_pair(
    registry: RunRegistry,
    label_or_hash_1: str,
    label_or_hash_2: str,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export two runs' josh sources to files for diffing.

    Args:
        registry: RunRegistry instance.
        label_or_hash_1: First run label or run_hash.
        label_or_hash_2: Second run label or run_hash.
        output_dir: Directory for exported files. Uses a tempdir if None.

    Returns:
        Tuple of (path1, path2) to the exported josh files.

    Raises:
        KeyError: If a label or hash is not found, or josh content
            is not stored.
    """
    out = _make_output_dir(output_dir)
    path1 = registry.export_josh(label_or_hash_1, out)
    path2 = registry.export_josh(label_or_hash_2, out)
    return path1, path2


def text_josh_diff(
    registry: RunRegistry,
    label_or_hash_1: str,
    label_or_hash_2: str,
) -> str:
    """Produce a unified text diff of two runs' stored josh sources.

    Works headless (terminal, CI, SSH) — no IDE required.

    Args:
        registry: RunRegistry instance.
        label_or_hash_1: First run label or run_hash.
        label_or_hash_2: Second run label or run_hash.

    Returns:
        Unified diff as a string (empty if the two sources are identical).

    Raises:
        KeyError: If a label or hash is not found, or josh content is not stored.
    """
    a = view_josh(registry, label_or_hash_1).splitlines(keepends=True)
    b = view_josh(registry, label_or_hash_2).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(a, b, fromfile=label_or_hash_1, tofile=label_or_hash_2)
    )


def open_josh_diff(
    registry: RunRegistry,
    label_or_hash_1: str,
    label_or_hash_2: str,
    ide: str = "vscode",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Open a side-by-side diff of two runs' josh sources in the IDE.

    For each run, if the original ``.josh`` file still exists on disk with
    matching content, uses that path directly. Otherwise exports a read-only
    snapshot.

    Args:
        registry: RunRegistry instance.
        label_or_hash_1: First run label or run_hash.
        label_or_hash_2: Second run label or run_hash.
        ide: IDE to open diff in (``"vscode"`` or ``"cursor"``).
        output_dir: Directory for exported files if needed. Uses a tempdir if None.

    Returns:
        Tuple of (path1, path2) to the opened josh files.

    Raises:
        KeyError: If a label or hash is not found, or josh content
            is not stored.
        ValueError: If *ide* is not a supported IDE name.
        RuntimeError: If the IDE CLI is not found in PATH.
    """
    out = _make_output_dir(output_dir)
    run_hash_1 = _resolve_label(registry, label_or_hash_1)
    run_hash_2 = _resolve_label(registry, label_or_hash_2)
    path1 = _resolve_josh_path_for_run(
        registry, label_or_hash_1, run_hash_1, out
    )
    path2 = _resolve_josh_path_for_run(
        registry, label_or_hash_2, run_hash_2, out
    )
    _launch_ide(ide, ["--diff", str(path1), str(path2)])
    return path1, path2


# ========== Query Utilities ==========


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a list of rows as an aligned table with headers."""
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    for row in rows:
        lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    return "\n".join(lines)


def format_labels(registry: RunRegistry) -> str:
    """List all labeled runs with their run_hash and creation date.

    Args:
        registry: RunRegistry instance.

    Returns:
        Formatted table string, or a message if no labels exist.
    """
    labels = registry.list_labels()
    if not labels:
        return "No labels found."

    rows: list[list[str]] = []
    for label, run_hash in labels:
        config = registry.get_config_by_hash(run_hash)
        created = (
            config.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if config and config.created_at
            else "-"
        )
        # Derive replicate count from cell_data (source of truth for pooled runs).
        # Fall back to job_runs count when cell_data hasn't been loaded yet.
        rep_count = registry.get_replicate_count(run_hash)
        if rep_count == 0:
            rep_count = len(registry.get_runs_for_hash(run_hash))
        reps = str(rep_count)
        rows.append([label, run_hash, reps, created])

    return _format_table(["LABEL", "RUN_HASH", "REPS", "CREATED"], rows)


def format_sessions(registry: RunRegistry) -> str:
    """List all sessions with experiment name, status, and run counts.

    Args:
        registry: RunRegistry instance.

    Returns:
        Formatted table string, or a message if no sessions exist.
    """
    sessions = registry.list_sessions()
    if not sessions:
        return "No sessions found."

    rows: list[list[str]] = []
    for s in sessions:
        summary = registry.get_session_summary(s.session_id)
        if summary:
            runs = f"{summary.runs_succeeded}/{summary.runs_failed}/{summary.runs_pending}"
            jobs = str(summary.total_jobs)
        else:
            runs = "-"
            jobs = "-"
        sid = s.session_id[:8] + "..." if len(s.session_id) > 11 else s.session_id
        experiment = s.experiment_name or "(unnamed)"
        created = (
            s.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if s.created_at
            else "-"
        )
        rows.append([sid, experiment, s.status, jobs, runs, created])

    return _format_table(
        ["SESSION", "EXPERIMENT", "STATUS", "JOBS", "RUNS (ok/fail/pend)", "CREATED"],
        rows,
    )


def format_run_info(registry: RunRegistry, label_or_hash: str) -> str:
    """Show detailed info for a single run.

    Args:
        registry: RunRegistry instance.
        label_or_hash: Run label or run_hash.

    Returns:
        Multi-section detail string.

    Raises:
        KeyError: If the label or hash is not found in the registry.
    """
    # get_run_info aggregates config + runs + replicate count and raises
    # KeyError if the run is not found.
    detail = registry.get_run_info(label_or_hash)
    config = detail.config
    run_hash = detail.run_hash

    # Header
    if config.label:
        header = f"Run: {config.label} ({run_hash})"
    else:
        header = f"Run: {run_hash}"
    lines = [header, "=" * len(header)]

    # Metadata
    lines.append(f"Session:  {config.session_id}")
    lines.append(f"Josh:     {config.josh_path or '(not stored)'}")
    created = (
        config.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if config.created_at
        else "-"
    )
    lines.append(f"Created:  {created}")

    # Parameters
    lines.append("")
    lines.append("Parameters:")
    if config.parameters:
        for key, value in sorted(config.parameters.items()):
            lines.append(f"  {key} = {value}")
    else:
        lines.append("  (none)")

    # Data files
    lines.append("")
    lines.append("Data files:")
    if config.file_mappings:
        for name, info in sorted(config.file_mappings.items()):
            path = info.get("path", "?") if isinstance(info, dict) else str(info)
            lines.append(f"  {name}: {path}")
    else:
        lines.append("  (none)")

    # Replicates (from cell_data, the source of truth for pooled runs)
    rep_count = detail.replicate_count
    if rep_count > 0:
        lines.append("")
        lines.append(f"Replicates: {rep_count}")

    # Runs
    runs = detail.runs
    lines.append("")
    if not runs:
        lines.append("Runs: (none recorded)")
    else:
        lines.append(
            f"Runs: {detail.succeeded} succeeded, "
            f"{detail.failed} failed, {detail.pending} pending"
        )

        run_rows: list[list[str]] = []
        for r in runs:
            rep = str(r.replicate)
            exit_code = str(r.exit_code) if r.exit_code is not None else "-"
            started = (
                r.started_at.strftime("%Y-%m-%d %H:%M:%S")
                if r.started_at
                else "-"
            )
            output = r.output_path or "-"
            run_rows.append([rep, exit_code, started, output])
        lines.append(_format_table(["REP", "EXIT", "STARTED", "OUTPUT"], run_rows))

    return "\n".join(lines)


def format_summary(registry: RunRegistry) -> str:
    """Print a data summary for the entire registry.

    Args:
        registry: RunRegistry instance.

    Returns:
        Human-readable summary string.
    """
    return str(registry.get_data_summary())
