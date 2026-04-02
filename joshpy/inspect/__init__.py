"""Viewing and diffing utilities for run configurations and josh sources.

Usage from Python::

    registry.compare_configs("baseline", "high_growth")
    registry.compare_josh("baseline", "high_growth")

Usage from the command line::

    python -m joshpy.inspect registry.duckdb --view baseline
    python -m joshpy.inspect registry.duckdb --view baseline --export-only
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth --ide cursor
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth --export-only
    python -m joshpy.inspect registry.duckdb --view baseline --type josh
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth --type josh
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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
            resolved_label = registry.resolve_label(label_or_hash)
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

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="joshpy_inspect_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    path = _resolve_path_for_run(registry, label_or_hash, run_hash, output_dir)
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
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="joshpy_inspect_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    path1 = registry.export_config(label_or_hash_1, output_dir)
    path2 = registry.export_config(label_or_hash_2, output_dir)
    return path1, path2


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
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="joshpy_inspect_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    run_hash_1 = _resolve_label(registry, label_or_hash_1)
    run_hash_2 = _resolve_label(registry, label_or_hash_2)

    path1 = _resolve_path_for_run(registry, label_or_hash_1, run_hash_1, output_dir)
    path2 = _resolve_path_for_run(registry, label_or_hash_2, run_hash_2, output_dir)

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

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="joshpy_inspect_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    path = _resolve_josh_path_for_run(registry, label_or_hash, run_hash, output_dir)
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
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="joshpy_inspect_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    path1 = registry.export_josh(label_or_hash_1, output_dir)
    path2 = registry.export_josh(label_or_hash_2, output_dir)
    return path1, path2


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
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="joshpy_inspect_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    run_hash_1 = _resolve_label(registry, label_or_hash_1)
    run_hash_2 = _resolve_label(registry, label_or_hash_2)

    path1 = _resolve_josh_path_for_run(
        registry, label_or_hash_1, run_hash_1, output_dir
    )
    path2 = _resolve_josh_path_for_run(
        registry, label_or_hash_2, run_hash_2, output_dir
    )

    _launch_ide(ide, ["--diff", str(path1), str(path2)])
    return path1, path2
