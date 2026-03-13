"""JSHD file loading and visualization utilities.

This module provides tools for loading and visualizing Josh's preprocessed
binary data files (.jshd). The primary use case is debugging preprocessing
to verify that NetCDF/GeoTIFF data was correctly interpolated to the simulation
grid.

Future versions may support bidirectional conversion between JSHD and Python
objects (xarray/pandas).

Example:
    >>> from joshpy import JoshCLI, load_jshd, plot_jshd
    >>> cli = JoshCLI()
    >>> data = load_jshd(cli, Path("soil_quality.jshd"))
    >>> plot_jshd(data, timestep=0)

Requires: pandas, matplotlib, numpy
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from joshpy.cli import JoshCLI


@dataclass
class JshdMetadata:
    """Metadata from JSHD file inspection via --to-csv.

    Contains grid bounds, timestep range, units, and the path to the
    CSV file written by the CLI command.

    Attributes:
        min_x: Minimum X grid index (inclusive).
        max_x: Maximum X grid index (inclusive).
        min_y: Minimum Y grid index (inclusive).
        max_y: Maximum Y grid index (inclusive).
        min_timestep: Minimum timestep (inclusive).
        max_timestep: Maximum timestep (inclusive).
        width: Grid width (max_x - min_x + 1).
        height: Grid height (max_y - min_y + 1).
        units: Unit string from JSHD header (may be empty string).
        csv_path: Absolute path to the written CSV file.
    """

    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_timestep: int
    max_timestep: int
    width: int
    height: int
    units: str
    csv_path: Path

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps in the file."""
        return self.max_timestep - self.min_timestep + 1

    @property
    def num_cells(self) -> int:
        """Total number of grid cells."""
        return self.width * self.height

    @classmethod
    def from_json(cls, data: dict) -> JshdMetadata:
        """Parse from JSON dict returned by CLI.

        Args:
            data: Dictionary parsed from CLI stdout JSON.

        Returns:
            JshdMetadata instance.
        """
        return cls(
            min_x=data["minX"],
            max_x=data["maxX"],
            min_y=data["minY"],
            max_y=data["maxY"],
            min_timestep=data["minTimestep"],
            max_timestep=data["maxTimestep"],
            width=data["width"],
            height=data["height"],
            units=data["units"],
            csv_path=Path(data["csv"]),
        )


@dataclass
class JshdData:
    """JSHD file contents as Python objects.

    Contains the metadata and a DataFrame with all values from the JSHD file.
    The DataFrame has columns: x, y, timestep, value.

    Attributes:
        metadata: Grid and timestep metadata.
        df: DataFrame with columns [x, y, timestep, value].
        source_path: Original JSHD file path (for display in plots).
    """

    metadata: JshdMetadata
    df: pd.DataFrame
    source_path: Path | None = None
    _temp_dir: Path | None = field(default=None, repr=False)

    def to_array(self, timestep: int | None = None) -> np.ndarray:
        """Convert to numpy array.

        The CSV row order is timestep → y → x (x varies fastest), so the
        values can be directly reshaped to (T, H, W) or (H, W).

        Args:
            timestep: If provided, return 2D array (height, width) for that
                timestep. If None, return 3D array (num_timesteps, height, width).

        Returns:
            Array with values reshaped to match grid dimensions.

        Raises:
            ValueError: If timestep is out of range.
        """
        meta = self.metadata

        if timestep is not None:
            if timestep < meta.min_timestep or timestep > meta.max_timestep:
                raise ValueError(
                    f"timestep {timestep} out of range [{meta.min_timestep}, {meta.max_timestep}]"
                )
            mask = self.df["timestep"] == timestep
            values = np.asarray(self.df.loc[mask, "value"].values)
            return values.reshape(meta.height, meta.width)
        else:
            values = np.asarray(self.df["value"].values)
            return values.reshape(meta.num_timesteps, meta.height, meta.width)

    def cleanup(self) -> None:
        """Remove temporary files created during inspection.

        Call this when you're done with the data to free disk space.
        Only removes files if they were created in a temp directory.
        """
        if self._temp_dir is not None and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None


def load_jshd(
    cli: JoshCLI,
    jshd_path: Path,
    output_csv: Path | None = None,
    variable: str = "data",
    timeout: float | None = None,
    cleanup: bool = True,
) -> JshdData:
    """Load JSHD contents as JshdData with DataFrame and metadata.

    Uses the CLI's ``inspectJshd --to-csv`` mode to extract all data from
    a JSHD file into a pandas DataFrame with metadata.

    Args:
        cli: JoshCLI instance to use for execution.
        jshd_path: Path to the .jshd file to load.
        output_csv: Path for CSV output. If None, uses a temp file.
        variable: Variable name in JSHD (typically "data").
        timeout: Command timeout in seconds.
        cleanup: If True and output_csv is None, delete the temp CSV after
            loading. If False, temp files are preserved and can be cleaned
            up later via JshdData.cleanup().

    Returns:
        JshdData containing metadata and DataFrame.

    Raises:
        RuntimeError: If CLI command fails.
        FileNotFoundError: If JSHD file doesn't exist.

    Example:
        >>> cli = JoshCLI()
        >>> data = load_jshd(cli, Path("temperature.jshd"))
        >>> print(f"Grid size: {data.metadata.width} x {data.metadata.height}")
        >>> arr = data.to_array(timestep=0)
    """
    jshd_path = Path(jshd_path)
    if not jshd_path.exists():
        raise FileNotFoundError(f"JSHD file not found: {jshd_path}")

    # Track temp directory for cleanup
    temp_dir: Path | None = None

    # Use temp file if no output path specified
    if output_csv is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="jshd_inspect_"))
        output_csv = temp_dir / "dump.csv"

    # Build command args
    # Note: --to-csv requires '=' syntax (--to-csv=path, not --to-csv path)
    args = [
        "inspectJshd",
        str(jshd_path.resolve()),
        variable,
        f"--to-csv={output_csv.resolve()}",
    ]

    result = cli._execute(args, timeout=timeout)

    if not result.success:
        # Clean up temp dir on failure
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise RuntimeError(
            f"inspectJshd --to-csv failed (exit {result.exit_code}): {result.stderr}"
        )

    # Parse JSON metadata from stdout
    try:
        metadata = JshdMetadata.from_json(json.loads(result.stdout))
    except (json.JSONDecodeError, KeyError) as e:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to parse CLI JSON output: {e}\nOutput: {result.stdout}")

    # Load CSV
    df = pd.read_csv(metadata.csv_path)

    data = JshdData(
        metadata=metadata,
        df=df,
        source_path=jshd_path,
        _temp_dir=temp_dir if not cleanup else None,
    )

    # Cleanup temp files if requested
    if cleanup and temp_dir is not None:
        shutil.rmtree(temp_dir)

    return data


def plot_jshd(
    data: JshdData,
    timestep: int = 0,
    cmap: str = "viridis",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    show: bool = True,
) -> Figure:
    """Plot JSHD data as a debug heatmap with metadata annotation.

    Creates a visualization showing:

    - Heatmap of values at the specified timestep
    - Red dashed bounding box showing grid extent (minX→maxX, minY→maxY)
    - Metadata text box with grid dimensions, timesteps, units, value range

    This is useful for verifying that preprocessing correctly interpolated
    external data to the simulation grid.

    Args:
        data: JshdData object from load_jshd().
        timestep: Which timestep to plot (default: 0).
        cmap: Matplotlib colormap name.
        title: Custom title. If None, uses source filename and timestep.
        figsize: Figure size in inches.
        show: If True, call plt.show().

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If timestep is out of range.

    Example:
        >>> data = load_jshd(cli, Path("soil_quality.jshd"))
        >>> fig = plot_jshd(data, timestep=0, cmap="YlGn")
        >>> fig.savefig("debug_plot.png", dpi=150, bbox_inches="tight")
    """
    meta = data.metadata

    # Validate timestep
    if timestep < meta.min_timestep or timestep > meta.max_timestep:
        raise ValueError(
            f"timestep {timestep} out of range [{meta.min_timestep}, {meta.max_timestep}]"
        )

    # Get data for this timestep as 2D array
    arr = data.to_array(timestep=timestep)

    # Compute value statistics
    vmin, vmax = float(arr.min()), float(arr.max())
    vmean = float(arr.mean())

    # Create figure with space below for metadata text
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap with extent matching grid bounds
    # Add 0.5 padding so cells are centered on integer indices
    extent: tuple[float, float, float, float] = (
        meta.min_x - 0.5,
        meta.max_x + 0.5,
        meta.min_y - 0.5,
        meta.max_y + 0.5,
    )
    im = ax.imshow(
        arr,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
    )

    # Add colorbar with units label
    cbar_label = f"Value ({meta.units})" if meta.units else "Value"
    fig.colorbar(im, ax=ax, label=cbar_label)

    # Draw bounding box (emphasize grid extent)
    rect = Rectangle(
        (meta.min_x - 0.5, meta.min_y - 0.5),
        meta.width,
        meta.height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

    # Labels
    ax.set_xlabel("X (grid index)")
    ax.set_ylabel("Y (grid index)")

    # Title
    if title is None:
        fname = data.source_path.name if data.source_path else "JSHD"
        title = f"{fname} (timestep {timestep})"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Metadata text - single line below plot
    meta_text = (
        f"Grid: {meta.width}\u00d7{meta.height} | "
        f"X: {meta.min_x}\u2013{meta.max_x} | "
        f"Y: {meta.min_y}\u2013{meta.max_y} | "
        f"Timesteps: {meta.min_timestep}\u2013{meta.max_timestep} ({meta.num_timesteps}) | "
        f"Units: {meta.units or 'none'} | "
        f"Range: {vmin:.4g}\u2013{vmax:.4g} | "
        f"Mean: {vmean:.4g}"
    )

    # Add metadata as figure text below the axes
    fig.text(
        0.5,
        0.02,
        meta_text,
        ha="center",
        va="bottom",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    # Adjust layout to make room for metadata text at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if show:
        plt.show()

    return fig
