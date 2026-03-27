"""Grid specification for Josh simulations.

A GridSpec defines the spatial geometry of a simulation grid and maintains an
inventory of data files (.jshd) preprocessed for that grid. It bridges
preprocessing and model execution:

1. Create a GridSpec with grid geometry
2. Preprocess rasters through it (accumulates file entries)
3. Save to YAML
4. Load later to feed ``JobConfig.file_mappings`` and ``template_vars``

A grid is NOT a model. It has no simulation name, no export paths, no debug
flags. Multiple models (simulations) share the same grid.

Requires: pyyaml (part of the ``[jobs]`` extra)

Example::

    from joshpy.grid import GridSpec

    # Build during preprocessing
    grid = GridSpec(
        name="dev_fine",
        output_dir=Path("data/grids/dev_fine"),
        size_m=30,
        low=(33.902, -116.0465),
        high=(33.908, -116.0395),
        steps=86,
    )
    grid.preprocess_geotiff(cli, josh_name="cover", data_file=..., ...)
    grid.save()

    # Load for model runs
    grid = GridSpec.from_yaml("data/grids/dev_fine/grid.yaml")
    config = JobConfig(
        template_vars={**grid.template_vars, "simulation_name": "MyModel"},
        file_mappings=grid.file_mappings,
        ...
    )
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from joshpy.cli import CLIResult, JoshCLI

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _check_yaml() -> None:
    if not HAS_YAML:
        raise ImportError(
            "pyyaml is required for GridSpec. Install with: pip install joshpy[jobs]"
        )


@dataclass
class GridSpec:
    """Spatial grid definition plus data file inventory.

    Attributes:
        name: Human-readable grid name (e.g., "dev_fine").
        output_dir: Directory where .jshd files live (and grid.yaml is saved).
        size_m: Grid cell size in meters.
        low: (latitude, longitude) of the grid's low corner.
        high: (latitude, longitude) of the grid's high corner.
        steps: Number of simulation timesteps.
        files: Inventory of preprocessed files.
            Keys are josh external names, values are dicts with
            ``"path"`` (relative to output_dir) and ``"units"``.
    """

    name: str
    output_dir: Path
    size_m: int | float
    low: tuple[float, float]  # (lat, lon)
    high: tuple[float, float]  # (lat, lon)
    steps: int
    files: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def file_mappings(self) -> dict[str, Path]:
        """Build a file_mappings dict with absolute paths.

        Returns:
            Dict mapping josh external names to absolute file Paths.
        """
        result: dict[str, Path] = {}
        for josh_name, info in self.files.items():
            result[josh_name] = (self.output_dir / info["path"]).resolve()
        return result

    @property
    def template_vars(self) -> dict[str, Any]:
        """Build a template_vars dict for .josh.j2 rendering.

        Returns:
            Dict with grid geometry values suitable for Jinja2 templates.
        """
        return {
            "size_m": self.size_m,
            "low_lat": self.low[0],
            "low_lon": self.low[1],
            "high_lat": self.high[0],
            "high_lon": self.high[1],
            "steps": self.steps,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> GridSpec:
        """Load a GridSpec from a YAML file.

        Args:
            path: Path to a grid.yaml file.

        Returns:
            GridSpec with paths resolved relative to the YAML file's directory.
        """
        _check_yaml()
        path = Path(path)
        data = yaml.safe_load(path.read_text())

        grid = data.get("grid", {})
        low = tuple(grid.get("low", [0, 0]))
        high = tuple(grid.get("high", [0, 0]))

        files = {}
        for josh_name, info in data.get("files", {}).items():
            if isinstance(info, dict):
                files[josh_name] = {
                    "path": info.get("path", ""),
                    "units": info.get("units", ""),
                }
            else:
                # Simple form: just a path string
                files[josh_name] = {"path": str(info), "units": ""}

        return cls(
            name=data.get("name", path.stem),
            output_dir=path.parent.resolve(),
            size_m=grid.get("size_m", 0),
            low=low,
            high=high,
            steps=grid.get("steps", 0),
            files=files,
        )

    def save(self, path: str | Path | None = None) -> Path:
        """Save the GridSpec to a YAML file.

        Args:
            path: Output path. Defaults to ``output_dir / "grid.yaml"``.

        Returns:
            Path to the written file.
        """
        _check_yaml()

        if path is None:
            path = self.output_dir / "grid.yaml"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "name": self.name,
            "grid": {
                "size_m": self.size_m,
                "low": list(self.low),
                "high": list(self.high),
                "steps": self.steps,
            },
        }

        if self.files:
            data["files"] = {}
            for josh_name, info in sorted(self.files.items()):
                data["files"][josh_name] = {
                    "path": info["path"],
                    "units": info["units"],
                }

        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        return path

    def _render_preprocess_script(self) -> Path:
        """Render a minimal .josh file for preprocessing.

        Returns:
            Path to a temporary .josh file. Caller must delete after use.
        """
        content = (
            f"start simulation _preprocess_{self.name}\n"
            f"  grid.size = {self.size_m} m\n"
            f"  grid.low = {self.low[0]} degrees latitude, "
            f"{self.low[1]} degrees longitude\n"
            f"  grid.high = {self.high[0]} degrees latitude, "
            f"{self.high[1]} degrees longitude\n"
            f"  steps.low = 0 count\n"
            f"  steps.high = {self.steps} count\n"
            f"end simulation\n"
            f"\n"
            f"start patch Default\n"
            f"end patch\n"
        )
        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".josh", prefix="preprocess_", delete=False
        )
        fd.write(content)
        fd.close()
        return Path(fd.name)

    def _compute_output_path(
        self, josh_name: str, subdirectory: str | None = None
    ) -> Path:
        """Compute the output .jshd path for a preprocessed file."""
        if subdirectory:
            out_dir = self.output_dir / subdirectory
        else:
            out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{josh_name}.jshd"

    def _relative_path(self, absolute_path: Path) -> str:
        """Compute a path relative to output_dir."""
        try:
            return str(absolute_path.relative_to(self.output_dir))
        except ValueError:
            return str(absolute_path)

    def _register_file(
        self, josh_name: str, output_path: Path, units: str
    ) -> None:
        """Register a successfully preprocessed file in the inventory."""
        self.files[josh_name] = {
            "path": self._relative_path(output_path),
            "units": units,
        }

    def preprocess_geotiff(
        self,
        cli: JoshCLI,
        *,
        josh_name: str,
        data_file: str | Path,
        band: int,
        units: str,
        timestep: int,
        crs: str | None = None,
        parallel: bool = False,
        amend: bool = False,
        subdirectory: str | None = None,
    ) -> CLIResult:
        """Preprocess a GeoTIFF file using this grid's geometry.

        Args:
            cli: JoshCLI instance.
            josh_name: Name the josh model uses for this external data.
            data_file: Path to the input GeoTIFF.
            band: Band index (0-based).
            units: Data units.
            timestep: Simulation timestep this data maps to.
            crs: Coordinate reference system (if not embedded in file).
            parallel: Enable parallel processing.
            amend: Append to existing .jshd file.
            subdirectory: Optional subdirectory within output_dir.

        Returns:
            CLIResult from the preprocessing command.
        """
        from joshpy.cli import GeotiffPreprocessConfig

        output_path = self._compute_output_path(josh_name, subdirectory)
        script_path = self._render_preprocess_script()
        try:
            config = GeotiffPreprocessConfig(
                script=script_path,
                simulation=f"_preprocess_{self.name}",
                data_file=Path(data_file),
                band=band,
                units=units,
                output=output_path,
                timestep=timestep,
                crs=crs,
                parallel=parallel,
                amend=amend,
            )
            result = cli.preprocess(config)
        finally:
            script_path.unlink(missing_ok=True)

        if result.success:
            self._register_file(josh_name, output_path, units)

        return result

    def preprocess_netcdf(
        self,
        cli: JoshCLI,
        *,
        josh_name: str,
        data_file: str | Path,
        variable: str,
        units: str,
        x_coord: str = "lon",
        y_coord: str = "lat",
        time_coord: str = "time",
        timestep: int | None = None,
        crs: str | None = None,
        parallel: bool = False,
        amend: bool = False,
        subdirectory: str | None = None,
    ) -> CLIResult:
        """Preprocess a NetCDF file using this grid's geometry.

        Args:
            cli: JoshCLI instance.
            josh_name: Name the josh model uses for this external data.
            data_file: Path to the input NetCDF file.
            variable: NetCDF variable name to extract.
            units: Data units.
            x_coord: Name of the X/longitude dimension.
            y_coord: Name of the Y/latitude dimension.
            time_coord: Name of the time dimension.
            timestep: Optional specific time slice to extract.
            crs: Coordinate reference system.
            parallel: Enable parallel processing.
            amend: Append to existing .jshd file.
            subdirectory: Optional subdirectory within output_dir.

        Returns:
            CLIResult from the preprocessing command.
        """
        from joshpy.cli import NetcdfPreprocessConfig

        output_path = self._compute_output_path(josh_name, subdirectory)
        script_path = self._render_preprocess_script()
        try:
            config = NetcdfPreprocessConfig(
                script=script_path,
                simulation=f"_preprocess_{self.name}",
                data_file=Path(data_file),
                variable=variable,
                units=units,
                output=output_path,
                x_coord=x_coord,
                y_coord=y_coord,
                time_coord=time_coord,
                timestep=timestep,
                crs=crs,
                parallel=parallel,
                amend=amend,
            )
            result = cli.preprocess(config)
        finally:
            script_path.unlink(missing_ok=True)

        if result.success:
            self._register_file(josh_name, output_path, units)

        return result

    def preprocess_csv(
        self,
        cli: JoshCLI,
        *,
        josh_name: str,
        data_file: str | Path,
        variable: str,
        units: str,
        timestep: int,
        crs: str | None = None,
        parallel: bool = False,
        amend: bool = False,
        subdirectory: str | None = None,
    ) -> CLIResult:
        """Preprocess a CSV point data file using this grid's geometry.

        Args:
            cli: JoshCLI instance.
            josh_name: Name the josh model uses for this external data.
            data_file: Path to the input CSV file.
            variable: Column name to extract.
            units: Data units.
            timestep: Simulation timestep this data maps to.
            crs: Coordinate reference system.
            parallel: Enable parallel processing.
            amend: Append to existing .jshd file.
            subdirectory: Optional subdirectory within output_dir.

        Returns:
            CLIResult from the preprocessing command.
        """
        from joshpy.cli import CsvPreprocessConfig

        output_path = self._compute_output_path(josh_name, subdirectory)
        script_path = self._render_preprocess_script()
        try:
            config = CsvPreprocessConfig(
                script=script_path,
                simulation=f"_preprocess_{self.name}",
                data_file=Path(data_file),
                variable=variable,
                units=units,
                output=output_path,
                timestep=timestep,
                crs=crs,
                parallel=parallel,
                amend=amend,
            )
            result = cli.preprocess(config)
        finally:
            script_path.unlink(missing_ok=True)

        if result.success:
            self._register_file(josh_name, output_path, units)

        return result
