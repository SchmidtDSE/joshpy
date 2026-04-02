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

import itertools
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


# Josh identifiers cannot contain underscores or start with special characters,
# so we use a fixed CamelCase name for the temporary preprocess script.
_PREPROCESS_SIM_NAME = "Preprocess"


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
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def file_mappings(self) -> dict[str, Path]:
        """Build a file_mappings dict with absolute paths.

        For files with ``template_path``, placeholders are resolved using
        each variant axis's default value.

        Returns:
            Dict mapping josh external names to absolute file Paths.
        """
        defaults = {k: v["default"] for k, v in self.variants.items()}
        result: dict[str, Path] = {}
        for josh_name, info in self.files.items():
            if "template_path" in info:
                resolved = info["template_path"].format(**defaults)
                result[josh_name] = (self.output_dir / resolved).resolve()
            else:
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

    def file_mappings_for(self, **variant_values: str) -> dict[str, Path]:
        """Resolve file mappings with specific variant values.

        Static ``path`` files pass through unchanged. ``template_path`` files
        are resolved using the provided values (falling back to defaults for
        unspecified axes).

        Args:
            **variant_values: Axis name → value overrides
                (e.g., ``scenario="ssp370"``).

        Returns:
            Dict mapping josh external names to absolute file Paths.

        Raises:
            ValueError: If an axis name or value is invalid.
        """
        for axis, value in variant_values.items():
            if axis not in self.variants:
                raise ValueError(
                    f"Unknown variant axis '{axis}'. "
                    f"Available: {list(self.variants.keys())}"
                )
            allowed = self.variants[axis]["values"]
            if value not in allowed:
                raise ValueError(
                    f"Invalid value '{value}' for axis '{axis}'. "
                    f"Allowed: {allowed}"
                )

        merged = {k: v["default"] for k, v in self.variants.items()}
        merged.update(variant_values)

        result: dict[str, Path] = {}
        for josh_name, info in self.files.items():
            if "template_path" in info:
                resolved = info["template_path"].format(**merged)
                result[josh_name] = (self.output_dir / resolved).resolve()
            else:
                result[josh_name] = (self.output_dir / info["path"]).resolve()
        return result

    def variant_sweep(
        self,
        axis: str | None = None,
        *,
        axes: list[str] | None = None,
        values: list[str] | None = None,
    ) -> CompoundSweepParameter:
        """Generate a CompoundSweepParameter from variant axes.

        Finds all ``template_path`` files referencing the given axis/axes,
        builds one ``FileSweepParameter`` per file, and wraps them in a
        ``CompoundSweepParameter`` so all files switch together.

        Args:
            axis: Single axis name (common case).
            axes: List of axis names for multi-axis cross-product.
            values: Subset of values to sweep (single-axis only).

        Returns:
            A ``CompoundSweepParameter`` ready for
            ``SweepConfig.compound_parameters``.

        Raises:
            ValueError: If axis/axes are invalid, both provided, or values
                used with multi-axis.
        """
        from joshpy.jobs import CompoundSweepParameter, FileSweepParameter

        if axis is not None and axes is not None:
            raise ValueError(
                "Cannot specify both 'axis' and 'axes'. "
                "Use 'axis' for single-axis or 'axes' for multi-axis."
            )
        if axis is None and axes is None:
            raise ValueError("Must specify either 'axis' or 'axes'.")

        if axis is not None:
            # Single-axis mode
            if axis not in self.variants:
                raise ValueError(
                    f"Unknown variant axis '{axis}'. "
                    f"Available: {list(self.variants.keys())}"
                )
            sweep_values = values if values is not None else self.variants[axis]["values"]
            if values is not None:
                allowed = self.variants[axis]["values"]
                for v in values:
                    if v not in allowed:
                        raise ValueError(
                            f"Invalid value '{v}' for axis '{axis}'. "
                            f"Allowed: {allowed}"
                        )

            # Find template_path files referencing this axis
            placeholder = "{" + axis + "}"
            defaults = {k: v["default"] for k, v in self.variants.items()}

            parameters: list[FileSweepParameter] = []
            for josh_name, info in sorted(self.files.items()):
                if "template_path" not in info:
                    continue
                if placeholder not in info["template_path"]:
                    continue
                paths = []
                for val in sweep_values:
                    resolved_vars = {**defaults, axis: val}
                    resolved = info["template_path"].format(**resolved_vars)
                    paths.append((self.output_dir / resolved).resolve())
                parameters.append(
                    FileSweepParameter(name=josh_name, paths=paths)
                )

            return CompoundSweepParameter(
                name=axis,
                parameters=parameters,
                labels=list(sweep_values),
            )
        else:
            # Multi-axis mode
            assert axes is not None
            if values is not None:
                raise ValueError(
                    "'values' parameter is only supported for single-axis sweeps."
                )
            for ax in axes:
                if ax not in self.variants:
                    raise ValueError(
                        f"Unknown variant axis '{ax}'. "
                        f"Available: {list(self.variants.keys())}"
                    )

            # Build cross-product of axis values
            axis_values = [self.variants[ax]["values"] for ax in axes]
            combos = list(itertools.product(*axis_values))

            defaults = {k: v["default"] for k, v in self.variants.items()}

            # Find template_path files referencing ANY of the axes
            placeholders = ["{" + ax + "}" for ax in axes]
            parameters = []
            for josh_name, info in sorted(self.files.items()):
                if "template_path" not in info:
                    continue
                if not any(ph in info["template_path"] for ph in placeholders):
                    continue
                paths = []
                for combo in combos:
                    resolved_vars = {**defaults}
                    for ax, val in zip(axes, combo):
                        resolved_vars[ax] = val
                    resolved = info["template_path"].format(**resolved_vars)
                    paths.append((self.output_dir / resolved).resolve())
                parameters.append(
                    FileSweepParameter(name=josh_name, paths=paths)
                )

            labels = ["_".join(combo) for combo in combos]
            return CompoundSweepParameter(
                name="_".join(axes),
                parameters=parameters,
                labels=labels,
            )

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
                entry: dict[str, str] = {"units": info.get("units", "")}
                if "template_path" in info:
                    entry["template_path"] = info["template_path"]
                else:
                    entry["path"] = info.get("path", "")
                files[josh_name] = entry
            else:
                # Simple form: just a path string
                files[josh_name] = {"path": str(info), "units": ""}

        variants = {}
        for axis_name, axis_info in data.get("variants", {}).items():
            variants[axis_name] = {
                "values": axis_info.get("values", []),
                "default": axis_info.get("default", ""),
            }

        return cls(
            name=data.get("name", path.stem),
            output_dir=path.parent.resolve(),
            size_m=grid.get("size_m", 0),
            low=low,
            high=high,
            steps=grid.get("steps", 0),
            files=files,
            variants=variants,
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

        if self.variants:
            data["variants"] = {}
            for axis_name, axis_info in sorted(self.variants.items()):
                data["variants"][axis_name] = {
                    "values": axis_info["values"],
                    "default": axis_info["default"],
                }

        if self.files:
            data["files"] = {}
            for josh_name, info in sorted(self.files.items()):
                entry: dict[str, str] = {}
                if "template_path" in info:
                    entry["template_path"] = info["template_path"]
                else:
                    entry["path"] = info["path"]
                entry["units"] = info["units"]
                data["files"][josh_name] = entry

        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        return path

    def _render_preprocess_script(self) -> Path:
        """Render a minimal .josh file for preprocessing.

        Returns:
            Path to a temporary .josh file. Caller must delete after use.
        """
        content = (
            f"start simulation {_PREPROCESS_SIM_NAME}\n"
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
        self,
        josh_name: str,
        subdirectory: str | None = None,
        variant: dict[str, str] | None = None,
    ) -> Path:
        """Compute the output .jshd path for a preprocessed file.

        When *variant* is provided and the file has a ``template_path``,
        the template is resolved with the variant values (plus defaults
        for unspecified axes) and used as the output path.
        """
        if variant is not None and josh_name in self.files:
            info = self.files[josh_name]
            if "template_path" in info:
                merged = {k: v["default"] for k, v in self.variants.items()}
                merged.update(variant)
                resolved = info["template_path"].format(**merged)
                out_path = self.output_dir / resolved
                out_path.parent.mkdir(parents=True, exist_ok=True)
                return out_path

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
        variant: dict[str, str] | None = None,
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
            variant: Variant values to resolve ``template_path``
                (e.g., ``{"scenario": "ssp370"}``). When provided, the
                output path is resolved from the file's template_path
                and ``_register_file()`` is skipped.

        Returns:
            CLIResult from the preprocessing command.
        """
        from joshpy.cli import GeotiffPreprocessConfig

        output_path = self._compute_output_path(josh_name, subdirectory, variant)
        script_path = self._render_preprocess_script()
        try:
            config = GeotiffPreprocessConfig(
                script=script_path,
                simulation=_PREPROCESS_SIM_NAME,
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

        if result.success and variant is None:
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
        variant: dict[str, str] | None = None,
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
            variant: Variant values to resolve ``template_path``
                (e.g., ``{"scenario": "ssp370"}``). When provided, the
                output path is resolved from the file's template_path
                and ``_register_file()`` is skipped.

        Returns:
            CLIResult from the preprocessing command.
        """
        from joshpy.cli import NetcdfPreprocessConfig

        output_path = self._compute_output_path(josh_name, subdirectory, variant)
        script_path = self._render_preprocess_script()
        try:
            config = NetcdfPreprocessConfig(
                script=script_path,
                simulation=_PREPROCESS_SIM_NAME,
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

        if result.success and variant is None:
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
        variant: dict[str, str] | None = None,
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
            variant: Variant values to resolve ``template_path``
                (e.g., ``{"scenario": "ssp370"}``). When provided, the
                output path is resolved from the file's template_path
                and ``_register_file()`` is skipped.

        Returns:
            CLIResult from the preprocessing command.
        """
        from joshpy.cli import CsvPreprocessConfig

        output_path = self._compute_output_path(josh_name, subdirectory, variant)
        script_path = self._render_preprocess_script()
        try:
            config = CsvPreprocessConfig(
                script=script_path,
                simulation=_PREPROCESS_SIM_NAME,
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

        if result.success and variant is None:
            self._register_file(josh_name, output_path, units)

        return result
