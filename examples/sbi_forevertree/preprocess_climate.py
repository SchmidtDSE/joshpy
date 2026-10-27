"""Preprocess the fire model's synthetic climate (climate/*.nc) to .jshd on the ForeverTree grid.

The NetCDF files were generated outside this repo by ``data/generate_fire_climate.py``; each records
its scenario, seed and regeneration command in its attributes. They hold yearly mean temperature
(K) and total precipitation (mm) on a 10 x 8 lat/lon grid over the ForeverTree extent:

- ``historical.nc``: 30 stationary years. The fire model's spin-up resamples them, and its present
  reads the last of them in order.
- ``normal.nc``, ``hot_dry.nc``, ``cool_wet.nc``: 40 years of each future, starting the year after
  the present. ``normal`` continues the historical climate; ``hot_dry`` ramps to about 25% less
  rain and 1.8 K warmer by its last decade, ``cool_wet`` to about 16% more rain and 0.9 K cooler.

Writes ``climate/{scenario}_{variable}.jshd``. Run from the repository root::

    python examples/sbi_forevertree/preprocess_climate.py
"""

from __future__ import annotations

from pathlib import Path

from joshpy.cli import JoshCLI
from joshpy.grid import GridSpec
from joshpy.jar import JarMode

CLIMATE = Path(__file__).resolve().parent / "climate"
SCENARIOS = {"historical": 30, "normal": 40, "hot_dry": 40, "cool_wet": 40}  # years in each file
VARIABLES = {"temperature": "K", "precipitation": "mm"}

# The ForeverTree grid (forevertree_fire.josh): 16 km patches, low = north-west corner.
GRID = GridSpec(
    name="forevertree",
    output_dir=CLIMATE,
    size_m=16000,
    low=(36.73, -119.52),
    high=(35.80, -117.98),
    steps=max(SCENARIOS.values()),  # required; each file declares its own length below
)


def preprocess(cli: JoshCLI) -> list[Path]:
    """Write one .jshd per scenario and variable; returns their paths."""
    written = []
    for scenario, years in SCENARIOS.items():
        for variable, units in VARIABLES.items():
            result = GRID.preprocess_netcdf(
                cli,
                josh_name=f"{scenario}_{variable}",
                data_file=CLIMATE / f"{scenario}.nc",
                variable=variable,
                units=units,
                time_type="count",
                time_start=0,
                time_unit="year",
                time_count=years,
                time_increment=1,
            )
            if not result.success:
                raise RuntimeError(f"Preprocessing {scenario} {variable} failed:\n{result.stderr}")
            written.append(CLIMATE / f"{scenario}_{variable}.jshd")
    return written


if __name__ == "__main__":
    for path in preprocess(JoshCLI(josh_jar=JarMode.DEV)):
        print(path.relative_to(CLIMATE.parent))
