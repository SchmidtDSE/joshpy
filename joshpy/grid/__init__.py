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

    grid = GridSpec.from_yaml("data/grids/dev_fine/grid.yaml")
    print(grid.describe())          # human-readable inventory
    summary = grid.to_summary_dict()  # structured form

CLI usage::

    python -m joshpy.grid data/grids/dev_fine/grid.yaml
    python -m joshpy.grid data/grids/dev_fine/grid.yaml --json
"""

from joshpy.grid._core import GridSpec  # noqa: F401
