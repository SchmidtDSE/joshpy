"""CLI entry point for inspecting a grid manifest.

Usage::

    python -m joshpy.grid data/grids/dev_fine/grid.yaml
    python -m joshpy.grid data/grids/dev_fine/grid.yaml --json

Summarizes a `grid.yaml`: geometry, variant axes, and the external-data
inventory — including whether each preprocessed `.jshd` file exists on disk.
Mirrors the ``python -m joshpy.{inspect,debug,jfr,jar,bottle}`` module-CLI
convention. Read-only: it never preprocesses or modifies anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from joshpy.grid import GridSpec


def main(argv: list[str] | None = None) -> int:
    """Summarize a grid.yaml manifest."""
    parser = argparse.ArgumentParser(
        prog="python -m joshpy.grid",
        description="Summarize a grid.yaml: geometry, variant axes, and the "
        "external-data inventory (with on-disk existence).",
    )
    parser.add_argument("grid_yaml", type=Path, help="Path to a grid.yaml manifest.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured summary as JSON instead of the text view.",
    )
    args = parser.parse_args(argv)

    if not args.grid_yaml.exists():
        print(f"Grid manifest not found: {args.grid_yaml}", file=sys.stderr)
        return 1

    grid = GridSpec.from_yaml(args.grid_yaml)

    if args.json:
        import json

        print(json.dumps(grid.to_summary_dict(), indent=2))
    else:
        print(grid.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
