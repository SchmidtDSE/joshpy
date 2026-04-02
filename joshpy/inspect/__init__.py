"""Viewing and diffing utilities for run configurations and josh sources.

Usage from Python::

    registry.compare_configs("baseline", "high_growth")
    registry.compare_josh("baseline", "high_growth")

Usage from the command line::

    python -m joshpy.inspect registry.duckdb --view baseline
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth
    python -m joshpy.inspect registry.duckdb --view baseline --type josh
"""

from joshpy.inspect._core import (  # noqa: F401
    export_josh_pair,
    export_pair,
    open_diff,
    open_josh_diff,
    open_josh_view,
    open_view,
    view_config,
    view_josh,
)
