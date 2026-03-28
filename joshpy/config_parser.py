"""Parser for Josh configuration files (.jshc format).

Extracts parameter names and numeric values from .jshc files, which use
the format::

    # Comment
    paramName = value unit

This enables auto-populating the ``config_parameters`` table for ad-hoc
runs that use ``config_path`` (raw .jshc, no sweep templating).

Example::

    from joshpy.config_parser import parse_jshc, parse_jshc_content

    params = parse_jshc(Path("configs/baseline.jshc"))
    # {"coverHighThreshold": 2.0, "treesHighCoverPerHa": 55, ...}

    params = parse_jshc_content("maxGrowth = 50 meters")
    # {"maxGrowth": 50}
"""

from __future__ import annotations

from pathlib import Path


def parse_jshc(path: str | Path) -> dict[str, int | float]:
    """Parse a .jshc configuration file into a parameters dict.

    Args:
        path: Path to the .jshc file.

    Returns:
        Dict mapping parameter names to numeric values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a non-comment, non-blank line cannot be parsed.
    """
    return parse_jshc_content(Path(path).read_text())


def parse_jshc_content(content: str) -> dict[str, int | float]:
    """Parse .jshc configuration content into a parameters dict.

    Args:
        content: Raw text content of a .jshc file.

    Returns:
        Dict mapping parameter names to numeric values.
        Returns int if the literal has no decimal point, float otherwise.
        Units are discarded (they are a Josh runtime concern).

    Raises:
        ValueError: If a non-comment, non-blank line cannot be parsed.
    """
    result: dict[str, int | float] = {}

    for line_num, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()

        # Skip comments and blank lines
        if not stripped or stripped.startswith("#"):
            continue

        # Split on '='
        if "=" not in stripped:
            raise ValueError(
                f"Line {line_num}: expected 'name = value unit', got: {stripped!r}"
            )

        name_part, value_part = stripped.split("=", 1)
        name = name_part.strip()
        value_part = value_part.strip()

        if not name:
            raise ValueError(f"Line {line_num}: empty parameter name")
        if not value_part:
            raise ValueError(f"Line {line_num}: empty value for parameter '{name}'")

        # First token is the numeric value, rest is the unit (discarded)
        tokens = value_part.split()
        value_str = tokens[0]

        try:
            if "." in value_str:
                result[name] = float(value_str)
            else:
                result[name] = int(value_str)
        except ValueError:
            raise ValueError(
                f"Line {line_num}: cannot parse numeric value from {value_str!r} "
                f"for parameter '{name}'"
            ) from None

    return result
