"""CLI entry point for viewing and diffing run configurations.

Usage::

    python -m joshpy.config registry.duckdb --view baseline
    python -m joshpy.config registry.duckdb --view baseline --export-only
    python -m joshpy.config registry.duckdb --diff baseline high_growth
    python -m joshpy.config registry.duckdb --diff baseline high_growth --export-only
    python -m joshpy.config registry.duckdb --diff baseline high_growth --ide cursor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m joshpy.config",
        description="View or diff run configurations from a joshpy registry.",
    )
    parser.add_argument(
        "registry",
        type=Path,
        help="Path to the .duckdb registry file.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--view",
        metavar="RUN",
        help="Print the stored config for a run (label or run_hash).",
    )
    mode.add_argument(
        "--diff",
        nargs=2,
        metavar=("RUN1", "RUN2"),
        help="Diff two run configs (labels or run_hashes).",
    )

    parser.add_argument(
        "--ide",
        default="vscode",
        help="IDE to open diff in (default: vscode). Supported: vscode, cursor.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export config file(s) without opening an IDE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write exported configs (default: system temp).",
    )
    return parser


def main() -> int:
    """Entry point for ``python -m joshpy.config``."""
    parser = _build_parser()
    args = parser.parse_args()

    from joshpy.config import export_pair, open_diff, open_view, view_config
    from joshpy.registry import RunRegistry

    if not args.registry.exists():
        print(f"Error: Registry not found: {args.registry}", file=sys.stderr)
        return 1

    registry = RunRegistry(str(args.registry))
    try:
        if args.view is not None and args.export_only:
            content = view_config(registry, args.view)
            print(content, end="" if content.endswith("\n") else "\n")
        elif args.view is not None:
            path = open_view(
                registry, args.view, args.ide, args.output_dir
            )
            print(f"Exported: {path}")
        elif args.export_only:
            path1, path2 = export_pair(
                registry, args.diff[0], args.diff[1], args.output_dir
            )
            print(path1)
            print(path2)
        else:
            path1, path2 = open_diff(
                registry, args.diff[0], args.diff[1], args.ide, args.output_dir
            )
            print(f"Exported: {path1}")
            print(f"Exported: {path2}")
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        registry.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
