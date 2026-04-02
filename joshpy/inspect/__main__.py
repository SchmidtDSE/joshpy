"""CLI entry point for viewing and diffing run configurations and josh sources.

Usage::

    python -m joshpy.inspect registry.duckdb --view baseline
    python -m joshpy.inspect registry.duckdb --view baseline --export-only
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth --export-only
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth --ide cursor
    python -m joshpy.inspect registry.duckdb --view baseline --type josh
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth --type josh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m joshpy.inspect",
        description="View or diff run configurations and josh sources from a joshpy registry.",
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
        help="Print the stored content for a run (label or run_hash).",
    )
    mode.add_argument(
        "--diff",
        nargs=2,
        metavar=("RUN1", "RUN2"),
        help="Diff two runs (labels or run_hashes).",
    )

    parser.add_argument(
        "--type",
        choices=["config", "josh"],
        default="config",
        help="What to view/diff: config (.jshc) or josh source (.josh). Default: config.",
    )
    parser.add_argument(
        "--ide",
        default="vscode",
        help="IDE to open diff in (default: vscode). Supported: vscode, cursor.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export file(s) without opening an IDE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write exported files (default: system temp).",
    )
    return parser


def main() -> int:
    """Entry point for ``python -m joshpy.inspect``."""
    parser = _build_parser()
    args = parser.parse_args()

    from joshpy.inspect import (
        export_josh_pair,
        export_pair,
        open_diff,
        open_josh_diff,
        open_josh_view,
        open_view,
        view_config,
        view_josh,
    )
    from joshpy.registry import RunRegistry

    if not args.registry.exists():
        print(f"Error: Registry not found: {args.registry}", file=sys.stderr)
        return 1

    registry = RunRegistry(str(args.registry))
    try:
        if args.type == "josh":
            if args.view is not None and args.export_only:
                content = view_josh(registry, args.view)
                print(content, end="" if content.endswith("\n") else "\n")
            elif args.view is not None:
                path = open_josh_view(
                    registry, args.view, args.ide, args.output_dir
                )
                print(f"Exported: {path}")
            elif args.export_only:
                path1, path2 = export_josh_pair(
                    registry, args.diff[0], args.diff[1], args.output_dir
                )
                print(path1)
                print(path2)
            else:
                path1, path2 = open_josh_diff(
                    registry, args.diff[0], args.diff[1], args.ide, args.output_dir
                )
                print(f"Exported: {path1}")
                print(f"Exported: {path2}")
        else:
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
