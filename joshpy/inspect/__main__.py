"""CLI entry point for querying, viewing, and diffing run configurations and josh sources.

Usage::

    python -m joshpy.inspect registry.duckdb --labels
    python -m joshpy.inspect registry.duckdb --sessions
    python -m joshpy.inspect registry.duckdb --info baseline
    python -m joshpy.inspect registry.duckdb --summary
    python -m joshpy.inspect registry.duckdb --view baseline
    python -m joshpy.inspect registry.duckdb --view baseline --export-only
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth
    python -m joshpy.inspect registry.duckdb --diff baseline high_growth --print
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
        "--labels",
        action="store_true",
        help="List all labeled runs with their run_hash and creation time.",
    )
    mode.add_argument(
        "--sessions",
        action="store_true",
        help="List all sessions with experiment name, status, and run counts.",
    )
    mode.add_argument(
        "--info",
        metavar="RUN",
        help="Show detailed info for a run (label or run_hash).",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="Print a data summary for the entire registry.",
    )
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
        "--print",
        dest="print_diff",
        action="store_true",
        help="Print a unified text diff to stdout (works headless; no IDE). "
        "Applies to --diff only.",
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
        format_labels,
        format_run_info,
        format_sessions,
        format_summary,
        open_diff,
        open_josh_diff,
        open_josh_view,
        open_view,
        text_diff,
        text_josh_diff,
        view_config,
        view_josh,
    )
    from joshpy.registry import RunRegistry

    if not args.registry.exists():
        print(f"Error: Registry not found: {args.registry}", file=sys.stderr)
        return 1

    registry = RunRegistry(str(args.registry))
    try:
        if args.labels:
            print(format_labels(registry))
        elif args.sessions:
            print(format_sessions(registry))
        elif args.info is not None:
            print(format_run_info(registry, args.info))
        elif args.summary:
            print(format_summary(registry))
        elif args.type == "josh":
            if args.view is not None and args.export_only:
                content = view_josh(registry, args.view)
                print(content, end="" if content.endswith("\n") else "\n")
            elif args.view is not None:
                path = open_josh_view(
                    registry, args.view, args.ide, args.output_dir
                )
                print(f"Exported: {path}")
            elif args.print_diff:
                diff = text_josh_diff(registry, args.diff[0], args.diff[1])
                if diff:
                    print(diff, end="" if diff.endswith("\n") else "\n")
                else:
                    print("(no differences)", file=sys.stderr)
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
            elif args.print_diff:
                diff = text_diff(registry, args.diff[0], args.diff[1])
                if diff:
                    print(diff, end="" if diff.endswith("\n") else "\n")
                else:
                    print("(no differences)", file=sys.stderr)
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
