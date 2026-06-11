"""CLI entry point for Josh JAR management.

Usage::

    python -m joshpy.jar [--force] [--jar-dir DIR]

Mirrors the ``python -m joshpy.{inspect,debug,jfr}`` module-CLI convention so
downstream projects don't need a wrapper script for JAR downloads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from joshpy.jar import download_jars


def main(argv: list[str] | None = None) -> int:
    """Download or refresh the Josh JAR files (prod + dev)."""
    parser = argparse.ArgumentParser(
        prog="python -m joshpy.jar",
        description="Download or refresh Josh JAR files (prod + dev).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if up to date."
    )
    parser.add_argument(
        "--jar-dir",
        type=Path,
        default=None,
        help="Target directory (default: joshpy-managed jar/ dir).",
    )
    args = parser.parse_args(argv)

    results = download_jars(jar_dir=args.jar_dir, force=args.force)

    failed = False
    for mode, result in results.items():
        if not result.success:
            failed = True
            print(
                f"  {mode.value}: FAILED ({result.error or 'unknown error'})",
                file=sys.stderr,
            )
        elif result.was_updated:
            old = result.old_version or "none"
            new = result.new_version or "unknown"
            print(f"  {mode.value}: updated ({old} -> {new})")
        else:
            print(f"  {mode.value}: up to date ({result.new_version or 'unknown'})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
