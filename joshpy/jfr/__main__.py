"""CLI entry point for JFR resource diagnostics.

Usage::

    python -m joshpy.jfr recording.jfr
    python -m joshpy.jfr recording.jfr --detailed
    python -m joshpy.jfr recording.jfr --no-color
    python -m joshpy.jfr recording.jfr --java /usr/lib/jvm/java-17/bin/java
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from joshpy.jfr import build_resource_profile

# JFR events to extract for resource diagnostics.
_EVENTS = ",".join([
    "jdk.CPULoad",
    "jdk.GCPhasePause",
    "jdk.GCCPUTime",
    "jdk.GCHeapSummary",
    "jdk.ResidentSetSize",
    "jdk.PhysicalMemory",
    "jdk.FileRead",
    "jdk.FileWrite",
    "jdk.JavaMonitorWait",
    "jdk.JavaMonitorEnter",
])


def _resolve_jfr_bin(java_path: str | None) -> str:
    """Resolve the jfr binary path.

    If *java_path* points into a JDK ``bin/`` directory, use the sibling
    ``jfr`` binary.  Otherwise fall back to ``jfr`` on PATH.
    """
    if java_path:
        java = Path(java_path)
        if java.parent.name == "bin":
            candidate = java.parent / "jfr"
            if candidate.exists():
                return str(candidate)
    # Fall back to PATH
    found = shutil.which("jfr")
    if found:
        return found
    # Last resort: check java on PATH and derive
    java_on_path = shutil.which("java")
    if java_on_path:
        java_bin = Path(java_on_path).resolve()
        if java_bin.parent.name == "bin":
            candidate = java_bin.parent / "jfr"
            if candidate.exists():
                return str(candidate)
    return "jfr"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m joshpy.jfr",
        description="Diagnose JFR recordings with user-friendly resource analysis.",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to .jfr recording file.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show advanced output with per-event breakdowns.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output (currently unused, reserved).",
    )
    parser.add_argument(
        "--java",
        default=None,
        help="Path to java binary (used to resolve jfr sibling).",
    )
    return parser


def main() -> int:
    """Entry point for ``python -m joshpy.jfr``."""
    parser = _build_parser()
    args = parser.parse_args()

    jfr_path: Path = args.file
    if not jfr_path.exists():
        print(f"File not found: {jfr_path}", file=sys.stderr)
        return 1

    jfr_bin = _resolve_jfr_bin(args.java)

    # Run jfr print for event data
    try:
        print_proc = subprocess.run(
            [jfr_bin, "print", "--events", _EVENTS, str(jfr_path.resolve())],
            capture_output=True,
            text=True,
        )
        if print_proc.returncode != 0:
            print(f"jfr print failed: {print_proc.stderr}", file=sys.stderr)
            return 1
    except FileNotFoundError:
        print(
            f"Could not find jfr binary at '{jfr_bin}'. "
            "Ensure a JDK is installed and jfr is on PATH, "
            "or use --java to specify the java binary path.",
            file=sys.stderr,
        )
        return 1

    # Run jfr summary for recording metadata
    try:
        summary_proc = subprocess.run(
            [jfr_bin, "summary", str(jfr_path.resolve())],
            capture_output=True,
            text=True,
        )
        if summary_proc.returncode != 0:
            print(f"jfr summary failed: {summary_proc.stderr}", file=sys.stderr)
            return 1
    except FileNotFoundError:
        print(f"Could not find jfr binary at '{jfr_bin}'.", file=sys.stderr)
        return 1

    profile = build_resource_profile(print_proc.stdout, summary_proc.stdout)
    print(profile.summary(detailed=args.detailed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
