"""CLI entry point for Josh debug log analysis.

Usage::

    python -m joshpy.debug debug.txt --summary
    python -m joshpy.debug debug.txt --step 5 --entity-type organism
    python -m joshpy.debug debug.txt --trace a1b2c3d4
    python -m joshpy.debug debug.txt --find "resprout" --before 3 --after 3
    python -m joshpy.debug patch.txt organism.txt --find "burned" --count
    python -m joshpy.debug simulation.josh --summary
    python -m joshpy.debug simulation.josh --simulation Main --trace a1b2
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from joshpy.debug import (
    DebugMessageStore,
    EventMatch,
    _Ansi,
    _supports_color,
    format_message,
    format_trace,
    load_debug_file,
    load_debug_from_script,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m joshpy.debug",
        description="Inspect Josh simulation debug output.",
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        metavar="FILE",
        help=(
            "Debug output file(s) (.txt) or a single Josh script (.josh). "
            "Multiple .txt files are merged into one store."
        ),
    )
    parser.add_argument(
        "--simulation",
        default="Main",
        help="Simulation name (only used with .josh files, default: Main).",
    )
    parser.add_argument(
        "--run-hash",
        default=None,
        help="Run hash to resolve {run_hash} in debug file paths (.josh only).",
    )

    # -- Filtering -----------------------------------------------------------
    filt = parser.add_argument_group("filtering")
    filt.add_argument(
        "--step",
        type=int,
        default=None,
        help="Filter to a specific step number.",
    )
    filt.add_argument(
        "--step-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Filter to step range (inclusive).",
    )
    filt.add_argument(
        "--entity-type",
        default=None,
        help="Filter by entity type (e.g., 'organism').",
    )
    filt.add_argument(
        "--entity-id",
        default=None,
        help="Filter by entity hex ID (prefix match supported).",
    )
    filt.add_argument(
        "--location",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Filter by (x, y) coordinates.",
    )
    filt.add_argument(
        "--content",
        default=None,
        help="Filter messages whose content contains PATTERN.",
    )

    # -- Modes ---------------------------------------------------------------
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics.",
    )
    modes.add_argument(
        "--trace",
        metavar="ID",
        default=None,
        help="Trace a specific entity across all timesteps.",
    )
    modes.add_argument(
        "--find",
        metavar="KEYWORD",
        default=None,
        help=(
            "Find entities whose traces contain KEYWORD. "
            "Use --before/--after for context window."
        ),
    )

    # --count can combine with --find or stand alone
    parser.add_argument(
        "--count",
        action="store_true",
        help=(
            "Show counts. With --find: summary of matched entities. "
            "Without: message count per step."
        ),
    )

    # -- Find context --------------------------------------------------------
    find_ctx = parser.add_argument_group("find context")
    find_ctx.add_argument(
        "--before",
        type=int,
        default=None,
        metavar="N",
        help="With --find: include N trace messages before each event.",
    )
    find_ctx.add_argument(
        "--after",
        type=int,
        default=None,
        metavar="N",
        help="With --find: include N trace messages after each event.",
    )

    # -- Display -------------------------------------------------------------
    disp = parser.add_argument_group("display")
    disp.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )
    disp.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of messages to display.",
    )

    return parser


def _resolve_entity_id(
    store: DebugMessageStore, prefix: str
) -> str | None:
    """Resolve prefix to full ID, printing guidance on failure."""
    resolved = store.resolve_entity_id(prefix)
    if resolved is not None:
        return resolved

    # Check for ambiguous matches
    matches = [eid for eid in store.entity_ids() if eid.startswith(prefix)]
    if len(matches) > 1:
        print(f"Ambiguous entity ID prefix '{prefix}'. Matches:")
        for eid in matches[:15]:
            print(f"  {eid}")
        if len(matches) > 15:
            print(f"  ... and {len(matches) - 15} more")
        return None

    # No matches at all
    print(f"Entity ID '{prefix}' not found.")
    sample = store.entity_ids()[:10]
    if sample:
        print("Example entity IDs:")
        for eid in sample:
            print(f"  {eid}")
    return None


def _print_summary(store: DebugMessageStore) -> None:
    """Print summary statistics."""
    a = _Ansi
    s = store.summary()

    print(f"{a.BOLD}Debug Output Summary{a.RESET}")
    print("=" * 40)
    print(f"Messages:      {s.total_messages:,}", end="")
    if s.parse_errors > 0:
        print(f"  ({a.RED}{s.parse_errors} parse errors{a.RESET})")
    else:
        print()
    print(f"Steps:         {s.step_range[0]} - {s.step_range[1]}"
          f"  ({s.num_steps} unique)")
    print(f"Entity types:  {', '.join(s.entity_types)}")
    print(f"Entities:      {s.num_entities:,} unique IDs")
    print(f"Locations:     {s.num_locations:,} unique (x, y)")

    print(f"\n{a.BOLD}Messages per step:{a.RESET}")
    for step, count in s.messages_per_step.items():
        print(f"  Step {step:>4d}: {count:>8,}")

    print(f"\n{a.BOLD}Entities by type:{a.RESET}")
    for etype, count in s.entities_per_type.items():
        print(f"  {etype}: {count:,}")

    # Content pattern counts (first two words)
    print(f"\n{a.BOLD}Content patterns (first 2 words):{a.RESET}")
    patterns: Counter[str] = Counter()
    for msg in store.messages:
        words = msg.content.split(maxsplit=2)
        key = " ".join(words[:2]) if len(words) >= 2 else msg.content
        patterns[key] += 1
    for pattern, count in patterns.most_common(15):
        print(f"  {pattern}: {count:,}")
    remaining = len(patterns) - 15
    if remaining > 0:
        print(f"  ... and {remaining} more patterns")


def _print_trace(store: DebugMessageStore, entity_id: str,
                 use_color: bool) -> None:
    """Print trace output for an entity grouped by step."""
    a = _Ansi
    msgs = store.trace(entity_id)
    if not msgs:
        print(f"No messages for entity '{entity_id}'.")
        return

    print(f"{a.BOLD}Trace: {a.MAGENTA}{entity_id}{a.RESET}"
          f" ({len(msgs)} messages)")
    print()
    print(format_trace(msgs, use_color=use_color, limit=None))


def _print_count(store: DebugMessageStore) -> None:
    """Print message count per step."""
    a = _Ansi
    s = store.summary()
    print(f"{a.BOLD}{'Step':>6s}  {'Count':>8s}{a.RESET}")
    print(f"{'─' * 6}  {'─' * 8}")
    for step, count in s.messages_per_step.items():
        print(f"{step:>6d}  {count:>8,}")


def _print_find(
    matches: list[EventMatch],
    keyword: str,
    use_color: bool,
) -> None:
    """Print find_events results with context and event markers."""
    if not matches:
        print(f"No entities matching \"{keyword}\".")
        return

    for match in matches:
        print()
        print(match.format(use_color=use_color))


def _print_find_count(matches: list[EventMatch], keyword: str) -> None:
    """Print summary counts for find_events results."""
    a = _Ansi
    print(
        f"{a.BOLD}Found {len(matches)} entities matching "
        f"\"{keyword}\"{a.RESET}"
    )
    if not matches:
        return

    # Count entities per event step
    step_counts: Counter[int] = Counter()
    for match in matches:
        for s in match.event_steps:
            step_counts[s] += 1

    if step_counts:
        for s in sorted(step_counts):
            print(f"  Step {s}: {step_counts[s]} entities")

    locations = set(m.location for m in matches)
    print(f"Unique locations: {len(locations)}")

    ids = [m.entity_id for m in matches]
    preview = ids[:10]
    print(f"Entity IDs: {', '.join(preview)}", end="")
    if len(ids) > 10:
        print(f", ... ({len(ids) - 10} more)")
    else:
        print()


def _apply_filters(
    store: DebugMessageStore, args: argparse.Namespace
) -> DebugMessageStore:
    """Apply CLI filter arguments, returning a new store with matched msgs."""
    entity_id = args.entity_id
    if entity_id is not None:
        resolved = _resolve_entity_id(store, entity_id)
        if resolved is None:
            return DebugMessageStore()
        entity_id = resolved

    msgs = store.filter(
        step=args.step,
        step_range=tuple(args.step_range) if args.step_range else None,
        entity_type=args.entity_type,
        entity_id=entity_id,
        x=args.location[0] if args.location else None,
        y=args.location[1] if args.location else None,
        content_contains=args.content,
    )

    filtered = DebugMessageStore()
    filtered.parse_errors = store.parse_errors
    for msg in msgs:
        filtered.add(msg)
    return filtered


def _load_files(args: argparse.Namespace) -> DebugMessageStore:
    """Load one or more debug files into a merged store."""
    files: list[Path] = args.files

    # Single .josh script — use script loader
    if len(files) == 1 and files[0].suffix == ".josh":
        return load_debug_from_script(
            files[0],
            simulation=args.simulation,
            run_hash=args.run_hash,
        )

    # One or more .txt debug files — merge into one store
    store = DebugMessageStore()
    for path in files:
        if path.suffix == ".josh":
            raise RuntimeError(
                "Cannot mix .josh scripts with other files. "
                "Pass a single .josh file or one or more .txt files."
            )
        file_store = load_debug_file(path)
        for msg in file_store.messages:
            store.add(msg)
        store.parse_errors += file_store.parse_errors
    return store


def main() -> int:
    """Entry point for ``python -m joshpy.debug``."""
    parser = _build_parser()
    args = parser.parse_args()

    # Validate --before/--after only with --find
    if (args.before is not None or args.after is not None) and args.find is None:
        parser.error("--before and --after require --find")

    # Color setup
    use_color = not args.no_color and _supports_color()
    if use_color:
        _Ansi.enable()
    else:
        _Ansi.disable()

    # Load files
    try:
        store = _load_files(args)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if len(store) == 0:
        print("No debug messages found.", file=sys.stderr)
        return 0

    # Find mode — uses find_events() directly on the store
    if args.find is not None:
        find_limit = args.limit if args.limit is not None else 3
        matches = store.find_events(
            args.find,
            before=args.before,
            after=args.after,
            entity_type=args.entity_type,
            step=args.step,
            step_range=tuple(args.step_range) if args.step_range else None,
            limit=None if args.count else find_limit,
            print=False,
        )
        if args.count:
            _print_find_count(matches, args.find)
        else:
            _print_find(matches, args.find, use_color)
        return 0

    # Trace mode (resolve ID before filtering)
    if args.trace is not None:
        resolved = _resolve_entity_id(store, args.trace)
        if resolved is None:
            return 0
        # Apply other filters first, then trace within that set
        filtered = _apply_filters(store, args)
        _print_trace(filtered, resolved, use_color=use_color)
        return 0

    # Apply filters for other modes
    has_filters = any([
        args.step is not None,
        args.step_range is not None,
        args.entity_type is not None,
        args.entity_id is not None,
        args.location is not None,
        args.content is not None,
    ])
    filtered = _apply_filters(store, args) if has_filters else store

    # Summary mode
    if args.summary:
        _print_summary(filtered)
        return 0

    # Count mode (without --find)
    if args.count:
        _print_count(filtered)
        return 0

    # Default: print messages
    limit = args.limit
    for i, msg in enumerate(filtered.messages):
        if limit is not None and i >= limit:
            remaining = len(filtered) - limit
            print(f"\n... {remaining:,} more messages (use --limit to adjust)")
            break
        print(format_message(msg, use_color=use_color))

    return 0


if __name__ == "__main__":
    sys.exit(main())
