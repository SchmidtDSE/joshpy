"""Parse and inspect Josh simulation debug output.

Josh simulations can emit structured debug messages with metadata (step, entity
type, entity ID, coordinates). This module parses those messages and provides
filtering, summarization, and colored terminal output for tracing agent behavior.

Example usage::

    from pathlib import Path
    from joshpy.debug import load_debug_file

    store = load_debug_file(Path("debug.txt"))
    print(f"Parsed {len(store)} messages")

    # Filter by step and entity type
    for msg in store.filter(step=5, entity_type="organism"):
        print(msg.content)

    # Trace a specific entity across all timesteps
    for msg in store.trace("a1b2c3d4"):
        print(f"Step {msg.step}: {msg.content}")

CLI usage::

    python -m joshpy.debug debug.txt --summary
    python -m joshpy.debug debug.txt --step 5 --entity-type organism
    python -m joshpy.debug debug.txt --trace a1b2c3d4
"""

from joshpy.debug._core import (  # noqa: F401
    DebugMessage,
    DebugMessageStore,
    DebugSummary,
    EventMatch,
    _Ansi,
    _supports_color,
    format_message,
    format_trace,
    load_debug_file,
    load_debug_from_script,
    parse_debug_line,
)
