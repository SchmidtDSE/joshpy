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

from __future__ import annotations

import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from joshpy.cli import JoshCLI

# ---------------------------------------------------------------------------
# Regex for parsing debug lines
# ---------------------------------------------------------------------------

_DEBUG_LINE_RE = re.compile(
    r"^\[Step (\d+), (\w+) @ ([a-f0-9]+) \(([\d.]+), ([\d.]+)\)\] (.*)$"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebugMessage:
    """A single parsed debug message from Josh simulation output.

    Attributes:
        step: Simulation step number (0-indexed).
        entity_type: Entity category (e.g., "organism", "patch").
        entity_id: Hex identifier for the entity instance.
        x: X coordinate in grid space.
        y: Y coordinate in grid space.
        content: Free-text message content.
        line_number: 1-based line number in the source file.
    """

    step: int
    entity_type: str
    entity_id: str
    x: float
    y: float
    content: str
    line_number: int = 0


@dataclass(frozen=True)
class DebugSummary:
    """Statistical summary of debug messages.

    Attributes:
        total_messages: Total number of parsed messages.
        parse_errors: Number of lines that failed to parse.
        step_range: (min_step, max_step) tuple.
        num_steps: Count of unique steps.
        entity_types: Sorted unique entity types.
        num_entities: Count of unique entity IDs.
        num_locations: Count of unique (x, y) locations.
        messages_per_step: Mapping of step -> message count.
        entities_per_type: Mapping of entity_type -> count of unique IDs.
    """

    total_messages: int
    parse_errors: int
    step_range: tuple[int, int]
    num_steps: int
    entity_types: tuple[str, ...]
    num_entities: int
    num_locations: int
    messages_per_step: dict[int, int]
    entities_per_type: dict[str, int]


@dataclass
class DebugMessageStore:
    """Accumulator for parsed debug messages with filtering.

    Attributes:
        messages: All parsed messages in file order.
        parse_errors: Count of lines that failed to parse.
    """

    messages: list[DebugMessage] = field(default_factory=list)
    parse_errors: int = 0
    _locations: set[tuple[float, float]] = field(
        default_factory=set, repr=False
    )
    _steps: set[int] = field(default_factory=set, repr=False)
    _entity_ids: set[str] = field(default_factory=set, repr=False)
    _entity_types: set[str] = field(default_factory=set, repr=False)

    def add(self, msg: DebugMessage) -> None:
        """Add a parsed message to the store."""
        self.messages.append(msg)
        self._locations.add((msg.x, msg.y))
        self._steps.add(msg.step)
        self._entity_ids.add(msg.entity_id)
        self._entity_types.add(msg.entity_type)

    def filter(
        self,
        *,
        step: int | None = None,
        step_range: tuple[int, int] | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        x: float | None = None,
        y: float | None = None,
        content_contains: str | None = None,
    ) -> list[DebugMessage]:
        """Return messages matching all non-None criteria (AND logic).

        Args:
            step: Exact step number.
            step_range: Inclusive (min, max) step range.
            entity_type: Entity type string.
            entity_id: Exact entity ID (hex).
            x: Exact X coordinate.
            y: Exact Y coordinate.
            content_contains: Substring match on message content.

        Returns:
            List of matching messages in file order.
        """
        result = self.messages
        if step is not None:
            result = [m for m in result if m.step == step]
        if step_range is not None:
            lo, hi = step_range
            result = [m for m in result if lo <= m.step <= hi]
        if entity_type is not None:
            result = [m for m in result if m.entity_type == entity_type]
        if entity_id is not None:
            result = [m for m in result if m.entity_id == entity_id]
        if x is not None:
            result = [m for m in result if m.x == x]
        if y is not None:
            result = [m for m in result if m.y == y]
        if content_contains is not None:
            result = [m for m in result if content_contains in m.content]
        return result

    def steps(self) -> list[int]:
        """Return sorted list of unique step numbers."""
        return sorted(self._steps)

    def entity_types(self) -> list[str]:
        """Return sorted list of unique entity types."""
        return sorted(self._entity_types)

    def entity_ids(self) -> list[str]:
        """Return sorted list of unique entity IDs."""
        return sorted(self._entity_ids)

    def locations(self) -> list[tuple[float, float]]:
        """Return sorted list of unique (x, y) locations."""
        return sorted(self._locations)

    def trace(self, entity_id: str) -> list[DebugMessage]:
        """Get all messages for an entity, sorted by step then line number.

        Args:
            entity_id: Exact entity ID (hex string).

        Returns:
            Messages for that entity in chronological order.
        """
        msgs = [m for m in self.messages if m.entity_id == entity_id]
        msgs.sort(key=lambda m: (m.step, m.line_number))
        return msgs

    def resolve_entity_id(self, prefix: str) -> str | None:
        """Resolve a prefix to a full entity ID (like git short SHAs).

        Args:
            prefix: Hex prefix to match against known entity IDs.

        Returns:
            Full entity ID if exactly one match, None otherwise.
        """
        if prefix in self._entity_ids:
            return prefix
        matches = [eid for eid in self._entity_ids if eid.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        return None

    def summary(self) -> DebugSummary:
        """Compute summary statistics for the stored messages."""
        steps = self.steps()
        step_range = (steps[0], steps[-1]) if steps else (0, 0)

        messages_per_step: dict[int, int] = Counter()
        entities_per_type: dict[str, set[str]] = {}
        for msg in self.messages:
            messages_per_step[msg.step] += 1
            entities_per_type.setdefault(msg.entity_type, set()).add(
                msg.entity_id
            )

        return DebugSummary(
            total_messages=len(self.messages),
            parse_errors=self.parse_errors,
            step_range=step_range,
            num_steps=len(steps),
            entity_types=tuple(sorted(entities_per_type.keys())),
            num_entities=len(self._entity_ids),
            num_locations=len(self._locations),
            messages_per_step=dict(sorted(messages_per_step.items())),
            entities_per_type={
                k: len(v) for k, v in sorted(entities_per_type.items())
            },
        )

    def __len__(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_debug_line(line: str, line_number: int = 0) -> DebugMessage | None:
    """Parse a single debug output line into a DebugMessage.

    Args:
        line: Raw text line from a debug file.
        line_number: Optional 1-based line number for provenance.

    Returns:
        Parsed DebugMessage, or None if the line doesn't match.
    """
    match = _DEBUG_LINE_RE.match(line.strip())
    if match is None:
        return None
    return DebugMessage(
        step=int(match.group(1)),
        entity_type=match.group(2),
        entity_id=match.group(3),
        x=float(match.group(4)),
        y=float(match.group(5)),
        content=match.group(6),
        line_number=line_number,
    )


def load_debug_file(path: Path | str) -> DebugMessageStore:
    """Load and parse a Josh debug file into a DebugMessageStore.

    Reads line-by-line for memory efficiency on large files. Lines that don't
    match the expected format are counted as parse errors.

    Args:
        path: Path to the debug text file.

    Returns:
        DebugMessageStore containing all parsed messages.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Debug file not found: {path}")

    store = DebugMessageStore()
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            msg = parse_debug_line(stripped, line_number=line_number)
            if msg is not None:
                store.add(msg)
            else:
                store.parse_errors += 1
    if store.parse_errors > 0:
        warnings.warn(
            f"Skipped {store.parse_errors} unparseable lines in {path}",
            stacklevel=2,
        )
    return store


def load_debug_from_script(
    script: Path | str,
    simulation: str = "Main",
    entity_types: list[str] | None = None,
    cli: JoshCLI | None = None,
    run_hash: str | None = None,
    **template_vars: str,
) -> DebugMessageStore:
    """Discover debug file paths from a .josh script and load them.

    Uses ``inspect-exports`` to find configured debug output paths, then
    loads and merges all discovered debug files into a single store.

    Debug file paths in Josh scripts can contain template variables like
    ``{run_hash}`` or ``{replicate}``. Pass these as keyword arguments
    to resolve them before loading.

    Args:
        script: Path to a .josh simulation file.
        simulation: Simulation name (default: "Main").
        entity_types: Which debug file types to load. Default loads all
            configured types (organism, patch, agent, disturbance).
        cli: JoshCLI instance. Created automatically if not provided.
        run_hash: Run hash to resolve ``{run_hash}`` in debug file paths.
        **template_vars: Additional template variables to resolve in paths
            (e.g., ``replicate=0``).

    Returns:
        DebugMessageStore with messages from all discovered debug files.

    Raises:
        FileNotFoundError: If the script does not exist.
        RuntimeError: If inspect-exports fails.
    """
    from joshpy.cli import InspectExportsConfig, JoshCLI

    script = Path(script)
    if not script.exists():
        raise FileNotFoundError(f"Josh script not found: {script}")

    if cli is None:
        cli = JoshCLI()

    exports = cli.inspect_exports(
        InspectExportsConfig(script=script, simulation=simulation)
    )

    # Build template resolution kwargs
    resolve_kwargs: dict[str, str] = dict(template_vars)
    if run_hash is not None:
        resolve_kwargs["run_hash"] = run_hash

    all_types = ["organism", "patch", "agent", "disturbance"]
    types_to_load = entity_types if entity_types is not None else all_types

    store = DebugMessageStore()
    for etype in types_to_load:
        info = exports.debug_files.get(etype)
        if info is None:
            continue

        # Resolve template variables in the path
        raw_path = info.path
        if resolve_kwargs:
            try:
                resolved = raw_path.format(**resolve_kwargs)
            except KeyError as e:
                warnings.warn(
                    f"Debug path for {etype} contains unresolved template "
                    f"variable {e}: {raw_path}",
                    stacklevel=2,
                )
                continue
            debug_path = Path(resolved)
        else:
            debug_path = Path(raw_path)

        if "{" in str(debug_path):
            warnings.warn(
                f"Debug path for {etype} contains unresolved template "
                f"variables: {debug_path}. Pass run_hash= or template_vars "
                f"to resolve them.",
                stacklevel=2,
            )
            continue

        if not debug_path.exists():
            warnings.warn(
                f"Debug file for {etype} not found at {debug_path} "
                f"(has the simulation been run?)",
                stacklevel=2,
            )
            continue
        file_store = load_debug_file(debug_path)
        for msg in file_store.messages:
            store.add(msg)
        store.parse_errors += file_store.parse_errors

    return store


# ---------------------------------------------------------------------------
# Terminal color support
# ---------------------------------------------------------------------------


def _supports_color() -> bool:
    """Check if stdout supports ANSI color codes.

    Respects the NO_COLOR convention (https://no-color.org/) and checks
    whether stdout is a TTY.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


class _Ansi:
    """ANSI escape code container.

    All attributes are empty strings until ``enable()`` is called, so
    formatting code works unconditionally without branching.
    """

    RESET = ""
    BOLD = ""
    DIM = ""
    RED = ""
    GREEN = ""
    YELLOW = ""
    CYAN = ""
    MAGENTA = ""
    BRIGHT_CYAN = ""

    @classmethod
    def enable(cls) -> None:
        """Populate escape codes for color output."""
        cls.RESET = "\033[0m"
        cls.BOLD = "\033[1m"
        cls.DIM = "\033[2m"
        cls.RED = "\033[31m"
        cls.GREEN = "\033[32m"
        cls.YELLOW = "\033[33m"
        cls.CYAN = "\033[36m"
        cls.MAGENTA = "\033[35m"
        cls.BRIGHT_CYAN = "\033[96m"

    @classmethod
    def disable(cls) -> None:
        """Clear all escape codes (no-color mode)."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.CYAN = ""
        cls.MAGENTA = ""
        cls.BRIGHT_CYAN = ""


def _colorize_content(content: str) -> str:
    """Apply keyword highlighting to message content.

    Colors the words ``true`` and ``false`` green/red inline.
    """
    a = _Ansi
    # Replace whole-word occurrences only
    result = re.sub(
        r"\btrue\b",
        f"{a.GREEN}true{a.RESET}",
        content,
    )
    result = re.sub(
        r"\bfalse\b",
        f"{a.RED}false{a.RESET}",
        result,
    )
    return result


def format_message(msg: DebugMessage, use_color: bool = True) -> str:
    """Format a DebugMessage for terminal display.

    Reproduces the original log format with optional ANSI coloring.

    Args:
        msg: The debug message to format.
        use_color: Whether to apply ANSI color codes.

    Returns:
        Formatted string suitable for printing.
    """
    if use_color:
        a = _Ansi
        content = _colorize_content(msg.content)
        return (
            f"[{a.BOLD}{a.YELLOW}Step {msg.step}{a.RESET}, "
            f"{a.CYAN}{msg.entity_type}{a.RESET} @ "
            f"{a.MAGENTA}{msg.entity_id}{a.RESET} "
            f"{a.DIM}({msg.x}, {msg.y}){a.RESET}] "
            f"{content}"
        )
    return (
        f"[Step {msg.step}, {msg.entity_type} @ {msg.entity_id} "
        f"({msg.x}, {msg.y})] {msg.content}"
    )
