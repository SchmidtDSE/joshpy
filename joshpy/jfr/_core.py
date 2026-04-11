"""JFR resource diagnostics — parse JFR recordings into user-friendly profiles.

Parses the output of ``jfr print`` and ``jfr summary`` into typed dataclasses
with a human-readable ``summary()`` method. The parser is a pure function
(no subprocess calls) so it can be tested with string fixtures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

_DURATION_UNITS: dict[str, float] = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1000.0,
}

_SIZE_UNITS: dict[str, float] = {
    "bytes": 1.0,
    "kb": 1024.0,
    "mb": 1024.0**2,
    "gb": 1024.0**3,
}


def _parse_duration(value: str) -> float:
    """Parse a JFR duration string to milliseconds.

    Examples: ``"12.3 ms"``, ``"1.20 s"``, ``"500 us"``, ``"1000 ns"``.
    """
    value = value.strip()
    for suffix, factor in _DURATION_UNITS.items():
        if value.endswith(" " + suffix):
            return float(value[: -len(suffix)].strip()) * factor
    # Bare number — assume nanoseconds (some older JDK versions)
    try:
        return float(value) * 1e-6
    except ValueError:
        return 0.0


def _parse_size(value: str) -> float:
    """Parse a JFR size string to bytes.

    Examples: ``"123.4 MB"``, ``"1.2 GB"``, ``"1234 bytes"``.
    """
    value = value.strip()
    for suffix, factor in _SIZE_UNITS.items():
        if value.lower().endswith(" " + suffix):
            num = value[: -len(suffix)].strip()
            return float(num) * factor
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_percent(value: str) -> float:
    """Parse ``"12.34%"`` to ``12.34``."""
    return float(value.strip().rstrip("%"))


# ---------------------------------------------------------------------------
# Generic JFR event parser
# ---------------------------------------------------------------------------

_EVENT_HEADER_RE = re.compile(r"^(\S+)\s*\{")
_FIELD_RE = re.compile(r"^\s+(\w[\w.]*)\s*=\s*(.+)$")


def parse_jfr_events(text: str) -> list[dict[str, str]]:
    """Parse ``jfr print`` output into a list of event dicts.

    Each dict has a ``_type`` key (e.g. ``"jdk.CPULoad"``) plus one key per
    field with the raw string value.  Nested blocks (like ``heapSpace``) are
    flattened with dot-separated keys (``heapSpace.committedSize``).
    """
    events: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    prefix = ""

    for line in text.splitlines():
        stripped = line.strip()

        if current is None:
            m = _EVENT_HEADER_RE.match(stripped)
            if m:
                current = {"_type": m.group(1)}
            continue

        if stripped == "}":
            if prefix:
                # Closing a nested block
                prefix = ""
                continue
            events.append(current)
            current = None
            continue

        # Detect nested block opening: "heapSpace = {"
        if stripped.endswith("{"):
            key = stripped.split("=")[0].strip()
            prefix = key + "."
            continue

        m = _FIELD_RE.match(line)
        if m:
            key = prefix + m.group(1)
            current[key] = m.group(2).strip()

    return events


def _parse_recording_duration(summary_text: str) -> float:
    """Extract recording duration in seconds from ``jfr summary`` header."""
    for line in summary_text.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            parts = line.split(":", 1)[1].strip()
            # Format: "4 s" or "1.2 s" or "125 ms"
            for suffix, factor in _DURATION_UNITS.items():
                if parts.endswith(" " + suffix):
                    return float(parts[: -len(suffix)].strip()) * factor / 1000.0
            try:
                return float(parts)
            except ValueError:
                pass
    return 0.0


# ---------------------------------------------------------------------------
# Sub-profile dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CpuProfile:
    """CPU utilization from jdk.CPULoad events.

    Attributes:
        jvm_user_pct: Mean JVM user CPU percent.
        jvm_system_pct: Mean JVM system CPU percent.
        machine_total_pct: Mean total machine CPU percent.
        sample_count: Number of CPULoad samples.
    """

    jvm_user_pct: float
    jvm_system_pct: float
    machine_total_pct: float
    sample_count: int


@dataclass(frozen=True)
class GcProfile:
    """Garbage collection from jdk.GCPhasePause and jdk.GCCPUTime events.

    Attributes:
        total_pause_ms: Sum of all GC pause durations (ms).
        max_pause_ms: Longest single GC pause (ms).
        pause_count: Number of GC pause events.
        gc_user_time_ms: Total GC user CPU time (ms).
        gc_system_time_ms: Total GC system CPU time (ms).
        gc_real_time_ms: Total GC real (wall) time (ms).
    """

    total_pause_ms: float
    max_pause_ms: float
    pause_count: int
    gc_user_time_ms: float
    gc_system_time_ms: float
    gc_real_time_ms: float


@dataclass(frozen=True)
class MemoryProfile:
    """Memory usage from jdk.GCHeapSummary, jdk.ResidentSetSize, jdk.PhysicalMemory.

    All values in megabytes.

    Attributes:
        peak_heap_mb: Peak observed heap usage.
        reserved_heap_mb: Reserved (max) heap size.
        peak_rss_mb: Peak resident set size.
        physical_memory_mb: Total physical memory on the machine.
    """

    peak_heap_mb: float
    reserved_heap_mb: float
    peak_rss_mb: float
    physical_memory_mb: float


@dataclass(frozen=True)
class IoProfile:
    """File I/O from jdk.FileRead and jdk.FileWrite events.

    Attributes:
        read_count: Number of FileRead events.
        read_total_ms: Sum of FileRead durations.
        read_total_bytes: Sum of bytes read.
        write_count: Number of FileWrite events.
        write_total_ms: Sum of FileWrite durations.
        write_total_bytes: Sum of bytes written.
    """

    read_count: int
    read_total_ms: float
    read_total_bytes: int
    write_count: int
    write_total_ms: float
    write_total_bytes: int


@dataclass(frozen=True)
class ContentionProfile:
    """Thread contention from jdk.JavaMonitorWait and jdk.JavaMonitorEnter.

    Attributes:
        wait_count: Number of JavaMonitorWait events.
        wait_total_ms: Sum of wait durations.
        enter_count: Number of JavaMonitorEnter events.
        enter_total_ms: Sum of enter (blocked) durations.
    """

    wait_count: int
    wait_total_ms: float
    enter_count: int
    enter_total_ms: float


# ---------------------------------------------------------------------------
# Sub-profile builders
# ---------------------------------------------------------------------------


def _build_cpu_profile(events: list[dict[str, str]]) -> CpuProfile | None:
    cpu_events = [e for e in events if e["_type"] == "jdk.CPULoad"]
    if not cpu_events:
        return None
    n = len(cpu_events)
    return CpuProfile(
        jvm_user_pct=sum(_parse_percent(e.get("jvmUser", "0%")) for e in cpu_events) / n,
        jvm_system_pct=sum(_parse_percent(e.get("jvmSystem", "0%")) for e in cpu_events) / n,
        machine_total_pct=sum(_parse_percent(e.get("machineTotal", "0%")) for e in cpu_events) / n,
        sample_count=n,
    )


def _build_gc_profile(events: list[dict[str, str]]) -> GcProfile | None:
    pauses = [e for e in events if e["_type"] == "jdk.GCPhasePause"]
    cpu_events = [e for e in events if e["_type"] == "jdk.GCCPUTime"]
    if not pauses and not cpu_events:
        return None
    durations = [_parse_duration(e.get("duration", "0 ms")) for e in pauses]
    return GcProfile(
        total_pause_ms=sum(durations),
        max_pause_ms=max(durations) if durations else 0.0,
        pause_count=len(pauses),
        gc_user_time_ms=sum(_parse_duration(e.get("userTime", "0 ms")) for e in cpu_events),
        gc_system_time_ms=sum(_parse_duration(e.get("systemTime", "0 ms")) for e in cpu_events),
        gc_real_time_ms=sum(_parse_duration(e.get("realTime", "0 ms")) for e in cpu_events),
    )


def _build_memory_profile(events: list[dict[str, str]]) -> MemoryProfile | None:
    heap_events = [e for e in events if e["_type"] == "jdk.GCHeapSummary"]
    rss_events = [e for e in events if e["_type"] == "jdk.ResidentSetSize"]
    phys_events = [e for e in events if e["_type"] == "jdk.PhysicalMemory"]
    if not heap_events and not rss_events and not phys_events:
        return None

    mb = 1024.0**2

    # Peak heap from all GCHeapSummary events
    peak_heap = 0.0
    reserved_heap = 0.0
    for e in heap_events:
        used = _parse_size(e.get("heapUsed", "0"))
        reserved = _parse_size(e.get("heapSpace.reservedSize", "0"))
        peak_heap = max(peak_heap, used)
        reserved_heap = max(reserved_heap, reserved)

    peak_rss = 0.0
    for e in rss_events:
        peak_rss = max(peak_rss, _parse_size(e.get("peak", "0")))

    physical = 0.0
    for e in phys_events:
        physical = max(physical, _parse_size(e.get("totalSize", "0")))

    return MemoryProfile(
        peak_heap_mb=peak_heap / mb,
        reserved_heap_mb=reserved_heap / mb,
        peak_rss_mb=peak_rss / mb,
        physical_memory_mb=physical / mb,
    )


def _build_io_profile(events: list[dict[str, str]]) -> IoProfile | None:
    reads = [e for e in events if e["_type"] == "jdk.FileRead"]
    writes = [e for e in events if e["_type"] == "jdk.FileWrite"]
    if not reads and not writes:
        return None
    return IoProfile(
        read_count=len(reads),
        read_total_ms=sum(_parse_duration(e.get("duration", "0 ms")) for e in reads),
        read_total_bytes=sum(int(_parse_size(e.get("bytesRead", "0"))) for e in reads),
        write_count=len(writes),
        write_total_ms=sum(_parse_duration(e.get("duration", "0 ms")) for e in writes),
        write_total_bytes=sum(int(_parse_size(e.get("bytesWritten", "0"))) for e in writes),
    )


def _build_contention_profile(events: list[dict[str, str]]) -> ContentionProfile | None:
    waits = [e for e in events if e["_type"] == "jdk.JavaMonitorWait"]
    enters = [e for e in events if e["_type"] == "jdk.JavaMonitorEnter"]
    if not waits and not enters:
        return None
    return ContentionProfile(
        wait_count=len(waits),
        wait_total_ms=sum(_parse_duration(e.get("duration", "0 ms")) for e in waits),
        enter_count=len(enters),
        enter_total_ms=sum(_parse_duration(e.get("duration", "0 ms")) for e in enters),
    )


# ---------------------------------------------------------------------------
# ResourceProfile
# ---------------------------------------------------------------------------


def _fmt_size(mb: float) -> str:
    """Format megabytes to a human-readable string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _fmt_bytes(b: int) -> str:
    """Format bytes to a human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} bytes"


@dataclass(frozen=True)
class ResourceProfile:
    """Complete resource profile parsed from a JFR recording.

    Attributes:
        recording_duration_s: Total recording wall-clock duration in seconds.
        cpu: CPU utilization profile, or None if no data.
        gc: Garbage collection profile, or None if no data.
        memory: Memory usage profile, or None if no data.
        io: File I/O profile, or None if no data.
        contention: Thread contention profile, or None if no data.
    """

    recording_duration_s: float
    cpu: CpuProfile | None
    gc: GcProfile | None
    memory: MemoryProfile | None
    io: IoProfile | None
    contention: ContentionProfile | None

    def _gc_pct(self) -> float:
        if self.gc and self.recording_duration_s > 0:
            return self.gc.total_pause_ms / (self.recording_duration_s * 1000) * 100
        return 0.0

    def _io_pct(self) -> float:
        if self.io and self.recording_duration_s > 0:
            total = self.io.read_total_ms + self.io.write_total_ms
            return total / (self.recording_duration_s * 1000) * 100
        return 0.0

    def _contention_pct(self) -> float:
        if self.contention and self.recording_duration_s > 0:
            total = self.contention.wait_total_ms + self.contention.enter_total_ms
            return total / (self.recording_duration_s * 1000) * 100
        return 0.0

    def classify(self) -> tuple[str, str]:
        """Classify the likely bottleneck.

        Returns:
            (bottleneck_type, explanation) tuple.
        """
        # 1. GC-bound
        gc_pct = self._gc_pct()
        if gc_pct > 10:
            return (
                "GC-bound",
                f"GC pauses account for {gc_pct:.0f}% of recording time. "
                "This usually means too many entities creating allocation pressure, "
                "or the heap is too small. Try reducing entity count or increasing "
                "JVM heap size (-Xmx).",
            )

        # 2. Memory pressure
        if self.memory:
            if self.memory.reserved_heap_mb > 0:
                heap_pct = self.memory.peak_heap_mb / self.memory.reserved_heap_mb
                if heap_pct > 0.85:
                    return (
                        "memory pressure",
                        f"Peak heap is {_fmt_size(self.memory.peak_heap_mb)} of "
                        f"{_fmt_size(self.memory.reserved_heap_mb)} reserved "
                        f"({heap_pct:.0%}). The JVM may be running out of room. "
                        "Increase heap size (-Xmx) or reduce entity count.",
                    )
            if self.memory.physical_memory_mb > 0:
                rss_pct = self.memory.peak_rss_mb / self.memory.physical_memory_mb
                if rss_pct > 0.80:
                    return (
                        "memory pressure",
                        f"Process RSS ({_fmt_size(self.memory.peak_rss_mb)}) is "
                        f"{rss_pct:.0%} of physical memory "
                        f"({_fmt_size(self.memory.physical_memory_mb)}). "
                        "The machine may be swapping. Reduce entity count or "
                        "use a machine with more RAM.",
                    )

        # 3. I/O-bound
        io_pct = self._io_pct()
        if io_pct > 10:
            return (
                "I/O-bound",
                f"File I/O accounts for {io_pct:.0f}% of recording time. "
                "Check if data files can be pre-loaded or if export frequency "
                "can be reduced.",
            )

        # 4. Thread contention
        cont_pct = self._contention_pct()
        if cont_pct > 10:
            return (
                "thread contention",
                f"Lock contention accounts for {cont_pct:.0f}% of recording time. "
                "This typically occurs with --parallel. Try running without "
                "--parallel or reducing the number of parallel patches.",
            )

        # 5. CPU-bound
        if self.cpu and self.cpu.jvm_user_pct > 50:
            return (
                "compute-bound",
                f"JVM CPU usage is {self.cpu.jvm_user_pct:.0f}% with minimal "
                "GC/IO overhead. Use enable_profiler=True with .evalDuration "
                "to find which attributes are slowest.",
            )

        # 6. No clear bottleneck
        return (
            "no clear bottleneck",
            "No single resource dominates. The simulation may be running "
            "efficiently, or the recording was too short to capture the "
            "bottleneck. Try a longer recording.",
        )

    def summary(self, detailed: bool = False) -> str:
        """Format a human-readable diagnostic summary.

        Args:
            detailed: If True, include per-event breakdowns.
        """
        lines: list[str] = []
        dur = self.recording_duration_s
        lines.append(f"JFR Resource Diagnostic ({dur:.1f}s recording)")
        lines.append("=" * 40)

        # --- CPU ---
        lines.append("")
        lines.append("--- CPU ---")
        lines.append("How much processor time the JVM uses.")
        lines.append("")
        if self.cpu:
            lines.append(f"  JVM user:      {self.cpu.jvm_user_pct:.1f}%")
            lines.append(f"  JVM system:    {self.cpu.jvm_system_pct:.1f}%")
            lines.append(f"  Machine total: {self.cpu.machine_total_pct:.1f}%")
            lines.append(f"  ({self.cpu.sample_count} samples)")
        else:
            lines.append("  No CPU data recorded.")

        # --- MEMORY ---
        lines.append("")
        lines.append("--- MEMORY ---")
        lines.append("Java heap and OS-level memory footprint.")
        lines.append("")
        if self.memory:
            lines.append(
                f"  Heap used:     {_fmt_size(self.memory.peak_heap_mb)} "
                f"/ {_fmt_size(self.memory.reserved_heap_mb)} reserved"
            )
            if self.memory.peak_rss_mb > 0:
                lines.append(f"  Process RSS:   {_fmt_size(self.memory.peak_rss_mb)} peak")
            if self.memory.physical_memory_mb > 0:
                lines.append(
                    f"  Physical RAM:  {_fmt_size(self.memory.physical_memory_mb)}"
                )
        else:
            lines.append("  No memory data recorded.")

        # --- GARBAGE COLLECTION ---
        lines.append("")
        lines.append("--- GARBAGE COLLECTION ---")
        lines.append(
            "GC pauses stop all threads to reclaim memory. High GC time\n"
            "often means too many entities or too little heap."
        )
        lines.append("")
        if self.gc:
            lines.append(
                f"  Pauses: {self.gc.pause_count} collections, "
                f"{self.gc.total_pause_ms:.0f} ms total, "
                f"{self.gc.max_pause_ms:.1f} ms max"
            )
            gc_pct = self._gc_pct()
            lines.append(f"  GC fraction: {gc_pct:.1f}% of recording")
            if detailed:
                lines.append("")
                lines.append(f"  GC CPU — user: {self.gc.gc_user_time_ms:.0f} ms, "
                             f"system: {self.gc.gc_system_time_ms:.0f} ms, "
                             f"real: {self.gc.gc_real_time_ms:.0f} ms")
        else:
            lines.append("  No GC data recorded.")

        # --- FILE I/O ---
        lines.append("")
        lines.append("--- FILE I/O ---")
        lines.append("Time reading/writing files (data loading, CSV export).")
        lines.append("")
        if self.io:
            if self.io.read_count > 0:
                lines.append(
                    f"  Reads:  {self.io.read_count:,} ops, "
                    f"{_fmt_bytes(self.io.read_total_bytes)}, "
                    f"{self.io.read_total_ms:.0f} ms"
                )
            if self.io.write_count > 0:
                lines.append(
                    f"  Writes: {self.io.write_count:,} ops, "
                    f"{_fmt_bytes(self.io.write_total_bytes)}, "
                    f"{self.io.write_total_ms:.0f} ms"
                )
            if self.io.read_count == 0 and self.io.write_count == 0:
                lines.append("  No file I/O events recorded.")
        else:
            lines.append("  No file I/O events recorded.")

        # --- THREAD CONTENTION ---
        lines.append("")
        lines.append("--- THREAD CONTENTION ---")
        lines.append("Lock contention when running with --parallel.")
        lines.append("")
        if self.contention and (self.contention.wait_count > 0 or self.contention.enter_count > 0):
            if self.contention.wait_count > 0:
                lines.append(
                    f"  Monitor waits:  {self.contention.wait_count:,} events, "
                    f"{self.contention.wait_total_ms:.0f} ms total"
                )
            if self.contention.enter_count > 0:
                lines.append(
                    f"  Monitor enters: {self.contention.enter_count:,} events, "
                    f"{self.contention.enter_total_ms:.0f} ms total"
                )
        else:
            lines.append("  No contention events recorded.")

        # --- DIAGNOSIS ---
        lines.append("")
        lines.append("--- DIAGNOSIS ---")
        bottleneck, explanation = self.classify()
        lines.append(f"Likely bottleneck: {bottleneck}")
        lines.append(explanation)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_resource_profile(
    event_text: str,
    summary_text: str,
) -> ResourceProfile:
    """Build a ResourceProfile from raw ``jfr print`` and ``jfr summary`` output.

    This is a pure function with no subprocess calls.

    Args:
        event_text: Output of ``jfr print --events <event_list>``.
        summary_text: Output of ``jfr summary``.

    Returns:
        Parsed ResourceProfile.
    """
    events = parse_jfr_events(event_text)
    duration = _parse_recording_duration(summary_text)

    return ResourceProfile(
        recording_duration_s=duration,
        cpu=_build_cpu_profile(events),
        gc=_build_gc_profile(events),
        memory=_build_memory_profile(events),
        io=_build_io_profile(events),
        contention=_build_contention_profile(events),
    )
