"""Parse JFR recordings into user-friendly resource diagnostics.

JFR (Java Flight Recorder) captures detailed JVM profiling data. This module
parses recordings into actionable sections — CPU, Memory, GC, File I/O,
Thread Contention — with plain-language analysis.

Example usage::

    from joshpy.jfr import build_resource_profile

    profile = build_resource_profile(event_text, summary_text)
    print(profile.summary())

CLI usage::

    python -m joshpy.jfr recording.jfr
    python -m joshpy.jfr recording.jfr --detailed
"""

from joshpy.jfr._core import (  # noqa: F401
    ContentionProfile,
    CpuProfile,
    GcProfile,
    IoProfile,
    MemoryProfile,
    ResourceProfile,
    build_resource_profile,
    parse_jfr_events,
)
