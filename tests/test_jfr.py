"""Tests for joshpy.jfr — JFR resource diagnostics."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from joshpy.jfr._core import (
    ContentionProfile,
    CpuProfile,
    GcProfile,
    IoProfile,
    MemoryProfile,
    ResourceProfile,
    _parse_duration,
    _parse_percent,
    _parse_recording_duration,
    _parse_size,
    build_resource_profile,
    parse_jfr_events,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CPU_LOAD = """\
jdk.CPULoad {
  startTime = 17:58:26.823 (2026-04-11)
  jvmUser = 45.20%
  jvmSystem = 3.10%
  machineTotal = 52.80%
}

jdk.CPULoad {
  startTime = 17:58:27.823 (2026-04-11)
  jvmUser = 60.00%
  jvmSystem = 2.00%
  machineTotal = 68.00%
}
"""

SAMPLE_GC = """\
jdk.GCPhasePause {
  startTime = 17:58:26.615 (2026-04-11)
  duration = 6.35 ms
  gcId = 1
  name = "GC Pause"
}

jdk.GCPhasePause {
  startTime = 17:58:26.685 (2026-04-11)
  duration = 9.14 ms
  gcId = 2
  name = "GC Pause"
}

jdk.GCCPUTime {
  startTime = 17:58:26.621 (2026-04-11)
  gcId = 1
  userTime = 20.0 ms
  systemTime = 10.0 ms
  realTime = 10.0 ms
}

jdk.GCCPUTime {
  startTime = 17:58:26.694 (2026-04-11)
  gcId = 2
  userTime = 60.0 ms
  systemTime = 10.0 ms
  realTime = 10.0 ms
}
"""

SAMPLE_HEAP = """\
jdk.GCHeapSummary {
  startTime = 17:58:26.621 (2026-04-11)
  gcId = 1
  when = "After GC"
  heapSpace = {
    start = 0x706000000
    committedEnd = 0x715C00000
    committedSize = 252.0 MB
    reservedEnd = 0x800000000
    reservedSize = 3.9 GB
  }
  heapUsed = 16.6 MB
}

jdk.GCHeapSummary {
  startTime = 17:58:26.960 (2026-04-11)
  gcId = 3
  when = "Before GC"
  heapSpace = {
    start = 0x706000000
    committedEnd = 0x715C00000
    committedSize = 252.0 MB
    reservedEnd = 0x800000000
    reservedSize = 3.9 GB
  }
  heapUsed = 142.6 MB
}

jdk.ResidentSetSize {
  startTime = 17:58:26.823 (2026-04-11)
  size = 214.7 MB
  peak = 215.6 MB
}

jdk.PhysicalMemory {
  startTime = 17:58:25.872 (2026-04-11)
  totalSize = 15.6 GB
  usedSize = 6.1 GB
}
"""

SAMPLE_IO = """\
jdk.FileRead {
  startTime = 17:58:26.500 (2026-04-11)
  duration = 25.3 ms
  bytesRead = 1048576
}

jdk.FileWrite {
  startTime = 17:58:27.000 (2026-04-11)
  duration = 12.1 ms
  bytesWritten = 524288
}
"""

SAMPLE_CONTENTION = """\
jdk.JavaMonitorWait {
  startTime = 17:58:26.700 (2026-04-11)
  duration = 150.0 ms
}

jdk.JavaMonitorEnter {
  startTime = 17:58:26.800 (2026-04-11)
  duration = 30.0 ms
}
"""

SAMPLE_SUMMARY = """\
 Version: 2.1
 Chunks: 1
 Start: 2026-04-11 17:58:25 (UTC)
 Duration: 4 s

 Event Type                              Count  Size (bytes)
=============================================================
 jdk.GCPhaseParallel                      3327         86236
 jdk.CPULoad                                 3            57
"""


# ---------------------------------------------------------------------------
# Value parser tests
# ---------------------------------------------------------------------------


class TestParseDuration(unittest.TestCase):
    def test_milliseconds(self):
        self.assertAlmostEqual(_parse_duration("12.3 ms"), 12.3)

    def test_seconds(self):
        self.assertAlmostEqual(_parse_duration("1.5 s"), 1500.0)

    def test_microseconds(self):
        self.assertAlmostEqual(_parse_duration("500 us"), 0.5)

    def test_nanoseconds(self):
        self.assertAlmostEqual(_parse_duration("1000000 ns"), 1.0)

    def test_bare_number_fallback(self):
        # Bare number assumed nanoseconds
        result = _parse_duration("5000000")
        self.assertAlmostEqual(result, 5.0)

    def test_invalid(self):
        self.assertEqual(_parse_duration("bogus"), 0.0)


class TestParseSize(unittest.TestCase):
    def test_megabytes(self):
        self.assertAlmostEqual(_parse_size("252.0 MB"), 252.0 * 1024**2)

    def test_gigabytes(self):
        self.assertAlmostEqual(_parse_size("3.9 GB"), 3.9 * 1024**3)

    def test_bytes(self):
        self.assertAlmostEqual(_parse_size("1234 bytes"), 1234.0)

    def test_kilobytes(self):
        self.assertAlmostEqual(_parse_size("100 kB"), 100 * 1024.0)

    def test_invalid(self):
        self.assertEqual(_parse_size("bogus"), 0.0)


class TestParsePercent(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(_parse_percent("45.20%"), 45.20)

    def test_zero(self):
        self.assertAlmostEqual(_parse_percent("0.00%"), 0.0)


# ---------------------------------------------------------------------------
# Event parser tests
# ---------------------------------------------------------------------------


class TestParseJfrEvents(unittest.TestCase):
    def test_cpu_load(self):
        events = parse_jfr_events(SAMPLE_CPU_LOAD)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["_type"], "jdk.CPULoad")
        self.assertEqual(events[0]["jvmUser"], "45.20%")
        self.assertEqual(events[1]["jvmUser"], "60.00%")

    def test_nested_blocks(self):
        events = parse_jfr_events(SAMPLE_HEAP)
        heap_events = [e for e in events if e["_type"] == "jdk.GCHeapSummary"]
        self.assertTrue(len(heap_events) >= 1)
        # Nested heapSpace fields should be flattened
        self.assertIn("heapSpace.reservedSize", heap_events[0])
        self.assertEqual(heap_events[0]["heapSpace.reservedSize"], "3.9 GB")

    def test_empty_input(self):
        self.assertEqual(parse_jfr_events(""), [])

    def test_mixed_events(self):
        text = SAMPLE_CPU_LOAD + SAMPLE_GC
        events = parse_jfr_events(text)
        types = {e["_type"] for e in events}
        self.assertIn("jdk.CPULoad", types)
        self.assertIn("jdk.GCPhasePause", types)
        self.assertIn("jdk.GCCPUTime", types)


class TestParseRecordingDuration(unittest.TestCase):
    def test_seconds(self):
        self.assertAlmostEqual(_parse_recording_duration(SAMPLE_SUMMARY), 4.0)

    def test_missing(self):
        self.assertEqual(_parse_recording_duration("no duration here"), 0.0)


# ---------------------------------------------------------------------------
# Sub-profile builder tests
# ---------------------------------------------------------------------------


class TestBuildCpuProfile(unittest.TestCase):
    def test_builds_from_events(self):
        events = parse_jfr_events(SAMPLE_CPU_LOAD)
        from joshpy.jfr._core import _build_cpu_profile

        profile = _build_cpu_profile(events)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.sample_count, 2)
        self.assertAlmostEqual(profile.jvm_user_pct, (45.20 + 60.00) / 2)
        self.assertAlmostEqual(profile.jvm_system_pct, (3.10 + 2.00) / 2)

    def test_returns_none_for_no_events(self):
        from joshpy.jfr._core import _build_cpu_profile

        self.assertIsNone(_build_cpu_profile([]))


class TestBuildGcProfile(unittest.TestCase):
    def test_builds_from_events(self):
        events = parse_jfr_events(SAMPLE_GC)
        from joshpy.jfr._core import _build_gc_profile

        profile = _build_gc_profile(events)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.pause_count, 2)
        self.assertAlmostEqual(profile.total_pause_ms, 6.35 + 9.14)
        self.assertAlmostEqual(profile.max_pause_ms, 9.14)
        self.assertAlmostEqual(profile.gc_user_time_ms, 80.0)

    def test_returns_none_for_no_events(self):
        from joshpy.jfr._core import _build_gc_profile

        self.assertIsNone(_build_gc_profile([]))


class TestBuildMemoryProfile(unittest.TestCase):
    def test_builds_from_events(self):
        events = parse_jfr_events(SAMPLE_HEAP)
        from joshpy.jfr._core import _build_memory_profile

        profile = _build_memory_profile(events)
        self.assertIsNotNone(profile)
        # Peak heap should be 142.6 MB (the Before GC value)
        self.assertAlmostEqual(profile.peak_heap_mb, 142.6, places=0)
        # Reserved should be 3.9 GB
        self.assertAlmostEqual(profile.reserved_heap_mb, 3.9 * 1024, places=0)
        # RSS peak
        self.assertAlmostEqual(profile.peak_rss_mb, 215.6, places=0)
        # Physical memory
        self.assertAlmostEqual(profile.physical_memory_mb, 15.6 * 1024, places=0)

    def test_returns_none_for_no_events(self):
        from joshpy.jfr._core import _build_memory_profile

        self.assertIsNone(_build_memory_profile([]))


class TestBuildIoProfile(unittest.TestCase):
    def test_builds_from_events(self):
        events = parse_jfr_events(SAMPLE_IO)
        from joshpy.jfr._core import _build_io_profile

        profile = _build_io_profile(events)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.read_count, 1)
        self.assertAlmostEqual(profile.read_total_ms, 25.3)
        self.assertEqual(profile.read_total_bytes, 1048576)
        self.assertEqual(profile.write_count, 1)
        self.assertAlmostEqual(profile.write_total_ms, 12.1)
        self.assertEqual(profile.write_total_bytes, 524288)

    def test_returns_none_for_no_events(self):
        from joshpy.jfr._core import _build_io_profile

        self.assertIsNone(_build_io_profile([]))


class TestBuildContentionProfile(unittest.TestCase):
    def test_builds_from_events(self):
        events = parse_jfr_events(SAMPLE_CONTENTION)
        from joshpy.jfr._core import _build_contention_profile

        profile = _build_contention_profile(events)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.wait_count, 1)
        self.assertAlmostEqual(profile.wait_total_ms, 150.0)
        self.assertEqual(profile.enter_count, 1)
        self.assertAlmostEqual(profile.enter_total_ms, 30.0)


# ---------------------------------------------------------------------------
# ResourceProfile tests
# ---------------------------------------------------------------------------


class TestResourceProfileSummary(unittest.TestCase):
    def _make_profile(self, **overrides):
        defaults = dict(
            recording_duration_s=10.0,
            cpu=CpuProfile(jvm_user_pct=50.0, jvm_system_pct=2.0, machine_total_pct=55.0, sample_count=5),
            gc=GcProfile(total_pause_ms=100.0, max_pause_ms=20.0, pause_count=10,
                         gc_user_time_ms=50.0, gc_system_time_ms=10.0, gc_real_time_ms=60.0),
            memory=MemoryProfile(peak_heap_mb=500.0, reserved_heap_mb=4096.0, peak_rss_mb=800.0, physical_memory_mb=16384.0),
            io=None,
            contention=None,
        )
        defaults.update(overrides)
        return ResourceProfile(**defaults)

    def test_summary_has_all_sections(self):
        profile = self._make_profile()
        text = profile.summary()
        self.assertIn("--- CPU ---", text)
        self.assertIn("--- MEMORY ---", text)
        self.assertIn("--- GARBAGE COLLECTION ---", text)
        self.assertIn("--- FILE I/O ---", text)
        self.assertIn("--- THREAD CONTENTION ---", text)
        self.assertIn("--- DIAGNOSIS ---", text)

    def test_summary_detailed_has_gc_cpu(self):
        profile = self._make_profile()
        text = profile.summary(detailed=True)
        self.assertIn("GC CPU", text)

    def test_summary_missing_profiles(self):
        profile = self._make_profile(cpu=None, gc=None, memory=None)
        text = profile.summary()
        self.assertIn("No CPU data recorded", text)
        self.assertIn("No GC data recorded", text)
        self.assertIn("No memory data recorded", text)

    def test_summary_no_contention(self):
        profile = self._make_profile()
        text = profile.summary()
        self.assertIn("No contention events recorded", text)

    def test_summary_with_contention(self):
        profile = self._make_profile(
            contention=ContentionProfile(wait_count=5, wait_total_ms=200.0, enter_count=3, enter_total_ms=50.0),
        )
        text = profile.summary()
        self.assertIn("Monitor waits", text)
        self.assertIn("Monitor enters", text)


# ---------------------------------------------------------------------------
# Diagnosis decision tree tests
# ---------------------------------------------------------------------------


class TestClassifyBottleneck(unittest.TestCase):
    def _make_profile(self, **overrides):
        defaults = dict(
            recording_duration_s=10.0,
            cpu=CpuProfile(jvm_user_pct=30.0, jvm_system_pct=2.0, machine_total_pct=35.0, sample_count=5),
            gc=GcProfile(total_pause_ms=50.0, max_pause_ms=10.0, pause_count=5,
                         gc_user_time_ms=30.0, gc_system_time_ms=5.0, gc_real_time_ms=35.0),
            memory=MemoryProfile(peak_heap_mb=500.0, reserved_heap_mb=4096.0, peak_rss_mb=800.0, physical_memory_mb=16384.0),
            io=None,
            contention=None,
        )
        defaults.update(overrides)
        return ResourceProfile(**defaults)

    def test_gc_bound(self):
        profile = self._make_profile(
            gc=GcProfile(total_pause_ms=2000.0, max_pause_ms=100.0, pause_count=50,
                         gc_user_time_ms=1500.0, gc_system_time_ms=200.0, gc_real_time_ms=1700.0),
        )
        bottleneck, _ = profile.classify()
        self.assertEqual(bottleneck, "GC-bound")

    def test_memory_pressure_heap(self):
        profile = self._make_profile(
            memory=MemoryProfile(peak_heap_mb=3700.0, reserved_heap_mb=4096.0, peak_rss_mb=800.0, physical_memory_mb=16384.0),
        )
        bottleneck, _ = profile.classify()
        self.assertEqual(bottleneck, "memory pressure")

    def test_memory_pressure_rss(self):
        profile = self._make_profile(
            memory=MemoryProfile(peak_heap_mb=500.0, reserved_heap_mb=4096.0, peak_rss_mb=14000.0, physical_memory_mb=16384.0),
        )
        bottleneck, _ = profile.classify()
        self.assertEqual(bottleneck, "memory pressure")

    def test_io_bound(self):
        profile = self._make_profile(
            io=IoProfile(read_count=100, read_total_ms=1500.0, read_total_bytes=100_000_000,
                         write_count=50, write_total_ms=500.0, write_total_bytes=50_000_000),
        )
        bottleneck, _ = profile.classify()
        self.assertEqual(bottleneck, "I/O-bound")

    def test_contention(self):
        profile = self._make_profile(
            contention=ContentionProfile(wait_count=100, wait_total_ms=1500.0, enter_count=50, enter_total_ms=500.0),
        )
        bottleneck, _ = profile.classify()
        self.assertEqual(bottleneck, "thread contention")

    def test_cpu_bound(self):
        profile = self._make_profile(
            cpu=CpuProfile(jvm_user_pct=85.0, jvm_system_pct=5.0, machine_total_pct=92.0, sample_count=10),
        )
        bottleneck, _ = profile.classify()
        self.assertEqual(bottleneck, "compute-bound")

    def test_healthy(self):
        profile = self._make_profile()
        bottleneck, _ = profile.classify()
        self.assertEqual(bottleneck, "no clear bottleneck")


# ---------------------------------------------------------------------------
# build_resource_profile integration test
# ---------------------------------------------------------------------------


class TestBuildResourceProfile(unittest.TestCase):
    def test_full_build(self):
        event_text = SAMPLE_CPU_LOAD + SAMPLE_GC + SAMPLE_HEAP + SAMPLE_IO + SAMPLE_CONTENTION
        profile = build_resource_profile(event_text, SAMPLE_SUMMARY)
        self.assertAlmostEqual(profile.recording_duration_s, 4.0)
        self.assertIsNotNone(profile.cpu)
        self.assertIsNotNone(profile.gc)
        self.assertIsNotNone(profile.memory)
        self.assertIsNotNone(profile.io)
        self.assertIsNotNone(profile.contention)

    def test_empty_recording(self):
        profile = build_resource_profile("", SAMPLE_SUMMARY)
        self.assertAlmostEqual(profile.recording_duration_s, 4.0)
        self.assertIsNone(profile.cpu)
        self.assertIsNone(profile.gc)
        self.assertIsNone(profile.memory)
        self.assertIsNone(profile.io)
        self.assertIsNone(profile.contention)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestDiagnoseJfrCLI(unittest.TestCase):
    """Tests for JoshCLI.diagnose_jfr()."""

    from joshpy.jar import JarMode

    JAR_MODE = JarMode.DEV

    @patch("subprocess.run")
    def test_diagnose_jfr_calls_jfr_print(self, mock_run):
        # First call: jfr print, second call: jfr summary
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=SAMPLE_CPU_LOAD, stderr=""),
            MagicMock(returncode=0, stdout=SAMPLE_SUMMARY, stderr=""),
        ]

        from joshpy.cli import JoshCLI

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        profile = cli.diagnose_jfr(Path("/tmp/recording.jfr"))

        # Verify jfr print was called with --events
        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        self.assertEqual(cmd[1], "print")
        self.assertEqual(cmd[2], "--events")
        self.assertIn("jdk.CPULoad", cmd[3])

        # Verify we got a profile back
        self.assertIsNotNone(profile.cpu)
        self.assertEqual(profile.cpu.sample_count, 2)

    @patch("subprocess.run")
    def test_diagnose_jfr_raises_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="bad file")

        from joshpy.cli import JoshCLI

        cli = JoshCLI(josh_jar=self.JAR_MODE)
        with self.assertRaises(RuntimeError):
            cli.diagnose_jfr(Path("/tmp/bad.jfr"))


if __name__ == "__main__":
    unittest.main()
