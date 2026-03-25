"""Unit tests for the debug module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from joshpy.debug import (
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


class TestDebugMessage(unittest.TestCase):
    """Tests for DebugMessage frozen dataclass."""

    def test_construction(self):
        """All fields should be accessible after construction."""
        msg = DebugMessage(
            step=5,
            entity_type="organism",
            entity_id="a1b2c3d4",
            x=10.5,
            y=20.3,
            content="Water level is 0.5",
            line_number=42,
        )
        self.assertEqual(msg.step, 5)
        self.assertEqual(msg.entity_type, "organism")
        self.assertEqual(msg.entity_id, "a1b2c3d4")
        self.assertEqual(msg.x, 10.5)
        self.assertEqual(msg.y, 20.3)
        self.assertEqual(msg.content, "Water level is 0.5")
        self.assertEqual(msg.line_number, 42)

    def test_frozen(self):
        """DebugMessage should be immutable."""
        msg = DebugMessage(
            step=0, entity_type="patch", entity_id="abc",
            x=0.0, y=0.0, content="test",
        )
        with self.assertRaises(AttributeError):
            msg.step = 1  # type: ignore[misc]

    def test_default_line_number(self):
        """line_number should default to 0."""
        msg = DebugMessage(
            step=0, entity_type="patch", entity_id="abc",
            x=0.0, y=0.0, content="test",
        )
        self.assertEqual(msg.line_number, 0)


class TestParseDebugLine(unittest.TestCase):
    """Tests for parse_debug_line()."""

    def test_standard_line(self):
        """Should parse a standard debug output line."""
        line = "[Step 5, organism @ a1b2c3d4 (10.5, 20.3)] Water level is 0.5"
        msg = parse_debug_line(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.step, 5)
        self.assertEqual(msg.entity_type, "organism")
        self.assertEqual(msg.entity_id, "a1b2c3d4")
        self.assertEqual(msg.x, 10.5)
        self.assertEqual(msg.y, 20.3)
        self.assertEqual(msg.content, "Water level is 0.5")

    def test_short_hex_id(self):
        """Should parse IDs with fewer than 8 hex characters."""
        line = "[Step 0, organism @ fc983ad (19.5, 8.5)] init adult age 90.4"
        msg = parse_debug_line(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.entity_id, "fc983ad")

    def test_integer_coordinates(self):
        """Should handle coordinates without decimals (e.g., 10.0)."""
        line = "[Step 0, patch @ abcdef12 (10.0, 20.0)] some message"
        msg = parse_debug_line(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.x, 10.0)
        self.assertEqual(msg.y, 20.0)

    def test_content_with_special_chars(self):
        """Content can contain brackets, equals, etc."""
        line = "[Step 1, organism @ abc123 (1.5, 2.5)] r = 0.5 [check] ok"
        msg = parse_debug_line(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.content, "r = 0.5 [check] ok")

    def test_line_number_passthrough(self):
        """line_number should be set from the argument."""
        line = "[Step 0, organism @ abc (1.0, 2.0)] test"
        msg = parse_debug_line(line, line_number=99)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.line_number, 99)

    def test_blank_line_returns_none(self):
        """Blank lines should return None."""
        self.assertIsNone(parse_debug_line(""))
        self.assertIsNone(parse_debug_line("   "))

    def test_malformed_line_returns_none(self):
        """Non-matching lines should return None."""
        self.assertIsNone(parse_debug_line("just some text"))
        self.assertIsNone(parse_debug_line("[Step bad]"))
        self.assertIsNone(parse_debug_line("# comment"))

    def test_leading_trailing_whitespace(self):
        """Should handle leading/trailing whitespace."""
        line = "  [Step 0, patch @ abc (1.0, 2.0)] test  "
        msg = parse_debug_line(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.content, "test")

    def test_step_zero(self):
        """Step 0 should parse correctly."""
        line = "[Step 0, organism @ 3282ea27 (19.5, 8.5)] init adult age 41.2"
        msg = parse_debug_line(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.step, 0)

    def test_large_step_number(self):
        """Large step numbers should parse correctly."""
        line = "[Step 99999, organism @ abc (1.0, 2.0)] test"
        msg = parse_debug_line(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.step, 99999)


class TestDebugMessageStore(unittest.TestCase):
    """Tests for DebugMessageStore."""

    def setUp(self):
        """Create a store with known test data."""
        self.store = DebugMessageStore()
        self.msgs = [
            DebugMessage(0, "organism", "aaa", 1.0, 2.0, "init adult", 1),
            DebugMessage(0, "organism", "bbb", 3.0, 4.0, "init seedling", 2),
            DebugMessage(0, "patch", "ccc", 1.0, 2.0, "fire check", 3),
            DebugMessage(1, "organism", "aaa", 1.0, 2.0, "survives true", 4),
            DebugMessage(1, "organism", "bbb", 3.0, 4.0, "survives false", 5),
            DebugMessage(2, "organism", "aaa", 1.0, 2.0, "flowering", 6),
        ]
        for msg in self.msgs:
            self.store.add(msg)

    def test_len(self):
        """__len__ should return message count."""
        self.assertEqual(len(self.store), 6)

    def test_add(self):
        """add() should append and update indexes."""
        store = DebugMessageStore()
        msg = DebugMessage(0, "organism", "abc", 1.0, 2.0, "test", 1)
        store.add(msg)
        self.assertEqual(len(store), 1)
        self.assertIn(0, store._steps)
        self.assertIn("abc", store._entity_ids)

    def test_filter_by_step(self):
        """filter(step=N) returns only messages at that step."""
        result = self.store.filter(step=0)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(m.step == 0 for m in result))

    def test_filter_by_step_range(self):
        """filter(step_range=(a, b)) returns messages in range."""
        result = self.store.filter(step_range=(0, 1))
        self.assertEqual(len(result), 5)

    def test_filter_by_entity_type(self):
        """filter(entity_type=X) returns only that type."""
        result = self.store.filter(entity_type="patch")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, "ccc")

    def test_filter_by_entity_id(self):
        """filter(entity_id=X) returns messages for that entity."""
        result = self.store.filter(entity_id="aaa")
        self.assertEqual(len(result), 3)

    def test_filter_by_coordinates(self):
        """filter(x=X, y=Y) returns messages at that location."""
        result = self.store.filter(x=3.0, y=4.0)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(m.entity_id == "bbb" for m in result))

    def test_filter_by_content(self):
        """filter(content_contains=X) matches substring."""
        result = self.store.filter(content_contains="survives")
        self.assertEqual(len(result), 2)

    def test_filter_combined(self):
        """Multiple filters are AND-combined."""
        result = self.store.filter(step=1, entity_type="organism")
        self.assertEqual(len(result), 2)

        result = self.store.filter(
            step=1, entity_type="organism", content_contains="true"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, "aaa")

    def test_filter_no_match(self):
        """Empty result for non-matching filters."""
        result = self.store.filter(step=999)
        self.assertEqual(result, [])

    def test_steps(self):
        """steps() returns sorted unique step numbers."""
        self.assertEqual(self.store.steps(), [0, 1, 2])

    def test_entity_types(self):
        """entity_types() returns sorted unique types."""
        self.assertEqual(self.store.entity_types(), ["organism", "patch"])

    def test_entity_ids(self):
        """entity_ids() returns sorted unique IDs."""
        self.assertEqual(self.store.entity_ids(), ["aaa", "bbb", "ccc"])

    def test_locations(self):
        """locations() returns sorted unique (x, y) tuples."""
        self.assertEqual(
            self.store.locations(), [(1.0, 2.0), (3.0, 4.0)]
        )

    def test_trace(self):
        """trace() returns messages for entity sorted by step."""
        result = self.store.trace("aaa")
        self.assertEqual(len(result), 3)
        self.assertEqual([m.step for m in result], [0, 1, 2])

    def test_trace_not_found(self):
        """trace() returns empty list for unknown entity."""
        self.assertEqual(self.store.trace("zzz"), [])

    def test_resolve_entity_id_exact(self):
        """resolve_entity_id returns exact match."""
        self.assertEqual(self.store.resolve_entity_id("aaa"), "aaa")

    def test_resolve_entity_id_prefix(self):
        """resolve_entity_id returns unique prefix match."""
        self.assertEqual(self.store.resolve_entity_id("cc"), "ccc")

    def test_resolve_entity_id_ambiguous(self):
        """resolve_entity_id returns None for ambiguous prefix."""
        # "a" matches "aaa" — only one match, so it resolves
        self.assertEqual(self.store.resolve_entity_id("a"), "aaa")
        # But add another "a"-prefixed entity
        self.store.add(
            DebugMessage(0, "organism", "aab", 0.0, 0.0, "test", 7)
        )
        self.assertIsNone(self.store.resolve_entity_id("a"))

    def test_resolve_entity_id_no_match(self):
        """resolve_entity_id returns None when nothing matches."""
        self.assertIsNone(self.store.resolve_entity_id("zzz"))


class TestDebugSummary(unittest.TestCase):
    """Tests for DebugMessageStore.summary()."""

    def test_summary_values(self):
        """summary() should compute correct statistics."""
        store = DebugMessageStore()
        store.add(DebugMessage(0, "organism", "a1", 1.0, 2.0, "init", 1))
        store.add(DebugMessage(0, "organism", "a2", 3.0, 4.0, "init", 2))
        store.add(DebugMessage(1, "organism", "a1", 1.0, 2.0, "step", 3))
        store.add(DebugMessage(1, "patch", "p1", 1.0, 2.0, "fire", 4))

        s = store.summary()
        self.assertIsInstance(s, DebugSummary)
        self.assertEqual(s.total_messages, 4)
        self.assertEqual(s.parse_errors, 0)
        self.assertEqual(s.step_range, (0, 1))
        self.assertEqual(s.num_steps, 2)
        self.assertEqual(s.entity_types, ("organism", "patch"))
        self.assertEqual(s.num_entities, 3)  # a1, a2, p1
        self.assertEqual(s.num_locations, 2)  # (1,2), (3,4)
        self.assertEqual(s.messages_per_step, {0: 2, 1: 2})
        self.assertEqual(s.entities_per_type, {"organism": 2, "patch": 1})

    def test_summary_parse_errors(self):
        """parse_errors should be reflected in summary."""
        store = DebugMessageStore()
        store.parse_errors = 5
        store.add(DebugMessage(0, "organism", "a1", 0.0, 0.0, "test", 1))
        s = store.summary()
        self.assertEqual(s.parse_errors, 5)

    def test_summary_empty_store(self):
        """summary() on empty store should not error."""
        store = DebugMessageStore()
        s = store.summary()
        self.assertEqual(s.total_messages, 0)
        self.assertEqual(s.step_range, (0, 0))
        self.assertEqual(s.num_steps, 0)


class TestLoadDebugFile(unittest.TestCase):
    """Tests for load_debug_file()."""

    def test_load_valid_file(self):
        """Should parse all valid lines."""
        content = (
            "[Step 0, organism @ abc (1.0, 2.0)] init\n"
            "[Step 1, organism @ abc (1.0, 2.0)] step\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            f.flush()
            store = load_debug_file(Path(f.name))
        self.assertEqual(len(store), 2)
        self.assertEqual(store.parse_errors, 0)
        self.assertEqual(store.messages[0].line_number, 1)
        self.assertEqual(store.messages[1].line_number, 2)

    def test_blank_lines_skipped(self):
        """Blank lines should be skipped without counting as errors."""
        content = (
            "[Step 0, organism @ abc (1.0, 2.0)] init\n"
            "\n"
            "   \n"
            "[Step 1, organism @ abc (1.0, 2.0)] step\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            f.flush()
            store = load_debug_file(Path(f.name))
        self.assertEqual(len(store), 2)
        self.assertEqual(store.parse_errors, 0)

    def test_malformed_lines_counted(self):
        """Malformed lines should be counted as parse errors."""
        content = (
            "[Step 0, organism @ abc (1.0, 2.0)] init\n"
            "this is not a debug line\n"
            "[Step 1, organism @ abc (1.0, 2.0)] step\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            f.flush()
            with self.assertWarns(UserWarning):
                store = load_debug_file(Path(f.name))
        self.assertEqual(len(store), 2)
        self.assertEqual(store.parse_errors, 1)

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_debug_file(Path("/nonexistent/debug.txt"))

    def test_accepts_string_path(self):
        """Should accept str paths in addition to Path objects."""
        content = "[Step 0, organism @ abc (1.0, 2.0)] init\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            f.flush()
            store = load_debug_file(f.name)
        self.assertEqual(len(store), 1)


class TestLoadDebugFromScript(unittest.TestCase):
    """Tests for load_debug_from_script()."""

    def _make_mock_cli(self, debug_files):
        """Create a mock JoshCLI with inspect_exports returning given debug_files."""
        from unittest.mock import MagicMock

        from joshpy.cli import ExportPaths

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={"patch": None, "meta": None, "entity": None},
            debug_files=debug_files,
        )
        return mock_cli

    def test_script_not_found(self):
        """Should raise FileNotFoundError for missing script."""
        with self.assertRaises(FileNotFoundError):
            load_debug_from_script(Path("/nonexistent/sim.josh"))

    def test_loads_discovered_debug_files(self):
        """Should call inspect_exports and load discovered debug files."""
        from joshpy.cli import ExportFileInfo

        # Create a real debug file
        debug_content = (
            "[Step 0, organism @ abc (1.0, 2.0)] init\n"
            "[Step 1, organism @ abc (1.0, 2.0)] step\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as debug_f:
            debug_f.write(debug_content)
            debug_f.flush()
            debug_path = debug_f.name

        # Create a fake .josh file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".josh", delete=False
        ) as josh_f:
            josh_f.write("# fake josh file")
            josh_f.flush()
            josh_path = josh_f.name

        mock_cli = self._make_mock_cli({
            "organism": ExportFileInfo(
                raw=f"file://{debug_path}",
                protocol="file",
                host="",
                path=debug_path,
                file_type="txt",
            ),
            "patch": None,
            "agent": None,
            "disturbance": None,
        })

        store = load_debug_from_script(Path(josh_path), cli=mock_cli)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.messages[0].content, "init")
        mock_cli.inspect_exports.assert_called_once()

    def test_warns_on_missing_debug_file(self):
        """Should warn when a debug file path doesn't exist on disk."""
        from joshpy.cli import ExportFileInfo

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".josh", delete=False
        ) as josh_f:
            josh_f.write("# fake josh file")
            josh_f.flush()
            josh_path = josh_f.name

        mock_cli = self._make_mock_cli({
            "organism": ExportFileInfo(
                raw="file:///nonexistent/debug.txt",
                protocol="file",
                host="",
                path="/nonexistent/debug.txt",
                file_type="txt",
            ),
            "patch": None,
            "agent": None,
            "disturbance": None,
        })

        with self.assertWarns(UserWarning):
            store = load_debug_from_script(Path(josh_path), cli=mock_cli)
        self.assertEqual(len(store), 0)

    def test_passes_simulation_name(self):
        """Should pass simulation name to inspect_exports."""
        from joshpy.cli import InspectExportsConfig

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".josh", delete=False
        ) as josh_f:
            josh_f.write("# fake josh file")
            josh_f.flush()
            josh_path = josh_f.name

        mock_cli = self._make_mock_cli({
            "organism": None, "patch": None,
            "agent": None, "disturbance": None,
        })

        load_debug_from_script(
            Path(josh_path), simulation="CustomSim", cli=mock_cli
        )

        call_args = mock_cli.inspect_exports.call_args
        config = call_args[0][0]
        self.assertIsInstance(config, InspectExportsConfig)
        self.assertEqual(config.simulation, "CustomSim")

    def test_no_debug_files_configured(self):
        """Should return empty store when no debug files are configured."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".josh", delete=False
        ) as josh_f:
            josh_f.write("# fake josh file")
            josh_f.flush()
            josh_path = josh_f.name

        mock_cli = self._make_mock_cli({
            "organism": None, "patch": None,
            "agent": None, "disturbance": None,
        })

        store = load_debug_from_script(Path(josh_path), cli=mock_cli)
        self.assertEqual(len(store), 0)

    def test_resolves_run_hash_in_path(self):
        """Should resolve {run_hash} template variable in debug paths."""
        from joshpy.cli import ExportFileInfo

        # Create debug file inside a "hash-named" directory
        tmp_dir = tempfile.mkdtemp()
        hash_dir = Path(tmp_dir) / "abc123def456"
        hash_dir.mkdir()
        debug_file = hash_dir / "debug_organism.txt"
        debug_file.write_text(
            "[Step 0, organism @ aaa (1.0, 2.0)] init\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".josh", delete=False
        ) as josh_f:
            josh_f.write("# fake josh file")
            josh_f.flush()
            josh_path = josh_f.name

        # Path template with {run_hash}
        template_path = str(tmp_dir) + "/{run_hash}/debug_organism.txt"
        mock_cli = self._make_mock_cli({
            "organism": ExportFileInfo(
                raw=f"file://{template_path}",
                protocol="file",
                host="",
                path=template_path,
                file_type="txt",
            ),
            "patch": None,
            "agent": None,
            "disturbance": None,
        })

        store = load_debug_from_script(
            Path(josh_path), cli=mock_cli, run_hash="abc123def456"
        )
        self.assertEqual(len(store), 1)

    def test_warns_on_unresolved_template_variable(self):
        """Should warn when path has {run_hash} but no run_hash provided."""
        from joshpy.cli import ExportFileInfo

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".josh", delete=False
        ) as josh_f:
            josh_f.write("# fake josh file")
            josh_f.flush()
            josh_path = josh_f.name

        mock_cli = self._make_mock_cli({
            "organism": ExportFileInfo(
                raw="file:///tmp/{run_hash}/debug.txt",
                protocol="file",
                host="",
                path="/tmp/{run_hash}/debug.txt",
                file_type="txt",
            ),
            "patch": None,
            "agent": None,
            "disturbance": None,
        })

        with self.assertWarns(UserWarning):
            store = load_debug_from_script(Path(josh_path), cli=mock_cli)
        self.assertEqual(len(store), 0)


class TestFormatMessage(unittest.TestCase):
    """Tests for format_message()."""

    def setUp(self):
        self.msg = DebugMessage(
            step=5,
            entity_type="organism",
            entity_id="a1b2c3d4",
            x=10.5,
            y=20.3,
            content="survival check r 0.5 survives true",
            line_number=1,
        )

    def test_no_color(self):
        """Without color, should reproduce the original format."""
        result = format_message(self.msg, use_color=False)
        self.assertEqual(
            result,
            "[Step 5, organism @ a1b2c3d4 (10.5, 20.3)]"
            " survival check r 0.5 survives true",
        )

    def test_with_color_contains_ansi(self):
        """With color enabled, output should contain ANSI codes."""
        _Ansi.enable()
        try:
            result = format_message(self.msg, use_color=True)
            self.assertIn("\033[", result)
        finally:
            _Ansi.disable()

    def test_color_true_keyword(self):
        """The word 'true' should be highlighted green."""
        _Ansi.enable()
        try:
            result = format_message(self.msg, use_color=True)
            self.assertIn("\033[32mtrue\033[0m", result)
        finally:
            _Ansi.disable()

    def test_color_false_keyword(self):
        """The word 'false' should be highlighted red."""
        _Ansi.enable()
        try:
            msg = DebugMessage(
                0, "organism", "abc", 1.0, 2.0, "survives false", 1
            )
            result = format_message(msg, use_color=True)
            self.assertIn("\033[31mfalse\033[0m", result)
        finally:
            _Ansi.disable()


class TestColorSupport(unittest.TestCase):
    """Tests for _supports_color() detection."""

    @patch.dict("os.environ", {"NO_COLOR": "1"})
    def test_no_color_env(self):
        """NO_COLOR environment variable should disable color."""
        self.assertFalse(_supports_color())

    @patch("sys.stdout")
    def test_non_tty(self, mock_stdout):
        """Non-TTY stdout should disable color."""
        mock_stdout.isatty.return_value = False
        # Also ensure NO_COLOR is not set
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_supports_color())

    @patch("sys.stdout")
    def test_tty_no_env(self, mock_stdout):
        """TTY stdout without NO_COLOR should enable color."""
        mock_stdout.isatty.return_value = True
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(_supports_color())


class TestByEntityIndex(unittest.TestCase):
    """Tests for the _by_entity incremental index."""

    def test_by_entity_populated_on_add(self):
        """_by_entity should group messages by entity_id during add()."""
        store = DebugMessageStore()
        m1 = DebugMessage(0, "organism", "aaa", 1.0, 2.0, "init", 1)
        m2 = DebugMessage(1, "organism", "aaa", 1.0, 2.0, "step", 2)
        m3 = DebugMessage(0, "organism", "bbb", 3.0, 4.0, "init", 3)
        store.add(m1)
        store.add(m2)
        store.add(m3)
        self.assertEqual(len(store._by_entity["aaa"]), 2)
        self.assertEqual(len(store._by_entity["bbb"]), 1)

    def test_trace_returns_sorted_via_index(self):
        """trace() should return chronologically sorted messages."""
        store = DebugMessageStore()
        # Add out of order
        store.add(DebugMessage(2, "organism", "aaa", 1.0, 2.0, "late", 3))
        store.add(DebugMessage(0, "organism", "aaa", 1.0, 2.0, "early", 1))
        store.add(DebugMessage(1, "organism", "aaa", 1.0, 2.0, "mid", 2))
        trace = store.trace("aaa")
        self.assertEqual([m.step for m in trace], [0, 1, 2])


class TestEventMatch(unittest.TestCase):
    """Tests for EventMatch dataclass."""

    def test_frozen(self):
        """EventMatch should be immutable."""
        match = EventMatch(
            entity_id="aaa",
            location=(1.0, 2.0),
            event_messages=[],
            context=[],
            full_trace=[],
        )
        with self.assertRaises(AttributeError):
            match.entity_id = "bbb"  # type: ignore[misc]

    def test_event_steps(self):
        """event_steps should return sorted unique steps from event_messages."""
        m1 = DebugMessage(5, "organism", "aaa", 1.0, 2.0, "ev", 1)
        m2 = DebugMessage(3, "organism", "aaa", 1.0, 2.0, "ev", 2)
        m3 = DebugMessage(5, "organism", "aaa", 1.0, 2.0, "ev2", 3)
        match = EventMatch(
            entity_id="aaa",
            location=(1.0, 2.0),
            event_messages=[m1, m2, m3],
            context=[],
            full_trace=[],
        )
        self.assertEqual(match.event_steps, [3, 5])


class TestFindEvents(unittest.TestCase):
    """Tests for DebugMessageStore.find_events()."""

    def setUp(self):
        """Create a store with multiple entities and known events."""
        self.store = DebugMessageStore()
        # Entity aaa: init -> survives -> burned -> resprout
        self.store.add(DebugMessage(0, "organism", "aaa", 1.0, 2.0, "init adult", 1))
        self.store.add(DebugMessage(1, "organism", "aaa", 1.0, 2.0, "survives true", 2))
        self.store.add(DebugMessage(2, "organism", "aaa", 1.0, 2.0, "burned true", 3))
        self.store.add(DebugMessage(3, "organism", "aaa", 1.0, 2.0, "resprout check", 4))
        self.store.add(DebugMessage(4, "organism", "aaa", 1.0, 2.0, "survives true", 5))

        # Entity bbb: init -> survives -> survives
        self.store.add(DebugMessage(0, "organism", "bbb", 3.0, 4.0, "init seedling", 6))
        self.store.add(DebugMessage(1, "organism", "bbb", 3.0, 4.0, "survives true", 7))
        self.store.add(DebugMessage(2, "organism", "bbb", 3.0, 4.0, "survives false", 8))

        # Entity ccc: patch messages
        self.store.add(DebugMessage(0, "patch", "ccc", 1.0, 2.0, "fire false", 9))
        self.store.add(DebugMessage(1, "patch", "ccc", 1.0, 2.0, "fire false", 10))
        self.store.add(DebugMessage(2, "patch", "ccc", 1.0, 2.0, "fire true", 11))

    def test_basic_keyword_match(self):
        """Should find entities containing the keyword."""
        matches = self.store.find_events("burned", print=False)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].entity_id, "aaa")

    def test_multiple_entity_matches(self):
        """Should find all entities containing the keyword."""
        matches = self.store.find_events("survives", print=False)
        # aaa and bbb both have "survives"
        ids = {m.entity_id for m in matches}
        self.assertEqual(ids, {"aaa", "bbb"})

    def test_no_match(self):
        """Should return empty list when no entities match."""
        matches = self.store.find_events("nonexistent", print=False)
        self.assertEqual(matches, [])

    def test_event_messages_populated(self):
        """event_messages should contain only the matching messages."""
        matches = self.store.find_events("burned", print=False)
        self.assertEqual(len(matches[0].event_messages), 1)
        self.assertIn("burned", matches[0].event_messages[0].content)

    def test_full_trace_always_complete(self):
        """full_trace should contain all messages for the entity."""
        matches = self.store.find_events("burned", print=False)
        self.assertEqual(len(matches[0].full_trace), 5)  # all aaa messages

    def test_context_equals_full_trace_when_no_window(self):
        """Without before/after, context should equal full_trace."""
        matches = self.store.find_events("burned", print=False)
        self.assertEqual(matches[0].context, matches[0].full_trace)

    def test_context_window_before_after(self):
        """before/after should limit context around event messages."""
        # "burned true" is at index 2 in aaa's trace (step 2)
        matches = self.store.find_events("burned", before=1, after=1, print=False)
        self.assertEqual(len(matches), 1)
        context = matches[0].context
        # Should include step 1 (before), step 2 (event), step 3 (after)
        self.assertEqual(len(context), 3)
        self.assertEqual([m.step for m in context], [1, 2, 3])

    def test_context_window_clamps_to_trace_bounds(self):
        """Context window should not extend beyond trace boundaries."""
        # "init adult" is at index 0 in aaa's trace
        matches = self.store.find_events("init adult", before=5, after=0, print=False)
        context = matches[0].context
        # before=5 but only index 0 exists before, so just the event itself
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0].content, "init adult")

    def test_context_window_merge_overlapping(self):
        """Overlapping context windows from multiple events should merge."""
        # aaa has "survives" at steps 1 and 4 (indices 1 and 4)
        matches = self.store.find_events("survives", before=1, after=1, print=False)
        aaa_match = next(m for m in matches if m.entity_id == "aaa")
        # Index 1: window [0, 2], Index 4: window [3, 4] -> merged [0,1,2,3,4]
        self.assertEqual(len(aaa_match.context), 5)

    def test_entity_type_filter(self):
        """entity_type should limit search to that type."""
        # "fire" appears in patch entity ccc
        matches = self.store.find_events("fire", entity_type="organism", print=False)
        self.assertEqual(matches, [])

        matches = self.store.find_events("fire", entity_type="patch", print=False)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].entity_id, "ccc")

    def test_step_filter(self):
        """step should limit matches to events at that step."""
        # "survives" appears at steps 1, 2, 4 across entities
        matches = self.store.find_events("survives", step=2, print=False)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].entity_id, "bbb")
        self.assertEqual(matches[0].event_messages[0].step, 2)

    def test_step_range_filter(self):
        """step_range should limit matches to events in that range."""
        matches = self.store.find_events("survives", step_range=(0, 1), print=False)
        # aaa has survives at step 1, bbb has survives at step 1
        ids = {m.entity_id for m in matches}
        self.assertEqual(ids, {"aaa", "bbb"})
        # Verify no event_messages outside range
        for match in matches:
            for em in match.event_messages:
                self.assertLessEqual(em.step, 1)

    def test_sorted_by_first_event_step(self):
        """Results should be sorted by the step of the first event."""
        matches = self.store.find_events("survives", print=False)
        steps = [m.event_messages[0].step for m in matches]
        self.assertEqual(steps, sorted(steps))

    def test_location_from_first_message(self):
        """location should come from the entity's first trace message."""
        matches = self.store.find_events("burned", print=False)
        self.assertEqual(matches[0].location, (1.0, 2.0))

    def test_context_before_zero_after_zero(self):
        """before=0 after=0 should return only the event messages."""
        matches = self.store.find_events("burned", before=0, after=0, print=False)
        self.assertEqual(len(matches[0].context), 1)
        self.assertEqual(matches[0].context[0].content, "burned true")


class TestFormatTrace(unittest.TestCase):
    """Tests for format_trace()."""

    def test_empty_messages(self):
        """Should return empty string for empty list."""
        self.assertEqual(format_trace([]), "")

    def test_contains_step_and_content(self):
        """Output should include step numbers and message content."""
        msgs = [
            DebugMessage(0, "organism", "aaa", 1.0, 2.0, "init", 1),
            DebugMessage(1, "organism", "aaa", 1.0, 2.0, "step one", 2),
        ]
        result = format_trace(msgs)
        self.assertIn("Step 0", result)
        self.assertIn("Step 1", result)
        self.assertIn("init", result)
        self.assertIn("step one", result)

    def test_groups_by_step(self):
        """Messages at the same step should share a single separator."""
        msgs = [
            DebugMessage(0, "organism", "aaa", 1.0, 2.0, "first", 1),
            DebugMessage(0, "organism", "aaa", 1.0, 2.0, "second", 2),
            DebugMessage(1, "organism", "aaa", 1.0, 2.0, "third", 3),
        ]
        result = format_trace(msgs)
        # Separator lines start with "---"; one per unique step
        separators = [l for l in result.split("\n") if l.lstrip().startswith("---")]
        self.assertEqual(len(separators), 2)  # step 0 and step 1


class TestPrintTrace(unittest.TestCase):
    """Tests for DebugMessageStore.print_trace()."""

    def test_print_trace_outputs(self):
        """print_trace should print formatted output to stdout."""
        store = DebugMessageStore()
        store.add(DebugMessage(0, "organism", "aaa", 1.0, 2.0, "init", 1))
        store.add(DebugMessage(1, "organism", "aaa", 1.0, 2.0, "step", 2))
        import io
        from unittest.mock import patch as mock_patch
        with mock_patch("sys.stdout", new_callable=io.StringIO) as out:
            store.print_trace("aaa")
        output = out.getvalue()
        self.assertIn("Step 0", output)
        self.assertIn("init", output)

    def test_print_trace_prefix_match(self):
        """print_trace should resolve prefix IDs."""
        store = DebugMessageStore()
        store.add(DebugMessage(0, "organism", "abcdef", 1.0, 2.0, "hi", 1))
        import io
        from unittest.mock import patch as mock_patch
        with mock_patch("sys.stdout", new_callable=io.StringIO) as out:
            store.print_trace("abc")
        self.assertIn("hi", out.getvalue())

    def test_print_trace_not_found(self):
        """print_trace should print error for unknown entity."""
        store = DebugMessageStore()
        store.add(DebugMessage(0, "organism", "aaa", 1.0, 2.0, "hi", 1))
        import io
        from unittest.mock import patch as mock_patch
        with mock_patch("sys.stdout", new_callable=io.StringIO) as out:
            store.print_trace("zzz")
        self.assertIn("not found", out.getvalue())


class TestEventMatchFormat(unittest.TestCase):
    """Tests for EventMatch.format() and print()."""

    def _make_match(self, before=None, after=None):
        """Create a store with gaps and find a match for testing."""
        store = DebugMessageStore()
        store.add(DebugMessage(3, "organism", "bbb", 5.0, 6.0, "before", 1))
        store.add(DebugMessage(4, "organism", "bbb", 5.0, 6.0, "target hit", 2))
        store.add(DebugMessage(5, "organism", "bbb", 5.0, 6.0, "middle", 3))
        store.add(DebugMessage(6, "organism", "bbb", 5.0, 6.0, "filler", 4))
        store.add(DebugMessage(7, "organism", "bbb", 5.0, 6.0, "filler2", 5))
        store.add(DebugMessage(8, "organism", "bbb", 5.0, 6.0, "target again", 6))
        store.add(DebugMessage(9, "organism", "bbb", 5.0, 6.0, "after", 7))
        matches = store.find_events("target", before=before, after=after, print=False)
        return matches[0]

    def test_format_contains_header(self):
        """format() should include entity ID and location in header."""
        match = self._make_match()
        result = match.format()
        self.assertIn("bbb", result)
        self.assertIn("5.0", result)
        self.assertIn("6.0", result)

    def test_format_marks_events(self):
        """format() should mark event messages with > and <- event."""
        match = self._make_match()
        result = match.format()
        self.assertIn("> ", result)
        self.assertIn("<- event", result)

    def test_format_shows_gap(self):
        """format() should show omitted message count for gaps."""
        match = self._make_match(before=0, after=0)
        result = match.format()
        self.assertIn("omitted", result)

    def test_format_no_gap_full_context(self):
        """format() without before/after should show no gaps."""
        match = self._make_match()
        result = match.format()
        self.assertNotIn("omitted", result)

    def test_print_outputs_format(self):
        """print() should output the same as format()."""
        match = self._make_match()
        import io
        from unittest.mock import patch as mock_patch
        with mock_patch("sys.stdout", new_callable=io.StringIO) as out:
            match.print()
        self.assertEqual(out.getvalue().strip(), match.format().strip())


if __name__ == "__main__":
    unittest.main()
