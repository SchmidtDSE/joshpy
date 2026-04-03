"""Tests for joshpy.inspect viewing and diffing utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


def _make_config():
    from joshpy.jobs import JobConfig
    return JobConfig(simulation="TestSim")


def _make_registry_with_two_runs():
    """Create an in-memory registry with two labeled runs."""
    from joshpy.registry import RunRegistry

    registry = RunRegistry(":memory:")
    session_id = registry.create_session(
        config=_make_config(), experiment_name="test"
    )
    josh_source = "start simulation Main\n  grid.size = 30 m\nend simulation\n"
    registry.register_run(
        session_id=session_id,
        run_hash="hash_baseline",
        josh_path="/path/to/sim.josh",
        config_content="maxGrowth = 50 meters\ninitialTreeCount = 10 count",
        file_mappings=None,
        parameters={"maxGrowth": 50, "initialTreeCount": 10},
        josh_content=josh_source,
    )
    registry.label_run("hash_baseline", "baseline")

    registry.register_run(
        session_id=session_id,
        run_hash="hash_highgrowth",
        josh_path="/path/to/sim.josh",
        config_content="maxGrowth = 100 meters\ninitialTreeCount = 10 count",
        file_mappings=None,
        parameters={"maxGrowth": 100, "initialTreeCount": 10},
        josh_content=josh_source,
    )
    registry.label_run("hash_highgrowth", "high_growth")

    return registry


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestExportPair(unittest.TestCase):
    """Tests for export_pair()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    def test_export_pair_by_label(self):
        from joshpy.inspect import export_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            path1, path2 = export_pair(
                self.registry, "baseline", "high_growth", tmpdir
            )
            self.assertTrue(path1.exists())
            self.assertTrue(path2.exists())
            self.assertIn("maxGrowth = 50", path1.read_text())
            self.assertIn("maxGrowth = 100", path2.read_text())

    def test_export_pair_by_hash(self):
        from joshpy.inspect import export_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            path1, path2 = export_pair(
                self.registry, "hash_baseline", "hash_highgrowth", tmpdir
            )
            self.assertTrue(path1.exists())
            self.assertTrue(path2.exists())

    def test_export_pair_default_tempdir(self):
        from joshpy.inspect import export_pair

        path1, path2 = export_pair(
            self.registry, "baseline", "high_growth"
        )
        self.assertTrue(path1.exists())
        self.assertTrue(path2.exists())
        self.assertTrue(str(path1.parent).startswith(tempfile.gettempdir()))

    def test_export_pair_missing_label(self):
        from joshpy.inspect import export_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(KeyError):
                export_pair(self.registry, "baseline", "nonexistent", tmpdir)

    def test_export_pair_creates_output_dir(self):
        from joshpy.inspect import export_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "dir"
            path1, path2 = export_pair(
                self.registry, "baseline", "high_growth", nested
            )
            self.assertTrue(nested.exists())
            self.assertTrue(path1.exists())


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestViewConfig(unittest.TestCase):
    """Tests for view_config()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    def test_view_by_label(self):
        from joshpy.inspect import view_config

        content = view_config(self.registry, "baseline")
        self.assertIn("maxGrowth = 50", content)

    def test_view_by_hash(self):
        from joshpy.inspect import view_config

        content = view_config(self.registry, "hash_baseline")
        self.assertIn("maxGrowth = 50", content)

    def test_view_missing_raises_key_error(self):
        from joshpy.inspect import view_config

        with self.assertRaises(KeyError):
            view_config(self.registry, "nonexistent")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestOpenView(unittest.TestCase):
    """Tests for open_view()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/code")
    def test_opens_file_in_ide(self, mock_which, mock_run):
        from joshpy.inspect import open_view

        path = open_view(self.registry, "baseline", ide="vscode")
        self.assertTrue(path.exists())
        content = path.read_text()
        self.assertIn("maxGrowth = 50", content)
        self.assertIn("READ-ONLY", content)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "/usr/bin/code")
        self.assertEqual(cmd[1], str(path))

    @patch("joshpy.inspect._core.shutil.which", return_value=None)
    def test_missing_cli_raises_runtime_error(self, mock_which):
        from joshpy.inspect import open_view

        with self.assertRaises(RuntimeError) as ctx:
            open_view(self.registry, "baseline", ide="vscode")
        self.assertIn("not found in PATH", str(ctx.exception))


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestOpenDiff(unittest.TestCase):
    """Tests for open_diff()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    def test_unknown_ide_raises_value_error(self):
        from joshpy.inspect import open_diff

        with self.assertRaises(ValueError) as ctx:
            open_diff(self.registry, "baseline", "high_growth", ide="emacs")
        self.assertIn("Unknown IDE", str(ctx.exception))
        self.assertIn("vscode", str(ctx.exception))

    @patch("joshpy.inspect._core.shutil.which", return_value=None)
    def test_missing_cli_raises_runtime_error(self, mock_which):
        from joshpy.inspect import open_diff

        with self.assertRaises(RuntimeError) as ctx:
            open_diff(self.registry, "baseline", "high_growth", ide="vscode")
        self.assertIn("not found in PATH", str(ctx.exception))

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/code")
    def test_opens_diff_in_ide(self, mock_which, mock_run):
        from joshpy.inspect import open_diff

        path1, path2 = open_diff(
            self.registry, "baseline", "high_growth", ide="vscode"
        )
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "/usr/bin/code")
        self.assertEqual(cmd[1], "--diff")
        self.assertEqual(cmd[2], str(path1))
        self.assertEqual(cmd[3], str(path2))

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/cursor")
    def test_cursor_ide(self, mock_which, mock_run):
        from joshpy.inspect import open_diff

        open_diff(
            self.registry, "baseline", "high_growth", ide="cursor"
        )
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "/usr/bin/cursor")
        self.assertEqual(cmd[1], "--diff")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestCompareConfigsMethod(unittest.TestCase):
    """Tests for RunRegistry.compare_configs()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/code")
    def test_compare_configs_delegates(self, mock_which, mock_run):
        path1, path2 = self.registry.compare_configs(
            "baseline", "high_growth"
        )
        self.assertTrue(path1.exists())
        self.assertTrue(path2.exists())
        mock_run.assert_called_once()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestDiffCLI(unittest.TestCase):
    """Tests for the CLI argument parser and main()."""

    def test_parser_view(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["my.duckdb", "--view", "baseline"])
        self.assertEqual(args.registry, Path("my.duckdb"))
        self.assertEqual(args.view, "baseline")
        self.assertIsNone(args.diff)

    def test_parser_diff(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["my.duckdb", "--diff", "a", "b"])
        self.assertIsNone(args.view)
        self.assertEqual(args.diff, ["a", "b"])
        self.assertEqual(args.ide, "vscode")
        self.assertFalse(args.export_only)

    def test_parser_diff_all_flags(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "my.duckdb", "--diff", "a", "b",
            "--ide", "cursor",
            "--export-only",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.ide, "cursor")
        self.assertTrue(args.export_only)
        self.assertEqual(args.output_dir, Path("/tmp/out"))

    def test_main_missing_registry(self):
        from joshpy.inspect.__main__ import main

        with patch("sys.argv", ["prog", "/nonexistent.duckdb", "--view", "a"]):
            result = main()
        self.assertEqual(result, 1)

    def test_main_view(self):
        from joshpy.inspect.__main__ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            from joshpy.registry import RunRegistry

            file_registry = RunRegistry(str(db_path))
            session_id = file_registry.create_session(
                config=_make_config(), experiment_name="test"
            )
            file_registry.register_run(
                session_id=session_id,
                run_hash="h1",
                josh_path="sim.josh",
                config_content="a = 1 count",
                file_mappings=None,
                parameters={"a": 1},
            )
            file_registry.label_run("h1", "run_a")
            file_registry.close()

            with patch("sys.argv", ["prog", str(db_path), "--view", "run_a"]):
                result = main()

            self.assertEqual(result, 0)

    def test_main_diff_export_only(self):
        from joshpy.inspect.__main__ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            from joshpy.registry import RunRegistry

            file_registry = RunRegistry(str(db_path))
            session_id = file_registry.create_session(
                config=_make_config(), experiment_name="test"
            )
            file_registry.register_run(
                session_id=session_id,
                run_hash="h1",
                josh_path="sim.josh",
                config_content="a = 1 count",
                file_mappings=None,
                parameters={"a": 1},
            )
            file_registry.label_run("h1", "run_a")
            file_registry.register_run(
                session_id=session_id,
                run_hash="h2",
                josh_path="sim.josh",
                config_content="a = 2 count",
                file_mappings=None,
                parameters={"a": 2},
            )
            file_registry.label_run("h2", "run_b")
            file_registry.close()

            out_dir = Path(tmpdir) / "exported"
            with patch("sys.argv", [
                "prog", str(db_path), "--diff", "run_a", "run_b",
                "--export-only", "--output-dir", str(out_dir),
            ]):
                result = main()

            self.assertEqual(result, 0)
            exported = list(out_dir.iterdir())
            self.assertEqual(len(exported), 2)


def _make_registry_with_real_config(tmpdir):
    """Create a registry where the session has a real config_path on disk."""
    from joshpy.jobs import JobConfig
    from joshpy.registry import RunRegistry

    config_file = Path(tmpdir) / "baseline.jshc"
    config_content = "maxGrowth = 50 meters\ninitialTreeCount = 10 count"
    config_file.write_text(config_content)

    config = JobConfig(
        simulation="Main",
        config_path=config_file,
    )

    db_path = Path(tmpdir) / "test.duckdb"
    registry = RunRegistry(str(db_path))
    session_id = registry.create_session(config=config, experiment_name="test")
    registry.register_run(
        session_id=session_id,
        run_hash="hash_real",
        josh_path="sim.josh",
        config_content=config_content,
        file_mappings=None,
        parameters={"maxGrowth": 50, "initialTreeCount": 10},
    )
    registry.label_run("hash_real", "baseline")
    return registry, config_file


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestResolveConfigSource(unittest.TestCase):
    """Tests for RunRegistry.resolve_config_source()."""

    def test_file_exists_and_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, config_file = _make_registry_with_real_config(tmpdir)
            try:
                source = registry.resolve_config_source("hash_real")
                self.assertEqual(source.path, config_file)
                self.assertTrue(source.exists)
                self.assertTrue(source.content_matches)
            finally:
                registry.close()

    def test_file_exists_but_drifted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, config_file = _make_registry_with_real_config(tmpdir)
            try:
                config_file.write_text("maxGrowth = 999 meters")
                source = registry.resolve_config_source("hash_real")
                self.assertEqual(source.path, config_file)
                self.assertTrue(source.exists)
                self.assertFalse(source.content_matches)
            finally:
                registry.close()

    def test_file_deleted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, config_file = _make_registry_with_real_config(tmpdir)
            try:
                config_file.unlink()
                source = registry.resolve_config_source("hash_real")
                self.assertEqual(source.path, config_file)
                self.assertFalse(source.exists)
                self.assertFalse(source.content_matches)
            finally:
                registry.close()

    def test_no_config_path(self):
        """Templated configs have no config_path — should return path=None."""
        registry = _make_registry_with_two_runs()
        try:
            source = registry.resolve_config_source("hash_baseline")
            self.assertIsNone(source.path)
            self.assertFalse(source.exists)
            self.assertFalse(source.content_matches)
        finally:
            registry.close()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestSmartOpenView(unittest.TestCase):
    """Tests for open_view() smart file resolution."""

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/code")
    def test_opens_real_file_when_unchanged(self, mock_which, mock_run):
        from joshpy.inspect import open_view

        with tempfile.TemporaryDirectory() as tmpdir:
            registry, config_file = _make_registry_with_real_config(tmpdir)
            try:
                path = open_view(registry, "baseline", ide="vscode")
                self.assertEqual(path, config_file)
                cmd = mock_run.call_args[0][0]
                self.assertEqual(cmd[1], str(config_file))
            finally:
                registry.close()

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/code")
    def test_opens_temp_with_warning_when_drifted(self, mock_which, mock_run):
        from joshpy.inspect import open_view

        with tempfile.TemporaryDirectory() as tmpdir:
            registry, config_file = _make_registry_with_real_config(tmpdir)
            try:
                config_file.write_text("maxGrowth = 999 meters")
                path = open_view(registry, "baseline", ide="vscode")
                # Should open temp, not the real file
                self.assertNotEqual(path, config_file)
                content = path.read_text()
                self.assertIn("READ-ONLY", content)
                # Header should mention the real file path
                self.assertIn(str(config_file), content)
                self.assertIn("modified since this run was registered", content)
                # Stored content should be the original, not the drifted version
                self.assertIn("maxGrowth = 50", content)
                self.assertNotIn("maxGrowth = 999", content)
            finally:
                registry.close()

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/code")
    def test_opens_temp_when_file_gone(self, mock_which, mock_run):
        from joshpy.inspect import open_view

        with tempfile.TemporaryDirectory() as tmpdir:
            registry, config_file = _make_registry_with_real_config(tmpdir)
            try:
                config_file.unlink()
                path = open_view(registry, "baseline", ide="vscode")
                self.assertNotEqual(path, config_file)
                self.assertIn("READ-ONLY", path.read_text())
            finally:
                registry.close()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestViewJosh(unittest.TestCase):
    """Tests for view_josh()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    def test_view_by_label(self):
        from joshpy.inspect import view_josh

        content = view_josh(self.registry, "baseline")
        self.assertIn("start simulation Main", content)

    def test_view_missing_raises_key_error(self):
        from joshpy.inspect import view_josh

        with self.assertRaises(KeyError):
            view_josh(self.registry, "nonexistent")

    def test_view_no_content_raises_key_error(self):
        """Runs registered without josh_content raise KeyError."""
        from joshpy.inspect import view_josh
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        session_id = registry.create_session(
            config=_make_config(), experiment_name="test"
        )
        registry.register_run(
            session_id=session_id,
            run_hash="hash_no_josh",
            josh_path="/path/to/sim.josh",
            config_content="a = 1 count",
            file_mappings=None,
            parameters={"a": 1},
        )
        registry.label_run("hash_no_josh", "no_josh")
        try:
            with self.assertRaises(KeyError):
                view_josh(registry, "no_josh")
        finally:
            registry.close()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestExportJoshPair(unittest.TestCase):
    """Tests for export_josh_pair()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    def test_export_pair_by_label(self):
        from joshpy.inspect import export_josh_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            path1, path2 = export_josh_pair(
                self.registry, "baseline", "high_growth", tmpdir
            )
            self.assertTrue(path1.exists())
            self.assertTrue(path2.exists())
            self.assertTrue(path1.name.endswith(".josh"))
            self.assertTrue(path2.name.endswith(".josh"))
            self.assertIn("start simulation Main", path1.read_text())


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestOpenJoshDiff(unittest.TestCase):
    """Tests for open_josh_diff()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    @patch("joshpy.inspect._core.subprocess.run")
    @patch("joshpy.inspect._core.shutil.which", return_value="/usr/bin/code")
    def test_opens_diff_in_ide(self, mock_which, mock_run):
        from joshpy.inspect import open_josh_diff

        path1, path2 = open_josh_diff(
            self.registry, "baseline", "high_growth", ide="vscode"
        )
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "/usr/bin/code")
        self.assertEqual(cmd[1], "--diff")
        self.assertTrue(path1.name.endswith(".josh"))
        self.assertTrue(path2.name.endswith(".josh"))


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestJoshCLI(unittest.TestCase):
    """Tests for CLI with --type josh."""

    def test_parser_type_josh(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "my.duckdb", "--view", "baseline", "--type", "josh"
        ])
        self.assertEqual(args.type, "josh")

    def test_main_view_josh(self):
        from joshpy.inspect.__main__ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            from joshpy.registry import RunRegistry

            file_registry = RunRegistry(str(db_path))
            session_id = file_registry.create_session(
                config=_make_config(), experiment_name="test"
            )
            file_registry.register_run(
                session_id=session_id,
                run_hash="h1",
                josh_path="sim.josh",
                config_content="a = 1 count",
                file_mappings=None,
                parameters={"a": 1},
                josh_content="start simulation Main\nend simulation\n",
            )
            file_registry.label_run("h1", "run_a")
            file_registry.close()

            with patch("sys.argv", [
                "prog", str(db_path), "--view", "run_a",
                "--type", "josh", "--export-only",
            ]):
                result = main()

            self.assertEqual(result, 0)

    def test_main_diff_josh_export_only(self):
        from joshpy.inspect.__main__ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            from joshpy.registry import RunRegistry

            file_registry = RunRegistry(str(db_path))
            session_id = file_registry.create_session(
                config=_make_config(), experiment_name="test"
            )
            file_registry.register_run(
                session_id=session_id,
                run_hash="h1",
                josh_path="sim.josh",
                config_content="a = 1 count",
                file_mappings=None,
                parameters={"a": 1},
                josh_content="start simulation Main\nend simulation\n",
            )
            file_registry.label_run("h1", "run_a")
            file_registry.register_run(
                session_id=session_id,
                run_hash="h2",
                josh_path="sim.josh",
                config_content="a = 2 count",
                file_mappings=None,
                parameters={"a": 2},
                josh_content="start simulation Main\n  grid.size = 30 m\nend simulation\n",
            )
            file_registry.label_run("h2", "run_b")
            file_registry.close()

            out_dir = Path(tmpdir) / "exported"
            with patch("sys.argv", [
                "prog", str(db_path), "--diff", "run_a", "run_b",
                "--type", "josh", "--export-only", "--output-dir", str(out_dir),
            ]):
                result = main()

            self.assertEqual(result, 0)
            exported = list(out_dir.iterdir())
            self.assertEqual(len(exported), 2)
            self.assertTrue(all(f.name.endswith(".josh") for f in exported))


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestFormatLabels(unittest.TestCase):
    """Tests for format_labels()."""

    def test_labels_with_data(self):
        from joshpy.inspect import format_labels

        registry = _make_registry_with_two_runs()
        try:
            result = format_labels(registry)
            self.assertIn("LABEL", result)
            self.assertIn("baseline", result)
            self.assertIn("high_growth", result)
            self.assertIn("hash_baseline", result)
            self.assertIn("hash_highgrowth", result)
        finally:
            registry.close()

    def test_labels_empty_registry(self):
        from joshpy.inspect import format_labels
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        try:
            result = format_labels(registry)
            self.assertEqual(result, "No labels found.")
        finally:
            registry.close()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestFormatSessions(unittest.TestCase):
    """Tests for format_sessions()."""

    def test_sessions_with_data(self):
        from joshpy.inspect import format_sessions

        registry = _make_registry_with_two_runs()
        try:
            result = format_sessions(registry)
            self.assertIn("SESSION", result)
            self.assertIn("test", result)  # experiment_name
            self.assertIn("pending", result)  # default status
        finally:
            registry.close()

    def test_sessions_empty_registry(self):
        from joshpy.inspect import format_sessions
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        try:
            result = format_sessions(registry)
            self.assertEqual(result, "No sessions found.")
        finally:
            registry.close()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestFormatRunInfo(unittest.TestCase):
    """Tests for format_run_info()."""

    def setUp(self):
        self.registry = _make_registry_with_two_runs()

    def tearDown(self):
        self.registry.close()

    def test_info_by_label(self):
        from joshpy.inspect import format_run_info

        result = format_run_info(self.registry, "baseline")
        self.assertIn("baseline", result)
        self.assertIn("hash_baseline", result)
        self.assertIn("maxGrowth", result)
        self.assertIn("50", result)
        self.assertIn("/path/to/sim.josh", result)

    def test_info_by_hash(self):
        from joshpy.inspect import format_run_info

        result = format_run_info(self.registry, "hash_baseline")
        self.assertIn("hash_baseline", result)
        self.assertIn("maxGrowth", result)

    def test_info_with_runs(self):
        from joshpy.inspect import format_run_info

        run_id = self.registry.start_run(
            "hash_baseline", replicate=0, output_path="/out/baseline/0"
        )
        self.registry.complete_run(run_id, exit_code=0)
        result = format_run_info(self.registry, "baseline")
        self.assertIn("1 succeeded", result)
        self.assertIn("0 failed", result)
        self.assertIn("REP", result)
        self.assertIn("/out/baseline/0", result)

    def test_info_missing_raises_key_error(self):
        from joshpy.inspect import format_run_info

        with self.assertRaises(KeyError):
            format_run_info(self.registry, "nonexistent")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestFormatSummary(unittest.TestCase):
    """Tests for format_summary()."""

    def test_summary_output(self):
        from joshpy.inspect import format_summary

        registry = _make_registry_with_two_runs()
        try:
            result = format_summary(registry)
            self.assertIn("Registry Data Summary", result)
            self.assertIn("Sessions:", result)
            self.assertIn("Configs:", result)
        finally:
            registry.close()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestQueryCLI(unittest.TestCase):
    """Tests for CLI query modes (--labels, --sessions, --info, --summary)."""

    def test_parser_labels(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["registry.duckdb", "--labels"])
        self.assertTrue(args.labels)

    def test_parser_sessions(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["registry.duckdb", "--sessions"])
        self.assertTrue(args.sessions)

    def test_parser_info(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["registry.duckdb", "--info", "baseline"])
        self.assertEqual(args.info, "baseline")

    def test_parser_summary(self):
        from joshpy.inspect.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["registry.duckdb", "--summary"])
        self.assertTrue(args.summary)

    def test_main_labels(self):
        from joshpy.inspect.__main__ import main
        from joshpy.registry import RunRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            file_registry = RunRegistry(str(db_path))
            session_id = file_registry.create_session(
                config=_make_config(), experiment_name="test"
            )
            file_registry.register_run(
                session_id=session_id,
                run_hash="h1",
                josh_path="sim.josh",
                config_content="a = 1 count",
                file_mappings=None,
                parameters={"a": 1},
            )
            file_registry.label_run("h1", "run_a")
            file_registry.close()

            with patch("sys.argv", ["prog", str(db_path), "--labels"]):
                result = main()
            self.assertEqual(result, 0)

    def test_main_info(self):
        from joshpy.inspect.__main__ import main
        from joshpy.registry import RunRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            file_registry = RunRegistry(str(db_path))
            session_id = file_registry.create_session(
                config=_make_config(), experiment_name="test"
            )
            file_registry.register_run(
                session_id=session_id,
                run_hash="h1",
                josh_path="sim.josh",
                config_content="a = 1 count",
                file_mappings=None,
                parameters={"a": 1},
            )
            file_registry.label_run("h1", "run_a")
            file_registry.close()

            with patch("sys.argv", ["prog", str(db_path), "--info", "run_a"]):
                result = main()
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
