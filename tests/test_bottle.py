"""Tests for joshpy.bottle module."""

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


def _make_job(
    tmpdir: str,
    run_hash: str = "abc123def456",
    simulation: str = "Main",
    replicates: int = 3,
    with_data: bool = True,
) -> "ExpandedJob":
    """Create an ExpandedJob with real files on disk."""
    from joshpy.jobs import ExpandedJob

    source = Path(tmpdir) / "model.josh"
    source.write_text("start simulation Main\nend simulation\n")

    config_path = Path(tmpdir) / "sweep_config.jshc"
    config_path.write_text("maxGrowth = 50 meters\n")

    file_mappings: dict[str, Path] = {}
    if with_data:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        soil = data_dir / "soil_quality.jshd"
        soil.write_bytes(b"fake jshd data")
        file_mappings["soil_quality"] = soil

    return ExpandedJob(
        config_content="maxGrowth = 50 meters\n",
        config_path=config_path,
        config_name="sweep_config",
        run_hash=run_hash,
        parameters={"maxGrowth": 50},
        simulation=simulation,
        replicates=replicates,
        source_path=source,
        file_mappings=file_mappings,
        custom_tags={"run_hash": run_hash, "maxGrowth": "50"},
    )


def _make_cli_result(success: bool = True):
    """Create a mock CLIResult."""
    from joshpy.cli import CLIResult

    return CLIResult(
        exit_code=0 if success else 1,
        stdout="output" if success else "",
        stderr="" if success else "java.lang.NullPointerException",
        command=["java", "-jar", "test.jar", "run"],
    )


def _make_mock_cli():
    """Create a mock JoshCLI with _resolved_jar."""
    cli = MagicMock()
    cli._resolved_jar = Path("/fake/joshsim-fat.jar")
    cli.java_path = "java"
    return cli


class TestCreateBottle(unittest.TestCase):
    """Tests for create_bottle()."""

    @patch("joshpy.jar.get_jar_version", return_value="0.5.0-dev")
    @patch("joshpy.jar.get_jar_hash", return_value="sha256abc123")
    def test_archive_structure(self, mock_hash, mock_ver):
        from joshpy.bottle import create_bottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir)
            result = _make_cli_result()
            cli = _make_mock_cli()

            out_dir = Path(tmpdir) / "bottles"
            archive = create_bottle(job, result, cli, output_dir=out_dir)

            self.assertTrue(archive.exists())
            self.assertTrue(archive.name.endswith(".tar.gz"))
            self.assertIn("bottle_abc123def456", archive.name)

            # Extract and check contents
            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()

            prefix = "bottle_abc123def456"
            self.assertIn(f"{prefix}/simulation.josh", names)
            self.assertIn(f"{prefix}/sweep_config.jshc", names)
            self.assertIn(f"{prefix}/run.sh", names)
            self.assertIn(f"{prefix}/manifest.json", names)
            self.assertIn(f"{prefix}/data/soil_quality.jshd", names)

    @patch("joshpy.jar.get_jar_version", return_value="0.5.0-dev")
    @patch("joshpy.jar.get_jar_hash", return_value="sha256abc123")
    def test_run_sh_content(self, mock_hash, mock_ver):
        from joshpy.bottle import create_bottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir)
            cli = _make_mock_cli()

            out_dir = Path(tmpdir) / "bottles"
            archive = create_bottle(job, cli=cli, output_dir=out_dir)

            extract_dir = Path(tmpdir) / "extracted"
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(extract_dir)

            run_sh = (extract_dir / "bottle_abc123def456" / "run.sh").read_text()
            self.assertTrue(run_sh.startswith("#!/bin/bash"))
            self.assertIn("java -jar", run_sh)
            self.assertIn("simulation.josh", run_sh)
            self.assertIn("Main", run_sh)
            self.assertIn("--data sweep_config.jshc=sweep_config.jshc", run_sh)
            self.assertIn("--data soil_quality=data/soil_quality.jshd", run_sh)
            self.assertIn("--replicates 3", run_sh)
            # Relative paths, not absolute
            self.assertNotIn(tmpdir, run_sh)

    @patch("joshpy.jar.get_jar_version", return_value="0.5.0-dev")
    @patch("joshpy.jar.get_jar_hash", return_value="sha256abc123")
    def test_manifest_fields(self, mock_hash, mock_ver):
        from joshpy.bottle import create_bottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir)
            result = _make_cli_result(success=False)
            cli = _make_mock_cli()

            out_dir = Path(tmpdir) / "bottles"
            archive = create_bottle(job, result, cli, output_dir=out_dir)

            extract_dir = Path(tmpdir) / "extracted"
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(extract_dir)

            manifest = json.loads(
                (extract_dir / "bottle_abc123def456" / "manifest.json").read_text()
            )
            self.assertEqual(manifest["run_hash"], "abc123def456")
            self.assertEqual(manifest["simulation"], "Main")
            self.assertEqual(manifest["replicates"], 3)
            self.assertEqual(manifest["parameters"], {"maxGrowth": 50})
            self.assertEqual(manifest["exit_code"], 1)
            self.assertIn("NullPointerException", manifest["stderr"])
            self.assertEqual(manifest["jar_version"], "0.5.0-dev")
            self.assertEqual(manifest["jar_sha256"], "sha256abc123")
            self.assertIn("bottled_at", manifest)
            self.assertIn("python_version", manifest)
            self.assertIn("original_josh_path", manifest)
            self.assertIn("original_data_paths", manifest)

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_no_data_files(self, mock_hash, mock_ver):
        from joshpy.bottle import create_bottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir, with_data=False)
            out_dir = Path(tmpdir) / "bottles"
            archive = create_bottle(job, output_dir=out_dir)

            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()

            prefix = "bottle_abc123def456"
            self.assertIn(f"{prefix}/simulation.josh", names)
            self.assertIn(f"{prefix}/sweep_config.jshc", names)
            # No data directory
            self.assertFalse(any("data/" in n for n in names))

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_missing_data_raises(self, mock_hash, mock_ver):
        from joshpy.bottle import create_bottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir, with_data=True)
            # Delete the data file
            for p in job.file_mappings.values():
                p.unlink()

            out_dir = Path(tmpdir) / "bottles"
            with self.assertRaises(FileNotFoundError):
                create_bottle(job, output_dir=out_dir)

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_omit_jshd_skips_data(self, mock_hash, mock_ver):
        from joshpy.bottle import create_bottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir, with_data=True)
            out_dir = Path(tmpdir) / "bottles"
            archive = create_bottle(job, output_dir=out_dir, omit_jshd=True)

            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()

            # No data files copied
            self.assertFalse(any("data/" in n for n in names))

            # run.sh still lists --data flags
            extract_dir = Path(tmpdir) / "extracted"
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(extract_dir)

            run_sh = (extract_dir / "bottle_abc123def456" / "run.sh").read_text()
            self.assertIn("--data soil_quality=data/soil_quality.jshd", run_sh)

            # Manifest records omit_jshd
            manifest = json.loads(
                (extract_dir / "bottle_abc123def456" / "manifest.json").read_text()
            )
            self.assertTrue(manifest["omit_jshd"])
            self.assertIn("soil_quality", manifest["original_data_paths"])

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_omit_jshd_missing_ok(self, mock_hash, mock_ver):
        from joshpy.bottle import create_bottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir, with_data=True)
            # Delete the data file
            for p in job.file_mappings.values():
                p.unlink()

            out_dir = Path(tmpdir) / "bottles"
            # Should NOT raise with omit_jshd=True
            archive = create_bottle(job, output_dir=out_dir, omit_jshd=True)
            self.assertTrue(archive.exists())


class TestShouldBottle(unittest.TestCase):
    """Tests for _should_bottle() logic."""

    def test_first_failure_triggers_on_first(self):
        from joshpy.bottle import _should_bottle

        self.assertTrue(_should_bottle("first_failure", False, False, False))

    def test_first_failure_skips_after_first(self):
        from joshpy.bottle import _should_bottle

        self.assertFalse(_should_bottle("first_failure", False, True, False))

    def test_first_failure_ignores_success(self):
        from joshpy.bottle import _should_bottle

        self.assertFalse(_should_bottle("first_failure", True, False, False))

    def test_all_failures(self):
        from joshpy.bottle import _should_bottle

        self.assertTrue(_should_bottle("all_failures", False, True, False))
        self.assertFalse(_should_bottle("all_failures", True, False, False))

    def test_first_success(self):
        from joshpy.bottle import _should_bottle

        self.assertTrue(_should_bottle("first_success", True, False, False))
        self.assertFalse(_should_bottle("first_success", True, False, True))
        self.assertFalse(_should_bottle("first_success", False, False, False))

    def test_all(self):
        from joshpy.bottle import _should_bottle

        self.assertTrue(_should_bottle("all", True, True, True))
        self.assertTrue(_should_bottle("all", False, True, True))


class TestBottleInRunSweep(unittest.TestCase):
    """Tests for bottle integration in run_sweep()."""

    def test_invalid_bottle_mode_raises(self):
        from joshpy.jobs import JobSet, run_sweep

        cli = MagicMock()
        job_set = JobSet(jobs=[])

        with self.assertRaises(ValueError) as ctx:
            run_sweep(cli, job_set, bottle="invalid_mode")
        self.assertIn("Invalid bottle mode", str(ctx.exception))

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_bottle_first_failure_creates_archive(self, mock_hash, mock_ver):
        from joshpy.jobs import run_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir)
            from joshpy.jobs import JobSet

            job_set = JobSet(jobs=[job])

            cli = MagicMock()
            cli._resolved_jar = Path("/fake/jar.jar")
            cli.java_path = "java"
            cli.run.return_value = _make_cli_result(success=False)

            bottle_dir = Path(tmpdir) / "bottles"
            result = run_sweep(
                cli,
                job_set,
                stop_on_failure=False,
                quiet=True,
                bottle="first_failure",
                bottle_dir=bottle_dir,
            )

            self.assertEqual(result.failed, 1)
            archives = list(bottle_dir.glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            self.assertIn("abc123def456", archives[0].name)

    @patch("joshpy.bottle.create_bottle", side_effect=RuntimeError("disk full"))
    def test_bottle_failure_does_not_abort_sweep(self, mock_bottle):
        from joshpy.jobs import run_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir)
            from joshpy.jobs import JobSet

            job_set = JobSet(jobs=[job])

            cli = MagicMock()
            cli.run.return_value = _make_cli_result(success=False)

            # Should complete without raising
            result = run_sweep(
                cli,
                job_set,
                stop_on_failure=False,
                quiet=True,
                bottle="first_failure",
            )

            self.assertEqual(result.failed, 1)
            mock_bottle.assert_called_once()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestRegistryBottle(unittest.TestCase):
    """Tests for registry.bottle() after-the-fact bottling."""

    def _make_registry_with_run(self, tmpdir):
        from joshpy.jobs import JobConfig
        from joshpy.registry import RunRegistry

        # Create a data file on disk
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        soil = data_dir / "soil.jshd"
        soil.write_bytes(b"fake jshd data")

        registry = RunRegistry(":memory:")
        session_id = registry.create_session(
            config=JobConfig(simulation="Main"),
            experiment_name="test",
        )
        registry.register_run(
            session_id=session_id,
            run_hash="hash_test",
            josh_path="/original/model.josh",
            josh_content="start simulation Main\nend simulation\n",
            config_content="maxGrowth = 50 meters\n",
            file_mappings={
                "soil": {"path": str(soil), "hash": "abc123"},
            },
            parameters={"maxGrowth": 50},
        )
        registry.label_run("hash_test", "baseline")
        return registry

    @patch("joshpy.jar.get_jar_version", return_value="0.5.0")
    @patch("joshpy.jar.get_jar_hash", return_value="sha256test")
    def test_bottle_by_label(self, mock_hash, mock_ver):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = self._make_registry_with_run(tmpdir)
            try:
                out_dir = Path(tmpdir) / "bottles"
                cli = _make_mock_cli()
                archive = registry.bottle("baseline", output_dir=out_dir, cli=cli)

                self.assertTrue(archive.exists())
                with tarfile.open(archive, "r:gz") as tar:
                    names = tar.getnames()

                prefix = "bottle_hash_test"
                self.assertIn(f"{prefix}/simulation.josh", names)
                self.assertIn(f"{prefix}/sweep_config.jshc", names)
                self.assertIn(f"{prefix}/run.sh", names)
                self.assertIn(f"{prefix}/manifest.json", names)
                self.assertIn(f"{prefix}/data/soil.jshd", names)

                # Verify josh content was written from registry
                extract_dir = Path(tmpdir) / "extracted"
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(extract_dir)

                josh = (extract_dir / prefix / "simulation.josh").read_text()
                self.assertIn("start simulation Main", josh)
            finally:
                registry.close()

    def test_bottle_missing_josh_content_raises(self):
        from joshpy.jobs import JobConfig
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        session_id = registry.create_session(
            config=JobConfig(simulation="Main"),
            experiment_name="test",
        )
        registry.register_run(
            session_id=session_id,
            run_hash="hash_no_josh",
            josh_path="/model.josh",
            config_content="a = 1 count",
            file_mappings=None,
            parameters={"a": 1},
            # No josh_content
        )
        try:
            with self.assertRaises(ValueError):
                registry.bottle("hash_no_josh")
        finally:
            registry.close()

    def test_bottle_missing_run_raises(self):
        from joshpy.jobs import JobConfig
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        registry.create_session(
            config=JobConfig(simulation="Main"),
            experiment_name="test",
        )
        try:
            with self.assertRaises(KeyError):
                registry.bottle("nonexistent")
        finally:
            registry.close()


class TestUnbottle(unittest.TestCase):
    """Tests for unbottle()."""

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_unbottle_basic(self, mock_hash, mock_ver):
        """Create a bottle with data, unbottle it, verify JobConfig."""
        from joshpy.bottle import create_bottle, unbottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir)
            archive = create_bottle(job, output_dir=Path(tmpdir) / "bottles")

            extract_dir = Path(tmpdir) / "unpacked"
            config = unbottle(archive, extract_dir=extract_dir)

            self.assertEqual(config.simulation, "Main")
            self.assertTrue(config.source_path.exists())
            self.assertIn("start simulation", config.source_path.read_text())
            self.assertTrue(config.config_path.exists())
            self.assertIn("maxGrowth", config.config_path.read_text())
            # Data file was included in bottle
            self.assertIn("soil_quality", config.file_mappings)
            self.assertTrue(config.file_mappings["soil_quality"].exists())

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_unbottle_with_data_dir(self, mock_hash, mock_ver):
        """Create bottle with omit_jshd, unbottle with data_dir remapping."""
        from joshpy.bottle import create_bottle, unbottle

        with tempfile.TemporaryDirectory() as tmpdir:
            job = _make_job(tmpdir)
            archive = create_bottle(
                job, output_dir=Path(tmpdir) / "bottles", omit_jshd=True
            )

            # Create a local data directory with the same files
            local_data = Path(tmpdir) / "local_data"
            local_data.mkdir()
            (local_data / "soil_quality.jshd").write_bytes(b"local data")

            extract_dir = Path(tmpdir) / "unpacked"
            config = unbottle(archive, extract_dir=extract_dir, data_dir=local_data)

            self.assertIn("soil_quality", config.file_mappings)
            self.assertEqual(
                config.file_mappings["soil_quality"],
                local_data / "soil_quality.jshd",
            )

    @patch("joshpy.jar.get_jar_version", return_value=None)
    @patch("joshpy.jar.get_jar_hash", return_value=None)
    def test_unbottle_preserves_subdirectory_structure(self, mock_hash, mock_ver):
        """data_dir remapping preserves subdirectory structure."""
        from joshpy.bottle import create_bottle, unbottle
        from joshpy.jobs import ExpandedJob

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create job with files in subdirectories
            source = Path(tmpdir) / "model.josh"
            source.write_text("start simulation Main\nend simulation\n")
            config_path = Path(tmpdir) / "config.jshc"
            config_path.write_text("a = 1 count\n")

            # Simulate nested data structure
            orig_data = Path(tmpdir) / "orig" / "grids" / "dev"
            orig_data.mkdir(parents=True)
            (orig_data / "cover.jshd").write_bytes(b"cover")
            monthly = orig_data / "monthly"
            monthly.mkdir()
            (monthly / "temp_jan.jshd").write_bytes(b"temp")

            job = ExpandedJob(
                config_content="a = 1 count\n",
                config_path=config_path,
                config_name="config",
                run_hash="nested_test",
                parameters={"a": 1},
                simulation="Main",
                replicates=1,
                source_path=source,
                file_mappings={
                    "cover": orig_data / "cover.jshd",
                    "futureTempJan": monthly / "temp_jan.jshd",
                },
            )

            archive = create_bottle(
                job, output_dir=Path(tmpdir) / "bottles", omit_jshd=True
            )

            # Create local data with same structure
            local = Path(tmpdir) / "local_grids"
            local.mkdir()
            (local / "cover.jshd").write_bytes(b"local cover")
            (local / "monthly").mkdir()
            (local / "monthly" / "temp_jan.jshd").write_bytes(b"local temp")

            config = unbottle(archive, data_dir=local)

            self.assertEqual(
                config.file_mappings["cover"], local / "cover.jshd"
            )
            self.assertEqual(
                config.file_mappings["futureTempJan"],
                local / "monthly" / "temp_jan.jshd",
            )

    def test_unbottle_missing_archive_raises(self):
        from joshpy.bottle import unbottle

        with self.assertRaises(FileNotFoundError):
            unbottle(Path("/nonexistent/bottle.tar.gz"))


if __name__ == "__main__":
    unittest.main()
