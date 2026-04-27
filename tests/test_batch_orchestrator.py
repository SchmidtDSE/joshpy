"""Tests for joshpy.batch_orchestrator.assemble_batch_workdir()."""

import tempfile
import unittest
from pathlib import Path

from joshpy.batch_orchestrator import assemble_batch_workdir
from joshpy.jobs import ExpandedJob


def _make_job(
    tmp: Path,
    *,
    run_hash: str = "abc123def456",
    file_mappings: dict[str, Path] | None = None,
    source_path: Path | None = None,
) -> ExpandedJob:
    """Build a minimal ExpandedJob for orchestrator tests."""
    if source_path is None:
        source_path = tmp / "model.josh"
        source_path.write_text("start simulation Main\nend simulation\n")

    config_path = tmp / "config.jshc"
    config_path.write_text("rendered_jshc_placeholder")

    return ExpandedJob(
        config_content="rendered_jshc_placeholder",
        config_path=config_path,
        # Real ExpandedJobs come out of JobExpander with .jshc-suffixed
        # config_name (jobs.py:1093). Mirror that here so staged-filename
        # assertions reflect the real shape.
        config_name="sweep_config.jshc",
        run_hash=run_hash,
        parameters={"p": 1},
        simulation="Main",
        replicates=1,
        source_path=source_path,
        file_mappings=file_mappings or {},
    )


class TestAssembleBatchWorkdir(unittest.TestCase):
    def test_creates_per_run_hash_subdir(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()
            job = _make_job(tmp)

            result = assemble_batch_workdir(job, workdir)

            self.assertEqual(result, workdir / job.run_hash)
            self.assertTrue(result.is_dir())

    def test_sim_josh_symlink_targets_source_path(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()
            job = _make_job(tmp)

            target = assemble_batch_workdir(job, workdir)

            sim_link = target / "sim.josh"
            self.assertTrue(sim_link.is_symlink())
            self.assertEqual(sim_link.resolve(), job.source_path.resolve())

    def test_config_jshc_written_from_config_content(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()
            job = _make_job(tmp)

            target = assemble_batch_workdir(job, workdir)

            jshc = target / job.config_name
            self.assertTrue(jshc.exists())
            self.assertFalse(jshc.is_symlink())
            self.assertEqual(jshc.read_text(), "rendered_jshc_placeholder")
            # Regression: must NOT silently fall back to a hardcoded "config.jshc"
            # when job.config_name is non-default. Pre-fix bug: the JAR's
            # RunCommand would resolve `config sweep_config.<key>` references
            # against an absent sweep_config.jshc and raise "Config value not
            # found" mid-run.
            if job.config_name != "config.jshc":
                self.assertFalse((target / "config.jshc").exists())

    def test_config_written_under_custom_config_name(self):
        """Regression for batch staging using job.config_name (not hardcoded).

        Models commonly leave JobConfig.config_name unset → JobExpander
        defaults to ``sweep_config.jshc`` and the .josh references
        ``config sweep_config.<key>``. Staging under hardcoded
        ``config.jshc`` causes the pod to fail at "Config value not found"
        because the JAR can't locate the named config file in /tmp/work.
        """
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()
            job = _make_job(tmp)
            # Override to a deliberately non-default name to lock the contract
            # against any future regression that hardcodes a single filename.
            object.__setattr__(job, "config_name", "my_special_config.jshc")

            target = assemble_batch_workdir(job, workdir)

            self.assertTrue((target / "my_special_config.jshc").exists())
            self.assertEqual(
                (target / "my_special_config.jshc").read_text(),
                "rendered_jshc_placeholder",
            )
            self.assertFalse((target / "config.jshc").exists())

    def test_file_mappings_become_symlinks_with_jshd_suffix(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()

            climate = tmp / "climate.jshd"
            climate.write_bytes(b"CLIMATE_DATA")
            cover = tmp / "cover_raw"  # no suffix; should get .jshd appended
            cover.write_bytes(b"COVER_DATA")

            job = _make_job(
                tmp,
                file_mappings={
                    "climate": climate,  # no ext in mapping key, has suffix in dest
                    "cover": cover,
                },
            )

            target = assemble_batch_workdir(job, workdir)

            self.assertTrue((target / "climate.jshd").is_symlink())
            self.assertEqual(
                (target / "climate.jshd").resolve(), climate.resolve(),
            )
            self.assertTrue((target / "cover.jshd").is_symlink())
            self.assertEqual(
                (target / "cover.jshd").resolve(), cover.resolve(),
            )

    def test_file_mapping_key_with_jshd_suffix_is_not_doubled(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()

            data = tmp / "temp.jshd"
            data.write_bytes(b"TEMP")

            job = _make_job(tmp, file_mappings={"temp.jshd": data})
            target = assemble_batch_workdir(job, workdir)

            self.assertTrue((target / "temp.jshd").is_symlink())
            self.assertFalse((target / "temp.jshd.jshd").exists())

    def test_jshdz_source_preserves_suffix_when_key_has_none(self):
        """Compressed .jshdz sources should symlink as .jshdz so the remote
        target's multi-format getter routes reads to the right strategy."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()

            compressed = tmp / "climate.jshdz"
            compressed.write_bytes(b"CLIMATE_XZ")

            job = _make_job(tmp, file_mappings={"climate": compressed})
            target = assemble_batch_workdir(job, workdir)

            self.assertTrue((target / "climate.jshdz").is_symlink())
            self.assertFalse((target / "climate.jshd").exists())
            self.assertFalse((target / "climate.jshdz.jshd").exists())

    def test_jshdz_key_suffix_is_not_doubled(self):
        """Key ending in .jshdz should be used as-is (no .jshdz.jshd)."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()

            compressed = tmp / "cover.jshdz"
            compressed.write_bytes(b"COVER_XZ")

            job = _make_job(tmp, file_mappings={"cover.jshdz": compressed})
            target = assemble_batch_workdir(job, workdir)

            self.assertTrue((target / "cover.jshdz").is_symlink())
            self.assertFalse((target / "cover.jshdz.jshd").exists())

    def test_raises_when_source_path_none(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()

            job = ExpandedJob(
                config_content="x",
                config_path=tmp / "c.jshc",
                config_name="sweep",
                run_hash="abc",
                parameters={},
                simulation="Main",
                replicates=1,
                source_path=None,
            )
            with self.assertRaises(ValueError):
                assemble_batch_workdir(job, workdir)

    def test_idempotent(self):
        """Re-running replaces existing entries without erroring."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            workdir = tmp / "work"
            workdir.mkdir()

            data = tmp / "a.jshd"
            data.write_bytes(b"v1")
            job = _make_job(tmp, file_mappings={"a": data})

            assemble_batch_workdir(job, workdir)
            # Second call must not raise
            target = assemble_batch_workdir(job, workdir)

            self.assertTrue((target / "sim.josh").is_symlink())
            self.assertTrue((target / "a.jshd").is_symlink())
            self.assertEqual(
                (target / job.config_name).read_text(), "rendered_jshc_placeholder",
            )


if __name__ == "__main__":
    unittest.main()
