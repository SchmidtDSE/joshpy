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
        config_name="sweep",
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

            jshc = target / "config.jshc"
            self.assertTrue(jshc.exists())
            self.assertFalse(jshc.is_symlink())
            self.assertEqual(jshc.read_text(), "rendered_jshc_placeholder")

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
                (target / "config.jshc").read_text(), "rendered_jshc_placeholder",
            )


if __name__ == "__main__":
    unittest.main()
