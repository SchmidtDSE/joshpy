"""Unit tests for the sweep module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

from joshpy.sweep import recover_sweep_results, _get_export_path


class TestGetExportPath(unittest.TestCase):
    """Tests for _get_export_path helper function."""

    def test_get_patch_path(self):
        """Should return patch path via get_patch_path method."""
        mock_export_paths = MagicMock()
        mock_export_paths.get_patch_path.return_value = "/tmp/output_{replicate}.csv"

        result = _get_export_path(mock_export_paths, "patch")
        self.assertEqual(result, "/tmp/output_{replicate}.csv")

    def test_get_meta_path(self):
        """Should return meta path from export_files dict."""
        mock_info = MagicMock()
        mock_info.path = "/tmp/meta_{replicate}.csv"
        mock_export_paths = MagicMock()
        mock_export_paths.export_files = {"meta": mock_info}

        result = _get_export_path(mock_export_paths, "meta")
        self.assertEqual(result, "/tmp/meta_{replicate}.csv")

    def test_get_entity_path(self):
        """Should return entity path from export_files dict."""
        mock_info = MagicMock()
        mock_info.path = "/tmp/entity_{replicate}.csv"
        mock_export_paths = MagicMock()
        mock_export_paths.export_files = {"entity": mock_info}

        result = _get_export_path(mock_export_paths, "entity")
        self.assertEqual(result, "/tmp/entity_{replicate}.csv")

    def test_returns_none_when_not_configured(self):
        """Should return None when export type is not configured."""
        mock_export_paths = MagicMock()
        mock_export_paths.export_files = {"meta": None}

        result = _get_export_path(mock_export_paths, "meta")
        self.assertIsNone(result)

    def test_raises_for_unknown_export_type(self):
        """Should raise ValueError for unknown export type."""
        mock_export_paths = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            _get_export_path(mock_export_paths, "unknown")

        self.assertIn("unknown", str(ctx.exception))


class TestRecoverSweepResults(unittest.TestCase):
    """Tests for recover_sweep_results function."""

    def test_raises_if_no_source_path(self):
        """Should raise ValueError if jobs have no source_path."""
        mock_cli = MagicMock()
        mock_registry = MagicMock()

        # Create mock job with no source_path
        mock_job = MagicMock()
        mock_job.source_path = None

        mock_job_set = MagicMock()
        mock_job_set.__iter__ = MagicMock(return_value=iter([mock_job]))
        mock_job_set.jobs = [mock_job]

        with self.assertRaises(ValueError) as ctx:
            recover_sweep_results(
                cli=mock_cli,
                job_set=mock_job_set,
                registry=mock_registry,
            )

        self.assertIn("source_path", str(ctx.exception))

    def test_raises_if_different_source_paths(self):
        """Should raise ValueError if jobs have different source_paths."""
        mock_cli = MagicMock()
        mock_registry = MagicMock()

        # Create mock jobs with different source_paths
        mock_job1 = MagicMock()
        mock_job1.source_path = Path("/path/to/sim1.josh")

        mock_job2 = MagicMock()
        mock_job2.source_path = Path("/path/to/sim2.josh")

        mock_job_set = MagicMock()
        mock_job_set.__iter__ = MagicMock(return_value=iter([mock_job1, mock_job2]))
        mock_job_set.jobs = [mock_job1, mock_job2]

        with self.assertRaises(ValueError) as ctx:
            recover_sweep_results(
                cli=mock_cli,
                job_set=mock_job_set,
                registry=mock_registry,
            )

        self.assertIn("same source_path", str(ctx.exception))

    def test_raises_if_no_export_configured(self):
        """Should raise RuntimeError if no export path is configured."""
        mock_cli = MagicMock()
        mock_registry = MagicMock()

        # Create mock job
        mock_job = MagicMock()
        mock_job.source_path = Path("/path/to/sim.josh")
        mock_job.simulation = "Main"

        mock_job_set = MagicMock()
        mock_job_set.__iter__ = MagicMock(return_value=iter([mock_job]))
        mock_job_set.jobs = [mock_job]

        # Mock inspect_exports to return no patch path
        mock_export_paths = MagicMock()
        mock_export_paths.get_patch_path.return_value = None
        mock_cli.inspect_exports.return_value = mock_export_paths

        with self.assertRaises(RuntimeError) as ctx:
            recover_sweep_results(
                cli=mock_cli,
                job_set=mock_job_set,
                registry=mock_registry,
            )

        self.assertIn("patch export configured", str(ctx.exception))

    @patch("joshpy.sweep.CellDataLoader")
    def test_skips_jobs_without_runs(self, mock_loader_class):
        """Should skip jobs that don't have recorded runs."""
        mock_cli = MagicMock()
        mock_registry = MagicMock()

        # Create mock job
        mock_job = MagicMock()
        mock_job.source_path = Path("/path/to/sim.josh")
        mock_job.simulation = "Main"
        mock_job.run_hash = "abc123def456"
        mock_job.replicates = 1
        mock_job.parameters = {}
        mock_job.custom_tags = {}

        mock_job_set = MagicMock()
        mock_job_set.__iter__ = MagicMock(return_value=iter([mock_job]))
        mock_job_set.jobs = [mock_job]

        # Mock inspect_exports
        mock_export_paths = MagicMock()
        mock_export_paths.get_patch_path.return_value = "/tmp/output_{replicate}.csv"
        mock_cli.inspect_exports.return_value = mock_export_paths

        # Mock registry to return no runs
        mock_registry.get_runs_for_hash.return_value = []

        rows = recover_sweep_results(
            cli=mock_cli,
            job_set=mock_job_set,
            registry=mock_registry,
            quiet=True,
        )

        self.assertEqual(rows, 0)

    @patch("joshpy.sweep.CellDataLoader")
    def test_loads_csv_files(self, mock_loader_class):
        """Should load CSV files that exist."""
        mock_cli = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock CSV file
            csv_path = Path(tmpdir) / "output_0.csv"
            csv_path.write_text("step,replicate,value\n0,0,42\n")

            # Create mock job
            mock_job = MagicMock()
            mock_job.source_path = Path("/path/to/sim.josh")
            mock_job.simulation = "Main"
            mock_job.run_hash = "abc123def456"
            mock_job.replicates = 1
            mock_job.parameters = {}
            mock_job.custom_tags = {}

            mock_job_set = MagicMock()
            # __iter__ needs to return a fresh iterator each time it's called
            mock_job_set.__iter__ = lambda self: iter([mock_job])
            mock_job_set.jobs = [mock_job]

            # Mock inspect_exports to return path template pointing to our temp file
            mock_export_paths = MagicMock()
            mock_export_paths.get_patch_path.return_value = str(tmpdir) + "/output_{replicate}.csv"
            mock_export_paths.resolve_path.return_value = csv_path
            mock_cli.inspect_exports.return_value = mock_export_paths

            # Create a real registry
            from joshpy.registry import RunRegistry
            registry = RunRegistry(":memory:")
            session_id = registry.create_session(experiment_name="test")

            # Register the config and run
            registry.register_run(
                session_id=session_id,
                run_hash="abc123def456",
                josh_path="/path/to/sim.josh",
                config_content="test config",
                file_mappings=None,
                parameters={},
            )
            run_id = registry.start_run(run_hash="abc123def456")

            # Mock loader
            mock_loader = MagicMock()
            mock_loader.load_csv.return_value = 10
            mock_loader_class.return_value = mock_loader

            rows = recover_sweep_results(
                cli=mock_cli,
                job_set=mock_job_set,
                registry=registry,
                quiet=True,
            )

            self.assertEqual(rows, 10)
            mock_loader.load_csv.assert_called_once()
            
            registry.close()

    @patch("joshpy.sweep.CellDataLoader")
    def test_loads_multiple_replicates(self, mock_loader_class):
        """Should load CSV files for all replicates."""
        mock_cli = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock CSV files for 3 replicates
            for i in range(3):
                csv_path = Path(tmpdir) / f"output_{i}.csv"
                csv_path.write_text(f"step,replicate,value\n0,{i},42\n")

            # Create mock job with 3 replicates
            mock_job = MagicMock()
            mock_job.source_path = Path("/path/to/sim.josh")
            mock_job.simulation = "Main"
            mock_job.run_hash = "abc123def456"
            mock_job.replicates = 3
            mock_job.parameters = {"param1": 10}
            mock_job.custom_tags = {"run_hash": "abc123def456"}

            mock_job_set = MagicMock()
            # __iter__ needs to return a fresh iterator each time it's called
            mock_job_set.__iter__ = lambda self: iter([mock_job])
            mock_job_set.jobs = [mock_job]

            # Mock inspect_exports
            mock_export_paths = MagicMock()
            mock_export_paths.get_patch_path.return_value = str(tmpdir) + "/output_{replicate}.csv"

            # resolve_path should return correct path for each replicate
            def resolve_path_side_effect(template, **kwargs):
                return Path(tmpdir) / f"output_{kwargs['replicate']}.csv"

            mock_export_paths.resolve_path.side_effect = resolve_path_side_effect
            mock_cli.inspect_exports.return_value = mock_export_paths

            # Create a real registry
            from joshpy.registry import RunRegistry
            registry = RunRegistry(":memory:")
            session_id = registry.create_session(experiment_name="test")

            # Register the config and run
            registry.register_run(
                session_id=session_id,
                run_hash="abc123def456",
                josh_path="/path/to/sim.josh",
                config_content="test config",
                file_mappings=None,
                parameters={"param1": 10},
            )
            run_id = registry.start_run(run_hash="abc123def456")

            # Mock loader
            mock_loader = MagicMock()
            mock_loader.load_csv.return_value = 5
            mock_loader_class.return_value = mock_loader

            rows = recover_sweep_results(
                cli=mock_cli,
                job_set=mock_job_set,
                registry=registry,
                quiet=True,
            )

            # Should have loaded 3 files * 5 rows = 15 total
            self.assertEqual(rows, 15)
            self.assertEqual(mock_loader.load_csv.call_count, 3)
            
            registry.close()


if __name__ == "__main__":
    unittest.main()
