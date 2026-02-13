"""Unit tests for the sweep module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

from joshpy.sweep import (
    recover_sweep_results,
    _get_export_path,
    SweepManager,
    SweepManagerBuilder,
)
from joshpy.jobs import JobConfig, SweepConfig, SweepParameter, JobExpander
from joshpy.registry import RunRegistry
from joshpy.cli import JoshCLI


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


class TestSweepManagerBuilder(unittest.TestCase):
    """Tests for SweepManagerBuilder class."""

    def setUp(self):
        """Create a temporary directory and sample config for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_path = Path(self.temp_dir) / "test_template.jshc.j2"
        self.template_path.write_text("simulation = {{ maxGrowth }}")
        
        self.source_path = Path(self.temp_dir) / "test.josh"
        self.source_path.write_text("// test josh file")
        
        self.config = JobConfig(
            template_path=self.template_path,
            source_path=self.source_path,
            simulation="TestSim",
            replicates=2,
            sweep=SweepConfig(
                parameters=[SweepParameter(name="maxGrowth", values=[10, 20])]
            ),
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_builder_creates_memory_registry_by_default(self):
        """Builder should create in-memory registry when not specified."""
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_cli().build()
        
        try:
            self.assertIsNotNone(manager.registry)
            self.assertTrue(manager._owns_registry)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_creates_cli_by_default(self):
        """Builder should create CLI when not specified."""
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_registry(":memory:").build()
        
        try:
            self.assertIsNotNone(manager.cli)
            self.assertTrue(manager._owns_cli)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_with_existing_registry(self):
        """Builder should use existing registry without ownership."""
        registry = RunRegistry(":memory:")
        
        try:
            builder = SweepManagerBuilder(self.config)
            manager = builder.with_registry(registry).with_cli().build()
            
            try:
                self.assertIs(manager.registry, registry)
                self.assertFalse(manager._owns_registry)
            finally:
                manager.cleanup()
                manager.close()
        finally:
            registry.close()

    def test_builder_with_existing_cli(self):
        """Builder should use existing CLI without ownership."""
        cli = JoshCLI()
        
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_registry(":memory:").with_cli(cli).build()
        
        try:
            self.assertIs(manager.cli, cli)
            self.assertFalse(manager._owns_cli)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_with_registry_path(self):
        """Builder should create registry from path with ownership."""
        db_path = Path(self.temp_dir) / "test.duckdb"
        
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_registry(db_path).with_cli().build()
        
        try:
            self.assertTrue(manager._owns_registry)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_with_defaults(self):
        """with_defaults should set up registry and CLI."""
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_defaults(registry=":memory:").build()
        
        try:
            self.assertIsNotNone(manager.registry)
            self.assertIsNotNone(manager.cli)
            self.assertTrue(manager._owns_registry)
            self.assertTrue(manager._owns_cli)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_expands_jobs(self):
        """Builder should expand jobs during build."""
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_defaults().build()
        
        try:
            # 2 parameter values = 2 jobs
            self.assertEqual(len(manager.job_set), 2)
            self.assertEqual(manager.job_set.total_jobs, 2)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_creates_session(self):
        """Builder should create session in registry."""
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_defaults(experiment_name="test_exp").build()
        
        try:
            self.assertIsNotNone(manager.session_id)
            session = manager.registry.get_session(manager.session_id)
            self.assertIsNotNone(session)
            self.assertEqual(session.experiment_name, "test_exp")
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_registers_runs(self):
        """Builder should register runs in registry."""
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_defaults().build()
        
        try:
            configs = manager.registry.get_configs_for_session(manager.session_id)
            # 2 parameter values = 2 configs
            self.assertEqual(len(configs), 2)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_verifies_hashes_on_existing_session(self):
        """Builder should verify hashes when using existing session."""
        # Create first manager
        registry = RunRegistry(":memory:")
        builder1 = SweepManagerBuilder(self.config)
        manager1 = builder1.with_registry(registry).with_cli().build()
        session_id = manager1.session_id
        manager1.cleanup()  # Only cleanup temp files, not registry
        
        # Create second manager with same config and existing session
        builder2 = SweepManagerBuilder(self.config)
        manager2 = builder2.with_registry(registry, session_id=session_id).with_cli().build()
        
        try:
            # Should succeed without error
            self.assertEqual(manager2.session_id, session_id)
        finally:
            manager2.cleanup()
            manager2.close()

    def test_builder_raises_on_hash_mismatch(self):
        """Builder should raise ValueError if hashes don't match existing session."""
        # Create first manager with one config
        registry = RunRegistry(":memory:")
        builder1 = SweepManagerBuilder(self.config)
        manager1 = builder1.with_registry(registry).with_cli().build()
        session_id = manager1.session_id
        manager1.cleanup()
        
        # Create different config
        different_config = JobConfig(
            template_path=self.template_path,
            source_path=self.source_path,
            simulation="TestSim",
            replicates=2,
            sweep=SweepConfig(
                parameters=[SweepParameter(name="maxGrowth", values=[30, 40])]  # Different values
            ),
        )
        
        # Try to use existing session with different config
        builder2 = SweepManagerBuilder(different_config)
        
        with self.assertRaises(ValueError) as ctx:
            builder2.with_registry(registry, session_id=session_id).with_cli().build()
        
        self.assertIn("hash mismatch", str(ctx.exception).lower())
        registry.close()


class TestSweepManager(unittest.TestCase):
    """Tests for SweepManager class."""

    def setUp(self):
        """Create a temporary directory and sample config for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_path = Path(self.temp_dir) / "test_template.jshc.j2"
        self.template_path.write_text("simulation = {{ maxGrowth }}")
        
        self.source_path = Path(self.temp_dir) / "test.josh"
        self.source_path.write_text("// test josh file")
        
        self.config = JobConfig(
            template_path=self.template_path,
            source_path=self.source_path,
            simulation="TestSim",
            replicates=1,
            sweep=SweepConfig(
                parameters=[SweepParameter(name="maxGrowth", values=[10, 20])]
            ),
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_from_config_creates_manager(self):
        """from_config should create a SweepManager from JobConfig."""
        manager = SweepManager.from_config(self.config, registry=":memory:")
        
        try:
            self.assertIsInstance(manager, SweepManager)
            self.assertEqual(len(manager.job_set), 2)
        finally:
            manager.cleanup()
            manager.close()

    def test_from_yaml_creates_manager(self):
        """from_yaml should create a SweepManager from YAML file."""
        yaml_path = Path(self.temp_dir) / "config.yaml"
        self.config.save_yaml(yaml_path)
        
        manager = SweepManager.from_yaml(yaml_path, registry=":memory:")
        
        try:
            self.assertIsInstance(manager, SweepManager)
            self.assertEqual(len(manager.job_set), 2)
        finally:
            manager.cleanup()
            manager.close()

    def test_builder_classmethod(self):
        """builder() should return a SweepManagerBuilder."""
        builder = SweepManager.builder(self.config)
        self.assertIsInstance(builder, SweepManagerBuilder)

    def test_run_remote_without_api_key(self):
        """run() with remote=True without api_key should work (for local servers)."""
        from unittest.mock import MagicMock
        from joshpy.cli import CLIResult
        
        manager = SweepManager.from_config(self.config, registry=":memory:")
        
        try:
            # Mock the CLI to avoid actual execution
            # Use a real CLIResult since RegistryCallback.record() checks isinstance
            mock_result = CLIResult(
                exit_code=0,
                stdout="",
                stderr="",
                command=["mock", "command"],
            )
            manager.cli = MagicMock()
            manager.cli.run_remote.return_value = mock_result
            
            # This should NOT raise - api_key is optional for local servers
            result = manager.run(remote=True, quiet=True)
            self.assertEqual(result.succeeded, len(manager.job_set))
        finally:
            manager.cleanup()
            manager.close()

    def test_run_dry_run_does_not_execute(self):
        """run() with dry_run=True should not execute jobs."""
        manager = SweepManager.from_config(self.config, registry=":memory:")
        
        try:
            result = manager.run(dry_run=True, quiet=True)
            
            # Dry run returns empty SweepResult
            self.assertEqual(len(result), 0)
            self.assertEqual(result.succeeded, 0)
            self.assertEqual(result.failed, 0)
        finally:
            manager.cleanup()
            manager.close()

    def test_cleanup_removes_temp_files(self):
        """cleanup() should remove temporary config files."""
        manager = SweepManager.from_config(self.config, registry=":memory:")
        temp_dir = manager.job_set.temp_dir
        
        # Temp dir should exist before cleanup
        self.assertIsNotNone(temp_dir)
        self.assertTrue(temp_dir.exists())
        
        manager.cleanup()
        
        # Temp dir should not exist after cleanup
        self.assertFalse(temp_dir.exists())
        manager.close()

    def test_context_manager_cleans_up(self):
        """Using as context manager should cleanup on exit."""
        with SweepManager.from_config(self.config, registry=":memory:") as manager:
            temp_dir = manager.job_set.temp_dir
            self.assertTrue(temp_dir.exists())
        
        # Should be cleaned up after context exit
        self.assertFalse(temp_dir.exists())

    def test_close_only_closes_owned_registry(self):
        """close() should only close registry if owned."""
        registry = RunRegistry(":memory:")
        
        builder = SweepManagerBuilder(self.config)
        manager = builder.with_registry(registry).with_cli().build()
        
        manager.cleanup()
        manager.close()
        
        # Registry should still be usable since manager didn't own it
        # If close() closed it incorrectly, this would raise
        sessions = registry.list_sessions()
        self.assertIsInstance(sessions, list)
        
        registry.close()

    @patch("joshpy.sweep.run_sweep")
    def test_run_calls_run_sweep(self, mock_run_sweep):
        """run() should call run_sweep for local execution."""
        from joshpy.jobs import SweepResult
        mock_run_sweep.return_value = SweepResult(succeeded=2, failed=0)
        
        manager = SweepManager.from_config(self.config, registry=":memory:")
        
        try:
            result = manager.run(quiet=True)
            
            mock_run_sweep.assert_called_once()
            self.assertEqual(result.succeeded, 2)
        finally:
            manager.cleanup()
            manager.close()

    @patch("joshpy.sweep.recover_sweep_results")
    def test_load_results_calls_recover(self, mock_recover):
        """load_results() should call recover_sweep_results."""
        mock_recover.return_value = 100
        
        manager = SweepManager.from_config(self.config, registry=":memory:")
        
        try:
            rows = manager.load_results(quiet=True)
            
            mock_recover.assert_called_once()
            self.assertEqual(rows, 100)
        finally:
            manager.cleanup()
            manager.close()


if __name__ == "__main__":
    unittest.main()
