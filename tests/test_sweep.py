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
from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter, JobExpander
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
                run_ids={},
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
                run_ids={},
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
                run_ids={"abc": "run-1"},
            )

        self.assertIn("patch export configured", str(ctx.exception))

    @patch("joshpy.sweep.CellDataLoader")
    def test_skips_jobs_without_run_id(self, mock_loader_class):
        """Should skip jobs whose run_hash is not in run_ids."""
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

        # Empty run_ids — job has no run_id mapping
        rows = recover_sweep_results(
            cli=mock_cli,
            job_set=mock_job_set,
            registry=mock_registry,
            run_ids={},
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
            config = JobConfig(simulation="TestSim")
            session_id = registry.create_session(config, experiment_name="test")

            # Register the config and run
            registry.register_run(
                session_id=session_id,
                run_hash="abc123def456",
                josh_path="/path/to/sim.josh",
                config_content="test config",
                file_mappings=None,
                parameters={},
            )
            run_id = registry.start_run(run_hash="abc123def456", session_id=session_id)

            # Mock loader
            mock_loader = MagicMock()
            mock_loader.load_csv.return_value = 10
            mock_loader_class.return_value = mock_loader

            rows = recover_sweep_results(
                cli=mock_cli,
                job_set=mock_job_set,
                registry=registry,
                run_ids={"abc123def456": run_id},
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
            config = JobConfig(simulation="TestSim")
            session_id = registry.create_session(config, experiment_name="test")

            # Register the config and run
            registry.register_run(
                session_id=session_id,
                run_hash="abc123def456",
                josh_path="/path/to/sim.josh",
                config_content="test config",
                file_mappings=None,
                parameters={"param1": 10},
            )
            run_id = registry.start_run(run_hash="abc123def456", session_id=session_id)

            # Mock loader
            mock_loader = MagicMock()
            mock_loader.load_csv.return_value = 5
            mock_loader_class.return_value = mock_loader

            rows = recover_sweep_results(
                cli=mock_cli,
                job_set=mock_job_set,
                registry=registry,
                run_ids={"abc123def456": run_id},
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
                config_parameters=[ConfigSweepParameter(name="maxGrowth", values=[10, 20])]
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
                config_parameters=[ConfigSweepParameter(name="maxGrowth", values=[30, 40])]  # Different values
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
                config_parameters=[ConfigSweepParameter(name="maxGrowth", values=[10, 20])]
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
            # Simulate run() having been called by setting run_ids
            manager._last_run_ids = {"hash1": "run-1"}
            rows = manager.load_results(quiet=True)

            mock_recover.assert_called_once()
            self.assertEqual(rows, 100)
        finally:
            manager.cleanup()
            manager.close()

    def test_load_results_raises_without_run(self):
        """load_results() should raise if run() hasn't been called."""
        manager = SweepManager.from_config(self.config, registry=":memory:")

        try:
            with self.assertRaises(RuntimeError):
                manager.load_results(quiet=True)
        finally:
            manager.cleanup()
            manager.close()

    @patch("joshpy.sweep.run_sweep")
    def test_run_forwards_jfr(self, mock_run_sweep):
        """run(jfr=...) should forward jfr to run_sweep."""
        from joshpy.cli import JfrConfig
        from joshpy.jobs import SweepResult

        mock_run_sweep.return_value = SweepResult(succeeded=2, failed=0)

        manager = SweepManager.from_config(self.config, registry=":memory:")

        try:
            jfr = JfrConfig(output=Path("/tmp/sweep_profile.jfr"))
            manager.run(quiet=True, jfr=jfr)

            call_kwargs = mock_run_sweep.call_args[1]
            self.assertIs(call_kwargs["jfr"], jfr)
        finally:
            manager.cleanup()
            manager.close()


def _has_optuna() -> bool:
    """Check if optuna is available."""
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


class TestSweepManagerAdaptiveDispatch(unittest.TestCase):
    """Tests for SweepManager adaptive strategy dispatch."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.josh_path = Path(self.temp_dir) / "test.josh"
        self.josh_path.write_text("start simulation Main\nend simulation\n")

    def tearDown(self):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_detects_cartesian_strategy(self):
        """run() should use run_sweep for CartesianStrategy."""
        from joshpy.strategies import CartesianStrategy

        config = JobConfig(
            template_string="# config",
            source_path=self.josh_path,
            sweep=SweepConfig(
                config_parameters=[ConfigSweepParameter(name="x", values=[1, 2])],
                strategy=CartesianStrategy(),
            ),
        )

        with patch("joshpy.sweep.run_sweep") as mock_run_sweep:
            from joshpy.jobs import SweepResult

            mock_run_sweep.return_value = SweepResult(succeeded=2, failed=0)

            manager = SweepManager.from_config(config, registry=":memory:")

            try:
                result = manager.run(quiet=True)
                mock_run_sweep.assert_called_once()
            finally:
                manager.cleanup()
                manager.close()

    @unittest.skipUnless(_has_optuna(), "optuna not installed")
    def test_run_detects_optuna_strategy(self):
        """run() should use run_adaptive_sweep for OptunaStrategy."""
        from joshpy.strategies import OptunaStrategy

        config = JobConfig(
            template_string="# config",
            source_path=self.josh_path,
            sweep=SweepConfig(
                config_parameters=[ConfigSweepParameter(name="x", values=[1, 2])],
                strategy=OptunaStrategy(n_trials=5, objective="math:sin"),
            ),
        )

        with patch("joshpy.strategies.run_adaptive_sweep") as mock_run_adaptive:
            from joshpy.jobs import AdaptiveSweepResult

            mock_run_adaptive.return_value = AdaptiveSweepResult(
                succeeded=3, failed=0, best_value=0.5
            )

            manager = SweepManager.from_config(config, registry=":memory:")

            try:
                result = manager.run(quiet=True)
                mock_run_adaptive.assert_called_once()
            finally:
                manager.cleanup()
                manager.close()

    def test_run_dry_run_with_adaptive_prints_message(self):
        """run(dry_run=True) should print message for adaptive without executing."""
        from joshpy.strategies import OptunaStrategy
        from joshpy.jobs import SweepResult

        config = JobConfig(
            template_string="# config",
            source_path=self.josh_path,
            sweep=SweepConfig(
                config_parameters=[ConfigSweepParameter(name="x", values=[1, 2])],
                strategy=OptunaStrategy(n_trials=10, objective="math:sin"),
            ),
        )

        manager = SweepManager.from_config(config, registry=":memory:")

        try:
            # Dry run should return empty SweepResult without calling run_adaptive_sweep
            result = manager.run(dry_run=True, quiet=True)
            self.assertIsInstance(result, SweepResult)
            self.assertEqual(len(result), 0)
        finally:
            manager.cleanup()
            manager.close()

    def test_run_without_sweep_uses_batch(self):
        """run() should use run_sweep when no sweep config is present."""
        config = JobConfig(
            template_string="# config",
            source_path=self.josh_path,
            # No sweep config
        )

        with patch("joshpy.sweep.run_sweep") as mock_run_sweep:
            from joshpy.jobs import SweepResult

            mock_run_sweep.return_value = SweepResult(succeeded=1, failed=0)

            manager = SweepManager.from_config(config, registry=":memory:")

            try:
                result = manager.run(quiet=True)
                mock_run_sweep.assert_called_once()
            finally:
                manager.cleanup()
                manager.close()


class TestSweepManagerCatalogIntegration(unittest.TestCase):
    """Tests for SweepManager + ProjectCatalog integration."""

    def test_catalog_stores_registry_path(self):
        """with_catalog() should store the registry path in the catalog (regression for db_path bug)."""
        from joshpy.catalog import ProjectCatalog

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = str(Path(tmpdir) / "experiment.duckdb")
            catalog_path = str(Path(tmpdir) / "catalog.duckdb")

            config = JobConfig(
                template_string="maxGrowth = {{ maxGrowth }} meters",
                simulation="Main",
            )

            catalog = ProjectCatalog(catalog_path)
            try:
                manager = (
                    SweepManagerBuilder(config)
                    .with_registry(registry_path, experiment_name="test_exp")
                    .with_catalog(catalog, experiment_name="test_exp")
                    .build()
                )

                # Verify the catalog recorded the registry path
                experiments = catalog.list_experiments()
                self.assertEqual(len(experiments), 1)
                self.assertEqual(experiments[0].registry_path, registry_path)
                self.assertNotEqual(experiments[0].registry_path, "")

                manager.cleanup()
                manager.close()
            finally:
                catalog.close()

    def test_with_label_single_job(self):
        """with_label() should apply label to a single-job config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = JobConfig(
                template_string="maxGrowth = 50 meters",
                simulation="Main",
            )

            manager = (
                SweepManagerBuilder(config)
                .with_registry(":memory:", experiment_name="test")
                .with_label("baseline")
                .build()
            )

            try:
                labels = manager.registry.list_labels()
                self.assertEqual(len(labels), 1)
                self.assertEqual(labels[0][0], "baseline")
            finally:
                manager.cleanup()
                manager.close()

    def test_with_label_multi_job_raises(self):
        """with_label() should raise for multi-job sweeps."""
        config = JobConfig(
            template_string="maxGrowth = {{ maxGrowth }} meters",
            simulation="Main",
            sweep=SweepConfig(
                config_parameters=[
                    ConfigSweepParameter(name="maxGrowth", values=[10, 20, 30]),
                ],
            ),
        )

        with self.assertRaises(ValueError) as ctx:
            SweepManagerBuilder(config).with_registry(":memory:").with_label("test").build()
        self.assertIn("single-job", str(ctx.exception))

    def test_with_label_force(self):
        """with_label(force=True) should reassign an existing label."""
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")

        config1 = JobConfig(
            template_string="maxGrowth = 50 meters",
            simulation="Main",
        )
        m1 = (
            SweepManagerBuilder(config1)
            .with_registry(registry, experiment_name="test1")
            .with_label("baseline")
            .build()
        )
        hash1 = m1.job_set.jobs[0].run_hash
        m1.cleanup()

        config2 = JobConfig(
            template_string="maxGrowth = 100 meters",
            simulation="Main",
        )
        m2 = (
            SweepManagerBuilder(config2)
            .with_registry(registry, experiment_name="test2")
            .with_label("baseline", force=True)
            .build()
        )
        hash2 = m2.job_set.jobs[0].run_hash
        m2.cleanup()

        self.assertEqual(registry.resolve_label("baseline"), hash2)
        # Old run has no label
        config_info = registry.get_config_by_hash(hash1)
        self.assertIsNone(config_info.label)
        registry.close()

    def test_with_label_on_collision_timestamp(self):
        """with_label(on_collision='timestamp') should archive old label."""
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")

        config1 = JobConfig(
            template_string="maxGrowth = 50 meters",
            simulation="Main",
        )
        m1 = (
            SweepManagerBuilder(config1)
            .with_registry(registry, experiment_name="test1")
            .with_label("baseline")
            .build()
        )
        hash1 = m1.job_set.jobs[0].run_hash
        m1.cleanup()

        config2 = JobConfig(
            template_string="maxGrowth = 100 meters",
            simulation="Main",
        )
        m2 = (
            SweepManagerBuilder(config2)
            .with_registry(registry, experiment_name="test2")
            .with_label("baseline", on_collision="timestamp")
            .build()
        )
        hash2 = m2.job_set.jobs[0].run_hash
        m2.cleanup()

        # Bare label points to new run
        self.assertEqual(registry.resolve_label("baseline"), hash2)
        # Old run has a timestamped label
        old_config = registry.get_config_by_hash(hash1)
        self.assertIsNotNone(old_config.label)
        self.assertRegex(old_config.label, r"^baseline_\d{8}_\d{6}")
        registry.close()


class TestIngestResults(unittest.TestCase):
    """Tests for ingest_results()."""

    def _make_registry_with_run(self, replicates=3):
        """Create an in-memory registry with a labeled run for testing."""
        registry = RunRegistry(":memory:")
        config = JobConfig(
            source_path=Path("/tmp/sim.josh"),
            simulation="Main",
            replicates=replicates,
        )
        session_id = registry.create_session(
            config=config,
            experiment_name="test",
        )
        # Register a config
        registry.register_run(
            session_id=session_id,
            run_hash="abc123def456",
            josh_path="/tmp/sim.josh",
            config_content="config_here",
            file_mappings=None,
            parameters={"maxGrowth": 50},
            josh_content="simulation Main { }",
        )
        registry.label_run("abc123def456", "test-label")

        # Start runs so _resolve_run_id_for_hash works and replicate count is right
        run_id = None
        for _ in range(replicates):
            run_id = registry.start_run("abc123def456", session_id=session_id)
            registry.complete_run(run_id, exit_code=0)

        return registry, session_id, run_id

    @patch("joshpy.sweep.CellDataLoader")
    def test_local_file_protocol(self, mock_loader_cls):
        """ingest_results with file:// protocol loads local CSVs."""
        from joshpy.sweep import ingest_results
        from joshpy.cli import ExportFileInfo, ExportPaths

        registry, _, run_id = self._make_registry_with_run()

        mock_loader = MagicMock()
        mock_loader.load_csv.return_value = 100
        mock_loader_cls.return_value = mock_loader

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw="file:///tmp/output_{replicate}.csv",
                    protocol="file",
                    host="",
                    path="/tmp/output_{replicate}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={"organism": None, "patch": None, "agent": None, "disturbance": None},
        )

        # Create fake CSV files
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            for rep in range(3):
                csv_path = Path(f"/tmp/output_{rep}.csv")
                csv_path.write_text("step,replicate,val\n0,0,1.0\n")

            try:
                rows = ingest_results(mock_cli, registry, "test-label", quiet=True)
                # Should have called load_csv 3 times
                self.assertEqual(mock_loader.load_csv.call_count, 3)
            finally:
                for rep in range(3):
                    Path(f"/tmp/output_{rep}.csv").unlink(missing_ok=True)

        registry.close()

    @patch("joshpy.sweep.CellDataLoader")
    def test_missing_replicate_skipped(self, mock_loader_cls):
        """Missing CSVs should be skipped gracefully."""
        from joshpy.sweep import ingest_results
        from joshpy.cli import ExportFileInfo, ExportPaths

        registry, _, _ = self._make_registry_with_run()

        mock_loader = MagicMock()
        mock_loader.load_csv.side_effect = FileNotFoundError("not found")
        mock_loader_cls.return_value = mock_loader

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw="file:///tmp/missing_{replicate}.csv",
                    protocol="file",
                    host="",
                    path="/tmp/missing_{replicate}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={"organism": None, "patch": None, "agent": None, "disturbance": None},
        )

        rows = ingest_results(mock_cli, registry, "test-label", quiet=True)
        self.assertEqual(rows, 0)
        registry.close()

    def test_unknown_label_raises(self):
        """ingest_results should raise KeyError for unknown label."""
        from joshpy.sweep import ingest_results

        registry = RunRegistry(":memory:")
        mock_cli = MagicMock()

        with self.assertRaises(KeyError):
            ingest_results(mock_cli, registry, "nonexistent-label")
        registry.close()

    @patch("joshpy.sweep.CellDataLoader")
    def test_minio_protocol_configures_s3(self, mock_loader_cls):
        """minio:// protocol should call configure_s3 and build s3:// URLs."""
        from joshpy.sweep import ingest_results
        from joshpy.cli import ExportFileInfo, ExportPaths

        registry, _, _ = self._make_registry_with_run()

        mock_loader = MagicMock()
        mock_loader.load_csv.return_value = 50
        mock_loader_cls.return_value = mock_loader

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw="minio://my-bucket/results/output_{replicate}.csv",
                    protocol="minio",
                    host="my-bucket",
                    path="/results/output_{replicate}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={"organism": None, "patch": None, "agent": None, "disturbance": None},
        )

        env = {
            "MINIO_ENDPOINT": "storage.example.com",
            "MINIO_ACCESS_KEY": "AKID",
            "MINIO_SECRET_KEY": "SECRET",
        }
        with patch("joshpy.sweep.configure_s3") as mock_configure, \
             patch.dict("os.environ", env):
            rows = ingest_results(mock_cli, registry, "test-label", quiet=True)

            # Should have configured S3
            mock_configure.assert_called_once()
            call_args = mock_configure.call_args
            self.assertEqual(call_args[0][1], "storage.example.com")

            # load_csv should have been called with s3:// URLs
            for call in mock_loader.load_csv.call_args_list:
                csv_arg = call[1].get("csv_path") or call[0][0]
                self.assertTrue(str(csv_arg).startswith("s3://my-bucket/"))

        registry.close()

    @patch("joshpy.sweep.CellDataLoader")
    def test_minio_missing_creds_raises(self, mock_loader_cls):
        """minio:// without env vars should raise RuntimeError."""
        from joshpy.sweep import ingest_results
        from joshpy.cli import ExportFileInfo, ExportPaths

        registry, _, _ = self._make_registry_with_run()

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw="minio://bucket/out_{replicate}.csv",
                    protocol="minio",
                    host="bucket",
                    path="/out_{replicate}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={"organism": None, "patch": None, "agent": None, "disturbance": None},
        )

        # Clear any minio env vars
        clean_env = {k: v for k, v in __import__("os").environ.items()
                     if not k.startswith("MINIO_")}
        with patch.dict("os.environ", clean_env, clear=True):
            with self.assertRaises(RuntimeError):
                ingest_results(mock_cli, registry, "test-label", quiet=True)

        registry.close()

    @patch("joshpy.sweep.CellDataLoader")
    def test_josh_content_fallback(self, mock_loader_cls):
        """Should use josh_content from registry when josh_path doesn't exist."""
        from joshpy.sweep import ingest_results
        from joshpy.cli import ExportFileInfo, ExportPaths

        registry, _, _ = self._make_registry_with_run()

        mock_loader = MagicMock()
        mock_loader.load_csv.return_value = 10
        mock_loader_cls.return_value = mock_loader

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw="file:///tmp/out_{replicate}.csv",
                    protocol="file",
                    host="",
                    path="/tmp/out_{replicate}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={"organism": None, "patch": None, "agent": None, "disturbance": None},
        )

        # josh_path is /tmp/sim.josh which doesn't exist — should fall back to josh_content
        rows = ingest_results(mock_cli, registry, "test-label", quiet=True)

        # inspect_exports should have been called with a temp file (not /tmp/sim.josh)
        call_config = mock_cli.inspect_exports.call_args[0][0]
        self.assertNotEqual(str(call_config.script), "/tmp/sim.josh")
        # Temp file has .josh suffix
        self.assertTrue(str(call_config.script).endswith(".josh"))

        registry.close()


class TestSweepManagerBuilderBatchRemote(unittest.TestCase):
    """Tests for SweepManagerBuilder.with_batch_remote() and the _UNSET
    sentinel-based default resolution in SweepManager.run()."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.template_path = Path(self.temp_dir) / "t.jshc.j2"
        self.template_path.write_text("x = {{ p }}")
        self.source_path = Path(self.temp_dir) / "sim.josh"
        self.source_path.write_text("// josh")
        self.config = JobConfig(
            template_path=self.template_path,
            source_path=self.source_path,
            simulation="Main",
            replicates=1,
            sweep=SweepConfig(
                config_parameters=[ConfigSweepParameter(name="p", values=[1])]
            ),
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_with_batch_remote_stashes_defaults(self):
        builder = SweepManagerBuilder(self.config)
        builder.with_batch_remote(
            "gke-test", no_wait=True, poll_interval=30,
            timeout=600, auto_ingest=False,
        )
        self.assertEqual(builder._batch_remote_target, "gke-test")
        self.assertTrue(builder._batch_no_wait)
        self.assertEqual(builder._batch_poll_interval, 30)
        self.assertEqual(builder._batch_timeout, 600)
        self.assertFalse(builder._batch_auto_ingest)

    def test_build_propagates_defaults_to_manager(self):
        manager = (
            SweepManagerBuilder(self.config)
            .with_registry(":memory:")
            .with_batch_remote("gke-test", timeout=900)
            .build()
        )
        try:
            self.assertEqual(manager._batch_remote_target, "gke-test")
            self.assertEqual(manager._batch_timeout, 900)
        finally:
            manager.cleanup()
            manager.close()

    def test_run_uses_builder_defaults_when_kwargs_omitted(self):
        """With `_batch_remote_target` set, `run()` should default to
        `batch_remote=True, target=<stashed>`."""
        from joshpy.jobs import SweepResult

        mock_cli = MagicMock()
        registry = RunRegistry(":memory:")
        try:
            manager = (
                SweepManagerBuilder(self.config)
                .with_registry(registry)
                .with_cli(mock_cli)
                .with_batch_remote("gke-test")
                .build()
            )

            with patch("joshpy.sweep.run_sweep") as mock_run_sweep:
                mock_run_sweep.return_value = SweepResult(
                    job_results=[], succeeded=0, failed=0, run_ids={},
                )
                manager.run(quiet=True)

            call_kwargs = mock_run_sweep.call_args.kwargs
            self.assertTrue(call_kwargs["batch_remote"])
            self.assertEqual(call_kwargs["target"], "gke-test")
            self.assertFalse(call_kwargs["batch_no_wait"])
            self.assertEqual(call_kwargs["poll_interval"], 10)
            self.assertIsNone(call_kwargs["batch_timeout"])
            self.assertTrue(call_kwargs["auto_ingest"])

            manager.cleanup()
            manager.close()
        finally:
            registry.close()

    def test_run_explicit_override_wins(self):
        """Passing `batch_remote=False` must override the builder default."""
        from joshpy.jobs import SweepResult

        mock_cli = MagicMock()
        registry = RunRegistry(":memory:")
        try:
            manager = (
                SweepManagerBuilder(self.config)
                .with_registry(registry)
                .with_cli(mock_cli)
                .with_batch_remote("gke-test")
                .build()
            )

            with patch("joshpy.sweep.run_sweep") as mock_run_sweep:
                mock_run_sweep.return_value = SweepResult(
                    job_results=[], succeeded=0, failed=0, run_ids={},
                )
                # Caller override: disable batch remote entirely
                manager.run(batch_remote=False, target=None, quiet=True)

            call_kwargs = mock_run_sweep.call_args.kwargs
            self.assertFalse(call_kwargs["batch_remote"])
            self.assertIsNone(call_kwargs["target"])

            manager.cleanup()
            manager.close()
        finally:
            registry.close()

    def test_run_target_override_wins(self):
        from joshpy.jobs import SweepResult

        mock_cli = MagicMock()
        registry = RunRegistry(":memory:")
        try:
            manager = (
                SweepManagerBuilder(self.config)
                .with_registry(registry)
                .with_cli(mock_cli)
                .with_batch_remote("gke-test")
                .build()
            )

            with patch("joshpy.sweep.run_sweep") as mock_run_sweep:
                mock_run_sweep.return_value = SweepResult(
                    job_results=[], succeeded=0, failed=0, run_ids={},
                )
                manager.run(target="other-target", quiet=True)

            call_kwargs = mock_run_sweep.call_args.kwargs
            self.assertTrue(call_kwargs["batch_remote"])  # still True via builder
            self.assertEqual(call_kwargs["target"], "other-target")

            manager.cleanup()
            manager.close()
        finally:
            registry.close()

    def test_run_without_builder_defaults_to_local(self):
        """Without with_batch_remote(), run() should default to
        batch_remote=False (backwards compatible)."""
        from joshpy.jobs import SweepResult

        mock_cli = MagicMock()
        registry = RunRegistry(":memory:")
        try:
            manager = (
                SweepManagerBuilder(self.config)
                .with_registry(registry)
                .with_cli(mock_cli)
                .build()
            )

            with patch("joshpy.sweep.run_sweep") as mock_run_sweep:
                mock_run_sweep.return_value = SweepResult(
                    job_results=[], succeeded=0, failed=0, run_ids={},
                )
                manager.run(quiet=True)

            call_kwargs = mock_run_sweep.call_args.kwargs
            self.assertFalse(call_kwargs["batch_remote"])
            self.assertIsNone(call_kwargs["target"])

            manager.cleanup()
            manager.close()
        finally:
            registry.close()


class TestConfigureS3(unittest.TestCase):
    """Tests for configure_s3()."""

    def test_executes_install_and_create_secret(self):
        """configure_s3 should call INSTALL httpfs and CREATE SECRET."""
        from joshpy.registry import configure_s3

        mock_conn = MagicMock()
        configure_s3(mock_conn, "storage.example.com", "AKID", "SECRET")

        # Should have called execute twice: INSTALL + CREATE SECRET
        self.assertEqual(mock_conn.execute.call_count, 2)
        first_call = mock_conn.execute.call_args_list[0]
        self.assertIn("INSTALL httpfs", first_call[0][0])
        second_call = mock_conn.execute.call_args_list[1]
        self.assertIn("CREATE OR REPLACE SECRET", second_call[0][0])

    def _secret_params(self, mock_conn):
        """Extract [access_key, secret_key, endpoint, url_style, use_ssl]."""
        return mock_conn.execute.call_args_list[1][0][1]

    def test_bare_hostname_passthrough(self):
        """Bare hostnames pass through unchanged with default use_ssl=True."""
        from joshpy.registry import configure_s3

        mock_conn = MagicMock()
        configure_s3(mock_conn, "storage.example.com", "AKID", "SECRET")

        params = self._secret_params(mock_conn)
        self.assertEqual(params[2], "storage.example.com")
        self.assertTrue(params[4])

    def test_https_scheme_stripped(self):
        """https:// prefix is stripped; use_ssl inferred True."""
        from joshpy.registry import configure_s3

        mock_conn = MagicMock()
        configure_s3(
            mock_conn, "https://storage.googleapis.com", "AKID", "SECRET"
        )

        params = self._secret_params(mock_conn)
        self.assertEqual(params[2], "storage.googleapis.com")
        self.assertTrue(params[4])

    def test_http_scheme_stripped_and_ssl_false(self):
        """http:// prefix is stripped; use_ssl inferred False."""
        from joshpy.registry import configure_s3

        mock_conn = MagicMock()
        configure_s3(mock_conn, "http://localhost:9000", "AKID", "SECRET")

        params = self._secret_params(mock_conn)
        self.assertEqual(params[2], "localhost:9000")
        self.assertFalse(params[4])

    def test_explicit_use_ssl_overrides_inference(self):
        """Explicit use_ssl wins over scheme inference."""
        from joshpy.registry import configure_s3

        mock_conn = MagicMock()
        configure_s3(
            mock_conn, "https://storage.example.com", "AKID", "SECRET",
            use_ssl=False,
        )

        params = self._secret_params(mock_conn)
        self.assertFalse(params[4])

    def test_trailing_slash_stripped(self):
        """Trailing slash on the endpoint is stripped."""
        from joshpy.registry import configure_s3

        mock_conn = MagicMock()
        configure_s3(
            mock_conn, "https://storage.example.com/", "AKID", "SECRET"
        )

        params = self._secret_params(mock_conn)
        self.assertEqual(params[2], "storage.example.com")

    def test_trailing_path_stripped(self):
        """Trailing path segments are stripped from the endpoint."""
        from joshpy.registry import configure_s3

        mock_conn = MagicMock()
        configure_s3(
            mock_conn, "https://storage.example.com/bucket/prefix",
            "AKID", "SECRET",
        )

        params = self._secret_params(mock_conn)
        self.assertEqual(params[2], "storage.example.com")


class TestIngestResultsExportPathsKwarg(unittest.TestCase):
    """Tests for ingest_results(export_paths=...) optimization."""

    def _setup_mocks(self, protocol: str = "minio"):
        """Build the minimum mock surface ingest_results needs."""
        mock_cli = MagicMock()
        mock_registry = MagicMock()

        # _resolve_ingest_metadata reads run hash + config + session
        mock_registry._resolve_label_or_hash.return_value = "abc123def456"
        config = MagicMock()
        config.session_id = "s1"
        config.label = None
        config.parameters = {}
        mock_registry.get_config_by_hash.return_value = config
        session = MagicMock()
        session.simulation = "Main"
        session.total_replicates = 1
        session.job_config = None
        mock_registry.get_session.return_value = session
        mock_registry.get_runs_for_hash.return_value = [MagicMock()]
        mock_registry._resolve_run_id_for_hash.return_value = "run-1"

        # ExportPaths fixture — .export_files with requested protocol
        export_info = MagicMock()
        export_info.protocol = protocol
        export_info.host = "bucket"
        export_info.path = "/prefix/output_{replicate}.csv"
        mock_export_paths = MagicMock()
        mock_export_paths.export_files = {"patch": export_info}
        return mock_cli, mock_registry, mock_export_paths

    @patch("joshpy.sweep._get_josh_source")
    @patch("joshpy.sweep._configure_minio_access", return_value=("bucket", None))
    @patch("joshpy.sweep._load_ingest_replicates", return_value=42)
    def test_inspect_exports_called_when_not_passed(
        self, _mock_load, _mock_cfg, mock_get_josh,
    ):
        """Default: ingest_results should call cli.inspect_exports once."""
        from joshpy.sweep import ingest_results

        mock_cli, mock_registry, mock_export_paths = self._setup_mocks()
        mock_cli.inspect_exports.return_value = mock_export_paths
        mock_get_josh.return_value = (Path("/tmp/sim.josh"), None)

        ingest_results(mock_cli, mock_registry, "abc123def456", quiet=True)
        mock_cli.inspect_exports.assert_called_once()

    @patch("joshpy.sweep._get_josh_source")
    @patch("joshpy.sweep._configure_minio_access", return_value=("bucket", None))
    @patch("joshpy.sweep._load_ingest_replicates", return_value=42)
    def test_inspect_exports_skipped_when_passed(
        self, _mock_load, _mock_cfg, mock_get_josh,
    ):
        """When export_paths= is provided, skip the inspect_exports subprocess."""
        from joshpy.sweep import ingest_results

        mock_cli, mock_registry, mock_export_paths = self._setup_mocks()
        mock_get_josh.return_value = (Path("/tmp/sim.josh"), None)

        ingest_results(
            mock_cli, mock_registry, "abc123def456",
            export_paths=mock_export_paths,
            quiet=True,
        )
        mock_cli.inspect_exports.assert_not_called()


class TestLoadJobResultsMinioRouting(unittest.TestCase):
    """Tests for load_job_results protocol dispatch (PR7)."""

    def _make_job(self, source_path: Path) -> "ExpandedJob":
        from joshpy.jobs import ExpandedJob

        return ExpandedJob(
            config_content="",
            config_path=source_path.parent / "c.jshc",
            config_name="c",
            run_hash="abcdef012345",
            parameters={"x": 1},
            simulation="Main",
            replicates=2,
            source_path=source_path,
            file_mappings={},
        )

    @patch("joshpy.sweep._load_job_results_minio", return_value=99)
    def test_minio_protocol_routes_to_helper(self, mock_minio):
        """export_info.protocol == 'minio' delegates to _load_job_results_minio."""
        from joshpy.sweep import load_job_results

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job = self._make_job(src)
            export_info = MagicMock()
            export_info.protocol = "minio"
            export_paths = MagicMock()
            export_paths.export_files = {"patch": export_info}

            rows = load_job_results(
                cli=MagicMock(),
                job=job,
                registry=MagicMock(),
                export_paths=export_paths,
                run_id="run-1",
            )

            self.assertEqual(rows, 99)
            mock_minio.assert_called_once()

    @patch("joshpy.sweep._load_job_results_minio")
    def test_file_protocol_uses_local_path(self, mock_minio):
        """export_info.protocol == 'file' does NOT call the minio helper."""
        from joshpy.sweep import load_job_results

        registry = RunRegistry(":memory:")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "sim.josh"
                src.write_text("x")
                job = self._make_job(src)
                export_info = MagicMock()
                export_info.protocol = "file"
                export_info.path = str(Path(tmp) / "missing_{replicate}.csv")
                export_paths = MagicMock()
                export_paths.export_files = {"patch": export_info}
                # Also mock get_patch_path since local path uses _get_export_path
                export_paths.get_patch_path.return_value = export_info.path
                export_paths.resolve_path.side_effect = lambda t, **kw: Path(
                    t.format(**kw)
                )

                rows = load_job_results(
                    cli=MagicMock(),
                    job=job,
                    registry=registry,
                    export_paths=export_paths,
                    run_id="run-1",
                    quiet=True,
                )

                # Local path: file doesn't exist, so 0 rows loaded; minio helper never called
                self.assertEqual(rows, 0)
                mock_minio.assert_not_called()
        finally:
            registry.close()


class TestStaticCollisionCheck(unittest.TestCase):
    """Tests for _check_export_path_safety + SweepManager.run(force=...)."""

    def _make_job(self, source_path: Path, run_hash: str = "abcdef012345") -> "ExpandedJob":
        from joshpy.jobs import ExpandedJob

        return ExpandedJob(
            config_content="",
            config_path=source_path.parent / "c.jshc",
            config_name="c",
            run_hash=run_hash,
            parameters={},
            simulation="Main",
            replicates=2,
            source_path=source_path,
            file_mappings={},
        )

    def _make_export_paths(self, *, protocol: str, path: str) -> MagicMock:
        export_info = MagicMock()
        export_info.protocol = protocol
        export_info.path = path
        export_paths = MagicMock()
        export_paths.export_files = {"patch": export_info}
        return export_paths

    def _make_job_set(self, jobs: list) -> MagicMock:
        job_set = MagicMock()
        job_set.__iter__ = lambda self: iter(jobs)
        return job_set

    # --- Helper-level tests --------------------------------------------------

    def test_local_file_protocol_skips_check(self):
        """protocol='file' is out of scope; never raises even with prior runs."""
        from joshpy.sweep import _check_export_path_safety

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job = self._make_job(src)
            cli = MagicMock()
            cli.inspect_exports.return_value = self._make_export_paths(
                protocol="file", path="/tmp/output_{replicate}.csv",
            )
            registry = MagicMock()
            registry.get_runs_for_hash.return_value = [MagicMock()]  # prior runs

            _check_export_path_safety(cli, self._make_job_set([job]), registry)
            registry.get_runs_for_hash.assert_not_called()

    def test_template_with_timestamp_skips_check(self):
        """Template containing {timestamp} disambiguates dispatches; safe."""
        from joshpy.sweep import _check_export_path_safety

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job = self._make_job(src)
            cli = MagicMock()
            cli.inspect_exports.return_value = self._make_export_paths(
                protocol="minio",
                path="/bucket/{timestamp}/output_{replicate}.csv",
            )
            registry = MagicMock()
            registry.get_runs_for_hash.return_value = [MagicMock()]

            _check_export_path_safety(cli, self._make_job_set([job]), registry)
            registry.get_runs_for_hash.assert_not_called()

    def test_template_with_run_hash_skips_check(self):
        """Template containing {run_hash} disambiguates per-simulation; safe."""
        from joshpy.sweep import _check_export_path_safety

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job = self._make_job(src)
            cli = MagicMock()
            cli.inspect_exports.return_value = self._make_export_paths(
                protocol="minio",
                path="/bucket/{run_hash}/output_{replicate}.csv",
            )
            registry = MagicMock()
            registry.get_runs_for_hash.return_value = [MagicMock()]

            _check_export_path_safety(cli, self._make_job_set([job]), registry)
            registry.get_runs_for_hash.assert_not_called()

    def test_minio_no_prior_runs_passes(self):
        """Plain template + empty registry: no collision, no raise."""
        from joshpy.sweep import _check_export_path_safety

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job = self._make_job(src)
            cli = MagicMock()
            cli.inspect_exports.return_value = self._make_export_paths(
                protocol="minio", path="/bucket/output_{replicate}.csv",
            )
            registry = MagicMock()
            registry.get_runs_for_hash.return_value = []  # no prior runs

            _check_export_path_safety(cli, self._make_job_set([job]), registry)
            registry.get_runs_for_hash.assert_called_once_with("abcdef012345")

    def test_minio_with_prior_runs_raises(self):
        """Plain minio template + prior run for hash → SweepCollisionError."""
        from joshpy.sweep import _check_export_path_safety, SweepCollisionError

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job = self._make_job(src, run_hash="hashAAA111")
            cli = MagicMock()
            cli.inspect_exports.return_value = self._make_export_paths(
                protocol="minio", path="/bucket/output_{replicate}.csv",
            )
            registry = MagicMock()
            registry.get_runs_for_hash.return_value = [MagicMock()]

            with self.assertRaises(SweepCollisionError) as ctx:
                _check_export_path_safety(cli, self._make_job_set([job]), registry)
            err = ctx.exception
            self.assertEqual(len(err.conflicts), 1)
            conflict_job, _template, priors = err.conflicts[0]
            self.assertEqual(conflict_job.run_hash, "hashAAA111")
            self.assertEqual(len(priors), 1)
            # Error message should be actionable.
            msg = str(err)
            self.assertIn("hashAAA111", msg)
            self.assertIn("{timestamp}", msg)
            self.assertIn("{run_hash}", msg)
            self.assertIn("force=True", msg)

    def test_cache_populated_and_reused(self):
        """Multiple jobs sharing (source, simulation) call inspect_exports once."""
        from joshpy.sweep import _check_export_path_safety

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job1 = self._make_job(src, run_hash="h1")
            job2 = self._make_job(src, run_hash="h2")
            cli = MagicMock()
            cli.inspect_exports.return_value = self._make_export_paths(
                protocol="minio", path="/bucket/{timestamp}/output_{replicate}.csv",
            )
            registry = MagicMock()
            registry.get_runs_for_hash.return_value = []

            cache = _check_export_path_safety(
                cli, self._make_job_set([job1, job2]), registry,
            )

            cli.inspect_exports.assert_called_once()  # cached for second job
            self.assertEqual(len(cache), 1)
            self.assertIn((str(src), "Main"), cache)

    def test_external_cache_passthrough(self):
        """Pre-populated cache bypasses inspect_exports entirely."""
        from joshpy.sweep import _check_export_path_safety

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            job = self._make_job(src)
            pre = self._make_export_paths(
                protocol="minio", path="/bucket/{timestamp}/output_{replicate}.csv",
            )
            cache: dict = {(str(src), "Main"): pre}
            cli = MagicMock()
            registry = MagicMock()
            registry.get_runs_for_hash.return_value = []

            _check_export_path_safety(
                cli, self._make_job_set([job]), registry,
                export_paths_cache=cache,
            )
            cli.inspect_exports.assert_not_called()

    # --- Integration through SweepManager.run --------------------------------

    def _build_manager_with_batch_remote(self, tmp: Path):
        """Construct a SweepManager wired to mock CLI/registry with batch-remote target."""
        src = tmp / "sim.josh"
        src.write_text("simulation Main { }")
        job = self._make_job(src, run_hash="hashCOLLIDE0")
        job_set = self._make_job_set([job])
        job_set.total_jobs = 1
        job_set.total_replicates = 2

        cli = MagicMock()
        cli.inspect_exports.return_value = self._make_export_paths(
            protocol="minio", path="/bucket/output_{replicate}.csv",
        )
        registry = MagicMock()
        registry.get_runs_for_hash.return_value = [MagicMock()]

        config = MagicMock()
        config.sweep = None  # non-adaptive

        manager = SweepManager(
            config=config,
            registry=registry,
            cli=cli,
            job_set=job_set,
            session_id="s1",
            _batch_remote_target="gke-test",
        )
        return manager, cli

    def test_run_raises_collision_by_default(self):
        """Manager.run() with batch-remote + plain path + prior run → raises."""
        from joshpy.sweep import SweepCollisionError

        with tempfile.TemporaryDirectory() as tmp:
            manager, _cli = self._build_manager_with_batch_remote(Path(tmp))
            with self.assertRaises(SweepCollisionError):
                manager.run(quiet=True)

    @patch("joshpy.sweep.run_sweep")
    def test_run_force_true_bypasses(self, mock_run_sweep):
        """force=True bypasses the collision check."""
        mock_run_sweep.return_value = MagicMock(
            succeeded=1, failed=0, run_ids={}, total_rows=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager, _cli = self._build_manager_with_batch_remote(Path(tmp))
            manager.run(force=True, quiet=True)
            mock_run_sweep.assert_called_once()

    @patch("joshpy.sweep.run_sweep")
    def test_run_dry_run_bypasses(self, mock_run_sweep):
        """dry_run=True skips the check (nothing dispatched anyway)."""
        mock_run_sweep.return_value = MagicMock(
            succeeded=0, failed=0, run_ids={}, total_rows=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager, _cli = self._build_manager_with_batch_remote(Path(tmp))
            manager.run(dry_run=True, quiet=True)
            mock_run_sweep.assert_called_once()

    @patch("joshpy.sweep.run_sweep")
    def test_run_local_does_not_check(self, mock_run_sweep):
        """Non-batch-remote (local) sweeps skip the collision check entirely."""
        mock_run_sweep.return_value = MagicMock(
            succeeded=1, failed=0, run_ids={}, total_rows=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager, cli = self._build_manager_with_batch_remote(Path(tmp))
            manager._batch_remote_target = None  # disable batch-remote
            # Should not even call inspect_exports for the collision check
            manager.run(quiet=True)
            cli.inspect_exports.assert_not_called()


class TestApplyCollisionPolicy(unittest.TestCase):
    """Tests for _apply_collision_policy pure function."""

    def test_empty_existing_dispatches_full(self):
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("pool", set(), n_requested=10)
        self.assertEqual(action.action, "dispatch")
        self.assertEqual(action.replicate_start, 0)
        self.assertEqual(action.replicates, 10)

    def test_fail_with_existing_returns_fail(self):
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("fail", {0, 1, 2}, n_requested=5)
        self.assertEqual(action.action, "fail")
        self.assertEqual(action.existing, frozenset({0, 1, 2}))

    def test_fail_with_empty_still_dispatches(self):
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("fail", set(), n_requested=5)
        self.assertEqual(action.action, "dispatch")

    def test_skip_with_any_existing_returns_skip(self):
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("skip", {0}, n_requested=10)
        self.assertEqual(action.action, "skip")

    def test_skip_with_empty_dispatches(self):
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("skip", set(), n_requested=5)
        self.assertEqual(action.action, "dispatch")

    def test_pool_fills_gap(self):
        """Pool dispatches the missing replicates starting at max(existing)+1."""
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("pool", {0, 1, 2, 3, 4}, n_requested=10)
        self.assertEqual(action.action, "dispatch")
        self.assertEqual(action.replicate_start, 5)
        self.assertEqual(action.replicates, 5)  # 10 - 5 remaining

    def test_pool_complete_skips(self):
        """Pool skips when existing already covers requested count."""
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("pool", {0, 1, 2, 3, 4}, n_requested=5)
        self.assertEqual(action.action, "skip")

    def test_pool_over_complete_skips(self):
        """Pool with existing > requested still skips (idempotent)."""
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("pool", {0, 1, 2, 3, 4, 5}, n_requested=5)
        self.assertEqual(action.action, "skip")

    def test_pool_sparse_existing_uses_max(self):
        """Pool uses max(existing)+1, not count(existing), as the offset."""
        from joshpy.sweep import _apply_collision_policy

        # {0, 7} — max is 7, so next is 8
        action = _apply_collision_policy("pool", {0, 7}, n_requested=10)
        self.assertEqual(action.action, "dispatch")
        self.assertEqual(action.replicate_start, 8)
        self.assertEqual(action.replicates, 2)

    def test_unknown_policy_raises(self):
        from joshpy.sweep import _apply_collision_policy

        with self.assertRaises(ValueError):
            _apply_collision_policy("replace", set(), n_requested=5)

    def test_overwrite_with_existing_dispatches_full(self):
        """overwrite ignores prior outputs and dispatches 0..N-1 over them."""
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("overwrite", {0, 1, 2}, n_requested=5)
        self.assertEqual(action.action, "dispatch")
        self.assertEqual(action.replicate_start, 0)
        self.assertEqual(action.replicates, 5)

    def test_overwrite_with_empty_dispatches_full(self):
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("overwrite", set(), n_requested=5)
        self.assertEqual(action.action, "dispatch")
        self.assertEqual(action.replicate_start, 0)
        self.assertEqual(action.replicates, 5)

    def test_overwrite_shrinking_sweep_leaves_orphans(self):
        """Shrinking sweep (old=10, new=5): dispatch only 0..4; 5..9 stay orphaned."""
        from joshpy.sweep import _apply_collision_policy

        action = _apply_collision_policy("overwrite", set(range(10)), n_requested=5)
        self.assertEqual(action.action, "dispatch")
        self.assertEqual(action.replicates, 5)
        # The action doesn't track orphans — that's a known limitation,
        # documented in with_collision_policy() docstring.


class TestListExistingReplicatesMinio(unittest.TestCase):
    """Tests for _list_existing_replicates_minio (MinIO listing via DuckDB glob)."""

    def _export_info(self, path: str, protocol: str = "minio", host: str = "bucket"):
        from joshpy.cli import ExportFileInfo
        return ExportFileInfo(
            raw=f"{protocol}://{host}{path}",
            protocol=protocol,
            host=host,
            path=path,
            file_type="csv",
        )

    def test_template_without_replicate_placeholder_returns_empty(self):
        from joshpy.sweep import _list_existing_replicates_minio

        info = self._export_info("/prefix/single_output.csv")
        result = _list_existing_replicates_minio(
            MagicMock(), MagicMock(), info, {}, quiet=True,
        )
        self.assertEqual(result, set())

    def test_template_with_timestamp_returns_empty(self):
        """{timestamp} signals per-dispatch isolation; don't pool across runs."""
        from joshpy.sweep import _list_existing_replicates_minio

        info = self._export_info("/prefix/{timestamp}/output_{replicate}.csv")
        result = _list_existing_replicates_minio(
            MagicMock(), MagicMock(), info, {}, quiet=True,
        )
        self.assertEqual(result, set())

    def test_template_with_unknown_placeholder_uses_wildcard(self):
        """Unknown placeholders (e.g. {step}) become wildcards, replicate still extracted."""
        from joshpy.sweep import _list_existing_replicates_minio

        info = self._export_info("/prefix/output_{step}_{replicate}.csv")
        mock_registry = MagicMock()
        # Multiple "step" variants per replicate — all dedupe to the same replicate index
        mock_registry.conn.execute.return_value.fetchall.return_value = [
            ("s3://bucket/prefix/output_0_0.csv",),
            ("s3://bucket/prefix/output_5_0.csv",),  # same replicate, different step
            ("s3://bucket/prefix/output_0_1.csv",),
            ("s3://bucket/prefix/output_5_1.csv",),
        ]
        with patch(
            "joshpy.sweep._configure_minio_access", return_value=("bucket", None),
        ):
            result = _list_existing_replicates_minio(
                MagicMock(), mock_registry, info, {}, quiet=True,
            )
        self.assertEqual(result, {0, 1})
        # Glob pattern should have BOTH {step} and {replicate} as *
        glob_arg = mock_registry.conn.execute.call_args[0][1][0]
        self.assertEqual(
            glob_arg, "s3://bucket/prefix/output_*_*.csv",
        )

    def test_run_hash_resolved_via_known_vars(self):
        """{run_hash} in the template is resolved via known_vars."""
        from joshpy.sweep import _list_existing_replicates_minio

        info = self._export_info("/prefix/{run_hash}/output_{replicate}.csv")
        mock_registry = MagicMock()
        mock_registry.conn.execute.return_value.fetchall.return_value = [
            ("s3://bucket/prefix/abc123/output_0.csv",),
            ("s3://bucket/prefix/abc123/output_1.csv",),
            ("s3://bucket/prefix/abc123/output_7.csv",),
        ]
        with patch(
            "joshpy.sweep._configure_minio_access", return_value=("bucket", None),
        ):
            result = _list_existing_replicates_minio(
                MagicMock(), mock_registry, info,
                known_vars={"run_hash": "abc123"}, quiet=True,
            )
        self.assertEqual(result, {0, 1, 7})

        # The glob pattern should have {run_hash} resolved but {replicate} as *
        call_args = mock_registry.conn.execute.call_args
        self.assertIn("s3://bucket/prefix/abc123/output_*.csv", call_args[0][1])

    def test_glob_failure_returns_empty(self):
        """DuckDB glob exception is swallowed and returns empty set."""
        from joshpy.sweep import _list_existing_replicates_minio

        info = self._export_info("/prefix/output_{replicate}.csv")
        mock_registry = MagicMock()
        mock_registry.conn.execute.side_effect = RuntimeError("S3 credentials missing")
        with patch(
            "joshpy.sweep._configure_minio_access", return_value=("bucket", None),
        ):
            result = _list_existing_replicates_minio(
                MagicMock(), mock_registry, info, {}, quiet=True,
            )
        self.assertEqual(result, set())

    def test_ignores_non_matching_names(self):
        """Files not matching the prefix/suffix regex are ignored."""
        from joshpy.sweep import _list_existing_replicates_minio

        info = self._export_info("/prefix/output_{replicate}.csv")
        mock_registry = MagicMock()
        mock_registry.conn.execute.return_value.fetchall.return_value = [
            ("s3://bucket/prefix/output_3.csv",),
            ("s3://bucket/prefix/output_something.csv",),  # not a number
            ("s3://bucket/prefix/unrelated.csv",),
            ("s3://bucket/prefix/output_12.csv",),
        ]
        with patch(
            "joshpy.sweep._configure_minio_access", return_value=("bucket", None),
        ):
            result = _list_existing_replicates_minio(
                MagicMock(), mock_registry, info, {}, quiet=True,
            )
        self.assertEqual(result, {3, 12})


class TestCollisionPolicyBuilder(unittest.TestCase):
    """Tests for SweepManagerBuilder.with_collision_policy."""

    def test_default_is_fail(self):
        from joshpy.jobs import JobConfig
        from joshpy.sweep import SweepManagerBuilder

        config = JobConfig(
            source_path=Path("/fake.josh"), simulation="Main", replicates=1,
        )
        builder = SweepManagerBuilder(config)
        self.assertEqual(builder._collision_policy, "fail")

    def test_with_collision_policy_stores(self):
        from joshpy.jobs import JobConfig
        from joshpy.sweep import SweepManagerBuilder

        config = JobConfig(
            source_path=Path("/fake.josh"), simulation="Main", replicates=1,
        )
        builder = SweepManagerBuilder(config).with_collision_policy("pool")
        self.assertEqual(builder._collision_policy, "pool")

    def test_invalid_policy_raises(self):
        from joshpy.jobs import JobConfig
        from joshpy.sweep import SweepManagerBuilder

        config = JobConfig(
            source_path=Path("/fake.josh"), simulation="Main", replicates=1,
        )
        with self.assertRaises(ValueError) as ctx:
            SweepManagerBuilder(config).with_collision_policy("destroy-all")
        self.assertIn("destroy-all", str(ctx.exception))
        self.assertIn("must be one of", str(ctx.exception))

    def test_with_collision_policy_accepts_overwrite(self):
        from joshpy.jobs import JobConfig
        from joshpy.sweep import SweepManagerBuilder

        config = JobConfig(
            source_path=Path("/fake.josh"), simulation="Main", replicates=1,
        )
        builder = SweepManagerBuilder(config).with_collision_policy("overwrite")
        self.assertEqual(builder._collision_policy, "overwrite")

    def test_builder_default_propagates_to_manager(self):
        from joshpy.jobs import JobConfig
        from joshpy.sweep import SweepManagerBuilder

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim.josh"
            src.write_text("x")
            config = JobConfig(
                source_path=src, simulation="Main", replicates=1,
            )
            builder = SweepManagerBuilder(config).with_collision_policy("skip")
            manager = builder.build()
            self.assertEqual(manager._collision_policy, "skip")
            manager.close()


class TestCollisionPolicyInRunSweep(unittest.TestCase):
    """Integration tests: collision policy applied inside run_sweep batch-remote."""

    def _make_job(self, tmp: Path, run_hash: str = "hashPool") -> "ExpandedJob":
        from joshpy.jobs import ExpandedJob
        src = tmp / "sim.josh"
        src.write_text("start simulation Main\nend simulation\n")
        return ExpandedJob(
            config_content="x = 1 count",
            config_path=tmp / "config.jshc",
            config_name="config.jshc",
            run_hash=run_hash,
            parameters={},
            simulation="Main",
            replicates=10,
            source_path=src,
            custom_tags={"run_hash": run_hash},
        )

    def _mock_cli(self, existing_files: list[str]):
        """Build a MockCLI that returns a minio:// patch export and the given existing files."""
        from joshpy.cli import CLIResult, ExportFileInfo, ExportPaths
        cli = MagicMock()
        cli.stage_to_minio.return_value = CLIResult(
            exit_code=0, stdout="", stderr="", command=["stageToMinio"],
        )
        cli.batch_remote.return_value = CLIResult(
            exit_code=0, stdout="", stderr="", command=["batchRemote"],
        )
        cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw="minio://bucket/prefix/output_{replicate}.csv",
                    protocol="minio",
                    host="bucket",
                    path="/prefix/output_{replicate}.csv",
                    file_type="csv",
                ),
            },
            debug_files={},
        )
        # Registry with glob() returning existing files
        mock_registry = MagicMock()
        mock_registry.conn.execute.return_value.fetchall.return_value = [
            (f,) for f in existing_files
        ]
        return cli, mock_registry

    @patch("joshpy.sweep._configure_minio_access", return_value=("bucket", None))
    def test_pool_policy_dispatches_with_offset(self, _mock_cfg):
        from joshpy.jobs import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            job = self._make_job(Path(tmp))
            job_set = MagicMock(
                total_jobs=1, total_replicates=10,
                __iter__=lambda self: iter([job]),
            )
            existing = [f"s3://bucket/prefix/output_{i}.csv" for i in range(5)]
            cli, registry = self._mock_cli(existing)

            run_sweep(
                cli, job_set,
                registry=registry,
                session_id="s1",
                batch_remote=True,
                target="gke-test",
                collision_policy="pool",
                quiet=True,
                manage_status=False,
            )

            # stage_to_minio and batch_remote called once each (we dispatched the gap)
            cli.stage_to_minio.assert_called_once()
            cli.batch_remote.assert_called_once()
            # The BatchRemoteConfig should have replicate_start=5, replicates=5
            br_config = cli.batch_remote.call_args[0][0]
            self.assertEqual(br_config.replicate_start, 5)
            self.assertEqual(br_config.replicates, 5)

    @patch("joshpy.sweep._configure_minio_access", return_value=("bucket", None))
    def test_skip_policy_skips_staging(self, _mock_cfg):
        from joshpy.jobs import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            job = self._make_job(Path(tmp))
            job_set = MagicMock(
                total_jobs=1, total_replicates=10,
                __iter__=lambda self: iter([job]),
            )
            existing = [f"s3://bucket/prefix/output_{i}.csv" for i in range(3)]
            cli, registry = self._mock_cli(existing)

            result = run_sweep(
                cli, job_set,
                registry=registry,
                session_id="s1",
                batch_remote=True,
                target="gke-test",
                collision_policy="skip",
                quiet=True,
                manage_status=False,
            )

            # No stage / no dispatch — skip policy is idempotent no-op
            cli.stage_to_minio.assert_not_called()
            cli.batch_remote.assert_not_called()
            # And we still count the skipped job as succeeded
            self.assertEqual(result.succeeded, 1)
            self.assertEqual(result.failed, 0)

    @patch("joshpy.sweep._configure_minio_access", return_value=("bucket", None))
    def test_pool_policy_complete_skips(self, _mock_cfg):
        """Pool with existing >= requested skips (idempotent)."""
        from joshpy.jobs import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            job = self._make_job(Path(tmp))
            job_set = MagicMock(
                total_jobs=1, total_replicates=10,
                __iter__=lambda self: iter([job]),
            )
            existing = [f"s3://bucket/prefix/output_{i}.csv" for i in range(10)]
            cli, registry = self._mock_cli(existing)

            run_sweep(
                cli, job_set,
                registry=registry,
                session_id="s1",
                batch_remote=True,
                target="gke-test",
                collision_policy="pool",
                quiet=True,
                manage_status=False,
            )

            cli.stage_to_minio.assert_not_called()
            cli.batch_remote.assert_not_called()

    @patch("joshpy.sweep._configure_minio_access", return_value=("bucket", None))
    def test_overwrite_policy_dispatches_full_over_existing(self, _mock_cfg):
        """overwrite dispatches 0..N-1 even when prior replicates exist on MinIO."""
        from joshpy.jobs import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            job = self._make_job(Path(tmp))
            job_set = MagicMock(
                total_jobs=1, total_replicates=10,
                __iter__=lambda self: iter([job]),
            )
            existing = [f"s3://bucket/prefix/output_{i}.csv" for i in range(5)]
            cli, registry = self._mock_cli(existing)

            run_sweep(
                cli, job_set,
                registry=registry,
                session_id="s1",
                batch_remote=True,
                target="gke-test",
                collision_policy="overwrite",
                quiet=True,
                manage_status=False,
            )

            cli.stage_to_minio.assert_called_once()
            cli.batch_remote.assert_called_once()
            br_config = cli.batch_remote.call_args[0][0]
            # overwrite dispatches the full range from 0
            self.assertEqual(br_config.replicate_start, 0)
            self.assertEqual(br_config.replicates, 10)

    @patch("joshpy.sweep._configure_minio_access", return_value=("bucket", None))
    def test_fail_policy_in_run_sweep_raises(self, _mock_cfg):
        """run_sweep called directly with policy='fail' + existing → SweepCollisionError."""
        from joshpy.jobs import run_sweep
        from joshpy.sweep import SweepCollisionError

        with tempfile.TemporaryDirectory() as tmp:
            job = self._make_job(Path(tmp))
            job_set = MagicMock(
                total_jobs=1, total_replicates=10,
                __iter__=lambda self: iter([job]),
            )
            existing = [f"s3://bucket/prefix/output_{i}.csv" for i in range(3)]
            cli, registry = self._mock_cli(existing)

            with self.assertRaises(SweepCollisionError):
                run_sweep(
                    cli, job_set,
                    registry=registry,
                    session_id="s1",
                    batch_remote=True,
                    target="gke-test",
                    collision_policy="fail",
                    quiet=True,
                    manage_status=False,
                )


if __name__ == "__main__":
    unittest.main()
