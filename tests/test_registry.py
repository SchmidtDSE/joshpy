"""Unit tests for the registry module."""

import unittest
import tempfile
from pathlib import Path

# Check if duckdb is available
try:
    import duckdb  # noqa: F401

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


def _make_config(simulation: str = "TestSim", **kwargs) -> "JobConfig":
    """Create a minimal JobConfig for testing.
    
    This helper creates a JobConfig with just enough attributes for create_session().
    """
    from joshpy.jobs import JobConfig
    return JobConfig(simulation=simulation, **kwargs)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestRunRegistry(unittest.TestCase):
    """Tests for RunRegistry class initialization and basic operations."""

    def test_create_in_memory(self):
        """Should create in-memory database."""
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        self.assertIsNotNone(registry.conn)
        registry.close()

    def test_context_manager(self):
        """Should work as context manager."""
        from joshpy.registry import RunRegistry

        with RunRegistry(":memory:") as registry:
            self.assertIsNotNone(registry.conn)
        # After close, _conn is None but conn property still works (returns None)
        self.assertIsNone(registry._conn)

    def test_schema_created(self):
        """Schema tables should be created on init."""
        from joshpy.registry import RunRegistry

        with RunRegistry(":memory:") as registry:
            # Check tables exist
            tables = registry.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            table_names = {t[0] for t in tables}
            self.assertIn("sweep_sessions", table_names)
            self.assertIn("job_configs", table_names)
            self.assertIn("job_runs", table_names)
            self.assertIn("run_outputs", table_names)

    def test_spatial_disabled(self):
        """Should work without spatial extension."""
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:", enable_spatial=False)
        self.assertIsNotNone(registry.conn)
        registry.close()

    def test_enable_spatial_default(self):
        """enable_spatial should default to True."""
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        self.assertTrue(registry.enable_spatial)
        registry.close()


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestSessionManagement(unittest.TestCase):
    """Tests for session CRUD operations."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_create_session_returns_id(self):
        """create_session should return a session ID."""
        config = _make_config(simulation="TestSim")
        session_id = self.registry.create_session(
            config=config,
            experiment_name="test_experiment",
        )
        self.assertIsInstance(session_id, str)
        self.assertTrue(len(session_id) > 0)

    def test_create_session_with_all_fields(self):
        """create_session should store all provided fields."""
        from joshpy.jobs import JobConfig
        
        config = JobConfig(
            simulation="JoshuaTreeSim",
            template_path=Path("/path/to/template.j2"),
        )
        session_id = self.registry.create_session(
            config=config,
            experiment_name="sensitivity_analysis",
        )

        session = self.registry.get_session(session_id)
        self.assertEqual(session.experiment_name, "sensitivity_analysis")
        self.assertEqual(session.simulation, "JoshuaTreeSim")
        # total_jobs and total_replicates are now computed from actual data
        self.assertIsNone(session.total_jobs)
        self.assertIsNone(session.total_replicates)
        self.assertEqual(session.template_path, "/path/to/template.j2")
        # template_hash is no longer used
        self.assertIsNone(session.template_hash)
        # metadata now auto-contains job_config
        self.assertIn("job_config", session.metadata)
        self.assertEqual(session.status, "pending")

    def test_get_session_not_found(self):
        """get_session should return None for non-existent session."""
        result = self.registry.get_session("nonexistent-id")
        self.assertIsNone(result)

    def test_update_session_status(self):
        """update_session_status should change session status."""
        config = _make_config()
        session_id = self.registry.create_session(config=config, experiment_name="test")

        self.registry.update_session_status(session_id, "running")
        session = self.registry.get_session(session_id)
        self.assertEqual(session.status, "running")

        self.registry.update_session_status(session_id, "completed")
        session = self.registry.get_session(session_id)
        self.assertEqual(session.status, "completed")

    def test_list_sessions_empty(self):
        """list_sessions should return empty list when no sessions."""
        sessions = self.registry.list_sessions()
        self.assertEqual(sessions, [])

    def test_list_sessions_all(self):
        """list_sessions should return all sessions."""
        self.registry.create_session(config=_make_config(), experiment_name="exp1")
        self.registry.create_session(config=_make_config(), experiment_name="exp2")
        self.registry.create_session(config=_make_config(), experiment_name="exp3")

        sessions = self.registry.list_sessions()
        self.assertEqual(len(sessions), 3)

    def test_list_sessions_by_experiment_name(self):
        """list_sessions should filter by experiment_name."""
        self.registry.create_session(config=_make_config(), experiment_name="sensitivity")
        self.registry.create_session(config=_make_config(), experiment_name="sensitivity")
        self.registry.create_session(config=_make_config(), experiment_name="calibration")

        sensitivity_sessions = self.registry.list_sessions(experiment_name="sensitivity")
        self.assertEqual(len(sensitivity_sessions), 2)
        for s in sensitivity_sessions:
            self.assertEqual(s.experiment_name, "sensitivity")

    def test_list_sessions_with_limit(self):
        """list_sessions should respect limit parameter."""
        for i in range(10):
            self.registry.create_session(config=_make_config(), experiment_name=f"exp{i}")

        sessions = self.registry.list_sessions(limit=5)
        self.assertEqual(len(sessions), 5)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestConfigRegistration(unittest.TestCase):
    """Tests for config registration and lookup."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config(), experiment_name="test")

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_register_run(self):
        """register_run should store config."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="abc123def456",
            josh_path="/path/to/sim.josh",
            config_content="survivalProbAdult = 85 %",
            file_mappings={"data": {"path": "/path/to/data.jshd", "hash": "abc123"}},
            parameters={"survivalProbAdult": 85},
        )

        config = self.registry.get_config_by_hash("abc123def456")
        self.assertIsNotNone(config)
        self.assertEqual(config.run_hash, "abc123def456")
        self.assertEqual(config.session_id, self.session_id)
        self.assertEqual(config.josh_path, "/path/to/sim.josh")
        self.assertEqual(config.config_content, "survivalProbAdult = 85 %")
        self.assertEqual(config.file_mappings, {"data": {"path": "/path/to/data.jshd", "hash": "abc123"}})
        self.assertEqual(config.parameters, {"survivalProbAdult": 85})

    def test_register_run_duplicate_hash(self):
        """register_run should ignore duplicate hashes."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="abc123",
            josh_path="/path/to/sim.josh",
            config_content="original content",
            file_mappings=None,
            parameters={"x": 1},
        )

        # Register same hash again - should not error
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="abc123",
            josh_path="/path/to/sim.josh",
            config_content="different content",
            file_mappings=None,
            parameters={"x": 2},
        )

        # Original should be preserved
        config = self.registry.get_config_by_hash("abc123")
        self.assertEqual(config.config_content, "original content")
        self.assertEqual(config.parameters, {"x": 1})

    def test_get_config_by_hash_not_found(self):
        """get_config_by_hash should return None for non-existent hash."""
        result = self.registry.get_config_by_hash("nonexistent")
        self.assertIsNone(result)

    def test_get_configs_for_session(self):
        """get_configs_for_session should return all configs for session."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash1",
            josh_path="/path/to/sim.josh",
            config_content="content1",
            file_mappings=None,
            parameters={"x": 1},
        )
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash2",
            josh_path="/path/to/sim.josh",
            config_content="content2",
            file_mappings=None,
            parameters={"x": 2},
        )

        configs = self.registry.get_configs_for_session(self.session_id)
        self.assertEqual(len(configs), 2)
        hashes = {c.run_hash for c in configs}
        self.assertEqual(hashes, {"hash1", "hash2"})

    def test_get_configs_for_session_empty(self):
        """get_configs_for_session should return empty list for session with no configs."""
        configs = self.registry.get_configs_for_session(self.session_id)
        self.assertEqual(configs, [])


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestRunTracking(unittest.TestCase):
    """Tests for run start/complete tracking."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config(), experiment_name="test")
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="config123",
            josh_path="/path/to/sim.josh",
            config_content="test content",
            file_mappings=None,
            parameters={"x": 1},
        )

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_start_run(self):
        """start_run should create a run record."""
        run_id = self.registry.start_run(
            run_hash="config123",
            session_id=self.session_id,
            replicate=0,
            output_path="/output/path",
        )

        self.assertIsInstance(run_id, str)
        run = self.registry.get_run(run_id)
        self.assertIsNotNone(run)
        self.assertEqual(run.run_hash, "config123")
        self.assertEqual(run.replicate, 0)
        self.assertEqual(run.output_path, "/output/path")
        self.assertIsNotNone(run.started_at)
        self.assertIsNone(run.completed_at)
        self.assertIsNone(run.exit_code)

    def test_complete_run_success(self):
        """complete_run should update run with exit code."""
        run_id = self.registry.start_run(run_hash="config123", session_id=self.session_id)

        self.registry.complete_run(run_id=run_id, exit_code=0)

        run = self.registry.get_run(run_id)
        self.assertIsNotNone(run.completed_at)
        self.assertEqual(run.exit_code, 0)
        self.assertIsNone(run.error_message)

    def test_complete_run_failure(self):
        """complete_run should store error message on failure."""
        run_id = self.registry.start_run(run_hash="config123", session_id=self.session_id)

        self.registry.complete_run(
            run_id=run_id,
            exit_code=1,
            error_message="Simulation failed: out of memory",
        )

        run = self.registry.get_run(run_id)
        self.assertEqual(run.exit_code, 1)
        self.assertEqual(run.error_message, "Simulation failed: out of memory")

    def test_get_run_not_found(self):
        """get_run should return None for non-existent run."""
        result = self.registry.get_run("nonexistent-id")
        self.assertIsNone(result)

    def test_get_runs_for_hash(self):
        """get_runs_for_hash should return all runs for a run hash."""
        run1 = self.registry.start_run(run_hash="config123", session_id=self.session_id, replicate=0)
        run2 = self.registry.start_run(run_hash="config123", session_id=self.session_id, replicate=1)
        run3 = self.registry.start_run(run_hash="config123", session_id=self.session_id, replicate=2)

        runs = self.registry.get_runs_for_hash("config123")
        self.assertEqual(len(runs), 3)
        run_ids = {r.run_id for r in runs}
        self.assertEqual(run_ids, {run1, run2, run3})

    def test_run_with_metadata(self):
        """start_run should store metadata."""
        run_id = self.registry.start_run(
            run_hash="config123",
            session_id=self.session_id,
            metadata={"host": "node01", "pid": 12345},
        )

        run = self.registry.get_run(run_id)
        self.assertEqual(run.metadata, {"host": "node01", "pid": 12345})


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestOutputTracking(unittest.TestCase):
    """Tests for output file registration."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config(), experiment_name="test")
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="config123",
            josh_path="/path/to/sim.josh",
            config_content="test content",
            file_mappings=None,
            parameters={},
        )
        self.run_id = self.registry.start_run(run_hash="config123", session_id=self.session_id)

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_register_output(self):
        """register_output should create output record."""
        output_id = self.registry.register_output(
            run_id=self.run_id,
            output_type="csv",
            file_path="/output/results.csv",
            file_size=1024,
            row_count=100,
        )

        self.assertIsInstance(output_id, str)

        # Verify by querying directly
        result = self.registry.conn.execute(
            "SELECT * FROM run_outputs WHERE output_id = ?", [output_id]
        ).fetchone()
        self.assertIsNotNone(result)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestQueryMethods(unittest.TestCase):
    """Tests for query methods."""

    def setUp(self):
        """Create registry with test data."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(
            config=_make_config(simulation="TestSim"),
            experiment_name="test",
        )

        # Create configs with different parameters
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash1",
            josh_path="/path/to/sim.josh",
            config_content="x=1, y=a",
            file_mappings=None,
            parameters={"x": 1, "y": "a"},
        )
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash2",
            josh_path="/path/to/sim.josh",
            config_content="x=2, y=a",
            file_mappings=None,
            parameters={"x": 2, "y": "a"},
        )
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash3",
            josh_path="/path/to/sim.josh",
            config_content="x=1, y=b",
            file_mappings=None,
            parameters={"x": 1, "y": "b"},
        )

        # Create runs
        run1 = self.registry.start_run(run_hash="hash1", session_id=self.session_id)
        self.registry.complete_run(run1, exit_code=0)

        run2 = self.registry.start_run(run_hash="hash2", session_id=self.session_id)
        self.registry.complete_run(run2, exit_code=1, error_message="failed")

        run3 = self.registry.start_run(run_hash="hash3", session_id=self.session_id)
        self.registry.complete_run(run3, exit_code=0)

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_get_runs_by_parameters_no_filter(self):
        """get_runs_by_parameters with no filters returns all runs."""
        runs = self.registry.get_runs_by_parameters()
        self.assertEqual(len(runs), 3)

    def test_get_runs_by_parameters_single_filter(self):
        """get_runs_by_parameters with single filter."""
        runs = self.registry.get_runs_by_parameters(x=1)
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertEqual(run["parameters"]["x"], 1)

    def test_get_runs_by_parameters_multiple_filters(self):
        """get_runs_by_parameters with multiple filters."""
        runs = self.registry.get_runs_by_parameters(x=1, y="a")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_hash"], "hash1")

    def test_get_runs_by_parameters_no_match(self):
        """get_runs_by_parameters returns empty list when no match."""
        runs = self.registry.get_runs_by_parameters(x=999)
        self.assertEqual(runs, [])

    def test_get_session_summary(self):
        """get_session_summary returns correct counts."""
        summary = self.registry.get_session_summary(self.session_id)

        self.assertEqual(summary.session_id, self.session_id)
        self.assertEqual(summary.experiment_name, "test")
        self.assertEqual(summary.simulation, "TestSim")
        # total_jobs is computed from registered configs (3)
        self.assertEqual(summary.total_jobs, 3)
        # total_replicates is computed from actual runs (3)
        self.assertEqual(summary.total_replicates, 3)
        self.assertEqual(summary.runs_completed, 3)
        self.assertEqual(summary.runs_succeeded, 2)
        self.assertEqual(summary.runs_failed, 1)
        self.assertEqual(summary.runs_pending, 0)

    def test_get_session_summary_not_found(self):
        """get_session_summary returns None for non-existent session."""
        result = self.registry.get_session_summary("nonexistent")
        self.assertIsNone(result)

    def test_get_session_summary_with_pending(self):
        """get_session_summary correctly counts pending runs."""
        # Add a run that's started but not completed
        self.registry.start_run(run_hash="hash1", session_id=self.session_id)

        summary = self.registry.get_session_summary(self.session_id)
        self.assertEqual(summary.runs_completed, 3)  # Only the original 3
        self.assertEqual(summary.runs_pending, 1)  # The new incomplete one


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestExportResultsDf(unittest.TestCase):
    """Tests for export_results_df method."""

    def setUp(self):
        """Create registry with test data."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config(), experiment_name="test")

        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash1",
            josh_path="/path/to/sim.josh",
            config_content="content",
            file_mappings=None,
            parameters={"survival": 85, "growth": 1.5},
        )
        run_id = self.registry.start_run(run_hash="hash1", session_id=self.session_id)
        self.registry.complete_run(run_id, exit_code=0)

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_export_results_df_basic(self):
        """export_results_df returns DataFrame with flattened parameters."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = self.registry.export_results_df(self.session_id)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn("run_id", df.columns)
        self.assertIn("run_hash", df.columns)
        self.assertIn("survival", df.columns)
        self.assertIn("growth", df.columns)
        self.assertEqual(df.iloc[0]["survival"], 85)
        self.assertEqual(df.iloc[0]["growth"], 1.5)

    def test_export_results_df_empty(self):
        """export_results_df returns empty DataFrame for session with no runs."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        empty_session = self.registry.create_session(config=_make_config(), experiment_name="empty")
        df = self.registry.export_results_df(empty_session)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestRegistryCallback(unittest.TestCase):
    """Tests for RegistryCallback integration helper."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config(), experiment_name="test")
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="test_hash",
            josh_path="/path/to/sim.josh",
            config_content="test",
            file_mappings=None,
            parameters={"x": 1},
        )

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_callback_records_success(self):
        """Callback should record successful run."""
        from joshpy.cli import CLIResult
        from joshpy.jobs import ExpandedJob
        from joshpy.registry import RegistryCallback

        callback = RegistryCallback(self.registry, self.session_id)

        # Create mock job and result
        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/test.jshc"),
            config_name="test",
            run_hash="test_hash",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
        )
        result = CLIResult(
            exit_code=0,
            stdout="success",
            stderr="",
            command=["java", "-jar", "test.jar"],
        )

        # Call the callback's record method
        callback.record(job, result)

        # Verify run was recorded
        runs = self.registry.get_runs_for_hash("test_hash")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].exit_code, 0)
        self.assertIsNone(runs[0].error_message)

    def test_callback_records_failure(self):
        """Callback should record failed run with error message."""
        from joshpy.cli import CLIResult
        from joshpy.jobs import ExpandedJob
        from joshpy.registry import RegistryCallback

        callback = RegistryCallback(self.registry, self.session_id)

        job = ExpandedJob(
            config_content="test",
            config_path=Path("/tmp/test.jshc"),
            config_name="test",
            run_hash="test_hash",
            parameters={"x": 1},
            simulation="Main",
            replicates=1,
        )
        result = CLIResult(
            exit_code=1,
            stdout="",
            stderr="Error: simulation failed",
            command=["java", "-jar", "test.jar"],
        )

        callback.record(job, result)

        runs = self.registry.get_runs_for_hash("test_hash")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].exit_code, 1)
        self.assertEqual(runs[0].error_message, "Error: simulation failed")

    def test_callback_rejects_invalid_types(self):
        """Callback should raise TypeError for invalid arguments."""
        from joshpy.registry import RegistryCallback

        callback = RegistryCallback(self.registry, self.session_id)

        # Should raise TypeError for invalid job type
        with self.assertRaises(TypeError):
            callback.record("not a job", "not a result")

        # No runs should be recorded
        runs = self.registry.get_runs_for_hash("test_hash")
        self.assertEqual(len(runs), 0)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestPersistence(unittest.TestCase):
    """Tests for database persistence."""

    def test_data_persists_after_close(self):
        """Data should persist after closing and reopening."""
        import os
        import tempfile

        from joshpy.registry import RunRegistry

        # Create a temp directory and use a path within it
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.duckdb")

        try:
            # Create and populate
            with RunRegistry(db_path) as registry:
                session_id = registry.create_session(
                    config=_make_config(simulation="TestSim"),
                    experiment_name="persistence_test",
                )
                registry.register_run(
                    session_id=session_id,
                    run_hash="persist_hash",
                    josh_path="/path/to/sim.josh",
                    config_content="persistent content",
                    file_mappings=None,
                    parameters={"key": "value"},
                )

            # Reopen and verify
            with RunRegistry(db_path) as registry:
                sessions = registry.list_sessions(experiment_name="persistence_test")
                self.assertEqual(len(sessions), 1)
                self.assertEqual(sessions[0].experiment_name, "persistence_test")

                config = registry.get_config_by_hash("persist_hash")
                self.assertIsNotNone(config)
                self.assertEqual(config.config_content, "persistent content")
        finally:
            # Cleanup
            import shutil

            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestDataClasses(unittest.TestCase):
    """Tests for data class attributes and defaults."""

    def test_session_info_attributes(self):
        """SessionInfo should have all expected attributes."""
        from datetime import datetime

        from joshpy.registry import SessionInfo

        session = SessionInfo(
            session_id="123",
            experiment_name="test",
            created_at=datetime.now(),
            template_path="/path",
            template_hash="hash",
            simulation="Sim",
            total_jobs=10,
            total_replicates=30,
            status="pending",
            metadata={"key": "value"},
        )
        self.assertEqual(session.session_id, "123")
        self.assertEqual(session.experiment_name, "test")
        self.assertEqual(session.status, "pending")

    def test_config_info_attributes(self):
        """ConfigInfo should have all expected attributes."""
        from datetime import datetime

        from joshpy.registry import ConfigInfo

        config = ConfigInfo(
            run_hash="abc123",
            session_id="session1",
            josh_path="/path/to/sim.josh",
            josh_content="start simulation Main\nend simulation\n",
            config_content="content",
            file_mappings={"data": {"path": "/path/to/data.jshd", "hash": "abc"}},
            parameters={"x": 1},
            label=None,
            created_at=datetime.now(),
        )
        self.assertEqual(config.run_hash, "abc123")
        self.assertEqual(config.parameters, {"x": 1})
        self.assertIn("start simulation", config.josh_content)

    def test_run_info_attributes(self):
        """RunInfo should have all expected attributes."""
        from joshpy.registry import RunInfo

        run = RunInfo(
            run_id="run1",
            run_hash="config1",
            replicate=0,
            started_at=None,
            completed_at=None,
            exit_code=None,
            output_path=None,
            error_message=None,
            metadata=None,
        )
        self.assertEqual(run.run_id, "run1")
        self.assertIsNone(run.exit_code)

    def test_session_summary_attributes(self):
        """SessionSummary should have all expected attributes."""
        from joshpy.registry import SessionSummary

        summary = SessionSummary(
            session_id="123",
            experiment_name="test",
            simulation="Sim",
            status="completed",
            total_jobs=10,
            total_replicates=30,
            runs_completed=30,
            runs_succeeded=28,
            runs_failed=2,
            runs_pending=0,
        )
        self.assertEqual(summary.runs_succeeded, 28)
        self.assertEqual(summary.runs_failed, 2)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestExternalSessionId(unittest.TestCase):
    """Tests for external session ID support in create_session()."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_create_session_with_external_id(self):
        """create_session should use provided session_id."""
        external_id = "my-custom-project-id-123"
        session_id = self.registry.create_session(
            config=_make_config(simulation="TestSim"),
            experiment_name="test_experiment",
            session_id=external_id,
        )

        self.assertEqual(session_id, external_id)

        # Verify we can retrieve it
        session = self.registry.get_session(external_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, external_id)
        self.assertEqual(session.experiment_name, "test_experiment")

    def test_create_session_generates_id_when_none(self):
        """create_session should generate UUID when session_id is None."""
        session_id = self.registry.create_session(
            config=_make_config(),
            experiment_name="test_experiment",
            session_id=None,
        )

        # Should be a valid UUID format (36 chars with dashes)
        self.assertEqual(len(session_id), 36)
        self.assertEqual(session_id.count("-"), 4)

    def test_create_session_external_id_with_metadata(self):
        """create_session with external ID should store job_config in metadata."""
        external_id = "frontend-project-456"
        config = _make_config(simulation="JoshuaTreeSim", replicates=3)

        session_id = self.registry.create_session(
            config=config,
            experiment_name="sensitivity_analysis",
            session_id=external_id,
        )

        self.assertEqual(session_id, external_id)

        session = self.registry.get_session(external_id)
        self.assertEqual(session.experiment_name, "sensitivity_analysis")
        self.assertEqual(session.simulation, "JoshuaTreeSim")
        # metadata now auto-contains job_config
        self.assertIn("job_config", session.metadata)
        self.assertEqual(session.metadata["job_config"]["simulation"], "JoshuaTreeSim")

    def test_create_session_duplicate_external_id_raises(self):
        """create_session with duplicate external ID should raise error."""
        external_id = "duplicate-id"

        # First creation should succeed
        self.registry.create_session(
            config=_make_config(),
            experiment_name="first",
            session_id=external_id,
        )

        # Second creation with same ID should raise
        with self.assertRaises(Exception):  # DuckDB raises constraint violation
            self.registry.create_session(
                config=_make_config(),
                experiment_name="second",
                session_id=external_id,
            )


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestSessionInfoJobConfig(unittest.TestCase):
    """Tests for SessionInfo.job_config property."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_job_config_property_returns_jobconfig(self):
        """job_config property should return JobConfig when metadata contains it."""
        from joshpy.jobs import JobConfig

        # Create session with a config that has these values
        config = JobConfig(
            simulation="JoshuaTreeSim",
            replicates=3,
            source_path=Path("/path/to/sim.josh"),
        )
        session_id = self.registry.create_session(
            config=config,
            experiment_name="test",
        )

        session = self.registry.get_session(session_id)
        retrieved_config = session.job_config

        self.assertIsInstance(retrieved_config, JobConfig)
        self.assertEqual(retrieved_config.simulation, "JoshuaTreeSim")
        self.assertEqual(retrieved_config.replicates, 3)
        self.assertEqual(str(retrieved_config.source_path), "/path/to/sim.josh")

    def test_job_config_property_always_has_job_config(self):
        """job_config property should return JobConfig since metadata always contains it."""
        from joshpy.jobs import JobConfig

        config = _make_config()
        session_id = self.registry.create_session(
            config=config,
            experiment_name="test",
        )

        session = self.registry.get_session(session_id)
        # With new API, metadata always contains job_config
        self.assertIsNotNone(session.job_config)
        self.assertIsInstance(session.job_config, JobConfig)

    def test_job_config_property_with_full_config(self):
        """job_config property should handle full JobConfig with sweep."""
        from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter

        config = JobConfig(
            template_path=Path("/path/to/template.j2"),
            simulation="Main",
            replicates=5,
            source_path=Path("/path/to/sim.josh"),
            sweep=SweepConfig(
                config_parameters=[
                    ConfigSweepParameter(name="maxGrowth", values=[10, 20, 30]),
                    ConfigSweepParameter(name="survivalProb", values=[0.8, 0.9]),
                ],
            ),
            file_mappings={
                "climate": Path("/data/climate.jshd"),
            },
        )

        session_id = self.registry.create_session(
            config=config,
            experiment_name="sweep_test",
        )

        session = self.registry.get_session(session_id)
        retrieved_config = session.job_config

        self.assertIsInstance(retrieved_config, JobConfig)
        self.assertEqual(retrieved_config.replicates, 5)
        self.assertIsNotNone(retrieved_config.sweep)
        self.assertEqual(len(retrieved_config.sweep.parameters), 2)
        self.assertEqual(retrieved_config.sweep.parameters[0].name, "maxGrowth")
        self.assertEqual(retrieved_config.sweep.parameters[0].values, [10, 20, 30])
        self.assertIn("climate", retrieved_config.file_mappings)

    def test_job_config_enables_session_reconstruction(self):
        """job_config property should enable session reconstruction pattern."""
        from joshpy.jobs import JobConfig

        # Original config
        original_config = JobConfig(
            simulation="TestSim",
            replicates=2,
            template_string="testParam = {{ testParam }}",
        )

        # Store in session (now using new API)
        session_id = self.registry.create_session(
            config=original_config,
            experiment_name="reconstruction_test",
        )

        # Later: reconstruct from session
        session = self.registry.get_session(session_id)
        reconstructed_config = session.job_config

        # Should be equivalent
        self.assertEqual(reconstructed_config.simulation, original_config.simulation)
        self.assertEqual(reconstructed_config.replicates, original_config.replicates)
        self.assertEqual(reconstructed_config.template_string, original_config.template_string)

        # Config is ready for re-expansion (would need template to actually expand)
        self.assertIsNotNone(reconstructed_config.template_string)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestTypedVariableColumns(unittest.TestCase):
    """Tests for Phase 7: Typed Variable Columns feature."""

    def setUp(self):
        """Create a registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_list_variable_columns_empty(self):
        """list_variable_columns should return empty list when no data loaded."""
        result = self.registry.list_variable_columns()
        self.assertEqual(result, [])

    def test_ensure_variable_columns_creates_double(self):
        """_ensure_variable_columns should create DOUBLE columns."""
        self.registry._ensure_variable_columns({"testVar": "DOUBLE"})
        
        columns = self.registry.list_variable_columns()
        self.assertIn("testVar", columns)

    def test_ensure_variable_columns_creates_varchar(self):
        """_ensure_variable_columns should create VARCHAR columns."""
        self.registry._ensure_variable_columns({"status": "VARCHAR"})
        
        columns = self.registry.list_variable_columns()
        self.assertIn("status", columns)

    def test_ensure_variable_columns_rejects_type_mismatch(self):
        """_ensure_variable_columns should reject type mismatches."""
        # Create column as DOUBLE
        self.registry._ensure_variable_columns({"testVar": "DOUBLE"})
        
        # Try to add same column as VARCHAR - should raise
        with self.assertRaises(ValueError) as ctx:
            self.registry._ensure_variable_columns({"testVar": "VARCHAR"})
        
        self.assertIn("exists as", str(ctx.exception))
        self.assertIn("DOUBLE", str(ctx.exception))

    def test_ensure_variable_columns_allows_same_type(self):
        """_ensure_variable_columns should allow re-adding same type."""
        # Create column as DOUBLE
        self.registry._ensure_variable_columns({"testVar": "DOUBLE"})
        
        # Re-adding as DOUBLE should not raise
        self.registry._ensure_variable_columns({"testVar": "DOUBLE"})
        
        columns = self.registry.list_variable_columns()
        self.assertIn("testVar", columns)

    def test_check_sparsity_empty_table(self):
        """check_sparsity should handle empty table."""
        report = self.registry.check_sparsity()
        
        self.assertEqual(report.total_rows, 0)
        self.assertEqual(report.column_stats, [])
        self.assertFalse(report.should_warn)

    def test_check_sparsity_with_data(self):
        """check_sparsity should report column statistics."""
        from joshpy.registry import SPARSITY_WARN_MIN_ROWS
        
        # Create a column
        self.registry._ensure_variable_columns({"testVar": "DOUBLE"})
        
        # Insert some rows with NULLs
        for i in range(SPARSITY_WARN_MIN_ROWS + 100):
            val = "1.0" if i < SPARSITY_WARN_MIN_ROWS // 2 else "NULL"
            self.registry.conn.execute(
                f"INSERT INTO cell_data (step, replicate, testVar) VALUES ({i}, 0, {val})"
            )
        
        report = self.registry.check_sparsity()
        
        self.assertGreater(report.total_rows, SPARSITY_WARN_MIN_ROWS)
        self.assertEqual(len(report.column_stats), 1)
        self.assertEqual(report.column_stats[0].name, "testVar")

    def test_sparsity_report_str(self):
        """SparsityReport.__str__ should provide readable output."""
        from joshpy.registry import SparsityReport, ColumnStats
        
        # Report with sparse columns
        report = SparsityReport(
            total_rows=10000,
            column_stats=[
                ColumnStats(name="sparseCol", dtype="DOUBLE", total_rows=10000, null_count=7500),
            ],
            threshold_percent=50,
        )
        
        report_str = str(report)
        self.assertIn("SparsityWarning", report_str)
        self.assertIn("sparseCol", report_str)
        self.assertIn("75.0%", report_str)

    def test_column_stats_null_percent(self):
        """ColumnStats.null_percent should calculate correctly."""
        from joshpy.registry import ColumnStats
        
        stats = ColumnStats(name="test", dtype="DOUBLE", total_rows=100, null_count=25)
        self.assertEqual(stats.null_percent, 25.0)
        
        # Zero rows should return 0
        stats_empty = ColumnStats(name="test", dtype="DOUBLE", total_rows=0, null_count=0)
        self.assertEqual(stats_empty.null_percent, 0.0)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestQuoteIdentifier(unittest.TestCase):
    """Tests for _quote_identifier helper function."""

    def test_quote_simple_name(self):
        """Should quote simple names."""
        from joshpy.registry import _quote_identifier
        
        self.assertEqual(_quote_identifier("treeCount"), '"treeCount"')
        self.assertEqual(_quote_identifier("avgHeight"), '"avgHeight"')

    def test_quote_names_with_dots(self):
        """Should preserve dots in quoted names."""
        from joshpy.registry import _quote_identifier
        
        self.assertEqual(_quote_identifier("avg.height"), '"avg.height"')
        self.assertEqual(_quote_identifier("a.b.c"), '"a.b.c"')

    def test_quote_names_with_dashes(self):
        """Should preserve dashes in quoted names."""
        from joshpy.registry import _quote_identifier
        
        self.assertEqual(_quote_identifier("tree-count"), '"tree-count"')

    def test_quote_escapes_internal_quotes(self):
        """Should escape double quotes inside names."""
        from joshpy.registry import _quote_identifier
        
        # Edge case: name contains a double quote
        self.assertEqual(_quote_identifier('a"b'), '"a""b"')


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestInferType(unittest.TestCase):
    """Tests for _infer_type helper function."""

    def test_infer_type_int(self):
        """Should return DOUBLE for integers."""
        from joshpy.registry import _infer_type
        
        self.assertEqual(_infer_type(42), "DOUBLE")
        self.assertEqual(_infer_type(0), "DOUBLE")
        self.assertEqual(_infer_type(-10), "DOUBLE")

    def test_infer_type_float(self):
        """Should return DOUBLE for floats."""
        from joshpy.registry import _infer_type
        
        self.assertEqual(_infer_type(3.14), "DOUBLE")
        self.assertEqual(_infer_type(0.0), "DOUBLE")
        self.assertEqual(_infer_type(-2.5), "DOUBLE")

    def test_infer_type_numeric_string(self):
        """Should return DOUBLE for numeric strings."""
        from joshpy.registry import _infer_type
        
        self.assertEqual(_infer_type("42"), "DOUBLE")
        self.assertEqual(_infer_type("3.14"), "DOUBLE")
        self.assertEqual(_infer_type("-10.5"), "DOUBLE")

    def test_infer_type_non_numeric_string(self):
        """Should return VARCHAR for non-numeric strings."""
        from joshpy.registry import _infer_type
        
        self.assertEqual(_infer_type("hello"), "VARCHAR")
        self.assertEqual(_infer_type("baseline"), "VARCHAR")
        self.assertEqual(_infer_type("10abc"), "VARCHAR")

    def test_infer_type_empty_string(self):
        """Should return VARCHAR for empty strings."""
        from joshpy.registry import _infer_type
        
        self.assertEqual(_infer_type(""), "VARCHAR")

    def test_infer_type_none(self):
        """Should return VARCHAR for None."""
        from joshpy.registry import _infer_type
        
        self.assertEqual(_infer_type(None), "VARCHAR")

    def test_infer_type_bool(self):
        """Should return VARCHAR for booleans (not numeric in this context)."""
        from joshpy.registry import _infer_type
        
        self.assertEqual(_infer_type(True), "VARCHAR")
        self.assertEqual(_infer_type(False), "VARCHAR")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestTypedConfigParameters(unittest.TestCase):
    """Tests for typed config parameters feature."""

    def setUp(self):
        """Create a registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config(), experiment_name="test")

    def tearDown(self):
        """Close registry after each test."""
        self.registry.close()

    def test_config_parameters_table_exists(self):
        """config_parameters table should be created in schema."""
        tables = self.registry.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        self.assertIn("config_parameters", table_names)

    def test_list_config_columns_empty(self):
        """list_config_columns should return empty list when no data."""
        result = self.registry.list_config_columns()
        self.assertEqual(result, [])

    def test_ensure_config_columns_creates_double(self):
        """_ensure_config_columns should create DOUBLE columns for numeric params."""
        self.registry._ensure_config_columns({"maxGrowth": "DOUBLE"})
        
        columns = self.registry.list_config_columns()
        self.assertIn("maxGrowth", columns)

    def test_ensure_config_columns_creates_varchar(self):
        """_ensure_config_columns should create VARCHAR columns for string params."""
        self.registry._ensure_config_columns({"scenario": "VARCHAR"})
        
        columns = self.registry.list_config_columns()
        self.assertIn("scenario", columns)

    def test_ensure_config_columns_with_special_chars(self):
        """_ensure_config_columns should handle names with dots."""
        self.registry._ensure_config_columns({"soil.moisture": "DOUBLE"})
        
        columns = self.registry.list_config_columns()
        self.assertIn("soil.moisture", columns)

    def test_ensure_config_columns_rejects_type_mismatch(self):
        """_ensure_config_columns should reject type mismatches."""
        self.registry._ensure_config_columns({"testParam": "DOUBLE"})
        
        with self.assertRaises(ValueError) as ctx:
            self.registry._ensure_config_columns({"testParam": "VARCHAR"})
        
        self.assertIn("exists as", str(ctx.exception))

    def test_register_run_stores_typed_parameters(self):
        """register_run should store parameters as typed columns."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash123",
            josh_path="/path/to/sim.josh",
            config_content="test content",
            file_mappings=None,
            parameters={"maxGrowth": 50, "scenario": "baseline"},
        )
        
        # Check columns were created
        columns = self.registry.list_config_columns()
        self.assertIn("maxGrowth", columns)
        self.assertIn("scenario", columns)
        
        # Check values were stored
        result = self.registry.conn.execute(
            'SELECT "maxGrowth", scenario FROM config_parameters WHERE run_hash = ?',
            ["hash123"],
        ).fetchone()
        self.assertEqual(result[0], 50)
        self.assertEqual(result[1], "baseline")

    def test_register_run_with_mixed_types(self):
        """register_run should handle mixed numeric and string parameters."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash456",
            josh_path="/path/to/sim.josh",
            config_content="test content",
            file_mappings=None,
            parameters={"numericParam": 3.14, "stringParam": "hello", "intParam": 42},
        )
        
        columns = self.registry.list_config_columns()
        self.assertIn("numericParam", columns)
        self.assertIn("stringParam", columns)
        self.assertIn("intParam", columns)

    def test_get_config_by_hash_returns_typed_parameters(self):
        """get_config_by_hash should return parameters from typed columns."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash789",
            josh_path="/path/to/sim.josh",
            config_content="test content",
            file_mappings=None,
            parameters={"x": 10, "y": "test"},
        )
        
        config = self.registry.get_config_by_hash("hash789")
        self.assertIsNotNone(config)
        self.assertEqual(config.parameters["x"], 10)
        self.assertEqual(config.parameters["y"], "test")

    def test_get_configs_for_session_returns_typed_parameters(self):
        """get_configs_for_session should return parameters from typed columns."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hashA",
            josh_path="/path/to/sim.josh",
            config_content="content A",
            file_mappings=None,
            parameters={"param": 100},
        )
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hashB",
            josh_path="/path/to/sim.josh",
            config_content="content B",
            file_mappings=None,
            parameters={"param": 200},
        )
        
        configs = self.registry.get_configs_for_session(self.session_id)
        self.assertEqual(len(configs), 2)
        params = {c.run_hash: c.parameters["param"] for c in configs}
        self.assertEqual(params["hashA"], 100)
        self.assertEqual(params["hashB"], 200)

    def test_list_config_parameters_from_typed_columns(self):
        """list_config_parameters should list parameter names from typed columns."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash1",
            josh_path="/path/to/sim.josh",
            config_content="test",
            file_mappings=None,
            parameters={"alpha": 1, "beta": 2},
        )
        
        params = self.registry.list_config_parameters()
        self.assertIn("alpha", params)
        self.assertIn("beta", params)

    def test_sql_query_typed_parameters_directly(self):
        """Verify typed parameters enable clean SQL queries without JSON extraction."""
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash1",
            josh_path="/path/to/sim.josh",
            config_content="test",
            file_mappings=None,
            parameters={"maxGrowth": 50},
        )
        
        # This is the key benefit: clean SQL without json_extract
        result = self.registry.conn.execute(
            '''
            SELECT cp."maxGrowth"
            FROM config_parameters cp
            WHERE cp."maxGrowth" > 25
            '''
        ).fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 50)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestLabelSystem(unittest.TestCase):
    """Tests for the run label system."""

    def setUp(self):
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        config = _make_config()
        self.session_id = self.registry.create_session(config=config)
        # Register two runs
        self.registry.register_run(
            self.session_id, "hash_aaa111", "/sim.josh",
            "param = 10 count", None, {"param": 10},
        )
        self.registry.register_run(
            self.session_id, "hash_bbb222", "/sim.josh",
            "param = 20 count", None, {"param": 20},
        )

    def tearDown(self):
        self.registry.close()

    def test_label_run_basic(self):
        self.registry.label_run("hash_aaa111", "baseline")
        labels = self.registry.list_labels()
        self.assertEqual(labels, [("baseline", "hash_aaa111")])

    def test_label_multiple_runs(self):
        self.registry.label_run("hash_aaa111", "baseline")
        self.registry.label_run("hash_bbb222", "high_mortality")
        labels = self.registry.list_labels()
        self.assertEqual(len(labels), 2)
        self.assertIn(("baseline", "hash_aaa111"), labels)
        self.assertIn(("high_mortality", "hash_bbb222"), labels)

    def test_label_uniqueness_error(self):
        self.registry.label_run("hash_aaa111", "baseline")
        with self.assertRaises(ValueError) as ctx:
            self.registry.label_run("hash_bbb222", "baseline")
        self.assertIn("already assigned", str(ctx.exception))
        self.assertIn("force=True", str(ctx.exception))

    def test_label_force_reassign(self):
        self.registry.label_run("hash_aaa111", "baseline")
        self.registry.label_run("hash_bbb222", "baseline", force=True)
        labels = self.registry.list_labels()
        self.assertEqual(labels, [("baseline", "hash_bbb222")])

    def test_label_same_run_same_label_ok(self):
        """Re-labeling the same run with the same label should succeed."""
        self.registry.label_run("hash_aaa111", "baseline")
        self.registry.label_run("hash_aaa111", "baseline")  # no error

    def test_label_run_missing_hash(self):
        with self.assertRaises(KeyError):
            self.registry.label_run("nonexistent", "baseline")

    def test_resolve_label(self):
        self.registry.label_run("hash_aaa111", "baseline")
        self.assertEqual(self.registry.resolve_label("baseline"), "hash_aaa111")

    def test_resolve_label_missing(self):
        with self.assertRaises(KeyError):
            self.registry.resolve_label("nonexistent")

    def test_list_labels_empty(self):
        self.assertEqual(self.registry.list_labels(), [])

    def test_export_config_by_label(self):
        import tempfile

        self.registry.label_run("hash_aaa111", "baseline")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.registry.export_config("baseline", tmpdir)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "baseline.jshc")
            content = path.read_text()
            self.assertIn("READ-ONLY", content)
            self.assertIn("param = 10 count", content)

    def test_export_config_by_hash(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.registry.export_config("hash_aaa111", tmpdir)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "hash_aaa111.jshc")

    def test_export_config_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(KeyError):
                self.registry.export_config("nonexistent", tmpdir)

    def test_config_info_includes_label(self):
        self.registry.label_run("hash_aaa111", "baseline")
        config = self.registry.get_config_by_hash("hash_aaa111")
        self.assertEqual(config.label, "baseline")

    def test_config_info_label_none_by_default(self):
        config = self.registry.get_config_by_hash("hash_aaa111")
        self.assertIsNone(config.label)

    def test_label_on_collision_timestamp(self):
        """on_collision='timestamp' archives the old label with a timestamp suffix."""
        self.registry.label_run("hash_aaa111", "baseline")
        self.registry.label_run("hash_bbb222", "baseline", on_collision="timestamp")
        # Bare label now points to the new run
        self.assertEqual(self.registry.resolve_label("baseline"), "hash_bbb222")
        # Old run has a timestamped label
        labels = self.registry.list_labels()
        old_labels = [l for l, h in labels if h == "hash_aaa111"]
        self.assertEqual(len(old_labels), 1)
        self.assertRegex(old_labels[0], r"^baseline_\d{8}_\d{6}$")

    def test_label_on_collision_timestamp_disambiguation(self):
        """Same-second collisions get _2, _3 suffixes."""
        # Register a third run
        self.registry.register_run(
            self.session_id, "hash_ccc333", "/sim.josh",
            "param = 30 count", None, {"param": 30},
        )
        self.registry.label_run("hash_aaa111", "baseline")
        self.registry.label_run("hash_bbb222", "baseline", on_collision="timestamp")
        self.registry.label_run("hash_ccc333", "baseline", on_collision="timestamp")
        # Bare label -> hash_ccc333
        self.assertEqual(self.registry.resolve_label("baseline"), "hash_ccc333")
        # Two archived labels, one with _2 suffix if same second
        labels = {l: h for l, h in self.registry.list_labels()}
        archived = [l for l in labels if l.startswith("baseline_")]
        self.assertEqual(len(archived), 2)
        # All three hashes should be labeled
        self.assertEqual(len(labels), 3)

    def test_label_force_and_on_collision_mutually_exclusive(self):
        """force=True and on_collision cannot be used together."""
        self.registry.label_run("hash_aaa111", "baseline")
        with self.assertRaises(ValueError) as ctx:
            self.registry.label_run(
                "hash_bbb222", "baseline", force=True, on_collision="timestamp"
            )
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_label_on_collision_invalid_value(self):
        """Invalid on_collision value raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.registry.label_run("hash_aaa111", "test", on_collision="invalid")
        self.assertIn("Invalid on_collision", str(ctx.exception))

    def test_resolve_latest_single_match(self):
        """resolve_latest returns the only matching run."""
        self.registry.label_run("hash_aaa111", "baseline")
        self.assertEqual(self.registry.resolve_latest("baseline"), "hash_aaa111")

    def test_resolve_latest_multiple_matches(self):
        """resolve_latest returns the most recently created run."""
        self.registry.label_run("hash_aaa111", "baseline")
        self.registry.label_run("hash_bbb222", "baseline", on_collision="timestamp")
        # hash_bbb222 was registered after hash_aaa111 so it has a later created_at
        self.assertEqual(self.registry.resolve_latest("baseline"), "hash_bbb222")

    def test_resolve_latest_no_match(self):
        """resolve_latest raises KeyError when no labels match."""
        with self.assertRaises(KeyError):
            self.registry.resolve_latest("nonexistent")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestInspectTextDiff(unittest.TestCase):
    """Tests for the headless --print text-diff mode of joshpy.inspect."""

    def setUp(self):
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config())
        self.registry.register_run(
            self.session_id, "hash_aaa111", "/sim.josh",
            "survivalProbAdult = 85 %\ngrowthRate = 1.2\n", None, {"param": 10},
        )
        self.registry.register_run(
            self.session_id, "hash_bbb222", "/sim.josh",
            "survivalProbAdult = 90 %\ngrowthRate = 1.2\n", None, {"param": 20},
        )

    def tearDown(self):
        self.registry.close()

    def test_text_diff_produces_unified_diff(self):
        from joshpy.inspect._core import text_diff

        diff = text_diff(self.registry, "hash_aaa111", "hash_bbb222")
        self.assertIn("---", diff)
        self.assertIn("+++", diff)
        self.assertIn("@@", diff)
        self.assertIn("-survivalProbAdult = 85 %", diff)
        self.assertIn("+survivalProbAdult = 90 %", diff)

    def test_text_diff_uses_labels_in_headers(self):
        from joshpy.inspect._core import text_diff

        self.registry.label_run("hash_aaa111", "baseline")
        self.registry.label_run("hash_bbb222", "high_survival")
        diff = text_diff(self.registry, "baseline", "high_survival")
        self.assertIn("--- baseline", diff)
        self.assertIn("+++ high_survival", diff)

    def test_text_diff_identical_is_empty(self):
        from joshpy.inspect._core import text_diff

        self.assertEqual(text_diff(self.registry, "hash_aaa111", "hash_aaa111"), "")

    def test_text_diff_missing_run_raises(self):
        from joshpy.inspect._core import text_diff

        with self.assertRaises(KeyError):
            text_diff(self.registry, "hash_aaa111", "nonexistent")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestRegistryDescribeMethods(unittest.TestCase):
    """Tests for the describe_*() convenience wrappers on RunRegistry."""

    def setUp(self):
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(
            config=_make_config(), experiment_name="describe_demo"
        )
        self.registry.register_run(
            self.session_id, "hash_aaa111", "/sim.josh",
            "maxGrowth = 5 meters", None, {"maxGrowth": 5},
        )
        self.registry.label_run("hash_aaa111", "baseline")

    def tearDown(self):
        self.registry.close()

    def test_describe_labels_matches_format_labels(self):
        from joshpy.inspect import format_labels

        self.assertEqual(
            self.registry.describe_labels(), format_labels(self.registry)
        )
        self.assertIn("baseline", self.registry.describe_labels())

    def test_describe_sessions_matches_format_sessions(self):
        from joshpy.inspect import format_sessions

        self.assertEqual(
            self.registry.describe_sessions(), format_sessions(self.registry)
        )
        self.assertIn("describe_demo", self.registry.describe_sessions())

    def test_describe_run_matches_format_run_info(self):
        from joshpy.inspect import format_run_info

        self.assertEqual(
            self.registry.describe_run("baseline"),
            format_run_info(self.registry, "baseline"),
        )
        self.assertIn("maxGrowth", self.registry.describe_run("baseline"))

    def test_describe_run_missing_raises(self):
        with self.assertRaises(KeyError):
            self.registry.describe_run("nonexistent")

    def test_describe_summary_matches_format_summary(self):
        from joshpy.inspect import format_summary

        self.assertEqual(
            self.registry.describe_summary(), format_summary(self.registry)
        )


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestGetRunInfo(unittest.TestCase):
    """Tests for the structured get_run_info() / RunDetail aggregate."""

    def setUp(self):
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(
            config=_make_config(), experiment_name="run_info_demo"
        )
        self.registry.register_run(
            self.session_id, "hash_aaa111", "/sim.josh",
            "maxGrowth = 5 meters",
            {"soil": {"path": "/data/soil.jshd", "hash": "abc"}},
            {"maxGrowth": 5},
        )
        self.registry.label_run("hash_aaa111", "baseline")
        # Two completed runs (one ok, one failed) plus one pending.
        ok = self.registry.start_run(
            run_hash="hash_aaa111", session_id=self.session_id,
            replicate=0, output_path="out/0.csv",
        )
        self.registry.complete_run(ok, exit_code=0)
        bad = self.registry.start_run(
            run_hash="hash_aaa111", session_id=self.session_id,
            replicate=1, output_path="out/1.csv",
        )
        self.registry.complete_run(bad, exit_code=1)
        self.registry.start_run(
            run_hash="hash_aaa111", session_id=self.session_id,
            replicate=2, output_path="out/2.csv",
        )  # left pending (no complete_run)

    def tearDown(self):
        self.registry.close()

    def test_returns_run_detail_with_config_and_runs(self):
        from joshpy.registry import ConfigInfo, RunDetail

        detail = self.registry.get_run_info("baseline")
        self.assertIsInstance(detail, RunDetail)
        self.assertIsInstance(detail.config, ConfigInfo)
        self.assertEqual(detail.run_hash, "hash_aaa111")
        self.assertEqual(detail.label, "baseline")
        self.assertEqual(detail.parameters, {"maxGrowth": 5})
        self.assertEqual(len(detail.runs), 3)

    def test_resolves_by_hash_too(self):
        detail = self.registry.get_run_info("hash_aaa111")
        self.assertEqual(detail.label, "baseline")

    def test_status_counts(self):
        detail = self.registry.get_run_info("baseline")
        self.assertEqual(detail.succeeded, 1)
        self.assertEqual(detail.failed, 1)
        self.assertEqual(detail.pending, 1)

    def test_missing_run_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.registry.get_run_info("nonexistent")

    def test_describe_run_is_consistent_with_get_run_info(self):
        """describe_run should report the same status counts get_run_info exposes."""
        detail = self.registry.get_run_info("baseline")
        text = self.registry.describe_run("baseline")
        self.assertIn(
            f"Runs: {detail.succeeded} succeeded, "
            f"{detail.failed} failed, {detail.pending} pending",
            text,
        )


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestGitHash(unittest.TestCase):
    """Tests for git hash capture in session metadata."""

    def test_git_hash_in_session_metadata(self):
        from joshpy.registry import RunRegistry

        registry = RunRegistry(":memory:")
        config = _make_config()
        session_id = registry.create_session(config=config)
        session = registry.get_session(session_id)
        # We're in a git repo, so git_hash should be present
        import json
        metadata = json.loads(session.metadata) if isinstance(session.metadata, str) else session.metadata
        self.assertIn("git_hash", metadata)
        git_hash = metadata["git_hash"]
        # Should be 12 hex chars, optionally with +dirty
        base = git_hash.replace("+dirty", "")
        self.assertEqual(len(base), 12)
        self.assertTrue(all(c in "0123456789abcdef" for c in base))
        registry.close()


class TestGetFileMappings(unittest.TestCase):
    """Tests for the get_file_mappings convenience method."""

    def setUp(self):
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config())

    def tearDown(self):
        self.registry.close()

    def test_returns_paths(self):
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash_with_fm",
            josh_path="sim.josh",
            config_content="x = 1",
            file_mappings={
                "soil": {"path": "/data/soil.jshd", "hash": "aaa"},
                "climate": {"path": "/data/climate.jshd", "hash": "bbb"},
            },
            parameters={},
        )
        result = self.registry.get_file_mappings("hash_with_fm")
        self.assertEqual(result, {
            "soil": Path("/data/soil.jshd"),
            "climate": Path("/data/climate.jshd"),
        })

    def test_by_label(self):
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash_labeled",
            josh_path="sim.josh",
            config_content="x = 1",
            file_mappings={"soil": {"path": "/data/soil.jshd", "hash": "aaa"}},
            parameters={},
        )
        self.registry.label_run("hash_labeled", "my_run")
        result = self.registry.get_file_mappings("my_run")
        self.assertEqual(result, {"soil": Path("/data/soil.jshd")})

    def test_none_when_no_mappings(self):
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash_no_fm",
            josh_path="sim.josh",
            config_content="x = 1",
            file_mappings=None,
            parameters={},
        )
        self.assertIsNone(self.registry.get_file_mappings("hash_no_fm"))

    def test_unknown_hash_raises(self):
        with self.assertRaises(KeyError):
            self.registry.get_file_mappings("nonexistent")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestDebugOutputAccess(unittest.TestCase):
    """Tests for registry debug output lookup and loading."""

    def setUp(self):
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(config=_make_config())
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash_debug",
            josh_path="sim.josh",
            config_content="x = 1",
            file_mappings=None,
            parameters={},
        )
        self.registry.label_run("hash_debug", "baseline")

        self.tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self.tmpdir.name)
        self.org_1 = tmp / "org_1.txt"
        self.patch_1 = tmp / "patch_1.txt"
        self.org_2 = tmp / "org_2.txt"

        self.org_1.write_text("[Step 0, organism @ aaa (1.0, 2.0)] init\n")
        self.patch_1.write_text("[Step 0, patch @ fff (1.0, 2.0)] fire false\n")
        self.org_2.write_text("[Step 1, organism @ bbb (3.0, 4.0)] grow\n")

        self.run_id_1 = self.registry.start_run(
            run_hash="hash_debug", session_id=self.session_id
        )
        self.registry.complete_run(self.run_id_1, exit_code=0)
        self.registry.register_output(
            run_id=self.run_id_1,
            output_type="debug.organism",
            file_path=str(self.org_1),
        )
        self.registry.register_output(
            run_id=self.run_id_1,
            output_type="debug.patch",
            file_path=str(self.patch_1),
        )

        self.run_id_2 = self.registry.start_run(
            run_hash="hash_debug", session_id=self.session_id
        )
        self.registry.complete_run(self.run_id_2, exit_code=0)
        self.registry.register_output(
            run_id=self.run_id_2,
            output_type="debug.organism",
            file_path=str(self.org_2),
        )

    def tearDown(self):
        self.registry.close()
        self.tmpdir.cleanup()

    def test_get_debug_output_files_uses_latest_run_by_default(self):
        paths = self.registry.get_debug_output_files("baseline")
        self.assertEqual(paths, [self.org_2])

    def test_get_debug_output_files_can_target_specific_run_id(self):
        paths = self.registry.get_debug_output_files("baseline", run_id=self.run_id_1)
        self.assertEqual(paths, [self.org_1, self.patch_1])

    def test_get_debug_output_files_entity_type_filter(self):
        paths = self.registry.get_debug_output_files(
            "baseline", run_id=self.run_id_1, entity_types=["patch"]
        )
        self.assertEqual(paths, [self.patch_1])

    def test_get_debug_output_files_existing_only_filter(self):
        missing = Path(self.tmpdir.name) / "missing.txt"
        self.registry.register_output(
            run_id=self.run_id_2,
            output_type="debug.patch",
            file_path=str(missing),
        )

        paths_existing = self.registry.get_debug_output_files(
            "baseline", existing_only=True
        )
        self.assertEqual(paths_existing, [self.org_2])

        paths_all = self.registry.get_debug_output_files(
            "baseline", existing_only=False
        )
        self.assertEqual(paths_all, [self.org_2, missing])

    def test_get_debug_output_files_bad_run_id(self):
        with self.assertRaises(KeyError):
            self.registry.get_debug_output_files("baseline", run_id="unknown-run-id")

    def test_get_debug_output_files_run_id_mismatch(self):
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="other_hash",
            josh_path="sim2.josh",
            config_content="x = 2",
            file_mappings=None,
            parameters={},
        )
        other_run = self.registry.start_run(
            run_hash="other_hash", session_id=self.session_id
        )
        self.registry.complete_run(other_run, exit_code=0)

        with self.assertRaises(ValueError):
            self.registry.get_debug_output_files("baseline", run_id=other_run)

    def test_load_debug_merges_messages(self):
        store = self.registry.load_debug("baseline", run_id=self.run_id_1)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.messages[0].entity_type, "organism")
        self.assertEqual(store.messages[1].entity_type, "patch")

    def test_load_debug_no_outputs_raises(self):
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash_no_debug",
            josh_path="sim3.josh",
            config_content="x = 3",
            file_mappings=None,
            parameters={},
        )
        self.registry.label_run("hash_no_debug", "no_debug")
        run_id = self.registry.start_run(
            run_hash="hash_no_debug", session_id=self.session_id
        )
        self.registry.complete_run(run_id, exit_code=0)

        with self.assertRaises(ValueError):
            self.registry.load_debug("no_debug")


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestPooledRuns(unittest.TestCase):
    """Tests for pooled runs — same run_hash across multiple sessions."""

    def setUp(self):
        """Create a registry with two sessions sharing a run_hash."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session1 = self.registry.create_session(
            config=_make_config(), experiment_name="sweep_v1"
        )
        self.session2 = self.registry.create_session(
            config=_make_config(), experiment_name="sweep_v2"
        )

        # Register the same config in both sessions
        for sid in [self.session1, self.session2]:
            self.registry.register_run(
                session_id=sid,
                run_hash="shared_hash",
                josh_path="/path/to/sim.josh",
                config_content="maxGrowth = 5 meters",
                file_mappings=None,
                parameters={"maxGrowth": 5},
            )

    def tearDown(self):
        self.registry.close()

    def test_configs_for_both_sessions(self):
        """get_configs_for_session returns config for both sessions."""
        configs1 = self.registry.get_configs_for_session(self.session1)
        configs2 = self.registry.get_configs_for_session(self.session2)
        self.assertEqual(len(configs1), 1)
        self.assertEqual(len(configs2), 1)
        self.assertEqual(configs1[0].run_hash, "shared_hash")
        self.assertEqual(configs2[0].run_hash, "shared_hash")

    def test_session_summary_counts_jobs_for_both(self):
        """get_session_summary shows correct job count for both sessions."""
        # Add a run in each session
        run1 = self.registry.start_run(run_hash="shared_hash", session_id=self.session1, replicate=0)
        self.registry.complete_run(run1, exit_code=0)
        run2 = self.registry.start_run(run_hash="shared_hash", session_id=self.session2, replicate=0)
        self.registry.complete_run(run2, exit_code=0)

        summary1 = self.registry.get_session_summary(self.session1)
        summary2 = self.registry.get_session_summary(self.session2)

        # Both sessions should show 1 job
        self.assertEqual(summary1.total_jobs, 1)
        self.assertEqual(summary2.total_jobs, 1)

        # Each session sees only its own run (session_id scopes runs)
        self.assertEqual(summary1.runs_succeeded, 1)
        self.assertEqual(summary2.runs_succeeded, 1)

    def test_get_replicate_count_from_cell_data(self):
        """get_replicate_count returns count of distinct replicates."""
        # Insert cell_data rows for 5 replicates
        self.registry._ensure_variable_columns({"height": "DOUBLE"})
        for rep in range(5):
            self.registry.conn.execute(
                """
                INSERT INTO cell_data (run_hash, step, replicate, position_x, position_y)
                VALUES (?, ?, ?, ?, ?)
                """,
                ["shared_hash", 0, rep, 0.0, 0.0],
            )

        count = self.registry.get_replicate_count("shared_hash")
        self.assertEqual(count, 5)

    def test_get_replicate_count_returns_zero_when_no_data(self):
        """get_replicate_count returns 0 when no cell_data loaded."""
        count = self.registry.get_replicate_count("shared_hash")
        self.assertEqual(count, 0)

    def test_session_configs_junction_populated(self):
        """session_configs junction table has entries for both sessions."""
        result = self.registry.conn.execute(
            "SELECT session_id, run_hash FROM session_configs ORDER BY session_id"
        ).fetchall()
        session_ids = {row[0] for row in result}
        self.assertEqual(len(result), 2)
        self.assertIn(self.session1, session_ids)
        self.assertIn(self.session2, session_ids)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestPooledRunsInspect(unittest.TestCase):
    """Tests for inspect display functions with pooled runs."""

    def setUp(self):
        """Create a registry with pooled data."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session1 = self.registry.create_session(
            config=_make_config(), experiment_name="sweep_v1"
        )
        self.session2 = self.registry.create_session(
            config=_make_config(), experiment_name="sweep_v2"
        )

        for sid in [self.session1, self.session2]:
            self.registry.register_run(
                session_id=sid,
                run_hash="pooled_hash",
                josh_path="/path/to/sim.josh",
                config_content="maxGrowth = 5 meters",
                file_mappings=None,
                parameters={"maxGrowth": 5},
            )

        self.registry.label_run("pooled_hash", "growth_5")

        # Add cell_data for 5 replicates (simulating 2 + 3 pooled)
        for rep in range(5):
            self.registry.conn.execute(
                """
                INSERT INTO cell_data (run_hash, step, replicate, position_x, position_y)
                VALUES (?, ?, ?, ?, ?)
                """,
                ["pooled_hash", 0, rep, 0.0, 0.0],
            )

    def tearDown(self):
        self.registry.close()

    def test_format_labels_shows_correct_reps(self):
        """format_labels REPS column reflects cell_data replicate count."""
        from joshpy.inspect import format_labels

        output = format_labels(self.registry)
        # Should contain "5" (from 5 distinct replicates in cell_data)
        self.assertIn("5", output)
        self.assertIn("growth_5", output)

    def test_format_run_info_shows_replicates(self):
        """format_run_info shows Replicates line from cell_data."""
        from joshpy.inspect import format_run_info

        output = format_run_info(self.registry, "growth_5")
        self.assertIn("Replicates: 5", output)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestRegistryBusyError(unittest.TestCase):
    """Tests for the typed lock error on a concurrently-held registry."""

    def test_locked_registry_raises_registry_busy_error(self):
        """Opening a registry locked by another process should raise a typed,
        actionable RegistryBusyError rather than a raw DuckDB IOException."""
        import os
        import subprocess
        import sys
        import time

        from joshpy.registry import RegistryBusyError, RunRegistry

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "busy.duckdb")
            # Hold a write lock from a separate process (DuckDB is single-writer
            # across processes; same-process connections are shared).
            holder = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import duckdb, time, sys; "
                    "c = duckdb.connect(sys.argv[1]); "
                    "c.execute('CREATE TABLE t(x INTEGER)'); "
                    "time.sleep(15)",
                    db_path,
                ]
            )
            try:
                # Wait for the holder to acquire the lock.
                deadline = time.monotonic() + 10
                while not os.path.exists(db_path) and time.monotonic() < deadline:
                    time.sleep(0.1)
                time.sleep(1.0)

                with self.assertRaises(RegistryBusyError) as ctx:
                    RunRegistry(db_path)
                msg = str(ctx.exception)
                self.assertIn("locked by another process", msg)
                self.assertIn(db_path, msg)
            finally:
                holder.terminate()
                holder.wait()

    def test_registry_busy_error_is_runtime_error(self):
        """RegistryBusyError should subclass RuntimeError for easy catching."""
        from joshpy.registry import RegistryBusyError

        self.assertTrue(issubclass(RegistryBusyError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
