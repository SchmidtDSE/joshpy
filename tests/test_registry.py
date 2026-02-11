"""Unit tests for the registry module."""

import unittest
from pathlib import Path

# Check if duckdb is available
try:
    import duckdb  # noqa: F401

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


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
        session_id = self.registry.create_session(
            experiment_name="test_experiment",
            simulation="TestSim",
        )
        self.assertIsInstance(session_id, str)
        self.assertTrue(len(session_id) > 0)

    def test_create_session_with_all_fields(self):
        """create_session should store all provided fields."""
        session_id = self.registry.create_session(
            experiment_name="sensitivity_analysis",
            simulation="JoshuaTreeSim",
            template_path="/path/to/template.j2",
            template_hash="abc123def456",
            metadata={"author": "test", "version": "1.0"},
        )

        session = self.registry.get_session(session_id)
        self.assertEqual(session.experiment_name, "sensitivity_analysis")
        self.assertEqual(session.simulation, "JoshuaTreeSim")
        # total_jobs and total_replicates are now computed from actual data
        self.assertIsNone(session.total_jobs)
        self.assertIsNone(session.total_replicates)
        self.assertEqual(session.template_path, "/path/to/template.j2")
        self.assertEqual(session.template_hash, "abc123def456")
        self.assertEqual(session.metadata, {"author": "test", "version": "1.0"})
        self.assertEqual(session.status, "pending")

    def test_get_session_not_found(self):
        """get_session should return None for non-existent session."""
        result = self.registry.get_session("nonexistent-id")
        self.assertIsNone(result)

    def test_update_session_status(self):
        """update_session_status should change session status."""
        session_id = self.registry.create_session(experiment_name="test")

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
        self.registry.create_session(experiment_name="exp1")
        self.registry.create_session(experiment_name="exp2")
        self.registry.create_session(experiment_name="exp3")

        sessions = self.registry.list_sessions()
        self.assertEqual(len(sessions), 3)

    def test_list_sessions_by_experiment_name(self):
        """list_sessions should filter by experiment_name."""
        self.registry.create_session(experiment_name="sensitivity")
        self.registry.create_session(experiment_name="sensitivity")
        self.registry.create_session(experiment_name="calibration")

        sensitivity_sessions = self.registry.list_sessions(experiment_name="sensitivity")
        self.assertEqual(len(sensitivity_sessions), 2)
        for s in sensitivity_sessions:
            self.assertEqual(s.experiment_name, "sensitivity")

    def test_list_sessions_with_limit(self):
        """list_sessions should respect limit parameter."""
        for i in range(10):
            self.registry.create_session(experiment_name=f"exp{i}")

        sessions = self.registry.list_sessions(limit=5)
        self.assertEqual(len(sessions), 5)


@unittest.skipIf(not HAS_DUCKDB, "duckdb not installed")
class TestConfigRegistration(unittest.TestCase):
    """Tests for config registration and lookup."""

    def setUp(self):
        """Create a fresh in-memory registry for each test."""
        from joshpy.registry import RunRegistry

        self.registry = RunRegistry(":memory:")
        self.session_id = self.registry.create_session(experiment_name="test")

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
        self.session_id = self.registry.create_session(experiment_name="test")
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
        run_id = self.registry.start_run(run_hash="config123")

        self.registry.complete_run(run_id=run_id, exit_code=0)

        run = self.registry.get_run(run_id)
        self.assertIsNotNone(run.completed_at)
        self.assertEqual(run.exit_code, 0)
        self.assertIsNone(run.error_message)

    def test_complete_run_failure(self):
        """complete_run should store error message on failure."""
        run_id = self.registry.start_run(run_hash="config123")

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
        run1 = self.registry.start_run(run_hash="config123", replicate=0)
        run2 = self.registry.start_run(run_hash="config123", replicate=1)
        run3 = self.registry.start_run(run_hash="config123", replicate=2)

        runs = self.registry.get_runs_for_hash("config123")
        self.assertEqual(len(runs), 3)
        run_ids = {r.run_id for r in runs}
        self.assertEqual(run_ids, {run1, run2, run3})

    def test_run_with_metadata(self):
        """start_run should store metadata."""
        run_id = self.registry.start_run(
            run_hash="config123",
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
        self.session_id = self.registry.create_session(experiment_name="test")
        self.registry.register_run(
            session_id=self.session_id,
            run_hash="config123",
            josh_path="/path/to/sim.josh",
            config_content="test content",
            file_mappings=None,
            parameters={},
        )
        self.run_id = self.registry.start_run(run_hash="config123")

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
            experiment_name="test",
            simulation="TestSim",
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
        run1 = self.registry.start_run(run_hash="hash1")
        self.registry.complete_run(run1, exit_code=0)

        run2 = self.registry.start_run(run_hash="hash2")
        self.registry.complete_run(run2, exit_code=1, error_message="failed")

        run3 = self.registry.start_run(run_hash="hash3")
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
        self.registry.start_run(run_hash="hash1")

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
        self.session_id = self.registry.create_session(experiment_name="test")

        self.registry.register_run(
            session_id=self.session_id,
            run_hash="hash1",
            josh_path="/path/to/sim.josh",
            config_content="content",
            file_mappings=None,
            parameters={"survival": 85, "growth": 1.5},
        )
        run_id = self.registry.start_run(run_hash="hash1")
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

        empty_session = self.registry.create_session(experiment_name="empty")
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
        self.session_id = self.registry.create_session(experiment_name="test")
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
                    experiment_name="persistence_test",
                    simulation="TestSim",
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
            config_content="content",
            file_mappings={"data": {"path": "/path/to/data.jshd", "hash": "abc"}},
            parameters={"x": 1},
            created_at=datetime.now(),
        )
        self.assertEqual(config.run_hash, "abc123")
        self.assertEqual(config.parameters, {"x": 1})

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
            experiment_name="test_experiment",
            simulation="TestSim",
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
            experiment_name="test_experiment",
            session_id=None,
        )

        # Should be a valid UUID format (36 chars with dashes)
        self.assertEqual(len(session_id), 36)
        self.assertEqual(session_id.count("-"), 4)

    def test_create_session_external_id_with_metadata(self):
        """create_session with external ID should store all fields correctly."""
        external_id = "frontend-project-456"
        job_config_dict = {
            "simulation": "JoshuaTreeSim",
            "replicates": 3,
            "source_path": "/path/to/sim.josh",
        }

        session_id = self.registry.create_session(
            experiment_name="sensitivity_analysis",
            simulation="JoshuaTreeSim",
            metadata={"job_config": job_config_dict, "author": "test"},
            session_id=external_id,
        )

        self.assertEqual(session_id, external_id)

        session = self.registry.get_session(external_id)
        self.assertEqual(session.experiment_name, "sensitivity_analysis")
        self.assertEqual(session.simulation, "JoshuaTreeSim")
        self.assertEqual(session.metadata["job_config"], job_config_dict)
        self.assertEqual(session.metadata["author"], "test")

    def test_create_session_duplicate_external_id_raises(self):
        """create_session with duplicate external ID should raise error."""
        external_id = "duplicate-id"

        # First creation should succeed
        self.registry.create_session(
            experiment_name="first",
            session_id=external_id,
        )

        # Second creation with same ID should raise
        with self.assertRaises(Exception):  # DuckDB raises constraint violation
            self.registry.create_session(
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

        job_config_dict = {
            "simulation": "JoshuaTreeSim",
            "replicates": 3,
            "source_path": "/path/to/sim.josh",
        }

        session_id = self.registry.create_session(
            experiment_name="test",
            metadata={"job_config": job_config_dict},
        )

        session = self.registry.get_session(session_id)
        config = session.job_config

        self.assertIsInstance(config, JobConfig)
        self.assertEqual(config.simulation, "JoshuaTreeSim")
        self.assertEqual(config.replicates, 3)
        self.assertEqual(str(config.source_path), "/path/to/sim.josh")

    def test_job_config_property_returns_none_when_missing(self):
        """job_config property should return None when metadata has no job_config."""
        session_id = self.registry.create_session(
            experiment_name="test",
            metadata={"other_key": "value"},
        )

        session = self.registry.get_session(session_id)
        self.assertIsNone(session.job_config)

    def test_job_config_property_returns_none_when_no_metadata(self):
        """job_config property should return None when session has no metadata."""
        session_id = self.registry.create_session(
            experiment_name="test",
            metadata=None,
        )

        session = self.registry.get_session(session_id)
        self.assertIsNone(session.job_config)

    def test_job_config_property_with_full_config(self):
        """job_config property should handle full JobConfig with sweep."""
        from joshpy.jobs import JobConfig

        job_config_dict = {
            "template_path": "/path/to/template.j2",
            "simulation": "Main",
            "replicates": 5,
            "source_path": "/path/to/sim.josh",
            "sweep": {
                "parameters": [
                    {"name": "maxGrowth", "values": [10, 20, 30]},
                    {"name": "survivalProb", "values": [0.8, 0.9]},
                ],
            },
            "file_mappings": {
                "climate": "/data/climate.jshd",
            },
        }

        session_id = self.registry.create_session(
            experiment_name="sweep_test",
            metadata={"job_config": job_config_dict},
        )

        session = self.registry.get_session(session_id)
        config = session.job_config

        self.assertIsInstance(config, JobConfig)
        self.assertEqual(config.replicates, 5)
        self.assertIsNotNone(config.sweep)
        self.assertEqual(len(config.sweep.parameters), 2)
        self.assertEqual(config.sweep.parameters[0].name, "maxGrowth")
        self.assertEqual(config.sweep.parameters[0].values, [10, 20, 30])
        self.assertIn("climate", config.file_mappings)

    def test_job_config_enables_session_reconstruction(self):
        """job_config property should enable session reconstruction pattern."""
        from joshpy.jobs import JobConfig

        # Original config
        original_config = JobConfig(
            simulation="TestSim",
            replicates=2,
            template_string="testParam = {{ testParam }}",
        )

        # Store in session
        session_id = self.registry.create_session(
            experiment_name="reconstruction_test",
            metadata={"job_config": original_config.to_dict()},
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


if __name__ == "__main__":
    unittest.main()
