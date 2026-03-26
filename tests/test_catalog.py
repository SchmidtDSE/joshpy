"""Unit tests for the catalog module."""

import tempfile
import unittest
from pathlib import Path

from joshpy.catalog import (
    ProjectCatalog,
    ExperimentInfo,
    compute_model_hash,
    compute_config_hash,
    compute_data_manifest_hash,
)
from joshpy.jobs import JobConfig, SweepConfig, ConfigSweepParameter


class TestHashUtilities(unittest.TestCase):
    """Tests for content hashing functions."""

    def test_compute_model_hash(self):
        """Model hash should be 12-char hex from file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.josh"
            model.write_text("start simulation Main\nend simulation")
            h = compute_model_hash(model)
            self.assertEqual(len(h), 12)
            self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_model_hash_deterministic(self):
        """Same content should produce same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = Path(tmpdir) / "a.josh"
            m2 = Path(tmpdir) / "b.josh"
            m1.write_text("content")
            m2.write_text("content")
            self.assertEqual(compute_model_hash(m1), compute_model_hash(m2))

    def test_model_hash_content_sensitive(self):
        """Different content should produce different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = Path(tmpdir) / "a.josh"
            m2 = Path(tmpdir) / "b.josh"
            m1.write_text("content A")
            m2.write_text("content B")
            self.assertNotEqual(compute_model_hash(m1), compute_model_hash(m2))

    def test_model_hash_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            compute_model_hash(Path("/nonexistent/model.josh"))

    def test_compute_config_hash(self):
        """Config hash should be 12-char hex from string content."""
        h = compute_config_hash("maxGrowth = 50 meters")
        self.assertEqual(len(h), 12)

    def test_compute_data_manifest_hash(self):
        """Manifest hash should reflect all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "a.jshd"
            f2 = Path(tmpdir) / "b.jshd"
            f1.write_bytes(b"data a")
            f2.write_bytes(b"data b")
            h = compute_data_manifest_hash({"a": f1, "b": f2})
            self.assertEqual(len(h), 12)

    def test_manifest_hash_order_independent(self):
        """Hash should be the same regardless of dict insertion order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "a.jshd"
            f2 = Path(tmpdir) / "b.jshd"
            f1.write_bytes(b"data a")
            f2.write_bytes(b"data b")
            h1 = compute_data_manifest_hash({"a": f1, "b": f2})
            h2 = compute_data_manifest_hash({"b": f2, "a": f1})
            self.assertEqual(h1, h2)

    def test_manifest_hash_content_sensitive(self):
        """Changing a file's content should change the manifest hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "a.jshd"
            f1.write_bytes(b"version 1")
            h1 = compute_data_manifest_hash({"a": f1})
            f1.write_bytes(b"version 2")
            h2 = compute_data_manifest_hash({"a": f1})
            self.assertNotEqual(h1, h2)


class TestProjectCatalogModels(unittest.TestCase):
    """Tests for model registration."""

    def setUp(self):
        self.catalog = ProjectCatalog(":memory:")
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        self.catalog.close()

    def test_register_model(self):
        """Should register a model and return its hash."""
        model = Path(self.tmpdir) / "model.josh"
        model.write_text("simulation Main")
        h = self.catalog.register_model(model)
        self.assertEqual(len(h), 12)

    def test_register_model_idempotent(self):
        """Re-registering same model should be a no-op."""
        model = Path(self.tmpdir) / "model.josh"
        model.write_text("simulation Main")
        h1 = self.catalog.register_model(model)
        h2 = self.catalog.register_model(model)
        self.assertEqual(h1, h2)

    def test_get_model(self):
        """Should retrieve model info by hash."""
        model = Path(self.tmpdir) / "model.josh"
        model.write_text("simulation Main")
        h = self.catalog.register_model(model, name="canonical")
        info = self.catalog.get_model(h)
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "canonical")
        self.assertIn("model.josh", info.path)

    def test_get_model_not_found(self):
        """Should return None for unknown hash."""
        self.assertIsNone(self.catalog.get_model("nonexistent"))

    def test_register_model_default_name(self):
        """Name should default to filename stem."""
        model = Path(self.tmpdir) / "canonical_v2.josh"
        model.write_text("simulation Main")
        h = self.catalog.register_model(model)
        info = self.catalog.get_model(h)
        self.assertEqual(info.name, "canonical_v2")


class TestProjectCatalogData(unittest.TestCase):
    """Tests for data manifest registration."""

    def setUp(self):
        self.catalog = ProjectCatalog(":memory:")
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        self.catalog.close()

    def test_register_data(self):
        """Should register data manifest and return hash."""
        f1 = Path(self.tmpdir) / "cover.jshd"
        f2 = Path(self.tmpdir) / "fire.jshd"
        f1.write_bytes(b"cover data")
        f2.write_bytes(b"fire data")
        h = self.catalog.register_data({"cover": f1, "fire": f2}, name="dev_fine")
        self.assertEqual(len(h), 12)

    def test_register_data_idempotent(self):
        """Re-registering same data should be a no-op."""
        f1 = Path(self.tmpdir) / "cover.jshd"
        f1.write_bytes(b"data")
        h1 = self.catalog.register_data({"cover": f1})
        h2 = self.catalog.register_data({"cover": f1})
        self.assertEqual(h1, h2)

    def test_get_data_manifest(self):
        """Should retrieve manifest info including inventory."""
        f1 = Path(self.tmpdir) / "cover.jshd"
        f1.write_bytes(b"cover data")
        h = self.catalog.register_data({"cover": f1}, name="test_data")
        info = self.catalog.get_data_manifest(h)
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "test_data")
        self.assertEqual(info.file_count, 1)
        self.assertIn("cover", info.file_inventory)


class TestProjectCatalogExperiments(unittest.TestCase):
    """Tests for experiment registration and lookup."""

    def setUp(self):
        self.catalog = ProjectCatalog(":memory:")
        self.tmpdir = tempfile.mkdtemp()
        # Create test files
        self.model = Path(self.tmpdir) / "model.josh"
        self.model.write_text("start simulation Main\nend simulation")
        self.config = Path(self.tmpdir) / "config.jshc"
        self.config.write_text("maxGrowth = 50 meters")
        self.data = Path(self.tmpdir) / "cover.jshd"
        self.data.write_bytes(b"cover data")

    def tearDown(self):
        self.catalog.close()

    def _make_job_config(self):
        return JobConfig(
            config_path=self.config,
            source_path=self.model,
            simulation="Main",
            replicates=1,
            file_mappings={"cover": self.data},
        )

    def test_register_experiment(self):
        """Should register an experiment and return its ID."""
        config = self._make_job_config()
        exp_id = self.catalog.register_experiment(
            config, registry_path="experiments/test.duckdb", name="test_exp"
        )
        self.assertIsNotNone(exp_id)
        self.assertTrue(len(exp_id) > 0)

    def test_register_experiment_auto_registers_model_and_data(self):
        """Registering experiment should also register model and data."""
        config = self._make_job_config()
        self.catalog.register_experiment(
            config, registry_path="experiments/test.duckdb"
        )
        # Model should be registered
        model_hash = compute_model_hash(self.model)
        self.assertIsNotNone(self.catalog.get_model(model_hash))
        # Data should be registered
        data_hash = compute_data_manifest_hash({"cover": self.data})
        self.assertIsNotNone(self.catalog.get_data_manifest(data_hash))

    def test_find_experiment(self):
        """Should find experiment matching the same config."""
        config = self._make_job_config()
        exp_id = self.catalog.register_experiment(
            config, registry_path="experiments/test.duckdb", name="findable"
        )
        found = self.catalog.find_experiment(config)
        self.assertIsNotNone(found)
        self.assertEqual(found.experiment_id, exp_id)
        self.assertEqual(found.name, "findable")

    def test_find_experiment_no_match(self):
        """Should return None when no match exists."""
        config = self._make_job_config()
        self.assertIsNone(self.catalog.find_experiment(config))

    def test_find_experiment_different_config(self):
        """Should not match when config content differs."""
        config1 = self._make_job_config()
        self.catalog.register_experiment(
            config1, registry_path="experiments/test.duckdb"
        )
        # Change config content
        self.config.write_text("maxGrowth = 100 meters")
        config2 = self._make_job_config()
        self.assertIsNone(self.catalog.find_experiment(config2))

    def test_get_experiment(self):
        """Should retrieve experiment by ID."""
        config = self._make_job_config()
        exp_id = self.catalog.register_experiment(
            config, registry_path="experiments/test.duckdb", name="by_id"
        )
        exp = self.catalog.get_experiment(exp_id)
        self.assertIsNotNone(exp)
        self.assertEqual(exp.name, "by_id")
        self.assertEqual(exp.simulation, "Main")
        self.assertEqual(exp.replicates, 1)

    def test_list_experiments(self):
        """Should list all experiments."""
        config = self._make_job_config()
        self.catalog.register_experiment(
            config, registry_path="a.duckdb", name="exp_a"
        )
        # Change config so it's a different experiment
        self.config.write_text("maxGrowth = 100 meters")
        config2 = self._make_job_config()
        self.catalog.register_experiment(
            config2, registry_path="b.duckdb", name="exp_b"
        )
        experiments = self.catalog.list_experiments()
        self.assertEqual(len(experiments), 2)

    def test_list_experiments_filter_status(self):
        """Should filter by status."""
        config = self._make_job_config()
        exp_id = self.catalog.register_experiment(
            config, registry_path="a.duckdb"
        )
        self.catalog.update_experiment_status(exp_id, "completed")
        pending = self.catalog.list_experiments(status="pending")
        completed = self.catalog.list_experiments(status="completed")
        self.assertEqual(len(pending), 0)
        self.assertEqual(len(completed), 1)

    def test_update_experiment_status(self):
        """Should update status and summary."""
        config = self._make_job_config()
        exp_id = self.catalog.register_experiment(
            config, registry_path="a.duckdb"
        )
        self.catalog.update_experiment_status(
            exp_id, "completed", summary={"succeeded": 3, "failed": 0}
        )
        exp = self.catalog.get_experiment(exp_id)
        self.assertEqual(exp.status, "completed")
        self.assertIsNotNone(exp.completed_at)
        self.assertEqual(exp.summary["succeeded"], 3)

    def test_orchestration_metadata(self):
        """Experiment should have joshpy orchestration metadata."""
        config = self._make_job_config()
        exp_id = self.catalog.register_experiment(
            config, registry_path="a.duckdb"
        )
        exp = self.catalog.get_experiment(exp_id)
        self.assertIsNotNone(exp.orchestration)
        self.assertEqual(exp.orchestration["tool"], "joshpy")
        self.assertIn("job_config", exp.orchestration)

    def test_orchestration_sweep_summary(self):
        """Orchestration should include sweep summary when sweep is configured."""
        config = JobConfig(
            template_string="maxGrowth = {{ maxGrowth }} meters",
            source_path=self.model,
            sweep=SweepConfig(
                config_parameters=[
                    ConfigSweepParameter(name="maxGrowth", values=[10, 50, 100]),
                ],
            ),
        )
        exp_id = self.catalog.register_experiment(
            config, registry_path="a.duckdb"
        )
        exp = self.catalog.get_experiment(exp_id)
        self.assertIn("sweep_summary", exp.orchestration)
        self.assertEqual(exp.orchestration["sweep_summary"]["parameters"], ["maxGrowth"])
        self.assertEqual(exp.orchestration["sweep_summary"]["total_combinations"], 3)


class TestProjectCatalogCrossQuery(unittest.TestCase):
    """Tests for cross-experiment query helper."""

    def test_open_registries_with_no_experiments(self):
        """Should yield a connection even with empty list."""
        catalog = ProjectCatalog(":memory:")
        try:
            with catalog.open_registries([]) as conn:
                self.assertIsNotNone(conn)
        finally:
            catalog.close()

    def test_open_registries_skips_missing_files(self):
        """Should skip experiments with missing registry files."""
        catalog = ProjectCatalog(":memory:")
        exp = ExperimentInfo(
            experiment_id="test",
            name="test",
            model_hash=None,
            config_hash=None,
            data_manifest_hash=None,
            simulation=None,
            replicates=None,
            registry_path="/nonexistent/registry.duckdb",
        )
        try:
            with catalog.open_registries([exp]) as conn:
                # Should succeed (skips missing file)
                self.assertIsNotNone(conn)
        finally:
            catalog.close()


class TestProjectCatalogLifecycle(unittest.TestCase):
    """Tests for catalog lifecycle management."""

    def test_context_manager(self):
        """Should work as context manager."""
        with ProjectCatalog(":memory:") as catalog:
            self.assertIsNotNone(catalog)

    def test_persistent_storage(self):
        """Data should persist across reopens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.duckdb"
            model = Path(tmpdir) / "model.josh"
            model.write_text("simulation")

            # Write
            with ProjectCatalog(db_path) as catalog:
                h = catalog.register_model(model, name="persistent")

            # Read
            with ProjectCatalog(db_path) as catalog:
                info = catalog.get_model(h)
                self.assertIsNotNone(info)
                self.assertEqual(info.name, "persistent")


if __name__ == "__main__":
    unittest.main()
