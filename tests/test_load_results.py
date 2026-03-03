"""Unit tests for load_job_results and LoadConfig."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from joshpy.sweep import (
    LoadConfig,
    ResultLoadError,
    _wait_for_file,
    load_job_results,
)
from joshpy.jobs import ExpandedJob


class TestLoadConfig(unittest.TestCase):
    """Tests for LoadConfig dataclass."""

    def test_default_values(self):
        """Default values should be reasonable."""
        config = LoadConfig()
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_delay, 0.5)
        self.assertEqual(config.settle_delay, 0.2)
        self.assertFalse(config.raise_on_missing)

    def test_custom_values(self):
        """Custom values should be set."""
        config = LoadConfig(
            max_retries=5,
            retry_delay=1.0,
            settle_delay=0.5,
            raise_on_missing=True,
        )
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.retry_delay, 1.0)
        self.assertEqual(config.settle_delay, 0.5)
        self.assertTrue(config.raise_on_missing)


class TestResultLoadError(unittest.TestCase):
    """Tests for ResultLoadError exception."""

    def test_message_format(self):
        """Error message should include job hash and context."""
        job = MagicMock()
        job.run_hash = "abc123def456"

        error = ResultLoadError(
            job=job,
            succeeded_before=5,
            message="CSV not found",
        )

        self.assertIn("abc123def456", str(error))
        self.assertIn("5 jobs succeeded", str(error))
        self.assertIn("CSV not found", str(error))

    def test_attributes(self):
        """Error should have accessible attributes."""
        job = MagicMock()
        job.run_hash = "abc123def456"

        error = ResultLoadError(
            job=job,
            succeeded_before=3,
            message="Test message",
        )

        self.assertIs(error.job, job)
        self.assertEqual(error.succeeded_before, 3)
        self.assertEqual(error.message, "Test message")


class TestWaitForFile(unittest.TestCase):
    """Tests for _wait_for_file function."""

    def test_returns_true_for_existing_file(self):
        """Should return True for existing non-empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"content")
            path = Path(f.name)

        try:
            config = LoadConfig(settle_delay=0.01)  # Speed up test
            result = _wait_for_file(path, config)
            self.assertTrue(result)
        finally:
            path.unlink()

    def test_returns_false_for_missing_file(self):
        """Should return False after retries for missing file."""
        path = Path("/nonexistent/file.csv")
        config = LoadConfig(max_retries=2, retry_delay=0.01, settle_delay=0.01)
        result = _wait_for_file(path, config)
        self.assertFalse(result)

    def test_returns_false_for_empty_file(self):
        """Should return False for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = Path(f.name)

        try:
            config = LoadConfig(max_retries=2, retry_delay=0.01, settle_delay=0.01)
            result = _wait_for_file(path, config)
            self.assertFalse(result)
        finally:
            path.unlink()

    def test_retries_on_missing(self):
        """Should retry multiple times for missing file."""
        path = Path("/nonexistent/file.csv")
        config = LoadConfig(max_retries=3, retry_delay=0.01, settle_delay=0.01)

        start = time.time()
        _wait_for_file(path, config)
        elapsed = time.time() - start

        # Should have waited at least (max_retries - 1) * retry_delay
        self.assertGreaterEqual(elapsed, 0.02)


class TestLoadJobResults(unittest.TestCase):
    """Tests for load_job_results function."""

    def setUp(self):
        """Set up mock objects."""
        self.cli = MagicMock()
        self.registry = MagicMock()
        self.export_paths = MagicMock()

        # Create a mock job
        self.job = MagicMock(spec=ExpandedJob)
        self.job.run_hash = "abc123def456"
        self.job.simulation = "Main"
        self.job.replicates = 1
        self.job.parameters = {"x": 10}
        self.job.custom_tags = {"run_hash": "abc123def456"}

    def test_returns_zero_for_no_path_template(self):
        """Should return 0 if no export path configured."""
        self.export_paths.get_patch_path.return_value = None

        with patch("joshpy.sweep._get_export_path", return_value=None):
            result = load_job_results(
                cli=self.cli,
                job=self.job,
                registry=self.registry,
                export_paths=self.export_paths,
            )

        self.assertEqual(result, 0)

    def test_returns_zero_for_no_runs(self):
        """Should return 0 if job not in registry."""
        self.registry.get_runs_for_hash.return_value = []

        with patch("joshpy.sweep._get_export_path", return_value="/path/to/{replicate}.csv"):
            result = load_job_results(
                cli=self.cli,
                job=self.job,
                registry=self.registry,
                export_paths=self.export_paths,
            )

        self.assertEqual(result, 0)

    def test_raises_on_missing_with_flag(self):
        """Should raise ResultLoadError if raise_on_missing=True."""
        run_info = MagicMock()
        run_info.run_id = "run-123"
        self.registry.get_runs_for_hash.return_value = [run_info]

        self.export_paths.resolve_path.return_value = Path("/nonexistent/file.csv")

        config = LoadConfig(
            raise_on_missing=True,
            max_retries=1,
            retry_delay=0.01,
            settle_delay=0.01,
        )

        mock_loader = MagicMock()
        
        with patch("joshpy.sweep._get_export_path", return_value="/path/{replicate}.csv"):
            with patch("joshpy.sweep.CellDataLoader", return_value=mock_loader):
                with self.assertRaises(ResultLoadError) as ctx:
                    load_job_results(
                        cli=self.cli,
                        job=self.job,
                        registry=self.registry,
                        export_paths=self.export_paths,
                        load_config=config,
                        succeeded_before=5,
                    )

                self.assertEqual(ctx.exception.succeeded_before, 5)
                self.assertIs(ctx.exception.job, self.job)

    def test_skips_missing_without_raise(self):
        """Should skip missing files when raise_on_missing=False."""
        run_info = MagicMock()
        run_info.run_id = "run-123"
        self.registry.get_runs_for_hash.return_value = [run_info]

        self.export_paths.resolve_path.return_value = Path("/nonexistent/file.csv")

        config = LoadConfig(
            raise_on_missing=False,
            max_retries=1,
            retry_delay=0.01,
            settle_delay=0.01,
        )

        mock_loader = MagicMock()

        with patch("joshpy.sweep._get_export_path", return_value="/path/{replicate}.csv"):
            with patch("joshpy.sweep.CellDataLoader", return_value=mock_loader):
                result = load_job_results(
                    cli=self.cli,
                    job=self.job,
                    registry=self.registry,
                    export_paths=self.export_paths,
                    load_config=config,
                    quiet=True,
                )

        self.assertEqual(result, 0)

    def test_loads_existing_file(self):
        """Should load existing CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n")
            csv_path = Path(f.name)

        try:
            run_info = MagicMock()
            run_info.run_id = "run-123"
            self.registry.get_runs_for_hash.return_value = [run_info]

            self.export_paths.resolve_path.return_value = csv_path

            # Mock CellDataLoader
            mock_loader = MagicMock()
            mock_loader.load_csv.return_value = 100

            config = LoadConfig(settle_delay=0.01)

            with patch("joshpy.sweep._get_export_path", return_value="/path/{replicate}.csv"):
                with patch("joshpy.sweep.CellDataLoader", return_value=mock_loader):
                    result = load_job_results(
                        cli=self.cli,
                        job=self.job,
                        registry=self.registry,
                        export_paths=self.export_paths,
                        load_config=config,
                        quiet=True,
                    )

            self.assertEqual(result, 100)
            mock_loader.load_csv.assert_called_once()
        finally:
            csv_path.unlink()


class TestRetryBehavior(unittest.TestCase):
    """Tests specifically for retry behavior."""

    def test_file_appears_after_first_retry(self):
        """Should succeed if file appears after first retry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"

            call_count = 0
            original_exists = Path.exists

            def delayed_exists(self):
                nonlocal call_count
                if self == csv_path:
                    call_count += 1
                    if call_count < 2:
                        return False
                    return original_exists(self)
                return original_exists(self)

            # Create the file
            csv_path.write_text("col1,col2\n1,2\n")

            config = LoadConfig(max_retries=3, retry_delay=0.01, settle_delay=0.01)

            with patch.object(Path, "exists", delayed_exists):
                result = _wait_for_file(csv_path, config)

            self.assertTrue(result)
            self.assertGreaterEqual(call_count, 2)

    def test_settle_delay_applied(self):
        """Should apply settle delay after file exists."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"content")
            path = Path(f.name)

        try:
            config = LoadConfig(settle_delay=0.1)

            with patch("time.sleep") as mock_sleep:
                _wait_for_file(path, config)

                # Should have called sleep with settle_delay
                mock_sleep.assert_called_with(0.1)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
