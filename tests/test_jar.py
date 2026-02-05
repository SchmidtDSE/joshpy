"""Unit tests for the jar module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from joshpy.jar import (
    JarMode,
    JarManager,
    get_jar,
    download_jars,
    get_jar_hash,
    get_jar_version,
    JAR_URLS,
    JAR_FILENAMES,
    DEFAULT_JAR_DIR,
    DownloadResult,
)


class TestJarMode(unittest.TestCase):
    """Tests for JarMode enum."""

    def test_modes_exist(self):
        """All expected modes should exist."""
        self.assertEqual(JarMode.PROD.value, "prod")
        self.assertEqual(JarMode.DEV.value, "dev")
        self.assertEqual(JarMode.LOCAL.value, "local")

    def test_urls_defined(self):
        """URLs should be defined for PROD and DEV."""
        self.assertIn(JarMode.PROD, JAR_URLS)
        self.assertIn(JarMode.DEV, JAR_URLS)
        self.assertNotIn(JarMode.LOCAL, JAR_URLS)

    def test_filenames_defined(self):
        """Filenames should be defined for all modes."""
        for mode in JarMode:
            self.assertIn(mode, JAR_FILENAMES)


class TestGetJarHash(unittest.TestCase):
    """Tests for get_jar_hash function."""

    def test_nonexistent_file(self):
        """Should return None for nonexistent file."""
        result = get_jar_hash(Path("/nonexistent/file.jar"))
        self.assertIsNone(result)

    def test_existing_file(self):
        """Should return hash for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()

            result = get_jar_hash(Path(f.name))
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 64)  # SHA256 hex digest

            Path(f.name).unlink()

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        content = b"identical content"

        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(content)
            f1.flush()
            hash1 = get_jar_hash(Path(f1.name))
            Path(f1.name).unlink()

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(content)
            f2.flush()
            hash2 = get_jar_hash(Path(f2.name))
            Path(f2.name).unlink()

        self.assertEqual(hash1, hash2)


class TestJarManager(unittest.TestCase):
    """Tests for JarManager class."""

    def test_default_jar_dir(self):
        """Should use default jar directory."""
        manager = JarManager()
        self.assertEqual(manager.jar_dir, DEFAULT_JAR_DIR)

    def test_custom_jar_dir(self):
        """Should use custom jar directory."""
        custom_dir = Path("/custom/jars")
        manager = JarManager(jar_dir=custom_dir)
        self.assertEqual(manager.jar_dir, custom_dir)

    def test_get_jar_path(self):
        """Should return correct paths for each mode."""
        manager = JarManager(jar_dir=Path("/jars"))

        self.assertEqual(
            manager.get_jar_path(JarMode.PROD),
            Path("/jars/joshsim-fat-prod.jar")
        )
        self.assertEqual(
            manager.get_jar_path(JarMode.DEV),
            Path("/jars/joshsim-fat-dev.jar")
        )
        self.assertEqual(
            manager.get_jar_path(JarMode.LOCAL),
            Path("/jars/joshsim-fat.jar")
        )

    def test_jar_exists_false(self):
        """Should return False for nonexistent jar."""
        manager = JarManager(jar_dir=Path("/nonexistent"))
        self.assertFalse(manager.jar_exists(JarMode.PROD))

    def test_jar_exists_true(self):
        """Should return True for existing jar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jar_dir = Path(tmpdir)
            jar_path = jar_dir / JAR_FILENAMES[JarMode.PROD]
            jar_path.write_bytes(b"fake jar")

            manager = JarManager(jar_dir=jar_dir)
            self.assertTrue(manager.jar_exists(JarMode.PROD))

    def test_download_local_raises(self):
        """Should raise ValueError for LOCAL mode download."""
        manager = JarManager()
        with self.assertRaises(ValueError):
            manager.download_jar(JarMode.LOCAL)

    @patch('requests.get')
    def test_download_success(self, mock_get):
        """Should download jar successfully."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = MagicMock(return_value=[b"x" * 100])
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JarManager(jar_dir=Path(tmpdir))
            result = manager.download_jar(JarMode.PROD, force=True)

            self.assertTrue(result.success)
            self.assertIsNotNone(result.new_hash)
            self.assertTrue(result.jar_path.exists())

    @patch('requests.get')
    def test_download_failure(self, mock_get):
        """Should handle download failure gracefully."""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JarManager(jar_dir=Path(tmpdir))
            result = manager.download_jar(JarMode.PROD, force=True)

            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)

    def test_get_jar_local_not_found(self):
        """Should raise FileNotFoundError for missing local jar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JarManager(jar_dir=Path(tmpdir))
            with self.assertRaises(FileNotFoundError):
                manager.get_jar(JarMode.LOCAL, auto_download=False)

    def test_get_jar_local_exists(self):
        """Should return path for existing local jar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jar_dir = Path(tmpdir)
            local_jar = jar_dir / JAR_FILENAMES[JarMode.LOCAL]
            local_jar.write_bytes(b"fake jar")

            manager = JarManager(jar_dir=jar_dir)
            result = manager.get_jar(JarMode.LOCAL)
            self.assertEqual(result, local_jar)

    def test_get_jar_custom_local_path(self):
        """Should use custom local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_jar = Path(tmpdir) / "custom.jar"
            custom_jar.write_bytes(b"custom jar")

            manager = JarManager()
            result = manager.get_jar(JarMode.LOCAL, local_path=custom_jar)
            self.assertEqual(result, custom_jar)

    def test_get_info(self):
        """Should return correct info dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jar_dir = Path(tmpdir)
            jar_path = jar_dir / JAR_FILENAMES[JarMode.PROD]
            jar_path.write_bytes(b"fake jar content")

            manager = JarManager(jar_dir=jar_dir)
            info = manager.get_info(JarMode.PROD)

            self.assertEqual(info["path"], str(jar_path))
            self.assertTrue(info["exists"])
            self.assertIsNotNone(info["hash"])


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module-level convenience functions."""

    def test_get_jar_local(self):
        """get_jar should work with LOCAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jar_dir = Path(tmpdir)
            local_jar = jar_dir / JAR_FILENAMES[JarMode.LOCAL]
            local_jar.write_bytes(b"fake jar")

            result = get_jar(JarMode.LOCAL, jar_dir=jar_dir)
            self.assertEqual(result, local_jar)


if __name__ == '__main__':
    unittest.main()
