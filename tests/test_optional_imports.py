"""Regression tests for joshpy's optional-dependency import guards.

These exercise `import joshpy` in a fresh interpreter with numpy/pandas/matplotlib
made unavailable, to confirm the package's thin surface (jar/cli/jfr/config_parser/
debug/targets) really only needs `requests`. This can't be done in-process: by the
time this test module runs, pytest's own collection has likely already imported
numpy/pandas via other test modules, so blocking them here wouldn't reflect a real
"not installed" environment.
"""

import subprocess
import sys
import unittest

_BLOCK_HEAVY_DEPS_AND_IMPORT_JOSHPY = """
import sys
import importlib.abc

class _BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".")[0]
        if top in ("numpy", "pandas", "matplotlib"):
            raise ImportError(f"No module named {top!r} (simulated block)")
        return None

sys.meta_path.insert(0, _BlockFinder())

import joshpy

assert joshpy.HAS_JSHD is False, "HAS_JSHD should be False when numpy/pandas/matplotlib are absent"
assert joshpy.JarManager is not None
assert joshpy.JoshCLI is not None
print("OK")
"""


class TestThinInstallGuard(unittest.TestCase):
    """`import joshpy` must succeed without numpy/pandas/matplotlib."""

    def test_import_succeeds_without_heavy_deps(self):
        result = subprocess.run(
            [sys.executable, "-c", _BLOCK_HEAVY_DEPS_AND_IMPORT_JOSHPY],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"import joshpy failed with numpy/pandas/matplotlib blocked:\n{result.stderr}",
        )
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
