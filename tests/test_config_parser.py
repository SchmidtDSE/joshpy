"""Tests for joshpy.config_parser."""

import tempfile
import unittest
from pathlib import Path

from joshpy.config_parser import parse_jshc, parse_jshc_content


class TestParseJshcContent(unittest.TestCase):
    """Tests for parse_jshc_content()."""

    def test_basic_parsing(self):
        content = "maxGrowth = 50 meters\nmortalityRate = 0.1 percent"
        result = parse_jshc_content(content)
        self.assertEqual(result, {"maxGrowth": 50, "mortalityRate": 0.1})

    def test_int_vs_float(self):
        content = "intParam = 55 count\nfloatParam = 35.5 %"
        result = parse_jshc_content(content)
        self.assertIsInstance(result["intParam"], int)
        self.assertEqual(result["intParam"], 55)
        self.assertIsInstance(result["floatParam"], float)
        self.assertEqual(result["floatParam"], 35.5)

    def test_comments_skipped(self):
        content = "# This is a comment\nparam = 10 count\n# Another comment"
        result = parse_jshc_content(content)
        self.assertEqual(result, {"param": 10})

    def test_indented_comments_skipped(self):
        content = "  # Indented comment\nparam = 10 count"
        result = parse_jshc_content(content)
        self.assertEqual(result, {"param": 10})

    def test_blank_lines_skipped(self):
        content = "param1 = 10 count\n\n\nparam2 = 20 count\n"
        result = parse_jshc_content(content)
        self.assertEqual(result, {"param1": 10, "param2": 20})

    def test_empty_content(self):
        result = parse_jshc_content("")
        self.assertEqual(result, {})

    def test_comments_only(self):
        result = parse_jshc_content("# Just a comment\n# Another")
        self.assertEqual(result, {})

    def test_negative_values(self):
        content = "offset = -5 meters"
        result = parse_jshc_content(content)
        self.assertEqual(result, {"offset": -5})

    def test_negative_float(self):
        content = "offset = -0.5 percent"
        result = parse_jshc_content(content)
        self.assertEqual(result, {"offset": -0.5})

    def test_whitespace_handling(self):
        content = "  param  =  42  count  "
        result = parse_jshc_content(content)
        self.assertEqual(result, {"param": 42})

    def test_malformed_no_equals(self):
        with self.assertRaises(ValueError) as ctx:
            parse_jshc_content("this is not valid")
        self.assertIn("Line 1", str(ctx.exception))

    def test_malformed_empty_name(self):
        with self.assertRaises(ValueError):
            parse_jshc_content("= 10 count")

    def test_malformed_empty_value(self):
        with self.assertRaises(ValueError):
            parse_jshc_content("param =")

    def test_malformed_non_numeric(self):
        with self.assertRaises(ValueError) as ctx:
            parse_jshc_content("param = notanumber count")
        self.assertIn("cannot parse numeric value", str(ctx.exception))

    def test_real_world_jshc(self):
        """Parse a realistic .jshc file snippet."""
        content = """\
# Joshua Tree Model Configuration
# All literature citations inline

# --- COVER MAP INITIALISATION ---
coverHighThreshold = 2 %
coverMedThreshold = 0.4 %
coverLowThreshold = 0.1 %

# Tree densities per hectare
treesHighCoverPerHa = 55 count
treesMedCoverPerHa = 30 count

# Patch size
patchSizeMeters = 30 count

# --- FIRE ---
fireYear = 75 count

# --- SEED BANK ---
seedBankSurvival = 35 %
seedPerTree = 3800 count
harvestedByRodents = 95 %
"""
        result = parse_jshc_content(content)
        self.assertEqual(len(result), 10)
        self.assertEqual(result["coverHighThreshold"], 2)
        self.assertEqual(result["coverMedThreshold"], 0.4)
        self.assertEqual(result["seedPerTree"], 3800)
        self.assertEqual(result["harvestedByRodents"], 95)

    def test_value_without_unit(self):
        """A value with no unit token should still parse."""
        content = "param = 42"
        result = parse_jshc_content(content)
        self.assertEqual(result, {"param": 42})


class TestParseJshc(unittest.TestCase):
    """Tests for parse_jshc() (file-based)."""

    def test_parse_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jshc", delete=False) as f:
            f.write("maxGrowth = 50 meters\nfireYear = 75 count\n")
            f.flush()
            result = parse_jshc(f.name)
        self.assertEqual(result, {"maxGrowth": 50, "fireYear": 75})

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_jshc("/nonexistent/path/config.jshc")

    def test_accepts_path_object(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jshc", delete=False) as f:
            f.write("param = 10 count\n")
            f.flush()
            result = parse_jshc(Path(f.name))
        self.assertEqual(result, {"param": 10})


if __name__ == "__main__":
    unittest.main()
