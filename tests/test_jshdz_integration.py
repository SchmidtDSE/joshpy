"""End-to-end integration test for compress=True .jshdz output.

Verifies:
  1. preprocess_csv(compress=True) produces a .jshdz file at the expected path.
  2. The file is valid XZ-compressed (XZ magic header + decompresses cleanly).
  3. A trivial Josh simulation can read the .jshdz via ``external <name>`` and
     export a CSV (no joshpy/JAR mocking — real JAR, real disk I/O).

Requires: Josh JAR (``pixi run get-jars``). Does NOT require MinIO.

Run with::

    pixi run -e dev test-integration tests/test_jshdz_integration.py -v
"""
from __future__ import annotations

import lzma
from pathlib import Path
from textwrap import dedent

import pytest

from joshpy.cli import RunConfig
from joshpy.grid import GridSpec

pytestmark = pytest.mark.integration

XZ_MAGIC = b"\xfd7zXZ\x00"


class TestJshdzCompressIntegration:
    """End-to-end: produce .jshdz via compress=True, then consume it in a sim."""

    def test_csv_compress_roundtrip(self, josh_cli, tmp_path):
        # 1. Build a tiny CSV input that fits inside a small grid.
        csv_input = tmp_path / "soil.csv"
        csv_input.write_text(dedent("""\
            longitude,latitude,value
            -116.025,33.925,5.0
            -116.020,33.920,7.5
            -116.015,33.915,10.0
        """))

        # 2. Preprocess with compress=True. Grid covers the CSV points.
        grid = GridSpec(
            name="jshdz-integration",
            output_dir=tmp_path / "out",
            size_m=30,
            # Josh's preprocess command treats `low` as the top-left (NW) corner
            # and `high` as the bottom-right (SE) corner — top-left.Y must
            # exceed bottom-right.Y. Matches test4_fixture.josh.
            low=(33.95, -116.05),
            high=(33.9, -116.0),
            steps=1,
        )
        result = grid.preprocess_csv(
            josh_cli,
            josh_name="soil_quality",
            data_file=csv_input,
            variable="value",
            units="count",
            timestep=0,
            compress=True,
        )
        assert result.success, f"preprocess failed: {result.stderr}"

        # 3. The .jshdz file exists at the expected location.
        jshdz = tmp_path / "out" / "soil_quality.jshdz"
        assert jshdz.exists(), f"expected {jshdz} to exist"
        assert grid.files["soil_quality"]["path"] == "soil_quality.jshdz"

        # 4. The file is valid XZ-compressed.
        with open(jshdz, "rb") as f:
            magic = f.read(6)
        assert magic == XZ_MAGIC, (
            f"expected XZ magic {XZ_MAGIC.hex()}, got {magic.hex()}. "
            "JAR did not honor the .jshdz suffix on the write side — "
            "this is a josh-side regression; document in JOSH_FIXES.md."
        )
        with lzma.open(jshdz, "rb") as f:
            decompressed = f.read()
        assert len(decompressed) > 0

        # 5. A trivial Josh sim consumes the .jshdz via ``external`` and exports.
        output_csv = tmp_path / "output.csv"
        sim = tmp_path / "consume.josh"
        sim.write_text(dedent(f"""\
            start simulation Main
              grid.size = 30 m
              grid.low = 33.95 degrees latitude, -116.05 degrees longitude
              grid.high = 33.9 degrees latitude, -116.0 degrees longitude
              grid.patch = "Default"
              steps.low = 0 count
              steps.high = 0 count
              exportFiles.patch = "file://{output_csv}"
            end simulation

            start patch Default
              soil_quality.step = external soil_quality
              export.meanSoil.step = mean(soil_quality)
            end patch
        """))

        run_result = josh_cli.run(RunConfig(
            script=sim,
            simulation="Main",
            replicates=1,
            data={"soil_quality": jshdz},  # explicit .jshdz path
        ))
        assert run_result.success, f"run failed: {run_result.stderr}"

        # 6. Output CSV exists and is non-empty (header + at least one row).
        assert output_csv.exists(), (
            f"expected {output_csv}; stdout: {run_result.stdout!r}"
        )
        rows = output_csv.read_text().strip().splitlines()
        assert len(rows) >= 2, f"expected header+data, got {rows!r}"

    def test_compress_default_false_unchanged_e2e(self, josh_cli, tmp_path):
        """Regression sanity: compress=False (default) still produces .jshd."""
        csv_input = tmp_path / "soil.csv"
        csv_input.write_text(
            "longitude,latitude,value\n-116.025,33.925,5.0\n"
        )

        grid = GridSpec(
            name="default-integration",
            output_dir=tmp_path / "out",
            size_m=30,
            # Josh's preprocess command treats `low` as the top-left (NW) corner
            # and `high` as the bottom-right (SE) corner — top-left.Y must
            # exceed bottom-right.Y. Matches test4_fixture.josh.
            low=(33.95, -116.05),
            high=(33.9, -116.0),
            steps=1,
        )
        result = grid.preprocess_csv(
            josh_cli,
            josh_name="soil_quality",
            data_file=csv_input,
            variable="value",
            units="count",
            timestep=0,
        )
        assert result.success, f"preprocess failed: {result.stderr}"
        assert (tmp_path / "out" / "soil_quality.jshd").exists()
        assert not (tmp_path / "out" / "soil_quality.jshdz").exists()
