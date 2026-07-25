"""End-to-end integration tests for .jshd/.jshdz output.

Verifies:
  1. CSV preprocessing produces a scalar .jshdz that a simulation can read.
  2. NetCDF preprocessing persists declared temporal metadata in both .jshdz
     and .jshd that a simulation uses through ``length of external`` and
     ``at year <coordinate>``.
  3. Compressed output has valid XZ framing (no joshpy/JAR mocking — real JAR,
     real disk I/O).

Requires: Josh JAR (``pixi run get-jars``). Does NOT require MinIO.

Run with::

    pixi run -e dev test-integration tests/test_jshdz_integration.py -v
"""

from __future__ import annotations

import csv
import io
import lzma
from textwrap import dedent

import pytest
from scipy.io import netcdf_file

from joshpy.cli import RunConfig
from joshpy.grid import GridSpec, TimeAxis

pytestmark = pytest.mark.integration

XZ_MAGIC = b"\xfd7zXZ\x00"


class TestJshdzCompressIntegration:
    """End-to-end: produce .jshdz via compress=True, then consume it in a sim."""

    def test_csv_compress_roundtrip(self, josh_cli, tmp_path):
        # 1. Build a tiny CSV input that fits inside a small grid.
        csv_input = tmp_path / "soil.csv"
        csv_input.write_text(
            dedent("""\
            longitude,latitude,value
            -116.025,33.925,5.0
            -116.020,33.920,7.5
            -116.015,33.915,10.0
        """)
        )

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
        sim.write_text(
            dedent(f"""\
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
        """)
        )

        run_result = josh_cli.run(
            RunConfig(
                script=sim,
                simulation="Main",
                replicates=1,
                data={"soil_quality": jshdz},  # explicit .jshdz path
            )
        )
        assert run_result.success, f"run failed: {run_result.stderr}"

        # 6. Output CSV exists and is non-empty (header + at least one row).
        assert output_csv.exists(), f"expected {output_csv}; stdout: {run_result.stdout!r}"
        rows = output_csv.read_text().strip().splitlines()
        assert len(rows) >= 2, f"expected header+data, got {rows!r}"

    def _write_temporal_netcdf(self, tmp_path):
        """Two-timestep, 2x2 NetCDF fixture shared by jshd/jshdz temporal tests."""
        netcdf_input = tmp_path / "temperature.nc"
        with netcdf_file(netcdf_input, "w") as dataset:
            dataset.createDimension("time", 2)
            dataset.createDimension("lat", 2)
            dataset.createDimension("lon", 2)
            dataset.createVariable("time", "f8", ("time",))[:] = [2015, 2016]
            dataset.createVariable("lat", "f8", ("lat",))[:] = [33.901, 33.9]
            dataset.createVariable("lon", "f8", ("lon",))[:] = [-116.001, -116.0]
            dataset.createVariable(
                "temperature",
                "f4",
                ("time", "lat", "lon"),
            )[:] = [[[10, 11], [12, 13]], [[20, 21], [22, 23]]]
        return netcdf_input

    def _run_temporal_netcdf_roundtrip(self, josh_cli, tmp_path, *, compress, grid_name):
        """Preprocess temporal metadata, consume it in a run, return (result, output_csv).

        Shared by the compressed (.jshdz) and uncompressed (.jshd) temporal
        roundtrip tests below -- both exercise the same declared count/year
        axis and the same ``external ... at year ...`` read.
        """
        netcdf_input = self._write_temporal_netcdf(tmp_path)

        grid = GridSpec(
            name=grid_name,
            output_dir=tmp_path / "out",
            size_m=30,
            low=(33.901, -116.001),
            high=(33.9, -116.0),
            time=TimeAxis(
                type="count",
                start=2015,
                unit="year",
                count=2,
                increment=1,
            ),
        )
        preprocess_result = grid.preprocess_netcdf(
            josh_cli,
            josh_name="temperature",
            data_file=netcdf_input,
            variable="temperature",
            units="celsius",
            compress=compress,
        )
        assert preprocess_result.success, f"preprocess failed: {preprocess_result.stderr}"

        extension = "jshdz" if compress else "jshd"
        data_file = tmp_path / "out" / f"temperature.{extension}"
        assert data_file.exists(), f"expected {data_file} to exist"

        output_csv = tmp_path / "temporal-output.csv"
        sim = tmp_path / "consume-temporal.josh"
        sim.write_text(
            dedent(f"""\
            start simulation Main
              grid.size = 30 m
              grid.low = 33.901 degrees latitude, -116.001 degrees longitude
              grid.high = 33.9 degrees latitude, -116.0 degrees longitude
              grid.patch = "Default"
              axisLength.constant = length of external temperature
              steps.low = 0 count
              steps.high = axisLength - 1 count
              exportFiles.patch = "file://{output_csv}"
            end simulation

            start unit year
              alias years
            end unit

            start patch Default
              # Declared axis unit is "year" (--time-unit year), so the read
              # clause keyword must be "year", not "time" -- "at time" is only
              # for ISO axes. Without a simulation-level calendar declaration,
              # meta.year yields the raw 0-based step, so the real axis
              # coordinate (2015, 2016, ...) has to be computed explicitly.
              year.step = meta.year + 2015 year
              temperature.step = external temperature at year year
              export.meanTemperature.step = mean(temperature)
            end patch
        """)
        )

        run_result = josh_cli.run(
            RunConfig(
                script=sim,
                simulation="Main",
                replicates=1,
                data={"temperature": data_file},
            )
        )
        return data_file, run_result, output_csv

    def _assert_temporal_output(self, run_result, output_csv):
        assert run_result.success, f"run failed: {run_result.stderr}"

        # exportFiles.patch writes one row per patch per step (not a
        # grid-level aggregate), so check per-step value sets rather than
        # a fixed row count.
        rows = list(csv.DictReader(io.StringIO(output_csv.read_text())))
        assert rows, f"expected at least one exported row, got {output_csv.read_text()!r}"

        by_step = {"0": {10, 11, 12, 13}, "1": {20, 21, 22, 23}}
        seen_steps = {row["step"] for row in rows}
        assert seen_steps == set(by_step), f"expected steps 0 and 1, got {seen_steps}"
        for step, expected_values in by_step.items():
            # Patches outside the source's coverage default to 0; only
            # covered cells are checked against the source data.
            observed = {
                float(row["meanTemperature"])
                for row in rows
                if row["step"] == step and float(row["meanTemperature"]) != 0
            }
            assert observed == expected_values, (
                f"step {step}: expected {expected_values}, got {observed}"
            )

    def test_netcdf_temporal_compress_roundtrip(self, josh_cli, tmp_path):
        """Preprocess temporal metadata into .jshdz and consume it in a run."""
        jshdz, run_result, output_csv = self._run_temporal_netcdf_roundtrip(
            josh_cli, tmp_path, compress=True, grid_name="temporal-jshdz-integration"
        )
        with open(jshdz, "rb") as f:
            assert f.read(6) == XZ_MAGIC
        self._assert_temporal_output(run_result, output_csv)

    def test_netcdf_temporal_default_uncompressed_e2e(self, josh_cli, tmp_path):
        """Same temporal roundtrip as above, but with compress=False (.jshd).

        The --time-* flag support and interpolation fix documented in
        JOSH_FIXES.md were only ever verified against .jshdz; this confirms
        the plain, uncompressed .jshd path reads declared temporal metadata
        identically.
        """
        jshd, run_result, output_csv = self._run_temporal_netcdf_roundtrip(
            josh_cli, tmp_path, compress=False, grid_name="temporal-jshd-integration"
        )
        with open(jshd, "rb") as f:
            assert f.read(6) != XZ_MAGIC
        self._assert_temporal_output(run_result, output_csv)

    def test_compress_default_false_unchanged_e2e(self, josh_cli, tmp_path):
        """Regression sanity: compress=False (default) still produces .jshd."""
        csv_input = tmp_path / "soil.csv"
        csv_input.write_text("longitude,latitude,value\n-116.025,33.925,5.0\n")

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


class TestPreprocessMismatchedNativeLengths:
    """Regression for josh#494's --time-count invariant.

    A single GridSpec is often reused to preprocess multiple NetCDF sources
    with different native temporal lengths against the same spatial grid --
    e.g. a shorter historical/spinup series alongside a longer future/
    scenario series (the josh-models ``dev_fine`` grid's standard pattern).
    Josh validates a declared ``--time-count`` against the preprocessing
    stub's own step range, so that stub must be sized per-call from each
    source's own declared axis, not from the grid's nominal ``steps``.
    """

    @staticmethod
    def _write_netcdf(path, values):
        with netcdf_file(path, "w") as dataset:
            dataset.createDimension("time", len(values))
            dataset.createDimension("lat", 2)
            dataset.createDimension("lon", 2)
            dataset.createVariable("time", "f8", ("time",))[:] = list(range(len(values)))
            dataset.createVariable("lat", "f8", ("lat",))[:] = [33.901, 33.9]
            dataset.createVariable("lon", "f8", ("lon",))[:] = [-116.001, -116.0]
            dataset.createVariable("temperature", "f4", ("time", "lat", "lon"))[:] = values

    def test_shorter_and_longer_sources_both_preprocess_against_shared_grid(
        self, josh_cli, tmp_path
    ):
        # grid.steps (5) intentionally matches neither source's native length,
        # to prove the stub is sized per-call rather than from the grid.
        grid = GridSpec(
            name="mismatched-lengths",
            output_dir=tmp_path / "out",
            size_m=30,
            low=(33.901, -116.001),
            high=(33.9, -116.0),
            steps=5,
        )

        historical_nc = tmp_path / "historical.nc"
        self._write_netcdf(historical_nc, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2 slices

        future_nc = tmp_path / "future.nc"
        self._write_netcdf(
            future_nc,
            [[[10, 11], [12, 13]], [[20, 21], [22, 23]], [[30, 31], [32, 33]]],  # 3 slices
        )

        historical_result = grid.preprocess_netcdf(
            josh_cli,
            josh_name="historicalTemp",
            data_file=historical_nc,
            variable="temperature",
            units="celsius",
            time_type="count",
            time_start=1950,
            time_unit="year",
            time_count=2,
            time_increment=1,
        )
        assert historical_result.success, f"preprocess failed: {historical_result.stderr}"

        future_result = grid.preprocess_netcdf(
            josh_cli,
            josh_name="futureTemp",
            data_file=future_nc,
            variable="temperature",
            units="celsius",
            time_type="count",
            time_start=2015,
            time_unit="year",
            time_count=3,
            time_increment=1,
        )
        assert future_result.success, f"preprocess failed: {future_result.stderr}"

        assert (tmp_path / "out" / "historicalTemp.jshd").exists()
        assert (tmp_path / "out" / "futureTemp.jshd").exists()
