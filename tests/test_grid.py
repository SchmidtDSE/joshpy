"""Tests for joshpy.grid (GridSpec)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from joshpy.grid import GridSpec


class TestGridSpecConstruction(unittest.TestCase):
    """Tests for GridSpec constructor and properties."""

    def _make_grid(self, **kwargs):
        defaults = {
            "name": "dev_fine",
            "output_dir": Path("/tmp/test_grid"),
            "size_m": 30,
            "low": (33.902, -116.0465),
            "high": (33.908, -116.0395),
            "steps": 86,
        }
        defaults.update(kwargs)
        return GridSpec(**defaults)

    def test_constructor(self):
        grid = self._make_grid()
        self.assertEqual(grid.name, "dev_fine")
        self.assertEqual(grid.size_m, 30)
        self.assertEqual(grid.low, (33.902, -116.0465))
        self.assertEqual(grid.high, (33.908, -116.0395))
        self.assertEqual(grid.steps, 86)
        self.assertEqual(grid.files, {})

    def test_template_vars(self):
        grid = self._make_grid()
        tv = grid.template_vars
        self.assertEqual(tv["size_m"], 30)
        self.assertEqual(tv["low_lat"], 33.902)
        self.assertEqual(tv["low_lon"], -116.0465)
        self.assertEqual(tv["high_lat"], 33.908)
        self.assertEqual(tv["high_lon"], -116.0395)
        self.assertEqual(tv["steps"], 86)

    def test_file_mappings_empty(self):
        grid = self._make_grid()
        self.assertEqual(grid.file_mappings, {})

    def test_file_mappings_resolves_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(
                output_dir=Path(tmpdir),
                files={
                    "cover": {"path": "cover.jshd", "units": "percent"},
                    "fire": {"path": "subdir/fire.jshd", "units": "rbr"},
                },
            )
            mappings = grid.file_mappings
            self.assertEqual(
                mappings["cover"], (Path(tmpdir) / "cover.jshd").resolve()
            )
            self.assertEqual(
                mappings["fire"], (Path(tmpdir) / "subdir" / "fire.jshd").resolve()
            )


@unittest.skipIf(not HAS_YAML, "pyyaml not installed")
class TestGridSpecYaml(unittest.TestCase):
    """Tests for GridSpec YAML save/load."""

    def _make_grid(self, output_dir):
        return GridSpec(
            name="test_grid",
            output_dir=Path(output_dir),
            size_m=30,
            low=(33.9, -116.05),
            high=(33.95, -116.0),
            steps=100,
            files={
                "cover": {"path": "cover.jshd", "units": "percent"},
                "temp": {"path": "monthly/temp.jshd", "units": "K"},
            },
        )

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            saved_path = grid.save()

            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.name, "grid.yaml")

            loaded = GridSpec.from_yaml(saved_path)
            self.assertEqual(loaded.name, "test_grid")
            self.assertEqual(loaded.size_m, 30)
            self.assertEqual(loaded.low, (33.9, -116.05))
            self.assertEqual(loaded.high, (33.95, -116.0))
            self.assertEqual(loaded.steps, 100)
            self.assertIn("cover", loaded.files)
            self.assertIn("temp", loaded.files)
            self.assertEqual(loaded.files["cover"]["units"], "percent")
            self.assertEqual(loaded.files["temp"]["path"], "monthly/temp.jshd")

    def test_save_custom_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            custom = Path(tmpdir) / "custom" / "my_grid.yaml"
            saved_path = grid.save(custom)
            self.assertEqual(saved_path, custom)
            self.assertTrue(saved_path.exists())

    def test_save_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dir = Path(tmpdir) / "a" / "b" / "c"
            grid = GridSpec(
                name="deep",
                output_dir=deep_dir,
                size_m=30,
                low=(0, 0),
                high=(1, 1),
                steps=10,
            )
            saved = grid.save()
            self.assertTrue(saved.exists())

    def test_from_yaml_resolves_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            grid.save()
            loaded = GridSpec.from_yaml(Path(tmpdir) / "grid.yaml")
            self.assertEqual(loaded.output_dir, Path(tmpdir).resolve())

    def test_yaml_content_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            saved = grid.save()
            data = yaml.safe_load(saved.read_text())
            self.assertEqual(data["name"], "test_grid")
            self.assertIn("grid", data)
            self.assertEqual(data["grid"]["size_m"], 30)
            self.assertIn("files", data)
            self.assertIn("cover", data["files"])

    def test_empty_files_no_files_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="empty",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(0, 0),
                high=(1, 1),
                steps=10,
            )
            saved = grid.save()
            data = yaml.safe_load(saved.read_text())
            self.assertNotIn("files", data)


class TestGridSpecPreprocessScript(unittest.TestCase):
    """Tests for the internal preprocess script rendering."""

    def test_render_preprocess_script(self):
        grid = GridSpec(
            name="dev_fine",
            output_dir=Path("/tmp/test"),
            size_m=30,
            low=(33.902, -116.0465),
            high=(33.908, -116.0395),
            steps=86,
        )
        script_path = grid._render_preprocess_script()
        try:
            self.assertTrue(script_path.exists())
            content = script_path.read_text()
            self.assertIn("start simulation Preprocess", content)
            self.assertIn("grid.size = 30 m", content)
            self.assertIn("33.902 degrees latitude", content)
            self.assertIn("-116.0465 degrees longitude", content)
            self.assertIn("33.908 degrees latitude", content)
            self.assertIn("-116.0395 degrees longitude", content)
            self.assertIn("steps.high = 86 count", content)
            self.assertIn("start patch Default", content)
            self.assertIn("end patch", content)
        finally:
            script_path.unlink(missing_ok=True)


class TestGridSpecPreprocess(unittest.TestCase):
    """Tests for preprocess methods with mocked CLI."""

    def _make_grid(self, tmpdir):
        return GridSpec(
            name="test",
            output_dir=Path(tmpdir),
            size_m=30,
            low=(33.9, -116.05),
            high=(33.95, -116.0),
            steps=10,
        )

    def _mock_cli(self, success=True):
        cli = MagicMock()
        mock_result = MagicMock()
        mock_result.success = success
        mock_result.stdout = ""
        mock_result.stderr = ""
        cli.preprocess.return_value = mock_result
        return cli, mock_result

    def test_preprocess_geotiff_calls_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            cli, _ = self._mock_cli()

            tif = Path(tmpdir) / "test.tif"
            tif.write_bytes(b"fake tif")

            result = grid.preprocess_geotiff(
                cli,
                josh_name="cover",
                data_file=tif,
                band=0,
                units="percent",
                timestep=0,
            )

            cli.preprocess.assert_called_once()
            config_arg = cli.preprocess.call_args[0][0]
            self.assertEqual(config_arg.band, 0)
            self.assertEqual(config_arg.units, "percent")
            self.assertEqual(config_arg.simulation, "Preprocess")

    def test_preprocess_geotiff_registers_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            cli, _ = self._mock_cli(success=True)

            result = grid.preprocess_geotiff(
                cli,
                josh_name="cover",
                data_file=Path(tmpdir) / "test.tif",
                band=0,
                units="percent",
                timestep=0,
            )

            self.assertIn("cover", grid.files)
            self.assertEqual(grid.files["cover"]["path"], "cover.jshd")
            self.assertEqual(grid.files["cover"]["units"], "percent")

    def test_preprocess_geotiff_failure_does_not_register(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            cli, _ = self._mock_cli(success=False)

            result = grid.preprocess_geotiff(
                cli,
                josh_name="cover",
                data_file=Path(tmpdir) / "test.tif",
                band=0,
                units="percent",
                timestep=0,
            )

            self.assertNotIn("cover", grid.files)

    def test_preprocess_with_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            cli, _ = self._mock_cli(success=True)

            result = grid.preprocess_netcdf(
                cli,
                josh_name="futureTempJan",
                data_file=Path(tmpdir) / "tas.nc",
                variable="tas",
                units="K",
                subdirectory="monthly",
            )

            self.assertIn("futureTempJan", grid.files)
            # Path should be relative and include subdirectory
            self.assertEqual(
                grid.files["futureTempJan"]["path"],
                str(Path("monthly") / "futureTempJan.jshd"),
            )

    def test_preprocess_csv_calls_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            cli, _ = self._mock_cli()

            result = grid.preprocess_csv(
                cli,
                josh_name="stations",
                data_file=Path(tmpdir) / "data.csv",
                variable="temp",
                units="celsius",
                timestep=0,
            )

            cli.preprocess.assert_called_once()
            config_arg = cli.preprocess.call_args[0][0]
            self.assertEqual(config_arg.variable, "temp")
            self.assertEqual(config_arg.units, "celsius")

    def test_preprocess_cleans_up_temp_script(self):
        """Temp .josh file should be cleaned up even on failure."""
        import glob

        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            cli, _ = self._mock_cli(success=False)

            # Count .josh files in temp before
            before = set(glob.glob("/tmp/preprocess_*.josh"))

            grid.preprocess_geotiff(
                cli,
                josh_name="cover",
                data_file=Path(tmpdir) / "test.tif",
                band=0,
                units="percent",
                timestep=0,
            )

            # No new .josh files should remain
            after = set(glob.glob("/tmp/preprocess_*.josh"))
            new_files = after - before
            self.assertEqual(len(new_files), 0)

    def test_preprocess_idempotent(self):
        """Re-preprocessing with same josh_name updates the entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_grid(tmpdir)
            cli, _ = self._mock_cli(success=True)

            grid.preprocess_geotiff(
                cli,
                josh_name="cover",
                data_file=Path(tmpdir) / "v1.tif",
                band=0,
                units="percent",
                timestep=0,
            )
            grid.preprocess_geotiff(
                cli,
                josh_name="cover",
                data_file=Path(tmpdir) / "v2.tif",
                band=0,
                units="fraction",
                timestep=0,
            )

            self.assertEqual(len(grid.files), 1)
            self.assertEqual(grid.files["cover"]["units"], "fraction")


class TestGridSpecVariants(unittest.TestCase):
    """Tests for variant resolution (file_mappings, file_mappings_for)."""

    def _make_variant_grid(self, tmpdir):
        return GridSpec(
            name="dev_fine",
            output_dir=Path(tmpdir),
            size_m=30,
            low=(33.9, -116.05),
            high=(33.95, -116.0),
            steps=86,
            files={
                "cover": {"path": "cover.jshd", "units": "percent"},
                "fireRbr": {"path": "fireRbr.jshd", "units": "rbr"},
                "futureTempJan": {
                    "template_path": "monthly/tas_{scenario}_jan.jshd",
                    "units": "K",
                },
                "futureTempFeb": {
                    "template_path": "monthly/tas_{scenario}_feb.jshd",
                    "units": "K",
                },
                "futurePrecipJan": {
                    "template_path": "monthly/pr_{scenario}_jan.jshd",
                    "units": "mm/year",
                },
            },
            variants={
                "scenario": {
                    "values": ["ssp245", "ssp370", "ssp585"],
                    "default": "ssp245",
                },
            },
        )

    def test_file_mappings_resolves_template_path_with_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            mappings = grid.file_mappings
            expected = (Path(tmpdir) / "monthly/tas_ssp245_jan.jshd").resolve()
            self.assertEqual(mappings["futureTempJan"], expected)

    def test_file_mappings_static_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            mappings = grid.file_mappings
            self.assertEqual(
                mappings["cover"], (Path(tmpdir) / "cover.jshd").resolve()
            )

    def test_file_mappings_for_specific_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            mappings = grid.file_mappings_for(scenario="ssp370")
            expected = (Path(tmpdir) / "monthly/tas_ssp370_jan.jshd").resolve()
            self.assertEqual(mappings["futureTempJan"], expected)
            # Static files unchanged
            self.assertEqual(
                mappings["cover"], (Path(tmpdir) / "cover.jshd").resolve()
            )

    def test_file_mappings_for_invalid_axis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                grid.file_mappings_for(gcm="cesm2")
            self.assertIn("Unknown variant axis", str(ctx.exception))

    def test_file_mappings_for_invalid_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                grid.file_mappings_for(scenario="rcp45")
            self.assertIn("Invalid value", str(ctx.exception))

    def test_variants_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            self.assertIn("scenario", grid.variants)
            self.assertEqual(
                grid.variants["scenario"]["values"],
                ["ssp245", "ssp370", "ssp585"],
            )
            self.assertEqual(grid.variants["scenario"]["default"], "ssp245")


class TestGridSpecVariantSweep(unittest.TestCase):
    """Tests for variant_sweep()."""

    def _make_variant_grid(self, tmpdir):
        return GridSpec(
            name="dev_fine",
            output_dir=Path(tmpdir),
            size_m=30,
            low=(33.9, -116.05),
            high=(33.95, -116.0),
            steps=86,
            files={
                "cover": {"path": "cover.jshd", "units": "percent"},
                "futureTempJan": {
                    "template_path": "monthly/tas_{scenario}_jan.jshd",
                    "units": "K",
                },
                "futureTempFeb": {
                    "template_path": "monthly/tas_{scenario}_feb.jshd",
                    "units": "K",
                },
            },
            variants={
                "scenario": {
                    "values": ["ssp245", "ssp370", "ssp585"],
                    "default": "ssp245",
                },
            },
        )

    def test_variant_sweep_single_axis(self):
        from joshpy.jobs import CompoundSweepParameter, FileSweepParameter

        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            param = grid.variant_sweep("scenario")

            self.assertIsInstance(param, CompoundSweepParameter)
            self.assertEqual(param.name, "scenario")
            self.assertEqual(param.labels, ["ssp245", "ssp370", "ssp585"])
            self.assertEqual(len(param.parameters), 2)  # two template_path files

            # Check that each is a FileSweepParameter with 3 paths
            for fp in param.parameters:
                self.assertIsInstance(fp, FileSweepParameter)
                self.assertEqual(len(fp.paths), 3)

            # Verify specific paths
            jan_param = next(p for p in param.parameters if p.name == "futureTempJan")
            self.assertEqual(
                jan_param.paths[1],
                (Path(tmpdir) / "monthly/tas_ssp370_jan.jshd").resolve(),
            )

    def test_variant_sweep_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            param = grid.variant_sweep("scenario", values=["ssp245", "ssp585"])

            self.assertEqual(param.labels, ["ssp245", "ssp585"])
            for fp in param.parameters:
                self.assertEqual(len(fp.paths), 2)

    def test_variant_sweep_skips_static_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            param = grid.variant_sweep("scenario")

            names = [p.name for p in param.parameters]
            self.assertNotIn("cover", names)
            self.assertIn("futureTempJan", names)
            self.assertIn("futureTempFeb", names)

    def test_variant_sweep_unknown_axis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                grid.variant_sweep("gcm")
            self.assertIn("Unknown variant axis", str(ctx.exception))

    def test_variant_sweep_multi_axis(self):
        from joshpy.jobs import CompoundSweepParameter

        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="multi",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(33.9, -116.05),
                high=(33.95, -116.0),
                steps=86,
                files={
                    "cover": {"path": "cover.jshd", "units": "percent"},
                    "futureTempJan": {
                        "template_path": "monthly/tas_{scenario}_{gcm}_jan.jshd",
                        "units": "K",
                    },
                },
                variants={
                    "scenario": {
                        "values": ["ssp245", "ssp370"],
                        "default": "ssp245",
                    },
                    "gcm": {
                        "values": ["cesm2", "miroc6"],
                        "default": "cesm2",
                    },
                },
            )
            param = grid.variant_sweep(axes=["scenario", "gcm"])

            self.assertIsInstance(param, CompoundSweepParameter)
            self.assertEqual(param.name, "scenario_gcm")
            # 2 scenarios × 2 gcms = 4 combos
            self.assertEqual(len(param.labels), 4)
            self.assertIn("ssp245_cesm2", param.labels)
            self.assertIn("ssp370_miroc6", param.labels)

            # Only 1 template_path file (futureTempJan), cover is static
            self.assertEqual(len(param.parameters), 1)
            self.assertEqual(len(param.parameters[0].paths), 4)

    def test_variant_sweep_axis_and_axes_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                grid.variant_sweep("scenario", axes=["scenario"])
            self.assertIn("Cannot specify both", str(ctx.exception))

    def test_variant_sweep_neither_axis_nor_axes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = self._make_variant_grid(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                grid.variant_sweep()
            self.assertIn("Must specify either", str(ctx.exception))

    def test_variant_sweep_values_with_multi_axis_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="multi",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(33.9, -116.05),
                high=(33.95, -116.0),
                steps=86,
                files={
                    "temp": {
                        "template_path": "tas_{scenario}_{gcm}_jan.jshd",
                        "units": "K",
                    },
                },
                variants={
                    "scenario": {
                        "values": ["ssp245", "ssp370"],
                        "default": "ssp245",
                    },
                    "gcm": {
                        "values": ["cesm2", "miroc6"],
                        "default": "cesm2",
                    },
                },
            )
            with self.assertRaises(ValueError) as ctx:
                grid.variant_sweep(axes=["scenario", "gcm"], values=["ssp245"])
            self.assertIn("only supported for single-axis", str(ctx.exception))


@unittest.skipIf(not HAS_YAML, "pyyaml not installed")
class TestGridSpecVariantYaml(unittest.TestCase):
    """Tests for variant YAML round-trip."""

    def test_save_load_round_trip_with_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="variant_grid",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(33.9, -116.05),
                high=(33.95, -116.0),
                steps=100,
                files={
                    "cover": {"path": "cover.jshd", "units": "percent"},
                    "futureTempJan": {
                        "template_path": "monthly/tas_{scenario}_jan.jshd",
                        "units": "K",
                    },
                },
                variants={
                    "scenario": {
                        "values": ["ssp245", "ssp370", "ssp585"],
                        "default": "ssp245",
                    },
                },
            )
            grid.save()
            loaded = GridSpec.from_yaml(Path(tmpdir) / "grid.yaml")

            # Variants round-trip
            self.assertIn("scenario", loaded.variants)
            self.assertEqual(
                loaded.variants["scenario"]["values"],
                ["ssp245", "ssp370", "ssp585"],
            )
            self.assertEqual(loaded.variants["scenario"]["default"], "ssp245")

            # Files round-trip: static has path, template has template_path
            self.assertIn("path", loaded.files["cover"])
            self.assertNotIn("template_path", loaded.files["cover"])
            self.assertIn("template_path", loaded.files["futureTempJan"])
            self.assertNotIn("path", loaded.files["futureTempJan"])
            self.assertEqual(
                loaded.files["futureTempJan"]["template_path"],
                "monthly/tas_{scenario}_jan.jshd",
            )

    def test_no_variants_backward_compat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="plain",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(0, 0),
                high=(1, 1),
                steps=10,
                files={"cover": {"path": "cover.jshd", "units": "percent"}},
            )
            grid.save()
            loaded = GridSpec.from_yaml(Path(tmpdir) / "grid.yaml")
            self.assertEqual(loaded.variants, {})
            self.assertEqual(loaded.files["cover"]["path"], "cover.jshd")

    def test_yaml_structure_with_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="test",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(0, 0),
                high=(1, 1),
                steps=10,
                files={
                    "cover": {"path": "cover.jshd", "units": "percent"},
                    "temp": {
                        "template_path": "tas_{scenario}.jshd",
                        "units": "K",
                    },
                },
                variants={
                    "scenario": {
                        "values": ["ssp245", "ssp370"],
                        "default": "ssp245",
                    },
                },
            )
            saved = grid.save()
            data = yaml.safe_load(saved.read_text())

            # variants section exists
            self.assertIn("variants", data)
            self.assertIn("scenario", data["variants"])
            self.assertEqual(
                data["variants"]["scenario"]["values"], ["ssp245", "ssp370"]
            )

            # files: cover has path, temp has template_path
            self.assertIn("path", data["files"]["cover"])
            self.assertNotIn("template_path", data["files"]["cover"])
            self.assertIn("template_path", data["files"]["temp"])
            self.assertNotIn("path", data["files"]["temp"])


class TestGridSpecPreprocessVariant(unittest.TestCase):
    """Tests for preprocessing with variant kwarg."""

    def _mock_cli(self, success=True):
        cli = MagicMock()
        mock_result = MagicMock()
        mock_result.success = success
        mock_result.stdout = ""
        mock_result.stderr = ""
        cli.preprocess.return_value = mock_result
        return cli, mock_result

    def test_preprocess_with_variant_resolves_template_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="test",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(33.9, -116.05),
                high=(33.95, -116.0),
                steps=10,
                files={
                    "futureTempJan": {
                        "template_path": "monthly/tas_{scenario}_jan.jshd",
                        "units": "K",
                    },
                },
                variants={
                    "scenario": {
                        "values": ["ssp245", "ssp370"],
                        "default": "ssp245",
                    },
                },
            )
            cli, _ = self._mock_cli(success=True)

            grid.preprocess_netcdf(
                cli,
                josh_name="futureTempJan",
                data_file=Path(tmpdir) / "tas.nc",
                variable="tas",
                units="K",
                variant={"scenario": "ssp370"},
            )

            # Check the output path passed to CLI
            config_arg = cli.preprocess.call_args[0][0]
            expected = Path(tmpdir) / "monthly" / "tas_ssp370_jan.jshd"
            self.assertEqual(config_arg.output, expected)

    def test_preprocess_with_variant_skips_registration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="test",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(33.9, -116.05),
                high=(33.95, -116.0),
                steps=10,
                files={
                    "futureTempJan": {
                        "template_path": "monthly/tas_{scenario}_jan.jshd",
                        "units": "K",
                    },
                },
                variants={
                    "scenario": {
                        "values": ["ssp245", "ssp370"],
                        "default": "ssp245",
                    },
                },
            )
            cli, _ = self._mock_cli(success=True)

            grid.preprocess_netcdf(
                cli,
                josh_name="futureTempJan",
                data_file=Path(tmpdir) / "tas.nc",
                variable="tas",
                units="K",
                variant={"scenario": "ssp370"},
            )

            # File entry should still have template_path, not be overwritten
            self.assertIn("template_path", grid.files["futureTempJan"])
            self.assertNotIn("path", grid.files["futureTempJan"])

    def test_preprocess_without_variant_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = GridSpec(
                name="test",
                output_dir=Path(tmpdir),
                size_m=30,
                low=(33.9, -116.05),
                high=(33.95, -116.0),
                steps=10,
            )
            cli, _ = self._mock_cli(success=True)

            grid.preprocess_netcdf(
                cli,
                josh_name="cover",
                data_file=Path(tmpdir) / "cover.nc",
                variable="cover",
                units="percent",
            )

            # Should register as a concrete path entry
            self.assertIn("cover", grid.files)
            self.assertIn("path", grid.files["cover"])
            self.assertEqual(grid.files["cover"]["path"], "cover.jshd")


if __name__ == "__main__":
    unittest.main()
