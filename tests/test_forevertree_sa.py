"""The sensitivity study's model configuration (examples/sbi_forevertree/forevertree_sa_*).

Jinja renders a variable it isn't given as an empty string, so these check that the template and
the case module's BASELINE name the same variables, and that the baseline is the fire model's
configuration.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

from jinja2 import Environment, meta

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sbi_forevertree"
sys.path.insert(0, str(EXAMPLE))

import forevertree_sa_case as sa  # noqa: E402


def _render(template: Path, values: dict) -> dict[str, float]:
    """The ``name = number unit`` lines of a rendered .jshc, as numbers."""
    text = Environment().from_string(template.read_text(encoding="utf-8")).render(**values)
    lines = re.findall(r"^(\w+) = (-?[\d.]+)", text, flags=re.MULTILINE)
    return {name: float(value) for name, value in lines}


class TestSensitivityTemplate(unittest.TestCase):
    def test_template_and_baseline_name_the_same_variables(self):
        source = (EXAMPLE / "forevertree_sa.jshc.j2").read_text(encoding="utf-8")
        variables = meta.find_undeclared_variables(Environment().parse(source))
        self.assertEqual(variables, set(sa.BASELINE))

    def test_every_variable_renders_to_a_number(self):
        rendered = _render(EXAMPLE / "forevertree_sa.jshc.j2", sa.BASELINE)
        self.assertEqual(set(rendered), set(sa.BASELINE))

    def test_baseline_is_the_fire_models_truth(self):
        """At BASELINE, every input the fire model has takes the fire model's value."""
        sa_values = _render(EXAMPLE / "forevertree_sa.jshc.j2", sa.BASELINE)
        fire_values = _render(
            EXAMPLE / "forevertree_fire.jshc.j2",
            {"recruitment": 50.0, "juvenileMortality": 5.0, "projectionYears": 40},
        )
        shared = set(sa_values) & set(fire_values)
        self.assertGreaterEqual(len(shared), 18)
        for name in sorted(shared):
            self.assertEqual(sa_values[name], fire_values[name], name)

    def test_neutral_knobs(self):
        """The knobs the fire model lacks are neutral at the baseline."""
        knobs = {
            "rainScale": 100, "temperatureOffset": 0, "fuelTreatmentInterval": 0,
            "fuelReduction": 0, "plantingDensity": 0, "juvenileProtection": 0,
            "initialFuel": 50,  # the fire model's hard-coded fuel.init
        }
        for name, value in knobs.items():
            self.assertEqual(sa.BASELINE[name], value, name)


class TestSensitivityDesign(unittest.TestCase):
    def test_labels_are_unique_and_overrides_are_template_variables(self):
        design = sa.design()
        self.assertTrue(design["label"].is_unique)
        for overrides in design["overrides"]:
            self.assertLessEqual(set(overrides), set(sa.BASELINE))

    def test_each_factor_moves_one_sd_each_way(self):
        for factor in sa.FACTORS:
            low, high = factor.at(-1), factor.at(1)
            self.assertAlmostEqual(high - low, 2 * factor.sd)
            self.assertIs(type(low), type(factor.baseline))

    def test_spreads_are_shares_of_the_baseline(self):
        """Each SD is its spread's share of the baseline: a whole step for counts and ages."""
        for factor in sa.FACTORS:
            if factor.spread is None:
                self.assertEqual(factor.baseline, 0, factor.name)  # only a zero baseline
                self.assertIsNotNone(factor.step, factor.name)
                continue
            share = sa.SPREADS[factor.spread] * factor.baseline
            if isinstance(factor.baseline, int):
                self.assertEqual(factor.sd, max(1, round(share)), factor.name)
            else:
                self.assertAlmostEqual(factor.sd, share, msg=factor.name)
            self.assertGreater(factor.at(-1), 0, factor.name)  # no input moves to zero or below
        used = {factor.spread for factor in sa.FACTORS if factor.spread}
        self.assertEqual(used, set(sa.SPREADS))

    def test_pulse_burns_in_the_first_projected_year(self):
        run = sa.settings(sa.PULSE)
        self.assertEqual(run["firstFire"], run["spinupYears"] + run["presentYears"])
        self.assertGreater(run["fireInterval"], run["projectionYears"])


if __name__ == "__main__":
    unittest.main()
