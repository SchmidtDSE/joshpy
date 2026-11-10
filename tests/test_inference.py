"""Tests for joshpy.inference (known-truth studies).

The table methods of LoadedStudy run on a synthetic reference table. The
end-to-end tests use a fake CLI and a stand-in model whose observed data are a
function of the parameters, as in test_sbi.py, so they exercise the real
SweepManager path without the JAR.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from joshpy.inference import (
    KnownTruthStudy,
    LoadedStudy,
    PropertyFilter,
    effective_sample_size,
    prior_interval,
    weighted_quantile,
)
from joshpy.jobs import JobConfig, SweepConfig
from tests.test_sbi import _export_paths, _has_sbi

PRIOR = {
    "growth": {"uniform": [0.2, 2.0]},
    "precip": {"uniform": [0.0, 100.0]},
}
TRUTH = {"growth": 1.0, "precip": 40.0}


def stand_in(params, replicate=0):
    """Observed data of the stand-in model: a function of the parameters, offset by replicate."""
    growth, precip = float(params["growth"]), float(params["precip"])
    return {
        "height": growth * (1 + precip / 100) + 0.001 * replicate,
        "spread": precip / 100,
        "constant": 1.0 + 0.01 * np.sin(1000 * growth),
    }


def registry_observe(registry, run_hash, replicate):
    """observe(): read the run's parameters back from the registry."""
    return stand_in(registry.get_config_by_hash(run_hash).parameters, replicate)


def statistics(data):
    return dict(data)


def measure(registry, run_hash, replicate):
    """A property filter's measurements: the stand-in's height and spread."""
    data = registry_observe(registry, run_hash, replicate)
    return {"height": data["height"], "spread": data["spread"]}


class _Model:
    """A JobConfig for the stand-in model, backed by a temporary .josh file."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.josh_path = self.tmp / "model.josh"
        self.josh_path.write_text("start simulation Main\nend simulation\n")

    def config(self, **kwargs):
        return JobConfig(
            template_string="growth = {{ growth }} m\nprecip = {{ precip }} %",
            source_path=self.josh_path,
            **kwargs,
        )

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def make_study(fake, **kwargs):
    defaults = dict(
        name="toy",
        model=fake.config(),
        prior=PRIOR,
        truth=TRUTH,
        observe=registry_observe,
        statistics=statistics,
        compare=["height", "spread"],
    )
    defaults.update(kwargs)
    return KnownTruthStudy(**defaults)


def synthetic_reference(n=200, seed=0):
    """A reference table drawn from the prior, with the stand-in's statistics."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        params = {name: rng.uniform(*spec["uniform"]) for name, spec in PRIOR.items()}
        rows.append({"run_hash": f"run{i}", **params, **stand_in(params)})
    return pd.DataFrame(rows)


class TestHelpers(unittest.TestCase):
    def test_uniform_interval(self):
        np.testing.assert_allclose(prior_interval({"uniform": [0.0, 10.0]}, 0.9), (0.5, 9.5))
        self.assertEqual(prior_interval({"uniform": [0.0, 10.0]}, 1.0), (0.0, 10.0))

    def test_normal_interval(self):
        low, high = prior_interval({"normal": [5.0, 2.0]}, 0.95)
        self.assertAlmostEqual(low, 5.0 - 1.95996 * 2, places=4)
        self.assertAlmostEqual(high, 5.0 + 1.95996 * 2, places=4)

    def test_unknown_prior_kind_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported prior kind"):
            prior_interval({"beta": [1.0, 1.0]}, 0.9)

    def test_weighted_quantile(self):
        values = np.array([3.0, 1.0, 2.0, 4.0])
        self.assertEqual(weighted_quantile(values, np.ones(4), 0.5), 2.0)
        self.assertEqual(weighted_quantile(values, np.array([0, 0, 0, 1.0]), 0.5), 4.0)
        self.assertEqual(weighted_quantile(values, np.ones(4), 1.0), 4.0)

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size(np.ones(30)), 30.0)
        self.assertAlmostEqual(effective_sample_size([1.0, 0, 0, 0]), 1.0)


class TestStudyValidation(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.addCleanup(self.model.cleanup)

    def test_model_with_a_sweep_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must not have a sweep"):
            make_study(self.model, model=self.model.config(sweep=SweepConfig()))

    def test_truth_must_name_the_prior_parameters(self):
        with self.assertRaisesRegex(ValueError, "same parameters"):
            make_study(self.model, truth={"growth": 1.0})

    def test_prior_kinds_are_checked(self):
        with self.assertRaisesRegex(ValueError, "Unsupported prior kind"):
            make_study(self.model, prior={**PRIOR, "precip": {"beta": [1.0, 1.0]}})

    def test_compare_is_required(self):
        with self.assertRaisesRegex(ValueError, "at least one statistic"):
            make_study(self.model, compare=[])


class TestPropertyFilter(unittest.TestCase):
    """Criteria, shares and passing, from measurements given directly (nothing runs)."""

    VALUES = {  # run_hash -> height of replicates 0, 1, 2, 3
        "inside": [1.1, 1.2, 1.3, 1.4],
        "one_out": [1.1, 1.2, 1.3, 9.0],
        "outside": [0.1, 0.2, 0.3, np.nan],
    }

    @classmethod
    def measure(cls, registry, run_hash, replicate):
        return {"height": cls.VALUES[run_hash][replicate], "spread": 0.5}

    def make(self, **kwargs):
        defaults = dict(measure=self.measure, criteria={"tall": ("height", 1.0, 1.5)}, replicates=4)
        defaults.update(kwargs)
        return PropertyFilter(**defaults)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one criterion"):
            self.make(criteria={})
        with self.assertRaisesRegex(ValueError, "needs a low or a high bound"):
            self.make(criteria={"tall": ("height", None, None)})
        with self.assertRaisesRegex(ValueError, "above high"):
            self.make(criteria={"tall": ("height", 2.0, 1.0)})
        with self.assertRaisesRegex(ValueError, "replicates"):
            self.make(replicates=0)
        with self.assertRaisesRegex(ValueError, "min_share"):
            self.make(min_share=0.0)

    def test_open_bounds_and_nan(self):
        table = pd.DataFrame({"height": [0.5, 1.0, 2.0, np.nan], "spread": [0.1, 0.9, 0.5, 0.5]})
        criteria = {"tall": ("height", 1.0, None), "narrow": ("spread", None, 0.5)}
        met = self.make(criteria=criteria).meets(table)
        self.assertEqual(list(met.columns), ["tall", "narrow"])
        self.assertEqual(list(met["tall"]), [False, True, True, False])
        self.assertEqual(list(met["narrow"]), [True, False, True, True])

    def test_measurements_cover_every_replicate(self):
        table = self.make().measurements(None, ["inside", "outside"])
        self.assertEqual(list(table.columns), ["run_hash", "replicate", "height", "spread"])
        self.assertEqual(list(table["replicate"]), [0, 1, 2, 3] * 2)

    def test_evaluate_shares_and_passes(self):
        runs = ["inside", "one_out", "outside"]
        table = self.make(min_share=0.75).evaluate(None, runs)
        self.assertEqual(list(table.index), runs)
        self.assertEqual(list(table.columns), ["tall", "passes"])
        self.assertEqual(list(table["tall"]), [1.0, 0.75, 0.0])
        self.assertEqual(list(table["passes"]), [True, True, False])
        strict = self.make(min_share=1.0).evaluate(None, runs)
        self.assertEqual(list(strict["passes"]), [True, False, False])

    def test_a_set_passes_only_if_it_passes_every_criterion(self):
        criteria = {"tall": ("height", 1.0, 1.5), "narrow": ("spread", None, 0.4)}
        table = self.make(criteria=criteria, min_share=0.5).evaluate(None, ["inside"])
        self.assertEqual(table.loc["inside", "tall"], 1.0)
        self.assertEqual(table.loc["inside", "narrow"], 0.0)
        self.assertFalse(table.loc["inside", "passes"])


class TestLoadedStudy(unittest.TestCase):
    """LoadedStudy's tables, from a synthetic reference table (nothing runs)."""

    def setUp(self):
        self.model = _Model()
        self.addCleanup(self.model.cleanup)
        self.study = make_study(self.model)
        self.reference = synthetic_reference()
        self.loaded = LoadedStudy(
            self.study, MagicMock(run_hash="truth"), stand_in(TRUTH), self.reference
        )

    def test_observed_is_the_compared_statistics_of_the_truth(self):
        self.assertEqual(set(self.loaded.candidates), {"height", "spread", "constant"})
        self.assertEqual(
            self.loaded.observed, {k: stand_in(TRUTH)[k] for k in ["height", "spread"]}
        )

    def test_zscore_uses_the_prior_simulations(self):
        z = self.loaded.zscore(self.reference)
        self.assertEqual(list(z.columns), ["height", "spread"])
        for name in z:
            self.assertAlmostEqual(z[name].mean(), 0.0)
            self.assertAlmostEqual(z[name].std(), 1.0)
        one = self.loaded.zscore(self.reference.iloc[3])
        self.assertAlmostEqual(one["height"], z["height"].iloc[3])

    def test_distance_is_unchanged_by_centering(self):
        distance = self.loaded.distance(self.reference)
        scale = self.reference[["height", "spread"]].std()
        offsets = (self.reference[["height", "spread"]] - pd.Series(self.loaded.observed)) / scale
        np.testing.assert_allclose(distance, np.sqrt((offsets**2).sum(axis=1)))
        self.assertEqual(len(self.loaded.distance(self.reference, ["height"])), len(self.reference))

    def test_correlations(self):
        table = self.loaded.correlations()
        self.assertEqual(list(table.index), ["height", "spread", "constant"])
        self.assertEqual(list(table.columns), ["growth", "precip"])
        self.assertAlmostEqual(table.loc["spread", "precip"], 1.0)

    def test_abc_target_scores_each_run_as_s_obs_is_scored(self):
        summaries, observed = self.loaded.abc_target(["spread"])
        self.assertEqual(observed, self.loaded.zscore(self.loaded.candidates, ["spread"]))
        registry = MagicMock()
        registry.get_config_by_hash.return_value = MagicMock(parameters=TRUTH)
        self.assertEqual(summaries(registry, "truth", None), observed)

    def test_leave_one_out(self):
        k = 20
        both = self.loaded.leave_one_out(k=k)
        self.assertEqual(list(both.index), ["growth", "precip"])
        expected = (k - 1) / (k + 1)
        for name in both.index:
            self.assertGreater(both.at[name, "coverage"], expected - 0.1)
            self.assertLess(both.at[name, "width"], 0.5)
        # The spread says nothing about growth: its accepted range is the prior's.
        spread_only = self.loaded.leave_one_out(["spread"], k=k)
        self.assertGreater(spread_only.at["growth", "width"], 0.8)
        self.assertLess(spread_only.at["precip", "width"], 0.2)

    def test_posterior(self):
        particles = pd.DataFrame(
            {"growth": np.linspace(0.9, 1.1, 21), "precip": np.linspace(50, 70, 21)}
        )
        particles["weight"] = 1 / len(particles)
        result = MagicMock()
        result.to_dataframe.return_value = particles
        table = self.loaded.posterior(result)
        self.assertEqual(
            list(table.columns), ["truth", "mean", "5%", "95%", "width", "truth inside"]
        )
        self.assertAlmostEqual(table.at["growth", "mean"], 1.0)
        self.assertTrue(table.at["growth", "truth inside"])
        self.assertFalse(table.at["precip", "truth inside"])
        low, high = table.at["growth", "5%"], table.at["growth", "95%"]
        self.assertAlmostEqual(table.at["growth", "width"], (high - low) / (0.9 * 1.8))


@unittest.skipUnless(_has_sbi(), "sbi not installed")
class TestKnownTruthStudyRuns(unittest.TestCase):
    """load, replicates and search through SweepManager, with a fake CLI."""

    def setUp(self):
        from joshpy.cli import CLIResult
        from joshpy.registry import RunRegistry

        self.model = _Model()
        self.addCleanup(self.model.cleanup)
        self.cli = MagicMock()
        self.cli.run.return_value = CLIResult(0, "", "", [])
        self.cli.inspect_exports.return_value = _export_paths(
            str(self.model.tmp / "out_{run_hash}_{replicate}.csv")
        )
        patcher = patch("joshpy.sweep.load_job_results", side_effect=self._fake_load)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = RunRegistry(":memory:")
        self.addCleanup(self.registry.close)
        self.study = make_study(self.model)

    @staticmethod
    def _fake_load(cli, job, registry, export_paths, *, run_id, **kwargs):
        loaded = registry.loaded_replicates(job.run_hash)
        rows = 0
        for rep in range(job.replicates):
            if rep not in loaded:
                registry.conn.execute(
                    "INSERT INTO cell_data (run_id, run_hash, step, replicate) VALUES (?, ?, 0, ?)",
                    [run_id, job.run_hash, rep],
                )
                rows += 1
        return rows

    def experiments(self):
        rows = self.registry.conn.execute(
            "SELECT DISTINCT experiment_name FROM sweep_sessions ORDER BY 1"
        ).fetchall()
        return [row[0] for row in rows]

    def test_load_simulates_once_and_reuses_the_registry(self):
        loaded = self.study.load(self.registry, self.cli, n_prior=20)
        runs = self.cli.run.call_count
        self.assertEqual(runs, 21)  # the truth and 20 prior simulations
        self.assertEqual(len(loaded.reference), 20)
        self.assertEqual(
            list(loaded.reference.columns),
            ["run_hash", "growth", "precip", "height", "spread", "constant"],
        )
        self.assertEqual(loaded.truth_job.seed, 1)
        self.assertEqual(loaded.observed["spread"], 0.4)
        self.assertEqual(self.experiments(), ["toy_prior", "toy_truth"])

        again = self.study.load(self.registry, self.cli, n_prior=20)
        self.assertEqual(self.cli.run.call_count, runs)
        pd.testing.assert_frame_equal(again.reference, loaded.reference)

    def test_replicates(self):
        table = self.study.replicates(self.registry, self.cli, n=3, seed=2)
        self.assertEqual(list(table["replicate"]), [0, 1, 2])
        self.assertAlmostEqual(table["height"].iloc[2] - table["height"].iloc[0], 0.002)
        runs = self.cli.run.call_count
        self.study.replicates(self.registry, self.cli, n=3, seed=2)
        self.assertEqual(self.cli.run.call_count, runs)
        self.assertIn("toy_replicates", self.experiments())

    def test_search_starts_from_the_prior_simulations(self):
        loaded = self.study.load(self.registry, self.cli, n_prior=20)
        result, session_id = loaded.search(
            self.registry,
            self.cli,
            "smcabc",
            {
                "num_initial_pop": 20,
                "num_particles": 10,
                "num_simulations": 60,
                "epsilon_decay": 0.5,
            },
            "toy_smcabc",
        )
        evaluated = [job.run_hash for job, _ in result.job_results]
        self.assertEqual(evaluated[:20], list(loaded.reference["run_hash"]))
        sbi = self.registry.get_session(session_id).metadata["sbi"]
        self.assertEqual(sbi["observed"], loaded.zscore(loaded.observed))
        table = loaded.posterior(result)
        self.assertEqual(list(table.index), ["growth", "precip"])

    def test_search_records_a_named_batch_policy(self):
        """A target's search used "measured"; naming it replays that search locally."""
        loaded = self.study.load(self.registry, self.cli, n_prior=20)
        settings = {
            "num_initial_pop": 20,
            "num_particles": 10,
            "num_simulations": 60,
            "epsilon_decay": 0.5,
        }
        _, exact = loaded.search(self.registry, self.cli, "smcabc", settings, "toy_exact")
        _, measured = loaded.search(
            self.registry, self.cli, "smcabc", settings, "toy_measured", batch_policy="measured"
        )
        policies = {
            session_id: self.registry.get_session(session_id).metadata["sbi"].get("batch_policy")
            for session_id in (exact, measured)
        }
        self.assertIsNone(policies[exact])  # local default: "exact", not recorded
        self.assertEqual(policies[measured]["policy"], "measured")

    def test_filter_prior_tops_up_the_prior_simulations(self):
        loaded = self.study.load(self.registry, self.cli, n_prior=20)
        hashes = list(loaded.reference["run_hash"])
        first = dict(
            self.registry.conn.execute(
                "SELECT run_hash, run_id FROM cell_data WHERE replicate = 0"
            ).fetchall()
        )
        runs = self.cli.run.call_count
        height = PropertyFilter(
            measure=measure, criteria={"tall": ("height", 1.0, 1.5)}, replicates=3
        )
        table = self.study.filter_prior(self.registry, self.cli, height, n=20)

        self.assertEqual(list(table["run_hash"]), hashes)  # the same draws
        self.assertEqual(self.cli.run.call_count, runs + 20)  # each rerun with 3 replicates
        for run_hash in hashes:
            self.assertEqual(self.registry.loaded_replicates(run_hash), {0, 1, 2})
        kept = dict(
            self.registry.conn.execute(
                "SELECT run_hash, run_id FROM cell_data WHERE replicate = 0"
            ).fetchall()
        )
        self.assertEqual(kept, first)  # replicate 0 is the prior simulation's
        self.assertEqual(list(table.columns), ["run_hash", "growth", "precip", "tall", "passes"])
        for row in table.itertuples():
            expected = stand_in({"growth": row.growth, "precip": row.precip})["height"]
            heights = expected + 0.001 * np.arange(3)
            self.assertAlmostEqual(row.tall, np.mean((heights >= 1.0) & (heights <= 1.5)))
        self.assertIn("toy_filter", self.experiments())

        again = self.study.filter_prior(self.registry, self.cli, height, n=20)
        self.assertEqual(self.cli.run.call_count, runs + 20)
        pd.testing.assert_frame_equal(again, table)

    def sessions(self, experiment):
        return self.registry.conn.execute(
            "SELECT COUNT(*) FROM sweep_sessions WHERE experiment_name = ?", [experiment]
        ).fetchone()[0]

    def test_run_at_each_runs_the_sets_as_one_batch(self):
        sets = [
            {"growth": 1.0, "precip": 40.0},
            {"precip": 20.0, "growth": 1.5},  # key order doesn't matter
            {"growth": 1.0, "precip": 40.0},
        ]
        jobs = self.study.run_at_each(self.registry, self.cli, sets, "toy_batch", replicates=2)

        self.assertEqual(
            [job.parameters for job in jobs],
            [{"growth": 1.0, "precip": 40.0}, {"growth": 1.5, "precip": 20.0}] + [jobs[0].parameters],
        )
        self.assertIs(jobs[0], jobs[2])  # identical sets share one run
        self.assertEqual(self.cli.run.call_count, 2)
        self.assertEqual(self.sessions("toy_batch"), 1)
        for job in jobs:
            self.assertEqual(self.registry.loaded_replicates(job.run_hash), {0, 1})

        again = self.study.run_at_each(self.registry, self.cli, sets, "toy_batch", replicates=2)
        self.assertEqual([job.run_hash for job in again], [job.run_hash for job in jobs])
        self.assertEqual(self.cli.run.call_count, 2)

    def test_run_at_each_reproduces_a_searchs_runs(self):
        loaded = self.study.load(self.registry, self.cli, n_prior=20)
        smc = {"num_initial_pop": 20, "num_particles": 10, "num_simulations": 60, "epsilon_decay": 0.5}
        result, _ = loaded.search(self.registry, self.cli, "smcabc", smc, "toy_smcabc")
        particles = result.to_dataframe()
        runs = self.cli.run.call_count

        jobs = self.study.run_at_each(
            self.registry, self.cli, particles[["growth", "precip"]].to_dict("records"), "toy_again"
        )
        self.assertEqual([job.run_hash for job in jobs], list(particles["run_hash"]))
        self.assertEqual(self.cli.run.call_count, runs)  # every particle is a completed run

    def test_run_at_each_needs_every_parameter(self):
        with self.assertRaisesRegex(ValueError, "must name exactly"):
            self.study.run_at_each(self.registry, self.cli, [{"growth": 1.0}], "toy_batch")

    def test_a_target_dispatches_each_batch_together_and_replays_locally(self):
        import json

        from joshpy.cli import CLIResult

        events = []

        def dispatch(config, **kwargs):
            events.append("dispatch")
            self.assertTrue(config.no_wait)
            return CLIResult(0, json.dumps({"jobId": f"job-{len(events)}"}), "", [])

        def poll(config, **kwargs):
            events.append("poll")
            return CLIResult(0, "", "", [])

        self.cli.stage_to_minio.return_value = CLIResult(0, "", "", [])
        self.cli.batch_remote.side_effect = dispatch
        self.cli.poll_batch.side_effect = poll
        on_target = make_study(self.model, target="cluster")
        sets = [{"growth": g, "precip": 40.0} for g in (0.5, 1.0, 1.5)]

        jobs = on_target.run_at_each(self.registry, self.cli, sets, "toy_remote", replicates=2)

        self.cli.run.assert_not_called()
        self.assertEqual(events, ["dispatch"] * 3 + ["poll"] * 3)  # all dispatched, then polled
        for job in jobs:
            self.assertEqual(self.registry.loaded_replicates(job.run_hash), {0, 1})

        local = self.study.run_at_each(self.registry, self.cli, sets, "toy_local", replicates=2)
        self.assertEqual([job.run_hash for job in local], [job.run_hash for job in jobs])
        self.cli.run.assert_not_called()  # the target's runs replay from the registry


if __name__ == "__main__":
    unittest.main()
