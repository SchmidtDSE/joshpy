"""Tests for SbiStrategy (ABC through sbi, with joshpy as the simulator).

The sbi-dependent tests use a fake CLI and a deterministic stand-in model whose
summary statistics are a function of the parameters, so they exercise the full
joshpy path (expand, register, run_sweep, load, score) without the JAR.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from joshpy.strategies import SbiStrategy, strategy_from_dict


def _has_sbi() -> bool:
    try:
        import sbi  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _export_paths(path):
    """Real ExportPaths for a local patch export at *path* (a file:// URI path)."""
    from joshpy.cli import ExportFileInfo, ExportPaths

    return ExportPaths(
        simulation="Main",
        export_files={
            "patch": ExportFileInfo(
                raw=f"file://{path}", protocol="file", host="", path=path, file_type="csv"
            )
        },
        debug_files={},
    )


PRIOR = {
    "growth": {"uniform": [0.2, 2.0]},
    "precip": {"uniform": [0.0, 100.0]},
}
TRUTH = {"growth": 1.0, "precip": 40.0}


def model_summaries(registry, run_hash, job):
    """Noise-free stand-in for a simulation's summary statistics."""
    growth = float(job.parameters["growth"])
    precip = float(job.parameters["precip"])
    return {"height": growth * (1 + precip / 100), "spread": precip / 100}


OBSERVED = model_summaries(None, None, MagicMock(parameters=TRUTH))


class TestSbiStrategyConfig(unittest.TestCase):
    """Validation and serialization (no sbi needed)."""

    def test_is_adaptive_and_cannot_expand(self):
        strategy = SbiStrategy(prior=PRIOR, observed=OBSERVED)
        self.assertTrue(strategy.is_adaptive)
        with self.assertRaises(RuntimeError):
            strategy.expand([], [])

    def test_param_names_follow_prior_order(self):
        self.assertEqual(SbiStrategy(prior=PRIOR).param_names, ["growth", "precip"])

    def test_rejects_bad_specs(self):
        bad = [
            dict(prior=PRIOR, method="npe"),
            dict(prior={}),
            dict(prior={"a": {"beta": [1, 2]}}),
            dict(prior={"a": {"uniform": [2, 1]}}),
            dict(prior={"a": {"normal": [0, -1]}}),
            dict(prior={"a": {"uniform": [0, 1], "normal": [0, 1]}}),
            dict(prior=PRIOR, method_kwargs={"kde": True}),
            dict(prior=PRIOR, method_kwargs={"lra": True}),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SbiStrategy(**kwargs)

    def test_round_trips_through_dict(self):
        strategy = SbiStrategy(
            prior=PRIOR,
            summaries="tests.test_sbi:model_summaries",
            observed=OBSERVED,
            method="mcabc",
            method_kwargs={"num_simulations": 50, "quantile": 0.1},
            seed=3,
        )
        restored = strategy_from_dict(strategy.to_dict())
        self.assertEqual(restored, strategy)

    def test_callable_summaries_serialize_as_path(self):
        data = SbiStrategy(prior=PRIOR, summaries=model_summaries).to_dict()
        self.assertEqual(data["summaries"], "tests.test_sbi:model_summaries")

    def test_closure_summaries_are_named_not_saved_as_a_path(self):
        def model_summaries(registry, run_hash, job):  # shadows the module-level name
            return {}

        data = SbiStrategy(prior=PRIOR, summaries=model_summaries).to_dict()
        self.assertNotIn("summaries", data)
        self.assertIn("<locals>.model_summaries", data["summaries_not_importable"])
        self.assertIsNone(strategy_from_dict(data).summaries)

    def test_observed_from_json_file(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.json"
            path.write_text(json.dumps({"height": 1.4, "spread": 0.4}))
            strategy = SbiStrategy(prior=PRIOR, observed=str(path))
            self.assertEqual(strategy.get_observed(), {"height": 1.4, "spread": 0.4})

    def test_observed_from_function_path(self):
        strategy = SbiStrategy(prior=PRIOR, observed="tests.test_sbi:_observed")
        self.assertEqual(strategy.get_observed(), OBSERVED)

    def test_missing_summaries_raises(self):
        with self.assertRaises(ValueError):
            SbiStrategy(prior=PRIOR).get_summaries()

    def test_job_config_keeps_parameterless_sbi_sweep(self):
        """A sweep listing no parameters (the prior has them) still serializes."""
        from joshpy.jobs import JobConfig, SweepConfig

        config = JobConfig(
            template_string="growth = {{ growth }} m",
            sweep=SweepConfig(strategy=SbiStrategy(prior=PRIOR, observed=OBSERVED)),
        )
        restored = JobConfig.from_dict(config.to_dict())
        self.assertEqual(restored.sweep.strategy, config.sweep.strategy)

    def test_empty_cartesian_sweep_still_omitted(self):
        from joshpy.jobs import JobConfig, SweepConfig

        self.assertNotIn("sweep", JobConfig(template_string="x", sweep=SweepConfig()).to_dict())


def _observed():
    return dict(OBSERVED)


@unittest.skipUnless(_has_sbi(), "sbi not installed")
class TestSbiStrategyRuns(unittest.TestCase):
    """End-to-end through run_adaptive_sweep with a fake CLI."""

    def setUp(self):
        from joshpy.cli import CLIResult

        self.tmp = Path(tempfile.mkdtemp())
        self.josh_path = self.tmp / "model.josh"
        self.josh_path.write_text("start simulation Main\nend simulation\n")
        self.cli = MagicMock()
        self.cli.run.return_value = CLIResult(0, "", "", [])
        self.cli.inspect_exports.return_value = _export_paths(
            str(self.tmp / "out_{run_hash}_{replicate}.csv")
        )
        patcher = patch("joshpy.sweep.load_job_results", side_effect=self._fake_load)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registries = []

    def tearDown(self):
        for registry in self.registries:
            registry.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _fake_load(cli, job, registry, export_paths, *, run_id, **kwargs):
        """Idempotent stand-in for loading: one cell_data row per replicate."""
        loaded = registry.loaded_replicates(job.run_hash)
        rows = 0
        for rep in range(job.replicates):
            if rep not in loaded:
                registry.conn.execute(
                    "INSERT INTO cell_data (run_id, run_hash, step, replicate) "
                    "VALUES (?, ?, 0, ?)",
                    [run_id, job.run_hash, rep],
                )
                rows += 1
        return rows

    def registry(self, name=None):
        from joshpy.registry import RunRegistry

        registry = RunRegistry(str(self.tmp / name) if name else ":memory:")
        self.registries.append(registry)
        return registry

    def run_sbi(self, registry, summaries=model_summaries, job_seed=None, **strategy_kwargs):
        from joshpy.jobs import JobConfig, SweepConfig
        from joshpy.strategies import run_adaptive_sweep

        kwargs = dict(
            prior=PRIOR,
            summaries=summaries,
            observed=OBSERVED,
            method_kwargs={
                "num_particles": 20,
                "num_initial_pop": 40,
                "num_simulations": 200,
                "epsilon_decay": 0.5,
            },
            seed=0,
        )
        kwargs.update(strategy_kwargs)
        config = JobConfig(
            template_string="growth = {{ growth }} m\nprecip = {{ precip }} %",
            source_path=self.josh_path,
            seed=job_seed,
            sweep=SweepConfig(strategy=SbiStrategy(**kwargs)),
        )
        session_id = registry.create_session(config=config)
        result = run_adaptive_sweep(
            self.cli, config, registry=registry, session_id=session_id,
            quiet=True, stop_on_failure=False,
        )
        result.job_set.cleanup()
        return result, session_id

    def test_smcabc_recovers_truth(self):
        registry = self.registry()
        result, session_id = self.run_sbi(registry)

        df = result.to_dataframe()
        mean = {p: (df[p] * df.weight).sum() for p in PRIOR}
        self.assertAlmostEqual(mean["growth"], TRUTH["growth"], delta=0.15)
        self.assertAlmostEqual(mean["precip"], TRUTH["precip"], delta=10)
        self.assertGreater(len(result.populations), 1)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=5)
        self.assertTrue(all(h is not None for h in result.run_hashes))
        # Epsilon shrinks generation over generation.
        epsilons = [p["epsilon"] for p in result.populations]
        self.assertEqual(epsilons, sorted(epsilons, reverse=True))

    def test_particles_carry_the_values_their_runs_were_rendered_with(self):
        registry = self.registry()
        result, _ = self.run_sbi(registry)
        rendered = {job.run_hash: job.parameters for job, _ in result.job_results}
        for particle in result.to_dataframe().to_dict("records"):
            self.assertEqual(
                {name: particle[name] for name in PRIOR}, rendered[particle["run_hash"]]
            )

    def test_every_simulation_is_registered_and_recorded(self):
        registry = self.registry()
        result, session_id = self.run_sbi(registry)

        n_configs = registry.conn.execute(
            "SELECT count(*) FROM session_configs WHERE session_id = ?", [session_id]
        ).fetchone()[0]
        self.assertEqual(n_configs, self.cli.run.call_count)
        self.assertEqual(result.succeeded, len(result.job_results))

        final_hash = result.run_hashes[0]
        attributes = registry.get_attributes(final_hash)
        self.assertTrue(attributes["sbi_accepted"])
        self.assertIn("sbi_weight", attributes)
        self.assertIn("sbi_call", attributes)

        sbi_meta = registry.get_session(session_id).metadata["sbi"]
        self.assertEqual(sbi_meta["param_names"], ["growth", "precip"])
        self.assertEqual(len(sbi_meta["populations"]), len(result.populations))
        self.assertEqual(sbi_meta["final_run_hashes"], result.run_hashes)
        self.assertIn("torch", sbi_meta["versions"])

    def test_mcabc(self):
        registry = self.registry()
        result, _ = self.run_sbi(
            registry, method="mcabc",
            method_kwargs={"num_simulations": 150, "quantile": 0.1},
        )
        self.assertEqual(len(result.samples), 15)
        self.assertEqual(len(result.populations), 1)
        self.assertEqual(self.cli.run.call_count, 150)

    def test_generations_account_for_every_simulation(self):
        registry = self.registry()
        result, session_id = self.run_sbi(registry)

        populations = result.populations
        self.assertEqual(
            sum(p["simulations"] for p in populations), result.succeeded + result.failed
        )
        self.assertEqual(populations[0]["simulations"], 40)  # num_initial_pop
        for pop in populations[:-1]:
            self.assertEqual(pop["carried_over"], 0)
            self.assertGreaterEqual(pop["simulations"], 20)  # num_particles
        recorded = registry.get_session(session_id).metadata["sbi"]["populations"]
        self.assertEqual(
            [p["simulations"] for p in recorded], [p["simulations"] for p in populations]
        )

    def test_budget_spent_mid_generation_is_carried_over(self):
        registry = self.registry()
        result, _ = self.run_sbi(
            registry,
            method_kwargs={
                "num_particles": 20,
                "num_initial_pop": 40,
                "num_simulations": 45,  # 5 left for generation 1
                "epsilon_decay": 0.5,
            },
        )
        initial, final = result.populations
        self.assertEqual((initial["simulations"], initial["carried_over"]), (40, 0))
        self.assertEqual(final["simulations"], 5)
        # At most 5 new particles; sbi fills the rest from generation 0.
        self.assertGreaterEqual(final["carried_over"], 15)
        initial_hashes = set(initial["run_hashes"])
        carried = [h for h in final["run_hashes"] if h in initial_hashes]
        self.assertEqual(len(carried), final["carried_over"])

    def test_warns_when_every_simulation_shares_a_seed(self):
        registry = self.registry()
        with self.assertWarnsRegex(UserWarning, "same Josh random draws"):
            self.run_sbi(
                registry, job_seed=3, method="mcabc",
                method_kwargs={"num_simulations": 10, "quantile": 0.5},
            )

    def test_no_seed_warning_without_a_job_seed(self):
        import warnings

        registry = self.registry()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.run_sbi(
                registry, method="mcabc",
                method_kwargs={"num_simulations": 10, "quantile": 0.5},
            )
        self.assertFalse(any("same Josh random draws" in str(w.message) for w in caught))

    def test_failed_runs_are_never_accepted(self):
        from joshpy.cli import CLIResult

        def run(run_config, **kwargs):
            growth = float(run_config.custom_tags["growth"])
            return CLIResult(1 if growth > 1.5 else 0, "", "OOM" if growth > 1.5 else "", [])

        self.cli.run.side_effect = run
        registry = self.registry()
        result, _ = self.run_sbi(registry)

        self.assertGreater(result.failed, 0)
        self.assertTrue((result.samples[:, 0] <= 1.5).all())

    def test_nan_statistics_reject_runs(self):
        def rejecting(registry, run_hash, job):
            if float(job.parameters["growth"]) > 1.5:
                return {"height": float("nan"), "spread": float("nan")}
            return model_summaries(registry, run_hash, job)

        result, _ = self.run_sbi(self.registry(), summaries=rejecting)
        self.assertTrue((result.samples[:, 0] <= 1.5).all())

    def test_summaries_exception_stops_the_search(self):
        def buggy(registry, run_hash, job):
            if float(job.parameters["growth"]) > 1.5:
                raise KeyError("meanHeight")
            return model_summaries(registry, run_hash, job)

        registry = self.registry()
        with self.assertRaises(RuntimeError) as raised:
            self.run_sbi(registry, summaries=buggy)
        self.assertIn("meanHeight", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, KeyError)
        # The batch (the 40 initial draws) finished first, so a rerun reuses its runs.
        runs = registry.conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
        self.assertEqual(runs, 40)

    def test_summaries_returning_none_fail_fast(self):
        with self.assertRaises(ValueError) as raised:
            self.run_sbi(self.registry(), summaries=lambda r, h, j: None)
        self.assertIn("returned None", str(raised.exception))

    def test_load_errors_are_reported_and_never_accepted(self):
        def flaky_load(cli, job, registry, export_paths, *, run_id, **kwargs):
            if float(job.parameters["growth"]) > 1.5:
                raise FileNotFoundError("no export written")
            return self._fake_load(cli, job, registry, export_paths, run_id=run_id, **kwargs)

        registry = self.registry()
        with patch("joshpy.sweep.load_job_results", side_effect=flaky_load):
            with self.assertWarns(UserWarning) as warned:
                result, session_id = self.run_sbi(registry)

        self.assertGreater(len(result.load_errors), 0)
        self.assertEqual(result.failed, 0)
        self.assertTrue(all("no export written" in m for m in result.load_errors.values()))
        self.assertTrue((result.samples[:, 0] <= 1.5).all())
        metadata = registry.get_session(session_id).metadata["sbi"]
        self.assertEqual(metadata["load_errors"], result.load_errors)
        self.assertTrue(
            any("could not be loaded" in str(w.message) for w in warned.warnings)
        )

    def test_initial_population_without_enough_usable_runs_stops(self):
        def mostly_nan(registry, run_hash, job):
            if float(job.parameters["growth"]) > 0.5:
                return {"height": float("nan"), "spread": float("nan")}
            return model_summaries(registry, run_hash, job)

        with self.assertRaises(RuntimeError) as raised:
            self.run_sbi(self.registry(), summaries=mostly_nan)
        self.assertIn("num_particles=20", str(raised.exception))
        self.assertIn("returned non-finite statistics", str(raised.exception))

    def test_mcabc_quantile_warns_when_it_keeps_unusable_runs(self):
        def rejecting(registry, run_hash, job):
            if float(job.parameters["growth"]) > 1.5:
                return {"height": float("nan"), "spread": float("nan")}
            return model_summaries(registry, run_hash, job)

        with self.assertWarns(UserWarning) as warned:
            self.run_sbi(
                self.registry(), summaries=rejecting, method="mcabc",
                method_kwargs={"num_simulations": 40, "quantile": 1.0},
            )
        self.assertTrue(
            any("accepted samples have non-finite" in str(w.message) for w in warned.warnings)
        )

    def test_objective_override_is_used_as_summaries(self):
        from joshpy.cli import CLIResult  # noqa: F401
        from joshpy.jobs import JobConfig, SweepConfig
        from joshpy.strategies import run_adaptive_sweep

        calls = []

        def summaries(registry, run_hash, job):
            calls.append(run_hash)
            return model_summaries(registry, run_hash, job)

        registry = self.registry()
        config = JobConfig(
            template_string="growth = {{ growth }} m\nprecip = {{ precip }} %",
            source_path=self.josh_path,
            sweep=SweepConfig(strategy=SbiStrategy(
                prior=PRIOR, observed=OBSERVED, method="mcabc",
                method_kwargs={"num_simulations": 20, "quantile": 0.5},
            )),
        )
        session_id = registry.create_session(config=config)
        result = run_adaptive_sweep(
            self.cli, config, registry=registry, session_id=session_id,
            objective=summaries, quiet=True,
        )
        result.job_set.cleanup()
        self.assertEqual(len(calls), 20)

    def test_sweep_manager_dispatches_parameterless_sbi_sweep(self):
        """SweepManager must treat an sbi sweep as adaptive even though its
        SweepConfig lists no parameters (and so is falsy)."""
        from joshpy.jobs import JobConfig, SweepConfig
        from joshpy.strategies import SbiResult
        from joshpy.sweep import SweepManager

        config = JobConfig(
            template_string="growth = {{ growth }} m\nprecip = {{ precip }} %",
            source_path=self.josh_path,
            sweep=SweepConfig(strategy=SbiStrategy(
                prior=PRIOR, summaries=model_summaries, observed=OBSERVED, method="mcabc",
                method_kwargs={"num_simulations": 20, "quantile": 0.25},
            )),
        )
        manager = (
            SweepManager.builder(config)
            .with_registry(self.registry())
            .with_cli(self.cli)
            .build()
        )
        try:
            result = manager.run(quiet=True, stop_on_failure=False)
            self.assertIsInstance(result, SbiResult)
            self.assertEqual(len(result.samples), 5)
            self.assertIs(manager.job_set, result.job_set)
            self.assertEqual(len(manager.job_set), 20)
        finally:
            manager.cleanup()

    def test_missing_summary_key_fails_fast(self):
        registry = self.registry()
        with self.assertRaises(ValueError) as raised:
            self.run_sbi(registry, summaries=lambda r, h, j: {"height": 1.0})
        self.assertIn("spread", str(raised.exception))

    def test_interrupted_run_replays_identically(self):
        """Resume: rerunning an interrupted sweep reuses completed runs and ends
        with exactly the particles of an uninterrupted run."""

        class Interrupted(BaseException):
            pass

        uninterrupted, _ = self.run_sbi(self.registry())
        total_runs = self.cli.run.call_count

        registry = self.registry("resume.duckdb")
        seen = []

        def interrupting(registry, run_hash, job):
            seen.append(run_hash)
            if len(seen) == 90:
                raise Interrupted()
            return model_summaries(registry, run_hash, job)

        self.cli.run.reset_mock()
        with self.assertRaises(Interrupted):
            self.run_sbi(registry, summaries=interrupting)
        runs_before_crash = self.cli.run.call_count

        self.cli.run.reset_mock()
        resumed, _ = self.run_sbi(registry)

        self.assertTrue(bool((resumed.samples == uninterrupted.samples).all()))
        self.assertEqual(resumed.run_hashes, uninterrupted.run_hashes)
        self.assertLess(self.cli.run.call_count, total_runs)
        self.assertEqual(self.cli.run.call_count, total_runs - runs_before_crash)


class TestAdaptiveBatchExportGuard(unittest.TestCase):
    """A multi-job batch must not share one export file."""

    def setUp(self):
        from joshpy.cli import CLIResult
        from joshpy.jobs import JobConfig, SweepConfig
        from joshpy.registry import RunRegistry
        from joshpy.strategies import AdaptiveContext

        self.tmp = Path(tempfile.mkdtemp())
        josh = self.tmp / "model.josh"
        josh.write_text("start simulation Main\nend simulation\n")
        self.registry = RunRegistry(":memory:")
        self.cli = MagicMock()
        self.cli.run.return_value = CLIResult(0, "", "", [])
        config = JobConfig(
            template_string="growth = {{ growth }} m",
            source_path=josh,
            sweep=SweepConfig(strategy=SbiStrategy(prior={"growth": {"uniform": [0, 1]}})),
        )
        self.ctx = AdaptiveContext(
            cli=self.cli, config=config, registry=self.registry,
            session_id=self.registry.create_session(config=config),
            quiet=True, stop_on_failure=False,
        )
        patcher = patch("joshpy.sweep.load_job_results", return_value=1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.ctx.cleanup()
        self.registry.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def export_path(self, path):
        self.cli.inspect_exports.return_value = _export_paths(path)

    def test_shared_export_path_raises_for_multi_job_batch(self):
        self.export_path(str(self.tmp / "results_{replicate}.csv"))
        with self.assertRaises(ValueError) as raised:
            self.ctx.evaluate([{"growth": 0.1}, {"growth": 0.2}])
        self.assertIn("{run_hash}", str(raised.exception))
        self.cli.run.assert_not_called()

    def test_single_job_batch_may_share_export_path(self):
        self.export_path(str(self.tmp / "results_{replicate}.csv"))
        self.ctx.evaluate([{"growth": 0.1}])
        self.cli.run.assert_called_once()

    def test_run_hash_in_export_path_is_fine(self):
        self.export_path(str(self.tmp / "results_{run_hash}_{replicate}.csv"))
        self.ctx.evaluate([{"growth": 0.1}, {"growth": 0.2}])
        self.assertEqual(self.cli.run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
