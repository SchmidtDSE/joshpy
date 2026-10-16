"""Known-truth ABC check on the ForeverTree demo model with SbiStrategy.

1. Run the model once at a chosen truth; its summary statistics become the
   "observed" data.
2. Run SMC-ABC (from the sbi package) over a uniform prior, with joshpy as the
   simulator: every proposal is rendered, run, registered and loaded.
3. Report whether the truth falls inside the final population's central 90%.

The two summaries are the final landscape mean of tree height and the
between-cell coefficient of variation. The mean alone cannot separate the two
parameters (faster growth with more drought sensitivity gives the same mean
height); ``--mean-only`` runs that version to show the resulting ridge.

Usage (from the repository root):

    pixi run python examples/sbi_forevertree/run_abc.py --registry abc.duckdb

Rerunning with the same registry and seed resumes an interrupted run: completed
simulations are reused, not rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from joshpy.cli import JoshCLI
from joshpy.jar import JarMode
from joshpy.jobs import ConfigSweepParameter, JobConfig, SweepConfig
from joshpy.registry import RunRegistry
from joshpy.strategies import SbiStrategy
from joshpy.sweep import SweepManager

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "demo"

TRUTH = {"maxNewGrowth": 1.0, "minPrecipImpactPct": 40.0}
PRIOR = {
    "maxNewGrowth": {"uniform": [0.2, 2.0]},
    "minPrecipImpactPct": {"uniform": [0.0, 100.0]},
}
NOMINAL_HEIGHT = 10.0  # m; scales mean height to roughly the range of the CV


def height_summaries(registry, run_hash, job):
    """Final-step landscape mean height (scaled) and between-cell CV."""
    mean, sd = registry.conn.execute(
        """
        SELECT avg(meanHeight), stddev_samp(meanHeight)
        FROM cell_data_current
        WHERE run_hash = ?
          AND step = (SELECT max(step) FROM cell_data_current WHERE run_hash = ?)
        """,
        [run_hash, run_hash],
    ).fetchone()
    return {"mean_height": mean / NOMINAL_HEIGHT, "height_cv": sd / mean}


def mean_height_only(registry, run_hash, job):
    """The confounded single summary, for the equifinality comparison."""
    return {"mean_height": height_summaries(registry, run_hash, job)["mean_height"]}


def model_config(**kwargs) -> JobConfig:
    return JobConfig(
        template_path=HERE / "forevertree.jshc.j2",
        config_name="forevertree",  # the model reads `config forevertree.*`
        source_path=HERE / "forevertree.josh",
        file_mappings={
            "temperature": DATA / "temperature.jshd",
            "precipitation": DATA / "precipitation.jshd",
        },
        **kwargs,
    )


def observe_truth(registry: RunRegistry, cli: JoshCLI, summaries) -> dict[str, float]:
    """Run the model at TRUTH and return its summaries as the observed data."""
    config = model_config(
        sweep=SweepConfig(
            config_parameters=[
                ConfigSweepParameter(name=name, values=[value]) for name, value in TRUTH.items()
            ]
        )
    )
    manager = (
        SweepManager.builder(config)
        .with_registry(registry, experiment_name="forevertree_truth")
        .with_cli(cli)
        .build()
    )
    try:
        manager.run(quiet=True)
        manager.load_results(quiet=True)
        job = manager.job_set.jobs[0]
        return summaries(registry, job.run_hash, job)
    finally:
        manager.cleanup()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registry", default="forevertree_abc.duckdb")
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--budget", type=int, default=300, help="total simulations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean-only", action="store_true")
    args = parser.parse_args()

    summaries = mean_height_only if args.mean_only else height_summaries
    cli = JoshCLI(josh_jar=JarMode.DEV)
    registry = RunRegistry(args.registry)
    try:
        observed = observe_truth(registry, cli, summaries)
        print(f"Observed at truth {TRUTH}: {observed}")

        strategy = SbiStrategy(
            prior=PRIOR,
            summaries=summaries,
            observed=observed,
            method_kwargs={
                "num_particles": args.particles,
                "num_initial_pop": 2 * args.particles,
                "num_simulations": args.budget,
                "epsilon_decay": 0.5,
            },
            seed=args.seed,
        )
        manager = (
            SweepManager.builder(model_config(sweep=SweepConfig(strategy=strategy)))
            .with_registry(registry, experiment_name="forevertree_abc")
            .with_cli(cli)
            .build()
        )
        try:
            result = manager.run(stop_on_failure=False)
        finally:
            manager.cleanup()

        df = result.to_dataframe()
        weights = df["weight"].to_numpy()
        print(f"\n{len(df)} particles, epsilons {[round(p['epsilon'], 4) for p in result.populations]}")
        all_inside = True
        for name, truth in TRUTH.items():
            values = df[name].to_numpy()
            low = weighted_quantile(values, weights, 0.05)
            high = weighted_quantile(values, weights, 0.95)
            mean = float(np.average(values, weights=weights))
            inside = low <= truth <= high
            all_inside &= inside
            print(
                f"  {name}: mean {mean:.3f}, 90% [{low:.3f}, {high:.3f}], "
                f"truth {truth} {'inside' if inside else 'OUTSIDE'}"
            )
        print("Truth recovered." if all_inside else "Truth NOT inside the central 90%.")
    finally:
        registry.close()


if __name__ == "__main__":
    main()
