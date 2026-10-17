"""Known-truth SMC-ABC on the ForeverTree survey, from the command line.

1. Run the model once at a chosen truth and survey it: eight fixed cells, each
   measured with error. The survey's mean height and between-cell CV are the
   pseudo-observed data.
2. Simulate the prior (200 runs) to scale the statistics.
3. Run SMC-ABC (from the sbi package), with joshpy as the simulator: every
   proposal is rendered, run, registered, loaded and surveyed the same way.
4. Report whether the truth falls inside the final population's central 90%.

The case study lives in ``forevertree_case.py``; the Inference docs use it too.
The mean alone cannot separate the two parameters (faster growth with more
drought sensitivity gives the same mean height); ``--mean-only`` runs that
version to show the resulting ridge.

Usage (from the repository root):

    pixi run python examples/sbi_forevertree/run_abc.py --registry abc.duckdb

The first run simulates about 450 parameter sets. Rerunning against the same
registry resumes: completed simulations are reused, not rerun.
"""

from __future__ import annotations

import argparse

import forevertree_case as case
import numpy as np

from joshpy.cli import JoshCLI
from joshpy.jar import JarMode
from joshpy.registry import RunRegistry


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registry", default="forevertree_abc.duckdb")
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--budget", type=int, default=300, help="total simulations")
    parser.add_argument("--mean-only", action="store_true")
    args = parser.parse_args()

    cli = JoshCLI(josh_jar=JarMode.DEV)
    registry = RunRegistry(args.registry)
    try:
        study = case.load(registry, cli)
        print(f"Pseudo-observed survey at {case.TRUTH}: {study.observed}")

        summaries, observed = study.abc_target(["mean_height"] if args.mean_only else None)
        result, _ = case.run_sbi(
            registry,
            cli,
            summaries,
            observed,
            "smcabc",
            {
                "num_particles": args.particles,
                "num_initial_pop": 2 * args.particles,
                "num_simulations": args.budget,
                "epsilon_decay": 0.5,
            },
            "forevertree_abc_mean_only" if args.mean_only else "forevertree_abc",
        )

        df = result.to_dataframe()
        weights = df["weight"].to_numpy()
        print(
            f"\n{len(df)} particles, epsilons {[round(p['epsilon'], 4) for p in result.populations]}"
        )
        all_inside = True
        for name, truth in case.TRUTH.items():
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
