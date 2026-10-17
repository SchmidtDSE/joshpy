"""ForeverTree known-truth case study, shared by the Inference docs and run_abc.py.

Put this directory on ``sys.path`` and import it::

    sys.path.insert(0, "examples/sbi_forevertree")
    import forevertree_case as case

    registry = RunRegistry("forevertree.duckdb")
    study = case.load(registry, JoshCLI(josh_jar=JarMode.DEV))

Nothing runs at import. ``load()`` makes sure the pseudo-observed run and the
prior simulations exist in the registry, reusing them by ``run_hash`` when they
are already there, so a docs page can start from the middle of the case study.

The pseudo-observed data are a **survey**: the final-year mean tree height of a
fixed set of cells, each measured with error. The same survey is applied to the
truth and to every simulation.

Only the pseudo-observed run and the replicate study are seeded. The prior
simulations draw their own noise, as independent realizations should, so a
registry built from scratch gives slightly different numbers than a reused one.
After changing this module, re-render the Inference pages in order
(``pixi run docs-inference``).
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joshpy.cell_data import DiagnosticQueries
from joshpy.cli import JoshCLI
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
TRUTH_SEED = 1  # Josh seed of the pseudo-observed run
REPLICATE_SEED = 2  # Josh seed of the replicate study
SBI_SEED = 0  # fixes sbi's proposals, not Josh's noise
N_PRIOR = 200
FINAL_STEP = 10  # steps.high in forevertree.josh

# Fixed survey sites, like monitoring plots: grid positions (x, y) of 8 of the
# 63 cells. x runs west to east, y north to south.
SURVEY_CELLS = [
    (0.5, 4.5),
    (1.5, 1.5),
    (2.5, 5.5),
    (3.5, 2.5),
    (4.5, 0.5),
    (5.5, 3.5),
    (6.5, 6.5),
    (8.5, 1.5),
]
MEASUREMENT_SD = 0.25  # m, error in each surveyed cell's mean height
STATISTICS = ["mean_height", "height_cv"]  # what ABC compares


def model_config(**kwargs: Any) -> JobConfig:
    """The ForeverTree job; kwargs add the sweep, seed, replicates, ..."""
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


def run_at(
    registry: RunRegistry, cli: JoshCLI, params: dict[str, float], experiment: str, **kwargs: Any
) -> Any:
    """Run the model once at fixed parameters, unless that run is already loaded.

    Returns the expanded job; its ``run_hash`` keys the results in the registry.
    """
    config = model_config(
        sweep=SweepConfig(
            config_parameters=[
                ConfigSweepParameter(name=name, values=[value]) for name, value in params.items()
            ]
        ),
        **kwargs,
    )
    manager = (
        SweepManager.builder(config)
        .with_registry(registry, experiment_name=experiment)
        .with_cli(cli)
        .build()
    )
    try:
        job = manager.job_set.jobs[0]
        if len(registry.loaded_replicates(job.run_hash)) < job.replicates:
            manager.run(quiet=True)
            manager.load_results(quiet=True)
        return job
    finally:
        manager.cleanup()


def run_sbi(
    registry: RunRegistry,
    cli: JoshCLI,
    summaries: Any,
    observed: dict[str, float],
    method: str,
    method_kwargs: dict[str, Any],
    experiment: str,
) -> tuple[Any, str]:
    """Run sbi with joshpy as the simulator; returns (SbiResult, session_id).

    A rerun with the same arguments against the same registry replays the
    search, reusing every completed run.
    """
    strategy = SbiStrategy(
        prior=PRIOR,
        summaries=summaries,
        observed=observed,
        method=method,
        method_kwargs=method_kwargs,
        seed=SBI_SEED,
    )
    manager = (
        SweepManager.builder(model_config(sweep=SweepConfig(strategy=strategy)))
        .with_registry(registry, experiment_name=experiment)
        .with_cli(cli)
        .build()
    )
    try:
        return manager.run(quiet=True, stop_on_failure=False), manager.session_id
    finally:
        manager.cleanup()


# --- Observation: raw output -> survey -> statistics ---------------------------------


def final_heights(registry: RunRegistry, run_hash: str, replicate: int = 0) -> pd.DataFrame:
    """Final-year mean tree height of every cell: columns x, y, height."""
    cells = DiagnosticQueries(registry).get_all_variables_at_step(FINAL_STEP, run_hash, replicate)
    if cells.empty:
        raise ValueError(f"No final-year cells for run {run_hash}, replicate {replicate}")
    cells = cells.rename(columns={"position_x": "x", "position_y": "y", "meanHeight": "height"})
    return cells[["x", "y", "height"]].sort_values(["y", "x"], ignore_index=True)


def measurement_rng(run_hash: str, replicate: int) -> np.random.Generator:
    """Measurement noise seeded by the run and replicate, so replay reproduces it."""
    digest = hashlib.sha256(f"{run_hash}:{replicate}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def survey(
    registry: RunRegistry, run_hash: str, replicate: int = 0, *, error: bool = True
) -> pd.DataFrame:
    """Survey one run: the true and measured final-year height of each surveyed cell."""
    cells = final_heights(registry, run_hash, replicate).set_index(["x", "y"])
    surveyed = cells.loc[SURVEY_CELLS].reset_index()
    noise = measurement_rng(run_hash, replicate).normal(0.0, MEASUREMENT_SD, len(surveyed))
    surveyed["measured"] = surveyed["height"] + (noise if error else 0.0)
    return surveyed


def statistics(heights: Any) -> dict[str, float]:
    """Candidate summary statistics of a set of measured heights."""
    values = np.asarray(heights, dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    return {"mean_height": mean, "height_sd": sd, "height_var": sd**2, "height_cv": sd / mean}


def survey_summaries(registry: RunRegistry, run_hash: str, job: Any) -> dict[str, float]:
    """S_sim for ABC: mean and CV of one run's measured survey (replicate 0)."""
    measured = statistics(survey(registry, run_hash)["measured"])
    return {name: measured[name] for name in STATISTICS}


# --- The case study ---------------------------------------------------------------


@dataclass
class CaseStudy:
    """The known-truth setup every Inference page starts from."""

    truth_job: Any  # the pseudo-observed run
    truth_survey: pd.DataFrame  # its surveyed cells: x, y, height, measured
    reference: pd.DataFrame  # prior simulations: run_hash, parameters, statistics
    scale: dict[str, float]  # prior SD of each statistic in STATISTICS

    @property
    def candidates(self) -> dict[str, float]:
        """Every candidate statistic of the pseudo-observed survey."""
        return statistics(self.truth_survey["measured"])

    @property
    def observed(self) -> dict[str, float]:
        """S_obs: the statistics ABC compares, unscaled."""
        return {name: self.candidates[name] for name in STATISTICS}

    @property
    def observed_scaled(self) -> dict[str, float]:
        """S_obs in units of each statistic's prior SD."""
        return {name: self.observed[name] / self.scale[name] for name in STATISTICS}

    def distance(self, table: pd.DataFrame) -> pd.Series:
        """Euclidean distance from S_obs of each row's statistics, in prior SDs."""
        offsets = [(table[name] - self.observed[name]) / self.scale[name] for name in STATISTICS]
        return np.sqrt(sum(offset**2 for offset in offsets))

    def abc_target(self, names: list[str] | None = None) -> tuple[Any, dict[str, float]]:
        """(summaries function, S_obs) for ABC on ``names``, both in prior SDs."""
        names = list(names or STATISTICS)
        scale = {name: self.scale[name] for name in names}

        def scaled_summaries(registry: RunRegistry, run_hash: str, job: Any) -> dict[str, float]:
            measured = statistics(survey(registry, run_hash)["measured"])
            return {name: measured[name] / scale[name] for name in names}

        return scaled_summaries, {name: self.observed[name] / scale[name] for name in names}


def observe_truth(registry: RunRegistry, cli: JoshCLI) -> tuple[Any, dict[str, float]]:
    """Run the model at the truth (seeded) and survey it: (job, S_obs)."""
    job = run_at(registry, cli, TRUTH, "forevertree_truth", seed=TRUTH_SEED)
    return job, survey_summaries(registry, job.run_hash, job)


def simulate_prior(
    registry: RunRegistry, cli: JoshCLI, observed: dict[str, float], n: int = N_PRIOR
) -> Any:
    """Simulate n parameter sets drawn from the prior; returns the SbiResult.

    sbi's rejection ABC keeping every draw (quantile=1.0) is a plain prior
    simulation. With the same seed, its first draws are SMC-ABC's first
    generation, so a later SMC-ABC search reuses these runs.
    """
    result, _ = run_sbi(
        registry,
        cli,
        survey_summaries,
        observed,
        "mcabc",
        {"num_simulations": n, "quantile": 1.0},
        "forevertree_prior",
    )
    if result.failed:
        raise RuntimeError(f"{result.failed} prior simulation(s) failed")
    return result


def reference_table(registry: RunRegistry, jobs: list[Any]) -> pd.DataFrame:
    """Parameters and candidate statistics of each run: the ABC reference table."""
    rows = []
    for job in jobs:
        measured = statistics(survey(registry, job.run_hash)["measured"])
        rows.append({"run_hash": job.run_hash, **job.parameters, **measured})
    return pd.DataFrame(rows)


def load(registry: RunRegistry, cli: JoshCLI, n_prior: int = N_PRIOR) -> CaseStudy:
    """The case study, simulating only what the registry doesn't already hold."""
    truth_job, observed = observe_truth(registry, cli)
    prior = simulate_prior(registry, cli, observed, n_prior)
    reference = reference_table(registry, [job for job, _ in prior.job_results])
    scale = {name: float(reference[name].std()) for name in STATISTICS}
    return CaseStudy(truth_job, survey(registry, truth_job.run_hash), reference, scale)


def replicate_study(registry: RunRegistry, cli: JoshCLI, n: int = 50) -> pd.DataFrame:
    """Survey n replicates at the truth, with and without measurement error.

    Returns one row per replicate and observation ("model only": the surveyed
    cells' true heights; "survey": with measurement error), with the candidate
    statistics.
    """
    job = run_at(registry, cli, TRUTH, "forevertree_replicates", seed=REPLICATE_SEED, replicates=n)
    rows = []
    for replicate in range(n):
        for observation, error in [("model only", False), ("survey", True)]:
            surveyed = survey(registry, job.run_hash, replicate, error=error)
            rows.append(
                {
                    "replicate": replicate,
                    "observation": observation,
                    **statistics(surveyed["measured"]),
                }
            )
    return pd.DataFrame(rows)


def show_source(*objects: Any) -> None:
    """Print objects' source as a fenced Python block (docs: use with output: asis)."""
    # A four-backtick fence, so source that contains ``` can't close it early.
    print("````python\n" + "\n\n".join(inspect.getsource(o).rstrip() for o in objects) + "\n````")


# --- Figures (each draws and shows one figure, and returns None) --------------------

SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE, ORANGE = "#2a78d6", "#eb6834"
HEIGHT_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
LABELS = {
    "mean_height": "mean",
    "height_sd": "SD",
    "height_var": "variance",
    "height_cv": "CV",
}


def _style(ax: Any, grid: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    if grid:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)


def _title(ax: Any, text: str) -> None:
    ax.set_title(text, color=INK, fontsize=10, loc="left")


def _figure(*args: Any, **kwargs: Any) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def plot_pipeline(study: CaseStudy, registry: RunRegistry, epsilon: float) -> None:
    """Raw output -> survey -> statistics -> comparison, for the pseudo-observed run."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, Rectangle

    run_hash = study.truth_job.run_hash
    cells = final_heights(registry, run_hash)
    surveyed = survey(registry, run_hash)
    order = surveyed.sort_values("x", ignore_index=True)
    order["site"] = np.arange(1, len(order) + 1)
    stats = statistics(surveyed["measured"])

    fig, axes = _figure(2, 2, figsize=(10, 7.6))
    (ax_map, ax_survey), (ax_stats, ax_compare) = axes

    # 1. Raw output: every cell's final-year height.
    grid = cells.pivot(index="y", columns="x", values="height")
    ramp = LinearSegmentedColormap.from_list("height", HEIGHT_RAMP)
    image = ax_map.imshow(
        grid.to_numpy(), cmap=ramp, origin="upper",
        extent=(0, grid.shape[1], grid.shape[0], 0),
    )  # fmt: skip
    for _, row in order.iterrows():
        ax_map.add_patch(
            Rectangle(
                (row["x"] - 0.5, row["y"] - 0.5), 1, 1,
                fill=False, edgecolor=INK, linewidth=1.6,
            )
        )  # fmt: skip
        ax_map.text(
            row["x"], row["y"], str(int(row["site"])), ha="center", va="center", fontsize=8,
            color=INK, bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"),
        )  # fmt: skip
    colorbar = fig.colorbar(image, ax=ax_map, fraction=0.035, pad=0.03)
    colorbar.set_label("height (m)", color=INK_2, fontsize=9)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    colorbar.outline.set_edgecolor(AXIS)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_xlabel("west → east", color=INK_2)
    ax_map.set_ylabel("north → south", color=INK_2)
    _title(ax_map, f"1  Model output: final-year height, {len(cells)} cells")

    # 2. Survey: the outlined cells, measured with error.
    _style(ax_survey)
    for _, row in order.iterrows():
        ax_survey.plot(
            [row["site"]] * 2, [row["height"], row["measured"]], color=AXIS, linewidth=1.2
        )
    ax_survey.scatter(
        order["site"], order["height"], s=46, facecolor=SURFACE, edgecolor=INK_2,
        linewidth=1.4, zorder=3, label="true",
    )  # fmt: skip
    ax_survey.scatter(
        order["site"], order["measured"], s=46, color=ORANGE, edgecolor=SURFACE,
        linewidth=1.2, zorder=4, label="measured",
    )  # fmt: skip
    ax_survey.set_xticks(order["site"])
    ax_survey.set_xlabel("survey site (west → east)", color=INK_2)
    ax_survey.set_ylabel("height (m)", color=INK_2)
    ax_survey.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    _title(ax_survey, f"2  Survey: {len(order)} sites, measurement SD {MEASUREMENT_SD} m")

    # 3. Statistics of the measured heights.
    _style(ax_stats)
    mean, sd = stats["mean_height"], stats["height_sd"]
    ax_stats.axvspan(mean - sd, mean + sd, color=GRID, zorder=0)
    ax_stats.axvline(mean, color=INK, linewidth=1.4)
    ax_stats.scatter(
        surveyed["measured"], np.zeros(len(surveyed)), s=46, color=ORANGE,
        edgecolor=SURFACE, linewidth=1.2, zorder=3,
    )  # fmt: skip
    ax_stats.set_ylim(-1, 1)
    ax_stats.set_yticks([])
    ax_stats.set_xlabel("measured height (m)", color=INK_2)
    ax_stats.text(
        0.02, 0.95,
        f"mean {mean:.2f} m\nSD {sd:.2f} m (band)\nCV = SD / mean = {stats['height_cv']:.3f}",
        transform=ax_stats.transAxes, va="top", fontsize=9, color=INK,
    )  # fmt: skip
    _title(ax_stats, "3  Summary statistics")

    # 4. Comparison with the prior simulations, in prior SDs.
    _style(ax_compare)
    reference = study.reference
    inside = study.distance(reference) <= epsilon
    sx, sy = (reference[name] / study.scale[name] for name in STATISTICS)
    ox, oy = (study.observed_scaled[name] for name in STATISTICS)
    ax_compare.scatter(
        sx[~inside], sy[~inside], s=20, facecolor="none", edgecolor=MUTED, linewidth=0.8,
        label="prior simulation",
    )  # fmt: skip
    ax_compare.scatter(
        sx[inside], sy[inside], s=26, color=BLUE, edgecolor=SURFACE, linewidth=0.8,
        label="within ε of S_obs", zorder=3,
    )  # fmt: skip
    ax_compare.add_patch(
        Circle((ox, oy), epsilon, fill=False, edgecolor=INK, linestyle="--", linewidth=1.2)
    )
    ax_compare.scatter([ox], [oy], marker="x", s=80, color=INK, linewidth=2, zorder=4)
    ax_compare.annotate(
        r"$S_\mathrm{obs}$", (ox, oy), xytext=(8, 6), textcoords="offset points",
        fontsize=9, color=INK,
    )  # fmt: skip
    ax_compare.set_aspect("equal", adjustable="datalim")
    ax_compare.set_xlabel("mean height (prior SDs)", color=INK_2)
    ax_compare.set_ylabel("CV (prior SDs)", color=INK_2)
    ax_compare.legend(
        fontsize=9,
        labelcolor=INK_2,
        loc="best",
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=0.9,
    )
    _title(ax_compare, "4  Comparison: within ε of S_obs?")

    fig.tight_layout()
    plt.show()


def plot_candidates(reference: pd.DataFrame, observed: dict[str, float]) -> None:
    """Each candidate statistic against each parameter, over the prior simulations.

    ``observed`` holds the pseudo-observed value of each statistic (dashed line).
    """
    import matplotlib.pyplot as plt

    fig, axes = _figure(2, len(LABELS), figsize=(11, 5.4), sharey="col")
    for row, param in enumerate(PRIOR):
        for col, (name, label) in enumerate(LABELS.items()):
            ax = axes[row, col]
            _style(ax)
            ax.scatter(
                reference[param], reference[name], s=14, color=BLUE, edgecolor=SURFACE,
                linewidth=0.4,
            )  # fmt: skip
            ax.axhline(observed[name], color=INK_2, linestyle="--", linewidth=0.9)
            ax.axvline(TRUTH[param], color=MUTED, linestyle=":", linewidth=0.9)
            rho = reference[param].corr(reference[name], method="spearman")
            ax.text(
                0.03, 0.95, f"ρ = {rho:+.2f}", transform=ax.transAxes, va="top", fontsize=9,
                color=INK, bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                                     edgecolor="none", alpha=0.85),
            )  # fmt: skip
            if row == 0:
                _title(ax, label)
            ax.set_xlabel(param, color=INK_2, fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_replicates(study: CaseStudy, replicates: pd.DataFrame, epsilon: float) -> None:
    """Replicates at the truth around S_obs, and the fraction within each tolerance."""
    import matplotlib.pyplot as plt

    series = [("model only", BLUE), ("survey", ORANGE)]
    fig, axes = _figure(1, 3, figsize=(11, 3.5))
    for ax, name in zip(axes[:2], STATISTICS, strict=True):
        _style(ax)
        for observation, color in series:
            values = replicates.loc[replicates["observation"] == observation, name]
            offset = (values - study.observed[name]) / study.scale[name]
            ax.hist(offset, bins=12, histtype="step", linewidth=2, color=color, label=observation)
        ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
        ax.set_xlabel(r"$S_\mathrm{sim} - S_\mathrm{obs}$ (prior SDs)", color=INK_2)
        _title(ax, LABELS[name])
    axes[0].set_ylabel("replicates", color=INK_2)

    ax = axes[2]
    _style(ax)
    smallest = np.inf
    for observation, color in series:
        rows = replicates[replicates["observation"] == observation]
        distances = np.sort(study.distance(rows).to_numpy())
        fraction = np.arange(1, len(distances) + 1) / len(distances)
        ax.step(distances, fraction, where="post", color=color, linewidth=2, label=observation)
        smallest = min(smallest, distances[0])
    ax.axvline(epsilon, color=INK, linestyle="--", linewidth=1.2)
    ax.annotate(
        "ε", (epsilon, 0.06), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=10,
    )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(smallest * 0.7, epsilon * 2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("tolerance ε (prior SDs, log scale)", color=INK_2)
    ax.set_ylabel("fraction within ε", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _title(ax, "Estimated ABC likelihood at the truth")
    fig.tight_layout()
    plt.show()
