"""ForeverTree known-truth case study, shared by the Inference docs and run_abc.py.

Put this directory on ``sys.path`` and import it::

    sys.path.insert(0, "examples/sbi_forevertree")
    import forevertree_case as case

    registry = RunRegistry("forevertree.duckdb")
    study = case.known_truth_study().load(registry, JoshCLI(josh_jar=JarMode.DEV))

joshpy's :class:`~joshpy.inference.KnownTruthStudy` runs the workflow: the
pseudo-observed run, the prior simulations, z-scores and searches, reusing runs
the registry already holds. This module holds what is specific to ForeverTree:
the model, its prior and truth, the survey and the candidate statistics, and
the docs' figures.

The pseudo-observed data are a **survey**: the final-year measured mean tree
height of a fixed set of cells. The model draws the measurement error itself
(``measuredHeight`` in forevertree.josh), so every run and replicate is measured
with its own error, and the same survey applies to the truth and to every
simulation.

Only the pseudo-observed run and the replicate study are seeded. The prior
simulations draw their own noise, as independent realizations should, so a
registry built from scratch gives slightly different numbers than a reused one.
After changing this module or the model, re-render the Inference pages in order
(``pixi run docs-inference``).
"""

from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joshpy.cell_data import DiagnosticQueries
from joshpy.cli import JoshCLI
from joshpy.inference import KnownTruthStudy, LoadedStudy
from joshpy.jobs import JobConfig
from joshpy.registry import RunRegistry

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "demo"

TRUTH = {"maxNewGrowth": 1.0, "minPrecipImpactPct": 40.0}
PRIOR = {
    "maxNewGrowth": {"uniform": [0.2, 2.0]},
    "minPrecipImpactPct": {"uniform": [0.0, 100.0]},
}
TRUTH_SEED = 3  # Josh seed of the pseudo-observed run
REPLICATE_SEED = 2  # Josh seed of the replicate study
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
MEASUREMENT_SD = 0.25  # m; drawn in forevertree.josh (measuredHeight), shown in figures
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


# --- Observation: raw output -> survey -> statistics ---------------------------------


def final_heights(registry: RunRegistry, run_hash: str, replicate: int = 0) -> pd.DataFrame:
    """Final-year mean tree height of every cell: columns x, y, height (true) and measured."""
    cells = DiagnosticQueries(registry).get_all_variables_at_step(FINAL_STEP, run_hash, replicate)
    if cells.empty:
        raise ValueError(f"No final-year cells for run {run_hash}, replicate {replicate}")
    cells = cells.rename(
        columns={
            "position_x": "x",
            "position_y": "y",
            "meanHeight": "height",
            "measuredHeight": "measured",
        }
    )
    return cells[["x", "y", "height", "measured"]].sort_values(["y", "x"], ignore_index=True)


def observe(
    registry: RunRegistry, run_hash: str, replicate: int = 0, *, error: bool = True
) -> pd.DataFrame:
    """The survey of one run: true and measured final-year height of each surveyed cell.

    With ``error=False``, the survey records the true heights (model noise only).
    """
    cells = final_heights(registry, run_hash, replicate).set_index(["x", "y"])
    surveyed = cells.loc[SURVEY_CELLS].reset_index()
    if not error:
        surveyed["measured"] = surveyed["height"]
    return surveyed


def statistics(survey: pd.DataFrame) -> dict[str, float]:
    """Candidate summary statistics of a survey's measured heights."""
    values = survey["measured"].to_numpy(dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    return {"mean_height": mean, "height_sd": sd, "height_var": sd**2, "height_cv": sd / mean}


def known_truth_study() -> KnownTruthStudy:
    """The ForeverTree study: what joshpy simulates, and how each run is observed."""
    return KnownTruthStudy(
        name="forevertree",  # runs are recorded as forevertree_truth, forevertree_prior, ...
        model=model_config(),
        prior=PRIOR,
        truth=TRUTH,
        observe=observe,
        statistics=statistics,
        compare=STATISTICS,
        truth_seed=TRUTH_SEED,
    )


def replicate_study(registry: RunRegistry, cli: JoshCLI, n: int = 50) -> pd.DataFrame:
    """Statistics of n replicates at the truth, surveyed without and with measurement error.

    One row per replicate and observation: "model only" (the surveyed cells'
    true heights) or "survey" (with measurement error). Both read the same runs.
    """
    study = known_truth_study()
    without_error = partial(observe, error=False)
    tables = {
        "model only": study.replicates(registry, cli, n, REPLICATE_SEED, observe=without_error),
        "survey": study.replicates(registry, cli, n, REPLICATE_SEED),
    }
    return pd.concat(
        [table.assign(observation=name) for name, table in tables.items()], ignore_index=True
    )


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
AXIS_LABELS = {"mean_height": "mean height (m)", "height_cv": "CV of height"}
Z_LABELS = {"mean_height": "mean height, z-score", "height_cv": "CV of height, z-score"}


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


def _legend(ax: Any, **kwargs: Any) -> None:
    ax.legend(
        fontsize=9, labelcolor=INK_2, facecolor=SURFACE, edgecolor="none", framealpha=0.9,
        **kwargs,
    )  # fmt: skip


def _draw_map(
    fig: Any,
    ax: Any,
    cells: pd.DataFrame,
    sites: pd.DataFrame,
    *,
    limits: tuple[float, float] | None = None,
) -> None:
    """Final-year height of every cell, with the surveyed cells outlined and numbered."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    grid = cells.pivot(index="y", columns="x", values="height")
    ramp = LinearSegmentedColormap.from_list("height", HEIGHT_RAMP)
    vmin, vmax = limits if limits else (None, None)
    image = ax.imshow(
        grid.to_numpy(), cmap=ramp, origin="upper", vmin=vmin, vmax=vmax,
        extent=(0, grid.shape[1], grid.shape[0], 0),
    )  # fmt: skip
    for _, row in sites.iterrows():
        ax.add_patch(
            Rectangle(
                (row["x"] - 0.5, row["y"] - 0.5), 1, 1,
                fill=False, edgecolor=INK, linewidth=1.6,
            )
        )  # fmt: skip
        ax.text(
            row["x"], row["y"], str(int(row["site"])), ha="center", va="center", fontsize=8,
            color=INK, bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"),
        )  # fmt: skip
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("true height (m)", color=INK_2, fontsize=9)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    colorbar.outline.set_edgecolor(AXIS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("west → east", color=INK_2)
    ax.set_ylabel("north → south", color=INK_2)


def _numbered_sites(surveyed: pd.DataFrame) -> pd.DataFrame:
    """Surveyed cells ordered west to east, numbered from 1."""
    order = surveyed.sort_values("x", ignore_index=True)
    order["site"] = np.arange(1, len(order) + 1)
    return order


def plot_observation(study: LoadedStudy, registry: RunRegistry) -> None:
    """Model output -> survey -> statistics, for the pseudo-observed run."""
    import matplotlib.pyplot as plt

    run_hash = study.truth_job.run_hash
    cells = final_heights(registry, run_hash)
    sites = _numbered_sites(study.truth_data)
    stats = statistics(sites)
    mean, sd, cv = stats["mean_height"], stats["height_sd"], stats["height_cv"]

    fig, (ax_map, ax_survey, ax_stats) = _figure(
        1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.25, 1.1, 0.75]}
    )

    # 1. Model output: every cell's final-year height.
    _draw_map(fig, ax_map, cells, sites)
    _title(
        ax_map,
        f"1  Model output: final-year height of\n    {len(cells)} cells, outlined if surveyed",
    )

    # 2. Survey: the outlined cells, measured with error.
    _style(ax_survey)
    for _, row in sites.iterrows():
        ax_survey.plot(
            [row["site"]] * 2, [row["height"], row["measured"]], color=AXIS, linewidth=1.2
        )
    ax_survey.scatter(
        sites["site"], sites["height"], s=46, facecolor=SURFACE, edgecolor=INK_2,
        linewidth=1.4, zorder=3, label="true",
    )  # fmt: skip
    ax_survey.scatter(
        sites["site"], sites["measured"], s=46, color=ORANGE, edgecolor=SURFACE,
        linewidth=1.2, zorder=4, label="measured",
    )  # fmt: skip
    ax_survey.set_xticks(sites["site"])
    ax_survey.set_xlabel("survey site (west → east)", color=INK_2)
    ax_survey.set_ylabel("height (m)", color=INK_2)
    _legend(ax_survey, loc="upper right")
    _title(
        ax_survey,
        f"2  Survey: {len(sites)} sites, each measured\n    with error (SD {MEASUREMENT_SD} m)",
    )

    # 3. Statistics: the measured heights reduce to a mean and a CV.
    _style(ax_stats)
    ax_stats.sharey(ax_survey)
    ax_stats.axhspan(mean - sd, mean + sd, color=GRID, zorder=0)
    ax_stats.axhline(mean, color=INK, linewidth=1.4)
    ax_stats.scatter(
        np.zeros(len(sites)), sites["measured"], s=46, color=ORANGE, edgecolor=SURFACE,
        linewidth=1.2, zorder=3,
    )  # fmt: skip
    ax_stats.set_xlim(-0.6, 1.6)
    ax_stats.set_xticks([])
    ax_stats.annotate(
        f"mean {mean:.2f} m", (0.35, mean), xytext=(0, 4), textcoords="offset points",
        fontsize=9, color=INK,
    )  # fmt: skip
    ax_stats.annotate(
        f"±1 SD ({sd:.2f} m)", (0.35, mean + sd), xytext=(0, -12), textcoords="offset points",
        fontsize=9, color=INK_2,
    )  # fmt: skip
    summary = (
        f"CV = SD / mean = {cv:.3f}\n"
        r"$S_\mathrm{obs}$ = (mean, CV)" + f"\n      = ({mean:.2f} m, {cv:.3f})"
    )
    ax_stats.text(
        0.04, 0.03, summary,
        transform=ax_stats.transAxes, va="bottom", fontsize=9, color=INK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=AXIS),
    )  # fmt: skip
    _title(ax_stats, "3  Summary statistics\n    of the measured heights")

    fig.tight_layout()
    plt.show()


def plot_comparison(study: LoadedStudy, epsilon: float) -> None:
    """Prior simulations against S_obs, in measured units (left) and as z-scores (right)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    x, y = STATISTICS
    reference = study.reference
    distance = study.distance(reference)
    inside = distance <= epsilon
    z = study.zscore(reference)
    observed, observed_z = study.observed, study.zscore(study.observed)

    fig, (ax_raw, ax_z) = _figure(1, 2, figsize=(11.5, 4.8))
    panels = [
        (ax_raw, reference, observed, AXIS_LABELS[x], AXIS_LABELS[y]),
        (ax_z, z, observed_z, Z_LABELS[x], Z_LABELS[y]),
    ]
    for ax, table, point, xlabel, ylabel in panels:
        _style(ax)
        ax.scatter(
            table.loc[~inside, x], table.loc[~inside, y], s=20, facecolor="none",
            edgecolor=MUTED, linewidth=0.8, label="prior simulation",
        )  # fmt: skip
        ax.scatter(
            table.loc[inside, x], table.loc[inside, y], s=26, color=BLUE, edgecolor=SURFACE,
            linewidth=0.8, label="within ε of S_obs (accepted)", zorder=3,
        )  # fmt: skip
        ax.scatter(
            [point[x]], [point[y]], marker="X", s=110, color=ORANGE, edgecolor=INK,
            linewidth=0.8, zorder=5, label=r"$S_\mathrm{obs}$",
        )  # fmt: skip
        ax.set_xlabel(xlabel, color=INK_2)
        ax.set_ylabel(ylabel, color=INK_2)

    # Left: in measured units, the region within ε is an ellipse, wider along the
    # statistic with the larger spread.
    ax_raw.add_patch(
        Ellipse(
            (observed[x], observed[y]), 2 * epsilon * study.scale[x], 2 * epsilon * study.scale[y],
            fill=False, edgecolor=INK, linestyle="--", linewidth=1.2,
        )
    )  # fmt: skip
    _legend(ax_raw, loc="best")
    _title(ax_raw, "4  Prior simulations and S_obs, in measured units")

    # Right: as z-scores, it is a circle of radius ε. One rejected simulation shows
    # the distance being compared.
    ax_z.add_patch(
        Circle(
            (observed_z[x], observed_z[y]), epsilon, fill=False, edgecolor=INK,
            linestyle="--", linewidth=1.2,
        )
    )  # fmt: skip
    example = (distance[~inside] - 3 * epsilon).abs().idxmin()
    ex, ey = z.at[example, x], z.at[example, y]
    ax_z.plot([observed_z[x], ex], [observed_z[y], ey], color=INK_2, linewidth=1.1, zorder=4)
    ax_z.annotate(
        f"distance {distance[example]:.2f} > ε: rejected", (ex, ey), xytext=(8, -4),
        textcoords="offset points", fontsize=9, color=INK,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85),
    )  # fmt: skip
    ax_z.set_aspect("equal", adjustable="datalim")
    _title(ax_z, f"5  As z-scores: accepted if distance ≤ ε = {epsilon:.2f}")

    fig.tight_layout()
    plt.show()


def plot_candidates(study: LoadedStudy) -> None:
    """Each candidate statistic against each parameter, over the prior simulations."""
    import matplotlib.pyplot as plt

    reference, observed = study.reference, study.candidates
    fig, axes = _figure(2, len(LABELS), figsize=(11, 5.9), sharey="col")
    for row, param in enumerate(PRIOR):
        for col, (name, label) in enumerate(LABELS.items()):
            ax = axes[row, col]
            _style(ax)
            ax.scatter(
                reference[param], reference[name], s=14, color=BLUE, edgecolor=SURFACE,
                linewidth=0.4,
            )  # fmt: skip
            ax.axhline(
                observed[name], color=ORANGE, linestyle="--", linewidth=1.5,
                label="pseudo-observed value of the statistic",
            )  # fmt: skip
            ax.axvline(TRUTH[param], color=INK, linewidth=1.0, label="true parameter value")
            rho = reference[param].corr(reference[name], method="spearman")
            ax.text(
                0.03, 0.95, f"ρ = {rho:+.2f}", transform=ax.transAxes, va="top", fontsize=9,
                color=INK, bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                                     edgecolor="none", alpha=0.85),
            )  # fmt: skip
            if row == 0:
                _title(ax, label)
            ax.set_xlabel(param, color=INK_2, fontsize=9)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
        fontsize=9, labelcolor=INK_2,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_replicates(study: LoadedStudy, replicates: pd.DataFrame, epsilon: float) -> None:
    """Replicates at the truth around S_obs, and the fraction within each tolerance."""
    import matplotlib.pyplot as plt

    series = [("model only", BLUE), ("survey", ORANGE)]
    observed_z = study.zscore(study.observed)
    fig, axes = _figure(1, 3, figsize=(11, 3.5))
    for ax, name in zip(axes[:2], STATISTICS, strict=True):
        _style(ax)
        for observation, color in series:
            rows = replicates[replicates["observation"] == observation]
            offset = study.zscore(rows)[name] - observed_z[name]
            ax.hist(offset, bins=12, histtype="step", linewidth=2, color=color, label=observation)
        ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
        ax.set_xlabel(r"$S_\mathrm{sim} - S_\mathrm{obs}$ (z-score units)", color=INK_2)
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
    ax.set_xlabel("tolerance ε (z-score units, log scale)", color=INK_2)
    ax.set_ylabel("fraction within ε", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _title(ax, "Estimated ABC likelihood at the truth")
    fig.tight_layout()
    plt.show()


# --- SMC-ABC results ---------------------------------------------------------------


def particle_parameters(result: Any) -> dict[str, dict[str, float]]:
    """run_hash -> parameters, for every run a search evaluated."""
    return {job.run_hash: job.parameters for job, _ in result.job_results}


def farthest_particle(result: Any) -> str:
    """run_hash of the final particle farthest from the truth, in units of the prior ranges."""
    df = result.to_dataframe()
    span = {name: spec["uniform"][1] - spec["uniform"][0] for name, spec in PRIOR.items()}
    offset = sum(((df[name] - TRUTH[name]) / span[name]) ** 2 for name in PRIOR)
    return str(df.loc[offset.idxmax(), "run_hash"])


def carried_into(populations: list[dict[str, Any]], t: int) -> set[str]:
    """Members of population t carried over from population t - 1 (budget top-up)."""
    if t == 0:
        return set()
    return set(populations[t - 1]["run_hashes"]) & set(populations[t]["run_hashes"])


def _draw_population(
    ax: Any, hashes: list[str], params: dict[str, dict[str, float]], title: str, carried: set[str]
) -> None:
    """Accepted parameter sets over the prior box; hollow points were carried over."""
    _style(ax)
    names = list(PRIOR)
    for subset, filled in [([h for h in hashes if h not in carried], True),
                           ([h for h in hashes if h in carried], False)]:  # fmt: skip
        if not subset:
            continue
        x = [params[h][names[0]] for h in subset]
        y = [params[h][names[1]] for h in subset]
        if filled:
            ax.scatter(x, y, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=2)
        else:
            ax.scatter(x, y, s=36, facecolor="none", edgecolor=BLUE, linewidth=1.2, zorder=2)
    tx, ty = (TRUTH[name] for name in names)
    ax.scatter([tx], [ty], marker="x", s=90, color=INK, linewidth=2, zorder=3)
    ax.annotate(
        "truth", (tx, ty), xytext=(7, 7), textcoords="offset points", color=INK, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85),
        zorder=4,
    )  # fmt: skip
    ax.set_xlim(*PRIOR[names[0]]["uniform"])
    ax.set_ylim(*PRIOR[names[1]]["uniform"])
    ax.set_xlabel(f"{names[0]} (m/yr)", color=INK_2)
    _title(ax, title)


def plot_generations(result: Any, columns: int = 3) -> None:
    """Each SMC-ABC generation's accepted parameter sets, over the prior box."""
    import matplotlib.pyplot as plt

    params = particle_parameters(result)
    populations = result.populations
    columns = min(columns, len(populations))
    rows = -(-len(populations) // columns)
    fig, axes = _figure(rows, columns, figsize=(3.4 * columns, 3.4 * rows), squeeze=False)
    for t, ax in enumerate(axes.flat):
        if t >= len(populations):
            ax.set_visible(False)
            continue
        population = populations[t]
        _draw_population(
            ax, population["run_hashes"], params,
            f"Generation {population['generation']}  ε = {population['epsilon']:.2g}",
            carried_into(populations, t),
        )  # fmt: skip
    for row in axes:
        row[0].set_ylabel(f"{list(PRIOR)[1]} (%)", color=INK_2)
    fig.tight_layout()
    plt.show()


def plot_final_populations(searches: list[tuple[str, Any]]) -> None:
    """Final populations of several searches side by side: (title, SbiResult) pairs."""
    import matplotlib.pyplot as plt

    fig, axes = _figure(
        1, len(searches), figsize=(3.7 * len(searches), 3.7), sharex=True, sharey=True,
        squeeze=False,
    )  # fmt: skip
    for ax, (title, result) in zip(axes[0], searches, strict=True):
        last = len(result.populations) - 1
        _draw_population(
            ax, result.populations[last]["run_hashes"], particle_parameters(result), title,
            carried_into(result.populations, last),
        )  # fmt: skip
    axes[0][0].set_ylabel(f"{list(PRIOR)[1]} (%)", color=INK_2)
    fig.tight_layout()
    plt.show()


def plot_final_states(registry: RunRegistry, runs: list[tuple[str, str]]) -> None:
    """Final-year maps and measured surveys of several runs: (label, run_hash) pairs.

    The maps share one color scale, so differences between rows are real.
    """
    import matplotlib.pyplot as plt

    cells = {run_hash: final_heights(registry, run_hash) for _, run_hash in runs}
    low = min(c["height"].min() for c in cells.values())
    high = max(c["height"].max() for c in cells.values())
    fig, axes = _figure(len(runs), 2, figsize=(10, 3.4 * len(runs)), squeeze=False)
    for (label, run_hash), (ax_map, ax_survey) in zip(runs, axes, strict=True):
        params = registry.get_config_by_hash(run_hash).parameters
        theta = ", ".join(f"{name} = {float(params[name]):.3g}" for name in PRIOR)
        sites = _numbered_sites(observe(registry, run_hash))
        _draw_map(fig, ax_map, cells[run_hash], sites, limits=(low, high))
        _title(ax_map, f"{label}\n{theta}")

        _style(ax_survey)
        stats = statistics(sites)
        mean, sd = stats["mean_height"], stats["height_sd"]
        ax_survey.axvspan(mean - sd, mean + sd, color=GRID, zorder=0)
        ax_survey.axvline(mean, color=INK, linewidth=1.4)
        ax_survey.scatter(
            sites["measured"], np.zeros(len(sites)), s=46, color=ORANGE, edgecolor=SURFACE,
            linewidth=1.2, zorder=3,
        )  # fmt: skip
        for _, row in sites.iterrows():
            ax_survey.annotate(
                str(int(row["site"])), (row["measured"], 0), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8, color=INK_2,
            )  # fmt: skip
        ax_survey.set_xlim(low - 0.5, high + 0.5)
        ax_survey.set_ylim(-1, 1)
        ax_survey.set_yticks([])
        ax_survey.set_xlabel("measured height (m)", color=INK_2)
        ax_survey.text(
            0.02, 0.95, f"mean {mean:.2f} m   CV {stats['height_cv']:.3f}",
            transform=ax_survey.transAxes, va="top", fontsize=9, color=INK,
        )  # fmt: skip
        _title(ax_survey, "Survey: measured heights (band: ±1 SD)")
    fig.tight_layout()
    plt.show()
