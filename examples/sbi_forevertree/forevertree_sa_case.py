"""The one-at-a-time sensitivity study of the fire model, for the local-sensitivity page.

Put this directory on ``sys.path`` and import it::

    sys.path.insert(0, "examples/sbi_forevertree")
    import forevertree_sa_case as sa

    registry = RunRegistry("forevertree_sa.duckdb")
    cli = JoshCLI(josh_jar=JarMode.DEV)
    hashes = sa.run_settings(registry, cli, [{}], "forevertree_sa_baseline")  # the baseline

The model (forevertree_sa.josh) is the fire model with knobs: initial conditions, the
projection's climate, a fire pulse and three management interventions. At ``BASELINE`` it is the
fire-model case study's configuration at its truth, a direct parameterization. Each setting
perturbs one input around it.

**Replicates.** Each setting runs ``REPLICATES`` independent, unseeded replicates, and settings
are compared by their means. Josh can't yet give two settings the same random numbers (fires in
the same years, the same weather), so every difference carries both settings' replicate noise.
Each estimate comes with a Welch standard error, and an effect within about two of them can't be
told from zero. The standard error falls as 1/sqrt(REPLICATES).

Every run writes to ``minio://`` (``exportPrefix``), as the fire model's do: locally given MinIO
keys in the environment, or on a batch target named by ``JOSH_BATCH_TARGET``.
"""

from __future__ import annotations

import ast
import inspect
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joshpy.cli import JoshCLI
from joshpy.jobs import JobConfig, SweepConfig
from joshpy.registry import RunRegistry
from joshpy.strategies import ParameterSetsStrategy
from joshpy.sweep import SweepManager

HERE = Path(__file__).resolve().parent
CLIMATE = HERE / "climate"

# Independent replicates per setting. Unseeded, as inference runs are (SBI_IMPLEMENTATION.md):
# a seed would force serial patches and shares nothing between settings.
REPLICATES = 30
TARGET = os.environ.get("JOSH_BATCH_TARGET") or None  # None: run locally

SPINUP_YEARS, PRESENT_YEARS, PROJECTION_YEARS = 200, 20, 40
FIRST_PROJECTED_STEP = SPINUP_YEARS + PRESENT_YEARS  # year 1 of the projection
TARGET_LAMBDA = 0.99  # the fire pages' management target: adults' yearly growth factor
TARGET_RETAINED = TARGET_LAMBDA**PROJECTION_YEARS  # = 67% of the adults after 40 years

# --- The baseline: a direct parameterization ------------------------------------------

BASELINE: dict[str, Any] = {
    # The run
    "spinupYears": SPINUP_YEARS,
    "presentYears": PRESENT_YEARS,
    "projectionYears": PROJECTION_YEARS,
    "historicalYears": 30,
    "futureYears": 40,
    # The projection's climate (the normal future)
    "rainScale": 100.0,
    "temperatureOffset": 0.0,
    "resampleWeather": 1,
    # Growth
    "maxNewGrowth": 1.0,
    "minPrecipImpactPct": 40.0,
    # Demography (recruitment and juvenileMortality at the fire pages' truth)
    "initialTrees": 10,
    "maxTrees": 40,
    "maturityAge": 8,
    "recruitment": 50.0,
    "adultMortality": 2.0,
    "juvenileMortality": 5.0,
    "droughtMortality": 10.0,
    # Fuel and fire (stochastic fire only)
    "initialFuel": 50.0,
    "fuelGrowth": 40.0,
    "fireInterval": 0,
    "firstFire": 10,
    "fireChance": 10.0,
    "adultFireMortality": 20.0,
    # Management (off)
    "fuelTreatmentInterval": 0,
    "fuelReduction": 0.0,
    "plantingDensity": 0,
    "juvenileProtection": 0.0,
}

MODEL = JobConfig(
    template_path=HERE / "forevertree_sa.jshc.j2",
    config_name="forevertreeSa",  # the model reads `config forevertreeSa.*`
    source_path=HERE / "forevertree_sa.josh",
    file_mappings={
        "historicalTemperature": CLIMATE / "historical_temperature.jshd",
        "historicalPrecipitation": CLIMATE / "historical_precipitation.jshd",
        "futureTemperature": CLIMATE / "normal_temperature.jshd",
        "futurePrecipitation": CLIMATE / "normal_precipitation.jshd",
    },
    template_vars=dict(BASELINE),  # every variable has a value, so none renders empty
    custom_tags={"exportPrefix": "forevertree-sa"},
    replicates=REPLICATES,
    output_steps=f"{FIRST_PROJECTED_STEP - 1}-{FIRST_PROJECTED_STEP + PROJECTION_YEARS - 1}",
)


# --- The inputs, each perturbed by an illustrative uncertainty ------------------------

# The model is invented, so no input's uncertainty is known. Each input gets one of three
# illustrative spreads, one standard deviation as a share of its baseline. The spreads are mixed
# within each group, so that the page can show how sensitivity and uncertainty combine: a
# sensitive input known well and an insensitive one known poorly can contribute alike
# (Malchow & Hartig 2026, Fig. 2).
SPREADS = {"narrow": 0.10, "medium": 0.25, "wide": 0.50}


@dataclass(frozen=True)
class Factor:
    """One input of the one-at-a-time design.

    Attributes:
        name: The template variable.
        group: Initial conditions, climate, life history or fire (Dietze's terms).
        label: How the page names it.
        unit: Its unit, as the template writes it.
        spread: ``narrow``, ``medium`` or ``wide`` (:data:`SPREADS`), or None with ``step``.
        step: An absolute SD, for an input whose baseline is 0 (a share of it is no step).
    """

    name: str
    group: str
    label: str
    unit: str
    spread: str | None
    step: float | None = None

    @property
    def baseline(self) -> Any:
        return BASELINE[self.name]

    @property
    def sd(self) -> Any:
        """One standard deviation of the illustrative uncertainty: the step each way.

        A share of the baseline, rounded to a whole step (at least 1) for a count or an age.
        """
        if self.step is not None:
            return self.step
        sd = SPREADS[self.spread] * self.baseline
        return max(1, round(sd)) if isinstance(self.baseline, int) else float(sd)

    def at(self, direction: int) -> Any:
        """The input's value one SD below (-1) or above (+1) the baseline, in its own type."""
        value = self.baseline + direction * self.sd
        return int(round(value)) if isinstance(self.baseline, int) else float(value)


FACTORS = [
    Factor("initialTrees", "initial conditions", "seedlings per patch at the start", "count",
           "wide"),
    Factor("initialFuel", "initial conditions", "fuel cover at the start", "%", "wide"),
    Factor("rainScale", "climate", "projected rain", "% of the future's", "narrow"),
    Factor("temperatureOffset", "climate", "projected temperature", "K", None, step=1.0),
    Factor("recruitment", "life history", "recruitment", "%", "medium"),
    Factor("juvenileMortality", "life history", "juvenile mortality", "%", "wide"),
    Factor("adultMortality", "life history", "adult mortality", "%", "narrow"),
    Factor("droughtMortality", "life history", "added juvenile mortality in drought", "%", "wide"),
    Factor("maturityAge", "life history", "age at maturity", "years", "narrow"),
    Factor("maxTrees", "life history", "trees a patch holds", "count", "medium"),
    Factor("fireChance", "fire", "yearly chance of fire at full fuel", "%", "wide"),
    Factor("adultFireMortality", "fire", "adults a full fire kills", "%", "medium"),
    Factor("fuelGrowth", "fire", "fuel regrowth in a wet year", "%", "narrow"),
]
FACTOR = {factor.name: factor for factor in FACTORS}

# Settings beyond the ±1 SD steps.
SPINUPS = [50, 100, 400]  # with the baseline's 200: is the spin-up long enough?
PULSE = {"fireInterval": 10_000, "firstFire": FIRST_PROJECTED_STEP}  # one fire, every patch, year 1
PULSE_FACTORS = [
    "recruitment", "juvenileMortality", "adultMortality", "maturityAge", "adultFireMortality",
    "fuelGrowth", "rainScale",
]
INTERVENTIONS = {
    "fuel treatment": [
        {"fuelTreatmentInterval": 5, "fuelReduction": 25.0},
        {"fuelTreatmentInterval": 5, "fuelReduction": 50.0},
    ],
    "post-fire planting": [{"plantingDensity": 5}, {"plantingDensity": 10}],
    "juvenile protection": [{"juvenileProtection": 25.0}, {"juvenileProtection": 50.0}],
}

# Representative sites: three columns of 7 patches along the gradient, wet in the west to the
# stand's dry margin (x is the patch's column; rain falls from about 550 mm to 400 mm across them).
SITES = {"wet": 0.5, "mid": 2.5, "margin": 4.5}
SITE_ORDER = [*SITES, "landscape"]


def design() -> pd.DataFrame:
    """Every setting of the study, one row each.

    Columns: ``label``; ``branch`` (``ic``: initial conditions and spin-up, read from step 0;
    ``oat``: the other inputs ±1 SD; ``weather``; ``pulse``; ``intervention``); ``factor``;
    ``direction`` (-1, 0 or +1; the dose for interventions); ``pulse`` (bool); and
    ``overrides``, the template variables the setting changes from ``BASELINE``.
    """
    rows: list[dict[str, Any]] = []

    def add(label: str, branch: str, overrides: dict[str, Any], factor: str = "baseline",
            direction: int = 0, pulse: bool = False) -> None:
        rows.append({"label": label, "branch": branch, "factor": factor, "direction": direction,
                     "pulse": pulse, "overrides": overrides})

    add("baseline", "ic", {})
    for factor in FACTORS:
        branch = "ic" if factor.group == "initial conditions" else "oat"
        for direction, sign in [(-1, "-"), (1, "+")]:
            add(f"{factor.name}{sign}", branch, {factor.name: factor.at(direction)},
                factor.name, direction)
    for years in SPINUPS:
        add(f"spinupYears={years}", "ic", {"spinupYears": years}, "spinupYears")
    add("fixed weather", "weather", {"resampleWeather": 0}, "resampleWeather")
    add("pulse", "pulse", dict(PULSE), pulse=True)
    for name in PULSE_FACTORS:
        factor = FACTOR[name]
        for direction, sign in [(-1, "-"), (1, "+")]:
            add(f"pulse {name}{sign}", "pulse", {**PULSE, name: factor.at(direction)},
                name, direction, pulse=True)
    for intervention, doses in INTERVENTIONS.items():
        for dose, overrides in enumerate(doses, start=1):
            for pulse in (False, True):
                prefix = "pulse " if pulse else ""
                add(f"{prefix}{intervention} {dose}", "intervention",
                    {**(PULSE if pulse else {}), **overrides}, intervention, dose, pulse)
    return pd.DataFrame(rows)


def settings(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """A setting's full template variables: the baseline with its overrides."""
    unknown = set(overrides) - set(BASELINE)
    if unknown:
        raise ValueError(f"Not a template variable of forevertree_sa.jshc.j2: {sorted(unknown)}")
    return {**BASELINE, **overrides}


# --- Running -------------------------------------------------------------------------


def _key(parameters: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(parameters[name] for name in BASELINE)


def run_settings(
    registry: RunRegistry,
    cli: JoshCLI,
    overrides: Sequence[Mapping[str, Any]],
    experiment: str,
    *,
    output_steps: str | None = None,
    replicates: int = REPLICATES,
) -> list[str]:
    """Run each setting, as one batch, and return their run_hashes in order.

    Each row is the baseline with its overrides, so the registry records every variable of every
    run. Runs the registry already holds with every replicate are reused (a re-render replays),
    a run with fewer is topped up, and the rest run together: on ``TARGET``, dispatched together
    and polled. This is :meth:`joshpy.inference.KnownTruthStudy.run_at_each` for any variables.

    Args:
        registry: Where the runs are recorded and looked up.
        cli: How the model runs.
        overrides: One mapping per setting, of the variables it changes from ``BASELINE``.
        experiment: Experiment name of the session.
        output_steps: Steps loaded into the registry (default: year 0 to the end of the
            projection). Not part of ``run_hash``: a reused run keeps the steps it was first
            loaded with, so a setting read from step 0 must first run with every step.
        replicates: Replicates per setting.

    Returns:
        One run_hash per setting (identical settings share one).
    """
    rows = [settings(row) for row in overrides]
    unique = list({_key(row): row for row in rows}.values())
    config = replace(
        MODEL,
        sweep=SweepConfig(strategy=ParameterSetsStrategy(sets=unique)),
        replicates=replicates,
        output_steps=output_steps or MODEL.output_steps,
    )
    builder = (
        SweepManager.builder(config).with_registry(registry, experiment_name=experiment).with_cli(cli)
    )
    if TARGET is not None:
        builder = builder.with_batch_remote(TARGET, no_wait=True)
    manager = builder.build()
    try:
        # "pool" tops a run up to its replicate count instead of re-running it.
        result = manager.run(quiet=True, collision_policy="pool")
    finally:
        manager.cleanup()
    by_key = {_key(run.parameters): run.run_hash for run, _ in result.job_results}
    return [by_key[_key(row)] for row in rows]


def last_step(overrides: Mapping[str, Any]) -> int:
    """The last step of a setting's run."""
    run = settings(overrides)
    return run["spinupYears"] + run["presentYears"] + run["projectionYears"] - 1


def all_steps(overrides: Sequence[Mapping[str, Any]]) -> str:
    """Every step of the longest of these settings' runs, for reading them from step 0."""
    return f"0-{max(last_step(row) for row in overrides)}"


# --- Reading the runs ----------------------------------------------------------------


def trajectories(
    registry: RunRegistry, run_hashes: Sequence[str], *, from_start: bool = False
) -> pd.DataFrame:
    """Each run's sites by replicate and year: adults, juveniles, new trees, new adults, fire, fuel.

    Sums each site's 7 patches, and every patch for ``landscape``. ``years`` counts from the
    projection's start (0: the last present year; negative: the spin-up and present), and
    ``burned`` is the share of the site's patches that burned. Years 0 to 40 only, unless
    ``from_start``.
    """
    window = "" if from_start else f"AND yearsFromSurvey BETWEEN 0 AND {PROJECTION_YEARS}"
    cells = registry.conn.execute(
        f"""
        SELECT run_hash, replicate, position_x AS x,
               CAST(yearsFromSurvey AS INTEGER) AS years,
               SUM(nAdults) AS adults, SUM(nJuveniles) AS juveniles,
               SUM(newTrees) AS new_trees, SUM(newAdults) AS new_adults,
               AVG(CASE WHEN lower(CAST(burned AS VARCHAR)) IN ('true', '1', '1.0')
                        THEN 1.0 ELSE 0.0 END) AS burned,
               AVG(fuel) AS fuel, COUNT(*) AS patches
        FROM cell_data_current
        WHERE run_hash IN (SELECT unnest(?)) {window}
        GROUP BY ALL
        """,
        [list(dict.fromkeys(run_hashes))],
    ).df()
    if cells.empty:
        raise ValueError("No cells for these runs: are they loaded?")
    sites = cells[cells["x"].isin(SITES.values())].copy()
    sites["site"] = sites["x"].map({x: name for name, x in SITES.items()})
    sums = ["adults", "juveniles", "new_trees", "new_adults"]
    keys = ["run_hash", "replicate", "years"]
    landscape = cells.groupby(keys)[[*sums, "patches"]].sum()
    weighted = cells.assign(
        burned=cells["burned"] * cells["patches"], fuel=cells["fuel"] * cells["patches"]
    ).groupby(keys)[["burned", "fuel"]].sum()
    landscape = landscape.join(weighted)
    landscape[["burned", "fuel"]] = landscape[["burned", "fuel"]].div(landscape["patches"], axis=0)
    landscape = landscape.reset_index().assign(site="landscape")
    columns = ["run_hash", "replicate", "site", "years", *sums, "burned", "fuel"]
    out = pd.concat([sites[columns], landscape[columns]], ignore_index=True)
    out["site"] = pd.Categorical(out["site"], SITE_ORDER, ordered=True)
    return out.sort_values(["run_hash", "replicate", "site", "years"], ignore_index=True)


OUTCOMES = {
    "recruits_per_adult": "new plants per adult per year",
    "survival_to_maturity": "survival to maturity",
    "adult_survival": "adult survival per year",
    "adult_lambda": "adults' yearly growth factor λ",
    "retained": "adults retained after 40 years",
}


def chain(years: pd.DataFrame, maturity_age: int) -> dict[str, float]:
    """One replicate's projection at one site, from life history up to the outcome.

    With A(y) adults in year y, new trees T(y) (first counted that year) and new adults N(y):

    - ``recruits_per_adult``: sum T(1..40) / sum A(0..39), new plants per adult per year;
    - ``survival_to_maturity``: new adults over the new trees they came from, maturity - 1
      years earlier (a tree is 1 the year it is first counted);
    - ``adult_survival``: sum (A(y+1) - N(y+1)) / sum A(y), the adults that live a year;
    - ``adult_lambda`` and ``retained``: A(40) / A(0), per year and over the 40 years.

    Since A(y+1) = survivors + N(y+1), λ is about adult survival plus new adults per adult, and
    new adults per adult are new plants per adult times survival to maturity, a few years on.
    """
    y = years.set_index("years").reindex(range(PROJECTION_YEARS + 1))
    adults, new_trees, new_adults = (y[c].to_numpy(dtype=float) for c in
                                     ["adults", "new_trees", "new_adults"])
    exposure = adults[:-1].sum()
    lag = maturity_age - 1
    cohorts = new_trees[1:PROJECTION_YEARS + 1 - lag].sum()
    retained = adults[-1] / adults[0] if adults[0] > 0 else np.nan
    return {
        "adults_at_start": adults[0],
        "recruits_per_adult": new_trees[1:].sum() / exposure if exposure > 0 else np.nan,
        "survival_to_maturity": new_adults[1 + lag:].sum() / cohorts if cohorts > 0 else np.nan,
        "adult_survival": (adults[1:] - new_adults[1:]).sum() / exposure if exposure > 0 else np.nan,
        "adult_lambda": retained ** (1 / PROJECTION_YEARS) if retained == retained else np.nan,
        "retained": retained,
        "burned": y["burned"].iloc[1:].mean(),
        "fuel": y["fuel"].iloc[1:].mean(),
    }


def measure(registry: RunRegistry, runs: pd.DataFrame) -> pd.DataFrame:
    """Every setting's chain, by replicate and site.

    Args:
        registry: Where the runs are.
        runs: One row per setting with ``label``, ``run_hash`` and ``overrides``
            (``design()`` plus run_hashes).

    Returns:
        One row per setting, replicate and site: ``label``, ``replicate``, ``site`` and the
        :func:`chain` measurements.
    """
    years = trajectories(registry, runs["run_hash"])
    maturity = {
        row.run_hash: settings(row.overrides)["maturityAge"] for row in runs.itertuples()
    }
    by_run = {
        key: chain(group, maturity[key[0]])
        for key, group in years.groupby(["run_hash", "replicate", "site"], observed=True)
    }
    measured = pd.DataFrame(
        [{"run_hash": h, "replicate": r, "site": s, **values} for (h, r, s), values in by_run.items()]
    )
    out = runs[["label", "run_hash"]].merge(measured, on="run_hash")
    out["site"] = pd.Categorical(out["site"], SITE_ORDER, ordered=True)
    return out


# --- Comparisons between settings ----------------------------------------------------


def summarize(measured: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Each setting's mean, variance and replicate count of ``outcome``, by site."""
    return (
        measured.dropna(subset=[outcome])
        .groupby(["label", "site"], observed=True)[outcome]
        .agg(mean="mean", var=lambda v: v.var(ddof=1), n="count")
    )


def compare(measured: pd.DataFrame, a: str, b: str, outcome: str) -> pd.DataFrame:
    """Setting ``b`` minus setting ``a`` at each site, from independent replicates.

    Returns one row per site: the difference of the means and its Welch standard error,
    sqrt(var_a / n_a + var_b / n_b).
    """
    stats = summarize(measured[measured["label"].isin([a, b])], outcome)
    sa_, sb = stats.loc[a], stats.loc[b]
    out = pd.DataFrame({
        "difference": sb["mean"] - sa_["mean"],
        "se": np.sqrt(sa_["var"] / sa_["n"] + sb["var"] / sb["n"]),
    })
    out.index.name = "site"
    return out


def oat_table(measured: pd.DataFrame, outcome: str, *, prefix: str = "",
              baseline: str = "baseline", factors: Sequence[str] | None = None) -> pd.DataFrame:
    """Each factor's one-at-a-time effect on ``outcome``, by site.

    For factor x with baseline x0 and uncertainty σ, run at x0 - σ and x0 + σ, with means Y-
    and Y+ over independent replicates, and Y0 at the baseline:

    - ``effect``: (Y+ - Y-) / 2, the change in Y for a 1-SD change in x, a central difference;
      ``se`` is its Welch standard error;
    - ``elasticity``: ((Y+ - Y-) / 2σ) · x0 / Y0, the % change in Y for a 1% change in x, the
      unscaled sensitivity;
    - ``contribution``: effect² - se², Dietze's (∂Y/∂x)² Var[x]. Replicate noise inflates
      effect² by se² on average, so it is subtracted; a negative value means the effect is
      lost in the noise;
    - ``curvature``: (Y+ + Y- - 2 Y0) / |Y+ - Y-|, far from 0 where the response bends or
      crosses a threshold within ±1 SD, so the linear terms don't hold.

    Args:
        measured: :func:`measure`'s output.
        outcome: One of :data:`OUTCOMES`, or another column of ``measured``.
        prefix: Label prefix of the branch (``"pulse "`` for the fire pulse).
        baseline: The branch's baseline label.
        factors: Factor names (default: every factor with both settings in ``measured``).

    Returns:
        One row per factor and site.
    """
    labels = set(measured["label"])
    names = factors or [f.name for f in FACTORS if {f"{prefix}{f.name}-", f"{prefix}{f.name}+"}
                        <= labels]
    stats = summarize(measured, outcome)
    rows = []
    for name in names:
        factor = FACTOR[name]
        minus, plus = f"{prefix}{name}-", f"{prefix}{name}+"
        diff = compare(measured, minus, plus, outcome)
        step = factor.at(1) - factor.at(-1)
        for site, d in diff.iterrows():
            y0 = stats.loc[(baseline, site), "mean"]
            y_minus, y_plus = stats.loc[(minus, site), "mean"], stats.loc[(plus, site), "mean"]
            effect, se = d["difference"] / 2, d["se"] / 2
            rows.append({
                "factor": name,
                "group": factor.group,
                "site": site,
                "baseline": y0,
                "minus": y_minus,
                "plus": y_plus,
                "effect": effect,
                "se": se,
                "elasticity": (d["difference"] / step) * factor.baseline / y0
                if y0 and factor.baseline else np.nan,  # no relative change of a zero offset
                "contribution": effect**2 - se**2,
                "curvature": (y_plus + y_minus - 2 * y0) / abs(y_plus - y_minus)
                if y_plus != y_minus else np.nan,
            })
    out = pd.DataFrame(rows)
    out["site"] = pd.Categorical(out["site"], SITE_ORDER, ordered=True)
    return out


def rescaled(effects: pd.DataFrame, spread: float) -> pd.Series:
    """Each input's contribution, had its SD been ``spread`` of its baseline.

    While the response is straight, an effect grows with the size of the change, so a
    contribution grows with the SD squared: C · (spread · x0 / σ)². With every input given the
    same spread, contributions rank by sensitivity alone. Inputs whose baseline is 0 get NaN.

    Args:
        effects: :func:`oat_table`'s output.
        spread: The SD, as a share of each input's baseline.
    """
    ratio = effects["factor"].map(
        lambda name: spread * FACTOR[name].baseline / FACTOR[name].sd
        if FACTOR[name].baseline else np.nan
    )
    return effects["contribution"] * ratio.astype(float) ** 2


def process_variance(measured: pd.DataFrame, outcome: str, label: str = "baseline") -> pd.Series:
    """The variance of ``outcome`` across a setting's replicates, by site: Dietze's process term."""
    return measured[measured["label"] == label].groupby("site", observed=True)[outcome].var(ddof=1)


def show_source(*objects: Any) -> None:
    """Print objects' source as a fenced Python block (docs: use with output: asis).

    A string names an assignment in this module, such as ``"BASELINE"``.
    """
    module = Path(__file__).read_text(encoding="utf-8")
    parts = []
    for obj in objects:
        if isinstance(obj, str):
            node = next(
                node
                for node in ast.parse(module).body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    getattr(target, "id", None) == obj
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                )
            )
            parts.append(ast.get_source_segment(module, node))
        else:
            parts.append(inspect.getsource(obj).rstrip())
    print("````python\n" + "\n\n".join(parts) + "\n````")
