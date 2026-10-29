"""The fire model's known-truth case study, shared by the Inference pages on replacement.

Put this directory on ``sys.path`` and import it::

    sys.path.insert(0, "examples/sbi_forevertree")
    import forevertree_fire_case as case
    from joshpy.inference import KnownTruthStudy

    registry = RunRegistry("forevertree_fire.duckdb")
    study = KnownTruthStudy(**case.STUDY).load(registry, JoshCLI(josh_jar=JarMode.DEV))

As in ``forevertree_case.py``, joshpy's :class:`~joshpy.inference.KnownTruthStudy`
runs the workflow, and this module holds only what is specific to the model
(forevertree_fire.josh): its prior and truth, the survey, the candidate
statistics, and the projections the hypotheses are about.

Two kinds of run share one model:

- **Fit runs** stop at the survey (``projectionYears = 0``) and need no future
  climate. The truth, the prior simulations and the SMC-ABC search are fit runs.
- **Projections** run 40 more years under one future (``normal``, ``hot_dry`` or
  ``cool_wet``): :func:`projection_model`. Each is a separate run, so its present
  is a fresh realization of its parameters, not a fit run's.

The pseudo-observed data are a **survey of marked plots**: 10 fixed sites,
revisited every ``VISIT_EVERY`` years over the 20-year present. Each visit counts
the adults and juveniles, marks every plant, and finds which of the plants marked
at the last visit survive (by their stage when marked) and which plants are new.
Climate and fires are recorded every year. Adults are always counted; each
juvenile is counted with chance ``juvenileDetection`` (90%), drawn by the model
(``detectedJuveniles``). Marked plants are always refound. The model exports
every present year, so other schedules can be read from the same runs.

Every run writes to ``minio://`` (``exportPrefix`` below), so the same study runs
locally, given MinIO keys in the environment, or on a batch target: set
``JOSH_BATCH_TARGET`` (for example ``gke-test``). Josh's ``batchRemote`` takes no
seed, so on a target the seeds only name the runs, and the registry reproduces
them. After changing this module or the model, re-render the replacement pages
in order (``pixi run docs-inference``).
"""

from __future__ import annotations

import ast
import inspect
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joshpy.cell_data import DiagnosticQueries
from joshpy.jobs import JobConfig
from joshpy.registry import RunRegistry

HERE = Path(__file__).resolve().parent
CLIMATE = HERE / "climate"

TRUTH = {"recruitment": 50.0, "juvenileMortality": 5.0}
PRIOR = {
    "recruitment": {"uniform": [10.0, 90.0]},
    "juvenileMortality": {"uniform": [1.0, 20.0]},
}
TRUTH_SEED = 3  # names the pseudo-observed run (batchRemote takes no seed)
N_PRIOR = 200

# The run (forevertree_fire.jshc.j2): spin-up, present, projection.
SPINUP_YEARS, PRESENT_YEARS, PROJECTION_YEARS = 200, 20, 40
SURVEY_STEP = SPINUP_YEARS + PRESENT_YEARS - 1  # the present's last year
FINAL_STEP = SURVEY_STEP + PROJECTION_YEARS
WINDOW = range(FINAL_STEP - 9, FINAL_STEP + 1)  # the last 10 projected years
FUTURES = ["normal", "hot_dry", "cool_wet"]

# The hypothesis: the adults' yearly growth factor over the projection is at
# least TARGET_LAMBDA, so the stand keeps at least 0.99^40 = 67% of its adults:
# a management target fixed before the analysis. (A fixed juvenile:adult ratio
# target would reward poor juvenile survival: a settled stand's ratio rises as
# its juveniles die faster.)
TARGET_LAMBDA = 0.99

# Fixed survey sites, like monitoring plots: grid positions (x, y) of 10 of the
# 63 patches, two in each of the five western columns, where the trees are.
# x runs west to east (wet to dry), y north to south.
SURVEY_SITES = [
    (0.5, 1.5),
    (0.5, 5.5),
    (1.5, 0.5),
    (1.5, 3.5),
    (2.5, 2.5),
    (2.5, 6.5),
    (3.5, 1.5),
    (3.5, 4.5),
    (4.5, 3.5),
    (4.5, 5.5),
]
DETECTION = 0.9  # juvenileDetection in forevertree_fire.jshc.j2
VISIT_EVERY = 3  # years between visits to the marked plots: steps 201, 204, ..., 219
MAX_TREES, MATURITY_AGE = 40, 8  # maxTrees and maturityAge in forevertree_fire.jshc.j2
# What ABC compares: one statistic per parameter, each built from the model's
# equations, and the stage ratio, which in a settled stand also depends on
# juvenile survival (see statistics() and the fire-model page).
STATISTICS = ["r_hat", "jm_hat", "log_ratio"]


# --- Observation: raw output -> survey -> statistics ---------------------------------


def survey_year(registry: RunRegistry, run_hash: str, replicate: int = 0) -> pd.DataFrame:
    """Every patch in the survey year: x, y, precipitation, adults, juveniles and detected."""
    cells = DiagnosticQueries(registry).get_all_variables_at_step(SURVEY_STEP, run_hash, replicate)
    if cells.empty:
        raise ValueError(f"No survey-year cells for run {run_hash}, replicate {replicate}")
    cells = cells.rename(
        columns={
            "position_x": "x",
            "position_y": "y",
            "nAdults": "adults",
            "nJuveniles": "juveniles",
            "detectedJuveniles": "detected",
        }
    )
    columns = ["x", "y", "precipitation", "adults", "juveniles", "detected"]
    return cells[columns].sort_values(["y", "x"], ignore_index=True)


def visits(every: int = VISIT_EVERY, years: int = PRESENT_YEARS) -> list[int]:
    """Steps of the visits, every ``every`` years back from the survey year, within ``years``."""
    if not 1 <= every <= 5:
        raise ValueError(f"Revisits every 1 to 5 years are supported, not {every}")
    first = SURVEY_STEP - years + 1
    return sorted(range(SURVEY_STEP, first - 1, -every))


_PRESENTS: dict[tuple[int, str, int], pd.DataFrame] = {}


def present(registry: RunRegistry, run_hash: str, replicate: int = 0) -> pd.DataFrame:
    """The surveyed sites in every present year: counts, age structure, rain and fire.

    Read once per run and replicate, so that surveying one run on several
    schedules queries the registry once.
    """
    key = (id(registry), run_hash, replicate)
    if key in _PRESENTS:
        return _PRESENTS[key]
    cells = registry.conn.execute(
        """
        SELECT * FROM cell_data_current
        WHERE run_hash = ? AND replicate = ? AND step BETWEEN ? AND ?
        """,
        [run_hash, replicate, SURVEY_STEP - PRESENT_YEARS + 1, SURVEY_STEP],
    ).df()
    if cells.empty:
        raise ValueError(f"No present years for run {run_hash}, replicate {replicate}")
    cells = cells.rename(
        columns={
            "position_x": "x",
            "position_y": "y",
            "nAdults": "adults",
            "nJuveniles": "juveniles",
            "detectedJuveniles": "detected",
        }
    )
    cells = cells.set_index(["x", "y"]).loc[SURVEY_SITES].reset_index()
    cells["dryness"] = np.clip((500 - cells["precipitation"]) / 200, 0, 1)  # as the model maps it
    cells["burned"] = cells["burned"].astype(str).str.lower().eq("true")
    _PRESENTS[key] = cells = cells.sort_values(["step", "y", "x"], ignore_index=True)
    return cells


def observe(
    registry: RunRegistry,
    run_hash: str,
    replicate: int = 0,
    *,
    detection: bool = True,
    every: int = VISIT_EVERY,
    years: int = PRESENT_YEARS,
) -> pd.DataFrame:
    """The marked-plot survey of one run: one row per site and visit.

    Every visit counts adults and juveniles (``detected``: the juveniles found).
    From the second visit on, a row also records the interval since the last
    visit, k years: the plants marked then (``marked_juveniles``,
    ``marked_adults``), those still alive (``surviving_juveniles``,
    ``surviving_adults``, by their stage when marked), the plants new since
    (``new``), the open space at the last visit (1 - plants / maxTrees), the
    interval's wet years (the sum of 1 - dryness) and whether a fire burned.

    The fates come from the age structure the model exports: a tree is age 1 the
    year it is first counted, so the trees aged k + 1 or more were alive at the
    last visit, and those aged k + maturityAge or more were adults then.

    With ``detection=False``, the survey finds every juvenile (model noise only).
    """
    cells = present(registry, run_hash, replicate)
    steps = visits(every, years)
    at = cells.set_index(["step", "x", "y"])
    rows = []
    for i, step in enumerate(steps):
        now = at.loc[step]
        row = pd.DataFrame(
            {
                "step": step,
                "adults": now["adults"],
                "juveniles": now["juveniles"],
                "detected": now["detected"] if detection else now["juveniles"],
            }
        )
        if i > 0:
            last, k = at.loc[steps[i - 1]], step - steps[i - 1]
            between = cells[cells["step"].between(steps[i - 1], step - 1)].groupby(["x", "y"])
            plants = now["adults"] + now["juveniles"]
            row["marked_juveniles"] = last["juveniles"]
            row["marked_adults"] = last["adults"]
            row["surviving_juveniles"] = now[f"below{k + MATURITY_AGE}"] - now[f"below{k + 1}"]
            row["surviving_adults"] = plants - now[f"below{k + MATURITY_AGE}"]
            row["new"] = now[f"below{k + 1}"]
            row["open_space"] = (1 - (last["adults"] + last["juveniles"]) / MAX_TREES).clip(lower=0)
            row["wet_years"] = k - between["dryness"].sum()  # the sum of 1 - dryness
            row["burned"] = between["burned"].any()
        rows.append(row.reset_index())
    return pd.concat(rows, ignore_index=True)


def statistics(survey: pd.DataFrame) -> dict[str, float]:
    """Candidate summary statistics of a marked-plot survey.

    From the fates of marked plants, one statistic per parameter, each following
    the model's equations:

    - ``r_hat``: new plants per marked adult, per wet year, per unit of open
      space, in %. Recruits ~ Binomial(adults, recruitment x (1 - dryness) x open
      space), so this tracks ``recruitment`` (below it: recruits that die before
      the next visit are never seen).
    - ``jm_hat``: the yearly death rate of marked juveniles, over intervals
      without fire, in %. A juvenile dies with chance juvenileMortality + dryness x
      droughtMortality, so this tracks ``juvenileMortality``.

    From the counts alone, as a survey without marked plants would have, each
    pooled over the visits: ``log_ratio`` (juveniles found per adult),
    ``adult_density`` (adults per site) and ``density_gradient`` (log adults at
    the four dry sites minus the four wet ones). Pooled counts get half a plant,
    so that an empty survey still has a finite log, and the rates get a
    pseudo-observation.
    """
    intervals = survey.dropna(subset=["new"])
    wet, dry = survey[survey["x"] < 2], survey[survey["x"] > 3]
    out = {
        "log_ratio": float(
            np.log((survey["detected"].sum() + 0.5) / (survey["adults"].sum() + 0.5))
        ),
        "adult_density": float(survey["adults"].mean()),
        "density_gradient": float(
            np.log(dry["adults"].sum() + 0.5) - np.log(wet["adults"].sum() + 0.5)
        ),
        "r_hat": np.nan,
        "jm_hat": np.nan,
    }
    if not intervals.empty:
        k = int(np.diff(np.unique(survey["step"]))[0])
        exposure = intervals["marked_adults"] * intervals["open_space"] * intervals["wet_years"]
        unburned = intervals[~intervals["burned"].astype(bool)]
        survival = (unburned["surviving_juveniles"].sum() + 0.9**k) / (
            unburned["marked_juveniles"].sum() + 1
        )
        out["r_hat"] = float(100 * (intervals["new"].sum() + 0.5) / (exposure.sum() + 1))
        out["jm_hat"] = float(100 * (1 - min(survival, 1.0) ** (1 / k)))
    return out


# --- Projections: the future the hypotheses are about ----------------------------------


def projection_model(future: str) -> JobConfig:
    """The model run 40 years past the survey under one future, writing those years."""
    if future not in FUTURES:
        raise ValueError(f"Unknown future {future!r}; use one of {FUTURES}")
    return replace(
        MODEL,
        template_vars={"projectionYears": PROJECTION_YEARS},
        file_mappings={
            **MODEL.file_mappings,
            "futureTemperature": CLIMATE / f"{future}_temperature.jshd",
            "futurePrecipitation": CLIMATE / f"{future}_precipitation.jshd",
        },
        output_steps=f"{SURVEY_STEP}-{FINAL_STEP}",
    )


def landscape(registry: RunRegistry, run_hashes: Any) -> pd.DataFrame:
    """Juveniles and adults over the whole landscape, by run, replicate and step.

    Columns run_hash, replicate, step, years (since the survey), adults,
    juveniles and ratio (juveniles per adult; NaN without adults).
    """
    totals = registry.conn.execute(
        """
        SELECT run_hash, replicate, step,
               SUM(nAdults) AS adults, SUM(nJuveniles) AS juveniles
        FROM cell_data_current
        WHERE run_hash IN (SELECT unnest(?))
        GROUP BY ALL
        ORDER BY run_hash, replicate, step
        """,
        [list(run_hashes)],
    ).df()
    totals["years"] = totals["step"] - SURVEY_STEP
    totals["ratio"] = totals["juveniles"] / totals["adults"].where(totals["adults"] > 0)
    return totals


_LANDSCAPES: dict[tuple[int, str], pd.DataFrame] = {}


def _run_landscape(registry: RunRegistry, run_hash: str, replicate: int) -> pd.DataFrame:
    """One run's landscape, read once for all its replicates (a filter measures each)."""
    key = (id(registry), run_hash)
    years = _LANDSCAPES.get(key)
    if years is None or replicate not in set(years["replicate"]):  # new, or topped up since
        years = _LANDSCAPES[key] = landscape(registry, [run_hash])
    return years


def projected(registry: RunRegistry, run_hash: str, replicate: int) -> dict[str, float]:
    """One projection's measurements: the ratio over the window, and the adults' growth rate.

    - ``ratio``: the landscape's juveniles per adult, averaged over the last 10
      projected years (NaN if the adults died out);
    - ``adult_lambda``: the adults' yearly growth factor from the survey to the
      last projected year, (A_final / A_survey)^(1/40).
    """
    years = _run_landscape(registry, run_hash, replicate)
    years = years[years["replicate"] == replicate].set_index("step")
    if FINAL_STEP not in years.index:
        raise ValueError(f"Run {run_hash}, replicate {replicate} is not a projection")
    adults = years["adults"]
    return {
        "ratio": float(years.loc[list(WINDOW), "ratio"].mean(skipna=False)),
        "adult_lambda": float(
            (adults[FINAL_STEP] / adults[SURVEY_STEP]) ** (1 / PROJECTION_YEARS)
            if adults[SURVEY_STEP] > 0
            else np.nan
        ),
    }


# --- The study: KnownTruthStudy(**STUDY) -----------------------------------------------

MODEL = JobConfig(
    template_path=HERE / "forevertree_fire.jshc.j2",
    config_name="forevertreeFire",  # the model reads `config forevertreeFire.*`
    source_path=HERE / "forevertree_fire.josh",
    file_mappings={
        "historicalTemperature": CLIMATE / "historical_temperature.jshd",
        "historicalPrecipitation": CLIMATE / "historical_precipitation.jshd",
    },
    template_vars={"projectionYears": 0},  # fit runs stop at the survey
    custom_tags={"exportPrefix": "forevertree-fire"},  # where in the bucket runs write
    output_steps=f"{SURVEY_STEP - PRESENT_YEARS + 1}-{SURVEY_STEP}",  # the present
)

STUDY = dict(
    name="forevertree_fire",  # runs are recorded as forevertree_fire_truth, ...
    model=MODEL,
    prior=PRIOR,
    truth=TRUTH,
    observe=observe,
    statistics=statistics,
    compare=STATISTICS,
    truth_seed=TRUTH_SEED,
    target=os.environ.get("JOSH_BATCH_TARGET") or None,  # None: run locally
)


def show_source(*objects: Any) -> None:
    """Print objects' source as a fenced Python block (docs: use with output: asis).

    A string names an assignment in this module, such as ``"STUDY"``.
    """
    module = Path(__file__).read_text(encoding="utf-8")
    parts = []
    for obj in objects:
        if isinstance(obj, str):
            node = next(
                node
                for node in ast.parse(module).body
                if isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == obj for target in node.targets)
            )
            parts.append(ast.get_source_segment(module, node))
        else:
            parts.append(inspect.getsource(obj).rstrip())
    # A four-backtick fence, so source that contains ``` can't close it early.
    print("````python\n" + "\n\n".join(parts) + "\n````")
