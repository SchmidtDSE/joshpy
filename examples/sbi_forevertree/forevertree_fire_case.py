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

The pseudo-observed data are a **survey**: stage counts at fixed sites in the
survey year. Adults are always found; each juvenile is found with chance
``juvenileDetection`` (90%), drawn by the model (``detectedJuveniles``).

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

TRUTH = {"recruitment": 50.0, "droughtMortality": 10.0}
PRIOR = {
    "recruitment": {"uniform": [10.0, 90.0]},
    "droughtMortality": {"uniform": [0.0, 40.0]},
}
TRUTH_SEED = 3  # names the pseudo-observed run (batchRemote takes no seed)
N_PRIOR = 200

# The run (forevertree_fire.jshc.j2): spin-up, present, projection.
SPINUP_YEARS, PRESENT_YEARS, PROJECTION_YEARS = 200, 10, 40
SURVEY_STEP = SPINUP_YEARS + PRESENT_YEARS - 1  # the present's last year
FINAL_STEP = SURVEY_STEP + PROJECTION_YEARS
WINDOW = range(FINAL_STEP - 9, FINAL_STEP + 1)  # the last 10 projected years
FUTURES = ["normal", "hot_dry", "cool_wet"]

# The hypothesis: at least TARGET_RATIO juveniles per adult over the window,
# a management target fixed before the analysis.
TARGET_RATIO = 0.30

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
# What ABC compares. The survey's log ratio barely moves with either parameter
# (see the fire-model page), so the fit compares how full the stand is and how
# far east it reaches.
STATISTICS = ["adult_density", "density_gradient"]


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


def observe(
    registry: RunRegistry, run_hash: str, replicate: int = 0, *, detection: bool = True
) -> pd.DataFrame:
    """The survey of one run: each site's adults, and its juveniles found and present.

    With ``detection=False``, the survey finds every juvenile (model noise only).
    """
    cells = survey_year(registry, run_hash, replicate).set_index(["x", "y"])
    surveyed = cells.loc[SURVEY_SITES].reset_index()
    if not detection:
        surveyed["detected"] = surveyed["juveniles"]
    return surveyed


def statistics(survey: pd.DataFrame) -> dict[str, float]:
    """Candidate summary statistics of a survey's counts: adults, and juveniles found.

    Counts are pooled over sites, with half a tree added to each pooled count so
    that an empty survey still has a finite log. The gradients compare the four
    dry sites (x > 3, under 450 mm of rain) with the four wet ones (x < 2, about
    500 mm or more).
    """
    wet, dry = survey[survey["x"] < 2], survey[survey["x"] > 3]

    def log_ratio(sites: pd.DataFrame) -> float:
        return float(np.log((sites["detected"].sum() + 0.5) / (sites["adults"].sum() + 0.5)))

    def log_adults(sites: pd.DataFrame) -> float:
        return float(np.log(sites["adults"].sum() + 0.5))

    return {
        "log_ratio": log_ratio(survey),
        "adult_density": float(survey["adults"].mean()),
        "occupied": float((survey["adults"] > 0).mean()),
        "ratio_gradient": log_ratio(dry) - log_ratio(wet),
        "density_gradient": log_adults(dry) - log_adults(wet),
    }


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
