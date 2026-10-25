"""ForeverTree known-truth case study, shared by the Inference docs and run_abc.py.

Put this directory on ``sys.path`` and import it::

    sys.path.insert(0, "examples/sbi_forevertree")
    import forevertree_case as case
    from joshpy.inference import KnownTruthStudy

    registry = RunRegistry("forevertree.duckdb")
    study = KnownTruthStudy(**case.STUDY).load(registry, JoshCLI(josh_jar=JarMode.DEV))

joshpy's :class:`~joshpy.inference.KnownTruthStudy` runs the workflow: the
pseudo-observed run, the prior simulations, z-scores and searches, reusing runs
the registry already holds. This module holds only what is specific to
ForeverTree: the model, its prior and truth, the survey and the candidate
statistics, gathered in ``STUDY``. The docs draw their figures inline.

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

import ast
import inspect
from pathlib import Path
from typing import Any

import pandas as pd

from joshpy.cell_data import DiagnosticQueries
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
MEASUREMENT_SD = 0.25  # m; drawn in forevertree.josh (measuredHeight)
STATISTICS = ["mean_height", "height_cv"]  # what ABC compares


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


# --- The study: KnownTruthStudy(**STUDY) -----------------------------------------------

STUDY = dict(
    name="forevertree",  # runs are recorded as forevertree_truth, forevertree_prior, ...
    model=JobConfig(
        template_path=HERE / "forevertree.jshc.j2",
        config_name="forevertree",  # the model reads `config forevertree.*`
        source_path=HERE / "forevertree.josh",
        file_mappings={
            "temperature": DATA / "temperature.jshd",
            "precipitation": DATA / "precipitation.jshd",
        },
    ),
    prior=PRIOR,
    truth=TRUTH,
    observe=observe,
    statistics=statistics,
    compare=STATISTICS,
    truth_seed=TRUTH_SEED,
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
