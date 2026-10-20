# SBI Implementation for joshpy

Oct 17, 2026 · @nick · joshpy 0.0.9.32 · sbi 0.27.0 with CPU-only torch 2.14.1

How joshpy runs the filter-and-refine search in [SBI.md](SBI.md). The machinery is proved on
the ForeverTree demo (`examples/sbi_forevertree/`). It then scales to the grass–shrub and Joshua
tree experiments.

## Principles

- **sbi is the inference engine; joshpy implements no statistics.**
  - sbi owns the proposals, weights and tolerance schedule.
  - joshpy turns each batch of parameter sets into Josh runs, records them, and returns their
    summary statistics.
  - Neural estimators (NPE/NLE/NRE) and SBC come from the same package when needed.
- **The registry is the only store.** Every simulation is a registered run, whether accepted or not.
  Analyses are DuckDB queries over `config_parameters`, run attributes and `cell_data`.
- **Resume lives in the simulation layer.**
  - Runs are content-addressed by `run_hash`.
  - Completed runs are reused, and submitted batch jobs are reattached.
  - A seeded strategy replays its proposals.
  - This works the same for every strategy.

## Built (PR stack #86 → #87 → #88)

| PR | Branch | What it adds |
|----|--------|--------------|
| #86 | `feat/adaptive-evaluate-batch` | `SweepStrategy.run_adaptive(ctx)`: strategies own their loop. `AdaptiveContext.evaluate(rows)` expands, registers, runs (local, remote, batch_remote), loads and scores a batch. Optuna moved onto it. `JobExpander.expand(combinations=...)`. `SweepManager` adopts adaptive runs, so `load_results()`/`ingest()` work. `_create_single_job` deleted. Exports `AdaptiveContext` and `BatchResult`. |
| #87 | `feat/batch-no-wait-resume` | `batch_no_wait` records dispatch as *submitted* and completion on poll, loading as jobs finish. `evaluate` reuses completed runs and reattaches to submitted ones. `OptunaStrategy(seed=...)` replays. |
| #88 | `feat/sbi-strategy` | `SbiStrategy` (`smcabc`, `mcabc`) and `SbiResult`; details below. |

What #88 provides:

- **Configuration:** prior as a dict (uniform/normal), a summaries callable, observed values,
  `method_kwargs` passed through to sbi, and `seed`.
- **Run attributes:** every run gets `sbi_call`; population members get `sbi_accepted`,
  `sbi_generation`, `sbi_weight` and `sbi_distance`.
- **Session metadata** (`sbi`): settings, a replay fingerprint, sbi/torch versions, and every
  population with its `epsilon`, `simulations` and `carried_over`.
- **Warnings:** version drift between replayed sessions, and a shared `JobConfig.seed`.
- **Batch guard:** a multi-job batch is refused if its export path contains neither `{run_hash}`
  nor `{label}`.
- **Catalog:** an adaptive sweep's summary lists the strategy's parameter names with
  `total_combinations: null` (decided at run time). On `main`, the summary was skipped or crashed
  on `len(sweep)`.
- **Exports:** `SbiStrategy` and `SbiResult`.
- **Packaging:** the `joshpy[sbi]` extra (`sbi>=0.27`, currently open-ended); `pixi run
  install-sbi` and `test-sbi`; an `sbi-tests` CI job.
- **Example:** `examples/sbi_forevertree/run_abc.py`.

**Deviations from the original plan:**
- Generation checkpoints are written once, at the end of a run.
- The warm-start fallback is not built; seeded replay is.
- The version guard warns instead of refusing.

## Behavior notes

Contracts the stack implements; the tests named here pin them.

- **Failures.** A failed run gets NaN summaries, which sbi never accepts
  (`test_failed_runs_are_never_accepted`). With `stop_on_failure=True`, `evaluate` raises only after
  the failing run's batch has finished and its successful runs are loaded.
- **Config namespace.** The default differs by path: batch sweeps use `sweep_config`; adaptive
  sweeps use the template's stem. An explicit `JobConfig.config_name` wins on both.
- **Parameter-less adaptive sweeps.** `SweepConfig` presence is checked with `is not None`, not
  truthiness, so a sweep that has only a strategy still runs adaptively
  (`test_job_config_keeps_parameterless_sbi_sweep`). Before this fix, such sweeps ran as empty
  batch sweeps.
- **Resume.** Rerunning an interrupted search with the same seed against the same registry ends
  with identical particles (`test_interrupted_run_replays_identically`).
- **Run attributes are last-writer-wins.** The latest sbi session that touches a `run_hash` sets
  its `sbi_*` attributes. Read populations from the `SbiResult` or the session metadata.

## What we learned

These findings apply to every page and experiment.

- **Seeds.**
  - `SbiStrategy.seed` fixes sbi's proposals, not Josh's noise.
  - Leave `JobConfig.seed` unset for ABC runs, so each draws independent noise. A shared seed makes
    ABC condition on one noise realization; joshpy warns about it.
  - Seed only the pseudo-observed run and replicate studies. Unseeded Josh runs are random (verified).
- **sbi fills the last population when the budget runs out.** If `num_simulations` is exhausted
  mid-generation, sbi tops up the final population with particles from the previous one
  (`use_last_pop_samples=True`, sbi's default). Those particles may exceed the final ε.
  `carried_over` counts them; in the landscape demo, 14 of 30 were carried over. Passing
  `use_last_pop_samples=False` ends on the last complete generation instead.
- **ForeverTree's landscape statistics are nearly deterministic.**
  - Measured from 50 replicates at the truth, in prior SDs:
    - per-statistic replicate SDs: 0.0009 (mean height) and 0.0028 (CV);
    - replicate distances to S_obs: 0.0027 (median), 0.0097 (largest).
  - The largest accepted distance (0.345) is 35× the largest replicate distance, so the posterior
    width is all tolerance.
  - This is why the next pages observe a **survey**: fixed cells measured with error.
- **CV vs variance.**
  - (mean, var) and (mean, CV) carry the same information.
  - ForeverTree is exactly multiplicative in `maxNewGrowth`, so CV cancels it and isolates
    `minPrecipImpactPct`. That makes CV the better distance coordinate.
- **Replicate count is not part of `run_hash`.**
  - Runs at the same θ with 1 and 10 replicates share a hash. Reuse tops up the missing replicates.
  - Summaries for ABC must name a replicate (`replicate=0`).
- **`cell_data.cell_id` is a row ID, not a cell.** Select survey cells by
  `position_x`/`position_y`.
- **Prior simulations use `mcabc` with `quantile=1.0`.**
  - This is an accepted workaround: rejection ABC that keeps every draw.
  - With the same `seed`, its first draws equal SMC-ABC's first generation, which then reuses
    those runs.
  - Filters on model properties pass a placeholder `observed`.
- **Calibration intervals.** With k accepted values, use their range as the interval: it covers
  (k−1)/(k+1) if calibrated. The 5–95% quantiles of a small sample undercover.
- **Docs.**
  - CI has no sbi, so every page that needs it ships a committed `docs/_freeze/`. Rendering a single
    file re-executes it.
  - With `freeze: true`, a project render never re-executes a frozen page, even after its source
    changes (tested). It silently ships the old frozen output, prose included. So a source edit
    without a fresh, committed render deploys a stale page, not a broken build.
  - Commit a render with `git add docs/_freeze/<page>`. A re-render renames figure files, and the
    new PNGs are untracked, so `git commit -a` would miss them.
  - reticulate needs the torch stderr-logging workaround (a hidden chunk).
  - Figure code must not leave matplotlib return values as expression statements; reticulate prints
    them.
  - With a registry shared across pages, report reuse from provenance (sets of `run_hash`), not from
    run-count deltas or wall-clock time.
  - Check every freeze for leaks before committing:
    `grep -c 'Text(\|Line2D\|BarContainer' docs/_freeze/inference/*/execute-results/html.json`
    should print 0.
- **Devcontainer.** In standalone scripts, import scipy (or joshpy) before torch or sbi. Otherwise
  scipy's extensions can fail to load against the system libstdc++ (`CXXABI_1.3.15`). pytest and
  Quarto renders are unaffected.

## Plan: PR series, stacked after #88

Each PR is self-contained and commits its own frozen pages. Rebase with `--update-refs` when a lower
PR changes, and push with `--force-with-lease`.

**Current state of #89 (`docs/sbi-tutorial`):**
- Pushed: the landscape SMC-ABC tutorial (`c73e532`).
- Local only: commit `d885798`. It contains:
  - `references.bib` and `references.qmd`, with the navbar, sidebar and bibliography wiring;
  - the revised tutorial source.
- Its committed freeze is still the old page's, so pushing it as-is would deploy the old page.
- The render of the revised tutorial is uncommitted.

PR 1 drops the tutorial page, so neither render ships.

### PR 1: #89, re-scoped. Inference section and site-wide references (no render)

- Already done in `d885798`: `docs/references.bib`, `docs/references.qmd` (one section per page),
  the References tab, and the project-level bibliography.
- An "Inference" navbar tab and sidebar.
- `docs/inference/index.qmd`, with no executable code:
  - joshpy vs sbi responsibilities;
  - notation;
  - Hartig's three steps, and the Fig. 5 options (a–d) mapped to pages;
  - method families;
  - a ForeverTree → Joshua tree mapping table.
- `docs/inference/_sbi-setup.qmd`: a hidden partial with the torch workaround and pandas options.
- Remove `docs/tutorials/sbi-abc.qmd` and its freeze.
- Tutorials link to Inference; Adaptive Sweeps stays under Tutorials.
- **Verify:** `pixi run docs-build` executes nothing; there are no dangling links.

### PR 2: ForeverTree case study (render about 20 min)

- **`examples/sbi_forevertree/forevertree_case.py`**, an importable module with no import side
  effects. Later pages import it to "start from the middle". It holds:
  - constants: truth, prior, seeds, survey cells by position, measurement SD;
  - `model_config` and `run_at`, which checks the registry before re-running;
  - `run_sbi`;
  - the survey observation and statistics;
  - leak-free plot helpers;
  - `load(registry, cli)`, which ensures the truth run and prior simulations exist (reused by
    `run_hash`) and returns the observed values, scale and reference table.
- `run_abc.py` reuses the module.
- **`docs/inference/forevertree.qmd`:**
  - the model and its stochasticity;
  - pseudo-observed survey data (`seed=1`);
  - a **pipeline figure**: all cells → surveyed cells with measurement error → statistics →
    comparison with an ε-ball;
  - 200 prior simulations (`mcabc`, `quantile=1.0`);
  - a **candidate-statistics figure** (mean, SD, variance and CV against each parameter) with the CV
    argument;
  - scaling;
  - a replicate study at the truth (`seed=2`, 50 replicates).
- Measurement noise is seeded from `(run_hash, replicate)`, so replay reproduces it.
- **Shared registry** `docs/inference/forevertree.duckdb`:
  - Add it and its `.duckdb.wal` to `.gitignore`.
  - Extend `docs-clean` in `pixi.toml` to remove both. Today it only clears
    `docs/tutorials/*.duckdb*`.
  - Don't delete it at the end of a page, and don't add it to `render-docs.sh`. Either would wipe the
    cache between renders.
  - Add a `pixi run docs-inference` task that renders the pages in order.
- **Decision point:** the replicate study sets the measurement SD and PR 3's budget, so the final ε
  can approach the replicate spread.

### PR 3: SMC-ABC on the survey, Hartig Fig. 5a (render about 60–80 min)

- `docs/inference/smc-abc.qmd` starts from `forevertree_case.load(...)`. It covers:
  - the indicator kernel and ε, the simulated survey, and Wilkinson's alternative reading;
  - the run and generation table (acceptance, carried over);
  - contraction, ESS and credible intervals;
  - a **sufficiency test**, with a **final-state figure**: the truth vs the farthest accepted
    mean-only run vs the farthest accepted mean+CV run (survey maps and distributions);
  - the registry, resume, calibration by leave-one-out re-filtering of the prior simulations, and
    scaling up.
- If joshpy friction appears, fix it first in a small code PR placed before this one. Two known
  candidates: a batch whose summaries all error yields silent NaN; summary closures serialize as
  unimportable paths.

### PR 4: Rejection filter on a model property, Hartig Fig. 5d (render about 30–40 min)

- **Filter criteria:**
  - F1: at least 90% of 10 replicates have a final-window annual gain in [1.2g\*, 1.4g\*].
  - F2: the SD of the gain within the window is below s_max.
- **Rejection sampling from the prior:** `mcabc`, `quantile=1.0`, with a placeholder `observed`.
- **Report:** the share passing each criterion and both; a pairs plot; the edges of the accepted
  region ("what would it take").
- Say plainly that the result is not a posterior: no data entered it.
- Note that the 10-replicate runs share hashes with PR 2's prior runs, which get topped up.

### PR 5: Posterior probability of a model property (render about 5 min; reuses PR 3's runs)

- P(passes | S_obs) = Σ wᵢ·passᵢ / Σ wᵢ over PR 3's final particles, with a Monte Carlo SE from the
  ESS.
- **Recovery checks:** a band excluding g\* gives ≈ 0; a band including it gives a higher value.

## Parked

- **Summary primitives.** While building PRs 2–5, log every hand-written query and helper:
  - survey cells by position;
  - last step and step windows; lagged gains;
  - per-replicate selection;
  - run-seeded RNG for measurement error;
  - scale divisors;
  - recording the statistics spec in session metadata.

  After PR 5, propose the smallest primitive set those uses justify.

  Logged in PR 2 (`forevertree_case.py`):
  - No "last step" helper, so `FINAL_STEP = 10` is hard-coded from the model.
  - Per-cell values come from `DiagnosticQueries.get_all_variables_at_step`, which
    selects every variable; that is adequate.
  - Survey selection by `(x, y)`, measurement noise and the statistics are pandas/numpy
    helpers (`survey`, `measurement_rng`, `statistics`).
  - The scaled-summaries closure (`CaseStudy.abc_target`) would serialize as an
    unimportable path in `SbiStrategy.to_dict`.
- **A general known-truth toolkit (`joshpy.inference`).** It would turn the case-study
  workflow into an API for testing whether statistics are sufficient before fitting
  field data.
  - **The user supplies functions:**
    - `observe(registry, run_hash, replicate, rng)`, where `rng=None` means no
      observation error;
    - `statistics(data)`;
    - which statistics to compare;
    - a scale;
    - a distance, given as an sbi name or a callable. sbi accepts both, so one object
      serves both the diagnostics and the `SbiStrategy` search.
  - **joshpy supplies the workflow:**
    - seeding `rng` from `(run_hash, replicate)`, and naming the replicate explicitly;
    - the truth run and the prior simulations, reused from the registry;
    - the reference table, statistic × parameter correlations and distances;
    - the replicate study;
    - leave-one-out coverage.
  - **Can be layered on later.** It is additive, and runs are keyed by model, config,
    data and seed, not by how statistics are computed, so nothing needs re-simulating.
  - **Meanwhile:**
    - keep the example's functions in this shape: `observe` with that signature,
      `statistics(data)` separate, one named distance function, and the scale as plain
      data;
    - keep the noise-seed derivation, `sha256("{run_hash}:{replicate}")`, unchanged.
  - Extending `SbiStrategy.distance` to accept a callable is a small, separate change.
- **A batch prior-sampling strategy** (random or Latin hypercube via `scipy.stats.qmc`). Revisit if
  the `mcabc` workaround gets in the way: placeholder `observed` for filters, or more parameters
  where a Latin hypercube matters.
- **Warm start** from a checkpointed population. Only needed if replay fails; it relies on private sbi
  behaviour.
- **Neural methods** (`method="npe"`, `"nle"`, …) and sbi's SBC/coverage diagnostics.
  - Later ABC generations are not prior draws.
  - NLE/NRE accept any proposal; NPE needs its sequential, proposal-corrected variants.
- **Kubernetes run of the demo** (the old Phase 4). Measure per-Job overhead before choosing a scaling
  step.
- **Joshua tree prerequisites:**
  - Fire scenarios are per-set data files, but the prior only feeds template variables; this needs a
    file-parameter design.
  - Fix outcome-group attribute names (`check_stability`, `check_data`, `group`) in the rules sheet.
  - The stability check needs R > 1 replicates.
- **Float rendering.** Every continuous draw gets a unique `run_hash` and full-precision
  `custom_tags`. Watch this at 1,000+ runs per generation.
- **sbi API churn.** The extra is open-ended (`sbi>=0.27`). Replay needs the exact sbi and torch
  versions, so cap it (e.g. `sbi>=0.27,<0.28`) before the long Joshua tree runs, and keep the
  adapter tests.

## Execution on Kubernetes

- **Current mapping:** one parameter set → one `batchRemote` → one indexed Job, with completions
  equal to `replicates`.
  - Each set is staged under its own MinIO prefix, with its own copy of the `.jshd` inputs.
  - With `batch_no_wait=True`, a batch of B sets is B Jobs dispatched together and then polled.
  - Rerunning reattaches to Jobs still running.
- **Replicates are a scientific setting, not the scaling lever.**
  - ABC uses `replicates=1` per set.
  - Model-property filters use R > 1 and summarize over the replicates.
- **Scaling path:**
  1. **Now:** one Job per set.
  2. **When staging dominates** (large climate `.jshd`): stage shared data once per session.
  3. **When Job overhead dominates** (thousands of sets per wave): a Josh-side multi-config indexed
     Job.
