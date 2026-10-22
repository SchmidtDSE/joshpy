# SBI Implementation for joshpy

Oct 22, 2026 · @nick · joshpy 0.0.9.35 · sbi 0.27.0 with CPU-only torch 2.14.1

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
- **Summaries errors** (#92):
  - NaN statistics reject a run (`test_nan_statistics_reject_runs`).
  - An exception from `summaries` stops the search after its batch, chained to the original error,
    and a rerun reuses the batch (`test_summaries_exception_stops_the_search`). Returning `None`
    raises too.
  - A run whose results can't be loaded gets NaN and is listed in `SbiResult.load_errors` and the
    session metadata, with a warning even when quiet. It is not counted in `failed`.
  - SMC-ABC raises if its initial population has fewer than `num_particles` usable runs; sbi
    would otherwise keep runs without statistics and set ε to 1e8.
  - A quantile MCABC whose accepted samples include unusable runs warns. It doesn't raise, because
    the prior-simulation workaround (`quantile=1.0`) keeps every draw.
- **Serialization.** `to_dict` saves a callable as `module:qualname` only when that path loads it
  back, and never for `__main__`. Lambdas, closures, bound methods, partials and callable
  instances are named under `summaries_not_importable` / `objective_not_importable`, for
  provenance only.
- **Config namespace.** The default differs by path: batch sweeps use `sweep_config`; adaptive
  sweeps use the template's stem. An explicit `JobConfig.config_name` wins on both.
- **Parameter-less adaptive sweeps.** `SweepConfig` presence is checked with `is not None`, not
  truthiness, so a sweep that has only a strategy still runs adaptively
  (`test_job_config_keeps_parameterless_sbi_sweep`). Before this fix, such sweeps ran as empty
  batch sweeps.
- **Resume.** Rerunning an interrupted search with the same seed against the same registry ends
  with identical particles (`test_interrupted_run_replays_identically`).
- **Topping up replicates** (#94). Asking for R replicates of a run the registry holds with fewer
  finds the same `run_hash` and reruns it with all R: the CLI runs replicates 0..R−1, and loading
  skips the indices already loaded, so their data and `run_id` don't change
  (`test_run_with_fewer_replicates_is_rerun_without_reloading_it`,
  `test_filter_prior_tops_up_the_prior_simulations`). Before #94, `evaluate` first tried to load
  the missing replicates from the old run's export paths, waiting about 1 s per missing file.
  `collision_policy="pool"` would run only the missing indices, but saves nothing measurable
  locally (JVM startup dominates).
- **Particle values are the rendered values** (#94). `SbiResult.to_dataframe()` gives each
  parameter as the run was rendered (the float32 draw's shortest decimal, as in
  `job.parameters`), so rerunning a particle's parameters reproduces its `run_hash`
  (`test_particles_carry_the_values_their_runs_were_rendered_with`). `SbiResult.samples` stays
  sbi's float32 tensor.
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
    new PNGs are untracked, so `git commit -a` would miss them. Quarto doesn't delete the old
    PNGs either: `git rm` any figure the new freeze no longer references.
  - reticulate needs the torch stderr-logging workaround (a hidden chunk).
  - Figure code must not leave matplotlib return values as expression statements; reticulate prints
    them.
  - With a registry shared across pages, report reuse from provenance (sets of `run_hash`), not from
    run-count deltas or wall-clock time.
  - Check every freeze for leaks before committing:
    `grep -c 'Text(\|Line2D\|BarContainer' docs/_freeze/inference/*/execute-results/html.json`
    should print 0. The case study displays its module's source, so the module mustn't contain
    those names either (build legends from labelled lines, not `Line2D` handles).
  - **A running `quarto preview` re-executes a page whenever its `.qmd` changes**, including through
    `git stash`, `checkout` or `rebase`. Its R session holds the shared registry's lock while it
    runs, so a render started meanwhile fails with `RegistryBusyError` (and may print a glibc
    `malloc()` error as R exits). Wait for the preview's R process to exit, and don't touch a
    `.qmd` while a render runs.
  - A render can crash once at the end of a page's knit with `corrupted double-linked list`
    (glibc), seen once on a warm re-render. No freeze is written, `set -e` stops the script, and
    `docs/inference/<page>_files/` is left behind. Delete that and rerun; the retry passed. It
    costs little, because the registry replays.
- **Devcontainer.** In standalone scripts, import scipy (or joshpy) before torch or sbi. Otherwise
  scipy's extensions can fail to load against the system libstdc++ (`CXXABI_1.3.15`). pytest and
  Quarto renders are unaffected.

## Plan: PR series, stacked after #88

**Conventions:**
- Each PR is self-contained and commits its own frozen pages.
- Rebase with `--update-refs` when a lower PR changes, and push with `--force-with-lease`.
- Commits end with the `Co-Authored-By` trailer; PR descriptions end with the Claude Code line.
- Docs PRs stay drafts until the user has reviewed the render.

### Where things stand (Oct 22)

Merge order: #86 → #87 → #88 → #89 → #93 → #90 → #91 → #92 → #94 → #95 ("SBI stack n/10").

| PR | Branch | Status |
|----|--------|--------|
| #86, #87, #88 | `feat/adaptive-evaluate-batch`, `feat/batch-no-wait-resume`, `feat/sbi-strategy` | Open, awaiting merge |
| #89 (PR 1) | `docs/sbi-tutorial` | Done. Inference tab and landing page, site-wide references, old tutorial retired |
| #93 | `feat/inference-known-truth` | Open. Experimental `joshpy.inference`: `KnownTruthStudy` and `LoadedStudy`, extracted from the case study after the #90 review |
| #90 (PR 2) | `docs/inference-forevertree` | Done, **draft**. Review addressed (below); awaiting the user's look at the render |
| #91 (PR 3) | `docs/inference-smc-abc` | Done, **draft**. Re-rendered on #93 and the revised case study; awaiting the user's review |
| #92 | `fix/sbi-summaries` | Open. Callables saved only as loadable paths; summaries errors stop the search (Behavior notes) |
| #94 | `feat/inference-property-filter` | Open. `PropertyFilter` and `KnownTruthStudy.filter_prior`; a replicate top-up no longer waits for files the earlier run never wrote; `to_dataframe()` gives the rendered parameter values |
| #95 (PR 4) | `docs/inference-rejection-filter` | Done, **draft**. The rejection-filter page, rendered; awaiting the user's review |

### #90 review (Oct 21)

| # | Note | Change |
|---|------|--------|
| 1 | Why seed the measurement noise from `(run_hash, replicate)`? | The error is drawn in Josh: `forevertree.josh` exports `measuredHeight` (mean height plus normal error, SD 0.25 m) next to `meanHeight`, and `observe` keeps the surveyed cells. No Python RNG. The model edit changed every `run_hash`, so every run re-simulated |
| 2 | Generalize the case study | Extracted into `joshpy.inference` (#93), after checking it against `docs/architecture.qmd`. The one tension (one registry per experiment) is now a note there: related experiments that share runs belong in one registry |
| 3 | Dashed and dotted lines look alike; define the statistics | Orange dashed for S_obs, solid black for the truth, with a legend; a LaTeX table of definitions |
| 4 | "Why the CV and not the variance?" reads as a defense | A general six-step process for choosing statistics, with CV vs variance as the example of one step, and pointers to automatic methods (Joyce & Marjoram 2008; Fearnhead & Prangle 2012; Blum et al. 2013) |
| 5 | z-score instead of dividing by the SD | Statistics are z-scored with the prior simulations' mean and SD. Distances don't change, because the mean cancels |
| 6 | The pipeline figure is confusing | Two figures: model output → survey → S_obs, then the prior simulations against S_obs in measured units and as z-scores |
| 7 | The likelihood-approximation section | Kept, with prose placing it in Hartig et al.'s three steps and options (a) and (b) |

### PR 3: status

`docs/inference/smc-abc.qmd` is written, registered (render script, render list, navbar, sidebar,
references, roadmap) and rendered at the decided budget; the case study links to it. Both freezes
are committed. It runs on `joshpy.inference`: `study.search`, `study.posterior` and
`study.leave_one_out`. Its chunks were smoke-tested outside Quarto against a scratch registry at
small budgets before the full render.

**Results at 600 simulations** (re-rendered after the #90 review: 1,110 new simulations, about
50 minutes; the registry holds 3,735 `job_runs`, and runs from earlier versions of the model stay
in it unused):
- **The budget roughly suffices.** Final target ε 0.075, about the median replicate distance at the
  truth (0.07). Generation 4 had 80 simulations left and accepted 11, so 19 of 30 particles were
  carried over from generation 3; the largest distance (0.108) lies between the replicate median
  and the largest replicate distance (0.22).
- Mean + CV: both 90% intervals contain the truth, each about 10% of the prior's 90% width; ESS 27.6
  of 30; parameter correlation −0.63.
- Mean only: ε reaches 0.011 with the same budget. Both intervals contain the truth (widths 24%
  and 73% of the prior's), but the correlation is −0.99: the ridge.
- Leave-one-out coverage (k = 20, expected 0.905 ± 0.021): mean + CV 0.975 and 0.97 (over-covers);
  mean only 0.92 and 0.905, with the `minPrecipImpactPct` width at 0.91 against 0.36.
- Posterior predictive check (30 fresh seeded runs at the final particles): all 8 sites inside.
- Replay of 600 evaluations: 36 s, 0 new simulations, identical population. A warm re-render of
  both pages takes about 3.5 min.

**Working rules (for PR 4 onward):**
- **The shared registry `docs/inference/forevertree.duckdb`** is warm and holds over three hours of
  simulations. Don't delete it at the end of a page, and don't add it to `render-docs.sh`; only
  `docs-clean` removes it.
- **Run scripts through `pixi run`.** `render-inference.sh` needs `CONDA_PREFIX`.
- **`render-inference.sh` renders every page in order**, so a module change re-renders (and
  re-freezes) the case study too.
- **Renders run under an ASCII locale.** Read any displayed file with `encoding="utf-8"`.
- **Figure helpers draw and return `None`.** No other matplotlib call may be an expression statement
  in a chunk.
- **Build on `joshpy.inference`.** ForeverTree-specific code (model config, `observe`,
  `statistics`, figures) stays in `forevertree_case.py`; generic workflow goes into the toolkit.
  Any edit to `forevertree.josh`, including its measurement error, re-simulates every run.
- **Smoke-test against a copy of the warm registry** (in a scratch directory) with small SMC
  settings: the prior runs replay, so only the searches simulate.

### PR 3 outline: SMC-ABC on the survey (Hartig Fig. 5a)

- **Start from the middle:** `study = case.known_truth_study().load(registry, cli)`, then a recap
  of the truth, S_obs and the z-scores; `study.search(...)` runs each search.
- **The likelihood approximation:**
  - an indicator kernel, with sbi's `"l2"` on z-scored summaries;
  - ε as the approximation error [@hartig2011, Box 2; @cranmer2020]; more statistics force a
    larger ε;
  - the survey is simulated, so ε carries only approximation error, with Wilkinson's alternative
    reading [@wilkinson2013];
  - the budget sets the final ε.
- **Run (600 simulations):**
  - Report evaluations, with reuse taken from provenance. The first 60 draws are the first 60 prior
    simulations (same seed), so `len(set(result_hashes) & set(study.reference.run_hash))` should
    be 60.
  - No wall-clock prints except in the replay section.
- **Generation table:**
  - target ε, largest accepted distance, simulations, accepted, acceptance rate, carried over;
  - then compare with rejection sampling at the final ε (the share of prior simulations within
    it), and with the replicate distances.
- **Results:**
  - `plot_generations`, with carried-over particles hollow;
  - a posterior table: weighted mean, central 90% credible interval, its width as a share of the
    prior's 90% width, and truth inside; plus the ESS;
  - notes: the intervals are marginal; small-sample endpoints are noisy; "truth inside" is
    necessary, not sufficient.
- **Sufficiency test:**
  - the same search on the mean alone; `plot_final_populations`; parameter correlations;
  - a **final-state figure** via `plot_final_states`: the truth, and `farthest_particle` of the
    mean+CV and of the mean-only searches;
  - equifinality [@beven2006] (non-identifiability); marginal intervals can hide the ridge.
- **Registry:** run attributes and session metadata.
- **Resume:** replay the first search (0 new simulations, identical final population).
- **Calibration:**
  - leave-one-out rejection ABC on `study.reference`, keeping 20 of 199;
  - score whether the held-out value falls strictly inside the accepted range, which happens
    (k−1)/(k+1) ≈ 90% of the time if calibrated [@talts2018; @csillery2012];
  - expect over-coverage from ε > 0.
- **Closing sections:**
  - a misspecification callout and a posterior predictive check;
  - scaling up: clusters, replicates, failures, the limits of rejection ABC, regression adjustment
    [@beaumont2002], and the choice of tolerance, summaries and metric.
- **joshpy friction:** if it appears, fix it first in a small code PR placed before this one. Two
  known candidates: a batch whose summaries all error yields silent NaN; summary closures serialize
  as unimportable paths. Both were fixed in #92 (see Behavior notes).

### PR 4: status (rejection filter on a model property, Hartig option (d))

`docs/inference/rejection-filter.qmd` is written, registered (render script, render list, navbar,
sidebar, references, roadmap) and rendered; the code is #94, the page #95. The first render took
43 minutes (200 parameter sets topped up to 10 replicates; 3,935 `job_runs`); a warm re-render takes
about 80 s and runs nothing.

**What was built:**
- The property: $H_t$, the mean height of all 63 cells; the annual gain $g_t = H_t - H_{t-1}$;
  over the final window (steps 6–10), its mean $\bar g$ and SD $s_g$
  (`case.final_window_gain`). g\* and s\* are their means over the case study's 50 replicates at
  the truth: 0.6234 and 0.0166 m/yr.
- `case.gain_filter`: F1 $1.2g^* \le \bar g \le 1.4g^*$, F2 $s_g \le s^*$; 10 replicates; a
  criterion passes when 90% of replicates meet it.
- `spec.filter_prior(registry, cli, gain_filter, n=200)`: the case study's 200 prior draws (all 200
  reused by `run_hash`). Provenance shows replicate 0 recorded by `forevertree_prior` and 1–9 by
  `forevertree_filter`.

**Results:**
- F1 19, F2 139, both 5 of 200: P(passes) under the prior 0.025 ± 0.011.
- F1 alone is a trade-off band (the mean-height ridge); F2 keeps its drought-tolerant end. Accepted
  `maxNewGrowth` 0.915–1.175, `minPrecipImpactPct` 56.2–87.4 (truth 1.0, 40).
- ForeverTree is multiplicative in `maxNewGrowth`, so the gain's CV depends on
  `minPrecipImpactPct` alone; both criteria need a CV ≤ s\*/(1.2g\*), which puts the region's edge
  at about 47–48% (all 200 draws), below the smallest accepted draw.
- Replicates: CV between replicates 0.09% (mean gain) and 5.4% (its SD). They disagree on F1 for 2
  draws (on the band's edges) and on F2 for 12 (within 15% of s\*). Requiring any, 90% or all
  replicates accepts the same 5.
- The page says plainly that this is not a posterior, and points to PR 5 for P(passes | S_obs).

### PR 5: Posterior probability of a model property (render about 5 min; reuses PR 3's runs)

- P(passes | S_obs) = Σ wᵢ·passᵢ / Σ wᵢ over PR 3's final particles, with a Monte Carlo SE from the
  ESS.
- **Recovery checks:** a band excluding g\* gives ≈ 0; a band including it gives a higher value.
- **First, top up the particles.** Each of the mean + CV search's 30 final particles needs the
  filter's 10 replicates: `spec.run_at(registry, cli, params, experiment, replicates=10)`, then
  `case.gain_filter(g_star, s_star).evaluate(registry, particles["run_hash"])`.
- **A particle's parameters reproduce its run** (#94). Checked on a copy of the docs registry:
  `run_at` at `job.parameters` reproduces 30 of 30 particle hashes. Before #94,
  `to_dataframe()`'s columns reproduced 0 of 30, because they widened sbi's float32 values to
  float64 (`1.0123386` became `1.012338638305664`).

## Parked

- **Summary primitives.** While building PRs 2–5, log every hand-written query and helper:
  - survey cells by position;
  - last step and step windows; lagged gains;
  - per-replicate selection;
  - z-scores (now `LoadedStudy.zscore`);
  - recording the statistics spec in session metadata.

  After PR 5, propose the smallest primitive set those uses justify.

  Logged in PR 2 (`forevertree_case.py`):
  - No "last step" helper, so `FINAL_STEP = 10` is hard-coded from the model.
  - Per-cell values come from `DiagnosticQueries.get_all_variables_at_step`, which
    selects every variable; that is adequate.
  - Survey selection by `(x, y)` and the statistics are pandas/numpy helpers
    (`observe`, `statistics`). The measurement error is a Josh export.
  - The z-scored summaries closure (`LoadedStudy.abc_target`) has no importable path.
    `SbiStrategy.to_dict` records only its name (#92), so a strategy can't be rebuilt
    from saved config. Fixing that needs summaries given as a path plus arguments.

  Logged in PR 4 (`forevertree_case.py`):
  - The landscape mean per replicate and step (`landscape_heights`) and a step window
    (`final_window_gain`) are hand-written SQL on `cell_data_current`.
    `DiagnosticQueries.get_replicate_cv` builds the same per-replicate series internally but
    doesn't return it.
  - Lagged gains are `np.diff` over the window; the window is a constant (`WINDOW`), like
    `FINAL_STEP`.
- **Known-truth toolkit: follow-ups.** `joshpy.inference` (#93) is experimental. Not yet:
  - a distance other than sbi's `"l2"` (extending `SbiStrategy.distance` to accept a callable is
    a small, separate change);
  - summaries as an importable path plus arguments (see above);
  - generic figures (the docs' figures stay in the example module);
  - exporting it from `joshpy/__init__.py`, once PR 4 and PR 5 have used it;
  - `PropertyFilter` (#94) runs only on prior draws (`filter_prior`) or on runs that already hold
    its replicates (`evaluate`). Topping up other runs, such as PR 5's particles, goes through
    `run_at`.
- **Failures in filters and searches.** `filter_prior` and `load` raise if any run fails; searches
  reject failed runs. For the Joshua tree runs: retry infrastructure failures (evicted pods,
  staging errors), record model failures (out of memory when a population explodes) as missing,
  and report where in the parameter box they fall.
- **Refining a filter in waves.** Rejection from the prior is wasteful when the accepted region is
  small (PR 4: a handful of 200 draws, 2,000 replicate runs). SBI.md's design refines in waves. One
  route through sbi: summaries that give each criterion's distance outside its bounds (0 inside),
  `observed` all zeros, and SMC-ABC's tolerance shrinking to 0, which is the filter; it needs a
  stopping rule at ε = 0 and the replicate share folded into the distance.
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
