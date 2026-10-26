# SBI Implementation for joshpy

Oct 26, 2026 · @nick · joshpy 0.0.9.35 · sbi 0.27.0 with CPU-only torch 2.14.1

How joshpy runs the filter-and-refine search in [SBI.md](SBI.md). The machinery is proved on
the ForeverTree demo (`examples/sbi_forevertree/`). It then scales to a second ForeverTree model
(births, deaths, fuel and fire) and the Joshua tree experiments.

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

### Where things stand (Oct 26)

Merge order: #86 → #87 → #88 → #89 → #93 → #90 → #91 → #92 → #94 → #95 → #96 ("SBI stack n/11").

| PR | Branch | Status |
|----|--------|--------|
| #86, #87, #88 | `feat/adaptive-evaluate-batch`, `feat/batch-no-wait-resume`, `feat/sbi-strategy` | Open, awaiting merge |
| #89 (PR 1) | `docs/sbi-tutorial` | Done. Inference tab and landing page, site-wide references, old tutorial retired |
| #93 | `feat/inference-known-truth` | Open. Experimental `joshpy.inference`: `KnownTruthStudy` and `LoadedStudy`, extracted from the case study after the #90 review |
| #90 (PR 2) | `docs/inference-forevertree` | Done, **draft**. #90 review and Oct 25 review addressed (below); re-rendered |
| #91 (PR 3) | `docs/inference-smc-abc` | Done, **draft**. Oct 25 review addressed; re-rendered; awaiting the user's review |
| #92 | `fix/sbi-summaries` | Open. Callables saved only as loadable paths; summaries errors stop the search (Behavior notes) |
| #94 | `feat/inference-property-filter` | Open. `PropertyFilter` and `KnownTruthStudy.filter_prior`; a replicate top-up no longer waits for files the earlier run never wrote; `to_dataframe()` gives the rendered parameter values |
| #95 (PR 4) | `docs/inference-rejection-filter` | Done, **draft**. The rejection-filter page; Oct 25 review addressed; re-rendered; awaiting the user's review |
| #96 (PR 5) | `docs/inference-posterior-property` | Done, **draft**. The posterior-probability page; awaiting the user's review |

### Next phase: readiness (Oct 26)

| Work | Ready? | Notes |
|------|--------|-------|
| PR 5: posterior probability of the property | **Done** (#96, draft) | Rendered; awaiting review |
| `csv_precision` passthrough; `use_float64` in `run_hash` | **Ready** | Small code PR ("Josh CSV export bug") |
| SMC-ABC batch policy | **Ready** | Design settled ("SMC-ABC batch policy"); names are the open choice |
| Second ForeverTree model | **Drafted**, awaiting review | `forevertree_fire.josh` and a config per fire mode; open items in "Second ForeverTree model" |
| Parameter declarations (direct vs inverse) | Direction agreed | API to draft; the second model is its first user |
| Field data (a study without a truth run) | Ready, small | Needed for Joshua tree, not for ForeverTree |
| Test (2) as SMC-ABC on a stability target | Waits | Needs the second model's population growth rate |
| Pilot on the cluster | Blocked | A Josh CSV fix for Kubernetes, the batch policy, and a long-run model |
| Cleanup (Optuna, spin-up) | Parked | Wholesale, once SBI is stable |

Nothing in the stack has merged yet, so new code PRs stack on #96 unless the lower PRs merge first.

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

### Review of the Inference pages (Oct 25)

| # | Note | Change |
|---|------|--------|
| 1 | `case.known_truth_study()` is a wrapper with no purpose | The module holds `STUDY`, the arguments as a dict; pages call `KnownTruthStudy(**case.STUDY)`, and the case study shows the dict. `model_config()`, `replicate_study()`, `replicate_job()` and `gain_filter()` are gone too: the replicate study and the `PropertyFilter` are written out on the pages |
| 2 | Bless a few general plots (e.g. S_sim vs S_obs as a convergence signal), but avoid the maintenance burden | None blessed yet. Parked below, as a data method with the plot left to the page |
| 3 | Pattern-oriented modelling: how many patterns do mean and CV fix? Within-site CV, temporal patterns? | Answered from the prior simulations (Parked: patterns at more scales). No change to the statistics yet; the user decides |
| 4 | Is the posterior predictive check a held-out pattern? | Yes; the SMC-ABC page now says so (Grimm et al. 2005) |
| 5 | "Starting from the middle" is too meta | "Picking Up from the Case Study", stating what the page assumes; the rejection-filter page says the same |
| 6 | Use ggplot, inline and folded, not plot code buried in `forevertree_case.py` | Every figure is a folded ggplot2 chunk on its page, fed by a folded pandas chunk; the module keeps only what is specific to ForeverTree (constants, `final_heights`, `observe`, `statistics`, `STUDY`, `final_window_gain`) |

All three pages were re-rendered on their own branches with no new simulations; printed outputs
are unchanged apart from tables now showing the statistics' own names (`mean_height`,
`height_cv`) and the replay timing.

### Review of the PR 4 and PR 5 pages (Oct 26)

| # | Note | Change |
|---|------|--------|
| 1 | The docs are a generic guide; keep anything about the Joshua tree test out of them | Removed from the rejection-filter page, the posterior-property page, and the Inference index's roadmap sentence and mapping table. The removed text is kept in `JOTR_NOTES.md` in the repo root, a scratch file outside git. This plan keeps the Joshua tree design |
| 2 | F1, F2, R1, R2 read as opaque; put them in biological terms | `faster` (the stand gains 0.75–0.87 m of height a year, 20–40% more than at the truth), `steady` (its gain swings no more between years), and the positive control's `same_rate` and `same_swings`. ḡ and s_g are named the stand's growth rate and growth variability |
| 3 | Read the results as a manager would: bound the biological parameters, then state how likely the posterior is to meet a hypothesis about the landscape | The posterior page opens with that two-step workflow, shows the trait intervals, and ends with "Reading the Results". The rejection-filter page reads its accepted region as a planting choice (more drought-tolerant trees, not faster-growing ones) |
| 4 | Why estimate the prior probability of a band we don't expect to be realistic? | The band is a negative control, which the pages now say. The rejection filter's result is the accepted region (what it would take), not its prior share. On the posterior page the prior column is labelled a baseline, and the realistic statement is the posterior one |

Also added to the posterior page: the probability read two ways, by the 90% rule (a stand that
reliably behaves so) and by pooling replicates (one realization of the landscape). For growing as
at the truth, 0.93 against 0.99. Both pages were re-rendered with no new simulations; every number
is unchanged.

### Joshua tree design (Oct 25)

**Two separate tests, never one combined search.**
1. **Fit.** Which parameter sets reproduce the I&M patterns (pattern-oriented: several
   statistics, at the scales where the model has processes)? SMC-ABC, as on the SMC-ABC page.
   Whether the fitted sets replace themselves is checked **post hoc**:
   P(replacing | I&M) = Σ wᵢ·passᵢ / Σ wᵢ over the final particles, each topped up to N replicates.
   PR 5 does exactly this on ForeverTree.
2. **Counterfactual: an idealized scenario of stability** (revised Oct 26). Also SMC-ABC: target
   statistics that describe stability (e.g. a population growth rate of 1 and a low year-to-year
   CV) take the place of observed patterns, and ε plays the band's role. It runs from the prior,
   independently of (1), which de-risks the work: (2) doesn't depend on (1) fitting well.
   Starting from (1)'s posterior instead isn't an important distinction. PR 4's `PropertyFilter`
   is the rejection version and a cheap first look. Replacement is a counterfactual that a
   one-timestep survey can't characterize, so it never enters (1)'s fit.

This supersedes SBI.md's "one run, two checks" with four outcome groups, and its filter-and-refine
search over both checks.

**What the existing workflow covers, and the gaps:**
- Test (1): `SbiStrategy` SMC-ABC, then `PropertyFilter.evaluate` on the final particles after a
  top-up (`run_at` with `replicates=N`; since #94 a particle's parameters reproduce its run). PR 5
  shows whether a top-up helper earns a place in the toolkit.
- Test (2): `SbiStrategy` SMC-ABC with the stability target as `observed`; the second ForeverTree
  model provides the growth rate. `PropertyFilter` on prior draws (`filter_prior`) is the rejection
  version.
- **Gap: field data.** `KnownTruthStudy.load` runs a truth to get S_obs. With I&M data, S_obs comes
  from the data, and the prior simulations only set the z-score scale. Small generalization: a
  study without a truth, S_obs given.
- **Gap: SMC-ABC's round structure on Kubernetes** (next; decided Oct 26 in "SMC-ABC batch
  policy").

**Kubernetes: one Job per parameter set; rounds, not generations, set the wall-clock.**
- One proposal → one indexed Job with N completions; a batch of proposals is dispatched together
  and polled (`batch_no_wait`). Staging each set separately is fine: Joshua tree runs are slow
  enough that staging won't dominate.
- **During the fit, one replicate per proposal.** sbi's SMC-ABC compares one simulation per
  proposal with S_obs, and its acceptance across proposals handles the noise. N replicates are only
  needed for the post-hoc check, so top up the final particles (one extra round) rather than run N
  for every proposal (N× the cost, spent mostly on rejected proposals).
- **A generation is many rounds.** Within a generation, sbi proposes only as many as are still
  missing (`SMCABC._sample_next_population`), so batches shrink as particles are accepted. PR 3's
  600 simulations ran as **70 sequential batches** (generations 0–4: 1, 19, 17, 29 and 4 rounds),
  most late ones of 1–2 proposals. Wall-clock ≈ rounds × run time, with the cluster mostly idle.
  Decided Oct 26: a configurable batch policy ("SMC-ABC batch policy"). The alternatives were an
  engine built for parallel ABC-SMC (pyABC's distributed samplers keep workers busy) and sequential
  neural estimation with fixed-size rounds (parked in SBI.md for trust reasons).

**Early stopping and failures: test the workflow first.** Both are invasive, early stopping
especially. Run a pilot on the cluster, see which failures occur and what stopping would save,
then decide.

**Parameter declarations: one way for every strategy.** SBI is becoming joshpy's canonical
workflow (a plain random sweep is a prior simulation that keeps every draw). Today sweeps and
Optuna read `ConfigSweepParameter` (`values` as a list, `np.arange` or a range dict),
`FileSweepParameter` and `CompoundSweepParameter`, while `SbiStrategy` takes its own continuous
prior dict and no file parameters. Review before the next case study:
- **Terminology (Oct 26): direct vs inverse (indirect) parameterization** (Hartig et al. 2011), not
  "prior". A reader takes a "prior" to mean the parameter must be fit by inverse modelling, but
  direct parameterization is normal for some models.
  - *Direct* parameters carry values: fixed, or swept, with one search per combination (a
    condition).
  - *Inverse* parameters carry a distribution that SBI samples.
- declare both on the same parameter objects, and have `SbiStrategy` read the inverse ones from
  `SweepConfig`;
- file and compound parameters are direct only: a categorical can't take sbi's Gaussian
  perturbation kernel, so infer such inputs through continuous parameters instead;
- fire (Oct 26): per patch, not spatial, because the Joshua tree design simulates each point. Fire
  is sometimes deterministic and sometimes stochastic, and the model can't assume either, so both
  modes come from the jshc (see "Second ForeverTree model"). No file primitive is needed. Any
  observed fire history stays fixed files (`file_mappings`).

**Grass–shrub fire is not a case study (Oct 26).** At 1,000 steps it is stable at every damage
setting, so stability isn't something to find there ("Second ForeverTree model"). Its role moves
into ForeverTree.

### SMC-ABC batch policy (Oct 26)

**A generation needs `num_particles` passed runs (inside ε); every proposal is simulated.** ε, the
importance weights and the perturbation kernel change only between generations, because they need
the finished population. Rounds within a generation are repeated draws from one fixed proposal;
only the count of passes changes when a round returns. sbi proposes as many runs as passes are
missing, so it never overshoots (the fewest simulations) but needs many rounds. The tail is the
cost: with one particle missing at 20% acceptance, the last particle alone averages 5 rounds.

**The batch policy is a trade-off between wasted compute and wall-clock, so it's configurable on
`SbiStrategy`:**
- `exact`: sbi's own loop. Fewest simulations; 18–29 rounds per generation at 20% acceptance and
  78–122 at 5% (30–300 particles). Right when runs execute one at a time, as `SweepManager` does
  locally, where wall-clock tracks total simulations.
- `measured`: round 1 proposes a fraction (default 0.7) of missing ÷ the previous generation's
  acceptance rate, so it rarely overshoots. Later rounds size from this generation's measured rate
  with a margin (default 1.1), and an optional cap per round (cluster capacity) bounds every batch.
  Simulated: 2.0–2.2 rounds per generation, for extra simulations of +14% at 30 particles, +8% at
  100, +6% at 300 and +5% at 1,000. The simulation assumes the acceptance rate moves about 30%
  between generations, as in PR 3 (0.20, 0.24, 0.16).
- Rejected: one round sized from the previous generation's rate × 1.2 (1.6–2.0 rounds but about
  +30%). The waste comes from the rate guess, which a margin can't fix, rather than from counting
  noise, which shrinks with population size.
- **Default by dispatch mode:** `exact` for local runs and `measured` for `batch_remote`; users can
  override every setting. PR 3's page runs locally, so its replay is unchanged.
- **Validity:** keep the first missing passes in **proposal order**, never the closest, which
  would quietly tighten ε. Weights come from sbi's `_calculate_new_log_weights` against the
  previous population. Batch sizes depend only on counts already observed, as sbi's do.
- **Implementation:**
  - an `SMCABC` subclass overriding the private `_sample_next_population`, with the version
    pinned to `sbi<0.28` ("sbi API churn" in Parked); an upstream option later;
  - cap every batch at the remaining `num_simulations` budget, as sbi does;
  - record rounds and surplus per generation in `SbiResult.populations`.
- **Cluster saturation:** a round larger than the cluster runs in waves, with pods Pending until
  capacity frees up or the autoscaler adds nodes, so `batch_timeout` must allow it. The surplus then
  costs wall-clock too, which is what the cap is for.
- **Tests:** a toy simulator, where `exact` and `measured` give the same population distribution
  (moments over seeds); round counts; the budget; replay.

### Second ForeverTree model (Oct 26)

**Why:** the 1,000-step grass–shrub run (`examples/adaptive_fire.josh`: 16 damage settings, 10
replicates each, exported at full precision) is stable at every setting:
- it settles by about step 80;
- damage sets the level of total cover (80–128% of a possible 200%);
- fire feedback barely acts, because patches sit above the 30% fire threshold 75–97% of the time.

A separate example would add little, so ForeverTree gains its role.

**Drafted for review (Oct 26):** `examples/sbi_forevertree/forevertree_fire.josh`, with one config
per fire mode in `fire_modes/{scheduled,stochastic}/forevertreeFire.jshc` (the config name is
`forevertreeFire`). `forevertree.josh` and the four Inference pages are unchanged. It is not yet
wired into joshpy (no template, case module or registry); that follows the review. It runs with the
Oct 22 dev jar; the browser bundle for demo.joshsim.org exports to `memory://editor/patches`
instead of a file.

**The run.** `steps.high = spinupYears + 10`. The simulation has a `state`: "spinup", then
"observed" at step `spinupYears`. During the spin-up, `year.step = sample discrete uniform from 0
count to 10 count`, the same draw for every patch; in the observed years, `year` runs 0–10 in order.
Patches read `external X at meta.year`.
- **Finding:** in the Oct 22 dev jar, an unadorned `external X` still reads by step index (Josh warns
  that it is "interpreted as a zero-based JSHD index") and fails past the data's last step. `at
  meta.year` reads the resampled year, so the model writes it explicitly.
- `observedPrecipChange` scales precipitation in the observed years only (default 0%): the climate
  scenario test (2) needs (below).
- Josh quirks met: `%` parses as percent, not modulo (the schedule uses `floor`); `and` doesn't
  short-circuit (the schedule guards its division); `n` and `p` are keywords.

**Mechanisms**, per patch and year (parameters in brackets; d is dryness, the linear map of
precipitation from 300 mm (d = 1) to 500 mm (d = 0), clamped):
- **Growth**, as in forevertree.josh [`maxNewGrowth`, `minPrecipImpactPct`].
- **Deaths**, removed at the next step's start (the two-trees `dead` slice). A juvenile (below
  [`maturityHeight`]) dies with chance [`juvenileMortality`] + d × [`droughtMortality`], an adult
  with [`adultMortality`]. In a fire, a juvenile dies with chance equal to the fuel cover that
  burned, and an adult with [`adultFireMortality`] × that cover.
- **Births:** recruits ~ Binomial(adults, [`recruitment`] × (1 − d) × (1 − trees / [`maxTrees`])),
  created at 0 m in `.end`. The run starts from [`initialTrees`] seedlings per patch.
- **Fuel**, a cover fraction: fuel ← fuel + [`fuelGrowth`] × (1 − d) × (1 − fuel); 0 after a fire.
- **Fire:** a patch burns if scheduled or if stochastic fire strikes; both may be on.
  - *Scheduled:* every [`fireInterval`] years from step [`firstFire`] (0 turns it off). These are the
    same steps in every run and every patch.
  - *Stochastic:* a chance of [`fireChance`] × last year's fuel (0% turns it off).
- **Exports:** year, observedYear, precipitation, nTrees, nAdults, nJuveniles, meanHeight,
  heightSd, fuel, burned, deaths, recruits.

**Behaviour at the defaults** (200-year spin-up, 2 replicates per mode; about 10 s a replicate
locally):

| | Scheduled (every 20 years) | Stochastic (10% at full fuel) |
|---|---|---|
| Fires per patch-year | 0.054 | 0.06 |
| Patches occupied at the end, of 63 | 36–38 | 24–31 |
| Trees at the end (adults) | 1,180 (980–1,010) | 750–830 (600–640) |
| Landscape CV of trees, years 100–199 | 0.09 | 0.03 |
| λ over the observed years: all trees, adults | 1.005, 1.01 | 1.00, 1.00 |
| The same with `observedPrecipChange = -20 %` | 0.99, 1.00 | 0.98, 0.985 |

- **Trees persist on the wetter half:** occupied patches average 460–500 mm against 290–355 mm for
  the empty ones. There is no seed rain, so extinction is permanent and occupancy drifts down slowly
  (45 → 38 between years 50 and 200 in one scheduled run).
- **Fire regularity matters:** at the same fire frequency, the scheduled regime leaves more patches
  occupied, because each 20-year gap lets juveniles mature. Stochastic fires can return within a
  few years. At 10-year intervals, scheduled fire leaves 17–18 patches.
- **Density dependence returns every persisting population to λ ≈ 1 after the spin-up.** A stability
  target is met trivially unless the observed climate differs from the spin-up's. Under 20% less
  rain, recruitment falls by about a third, but the adults barely move within 11 years: they
  matured from earlier recruits.

**What it enables:**
- Test (2) gets its statistic: the juvenile:adult ratio over the observed years (decided Oct 26).
- Patterns at more scales: occupancy, density, juveniles against adults, within-site height SD
  (informative now that ages differ), and recruitment pulses.
- Direct and inverse parameters in one model.
- Long runs for timing on the cluster.
- Fire both ways.

**Open for review:**
1. **Direct vs inverse.** Candidates for inverse: `maxNewGrowth`, `minPrecipImpactPct`,
   `recruitment`, `juvenileMortality`, `droughtMortality`, `adultFireMortality`, and `fireChance`
   in the stochastic mode. Direct: `spinupYears`, `observedPrecipChange`, `initialTrees`,
   `maxTrees`, `maturityHeight`, `fuelGrowth`, `adultMortality`, `fireInterval` and `firstFire`.
2. **Statistics for test (1):** occupancy of the surveyed cells, density where occupied, the
   juvenile:adult ratio, mean height and within-site height SD.
3. **Test (2)'s statistic: decided Oct 26, the juvenile:adult ratio**, keeping the 11-year
   window (the Joshua tree test statistic most likely). The 11 years are too short for adults to
   show a drying, but the ratio responds. Landscape ratio over steps 200–210, 2 replicates:
   - stochastic fire: 0.27–0.33 at 0%; at −20% it falls from 0.26 to 0.19–0.21 within 5–10 years;
   - scheduled fire: the 20-year cycle dominates. The window sits between fires at steps 190 and
     210, so the ratio falls from 0.27 to 0.19 as the post-fire cohort matures, with or without
     drying; −20% lowers it by a further 0.02–0.05. Open: whether test (2) uses the stochastic
     mode, or aligns the window with the fire cycle.
4. **Seed rain:** a small yearly chance of one recruit from outside would make occupancy
   stationary. That is one direct parameter more.
5. **Stochastic fire reads fuel only**; climate enters through fuel growth. A dry-year multiplier
   would add one parameter.
6. **Fire mortality is by stage** (juvenile or adult), not continuous in height.
7. **Josh CSV bug exposure:** fuel can fall below 1e-6 only just above 300 mm, but run with
   `--csv-precision=-1` once joshpy passes it.
8. **The page plan:** a page introducing the model (both modes, the spin-up, the patterns), then
   test (1) on a survey, then test (2) under a drying scenario.

### Josh CSV export bug (Oct 26)

Found in the 1,000-step grass–shrub run: 7 of 4,004,000 values were exported as 220–846% cover.
- **Cause:**
  - Josh #388 (Apr 13; on main via #390) replaced #373's BigDecimal rounding with string
    truncation.
  - `MapSerializeStrategy.formatNumeric` cuts the `getAsString()` string `--csv-precision`
    characters (default 10) after the `.`. That drops the exponent of a value printed in
    scientific notation: `3.3144869433E-7` becomes `3.3144869433`.
  - With BigDecimal this hits |x| < 1e-6. With `--use-float-64` it hits |x| < 1e-3 and |x| ≥ 1e7.
  - Unchanged in the Oct 22 dev jar and in prod.
- **Opt-out:**
  - `--csv-precision=-1` works for `run`, but the CSVs are 4× larger.
  - There is none on the sandbox path (`MemoryExportFacade`: the browser and `runRemote` workers).
  - There is none on Kubernetes (`cloud-img/run-entrypoint.sh` passes no precision flag).
- **Reporting:** the user takes it to Josh, with three bottles: the minimal repro in both number
  modes, and the seeded fire run.
- **joshpy:**
  - `RunConfig` and `JobConfig` have no `csv_precision`; add a passthrough, including to a
    bottle's `run.sh`.
  - `run_hash` ignores `use_float64`, so a float64 run and a BigDecimal run of one config collide.
- **Before trusting Kubernetes runs** whose values can fall below 1e-6: the Josh fix has landed, or
  the entrypoint takes a precision.

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
- **Figures are folded ggplot2 chunks on the page** (`code-fold: true`), fed by a folded pandas
  chunk; multi-panel figures are Quarto subfigures (`layout-ncol`, `fig-subcap`), since
  patchwork isn't installed. `_sbi-setup.qmd` loads ggplot2 with the theme, named colours and a
  UTF-8 `LC_CTYPE` (otherwise ρ, ε and arrows print as `<U+...>` under the ASCII render locale).
  Python lists arrive in R as lists (`unlist` them); give manual scales named values.
- **Build on `joshpy.inference`, with no wrappers.** The module holds only what is specific to
  ForeverTree, with the study's arguments in `STUDY`; generic workflow goes into the toolkit.
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

### PR 5: status (posterior probability of a model property, test (1)'s post-hoc check)

`docs/inference/posterior-property.qmd` is written, registered (render script, render list, navbar,
sidebar, references, roadmap) and rendered; the rejection-filter page now links to it. The first
render ran 270 replicate simulations (30 particles topped up to 10 replicates, 7 minutes); a warm
re-render takes about 2.5 minutes and runs nothing.

**What was built (page only, no toolkit change):**
- Replay PR 3's mean + CV search (0 new simulations), and `filter_prior` replays PR 4's 200 draws.
- Top-up: `spec.run_at(registry, cli, params, "forevertree_posterior_property", replicates=10)`
  per particle, with `params` from `result.to_dataframe()`. All 30 hashes are the particles'
  (asserted); provenance shows replicate 0 recorded by `forevertree_smcabc` and 1–9 by
  `forevertree_posterior_property`.
- P = Σ wᵢ·passᵢ / Σ wᵢ, Monte Carlo SE sqrt(p(1 − p) / ESS), and the rule of three (3/ESS) as a
  95% bound when no particle passes, or 1 − 3/ESS when all do [Hanley & Lippman-Hand 1983].

**Results (ESS 27.6 of 30; criteria renamed Oct 26: F1 → `faster`, F2 → `steady`, R1 → `same_rate`,
R2 → `same_swings`):**

| Filter | Prior P (200 draws) | Posterior P |
|--------|---------------------|-------------|
| `faster`: 1.2–1.4 g\* | 0.095 | 0 (bound 0.11) |
| `steady`: s_g ≤ s\* | 0.695 | 0.495 ± 0.095 |
| both | 0.025 | 0 (bound 0.11) |
| `same_rate`: 0.9–1.1 g\* | 0.115 | 1 (bound 0.89) |
| `same_swings`: 0.8–1.2 s\* | 0.115 | 0.929 ± 0.049 |
| both ("grows as at the truth"; 0.986 pooled) | 0.010 | 0.929 ± 0.049 |

- None of the particles' 300 replicates reaches F1's band; the fastest particle (0.669 m/yr) is 11%
  below its lower edge (0.748).
- **The recovery check changed from the plan's "0.9–1.1 g\*, same F2".** s\* is the mean of the
  truth's replicate SDs, so only 52% of the truth's 50 replicates meet F2, and the truth itself
  would fail the 90% rule (the plan's version gave 0.495). The check's SD criterion is a band
  around s\* instead; all 50 truth replicates meet R1 and R2.
- The one particle failing R is the heaviest (weight 0.071, `minPrecipImpactPct` 36.7%, the
  posterior's drought-sensitive edge): 2 of 10 replicates vary more than 20% above s\*.
- F2's posterior P is a coin flip because its limit runs through the middle of the posterior.
- Figures: particles over the prior draws (window gain against its SD, sized by weight, with F1's
  band and F2's limit), and the prior and posterior predictive gain against both bands.

**Log: a top-up helper (`study.top_up`).** Not yet. The loop is three lines, but each `run_at` is
its own session (30 sessions for one top-up) and runs one job at a time. A batched `run_at` (one
session, one dispatch, `batch_remote`-ready) is the toolkit change worth making when the Joshua
tree posterior needs it; it would also serve the posterior predictive check on the SMC-ABC page.

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

  Logged in PR 4 (`forevertree_case.py` and the page):
  - The landscape mean per replicate and step (`landscape_gains` on the page) and a step window
    (`final_window_gain`) are hand-written SQL on `cell_data_current`.
    `DiagnosticQueries.get_replicate_cv` builds the same per-replicate series internally but
    doesn't return it.
  - Lagged gains are `np.diff` over the window; the window is a constant (`WINDOW`), like
    `FINAL_STEP`.
- **Known-truth toolkit: follow-ups.** `joshpy.inference` (#93) is experimental. Not yet:
  - a distance other than sbi's `"l2"` (extending `SbiStrategy.distance` to accept a callable is
    a small, separate change);
  - summaries as an importable path plus arguments (see above);
  - blessed plots (see below);
  - exporting it from `joshpy/__init__.py`, once PR 4 and PR 5 have used it;
  - `PropertyFilter` (#94) runs only on prior draws (`filter_prior`) or on runs that already hold
    its replicates (`evaluate`). Topping up other runs, such as PR 5's particles, goes through
    `run_at`.
- **Blessed plots** (Oct 25 review, item 2). The candidate is S_sim against S_obs by generation: each
  population's z-scored statistics closing in on S_obs, with the replicate cloud at the truth as
  the floor, as a convergence signal for any study. If blessed, add it as a data method returning
  a tidy table (like `DiagnosticQueries`), and leave the ggplot to the page, so joshpy carries no
  plotting code. Wait for PR 5 to see whether a second page needs it.
- **Patterns at more scales** (Oct 25 review, item 3). Mean and CV are two patterns at one scale:
  across 8 sites, in the final year. Measured over the 200 prior simulations (Spearman ρ with
  `maxNewGrowth`, `minPrecipImpactPct`):
  - between-site CV: −0.14, −0.99 (drought sensitivity);
  - year-to-year CV of the landscape's true growth (years 6–10): −0.11, −0.99, and 0.99 with the
    between-site CV. The same drought response seen in time instead of space: a consistency check
    across scales, not new information here;
  - the same from annual surveys of the 8 sites (0.25 m error each year): −0.71, −0.34, mostly
    measurement noise. Whether a temporal pattern is observable depends on the survey design;
  - growth between two surveys (years 5 and 10): 0.95, 0.36, like the final mean height.
  - Within-site CV (between trees in a cell) isn't exported; adding it edits the model and
    re-simulates everything. By the model's equations it would be about 1.5% for every parameter
    set (the per-tree noise, SD 5%, is the only difference between trees in a cell), so it would
    test the model's structure (a held-out pattern) rather than constrain these two parameters,
    unless the noise SD became a parameter.

  ForeverTree is too simple for patterns at more scales to add information; the Joshua tree model
  (stage structure, revisited plots) is where they should. Options: a short "patterns at other
  scales" note in the case study's step 1, with these numbers computed on the page (no new runs);
  or a temporal pattern as a second held-out check on the SMC-ABC page.
- **Failures in filters and searches.** `filter_prior` and `load` raise if any run fails; searches
  reject failed runs. For the Joshua tree runs: retry infrastructure failures (evicted pods,
  staging errors), record model failures (out of memory when a population explodes) as missing,
  and report where in the parameter box they fall.
- **Refining a filter in waves** (needed only for test (2), if its accepted region is small).
  Rejection from the prior is wasteful when the accepted region is small (PR 4: 5 of 200 draws,
  2,000 replicate runs). SBI.md's design refines in waves. One
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
- **A pilot on the cluster** (the old Phase 4). Measure the run time of a 1,000-year replicate and
  the per-Job overhead, watch which failures occur, and count rounds per generation. These check
  the batch policy's defaults and decide early stopping and failure handling ("Joshua tree
  design").
- **Cleanup once SBI is stable, wholesale:**
  - Remove Optuna: `OptunaStrategy` and its helpers in `strategies.py`, the `study` field on
    adaptive results in `jobs.py`, the adaptive-loop fixtures in `test_strategies.py`, the
    references in `test_sweep.py`, the adaptive-sweep tutorial, and the optional dependency.
  - Spin-up: `examples/spinup_spindown.josh` and the spin-up tutorial still use `start spinup`
    blocks, which Josh #479 removed (Jul 20). Rewrite them to the plain-state recipe.
- **Joshua tree prerequisites:**
  - Fire: per patch, both modes from the jshc ("Second ForeverTree model"); no file primitive.
  - Josh CSV exports: a fix, or a precision setting on Kubernetes ("Josh CSV export bug").
  - Fix outcome-group attribute names (`check_stability`, `check_data`, `group`) in the rules sheet.
    With the two-test design there are no four outcome groups; revisit the names.
  - The replacement check needs R > 1 replicates (`PropertyFilter`, #94).
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
- **Rounds per generation** come from the SMC-ABC batch policy (`measured` by default on
  `batch_remote`).
- **Replicates are a scientific setting, not the scaling lever.**
  - ABC uses `replicates=1` per set.
  - Model-property filters use R > 1 and summarize over the replicates.
- **Scaling path:**
  1. **Now:** one Job per set.
  2. **When staging dominates** (large climate `.jshd`): stage shared data once per session. Not
     expected for Joshua tree, whose runs are slow.
  3. **When Job overhead dominates** (thousands of sets per wave): a Josh-side multi-config indexed
     Job.
